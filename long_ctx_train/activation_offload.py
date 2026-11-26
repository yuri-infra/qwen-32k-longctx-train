"""
槽位安全锁页内存池 (`PinnedActivationPool`) + 双缓冲异步前向卸载与反向跨层预取流水线 (`DoubleBufferedOffloadPipeline`)

核心系统机制（彻底根除旧版环形回绕覆写 Bug 与同步阻塞气泡）：
1. 槽位引用追踪锁页内存池（`PinnedActivationPool`）：
   - 预分配 256 字节对齐的 `pin_memory=True` CPU 缓冲槽位，每个槽位显式维护 `in_use` 占用位、
     `d2h_done_event`（前向 D2H 落盘完成事件）与 `h2d_consumed_event`（反向 H2D 读取完成事件）；
   - 复用槽位前强制在 `d2h_stream` 上 `wait_event(h2d_consumed_event)`，确保前一步异步 H2D 未读完前绝不覆写；
   - 当所有预分配槽位均处于 `in_use` 状态时：若 `strict_capacity=True` 立即抛出 `MemoryError` 拒绝覆写，
     若 `strict_capacity=False` 自动安全扩容新槽位，**从数据结构层面 100% 杜绝未消费激活被静默踩踏**！
2. 双缓冲异步 D2H 卸载 + 反向跨层超前 H2D 预取（`DoubleBufferedOffloadPipeline`）：
   - 前向阶段：第 `l` 层输入激活在 `d2h_stream`（独立硬件 PCIe Copy Engine）上异步 DMA 卸载至主机锁页槽位，
     配合 `x.record_stream(d2h_stream)` 立即释放主计算流 Autograd 显存持有，与后续层的正向 GEMM 并行重叠；
   - 反向阶段：当第 `l` 层开始反向求导时，**第一时间在 `h2d_stream` 上异步触发第 `l - 1` 层激活的 H2D 预取**，
     写入双缓冲显存槽 `gpu_staging[(l - 1) % 2]`；使第 `l - 1` 层的 PCIe H2D 传输延迟完全掩盖在第 `l` 层繁重的
     反向 MLP + FlashAttention-2 GEMM 计算背后！
   - 提供 `sync_debug_mode=True`（禁用双流重叠，退化为同流串行拷贝），用于诚实消融验证硬件 DMA 重叠加速收益。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class ActivationSlot:
    slot_id: int
    buffer: torch.Tensor  # 1D uint8 pinned CPU buffer
    in_use: bool
    d2h_done_event: torch.cuda.Event
    h2d_consumed_event: torch.cuda.Event
    shape: Tuple[int, ...] = ()
    dtype: torch.dtype = torch.float16
    nbytes: int = 0


class PinnedActivationPool:
    """显式追踪槽位生命周期的预分配锁页内存池（零 `cudaHostAlloc` 运行期系统调用、零环绕覆写风险）"""

    def __init__(
        self,
        num_slots: int = 24,
        slot_bytes: int = 32 * 1024 * 1024,
        strict_capacity: bool = False,
    ):
        if num_slots < 1 or slot_bytes < 256:
            raise ValueError(f"num_slots ({num_slots}) 必须 >= 1 且 slot_bytes ({slot_bytes}) >= 256")
        self.aligned_slot_bytes = (slot_bytes + 255) & ~255
        self.strict_capacity = strict_capacity
        self.slots: List[ActivationSlot] = [
            ActivationSlot(
                slot_id=i,
                buffer=torch.empty((self.aligned_slot_bytes,), dtype=torch.uint8, device="cpu", pin_memory=True),
                in_use=False,
                d2h_done_event=torch.cuda.Event(enable_timing=False, blocking=False),
                h2d_consumed_event=torch.cuda.Event(enable_timing=False, blocking=False),
            )
            for i in range(num_slots)
        ]
        self.peak_active_slots: int = 0
        self.overflow_alloc_count: int = 0
        self.total_d2h_bytes: int = 0
        self.total_h2d_bytes: int = 0

    @property
    def num_active_slots(self) -> int:
        return sum(1 for s in self.slots if s.in_use)

    def acquire_slot(self, nbytes: int, shape: Tuple[int, ...], dtype: torch.dtype) -> ActivationSlot:
        aligned_req = (nbytes + 255) & ~255
        chosen: Optional[ActivationSlot] = None
        for s in self.slots:
            if not s.in_use:
                chosen = s
                break

        if chosen is None:
            if self.strict_capacity:
                raise MemoryError(
                    f"PinnedActivationPool 全部 {len(self.slots)} 个槽位均在使用中，拒绝覆写未消费激活张量！"
                )
            alloc_sz = max(self.aligned_slot_bytes, aligned_req)
            chosen = ActivationSlot(
                slot_id=len(self.slots),
                buffer=torch.empty((alloc_sz,), dtype=torch.uint8, device="cpu", pin_memory=True),
                in_use=False,
                d2h_done_event=torch.cuda.Event(enable_timing=False, blocking=False),
                h2d_consumed_event=torch.cuda.Event(enable_timing=False, blocking=False),
            )
            self.slots.append(chosen)
            self.overflow_alloc_count += 1

        if chosen.buffer.numel() < aligned_req:
            # 槽位当前空闲（in_use == False），可安全扩容而不影响任何在途激活
            chosen.d2h_done_event.synchronize()
            chosen.h2d_consumed_event.synchronize()
            chosen.buffer = torch.empty((aligned_req,), dtype=torch.uint8, device="cpu", pin_memory=True)

        chosen.in_use = True
        chosen.shape = shape
        chosen.dtype = dtype
        chosen.nbytes = nbytes
        active = self.num_active_slots
        if active > self.peak_active_slots:
            self.peak_active_slots = active
        return chosen

    def view_slot_tensor(self, slot: ActivationSlot) -> torch.Tensor:
        return slot.buffer[: slot.nbytes].view(slot.dtype).reshape(slot.shape)

    def release_slot(self, slot: ActivationSlot) -> None:
        slot.in_use = False


class DoubleBufferedOffloadPipeline:
    """
    双缓冲异步激活卸载与反向跨层超前预取调度器：
    - `offload_forward(x)`: 将 `x` 异步卸载至锁页槽位并返回 `ticket_id`（不保留 GPU 端 `x` 引用）；
    - `trigger_prefetch_latest()`: 在 Loss 反向开始时立即预取最后一个卸载层；
    - `fetch_for_backward_and_prefetch_prev(ticket_id)`: 立即发起前一层 `prev_ticket` 的异步 H2D 预取，
      并等待当前层 `ticket_id` 的预取缓冲就绪后交还计算流消费。
    """

    def __init__(
        self,
        pool: PinnedActivationPool,
        device: torch.device = torch.device("cuda:0"),
        sync_debug_mode: bool = False,
    ):
        self.pool = pool
        self.device = device
        self.sync_debug_mode = sync_debug_mode
        self.d2h_stream = torch.cuda.Stream(device=device, priority=-1)
        self.h2d_stream = torch.cuda.Stream(device=device, priority=-1)

        # 双缓冲 GPU 预取暂存区 (Ping-Pong Staging Buffers)
        self._staging_bufs: List[Optional[torch.Tensor]] = [None, None]
        self._staging_free_evs: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=False, blocking=False),
            torch.cuda.Event(enable_timing=False, blocking=False),
        ]

        self._next_ticket: int = 0
        self._active_tickets: Dict[int, ActivationSlot] = {}
        self._ticket_order: List[int] = []
        self._ticket_ping: Dict[int, int] = {}
        self._prefetched_gpu: Dict[int, Tuple[torch.Tensor, torch.cuda.Event, int]] = {}
        self.prefetch_hit_count: int = 0

    def reset_step(self) -> None:
        if self._active_tickets or self._prefetched_gpu:
            self.d2h_stream.synchronize()
            self.h2d_stream.synchronize()
        for slot in list(self._active_tickets.values()):
            self.pool.release_slot(slot)
        self._active_tickets.clear()
        self._ticket_order.clear()
        self._ticket_ping.clear()
        self._prefetched_gpu.clear()

    def _ensure_staging_buf(self, ping_idx: int, nbytes: int) -> torch.Tensor:
        aligned = (nbytes + 255) & ~255
        cur = self._staging_bufs[ping_idx]
        if cur is None or cur.numel() < aligned:
            self._staging_free_evs[ping_idx].synchronize()
            self.h2d_stream.synchronize()
            # 必须在 h2d_stream 上下文内分配，杜绝从 compute_stream 缓存池拿到仍在执行写核函数的回收块引发数据竞争！
            with torch.cuda.stream(self.h2d_stream):
                cur = torch.empty((aligned,), dtype=torch.uint8, device=self.device)
            self._staging_bufs[ping_idx] = cur
        return cur

    def offload_forward(self, x: torch.Tensor) -> int:
        x_c = x.detach().contiguous()
        nbytes = x_c.numel() * x_c.element_size()
        slot = self.pool.acquire_slot(nbytes, tuple(x_c.shape), x_c.dtype)
        host_view = self.pool.view_slot_tensor(slot)

        self._next_ticket += 1
        ticket = self._next_ticket
        ping_idx = len(self._ticket_order) & 1
        self._active_tickets[ticket] = slot
        self._ticket_order.append(ticket)
        self._ticket_ping[ticket] = ping_idx
        self.pool.total_d2h_bytes += nbytes

        compute_stream = torch.cuda.current_stream(self.device)
        if self.sync_debug_mode:
            # 同流串行对照路径：仍在 compute_stream 硬件队列内发射异步 DMA（不插入 CPU 同步气泡），严格隔离纯双流重叠收益
            compute_stream.wait_event(slot.h2d_consumed_event)
            host_view.copy_(x_c, non_blocking=True)
            slot.d2h_done_event.record(compute_stream)
            return ticket

        ready_ev = torch.cuda.Event(enable_timing=False, blocking=False)
        ready_ev.record(compute_stream)
        x_c.record_stream(self.d2h_stream)

        self.d2h_stream.wait_event(ready_ev)
        # 确保该主机槽位上一轮 H2D 读取已彻底结束再覆写
        self.d2h_stream.wait_event(slot.h2d_consumed_event)
        with torch.cuda.stream(self.d2h_stream):
            host_view.copy_(x_c, non_blocking=True)
            slot.d2h_done_event.record(self.d2h_stream)

        return ticket

    def enqueue_prefetch(self, ticket: int) -> None:
        """在 `h2d_stream` 上异步预取指定 `ticket` 对应的激活张量至双缓冲显存槽"""
        if self.sync_debug_mode or ticket in self._prefetched_gpu or ticket not in self._active_tickets:
            return

        slot = self._active_tickets[ticket]
        ping_idx = self._ticket_ping[ticket]

        staging_raw = self._ensure_staging_buf(ping_idx, slot.nbytes)
        host_view = self.pool.view_slot_tensor(slot)
        h2d_ready = torch.cuda.Event(enable_timing=False, blocking=False)

        # 等待：1) 该槽位前向 D2H 已落盘；2) 该乒乓显存槽上一层反向读取已完成
        self.h2d_stream.wait_event(slot.d2h_done_event)
        self.h2d_stream.wait_event(self._staging_free_evs[ping_idx])
        with torch.cuda.stream(self.h2d_stream):
            gpu_view = staging_raw[: slot.nbytes].view(slot.dtype).reshape(slot.shape)
            gpu_view.copy_(host_view, non_blocking=True)
            slot.h2d_consumed_event.record(self.h2d_stream)
            h2d_ready.record(self.h2d_stream)

        self.pool.total_h2d_bytes += slot.nbytes
        self._prefetched_gpu[ticket] = (gpu_view, h2d_ready, ping_idx)

    def trigger_prefetch_latest(self) -> None:
        if self._ticket_order:
            self.enqueue_prefetch(self._ticket_order[-1])

    def fetch_for_backward_and_prefetch_prev(self, ticket: int) -> torch.Tensor:
        """
        反向求导核心交接点：
        1. 立即查找前序待求导层 `prev_ticket` 并向 `h2d_stream` 发射超前预取（与当前层即将执行的反向 GEMM 重叠！）；
        2. 等待当前层 `ticket` 的预取缓冲就绪，拷贝出当前层激活并释放双缓冲槽与主机锁页槽。
        """
        slot = self._active_tickets.get(ticket)
        if slot is None:
            raise RuntimeError(
                f"卸载激活槽位 ticket={ticket} 已消费释放（CPU 激活卸载模式不支持 retain_graph=True 重复反向）"
            )

        compute_stream = torch.cuda.current_stream(self.device)

        if self.sync_debug_mode:
            compute_stream.wait_event(slot.d2h_done_event)
            host_view = self.pool.view_slot_tensor(slot)
            out = torch.empty(slot.shape, dtype=slot.dtype, device=self.device)
            out.copy_(host_view, non_blocking=True)
            slot.h2d_consumed_event.record(compute_stream)
            self.pool.total_h2d_bytes += slot.nbytes
            self._active_tickets.pop(ticket, None)
            self._ticket_ping.pop(ticket, None)
            if ticket in self._ticket_order:
                self._ticket_order.remove(ticket)
            self.pool.release_slot(slot)
            return out

        already_prefetched = ticket in self._prefetched_gpu
        if not already_prefetched:
            self.enqueue_prefetch(ticket)
        else:
            self.prefetch_hit_count += 1

        # 定位紧邻的下一待反向层（即前向顺序中的前一层）并发起异步 H2D 预取！
        if ticket in self._ticket_order:
            idx = self._ticket_order.index(ticket)
            if idx - 1 >= 0:
                prev_ticket = self._ticket_order[idx - 1]
                self.enqueue_prefetch(prev_ticket)

        gpu_view, h2d_ready, ping_idx = self._prefetched_gpu.pop(ticket)
        compute_stream.wait_event(h2d_ready)
        gpu_view.record_stream(compute_stream)
        # 在计算流上生成当前层独立张量副本并立即标记 ping_idx 乒乓缓冲可供后续第 idx-2 层复用
        restored = gpu_view.clone()
        self._staging_free_evs[ping_idx].record(compute_stream)

        self._active_tickets.pop(ticket, None)
        self._ticket_ping.pop(ticket, None)
        if ticket in self._ticket_order:
            self._ticket_order.remove(ticket)
        self.pool.release_slot(slot)
        return restored
