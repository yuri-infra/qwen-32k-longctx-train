"""
真实 `torch.distributed` 序列并行环形注意力引擎（支持标准 Contiguous 与 ZigZag 对称负载均衡 Causal Ring Attention）

核心分布式系统设计：
1. 真实多进程 P2P 环网通信（零单卡假拷贝、零丢弃计算结果）：
   - 基于 `torch.distributed.batch_isend_irecv` 在 `world_size` 个真实进程间沿环路传递 `(K, V)` 分片（反向同时回传累加 `(dK, dV)`）；
   - 偶奇 Rank 交错排列 `[isend, irecv]` / `[irecv, isend]` 杜绝同步后端环路死锁；
   - 原生兼容多 GPU `nccl`（独立高优先级 `comm_stream` 直接显存 P2P 重叠）与单机多进程 `gloo`（后台线程 Pinned 缓冲异步 P2P 重叠）。
2. ZigZag 对称首尾分片消除因果三角形负载失衡（面试核心高频考点）：
   - 普通顺序切分 `[0..S/P), [S/P..2S/P)...` 下，Rank 0 仅计算 1 个下三角块，而 Rank P-1 需计算 P-1 个满块 + 1 个三角块（负载相差 `2P - 1` 倍，产生严重木桶等待气泡）；
   - ZigZag 切分将全局序列均分为 `2P` 个子块 `c_0 .. c_{2P-1}`，令 Rank `r` 持有对称配对 `(c_r, c_{2P - 1 - r})`：
     每个 Rank 承担的有效因果 Token 对数恒为 `((r + 0.5) + (2P - 1 - r + 0.5)) * C^2 = 2P * C^2 / 1`，多卡计算负载比严格等于 **1.00 : 1.00**！
3. 在线精确对数配分函数（Online LSE）合并与全局 `(LSE, Delta)` 反向重算：
   - 前向跨 Ring 步以 FP32 精确合并各分片的局部 `(out_step, lse_step)`；
   - 反向复用全局收敛后的 `lse_total` 与 `Delta = sum(dO * O_total)` 逐步累加 `dQ_local` 并在环上流转归约 `dK, dV` 直至回到归属 Rank。
"""

from concurrent.futures import Future, ThreadPoolExecutor
import math
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.distributed as dist

from .flash_attn_triton import (
    compute_flash_bwd_delta,
    flash_attn_backward_with_lse,
    flash_attn_forward_with_lse,
)


def compute_ring_causal_load_balance(seq_len: int, world_size: int) -> Dict[str, object]:
    """
    定量计算标准顺序分片 (Contiguous) 与对称交错分片 (ZigZag) 在各个 Rank 上的真实有效因果注意力元素对数 `(i >= j)`：
    用于验证 ZigZag 消除因果掩码负载失衡的数学性质。
    """
    if world_size < 1 or seq_len % (2 * world_size) != 0:
        raise ValueError(f"要求 world_size >= 1 且 seq_len ({seq_len}) 可被 2 * world_size ({2 * world_size}) 整除")

    s_per_rank = seq_len // world_size
    contig_pairs: List[int] = []
    for r in range(world_size):
        q_start = r * s_per_rank
        q_end = (r + 1) * s_per_rank
        # sum_{i = q_start}^{q_end - 1} (i + 1)
        cnt = sum(i + 1 for i in range(q_start, q_end))
        contig_pairs.append(cnt)

    sub_chunk = seq_len // (2 * world_size)
    zigzag_pairs: List[int] = []
    for r in range(world_size):
        c_head = range(r * sub_chunk, (r + 1) * sub_chunk)
        mirror = 2 * world_size - 1 - r
        c_tail = range(mirror * sub_chunk, (mirror + 1) * sub_chunk)
        cnt = sum(i + 1 for i in c_head) + sum(i + 1 for i in c_tail)
        zigzag_pairs.append(cnt)

    return {
        "contiguous_pairs_per_rank": contig_pairs,
        "zigzag_pairs_per_rank": zigzag_pairs,
        "contiguous_max_min_ratio": max(contig_pairs) / float(max(1, min(contig_pairs))),
        "zigzag_max_min_ratio": max(zigzag_pairs) / float(max(1, min(zigzag_pairs))),
        "contiguous_straggler_overhead": max(contig_pairs) / (sum(contig_pairs) / float(world_size)),
        "zigzag_straggler_overhead": max(zigzag_pairs) / (sum(zigzag_pairs) / float(world_size)),
    }


def zigzag_shard_sequence(x: torch.Tensor, rank: int, world_size: int, seq_dim: int = 1) -> torch.Tensor:
    """将完整序列张量沿 `seq_dim` 切分为 `2 * world_size` 份，取出当前 `rank` 负责的对称块 `(c_r, c_{2P-1-r})`"""
    seq_len = x.shape[seq_dim]
    if seq_len % (2 * world_size) != 0:
        raise ValueError(f"ZigZag 切分要求序列长度 {seq_len} 可被 2*world_size ({2 * world_size}) 整除")
    chunks = torch.chunk(x, 2 * world_size, dim=seq_dim)
    return torch.cat([chunks[rank], chunks[2 * world_size - 1 - rank]], dim=seq_dim).contiguous()


def zigzag_unshard_sequence(rank_shards: List[torch.Tensor], seq_dim: int = 1) -> torch.Tensor:
    """将全部 `world_size` 个 Rank 的 ZigZag 分片还原拼回全局单调顺序 `[0 .. S-1]`"""
    world_size = len(rank_shards)
    sub_chunks: List[Optional[torch.Tensor]] = [None] * (2 * world_size)
    for r, shard in enumerate(rank_shards):
        first_half, second_half = torch.chunk(shard, 2, dim=seq_dim)
        sub_chunks[r] = first_half
        sub_chunks[2 * world_size - 1 - r] = second_half
    return torch.cat([c for c in sub_chunks if c is not None], dim=seq_dim).contiguous()


def build_zigzag_global_positions(
    batch_size: int,
    global_seq_len: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """生成 Rank `r` 本地 ZigZag 分片对应的真实全局位置索引 `[B, S_local]` (int32)"""
    if global_seq_len % (2 * world_size) != 0:
        raise ValueError(f"global_seq_len ({global_seq_len}) 必须可被 2 * world_size ({2 * world_size}) 整除")
    c = global_seq_len // (2 * world_size)
    head_pos = torch.arange(rank * c, (rank + 1) * c, dtype=torch.int32, device=device)
    mirror = 2 * world_size - 1 - rank
    tail_pos = torch.arange(mirror * c, (mirror + 1) * c, dtype=torch.int32, device=device)
    pos_1d = torch.cat([head_pos, tail_pos], dim=0)
    return pos_1d.unsqueeze(0).expand(batch_size, -1).contiguous()


def _merge_online_lse(
    acc_out: torch.Tensor,
    acc_lse: torch.Tensor,
    step_out: torch.Tensor,
    step_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    以 FP32 精度无损合并两段独立 KV 子集上的注意力结果：
    - `acc_out`: `[B, Sq, Hq, Dh]` (FP32)
    - `acc_lse`: `[B, Hq, Sq]` (FP32)
    - `step_out`: `[B, Sq, Hq, Dh]`
    - `step_lse`: `[B, Hq, Sq]` (FP32)
    """
    m_max = torch.maximum(acc_lse, step_lse)
    m_safe = torch.where(m_max == -float("inf"), torch.zeros_like(m_max), m_max)
    w_acc = torch.where(acc_lse == -float("inf"), torch.zeros_like(acc_lse), torch.exp(acc_lse - m_safe))
    w_step = torch.where(step_lse == -float("inf"), torch.zeros_like(step_lse), torch.exp(step_lse - m_safe))
    denom = w_acc + w_step
    has_any = denom > 0.0
    denom_safe = torch.where(has_any, denom, torch.ones_like(denom))

    new_lse = torch.where(has_any, m_safe + torch.log(denom_safe), torch.full_like(m_max, -float("inf")))
    w_acc_norm = torch.where(has_any, w_acc / denom_safe, torch.zeros_like(w_acc)).transpose(1, 2).unsqueeze(-1)
    w_step_norm = torch.where(has_any, w_step / denom_safe, torch.zeros_like(w_step)).transpose(1, 2).unsqueeze(-1)

    new_out = acc_out * w_acc_norm + step_out.to(torch.float32) * w_step_norm
    return new_out, new_lse


class _AsyncRingCommunicator:
    """
    自适应异步环网点对点交换器：
    - `nccl` 后端：直接在独立高优先级 `comm_stream` 上调用 `dist.batch_isend_irecv` 并通过 CUDA Event 与 `compute_stream` 同步；
    - `gloo` 后端（支持单卡多进程实测）：在独立 `comm_stream` + 单线程执行器上通过 Pinned Host 缓冲异步完成跨进程发送接收，
      使环网通信与主计算流 Triton FlashAttention 核函数真正并发推进！
    """

    def __init__(self, device: torch.device, group: Optional[dist.ProcessGroup] = None):
        self.device = device
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        self.backend = dist.get_backend(group)
        self.comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self._gloo_pool = ThreadPoolExecutor(max_workers=1) if self.backend != "nccl" else None

    def close(self) -> None:
        if self._gloo_pool is not None:
            self._gloo_pool.shutdown(wait=True)
            self._gloo_pool = None

    def launch_ring_exchange(self, send_tensors: List[torch.Tensor]) -> Callable[[], List[torch.Tensor]]:
        compute_stream = torch.cuda.current_stream(self.device)
        ready_ev = torch.cuda.Event(enable_timing=False, blocking=False)
        ready_ev.record(compute_stream)

        contig_sends = [t.contiguous() for t in send_tensors]
        for t in contig_sends:
            t.record_stream(self.comm_stream)

        if self.backend == "nccl":
            self.comm_stream.wait_event(ready_ev)
            ops: List[dist.P2POp] = []
            with torch.cuda.stream(self.comm_stream):
                recv_tensors = [torch.empty_like(t) for t in contig_sends]
                for s_t, r_t in zip(contig_sends, recv_tensors):
                    if self.rank % 2 == 0:
                        ops.append(dist.P2POp(dist.isend, s_t, self.next_rank, self.group))
                        ops.append(dist.P2POp(dist.irecv, r_t, self.prev_rank, self.group))
                    else:
                        ops.append(dist.P2POp(dist.irecv, r_t, self.prev_rank, self.group))
                        ops.append(dist.P2POp(dist.isend, s_t, self.next_rank, self.group))
                reqs = dist.batch_isend_irecv(ops)

            def _wait_nccl() -> List[torch.Tensor]:
                with torch.cuda.stream(self.comm_stream):
                    for rq in reqs:
                        rq.wait()
                    done_ev = torch.cuda.Event(enable_timing=False, blocking=False)
                    done_ev.record(self.comm_stream)
                curr = torch.cuda.current_stream(self.device)
                curr.wait_event(done_ev)
                for r_t in recv_tensors:
                    r_t.record_stream(curr)
                return recv_tensors

            return _wait_nccl

        # gloo 后端：独立线程 + comm_stream 锁页内存异步桥接，允许主线程立即继续发射本地注意力计算！
        assert self._gloo_pool is not None

        def _gloo_worker() -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.cuda.Event]:
            torch.cuda.set_device(self.device)
            self.comm_stream.wait_event(ready_ev)
            cpu_sends: List[torch.Tensor] = []
            cpu_recvs: List[torch.Tensor] = []
            with torch.cuda.stream(self.comm_stream):
                for s_t in contig_sends:
                    c_s = torch.empty(s_t.shape, dtype=s_t.dtype, device="cpu", pin_memory=True)
                    c_s.copy_(s_t, non_blocking=True)
                    c_r = torch.empty(s_t.shape, dtype=s_t.dtype, device="cpu", pin_memory=True)
                    cpu_sends.append(c_s)
                    cpu_recvs.append(c_r)
            self.comm_stream.synchronize()

            ops_g: List[dist.P2POp] = []
            for c_s, c_r in zip(cpu_sends, cpu_recvs):
                if self.rank % 2 == 0:
                    ops_g.append(dist.P2POp(dist.isend, c_s, self.next_rank, self.group))
                    ops_g.append(dist.P2POp(dist.irecv, c_r, self.prev_rank, self.group))
                else:
                    ops_g.append(dist.P2POp(dist.irecv, c_r, self.prev_rank, self.group))
                    ops_g.append(dist.P2POp(dist.isend, c_s, self.next_rank, self.group))
            reqs_g = dist.batch_isend_irecv(ops_g)
            for rq in reqs_g:
                rq.wait()

            gpu_recvs: List[torch.Tensor] = []
            done_ev = torch.cuda.Event(enable_timing=False, blocking=False)
            with torch.cuda.stream(self.comm_stream):
                for c_r in cpu_recvs:
                    g_r = torch.empty(c_r.shape, dtype=c_r.dtype, device=self.device)
                    g_r.copy_(c_r, non_blocking=True)
                    gpu_recvs.append(g_r)
                done_ev.record(self.comm_stream)
            return gpu_recvs, cpu_recvs, done_ev

        fut: Future = self._gloo_pool.submit(_gloo_worker)

        def _wait_gloo() -> List[torch.Tensor]:
            gpu_recvs, cpu_recvs, done_ev = fut.result()
            # 强引用保活锁页缓冲 cpu_recvs 直至异步 H2D DMA 事件物理完成，杜绝主机锁页内存提前归还引发释放后读！
            done_ev.synchronize()
            curr = torch.cuda.current_stream(self.device)
            curr.wait_event(done_ev)
            for g_r in gpu_recvs:
                g_r.record_stream(curr)
            del cpu_recvs
            return gpu_recvs

        return _wait_gloo


class RingAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        pos_local: Optional[torch.Tensor],
        is_causal: bool,
        zigzag: bool,
        group: Optional[dist.ProcessGroup],
    ) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("RingAttentionFunction 需要已初始化的 torch.distributed 进程组")

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        scale = 1.0 / math.sqrt(q_local.shape[-1])

        if world_size == 1:
            out, lse = flash_attn_forward_with_lse(
                q_local, k_local, v_local, is_causal=is_causal, scale=scale, q_pos=pos_local, k_pos=pos_local
            )
            ctx.save_for_backward(q_local, k_local, v_local, out, lse, *( [pos_local] if pos_local is not None else [] ))
            ctx.has_pos = pos_local is not None
            ctx.is_causal = is_causal
            ctx.zigzag = zigzag
            ctx.group = group
            ctx.scale = scale
            return out

        B, S_loc, Hq, Dh = q_local.shape
        comm = _AsyncRingCommunicator(q_local.device, group)
        try:
            acc_out = torch.zeros((B, S_loc, Hq, Dh), dtype=torch.float32, device=q_local.device)
            acc_lse = torch.full((B, Hq, S_loc), -float("inf"), dtype=torch.float32, device=q_local.device)

            k_curr = k_local.contiguous()
            v_curr = v_local.contiguous()
            pos_curr = pos_local.to(torch.int32).contiguous() if (zigzag and pos_local is not None) else None

            for step in range(world_size):
                wait_next = None
                if step + 1 < world_size:
                    payload = [k_curr, v_curr] + ([pos_curr] if pos_curr is not None else [])
                    wait_next = comm.launch_ring_exchange(payload)

                src_rank = (rank - step + world_size) % world_size
                if zigzag:
                    assert pos_local is not None and pos_curr is not None
                    step_out, step_lse = flash_attn_forward_with_lse(
                        q_local, k_curr, v_curr, is_causal=is_causal, scale=scale, q_pos=pos_local, k_pos=pos_curr
                    )
                    acc_out, acc_lse = _merge_online_lse(acc_out, acc_lse, step_out, step_lse)
                else:
                    if not (is_causal and src_rank > rank):
                        local_causal = bool(is_causal and (src_rank == rank))
                        step_out, step_lse = flash_attn_forward_with_lse(
                            q_local, k_curr, v_curr, is_causal=local_causal, scale=scale
                        )
                        acc_out, acc_lse = _merge_online_lse(acc_out, acc_lse, step_out, step_lse)

                if wait_next is not None:
                    recvd = wait_next()
                    k_curr, v_curr = recvd[0], recvd[1]
                    if pos_curr is not None:
                        pos_curr = recvd[2]

            out_final = acc_out.to(q_local.dtype)
            ctx.has_pos = pos_local is not None
            if ctx.has_pos:
                assert pos_local is not None
                ctx.save_for_backward(q_local, k_local, v_local, out_final, acc_lse, pos_local)
            else:
                ctx.save_for_backward(q_local, k_local, v_local, out_final, acc_lse)
            ctx.is_causal = is_causal
            ctx.zigzag = zigzag
            ctx.group = group
            ctx.scale = scale
            return out_final
        finally:
            comm.close()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.has_pos:
            q_local, k_local, v_local, out_final, lse_final, pos_local = ctx.saved_tensors
        else:
            q_local, k_local, v_local, out_final, lse_final = ctx.saved_tensors
            pos_local = None

        group = ctx.group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        scale = ctx.scale
        is_causal = ctx.is_causal
        zigzag = ctx.zigzag

        delta = compute_flash_bwd_delta(out_final, grad_out)

        if world_size == 1:
            dq, dk, dv = flash_attn_backward_with_lse(
                grad_out, q_local, k_local, v_local, lse_final, delta,
                is_causal=is_causal, scale=scale, q_pos=pos_local, k_pos=pos_local
            )
            return dq, dk, dv, None, None, None, None

        comm = _AsyncRingCommunicator(q_local.device, group)
        try:
            dq_acc = torch.zeros_like(q_local, dtype=torch.float32)
            k_curr = k_local.contiguous()
            v_curr = v_local.contiguous()
            dk_curr = torch.zeros_like(k_local, dtype=torch.float32)
            dv_curr = torch.zeros_like(v_local, dtype=torch.float32)
            pos_curr = pos_local.to(torch.int32).contiguous() if (zigzag and pos_local is not None) else None

            for step in range(world_size):
                wait_kv_next = None
                if step + 1 < world_size:
                    kv_payload = [k_curr, v_curr] + ([pos_curr] if pos_curr is not None else [])
                    wait_kv_next = comm.launch_ring_exchange(kv_payload)

                src_rank = (rank - step + world_size) % world_size
                if zigzag:
                    assert pos_local is not None and pos_curr is not None
                    dq_s, dk_s, dv_s = flash_attn_backward_with_lse(
                        grad_out, q_local, k_curr, v_curr, lse_final, delta,
                        is_causal=is_causal, scale=scale, q_pos=pos_local, k_pos=pos_curr
                    )
                    dq_acc += dq_s.to(torch.float32)
                    dk_curr += dk_s.to(torch.float32)
                    dv_curr += dv_s.to(torch.float32)
                else:
                    if not (is_causal and src_rank > rank):
                        local_causal = bool(is_causal and (src_rank == rank))
                        dq_s, dk_s, dv_s = flash_attn_backward_with_lse(
                            grad_out, q_local, k_curr, v_curr, lse_final, delta,
                            is_causal=local_causal, scale=scale
                        )
                        dq_acc += dq_s.to(torch.float32)
                        dk_curr += dk_s.to(torch.float32)
                        dv_curr += dv_s.to(torch.float32)

                if wait_kv_next is not None:
                    recvd_kv = wait_kv_next()
                    k_curr, v_curr = recvd_kv[0], recvd_kv[1]
                    if pos_curr is not None:
                        pos_curr = recvd_kv[2]

                # 将累加后的 (dk_curr, dv_curr) 沿环路传递一跳；经过全部 world_size 跳后精准归位至所属原 Rank！
                wait_dkv = comm.launch_ring_exchange([dk_curr, dv_curr])
                dk_curr, dv_curr = wait_dkv()

            return dq_acc.to(q_local.dtype), dk_curr.to(k_local.dtype), dv_curr.to(v_local.dtype), None, None, None, None
        finally:
            comm.close()


def ring_flash_attention(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    is_causal: bool = True,
    zigzag: bool = True,
    pos_local: Optional[torch.Tensor] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    分布式序列并行环形注意力入口：
    - `zigzag=True`（推荐）：配合 `zigzag_shard_sequence` 与 `build_zigzag_global_positions` 实现多卡 1:1 严格对称负载均衡；
    - `zigzag=False`：按连续块切分 `[r * S_loc, (r + 1) * S_loc)`。
    """
    if zigzag and pos_local is None and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        pos_local = build_zigzag_global_positions(
            q_local.shape[0], q_local.shape[1] * world_size, rank, world_size, q_local.device
        )
    return RingAttentionFunction.apply(q_local, k_local, v_local, pos_local, is_causal, zigzag, group)  # type: ignore[return-value]


def _ring_verification_worker(
    rank: int,
    world_size: int,
    port: int,
    zigzag: bool,
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
    do_cpu: torch.Tensor,
    result_file: str,
) -> None:
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda:0")
        q_full = q_cpu.to(device)
        k_full = k_cpu.to(device)
        v_full = v_cpu.to(device)
        do_full = do_cpu.to(device)

        if zigzag:
            q_loc = zigzag_shard_sequence(q_full, rank, world_size, seq_dim=1).detach().requires_grad_(True)
            k_loc = zigzag_shard_sequence(k_full, rank, world_size, seq_dim=1).detach().requires_grad_(True)
            v_loc = zigzag_shard_sequence(v_full, rank, world_size, seq_dim=1).detach().requires_grad_(True)
            do_loc = zigzag_shard_sequence(do_full, rank, world_size, seq_dim=1)
            pos_loc = build_zigzag_global_positions(q_full.shape[0], q_full.shape[1], rank, world_size, device)
        else:
            s_loc = q_full.shape[1] // world_size
            sl = slice(rank * s_loc, (rank + 1) * s_loc)
            q_loc = q_full[:, sl].contiguous().detach().requires_grad_(True)
            k_loc = k_full[:, sl].contiguous().detach().requires_grad_(True)
            v_loc = v_full[:, sl].contiguous().detach().requires_grad_(True)
            do_loc = do_full[:, sl].contiguous()
            pos_loc = None

        out_loc = ring_flash_attention(q_loc, k_loc, v_loc, is_causal=True, zigzag=zigzag, pos_local=pos_loc)
        out_loc.backward(do_loc)
        torch.cuda.synchronize(device)

        gather_out = [torch.zeros_like(out_loc.detach().cpu()) for _ in range(world_size)]
        gather_dq = [torch.zeros_like(q_loc.grad.detach().cpu()) for _ in range(world_size)]  # type: ignore[union-attr]
        gather_dk = [torch.zeros_like(k_loc.grad.detach().cpu()) for _ in range(world_size)]  # type: ignore[union-attr]
        gather_dv = [torch.zeros_like(v_loc.grad.detach().cpu()) for _ in range(world_size)]  # type: ignore[union-attr]

        dist.all_gather(gather_out, out_loc.detach().cpu())
        dist.all_gather(gather_dq, q_loc.grad.detach().cpu())  # type: ignore[union-attr]
        dist.all_gather(gather_dk, k_loc.grad.detach().cpu())  # type: ignore[union-attr]
        dist.all_gather(gather_dv, v_loc.grad.detach().cpu())  # type: ignore[union-attr]

        if rank == 0:
            if zigzag:
                full_out = zigzag_unshard_sequence(gather_out, seq_dim=1)
                full_dq = zigzag_unshard_sequence(gather_dq, seq_dim=1)
                full_dk = zigzag_unshard_sequence(gather_dk, seq_dim=1)
                full_dv = zigzag_unshard_sequence(gather_dv, seq_dim=1)
            else:
                full_out = torch.cat(gather_out, dim=1)
                full_dq = torch.cat(gather_dq, dim=1)
                full_dk = torch.cat(gather_dk, dim=1)
                full_dv = torch.cat(gather_dv, dim=1)
            torch.save({"out": full_out, "dq": full_dq, "dk": full_dk, "dv": full_dv}, result_file)
    finally:
        dist.destroy_process_group()


def run_multiprocess_ring_verification(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    world_size: int = 2,
    zigzag: bool = True,
) -> Dict[str, torch.Tensor]:
    """派生 `world_size` 个真实独立进程运行分布式 Ring Attention 前反向并返回拼接还原后的全局张量与梯度"""
    import os
    import socket
    import tempfile
    import torch.multiprocessing as mp

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    fd, tmp_path = tempfile.mkstemp(suffix=".pt", prefix="ring_attn_verify_")
    os.close(fd)
    try:
        mp.spawn(
            _ring_verification_worker,
            args=(
                world_size,
                port,
                zigzag,
                q.detach().cpu(),
                k.detach().cpu(),
                v.detach().cpu(),
                grad_out.detach().cpu(),
                tmp_path,
            ),
            nprocs=world_size,
            join=True,
        )
        res = torch.load(tmp_path, weights_only=True)
        return {key: val.to(q.device) for key, val in res.items()}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
