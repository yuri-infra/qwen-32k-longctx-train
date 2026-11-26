"""
真实 Triton FlashAttention-2 前反向全自定义算子（支持 GQA 异构头比、O(S) 在线 LSE 重算与任意位置因果掩码）

核心微架构设计：
1. 彻底根除 O(B * Hq * S^2) 显存物化：
   - 前向采用 Online Softmax 分块（SRAM 内维护 `m_i, l_i, acc`），全局显存仅写出输出 `Out: [B, S, Hq, Dh]`
     与一维对数配分函数 `LSE: [B, Hq, S]`（FP32，每 Token 每头仅 4 Bytes）；
   - 反向传播拆分为：
     a) `_flash_bwd_preprocess_kernel`：单 Pass 计算行内点积 `Delta_i = sum_d(dO_{i,d} * O_{i,d})`；
     b) `_flash_bwd_dk_dv_kernel`：按 `(cdiv(Sk, BLOCK_N), B * Hkv)` 网格组织，在寄存器内跨同组全部 `GROUP_SIZE` 个 Query 头
        及有效 Query 行无原子锁累加 `dK, dV`（零 `atomic_add` 确定性归约，原生适配 Qwen2.5 `14Q/2KV` 7:1 GQA！）；
     c) `_flash_bwd_dq_kernel`：按 `(cdiv(Sq, BLOCK_M), B * Hq)` 网格组织，在片上 SRAM 重算局部 `P_{ij} = exp(S_{ij} - LSE_i)`
        并直接写出连续 `dQ`。
2. 原生支持两种因果模式：
   - 标准下三角因果模式（`is_causal=True, q_pos=None`）：自动跳过上三角无效分块，减少 50% 无效 FLOPs；
   - 任意全局位置索引因果模式（传入 `q_pos: [B, Sq], k_pos: [B, Sk]`）：直接在寄存器比较 `q_pos >= k_pos`，
     无需物化二维 `[Sq, Sk]` 掩码即可驱动 **ZigZag 负载均衡 Ring Attention**！
3. 严格输入契约与步长保护：
   - 非连续输入视图（`stride(-1) != 1`）自动 `.contiguous()` 保护，输出统一由 `torch.empty` 显式分配连续显存；
   - 针对整行被掩码的边界（如 Ring Attention 远端未来分片）做 `lse == -inf` 防 NaN 保护。
"""

from typing import Dict, Optional, Tuple
import math
import torch
import triton
import triton.language as tl

FLASH_ATTN_COUNTERS: Dict[str, int] = {
    "fwd": 0,
    "bwd": 0,
}


def reset_flash_attn_counters() -> None:
    for k in FLASH_ATTN_COUNTERS:
        FLASH_ATTN_COUNTERS[k] = 0


def get_flash_attn_counters() -> Dict[str, int]:
    return dict(FLASH_ATTN_COUNTERS)


def _ensure_contiguous_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


# ==============================================================================
# 1. Triton FlashAttention-2 Forward Kernel
# ==============================================================================

@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    LSE_ptr,
    QPos_ptr,
    KPos_ptr,
    stride_qb, stride_qs, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_ob, stride_os, stride_oh,
    stride_lb, stride_lh,
    stride_qpb,
    stride_kpb,
    Sq,
    Sk,
    scale,
    Hq: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_POS_IDS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // Hq
    q_h = pid_bh % Hq
    kv_h = q_h // GROUP_SIZE

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Sq
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + b_idx * stride_qb + offs_m[:, None] * stride_qs + q_h * stride_qh + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    if HAS_POS_IDS:
        q_pos = tl.load(QPos_ptr + b_idx * stride_qpb + offs_m, mask=mask_m, other=-1)
    else:
        q_pos = offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL and not HAS_POS_IDS:
        max_k_limit = tl.minimum(Sk, start_m + BLOCK_M)
    else:
        max_k_limit = Sk

    for start_n in range(0, max_k_limit, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < Sk

        k_ptrs = K_ptr + b_idx * stride_kb + offs_n[:, None] * stride_ks + kv_h * stride_kh + offs_d[None, :]
        v_ptrs = V_ptr + b_idx * stride_vb + offs_n[:, None] * stride_vs + kv_h * stride_vh + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        if IS_FP32:
            scores = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * scale
        else:
            scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

        valid_mn = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            if HAS_POS_IDS:
                k_pos = tl.load(KPos_ptr + b_idx * stride_kpb + offs_n, mask=mask_n, other=2147483647)
                valid_mn = valid_mn & (q_pos[:, None] >= k_pos[None, :])
            else:
                valid_mn = valid_mn & (offs_m[:, None] >= offs_n[None, :])

        scores = tl.where(valid_mn, scores, -float("inf"))

        m_tile = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)

        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(valid_mn, tl.exp(scores - m_new_safe[:, None]), 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        if IS_FP32:
            acc = acc * alpha[:, None] + tl.dot(p, v_tile, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_new

    has_valid = l_i > 0.0
    denom_safe = tl.where(has_valid, l_i, 1.0)
    out_tile = tl.where(has_valid[:, None], acc / denom_safe[:, None], 0.0)
    lse_m = tl.where(has_valid, m_i + tl.log(denom_safe), -float("inf"))

    out_ptrs = Out_ptr + b_idx * stride_ob + offs_m[:, None] * stride_os + q_h * stride_oh + offs_d[None, :]
    tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])

    lse_ptrs = LSE_ptr + b_idx * stride_lb + q_h * stride_lh + offs_m
    tl.store(lse_ptrs, lse_m, mask=mask_m)


# ==============================================================================
# 2. Triton FlashAttention-2 Backward Kernels (Preprocess + Atomic-Free GQA dK/dV + dQ)
# ==============================================================================

@triton.jit
def _flash_bwd_preprocess_kernel(
    Out_ptr,
    dO_ptr,
    Delta_ptr,
    stride_ob, stride_os, stride_oh,
    stride_dob, stride_dos, stride_doh,
    stride_db, stride_dh,
    Sq,
    Hq: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // Hq
    q_h = pid_bh % Hq

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Sq
    offs_d = tl.arange(0, HEAD_DIM)

    o_ptrs = Out_ptr + b_idx * stride_ob + offs_m[:, None] * stride_os + q_h * stride_oh + offs_d[None, :]
    do_ptrs = dO_ptr + b_idx * stride_dob + offs_m[:, None] * stride_dos + q_h * stride_doh + offs_d[None, :]

    o_val = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do_val = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o_val * do_val, axis=1)

    d_ptrs = Delta_ptr + b_idx * stride_db + q_h * stride_dh + offs_m
    tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _flash_bwd_dk_dv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    LSE_ptr,
    Delta_ptr,
    dK_ptr,
    dV_ptr,
    QPos_ptr,
    KPos_ptr,
    stride_qb, stride_qs, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_dob, stride_dos, stride_doh,
    stride_lb, stride_lh,
    stride_dkb, stride_dks, stride_dkh,
    stride_dvb, stride_dvs, stride_dvh,
    stride_qpb,
    stride_kpb,
    Sq,
    Sk,
    scale,
    Hkv: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_POS_IDS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bkv = tl.program_id(1)
    b_idx = pid_bkv // Hkv
    kv_h = pid_bkv % Hkv

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < Sk
    offs_d = tl.arange(0, HEAD_DIM)

    k_ptrs = K_ptr + b_idx * stride_kb + offs_n[:, None] * stride_ks + kv_h * stride_kh + offs_d[None, :]
    v_ptrs = V_ptr + b_idx * stride_vb + offs_n[:, None] * stride_vs + kv_h * stride_vh + offs_d[None, :]
    k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    if HAS_POS_IDS:
        k_pos = tl.load(KPos_ptr + b_idx * stride_kpb + offs_n, mask=mask_n, other=2147483647)
    else:
        k_pos = offs_n

    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL and not HAS_POS_IDS:
        min_m_start = (start_n // BLOCK_M) * BLOCK_M
    else:
        min_m_start = 0

    for g in range(0, GROUP_SIZE):
        q_h = kv_h * GROUP_SIZE + g
        for start_m in range(min_m_start, Sq, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < Sq

            q_ptrs = Q_ptr + b_idx * stride_qb + offs_m[:, None] * stride_qs + q_h * stride_qh + offs_d[None, :]
            do_ptrs = dO_ptr + b_idx * stride_dob + offs_m[:, None] * stride_dos + q_h * stride_doh + offs_d[None, :]
            q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

            lse_m = tl.load(LSE_ptr + b_idx * stride_lb + q_h * stride_lh + offs_m, mask=mask_m, other=-float("inf"))
            delta_m = tl.load(Delta_ptr + b_idx * stride_lb + q_h * stride_lh + offs_m, mask=mask_m, other=0.0)

            if IS_FP32:
                scores = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * scale
            else:
                scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

            valid_mn = mask_m[:, None] & mask_n[None, :] & (lse_m[:, None] > -float("inf"))
            if IS_CAUSAL:
                if HAS_POS_IDS:
                    q_pos = tl.load(QPos_ptr + b_idx * stride_qpb + offs_m, mask=mask_m, other=-1)
                    valid_mn = valid_mn & (q_pos[:, None] >= k_pos[None, :])
                else:
                    valid_mn = valid_mn & (offs_m[:, None] >= k_pos[None, :])

            lse_safe = tl.where(lse_m == -float("inf"), 0.0, lse_m)
            p = tl.where(valid_mn, tl.exp(scores - lse_safe[:, None]), 0.0)

            if IS_FP32:
                dv_acc += tl.dot(tl.trans(p), do_tile, input_precision="ieee")
                dp = tl.dot(do_tile, tl.trans(v_tile), input_precision="ieee")
            else:
                dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile)
                dp = tl.dot(do_tile, tl.trans(v_tile))

            ds = p * (dp - delta_m[:, None]) * scale
            if IS_FP32:
                dk_acc += tl.dot(tl.trans(ds), q_tile, input_precision="ieee")
            else:
                dk_acc += tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile)

    dk_ptrs = dK_ptr + b_idx * stride_dkb + offs_n[:, None] * stride_dks + kv_h * stride_dkh + offs_d[None, :]
    dv_ptrs = dV_ptr + b_idx * stride_dvb + offs_n[:, None] * stride_dvs + kv_h * stride_dvh + offs_d[None, :]
    tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def _flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    LSE_ptr,
    Delta_ptr,
    dQ_ptr,
    QPos_ptr,
    KPos_ptr,
    stride_qb, stride_qs, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_dob, stride_dos, stride_doh,
    stride_lb, stride_lh,
    stride_dqb, stride_dqs, stride_dqh,
    stride_qpb,
    stride_kpb,
    Sq,
    Sk,
    scale,
    Hq: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_POS_IDS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // Hq
    q_h = pid_bh % Hq
    kv_h = q_h // GROUP_SIZE

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Sq
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + b_idx * stride_qb + offs_m[:, None] * stride_qs + q_h * stride_qh + offs_d[None, :]
    do_ptrs = dO_ptr + b_idx * stride_dob + offs_m[:, None] * stride_dos + q_h * stride_doh + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

    lse_m = tl.load(LSE_ptr + b_idx * stride_lb + q_h * stride_lh + offs_m, mask=mask_m, other=-float("inf"))
    delta_m = tl.load(Delta_ptr + b_idx * stride_lb + q_h * stride_lh + offs_m, mask=mask_m, other=0.0)
    lse_safe = tl.where(lse_m == -float("inf"), 0.0, lse_m)

    if HAS_POS_IDS:
        q_pos = tl.load(QPos_ptr + b_idx * stride_qpb + offs_m, mask=mask_m, other=-1)
    else:
        q_pos = offs_m

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL and not HAS_POS_IDS:
        max_k_limit = tl.minimum(Sk, start_m + BLOCK_M)
    else:
        max_k_limit = Sk

    for start_n in range(0, max_k_limit, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < Sk

        k_ptrs = K_ptr + b_idx * stride_kb + offs_n[:, None] * stride_ks + kv_h * stride_kh + offs_d[None, :]
        v_ptrs = V_ptr + b_idx * stride_vb + offs_n[:, None] * stride_vs + kv_h * stride_vh + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        if IS_FP32:
            scores = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * scale
        else:
            scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

        valid_mn = mask_m[:, None] & mask_n[None, :] & (lse_m[:, None] > -float("inf"))
        if IS_CAUSAL:
            if HAS_POS_IDS:
                k_pos = tl.load(KPos_ptr + b_idx * stride_kpb + offs_n, mask=mask_n, other=2147483647)
                valid_mn = valid_mn & (q_pos[:, None] >= k_pos[None, :])
            else:
                valid_mn = valid_mn & (q_pos[:, None] >= offs_n[None, :])

        p = tl.where(valid_mn, tl.exp(scores - lse_safe[:, None]), 0.0)
        if IS_FP32:
            dp = tl.dot(do_tile, tl.trans(v_tile), input_precision="ieee")
            ds = p * (dp - delta_m[:, None]) * scale
            dq_acc += tl.dot(ds, k_tile, input_precision="ieee")
        else:
            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = (p * (dp - delta_m[:, None]) * scale).to(k_tile.dtype)
            dq_acc += tl.dot(ds, k_tile)

    dq_ptrs = dQ_ptr + b_idx * stride_dqb + offs_m[:, None] * stride_dqs + q_h * stride_dqh + offs_d[None, :]
    tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=mask_m[:, None])


# ==============================================================================
# 3. Python Launch Wrappers & Differentiable Autograd Interface
# ==============================================================================

def _validate_qkv_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[int, int, int, int, int]:
    if not (q.ndim == 4 and k.ndim == 4 and v.ndim == 4):
        raise ValueError(f"Q, K, V 必须为 4D [B, S, H, Dh] 张量，实际维度: {q.shape}, {k.shape}, {v.shape}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Triton FlashAttention 要求 Q, K, V 位于 CUDA 设备上")
    if not (q.device == k.device == v.device):
        raise ValueError(f"Q, K, V 必须位于同一 CUDA 设备上，实际: {q.device}, {k.device}, {v.device}")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Q, K, V 数据类型必须一致且属于 FP16/BF16/FP32，实际: {q.dtype}, {k.dtype}, {v.dtype}")

    B, Sq, Hq, Dh = q.shape
    Bk, Sk, Hkv, Dhk = k.shape
    if (Bk, Sk, Hkv, Dhk) != tuple(v.shape) or B != Bk or Dh != Dhk:
        raise ValueError(f"Q {q.shape}, K {k.shape}, V {v.shape} 形状不匹配")
    if B < 1 or Sq < 1 or Sk < 1:
        raise ValueError(f"Batch 与序列长度必须 >= 1，实际 B={B}, Sq={Sq}, Sk={Sk}")
    if Hq < 1 or Hkv < 1 or Hq % Hkv != 0:
        raise ValueError(f"GQA 头数必须满足 Hq % Hkv == 0，实际 Hq={Hq}, Hkv={Hkv}")
    if Dh not in (16, 32, 64, 128):
        raise ValueError(f"HeadDim 必须为 {{16, 32, 64, 128}} 之一，实际 Dh={Dh}")
    return B, Sq, Sk, Hq, Hkv


def _select_tile_config(dtype: torch.dtype, head_dim: int) -> Tuple[int, int, int, int]:
    if dtype == torch.float32 or head_dim >= 128:
        return 32, 32, 4, 1
    return 64, 64, 4, 2


def flash_attn_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
    q_pos: Optional[torch.Tensor] = None,
    k_pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """执行底层 Triton FlashAttention-2 前向，返回连续 `(out: [B, Sq, Hq, Dh], lse: [B, Hq, Sq] FP32)`"""
    B, Sq, Sk, Hq, Hkv = _validate_qkv_inputs(q, k, v)
    Dh = q.shape[-1]
    has_pos = (q_pos is not None) or (k_pos is not None)
    if has_pos:
        if q_pos is None or k_pos is None or q_pos.shape != (B, Sq) or k_pos.shape != (B, Sk):
            raise ValueError(f"q_pos / k_pos 形状必须分别为 {(B, Sq)} 与 {(B, Sk)}")
        if q_pos.device != q.device or k_pos.device != q.device:
            raise ValueError(f"q_pos ({q_pos.device}) / k_pos ({k_pos.device}) 必须与 Q ({q.device}) 同设备")
        qp_in = q_pos.to(torch.int32).contiguous()
        kp_in = k_pos.to(torch.int32).contiguous()
        stride_qpb = qp_in.stride(0)
        stride_kpb = kp_in.stride(0)
    else:
        if is_causal and Sq != Sk:
            raise ValueError(f"未提供显式位置索引时，is_causal=True 要求 Sq == Sk，实际 {Sq} != {Sk}")
        qp_in = q  # dummy pointer
        kp_in = k
        stride_qpb = 0
        stride_kpb = 0

    q_in = _ensure_contiguous_last_dim(q)
    k_in = _ensure_contiguous_last_dim(k)
    v_in = _ensure_contiguous_last_dim(v)

    if scale is None:
        scale = 1.0 / math.sqrt(Dh)

    out = torch.empty((B, Sq, Hq, Dh), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, Hq, Sq), dtype=torch.float32, device=q.device)

    block_m, block_n, num_warps, num_stages = _select_tile_config(q.dtype, Dh)
    FLASH_ATTN_COUNTERS["fwd"] += 1

    grid = (triton.cdiv(Sq, block_m), B * Hq)
    _flash_attn_fwd_kernel[grid](
        q_in,
        k_in,
        v_in,
        out,
        lse,
        qp_in,
        kp_in,
        q_in.stride(0), q_in.stride(1), q_in.stride(2),
        k_in.stride(0), k_in.stride(1), k_in.stride(2),
        v_in.stride(0), v_in.stride(1), v_in.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        stride_qpb,
        stride_kpb,
        Sq,
        Sk,
        scale,
        Hq=Hq,
        GROUP_SIZE=Hq // Hkv,
        HEAD_DIM=Dh,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        HAS_POS_IDS=has_pos,
        IS_FP32=(q.dtype == torch.float32),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


def compute_flash_bwd_delta(out: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    """预计算全局 `Delta_{b, h, i} = sum_d(O_{b, i, h, d} * dO_{b, i, h, d})` (`[B, Hq, Sq]` FP32)"""
    if (
        out.ndim != 4
        or not (out.is_cuda and grad_out.is_cuda)
        or grad_out.shape != out.shape
        or grad_out.device != out.device
        or grad_out.dtype != out.dtype
        or out.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError(
            f"compute_flash_bwd_delta 要求 out 与 grad_out 形状/CUDA设备/dtype 一致: "
            f"{out.shape}/{out.dtype}@{out.device} vs {grad_out.shape}/{grad_out.dtype}@{grad_out.device}"
        )
    B, Sq, Hq, Dh = out.shape
    if B < 1 or Sq < 1 or Hq < 1 or Dh not in (16, 32, 64, 128):
        raise ValueError(f"out 形状不合法: {out.shape}")
    o_in = _ensure_contiguous_last_dim(out)
    do_in = _ensure_contiguous_last_dim(grad_out)
    delta = torch.empty((B, Hq, Sq), dtype=torch.float32, device=out.device)
    block_m = 64 if Sq >= 64 else 32
    _flash_bwd_preprocess_kernel[(triton.cdiv(Sq, block_m), B * Hq)](
        o_in,
        do_in,
        delta,
        o_in.stride(0), o_in.stride(1), o_in.stride(2),
        do_in.stride(0), do_in.stride(1), do_in.stride(2),
        delta.stride(0), delta.stride(1),
        Sq,
        Hq=Hq,
        HEAD_DIM=Dh,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return delta


def flash_attn_backward_with_lse(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
    q_pos: Optional[torch.Tensor] = None,
    k_pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于全局 `LSE` 与 `Delta` 在片上 SRAM 重算局部注意力概率并求得 `(dQ, dK, dV)`：
    既可直接用于单卡 FlashAttention-2 反向，也可直接作为 Ring Attention 每步分片梯度核函数！
    """
    B, Sq, Sk, Hq, Hkv = _validate_qkv_inputs(q, k, v)
    Dh = q.shape[-1]
    if tuple(grad_out.shape) != (B, Sq, Hq, Dh) or grad_out.device != q.device or grad_out.dtype != q.dtype:
        raise ValueError(
            f"grad_out 形状/设备/dtype 不匹配: 期望 {(B, Sq, Hq, Dh)} {q.dtype} @ {q.device}，"
            f"实际 {grad_out.shape} {grad_out.dtype} @ {grad_out.device}"
        )
    if (
        tuple(lse.shape) != (B, Hq, Sq)
        or tuple(delta.shape) != (B, Hq, Sq)
        or lse.dtype != torch.float32
        or delta.dtype != torch.float32
        or lse.device != q.device
        or delta.device != q.device
    ):
        raise ValueError(
            f"lse/delta 契约不匹配: 期望 {(B, Hq, Sq)} float32 @ {q.device}，"
            f"实际 lse={lse.shape}/{lse.dtype}@{lse.device}, delta={delta.shape}/{delta.dtype}@{delta.device}"
        )
    has_pos = (q_pos is not None) or (k_pos is not None)
    if has_pos:
        if q_pos is None or k_pos is None or q_pos.shape != (B, Sq) or k_pos.shape != (B, Sk):
            raise ValueError(f"q_pos / k_pos 形状必须分别为 {(B, Sq)} 与 {(B, Sk)}")
        if q_pos.device != q.device or k_pos.device != q.device:
            raise ValueError(f"q_pos ({q_pos.device}) / k_pos ({k_pos.device}) 必须与 Q ({q.device}) 同设备")
        qp_in = q_pos.to(torch.int32).contiguous()
        kp_in = k_pos.to(torch.int32).contiguous()
        stride_qpb = qp_in.stride(0)
        stride_kpb = kp_in.stride(0)
    else:
        if is_causal and Sq != Sk:
            raise ValueError(f"未提供显式位置索引时，is_causal=True 要求 Sq == Sk，实际 {Sq} != {Sk}")
        qp_in = q
        kp_in = k
        stride_qpb = 0
        stride_kpb = 0

    q_in = _ensure_contiguous_last_dim(q)
    k_in = _ensure_contiguous_last_dim(k)
    v_in = _ensure_contiguous_last_dim(v)
    do_in = _ensure_contiguous_last_dim(grad_out)
    lse_c = lse.contiguous()
    delta_c = delta.contiguous()

    if scale is None:
        scale = 1.0 / math.sqrt(Dh)

    dq = torch.empty((B, Sq, Hq, Dh), dtype=q.dtype, device=q.device)
    dk = torch.empty((B, Sk, Hkv, Dh), dtype=k.dtype, device=k.device)
    dv = torch.empty((B, Sk, Hkv, Dh), dtype=v.dtype, device=v.device)

    block_m, block_n, num_warps, num_stages = _select_tile_config(q.dtype, Dh)
    FLASH_ATTN_COUNTERS["bwd"] += 1

    _flash_bwd_dk_dv_kernel[(triton.cdiv(Sk, block_n), B * Hkv)](
        q_in,
        k_in,
        v_in,
        do_in,
        lse_c,
        delta_c,
        dk,
        dv,
        qp_in,
        kp_in,
        q_in.stride(0), q_in.stride(1), q_in.stride(2),
        k_in.stride(0), k_in.stride(1), k_in.stride(2),
        v_in.stride(0), v_in.stride(1), v_in.stride(2),
        do_in.stride(0), do_in.stride(1), do_in.stride(2),
        lse_c.stride(0), lse_c.stride(1),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        stride_qpb,
        stride_kpb,
        Sq,
        Sk,
        scale,
        Hkv=Hkv,
        GROUP_SIZE=Hq // Hkv,
        HEAD_DIM=Dh,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        HAS_POS_IDS=has_pos,
        IS_FP32=(q.dtype == torch.float32),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _flash_bwd_dq_kernel[(triton.cdiv(Sq, block_m), B * Hq)](
        q_in,
        k_in,
        v_in,
        do_in,
        lse_c,
        delta_c,
        dq,
        qp_in,
        kp_in,
        q_in.stride(0), q_in.stride(1), q_in.stride(2),
        k_in.stride(0), k_in.stride(1), k_in.stride(2),
        v_in.stride(0), v_in.stride(1), v_in.stride(2),
        do_in.stride(0), do_in.stride(1), do_in.stride(2),
        lse_c.stride(0), lse_c.stride(1),
        dq.stride(0), dq.stride(1), dq.stride(2),
        stride_qpb,
        stride_kpb,
        Sq,
        Sk,
        scale,
        Hq=Hq,
        GROUP_SIZE=Hq // Hkv,
        HEAD_DIM=Dh,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        HAS_POS_IDS=has_pos,
        IS_FP32=(q.dtype == torch.float32),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv


class TritonFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = True,
        scale: Optional[float] = None,
        q_pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eff_scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else float(scale)
        out, lse = flash_attn_forward_with_lse(
            q, k, v, is_causal=is_causal, scale=eff_scale, q_pos=q_pos, k_pos=k_pos
        )
        ctx.is_causal = is_causal
        ctx.scale = eff_scale
        ctx.has_pos = (q_pos is not None) and (k_pos is not None)
        if ctx.has_pos:
            ctx.save_for_backward(q, k, v, out, lse, q_pos, k_pos)
        else:
            ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.has_pos:
            q, k, v, out, lse, q_pos, k_pos = ctx.saved_tensors
        else:
            q, k, v, out, lse = ctx.saved_tensors
            q_pos, k_pos = None, None

        delta = compute_flash_bwd_delta(out, grad_out)
        dq, dk, dv = flash_attn_backward_with_lse(
            grad_out,
            q,
            k,
            v,
            lse,
            delta,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
            q_pos=q_pos,
            k_pos=k_pos,
        )
        return dq, dk, dv, None, None, None, None


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
    q_pos: Optional[torch.Tensor] = None,
    k_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """对外可求导 Triton GQA FlashAttention-2 接口 (`[B, S, Hq, Dh]` -> `[B, S, Hq, Dh]`)"""
    return TritonFlashAttentionFunction.apply(q, k, v, is_causal, scale, q_pos, k_pos)  # type: ignore[return-value]
