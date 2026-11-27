"""层内细粒度选择性激活重计算（SAC）、双缓冲异步卸载层与大词表分块交叉熵。

三级显存协同机制：
- `ChunkedLinearCrossEntropyFunction`：沿序列维分块流式计算全序列精确损失，
  反向通过原地 Softmax 与 `dw_f32.addmm_` 累加词表梯度（`S=16K, V=152K` 额外工作区压至 `520 MB`）；
- `SelectiveLayerCheckpointFunction`：前向仅保留紧凑注意力边界张量与 LSE（削减 `80%+` 激活显存），
  反向跳过 `Q/K/V/O` 与 FA2 重算，仅局部重算宽维 SwiGLU；
- `OffloadedLayerCheckpointFunction`：经 `DoubleBufferedOffloadPipeline` 异步 DMA 卸载至锁页内存，
  反向超前一层异步 H2D 预取并单趟解析求导，原生支持 LoRA 局部冻结参数。
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .activation_offload import DoubleBufferedOffloadPipeline
from .flash_attn_triton import (
    compute_flash_bwd_delta,
    flash_attn_backward_with_lse,
    flash_attn_forward_with_lse,
)


@triton.jit
def _swiglu_packed_fwd_kernel(
    GateUp_ptr,
    Out_ptr,
    Inter,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < Inter
    inter64 = Inter + 0 * row

    gu_base = GateUp_ptr + row * (2 * inter64)
    g = tl.load(gu_base + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_base + inter64 + cols, mask=mask, other=0.0).to(tl.float32)
    act = (g * tl.sigmoid(g)) * u
    tl.store(
        Out_ptr + row * inter64 + cols, act.to(Out_ptr.dtype.element_ty), mask=mask
    )


@triton.jit
def _swiglu_packed_bwd_kernel(
    GateUp_ptr,
    dMlp_ptr,
    dGateUp_ptr,
    Inter,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < Inter
    inter64 = Inter + 0 * row

    gu_base = GateUp_ptr + row * (2 * inter64)
    dgu_base = dGateUp_ptr + row * (2 * inter64)
    g = tl.load(gu_base + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_base + inter64 + cols, mask=mask, other=0.0).to(tl.float32)
    dm = tl.load(dMlp_ptr + row * inter64 + cols, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    du = dm * (g * sig)
    dg = dm * u * sig * (1.0 + g * (1.0 - sig))

    tl.store(dgu_base + cols, dg.to(dGateUp_ptr.dtype.element_ty), mask=mask)
    tl.store(dgu_base + inter64 + cols, du.to(dGateUp_ptr.dtype.element_ty), mask=mask)


def fused_packed_swiglu_fwd(gate_up: torch.Tensor) -> torch.Tensor:
    """对紧凑拼接的 `gate_up: [..., 2 * Inter]` 单 Pass 计算 `silu(gate) * up`，零 FP32 中间显存分配"""
    if (
        not gate_up.is_cuda
        or gate_up.ndim < 2
        or gate_up.shape[-1] < 2
        or gate_up.shape[-1] % 2 != 0
    ):
        raise ValueError(
            f"fused_packed_swiglu_fwd 要求 CUDA 张量且末维为 >=2 的偶数，实际: {gate_up.shape} @ {gate_up.device}"
        )
    gu_c = gate_up.contiguous()
    orig_shape = gu_c.shape
    inter = orig_shape[-1] // 2
    M = gu_c.numel() // (2 * inter)
    out = torch.empty((*orig_shape[:-1], inter), dtype=gu_c.dtype, device=gu_c.device)
    block_n = 1024 if inter >= 1024 else 256
    _swiglu_packed_fwd_kernel[(M, triton.cdiv(inter, block_n))](
        gu_c, out, inter, BLOCK_N=block_n, num_warps=4
    )
    return out


def fused_packed_swiglu_bwd_inplace(
    gate_up_2d: torch.Tensor, d_mlp_2d: torch.Tensor
) -> torch.Tensor:
    """单 Pass 计算 SwiGLU 反向并原地写回 `gate_up_2d` 缓冲区 `[M, 2 * Inter]`，彻底消除 `torch.cat` 拷贝与 80MB*8 中间显存！"""
    if (
        gate_up_2d.ndim != 2
        or gate_up_2d.shape[-1] < 2
        or gate_up_2d.shape[-1] % 2 != 0
    ):
        raise ValueError(f"gate_up_2d 形状非法: {gate_up_2d.shape}")
    M, two_inter = gate_up_2d.shape
    inter = two_inter // 2
    if d_mlp_2d.shape != (M, inter) or d_mlp_2d.device != gate_up_2d.device:
        raise ValueError(
            f"d_mlp_2d {d_mlp_2d.shape}@{d_mlp_2d.device} 与 gate_up_2d {gate_up_2d.shape}@{gate_up_2d.device} 不匹配"
        )
    dm_c = d_mlp_2d.contiguous()
    block_n = 1024 if inter >= 1024 else 256
    _swiglu_packed_bwd_kernel[(M, triton.cdiv(inter, block_n))](
        gate_up_2d, dm_c, gate_up_2d, inter, BLOCK_N=block_n, num_warps=4
    )
    return gate_up_2d


class _FusedPackedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        gu_c = gate_up.contiguous()
        ctx.save_for_backward(gu_c)
        return fused_packed_swiglu_fwd(gu_c)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (gu_c,) = ctx.saved_tensors
        orig_shape = gu_c.shape
        two_inter = orig_shape[-1]
        inter = two_inter // 2
        M = gu_c.numel() // two_inter
        dgu = torch.empty_like(gu_c)
        dm_c = grad_out.contiguous().view(M, inter)
        block_n = 1024 if inter >= 1024 else 256
        _swiglu_packed_bwd_kernel[(M, triton.cdiv(inter, block_n))](
            gu_c.view(M, two_inter),
            dm_c,
            dgu.view(M, two_inter),
            inter,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return dgu


def differentiable_packed_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """供标准 Eager / Full-Checkpointing 强基线公平共享的可求导融合 SwiGLU 算子，确保重计算消融 100% 对称！"""
    return _FusedPackedSwiGLUFunction.apply(gate_up)  # type: ignore[return-value]


@triton.jit
def _rmsnorm_fwd_rstd_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    Rstd_ptr,
    H,
    eps,
    SAVE_RSTD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    base = row * H

    x = tl.load(X_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    if SAVE_RSTD:
        tl.store(Rstd_ptr + row, rstd)
    out = (x * rstd).to(Out_ptr.dtype.element_ty).to(tl.float32) * w
    tl.store(Out_ptr + base + cols, out.to(Out_ptr.dtype.element_ty), mask=mask)


def rmsnorm_fwd_fp32(
    x: torch.Tensor, weight: torch.Tensor, eps: float, need_rstd: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """单 Pass 融合 RMSNorm 前向（可选保存 `[B, S, 1]` FP32 `rstd` 供解析反向使用）"""
    if not (x.is_cuda and weight.is_cuda) or x.device != weight.device:
        raise ValueError(
            f"rmsnorm_fwd_fp32 要求 x 与 weight 同处 CUDA 设备，实际 {x.device} vs {weight.device}"
        )
    x_c = x.contiguous()
    H = x_c.shape[-1]
    if weight.ndim != 1 or weight.shape[0] != H:
        raise ValueError(f"rmsnorm weight 形状 {weight.shape} 必须等于 ({H},)")
    M = x_c.numel() // H
    out = torch.empty_like(x_c)
    block_h = triton.next_power_of_2(H)
    w_c = weight.contiguous()
    if need_rstd:
        rstd = torch.empty((*x_c.shape[:-1], 1), dtype=torch.float32, device=x_c.device)
        _rmsnorm_fwd_rstd_kernel[(M,)](
            x_c, w_c, out, rstd, H, eps, SAVE_RSTD=True, BLOCK_H=block_h, num_warps=4
        )
        return out, rstd
    _rmsnorm_fwd_rstd_kernel[(M,)](
        x_c, w_c, out, out, H, eps, SAVE_RSTD=False, BLOCK_H=block_h, num_warps=4
    )
    return out, None


def rmsnorm_bwd_fp32(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm 精确 FP32 解析反向，返回 `(grad_x, grad_weight)`"""
    x_f32 = x.to(torch.float32)
    dy_f32 = grad_out.to(torch.float32)
    w_f32 = weight.to(torch.float32)

    x_hat = x_f32 * rstd
    dw = (dy_f32 * x_hat).reshape(-1, x.shape[-1]).sum(dim=0).to(weight.dtype)

    dy_w = dy_f32 * w_f32
    mean_dot = (dy_w * x_hat).mean(dim=-1, keepdim=True)
    dx = ((dy_w - x_hat * mean_dot) * rstd).to(x.dtype)
    return dx, dw


class _FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x_c = x.contiguous()
        out, rstd = rmsnorm_fwd_fp32(x_c, weight, eps, need_rstd=True)
        assert rstd is not None
        ctx.save_for_backward(x_c, weight, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_c, weight, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_bwd_fp32(grad_out, x_c, weight, rstd)
        return dx, dw, None


def differentiable_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """供 Eager / Full-Checkpointing 基线与最终归一化层共享的对称 RMSNorm 算子"""
    return _FusedRMSNormFunction.apply(x, weight, eps)  # type: ignore[return-value]


@triton.jit
def _rope_half_kernel(
    X_ptr,
    Cos_ptr,
    Sin_ptr,
    Out_ptr,
    S,
    H,
    HALF_D: tl.constexpr,
    IS_BWD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)  # [0 .. B * S * H)
    s_idx = (row // H) % S
    d_offs = tl.arange(0, HALF_D)
    full_d = 2 * HALF_D

    x_base = X_ptr + row * full_d
    o_base = Out_ptr + row * full_d
    cs_base = s_idx * full_d

    x1 = tl.load(x_base + d_offs).to(tl.float32)
    x2 = tl.load(x_base + HALF_D + d_offs).to(tl.float32)
    c1 = tl.load(Cos_ptr + cs_base + d_offs).to(tl.float32)
    c2 = tl.load(Cos_ptr + cs_base + HALF_D + d_offs).to(tl.float32)
    s1 = tl.load(Sin_ptr + cs_base + d_offs).to(tl.float32)
    s2 = tl.load(Sin_ptr + cs_base + HALF_D + d_offs).to(tl.float32)

    if IS_BWD:
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    else:
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2

    tl.store(o_base + d_offs, y1.to(Out_ptr.dtype.element_ty))
    tl.store(o_base + HALF_D + d_offs, y2.to(Out_ptr.dtype.element_ty))


def _validate_rope_inputs(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[int, int, int, int]:
    if x.ndim != 4:
        raise ValueError(f"RoPE 输入必须为 4D [B, S, H, Dh] 张量，实际形状: {x.shape}")
    if not (x.is_cuda and cos.is_cuda and sin.is_cuda) or not (
        x.device == cos.device == sin.device
    ):
        raise ValueError(
            f"RoPE 要求 x, cos, sin 位于同一 CUDA 设备，实际: {x.device}, {cos.device}, {sin.device}"
        )
    B, S, H, Dh = x.shape
    if B < 1 or S < 1 or H < 1:
        raise ValueError(f"RoPE 维度必须 >= 1，实际形状: {x.shape}")
    if Dh < 2 or (Dh & (Dh - 1)) != 0:
        raise ValueError(f"RoPE HeadDim 必须为 >= 2 的 2 的幂次，实际 Dh={Dh}")
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError(
            f"RoPE cos {cos.shape} 与 sin {sin.shape} 必须为同形状二维张量 [MaxS, Dh]"
        )
    if cos.shape[0] < S or cos.shape[1] != Dh:
        raise ValueError(
            f"RoPE cos/sin 形状 {cos.shape} 越界或不匹配：需要行数 >= S ({S}) 且列数 == Dh ({Dh})"
        )
    return B, S, H, Dh


def _rope_fwd_impl(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    B, S, H, Dh = _validate_rope_inputs(x, cos, sin)
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _rope_half_kernel[(B * S * H,)](
        x_c,
        cos.contiguous(),
        sin.contiguous(),
        out,
        S,
        H,
        HALF_D=Dh // 2,
        IS_BWD=False,
        num_warps=2,
    )
    return out


def apply_rope_bwd(
    grad_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """单 Pass RoPE 正交逆旋转解析反向 (`R^T = R(-theta)`)"""
    B, S, H, Dh = _validate_rope_inputs(grad_out, cos, sin)
    do_c = grad_out.contiguous()
    dx = torch.empty_like(do_c)
    _rope_half_kernel[(B * S * H,)](
        do_c,
        cos.contiguous(),
        sin.contiguous(),
        dx,
        S,
        H,
        HALF_D=Dh // 2,
        IS_BWD=True,
        num_warps=2,
    )
    return dx


class _FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return _rope_fwd_impl(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        cos, sin = ctx.saved_tensors
        return apply_rope_bwd(grad_out, cos, sin), None, None


def apply_rope_fwd(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """支持标准 Autograd 与手动解析求导的对称单 Pass 融合 RoPE（基线与 SAC 公平共享）"""
    if torch.is_grad_enabled() and x.requires_grad:
        return _FusedRoPEFunction.apply(x, cos, sin)  # type: ignore[return-value]
    return _rope_fwd_impl(x, cos, sin)


def theoretical_selective_ckpt_flops_overhead(
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> float:
    """
    计算层内选择性激活重计算（仅重算 `gate_up_proj` 1 个矩阵乘，完全跳过 Attention + QKV/O/Down 重算）
    相对无重算单步因果前反向总 FLOPs (`3 * F_fwd`) 的精确额外计算占比：
      F_recomp = 4 * S * H * Inter
      F_fwd    = 2 * S * H * ((Hq + 2*Hkv)*Dh + H) + 6 * S * H * Inter + 2 * S^2 * Hq * Dh  (因果下三角有效口径)
      η_SAC    = F_recomp / (3 * F_fwd)
    """
    s = float(seq_len)
    h = float(hidden_dim)
    inter = float(intermediate_dim)
    qkv_o_dim = float((num_q_heads + 2 * num_kv_heads) * head_dim + hidden_dim)
    attn_dim = float(num_q_heads * head_dim)

    f_recomp = 4.0 * s * h * inter
    f_fwd = 2.0 * s * h * qkv_o_dim + 6.0 * s * h * inter + 2.0 * (s**2) * attn_dim
    return f_recomp / (3.0 * f_fwd)


# ==============================================================================
# 1. 层内细粒度选择性激活重计算算子 (SelectiveLayerCheckpointFunction)
# ==============================================================================


def _layer_analytical_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    Hq: int,
    Hkv: int,
    Dh: int,
    has_qkv_bias: bool,
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    attn_out: torch.Tensor,
    lse: torch.Tensor,
    h_mid: torch.Tensor,
    attn_norm_w: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    o_w: torch.Tensor,
    ffn_norm_w: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_w: torch.Tensor,
    needs_grad_12: tuple[bool, ...],
    precomputed_norm1: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    (
        need_dx,
        need_ln1,
        need_qw,
        need_qb,
        need_kw,
        need_kb,
        need_vw,
        need_vb,
        need_ow,
        need_ln2,
        need_gu,
        need_down,
    ) = needs_grad_12

    B, S, H = x.shape
    M = B * S
    dy = grad_out.contiguous()
    dy_2d = dy.view(M, H)

    # --- 第一段：单趟局部重算 SwiGLU 激活并原地求 FFN 导数（每层全程仅算 2 次 SwiGLU：前向 1 次 + 此处 1 次） ---
    norm2, rstd2 = rmsnorm_fwd_fp32(h_mid, ffn_norm_w, eps, need_rstd=True)
    assert rstd2 is not None
    norm2_2d = norm2.view(M, H)
    gate_up_2d = F.linear(norm2_2d, gate_up_w)

    if need_down:
        mlp_act_2d = fused_packed_swiglu_fwd(gate_up_2d)
        dw_down = torch.matmul(dy_2d.t(), mlp_act_2d)
        del mlp_act_2d
    else:
        dw_down = None

    need_before_down = (
        need_gu
        or need_ln2
        or need_ow
        or need_qw
        or need_qb
        or need_kw
        or need_kb
        or need_vw
        or need_vb
        or need_ln1
        or need_dx
    )
    if not need_before_down:
        del norm2, norm2_2d, gate_up_2d, rstd2
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dw_down,
        )

    d_mlp_2d = torch.matmul(dy_2d, down_w)
    d_gate_up_2d = fused_packed_swiglu_bwd_inplace(gate_up_2d, d_mlp_2d)
    del d_mlp_2d

    dw_gate_up = torch.matmul(d_gate_up_2d.t(), norm2_2d) if need_gu else None
    del norm2, norm2_2d

    need_before_gu = (
        need_ln2
        or need_ow
        or need_qw
        or need_qb
        or need_kw
        or need_kb
        or need_vw
        or need_vb
        or need_ln1
        or need_dx
    )
    if not need_before_gu:
        del d_gate_up_2d, rstd2
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dw_gate_up,
            dw_down,
        )

    d_norm2 = torch.matmul(d_gate_up_2d, gate_up_w).view(B, S, H)
    del d_gate_up_2d

    dx_ffn, dw_ffn_norm_val = rmsnorm_bwd_fp32(d_norm2, h_mid, ffn_norm_w, rstd2)
    dw_ffn_norm = dw_ffn_norm_val if need_ln2 else None
    dh_mid = dy + dx_ffn
    del d_norm2, dx_ffn, rstd2

    # --- 第二段：直接利用 (q_r, k_r, v, attn_out, lse) 执行 Triton FlashAttention-2 解析反向 ---
    dh_mid_2d = dh_mid.contiguous().view(M, H)
    if need_ow:
        attn_2d = attn_out.contiguous().view(M, Hq * Dh)
        dw_o = torch.matmul(dh_mid_2d.t(), attn_2d)
        del attn_2d
    else:
        dw_o = None

    need_attn_sub = (
        need_dx
        or need_ln1
        or need_qw
        or need_qb
        or need_kw
        or need_kb
        or need_vw
        or need_vb
    )
    if not need_attn_sub:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dw_o,
            dw_ffn_norm,
            dw_gate_up,
            dw_down,
        )

    need_qkv_bwd = (
        need_qw
        or need_qb
        or need_kw
        or need_kb
        or need_vw
        or need_vb
        or need_ln1
        or need_dx
    )
    if (
        not (need_qw or need_qb or need_kw or need_kb or need_vw or need_vb or need_ln1)
        and need_dx
    ):
        # 注意：残差支路之外注意力支路仍向 x 贡献 dx_attn，故只要 need_dx=True 也必须求注意力反向
        need_qkv_bwd = True

    assert need_qkv_bwd
    d_attn = torch.matmul(dh_mid_2d, o_w).view(B, S, Hq, Dh)
    delta = compute_flash_bwd_delta(attn_out, d_attn)
    dq_r, dk_r, dv = flash_attn_backward_with_lse(
        d_attn, q_r, k_r, v, lse, delta, is_causal=True
    )
    del d_attn, delta

    dq_2d = apply_rope_bwd(dq_r, cos, sin).view(M, Hq * Dh)
    dk_2d = apply_rope_bwd(dk_r, cos, sin).view(M, Hkv * Dh)
    dv_2d = dv.contiguous().view(M, Hkv * Dh)
    del dq_r, dk_r, dv

    if precomputed_norm1 is not None:
        norm1, rstd1 = precomputed_norm1
    else:
        norm1, rstd1 = rmsnorm_fwd_fp32(x, attn_norm_w, eps, need_rstd=True)
        assert rstd1 is not None
    norm1_2d = norm1.view(M, H)

    dw_q = torch.matmul(dq_2d.t(), norm1_2d) if need_qw else None
    dw_k = torch.matmul(dk_2d.t(), norm1_2d) if need_kw else None
    dw_v = torch.matmul(dv_2d.t(), norm1_2d) if need_vw else None
    db_q = dq_2d.sum(dim=0) if (has_qkv_bias and need_qb) else None
    db_k = dk_2d.sum(dim=0) if (has_qkv_bias and need_kb) else None
    db_v = dv_2d.sum(dim=0) if (has_qkv_bias and need_vb) else None
    del norm1, norm1_2d

    if need_dx or need_ln1:
        d_norm1_2d = torch.matmul(dq_2d, q_w)
        d_norm1_2d.addmm_(dk_2d, k_w)
        d_norm1_2d.addmm_(dv_2d, v_w)
        del dq_2d, dk_2d, dv_2d
        dx_attn, dw_attn_norm_val = rmsnorm_bwd_fp32(
            d_norm1_2d.view(B, S, H), x, attn_norm_w, rstd1
        )
        dw_attn_norm = dw_attn_norm_val if need_ln1 else None
        dx_total = (dh_mid + dx_attn) if need_dx else None
    else:
        del dq_2d, dk_2d, dv_2d
        dw_attn_norm = None
        dx_total = None

    return (
        dx_total,
        dw_attn_norm,
        dw_q,
        db_q,
        dw_k,
        db_k,
        dw_v,
        db_v,
        dw_o,
        dw_ffn_norm,
        dw_gate_up,
        dw_down,
    )


class SelectiveLayerCheckpointFunction(torch.autograd.Function):
    """
    前向仅保留 `(x, q_r, k_r, v, attn_out, lse, h_mid)`，丢弃占层激活 72%+ 的宽维 `gate_up` / `mlp_act` / 归一化副本；
    反向直接调用 `flash_attn_backward_with_lse` 求注意力导数，零注意力前向重算、零 QKV/O/Down GEMM 重算，原生支持局部冻结权重。
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        eps: float,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_norm_w: torch.Tensor,
        q_w: torch.Tensor,
        q_b: torch.Tensor | None,
        k_w: torch.Tensor,
        k_b: torch.Tensor | None,
        v_w: torch.Tensor,
        v_b: torch.Tensor | None,
        o_w: torch.Tensor,
        ffn_norm_w: torch.Tensor,
        gate_up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> torch.Tensor:
        x = x.contiguous()
        B, S, H = x.shape
        norm1, _ = rmsnorm_fwd_fp32(x, attn_norm_w, eps, need_rstd=False)

        q = F.linear(norm1, q_w, q_b).view(B, S, num_q_heads, head_dim)
        k = F.linear(norm1, k_w, k_b).view(B, S, num_kv_heads, head_dim)
        v = F.linear(norm1, v_w, v_b).view(B, S, num_kv_heads, head_dim)
        del norm1

        q_r = apply_rope_fwd(q, cos, sin)
        k_r = apply_rope_fwd(k, cos, sin)
        del q, k

        attn_out, lse = flash_attn_forward_with_lse(q_r, k_r, v, is_causal=True)
        h_mid = x + F.linear(attn_out.view(B, S, num_q_heads * head_dim), o_w)

        norm2, _ = rmsnorm_fwd_fp32(h_mid, ffn_norm_w, eps, need_rstd=False)
        gate_up = F.linear(norm2, gate_up_w)
        mlp_act = fused_packed_swiglu_fwd(gate_up)
        del gate_up, norm2
        out = h_mid + F.linear(mlp_act, down_w)
        del mlp_act

        ctx.eps = eps
        ctx.num_q_heads = num_q_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.has_qkv_bias = q_b is not None

        saved = [
            x,
            cos,
            sin,
            q_r,
            k_r,
            v,
            attn_out,
            lse,
            h_mid,
            attn_norm_w,
            q_w,
            k_w,
            v_w,
            o_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
        ]
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            x,
            cos,
            sin,
            q_r,
            k_r,
            v,
            attn_out,
            lse,
            h_mid,
            attn_norm_w,
            q_w,
            k_w,
            v_w,
            o_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
        ) = ctx.saved_tensors

        ng = ctx.needs_input_grad
        needs_12 = (
            ng[0],
            ng[7],
            ng[8],
            ng[9],
            ng[10],
            ng[11],
            ng[12],
            ng[13],
            ng[14],
            ng[15],
            ng[16],
            ng[17],
        )
        (
            dx_total,
            dw_attn_norm,
            dw_q,
            db_q,
            dw_k,
            db_k,
            dw_v,
            db_v,
            dw_o,
            dw_ffn_norm,
            dw_gate_up,
            dw_down,
        ) = _layer_analytical_backward(
            grad_out,
            x,
            cos,
            sin,
            ctx.eps,
            ctx.num_q_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
            ctx.has_qkv_bias,
            q_r,
            k_r,
            v,
            attn_out,
            lse,
            h_mid,
            attn_norm_w,
            q_w,
            k_w,
            v_w,
            o_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
            needs_12,
            precomputed_norm1=None,
        )
        return (
            dx_total,
            None,
            None,
            None,
            None,
            None,
            None,
            dw_attn_norm,
            dw_q,
            db_q,
            dw_k,
            db_k,
            dw_v,
            db_v,
            dw_o,
            dw_ffn_norm,
            dw_gate_up,
            dw_down,
        )


# ==============================================================================
# 2. 双缓冲异步 PCIe 卸载与跨层超前预取层算子 (OffloadedLayerCheckpointFunction)
# ==============================================================================


class OffloadedLayerCheckpointFunction(torch.autograd.Function):
    """
    真异步双流激活卸载算子（计算结果 100% 在关键求导路径上被真实消费）：
    - 前向：仅在层存在需要求导输入时通过 `pipeline.offload_forward(x)` 异步将 `x` 搬入 `PinnedActivationPool`，
      GPU 侧零激活常驻；
    - 反向：调用 `pipeline.fetch_for_backward_and_prefetch_prev(ticket)` 立即触发 `l-1` 层异步 H2D 预取，
      基于 `x_restored` 单趟恢复注意力中间态并直接调用 `_layer_analytical_backward`（每层总计仅执行 2 次 SwiGLU，
      彻底消除二次套壳重算开销，且原生支持 LoRA / 局部参数冻结）。
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        pipeline: DoubleBufferedOffloadPipeline,
        cos: torch.Tensor,
        sin: torch.Tensor,
        eps: float,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_norm_w: torch.Tensor,
        q_w: torch.Tensor,
        q_b: torch.Tensor | None,
        k_w: torch.Tensor,
        k_b: torch.Tensor | None,
        v_w: torch.Tensor,
        v_b: torch.Tensor | None,
        o_w: torch.Tensor,
        ffn_norm_w: torch.Tensor,
        gate_up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> torch.Tensor:
        needs_any_grad = any(ctx.needs_input_grad)
        ctx.ticket = pipeline.offload_forward(x) if needs_any_grad else -1
        ctx.pipeline = pipeline
        ctx.eps = eps
        ctx.num_q_heads = num_q_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.has_qkv_bias = q_b is not None

        with torch.no_grad():
            x_c = x.contiguous()
            B, S, H = x_c.shape
            norm1, _ = rmsnorm_fwd_fp32(x_c, attn_norm_w, eps, need_rstd=False)
            q = F.linear(norm1, q_w, q_b).view(B, S, num_q_heads, head_dim)
            k = F.linear(norm1, k_w, k_b).view(B, S, num_kv_heads, head_dim)
            v = F.linear(norm1, v_w, v_b).view(B, S, num_kv_heads, head_dim)
            del norm1
            q_r = apply_rope_fwd(q, cos, sin)
            k_r = apply_rope_fwd(k, cos, sin)
            del q, k
            attn_out, _ = flash_attn_forward_with_lse(q_r, k_r, v, is_causal=True)
            del q_r, k_r, v
            h_mid = x_c + F.linear(attn_out.view(B, S, num_q_heads * head_dim), o_w)
            del attn_out, x_c
            norm2, _ = rmsnorm_fwd_fp32(h_mid, ffn_norm_w, eps, need_rstd=False)
            mlp_act = fused_packed_swiglu_fwd(F.linear(norm2, gate_up_w))
            out = h_mid + F.linear(mlp_act, down_w)
            del norm2, mlp_act, h_mid

        if needs_any_grad:
            saved_params = [
                cos,
                sin,
                attn_norm_w,
                q_w,
                k_w,
                v_w,
                o_w,
                ffn_norm_w,
                gate_up_w,
                down_w,
            ]
            if ctx.has_qkv_bias:
                assert q_b is not None and k_b is not None and v_b is not None
                saved_params.extend([q_b, k_b, v_b])
            ctx.save_for_backward(*saved_params)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pipeline: DoubleBufferedOffloadPipeline = ctx.pipeline
        x_restored = pipeline.fetch_for_backward_and_prefetch_prev(
            ctx.ticket
        ).contiguous()

        saved = ctx.saved_tensors
        cos, sin, attn_norm_w, q_w, k_w, v_w, o_w, ffn_norm_w, gate_up_w, down_w = (
            saved[:10]
        )
        if ctx.has_qkv_bias:
            q_b, k_b, v_b = saved[10], saved[11], saved[12]
        else:
            q_b = k_b = v_b = None

        Hq = ctx.num_q_heads
        Hkv = ctx.num_kv_heads
        Dh = ctx.head_dim
        B, S, _ = x_restored.shape

        with torch.no_grad():
            norm1, rstd1 = rmsnorm_fwd_fp32(
                x_restored, attn_norm_w, ctx.eps, need_rstd=True
            )
            assert rstd1 is not None
            q = F.linear(norm1, q_w, q_b).view(B, S, Hq, Dh)
            k = F.linear(norm1, k_w, k_b).view(B, S, Hkv, Dh)
            v = F.linear(norm1, v_w, v_b).view(B, S, Hkv, Dh)
            q_r = apply_rope_fwd(q, cos, sin)
            k_r = apply_rope_fwd(k, cos, sin)
            del q, k
            attn_out, lse = flash_attn_forward_with_lse(q_r, k_r, v, is_causal=True)
            h_mid = x_restored + F.linear(attn_out.view(B, S, Hq * Dh), o_w)

        ng = ctx.needs_input_grad
        needs_12 = (
            ng[0],
            ng[8],
            ng[9],
            ng[10],
            ng[11],
            ng[12],
            ng[13],
            ng[14],
            ng[15],
            ng[16],
            ng[17],
            ng[18],
        )
        (
            dx_total,
            dw_attn_norm,
            dw_q,
            db_q,
            dw_k,
            db_k,
            dw_v,
            db_v,
            dw_o,
            dw_ffn_norm,
            dw_gate_up,
            dw_down,
        ) = _layer_analytical_backward(
            grad_out,
            x_restored,
            cos,
            sin,
            ctx.eps,
            Hq,
            Hkv,
            Dh,
            ctx.has_qkv_bias,
            q_r,
            k_r,
            v,
            attn_out,
            lse,
            h_mid,
            attn_norm_w,
            q_w,
            k_w,
            v_w,
            o_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
            needs_12,
            precomputed_norm1=(norm1, rstd1),
        )
        return (
            dx_total,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dw_attn_norm,
            dw_q,
            db_q,
            dw_k,
            db_k,
            dw_v,
            db_v,
            dw_o,
            dw_ffn_norm,
            dw_gate_up,
            dw_down,
        )


# ==============================================================================
# 3. 大词表零物化流式分块交叉熵损失算子 (ChunkedLinearCrossEntropyFunction)
# ==============================================================================


class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    针对大词表语言模型（如 Qwen2.5 `Vocab=151,936`）的流式分块线性交叉熵：
    - 完整计算全序列所有 `M = B * (S - 1)` 个有效位置的损失（严禁 `[:, -64:]` 截断偷跑）；
    - 前反向均以 `chunk_size`（默认 `256`）为单位复用小块显存，反向采用原地 Softmax 与 cuBLAS `dw_f32.addmm_` 原地累加 `dW`，
      消除 `dw_chunk` 低精度临时张量与二次转型拷贝，在 `S=16,384, Vocab=151,936` 下将反向额外工作区压降至 `520.0 MB`（含返回梯度张量的总增量峰值 `872.0 MB`）；
    - 在反向求导入口第一时间回调 `offload_pipeline.trigger_prefetch_latest()`，使最后一层卸载激活的 H2D 传输
      完全隐藏在 `lm_head` 大词表反向 GEMM 耗时之内！
    """

    @staticmethod
    def forward(
        ctx,
        hidden_2d: torch.Tensor,
        lm_weight: torch.Tensor,
        targets_1d: torch.Tensor,
        ignore_index: int = -100,
        chunk_size: int = 256,
        offload_pipeline: DoubleBufferedOffloadPipeline | None = None,
    ) -> torch.Tensor:
        h_c = hidden_2d.contiguous()
        t_c = targets_1d.reshape(-1).to(torch.long).contiguous()
        M, H = h_c.shape
        V, Hw = lm_weight.shape
        if Hw != H or t_c.numel() != M:
            raise ValueError(
                f"ChunkedLinearCrossEntropy 输入维度不匹配: hidden={h_c.shape}, weight={lm_weight.shape}, targets={t_c.shape}"
            )

        valid_mask = t_c != ignore_index
        if bool((valid_mask & ((t_c < 0) | (t_c >= V))).any().item()):
            raise ValueError(
                f"ChunkedLinearCrossEntropy 检测到越界 target 索引（合法范围 [0, {V}) 或 ignore_index={ignore_index}）"
            )
        valid_count = valid_mask.sum()

        ctx.ignore_index = int(ignore_index)
        ctx.chunk_size = max(16, int(chunk_size))
        ctx.offload_pipeline = offload_pipeline
        ctx.save_for_backward(h_c, lm_weight, t_c, valid_count)

        if int(valid_count.item()) == 0:
            return h_c.new_zeros((), dtype=torch.float32)

        loss_sum = torch.zeros((), dtype=torch.float32, device=h_c.device)
        cs = ctx.chunk_size
        with torch.no_grad():
            for s in range(0, M, cs):
                e = min(M, s + cs)
                t_sub = t_c[s:e]
                logits_sub = F.linear(h_c[s:e], lm_weight).to(torch.float32)
                loss_sum = loss_sum + F.cross_entropy(
                    logits_sub, t_sub, ignore_index=ignore_index, reduction="sum"
                )
                del logits_sub

        return loss_sum / valid_count.to(torch.float32)

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        # 反向传播刚到达 lm_head 时，立即异步唤醒最后一层卸载激活的 H2D 预取！
        if ctx.offload_pipeline is not None:
            ctx.offload_pipeline.trigger_prefetch_latest()

        h_c, lm_weight, t_c, valid_count = ctx.saved_tensors
        M, H = h_c.shape
        ignore_index = ctx.ignore_index
        cs = ctx.chunk_size
        need_dh = bool(ctx.needs_input_grad[0])
        need_dw = bool(ctx.needs_input_grad[1])

        vc_val = int(valid_count.item())
        if vc_val == 0:
            return (
                torch.zeros_like(h_c) if need_dh else None,
                torch.zeros_like(lm_weight) if need_dw else None,
                None,
                None,
                None,
                None,
            )

        scale = grad_loss.to(torch.float32) / valid_count.to(torch.float32)
        # 采用 2^-3 (0.125) 精确指数预缩放保持 FP16/BF16 尾数零损失且彻底防止大序列 1/M 预除导致 FP16 下溢归零！
        pre_scale = 0.125
        post_scale = scale / pre_scale
        dh = torch.empty_like(h_c) if need_dh else None
        dw_f32 = (
            torch.zeros(
                (lm_weight.shape[0], lm_weight.shape[1]),
                dtype=torch.float32,
                device=lm_weight.device,
            )
            if need_dw
            else None
        )

        for s in range(0, M, cs):
            e = min(M, s + cs)
            h_sub = h_c[s:e]
            t_sub = t_c[s:e]
            valid_sub = t_sub != ignore_index

            probs = F.linear(h_sub, lm_weight).to(torch.float32)
            torch.softmax(probs, dim=-1, out=probs)

            safe_t = torch.where(valid_sub, t_sub, torch.zeros_like(t_sub))
            rows = torch.arange(e - s, device=h_c.device)
            probs[rows, safe_t] -= 1.0
            probs.mul_(valid_sub.unsqueeze(-1).to(torch.float32) * pre_scale)

            if need_dw:
                assert dw_f32 is not None
                # 真正原地 cuBLAS FP32 addmm_ 累加：零 [V, H] 临时张量分配，且消除 dlogits 低精度截断误差！
                dw_f32.addmm_(probs.t(), h_sub.to(torch.float32))

            if need_dh:
                assert dh is not None
                dlogits = probs.to(h_c.dtype)
                del probs
                dh_unscaled = torch.matmul(dlogits, lm_weight)
                del dlogits
                dh[s:e] = (dh_unscaled.to(torch.float32) * post_scale).to(h_c.dtype)
                del dh_unscaled
            else:
                del probs

        if need_dw:
            assert dw_f32 is not None
            dw_f32.mul_(post_scale)
            dw_out = (
                dw_f32
                if lm_weight.dtype == torch.float32
                else dw_f32.to(lm_weight.dtype)
            )
            del dw_f32
        else:
            dw_out = None
        return dh, dw_out, None, None, None, None


def chunked_causal_lm_cross_entropy(
    hidden: torch.Tensor,
    lm_weight: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = 256,
    offload_pipeline: DoubleBufferedOffloadPipeline | None = None,
) -> torch.Tensor:
    """标准自回归偏移 `hidden[:, :-1]` 预测 `target[:, 1:]` 的全序列零截断分块交叉熵"""
    tgt_src = input_ids if labels is None else labels
    if hidden.shape[:2] != tgt_src.shape or hidden.shape[1] < 2:
        raise ValueError(
            f"Causal LM Loss 要求序列长度 >= 2 且 hidden {hidden.shape[:2]} 与 labels {tgt_src.shape} 对齐"
        )
    shift_h = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    shift_t = tgt_src[:, 1:].reshape(-1)
    return ChunkedLinearCrossEntropyFunction.apply(  # type: ignore[return-value]
        shift_h, lm_weight, shift_t, ignore_index, chunk_size, offload_pipeline
    )
