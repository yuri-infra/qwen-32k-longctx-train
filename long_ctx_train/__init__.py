"""
AI Infra Project 2: Long-Context LLM Training & Context-Parallel Engine
(Custom Triton GQA FlashAttention-2 Fwd/Bwd + Real Distributed ZigZag Ring Attention + Selective Layer Checkpointing + Slot-Safe Double-Buffered Async Offload & Prefetch + Chunked Giant-Vocab Cross-Entropy + Real Qwen2.5-0.5B 16K Training)
"""

from .flash_attn_triton import (
    FLASH_ATTN_COUNTERS,
    reset_flash_attn_counters,
    get_flash_attn_counters,
    flash_attn_forward_with_lse,
    compute_flash_bwd_delta,
    flash_attn_backward_with_lse,
    triton_flash_attention,
)
from .ring_attention import (
    compute_ring_causal_load_balance,
    zigzag_shard_sequence,
    zigzag_unshard_sequence,
    build_zigzag_global_positions,
    ring_flash_attention,
    run_multiprocess_ring_verification,
)
from .activation_offload import (
    ActivationSlot,
    PinnedActivationPool,
    DoubleBufferedOffloadPipeline,
)
from .selective_checkpoint import (
    apply_rope_fwd,
    apply_rope_bwd,
    theoretical_selective_ckpt_flops_overhead,
    SelectiveLayerCheckpointFunction,
    OffloadedLayerCheckpointFunction,
    ChunkedLinearCrossEntropyFunction,
    chunked_causal_lm_cross_entropy,
)
from .trainer import (
    TrainModelConfig,
    AdaptiveHybridMemoryPlanner,
    LongContextTransformer,
    HybridCPUAdamW,
    estimate_step_theoretical_flops,
    run_training_step,
)

__all__ = [
    "FLASH_ATTN_COUNTERS",
    "reset_flash_attn_counters",
    "get_flash_attn_counters",
    "flash_attn_forward_with_lse",
    "compute_flash_bwd_delta",
    "flash_attn_backward_with_lse",
    "triton_flash_attention",
    "compute_ring_causal_load_balance",
    "zigzag_shard_sequence",
    "zigzag_unshard_sequence",
    "build_zigzag_global_positions",
    "ring_flash_attention",
    "run_multiprocess_ring_verification",
    "ActivationSlot",
    "PinnedActivationPool",
    "DoubleBufferedOffloadPipeline",
    "apply_rope_fwd",
    "apply_rope_bwd",
    "theoretical_selective_ckpt_flops_overhead",
    "SelectiveLayerCheckpointFunction",
    "OffloadedLayerCheckpointFunction",
    "ChunkedLinearCrossEntropyFunction",
    "chunked_causal_lm_cross_entropy",
    "TrainModelConfig",
    "AdaptiveHybridMemoryPlanner",
    "LongContextTransformer",
    "HybridCPUAdamW",
    "estimate_step_theoretical_flops",
    "run_training_step",
]
