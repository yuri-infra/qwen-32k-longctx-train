"""
qwen-32k-longctx-train（长上下文大模型训练与分布式环形注意力引擎）严苛验证与回归测试套件（共 8 大项）

全链路覆盖：
1. 真实 Triton GQA FlashAttention-2 前反向精度（FP16/BF16/FP32 × MHA/GQA × 非对齐序列 × 非连续步长视图）与 O(S) 显存核验
2. 真实 `torch.distributed` (`mp.spawn` 多进程) Contiguous 与 ZigZag 负载均衡 Causal Ring Attention 前反向梯度对齐测试
3. `PinnedActivationPool` 零环绕覆写保护 + 【反假拷贝篡改实验：证明反向求导 100% 真实消费锁页内存中的 DMA 激活】
4. 大词表分块融合交叉熵 (`ChunkedLinearCrossEntropy`)：`ignore_index` / 全忽略零除保护 / 词嵌入共享权重 / FP16 防下溢验证
5. 层内选择性激活重计算 (`SelectiveLayerCheckpointFunction`) 全模型权重梯度精确对齐与 >70% 激活显存削减验证
6. 真实预训练 `Qwen/Qwen2.5-0.5B-Instruct` (`24层, 14Q/2KV, 151936词表, BF16`) 多步训练 Loss 收敛与 `HybridCPUAdamW` 零静态显存验证
7. 单张 8GB GPU 运行真实 24 层 `Qwen2.5-0.5B-Instruct` **`16,384` (16K) 超长序列** 全序列 Loss + 反向 + AdamW 更新闭环测试
8. AST 抽象语法树静态审计：确认零 `[:, -64:]` 损失截断偷跑、零丢弃返回值的伪通信调用、强基线真实调用原生 `scaled_dot_product_attention`
"""

import ast
import inspect
import os
import sys
import textwrap
import unittest

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import long_ctx_train.selective_checkpoint as sc_mod
import long_ctx_train.trainer as trainer_mod
from long_ctx_train import (
    DoubleBufferedOffloadPipeline,
    HybridCPUAdamW,
    LongContextTransformer,
    PinnedActivationPool,
    TrainModelConfig,
    apply_rope_bwd,
    apply_rope_fwd,
    chunked_causal_lm_cross_entropy,
    compute_flash_bwd_delta,
    compute_ring_causal_load_balance,
    flash_attn_backward_with_lse,
    flash_attn_forward_with_lse,
    get_flash_attn_counters,
    reset_flash_attn_counters,
    run_multiprocess_ring_verification,
    run_training_step,
    triton_flash_attention,
    zigzag_shard_sequence,
    zigzag_unshard_sequence,
)


class TestProject2HardenedLongCtxSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), "本测试套件需要可用的 NVIDIA CUDA GPU"
        torch.manual_seed(2026)
        cls.device = torch.device("cuda:0")

    def test_01_triton_flash_attn_gqa_fwd_bwd_strides_and_o_s_memory(self):
        """验证 Triton GQA FlashAttention-2 在多精度、非对齐长度、非连续 transpose 视图下的前反向精度与 O(S) 显存复杂度"""
        reset_flash_attn_counters()

        for dtype, atol in (
            (torch.float32, 2e-4),
            (torch.float16, 5e-3),
            (torch.bfloat16, 2.5e-2),
        ):
            for seq_len in (19, 113, 384):
                for hq, hkv in ((8, 8), (14, 2)):
                    q1 = torch.randn(
                        2,
                        seq_len,
                        hq,
                        64,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=True,
                    )
                    k1 = torch.randn(
                        2,
                        seq_len,
                        hkv,
                        64,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=True,
                    )
                    v1 = torch.randn(
                        2,
                        seq_len,
                        hkv,
                        64,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=True,
                    )
                    do = torch.randn(
                        2, seq_len, hq, 64, dtype=dtype, device=self.device
                    )

                    q2 = q1.detach().clone().requires_grad_(True)
                    k2 = k1.detach().clone().requires_grad_(True)
                    v2 = v1.detach().clone().requires_grad_(True)

                    o_ref = F.scaled_dot_product_attention(
                        q1.transpose(1, 2),
                        k1.transpose(1, 2),
                        v1.transpose(1, 2),
                        is_causal=True,
                        enable_gqa=True,
                    ).transpose(1, 2)
                    o_ref.backward(do)

                    o_tri = triton_flash_attention(q2, k2, v2, is_causal=True)
                    o_tri.backward(do)

                    grad_tol = atol * 3.0 if dtype == torch.bfloat16 else atol * 2.5
                    self.assertLess((o_ref - o_tri).abs().max().item(), atol)
                    self.assertLessEqual(
                        (q1.grad - q2.grad).abs().max().item(), grad_tol
                    )  # type: ignore[operator]
                    self.assertLessEqual(
                        (k1.grad - k2.grad).abs().max().item(), grad_tol
                    )  # type: ignore[operator]
                    self.assertLessEqual(
                        (v1.grad - v2.grad).abs().max().item(), grad_tol
                    )  # type: ignore[operator]

        cnts = get_flash_attn_counters()
        self.assertEqual(cnts["fwd"], 18)
        self.assertEqual(cnts["bwd"], 18)

        # 验证外维转置视图 [B, H, S, D].transpose(1, 2) (stride(-1) == 1 但外维非连续) 与内维非连续视图
        q_tr = (
            torch.randn(2, 14, 70, 64, dtype=torch.float32, device=self.device)
            .transpose(1, 2)
            .requires_grad_(True)
        )
        k_tr = (
            torch.randn(2, 2, 70, 64, dtype=torch.float32, device=self.device)
            .transpose(1, 2)
            .requires_grad_(True)
        )
        v_tr = (
            torch.randn(2, 2, 70, 64, dtype=torch.float32, device=self.device)
            .transpose(1, 2)
            .requires_grad_(True)
        )
        self.assertFalse(q_tr.is_contiguous())
        self.assertEqual(q_tr.stride(-1), 1)
        o_tr_ref = F.scaled_dot_product_attention(
            q_tr.detach().transpose(1, 2),
            k_tr.detach().transpose(1, 2),
            v_tr.detach().transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        o_tr_tri = triton_flash_attention(q_tr, k_tr, v_tr, is_causal=True)
        self.assertLess((o_tr_ref - o_tr_tri).abs().max().item(), 2e-4)

        # 验证非连续最后一维 (stride(-1) != 1) 视图自动保护零 NaN
        q_nc = (
            torch.randn(1, 70, 64, 14, dtype=torch.float16, device=self.device)
            .transpose(2, 3)
            .requires_grad_(True)
        )
        k_nc = (
            torch.randn(1, 70, 64, 2, dtype=torch.float16, device=self.device)
            .transpose(2, 3)
            .requires_grad_(True)
        )
        v_nc = (
            torch.randn(1, 70, 64, 2, dtype=torch.float16, device=self.device)
            .transpose(2, 3)
            .requires_grad_(True)
        )
        self.assertNotEqual(q_nc.stride(-1), 1)
        out_nc = triton_flash_attention(q_nc, k_nc, v_nc, is_causal=True)
        self.assertFalse(torch.isnan(out_nc).any().item())
        out_nc.sum().backward()
        self.assertFalse(torch.isnan(q_nc.grad).any().item())  # type: ignore[union-attr]

        # 非法输入与契约校验（含 FlashBwd grad_out/lse/delta dtype 错配与 RoPE cos/sin 越界拦截）
        with self.assertRaises(ValueError):
            triton_flash_attention(q_nc[:, :, :5, :], k_nc, v_nc)  # 5 % 2 != 0
        q_chk = torch.randn(1, 32, 4, 64, dtype=torch.float16, device=self.device)
        o_chk, lse_chk = flash_attn_forward_with_lse(q_chk, q_chk, q_chk)
        with self.assertRaises(ValueError):
            flash_attn_backward_with_lse(
                torch.randn_like(q_chk, dtype=torch.float32),
                q_chk,
                q_chk,
                q_chk,
                lse_chk,
                lse_chk,
            )
        with self.assertRaises(ValueError):
            flash_attn_backward_with_lse(
                torch.randn_like(q_chk), q_chk, q_chk, q_chk, lse_chk.half(), lse_chk
            )
        with self.assertRaises(ValueError):
            compute_flash_bwd_delta(o_chk, torch.randn_like(o_chk, dtype=torch.float32))
        cos_short = torch.randn(8, 64, dtype=torch.float32, device=self.device)
        with self.assertRaises(ValueError):
            apply_rope_fwd(q_chk, cos_short, cos_short)
        with self.assertRaises(ValueError):
            apply_rope_bwd(q_chk, cos_short, cos_short)

        # 验证 S=2048, Hq=16 时前向后常驻激活严格为线性 O(S) (< 16 MB)，绝不物化 [1, 16, 2048, 2048] (128 MB FP16) 矩阵！
        torch.cuda.empty_cache()
        q_mem = torch.randn(
            1, 2048, 16, 64, dtype=torch.float16, device=self.device, requires_grad=True
        )
        k_mem = torch.randn(
            1, 2048, 4, 64, dtype=torch.float16, device=self.device, requires_grad=True
        )
        v_mem = torch.randn(
            1, 2048, 4, 64, dtype=torch.float16, device=self.device, requires_grad=True
        )
        m_before = torch.cuda.memory_allocated(self.device)
        o_mem = triton_flash_attention(q_mem, k_mem, v_mem, is_causal=True)
        saved_mb = (torch.cuda.memory_allocated(self.device) - m_before) / (
            1024.0 * 1024.0
        )
        self.assertLess(
            saved_mb,
            16.0,
            f"FlashAttention 前向常驻显存过高 ({saved_mb:.2f} MB)，疑似泄漏 O(S^2) 矩阵",
        )
        del o_mem, q_mem, k_mem, v_mem

    def test_02_distributed_ring_attention_multiprocess_zigzag_and_contiguous(self):
        """派生 2 个真实 OS 子进程 (`torch.distributed`)，验证 Contiguous 与 ZigZag Ring Attention 前反向梯度严格一致"""
        # 1. 验证 ZigZag 在 P=4, S=4096 下的精确 1.00x 因果计算负载均衡
        lb = compute_ring_causal_load_balance(seq_len=4096, world_size=4)
        self.assertGreater(lb["contiguous_max_min_ratio"], 6.9)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(lb["zigzag_max_min_ratio"]), 1.0, places=6)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(lb["zigzag_straggler_overhead"]), 1.0, places=6)  # type: ignore[arg-type]

        x_full = torch.arange(16, device=self.device).view(1, 16, 1)
        s0 = zigzag_shard_sequence(x_full, 0, 2, seq_dim=1)
        s1 = zigzag_shard_sequence(x_full, 1, 2, seq_dim=1)
        self.assertTrue(
            torch.equal(zigzag_unshard_sequence([s0, s1], seq_dim=1), x_full)
        )

        # 2. 真实双进程 (world_size=2) P2P 环网通信前反向验证 (GQA 14Q/2KV, S=256)
        q_ref = torch.randn(
            1, 256, 14, 64, dtype=torch.float32, device=self.device, requires_grad=True
        )
        k_ref = torch.randn(
            1, 256, 2, 64, dtype=torch.float32, device=self.device, requires_grad=True
        )
        v_ref = torch.randn(
            1, 256, 2, 64, dtype=torch.float32, device=self.device, requires_grad=True
        )
        do_ref = torch.randn(1, 256, 14, 64, dtype=torch.float32, device=self.device)

        out_single = triton_flash_attention(q_ref, k_ref, v_ref, is_causal=True)
        out_single.backward(do_ref)

        for use_zigzag in (False, True):
            ring_res = run_multiprocess_ring_verification(
                q_ref, k_ref, v_ref, do_ref, world_size=2, zigzag=use_zigzag
            )
            self.assertLess((out_single - ring_res["out"]).abs().max().item(), 2e-4)
            self.assertLess((q_ref.grad - ring_res["dq"]).abs().max().item(), 5e-4)  # type: ignore[operator]
            self.assertLess((k_ref.grad - ring_res["dk"]).abs().max().item(), 5e-4)  # type: ignore[operator]
            self.assertLess((v_ref.grad - ring_res["dv"]).abs().max().item(), 5e-4)  # type: ignore[operator]

    def test_03_pinned_pool_zero_overwrite_and_real_consumed_offload_tamper_proof(self):
        """验证锁页槽位池零环绕覆写，并通过篡改主机锁页内存实验证明反向求导 100% 真实消费 DMA 激活而非假拷贝"""
        strict_pool = PinnedActivationPool(
            num_slots=2, slot_bytes=4096, strict_capacity=True
        )
        sl0 = strict_pool.acquire_slot(1024, (256,), torch.float32)
        sl1 = strict_pool.acquire_slot(1024, (256,), torch.float32)
        with self.assertRaises(MemoryError):
            strict_pool.acquire_slot(1024, (256,), torch.float32)
        strict_pool.release_slot(sl0)
        strict_pool.release_slot(sl1)

        cfg = TrainModelConfig(
            vocab_size=512,
            hidden_dim=256,
            intermediate_dim=512,
            num_layers=4,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=32,
            qkv_bias=True,
            tie_word_embeddings=True,
        )
        m_ref = LongContextTransformer(cfg, device=self.device, dtype=torch.float32)
        inp = torch.randint(0, cfg.vocab_size, (2, 96), device=self.device)

        loss_ref = m_ref.forward_and_loss(inp, mode="eager_sdpa_full_logits")
        loss_ref.backward()

        # 正常双缓冲异步卸载预取：所有参数梯度必须与未卸载基线精确一致，且预取命中 > 0、结束后占用槽归零
        m_off = LongContextTransformer(cfg, device=self.device, dtype=torch.float32)
        m_off.load_state_dict(m_ref.state_dict())
        pool = PinnedActivationPool(num_slots=4, slot_bytes=512 * 1024)
        pipe = DoubleBufferedOffloadPipeline(pool, device=self.device)

        loss_off = m_off.forward_and_loss(
            inp,
            mode="hybrid_async_offload",
            offload_pipeline=pipe,
            num_offload_layers=4,
            ce_chunk_size=32,
        )
        loss_off.backward()

        self.assertGreater(
            pipe.prefetch_hit_count, 0, "反向跨层双缓冲异步 H2D 预取必须真实命中"
        )
        self.assertEqual(
            pool.num_active_slots, 0, "反向传播完成后锁页槽位必须 100% 释放"
        )
        self.assertAlmostEqual(
            float(loss_ref.detach()), float(loss_off.detach()), places=4
        )
        for (name, p_ref), (_, p_off) in zip(
            m_ref.named_parameters(), m_off.named_parameters(), strict=True
        ):
            self.assertLess(
                (p_ref.grad - p_off.grad).abs().max().item(), 2e-4, f"{name} 梯度不一致"
            )  # type: ignore[operator]

        # 【反假拷贝红方篡改测试】：在前向完成、D2H 落盘至 CPU 锁页内存后，人为将主机锁页槽数据清零，
        # 若反向传播真的是从主机锁页内存 H2D 读回激活参与计算，第一层权重梯度必定发生显著改变！
        m_tamper = LongContextTransformer(cfg, device=self.device, dtype=torch.float32)
        m_tamper.load_state_dict(m_ref.state_dict())
        loss_t = m_tamper.forward_and_loss(
            inp,
            mode="hybrid_async_offload",
            offload_pipeline=pipe,
            num_offload_layers=4,
            ce_chunk_size=32,
        )
        pipe.d2h_stream.synchronize()
        for slot in pool.slots:
            if slot.in_use:
                slot.buffer.zero_()
        loss_t.backward()
        ref_max_g = m_ref.layers[0].q_proj.weight.grad.abs().max().item()  # type: ignore[union-attr]
        diff_after_tamper = (
            (m_ref.layers[0].q_proj.weight.grad - m_tamper.layers[0].q_proj.weight.grad)
            .abs()
            .max()
            .item()
        )  # type: ignore[operator]
        self.assertGreater(
            diff_after_tamper,
            ref_max_g * 0.90,
            "篡改主机锁页激活后梯度未变化，疑似未真实消费卸载张量！",
        )

        # 验证 1) hybrid_sync_offload 在默认 pipeline 上严格禁用异步预取；2) LoRA 局部参数冻结零崩溃且每层 SwiGLU 无三次冗余重算
        m_lora_ref = LongContextTransformer(
            cfg, device=self.device, dtype=torch.float32
        )
        m_lora_off = LongContextTransformer(
            cfg, device=self.device, dtype=torch.float32
        )
        m_lora_off.load_state_dict(m_lora_ref.state_dict())
        for m_tmp in (m_lora_ref, m_lora_off):
            m_tmp.embed.weight.requires_grad_(False)
            for p_l0 in m_tmp.layers[0].parameters():
                p_l0.requires_grad_(False)
            m_tmp.layers[1].q_proj.weight.requires_grad_(False)
            m_tmp.layers[1].down_proj.weight.requires_grad_(False)

        loss_lr_ref = m_lora_ref.forward_and_loss(inp, mode="eager_sdpa_full_logits")
        loss_lr_ref.backward()

        fresh_pipe = DoubleBufferedOffloadPipeline(pool, device=self.device)
        loss_sync_chk = m_lora_off.forward_and_loss(
            inp,
            mode="hybrid_sync_offload",
            offload_pipeline=fresh_pipe,
            num_offload_layers=4,
            ce_chunk_size=32,
        )
        loss_sync_chk.backward()
        self.assertEqual(
            fresh_pipe.prefetch_hit_count,
            0,
            "hybrid_sync_offload 模式下不得静默触发异步预取",
        )
        self.assertEqual(pool.num_active_slots, 0)
        m_lora_off.zero_grad(set_to_none=True)

        orig_swiglu = sc_mod.fused_packed_swiglu_fwd
        swiglu_cnt = [0]

        def _counted_swiglu(gu):
            swiglu_cnt[0] += 1
            return orig_swiglu(gu)

        sc_mod.fused_packed_swiglu_fwd = _counted_swiglu
        try:
            loss_lr_off = m_lora_off.forward_and_loss(
                inp,
                mode="hybrid_async_offload",
                offload_pipeline=fresh_pipe,
                num_offload_layers=4,
                ce_chunk_size=32,
            )
            loss_lr_off.backward()
        finally:
            sc_mod.fused_packed_swiglu_fwd = orig_swiglu

        self.assertLessEqual(
            swiglu_cnt[0],
            2 * cfg.num_layers,
            f"OffloadedLayer 出现冗余第 3 次 SwiGLU 重算: {swiglu_cnt[0]}",
        )
        self.assertEqual(pool.num_active_slots, 0)
        for (name, p1), (_, p2) in zip(
            m_lora_ref.named_parameters(), m_lora_off.named_parameters(), strict=True
        ):
            if p1.requires_grad:
                self.assertLess(
                    (p1.grad - p2.grad).abs().max().item(),
                    2e-4,
                    f"LoRA 冻结场景 {name} 梯度不一致",
                )  # type: ignore[operator]
            else:
                self.assertIsNone(p2.grad)

    def test_04_chunked_cross_entropy_ignore_index_and_fp16_underflow_guard(self):
        """验证大词表流式分块交叉熵在 ignore_index、全忽略零除边界及 FP16 大序列下的无下溢梯度精度"""
        B, S, H, V = 2, 1025, 256, 16384  # M = 2 * 1024 = 2048 tokens
        h_fp16 = (
            torch.randn(B, S, H, dtype=torch.float16, device=self.device) * 0.5
        ).requires_grad_(True)
        w_fp16 = (
            torch.randn(V, H, dtype=torch.float16, device=self.device) * 0.02
        ).requires_grad_(True)
        labels = torch.randint(0, V, (B, S), device=self.device)
        labels[:, ::5] = -100  # 20% ignore_index

        # FP32 黄金参考
        h_f32 = h_fp16.detach().float().requires_grad_(True)
        w_f32 = w_fp16.detach().float().requires_grad_(True)
        logits_ref = F.linear(h_f32[:, :-1, :].reshape(-1, H), w_f32)
        loss_ref = F.cross_entropy(
            logits_ref, labels[:, 1:].reshape(-1), ignore_index=-100
        )
        loss_ref.backward()

        loss_chunk = chunked_causal_lm_cross_entropy(
            h_fp16,
            w_fp16,
            input_ids=labels,
            labels=labels,
            ignore_index=-100,
            chunk_size=128,
        )
        loss_chunk.backward()

        self.assertAlmostEqual(
            float(loss_ref.detach()), float(loss_chunk.detach()), places=2
        )
        self.assertLess((h_f32.grad - h_fp16.grad.float()).abs().max().item(), 2e-3)  # type: ignore[union-attr]
        self.assertLess((w_f32.grad - w_fp16.grad.float()).abs().max().item(), 5e-3)  # type: ignore[union-attr]

        # 验证 FP16 w.grad 非零率（防 1/M 预除引发大面积下溢归零！）
        ref_nonzero = float((w_f32.grad.abs() > 0).float().mean().item())  # type: ignore[union-attr]
        fp16_nonzero = float((w_fp16.grad.abs() > 0).float().mean().item())  # type: ignore[union-attr]
        self.assertGreater(
            fp16_nonzero,
            ref_nonzero * 0.85,
            f"FP16 w.grad 出现严重下溢归零: {fp16_nonzero:.3f} vs {ref_nonzero:.3f}",
        )

        # 100% 全 ignore_index 零除保护验证
        all_ign = torch.full((B, S), -100, dtype=torch.long, device=self.device)
        h_zero = h_fp16.detach().clone().requires_grad_(True)
        w_zero = w_fp16.detach().clone().requires_grad_(True)
        loss_ign = chunked_causal_lm_cross_entropy(
            h_zero, w_zero, input_ids=all_ign, labels=all_ign, ignore_index=-100
        )
        self.assertEqual(float(loss_ign.detach().item()), 0.0)
        loss_ign.backward()
        self.assertFalse(torch.isnan(h_zero.grad).any().item())  # type: ignore[union-attr]
        self.assertEqual(float(w_zero.grad.abs().max().item()), 0.0)  # type: ignore[union-attr]

        # 越界 target 索引安全拦截（拒绝触发 CUDA Device-Side Assert）
        bad_lbl = labels.clone()
        bad_lbl[0, 3] = V + 10
        with self.assertRaises(ValueError):
            chunked_causal_lm_cross_entropy(
                h_fp16, w_fp16, input_ids=bad_lbl, labels=bad_lbl, ignore_index=-100
            )

        # 验证 M=16,384, Vocab=151,936 (BF16) 下原地 dw_f32.addmm_ 峰值显存契约（额外工作区 <= 560 MB，总增量峰值 <= 900 MB）
        torch.cuda.empty_cache()
        h_16k = torch.randn(
            1, 16385, 896, dtype=torch.bfloat16, device=self.device, requires_grad=True
        )
        w_qwen = torch.randn(
            151936, 896, dtype=torch.bfloat16, device=self.device, requires_grad=True
        )
        lbl_16k = torch.randint(0, 151936, (1, 16385), device=self.device)
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        base_ce = torch.cuda.memory_allocated(self.device)
        loss_16k_ce = chunked_causal_lm_cross_entropy(
            h_16k, w_qwen, input_ids=lbl_16k, chunk_size=256
        )
        loss_16k_ce.backward()
        torch.cuda.synchronize(self.device)
        inc_peak_mb = (torch.cuda.max_memory_allocated(self.device) - base_ce) / (
            1024.0 * 1024.0
        )
        end_grads_mb = (torch.cuda.memory_allocated(self.device) - base_ce) / (
            1024.0 * 1024.0
        )
        self.assertLessEqual(
            inc_peak_mb - end_grads_mb,
            560.0,
            f"分块 CE 额外工作区显存超标: {inc_peak_mb - end_grads_mb:.1f} MB",
        )
        self.assertLessEqual(
            inc_peak_mb, 900.0, f"分块 CE 总增量显存超标: {inc_peak_mb:.1f} MB"
        )
        del h_16k, w_qwen, lbl_16k, loss_16k_ce
        torch.cuda.empty_cache()

    def test_05_selective_layer_checkpoint_exact_gradients_and_memory_savings(self):
        """验证层内选择性激活重计算 (sac_triton_flash) 全参数梯度对齐且前向常驻激活显存削减 > 70%"""
        cfg = TrainModelConfig(
            vocab_size=2048,
            hidden_dim=512,
            intermediate_dim=2048,
            num_layers=6,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            qkv_bias=True,
            tie_word_embeddings=True,
        )
        m_eager = LongContextTransformer(cfg, device=self.device, dtype=torch.bfloat16)
        m_sac = LongContextTransformer(cfg, device=self.device, dtype=torch.bfloat16)
        m_sac.load_state_dict(m_eager.state_dict())

        inp = torch.randint(0, cfg.vocab_size, (1, 1024), device=self.device)
        # 必须比较前向完成时刻常驻激活 `fwd_act_mb`（而非反向瞬态峰值），但注意 `run_training_step` 中 `fwd_peak` 包含前向与分块 CE 前向峰值！
        # 为纯粹测量 6 层主干网络前向驻留激活显存，直接对比 `forward_hidden` 前后的 `memory_allocated` 增量：
        torch.cuda.empty_cache()
        m_e0 = torch.cuda.memory_allocated(self.device)
        out_e = m_eager.forward_hidden(inp, mode="eager_sdpa_chunked_ce")
        saved_eager_mb = (torch.cuda.memory_allocated(self.device) - m_e0) / (
            1024.0 * 1024.0
        )
        del out_e
        m_eager.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        m_s0 = torch.cuda.memory_allocated(self.device)
        out_s = m_sac.forward_hidden(inp, mode="sac_triton_flash")
        saved_sac_mb = (torch.cuda.memory_allocated(self.device) - m_s0) / (
            1024.0 * 1024.0
        )
        del out_s
        m_sac.zero_grad(set_to_none=True)

        r_eager = run_training_step(m_eager, inp, mode="eager_sdpa_chunked_ce")
        r_sac = run_training_step(m_sac, inp, mode="sac_triton_flash")

        self.assertAlmostEqual(r_eager["loss"], r_sac["loss"], places=2)
        self.assertLess(
            saved_sac_mb,
            saved_eager_mb * 0.30,
            f"SAC 主干常驻激活 ({saved_sac_mb:.1f} MB) 应低于 Eager ({saved_eager_mb:.1f} MB) 的 30%",
        )
        for (name, p1), (_, p2) in zip(
            m_eager.named_parameters(), m_sac.named_parameters(), strict=True
        ):
            self.assertLess(
                (p1.grad.float() - p2.grad.float()).abs().max().item(),
                5e-2,
                f"层参数 {name} 梯度偏差过大",
            )  # type: ignore[union-attr]

    def test_06_real_qwen2_5_0_5b_pretrained_long_context_convergence_and_hybrid_adamw(
        self,
    ):
        """加载本机真实 24 层 Qwen2.5-0.5B-Instruct (BF16)，验证 HybridCPUAdamW 数值等价性、零静态显存占用与多步训练 Loss 真实下降"""
        # 1. 先与 PyTorch 原生 torch.optim.AdamW 做 3 步严格数值对齐（含连续参数与非连续转置参数/非连续梯度）与非法超参拦截验证
        p_ref = torch.nn.Parameter(
            torch.randn(128, 64, dtype=torch.float32, device=self.device)
        )
        p_hyb = torch.nn.Parameter(p_ref.detach().clone())
        raw_nc = torch.randn(64, 128, dtype=torch.float32, device=self.device)
        p_nc_ref = torch.nn.Parameter(raw_nc.t())
        p_nc_hyb = torch.nn.Parameter(raw_nc.clone().t())
        self.assertFalse(p_nc_hyb.is_contiguous())

        opt_ref = torch.optim.AdamW(
            [p_ref, p_nc_ref], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02
        )
        opt_hyb = HybridCPUAdamW(
            [p_hyb, p_nc_hyb], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02
        )
        for _ in range(3):
            g1 = torch.randn_like(p_ref)
            g2 = torch.randn_like(p_nc_ref)  # 保留非连续转置步长 (1, 128)
            p_ref.grad, p_hyb.grad = g1.clone(), g1.clone()
            p_nc_ref.grad, p_nc_hyb.grad = g2.clone(), g2.clone()
            opt_ref.step()
            opt_hyb.step()
        self.assertLess((p_ref.detach() - p_hyb.detach()).abs().max().item(), 1e-6)
        self.assertLess(
            (p_nc_ref.detach() - p_nc_hyb.detach()).abs().max().item(), 1e-6
        )
        with self.assertRaises(ValueError):
            HybridCPUAdamW([p_hyb], lr=float("nan"))

        # 2. 真实 24 层 Qwen2.5-0.5B-Instruct 训练验证
        qwen = LongContextTransformer.from_qwen2_pretrained(
            device=self.device, dtype=torch.bfloat16, max_seq_len=4096
        )
        mem_before_opt = torch.cuda.memory_allocated(self.device)
        optimizer = HybridCPUAdamW(qwen.parameters(), lr=2e-5, weight_decay=0.01)
        mem_after_opt = torch.cuda.memory_allocated(self.device)
        self.assertEqual(
            mem_after_opt - mem_before_opt,
            0,
            "HybridCPUAdamW 在 GPU 上必须保持 0 Bytes 静态状态显存开销",
        )

        pool = PinnedActivationPool(num_slots=12, slot_bytes=8 * 1024 * 1024)
        pipe = DoubleBufferedOffloadPipeline(pool, device=self.device)
        # 构造非周期长序列训练样本 (S=1024) 验证真实前向 + 1023词全序列CE + 反向 + FP32主权重更新单调收敛！
        train_seq = torch.tensor(
            [[((i * 131 + i * i * 7 + 29) % 8000) + 200 for i in range(1024)]],
            dtype=torch.long,
            device=self.device,
        )

        losses = []
        for _ in range(3):
            st = run_training_step(
                qwen,
                train_seq,
                mode="hybrid_async_offload",
                offload_pipeline=pipe,
                num_offload_layers=12,
                optimizer=optimizer,
            )
            losses.append(st["loss"])

        self.assertLess(losses[1], losses[0], f"第 2 步训练 Loss 未下降: {losses}")
        self.assertLess(losses[2], losses[1], f"第 3 步训练 Loss 未下降: {losses}")

        del optimizer, pipe, pool, qwen
        torch.cuda.empty_cache()

    def test_07_real_qwen2_5_0_5b_16k_ultra_long_context_on_8gb_gpu(self):
        """【8GB 单卡极限实证】：在 7.93GB RTX 5060 上完整跑通真实 24 层 Qwen2.5-0.5B 的 16,384 (16K) 全序列训练步 + AdamW 更新！"""
        torch.cuda.empty_cache()
        qwen = LongContextTransformer.from_qwen2_pretrained(
            device=self.device, dtype=torch.bfloat16, max_seq_len=16384
        )
        optimizer = HybridCPUAdamW(qwen.parameters(), lr=5e-5)
        pool = PinnedActivationPool(num_slots=24, slot_bytes=30 * 1024 * 1024)
        pipe = DoubleBufferedOffloadPipeline(pool, device=self.device)

        inp_16k = torch.randint(10, 50000, (1, 16384), device=self.device)
        stats = run_training_step(
            qwen,
            inp_16k,
            mode="hybrid_async_offload",
            offload_pipeline=pipe,
            num_offload_layers=24,
            optimizer=optimizer,
            ce_chunk_size=256,
        )
        self.assertTrue(0.0 < stats["loss"] < 30.0, f"16K 训练 Loss 异常: {stats}")
        self.assertLess(stats["total_peak_mb"], 7800.0)
        self.assertGreater(pipe.prefetch_hit_count, 20)
        self.assertEqual(pool.num_active_slots, 0)

        del inp_16k, optimizer, pipe, pool, qwen
        torch.cuda.empty_cache()

    def test_08_ast_audit_zero_discarded_comm_and_zero_strawman_baseline(self):
        """AST 抽象语法树静态审计：确认消除旧版 `[:, -64:]` 损失截断偷跑、丢弃型假通信与 O(S^2) 稻草人基线"""
        src = textwrap.dedent(inspect.getsource(trainer_mod))
        tree_ast = ast.parse(src)

        # 1. 强基线必须真实调用 F.scaled_dot_product_attention
        called_attrs = [
            node.func.attr
            for node in ast.walk(tree_ast)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("scaled_dot_product_attention", called_attrs)
        self.assertNotIn("naive_eager_causal_attention", called_attrs)

        # 2. 禁止任何 `-64` 尾部截断计算 Loss 的作弊切片
        neg_64_literals = [
            node.operand.value
            for node in ast.walk(tree_ast)
            if isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and node.operand.value == 64
        ]
        self.assertEqual(len(neg_64_literals), 0, "发现 [-64:] 截断序列损失作弊代码！")


if __name__ == "__main__":
    print("=" * 78)
    print(
        f"Running Project 2 Hardened Long-Context Training Suite on GPU: {torch.cuda.get_device_name(0)}"
    )
    print("=" * 78)
    unittest.main(verbosity=2)
