"""Unit tests for KV-cache memory calculations.

Verifies:
- MHA versus GQA analytical scaling
- Cache precision scaling (FP32, FP16, BF16, FP8, INT4)
- Context length linear scaling
- Discrete block allocation (PagedAttention ceiling and fragmentation)
- Model weight memory footprint
"""

import pytest
from inference_lab.hardware import A100_SXM4_80GB, T4_16GB
from inference_lab.memory import (
    kv_cache_blocked_bytes,
    kv_cache_bytes_per_token,
    kv_cache_elements_per_token,
    kv_cache_internal_fragmentation,
    kv_cache_num_blocks,
    kv_cache_total_bytes,
    max_active_tokens_in_vram,
    model_weights_bytes,
)
from inference_lab.models import LLAMA_1_7B, LLAMA_3_8B, LLAMA_3_70B, Precision


def test_kv_cache_elements_per_token():
    # LLaMA-1 7B: 2 * 32 layers * 32 kv_heads * 128 head_dim = 262,144 elements
    assert kv_cache_elements_per_token(LLAMA_1_7B) == 262_144

    # LLaMA-3 8B: 2 * 32 layers * 8 kv_heads * 128 head_dim = 65,536 elements
    assert kv_cache_elements_per_token(LLAMA_3_8B) == 65_536

    # LLaMA-3 70B: 2 * 80 layers * 8 kv_heads * 128 head_dim = 163,840 elements
    assert kv_cache_elements_per_token(LLAMA_3_70B) == 163_840


def test_mha_vs_gqa_memory_ratio():
    """Verify that GQA (8 kv_heads) saves exactly 4x memory compared to MHA (32 kv_heads)

    for equivalent layers (32) and head dimension (128).
    """
    mha_bytes = kv_cache_bytes_per_token(LLAMA_1_7B, Precision.FP16)
    gqa_bytes = kv_cache_bytes_per_token(LLAMA_3_8B, Precision.FP16)

    # LLaMA-1 7B: 262,144 * 2 = 524,288 bytes (512 KiB)
    assert mha_bytes == 524_288
    # LLaMA-3 8B: 65,536 * 2 = 131,072 bytes (128 KiB)
    assert gqa_bytes == 131_072

    ratio = mha_bytes / gqa_bytes
    assert ratio == 4.0


def test_cache_precision_scaling():
    """Verify exact numerical scaling across FP32, FP16/BF16, FP8, and INT4."""
    model = LLAMA_3_8B
    fp32_bytes = kv_cache_bytes_per_token(model, Precision.FP32)
    fp16_bytes = kv_cache_bytes_per_token(model, Precision.FP16)
    bf16_bytes = kv_cache_bytes_per_token(model, Precision.BF16)
    fp8_bytes = kv_cache_bytes_per_token(model, Precision.FP8)
    int4_bytes = kv_cache_bytes_per_token(model, Precision.INT4)

    assert fp16_bytes == bf16_bytes == 131_072
    assert fp32_bytes == fp16_bytes * 2.0 == 262_144
    assert fp8_bytes == fp16_bytes / 2.0 == 65_536
    assert int4_bytes == fp16_bytes / 4.0 == 32_768


def test_context_length_scaling():
    """Verify linear scaling with sequence length and batch size."""
    model = LLAMA_3_8B
    base = kv_cache_total_bytes(model, seq_len=512, precision=Precision.FP16, batch_size=1)
    double_seq = kv_cache_total_bytes(model, seq_len=1024, precision=Precision.FP16, batch_size=1)
    double_batch = kv_cache_total_bytes(model, seq_len=512, precision=Precision.FP16, batch_size=2)

    assert double_seq == base * 2
    assert double_batch == base * 2

    # Negative inputs must raise ValueError
    with pytest.raises(ValueError):
        kv_cache_total_bytes(model, seq_len=-1, precision=Precision.FP16)
    with pytest.raises(ValueError):
        kv_cache_total_bytes(model, seq_len=512, precision=Precision.FP16, batch_size=0)


def test_block_allocation_and_fragmentation():
    """Verify PagedAttention ceiling behavior and internal fragmentation calculations."""
    block_size = 16

    # Boundary: 0 tokens
    assert kv_cache_num_blocks(0, block_size) == 0
    frag_0 = kv_cache_internal_fragmentation(0, block_size)
    assert frag_0["num_blocks"] == 0
    assert frag_0["wasted_tokens"] == 0
    assert frag_0["fragmentation_ratio"] == 0.0

    # Boundary: 1 token requires 1 block (15 tokens wasted)
    assert kv_cache_num_blocks(1, block_size) == 1
    frag_1 = kv_cache_internal_fragmentation(1, block_size)
    assert frag_1["allocated_tokens"] == 16
    assert frag_1["wasted_tokens"] == 15
    assert frag_1["fragmentation_ratio"] == 15 / 16

    # Boundary: Exactly 16 tokens fits 1 block with zero waste
    assert kv_cache_num_blocks(16, block_size) == 1
    frag_16 = kv_cache_internal_fragmentation(16, block_size)
    assert frag_16["allocated_tokens"] == 16
    assert frag_16["wasted_tokens"] == 0
    assert frag_16["fragmentation_ratio"] == 0.0

    # Boundary: 17 tokens requires 2 blocks (15 tokens wasted)
    assert kv_cache_num_blocks(17, block_size) == 2
    frag_17 = kv_cache_internal_fragmentation(17, block_size)
    assert frag_17["allocated_tokens"] == 32
    assert frag_17["wasted_tokens"] == 15
    assert frag_17["fragmentation_ratio"] == 15 / 32

    # Long sequence: 2048 tokens = 128 blocks exactly
    assert kv_cache_num_blocks(2048, block_size) == 128
    frag_2048 = kv_cache_internal_fragmentation(2048, block_size)
    assert frag_2048["wasted_tokens"] == 0
    assert frag_2048["fragmentation_ratio"] == 0.0

    # Compare blocked bytes vs continuous bytes
    model = LLAMA_3_8B
    continuous_bytes = kv_cache_total_bytes(model, seq_len=17, precision=Precision.FP16)
    blocked_bytes = kv_cache_blocked_bytes(model, seq_len=17, precision=Precision.FP16, block_size=16)
    assert blocked_bytes > continuous_bytes
    assert blocked_bytes == 32 * kv_cache_bytes_per_token(model, Precision.FP16)


def test_model_weights_and_vram_capacity():
    model = LLAMA_3_8B
    # Weights in FP16: 8.03B params * 2 bytes = ~16.06 GB
    weights_fp16 = model_weights_bytes(model, Precision.FP16)
    assert weights_fp16 == 8_030_000_000 * 2.0

    # On A100 80GB: Weights ~16GB + 2GB headroom leaves ~62GB for KV-cache
    # 62 GB / 128 KiB per token = ~500,000 tokens
    tokens_a100 = max_active_tokens_in_vram(model, A100_SXM4_80GB, Precision.FP16, reserved_headroom_gb=2.0)
    assert tokens_a100 > 450_000

    # On T4 16GB: Weights (16.06 GB) do NOT fit into 16GB VRAM in FP16!
    # Available should be 0
    tokens_t4_fp16 = max_active_tokens_in_vram(model, T4_16GB, Precision.FP16, reserved_headroom_gb=2.0)
    assert tokens_t4_fp16 == 0

    # But in INT4 quantization (0.5 bytes): Weights = ~4.015 GB, fits in T4 16GB
    tokens_t4_int4 = max_active_tokens_in_vram(model, T4_16GB, Precision.INT4, reserved_headroom_gb=2.0)
    assert tokens_t4_int4 > 250_000
