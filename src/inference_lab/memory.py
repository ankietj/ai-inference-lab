"""Analytical calculations for KV-cache and model weight memory consumption.

Formulas:
- Per-token KV cache elements:
    Elements_per_token = 2 * n_layers * n_kv_heads * d_head
  where the factor of 2 accounts for Key and Value matrices.

- Per-token KV cache memory (bytes):
    Bytes_per_token = Elements_per_token * bytes_per_element

- Continuous KV cache for sequence length L:
    KV_total(L) = L * Bytes_per_token

- Block-allocated (PagedAttention) KV cache for sequence length L with block size B:
    Allocated_blocks = ceil(L / B)
    Allocated_tokens = Allocated_blocks * B
    Allocated_bytes = Allocated_tokens * Bytes_per_token
    Wasted_tokens = Allocated_tokens - L
    Fragmentation_ratio = Wasted_tokens / Allocated_tokens
"""

from __future__ import annotations

import math
from typing import Any

from inference_lab.hardware import HardwareConfig
from inference_lab.models import ModelConfig, Precision


def kv_cache_elements_per_token(model: ModelConfig) -> int:
    """Calculate the total number of tensor elements stored in the KV cache per token.

    Each layer stores 1 Key vector and 1 Value vector of dimension (n_kv_heads * d_head).
    Total elements = 2 * n_layers * n_kv_heads * d_head
    """
    return 2 * model.n_layers * model.n_kv_heads * model.d_head


def kv_cache_bytes_per_token(model: ModelConfig, precision: Precision) -> float:
    """Calculate KV-cache byte footprint for a single token across all layers."""
    elements = kv_cache_elements_per_token(model)
    return elements * precision.bytes_per_element


def kv_cache_total_bytes(
    model: ModelConfig,
    seq_len: int,
    precision: Precision,
    batch_size: int = 1,
) -> float:
    """Calculate ideal continuous KV-cache memory for a batch of sequences of length seq_len."""
    if seq_len < 0:
        raise ValueError(f"Sequence length must be non-negative, got {seq_len}")
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")
    return kv_cache_bytes_per_token(model, precision) * seq_len * batch_size


def kv_cache_num_blocks(seq_len: int, block_size: int = 16) -> int:
    """Compute number of discrete PagedAttention blocks required to store a sequence."""
    if seq_len <= 0:
        return 0
    if block_size < 1:
        raise ValueError(f"Block size must be >= 1, got {block_size}")
    return math.ceil(seq_len / block_size)


def kv_cache_blocked_bytes(
    model: ModelConfig,
    seq_len: int,
    precision: Precision,
    block_size: int = 16,
) -> float:
    """Compute memory allocated for a sequence using discrete block allocation (PagedAttention)."""
    blocks = kv_cache_num_blocks(seq_len, block_size)
    allocated_tokens = blocks * block_size
    return allocated_tokens * kv_cache_bytes_per_token(model, precision)


def kv_cache_internal_fragmentation(
    seq_len: int,
    block_size: int = 16,
) -> dict[str, Any]:
    """Compute internal fragmentation metrics caused by discrete block paging."""
    if seq_len <= 0:
        return {
            "active_tokens": 0,
            "allocated_tokens": 0,
            "wasted_tokens": 0,
            "fragmentation_ratio": 0.0,
            "num_blocks": 0,
        }
    blocks = kv_cache_num_blocks(seq_len, block_size)
    allocated = blocks * block_size
    wasted = allocated - seq_len
    ratio = wasted / allocated if allocated > 0 else 0.0
    return {
        "active_tokens": seq_len,
        "allocated_tokens": allocated,
        "wasted_tokens": wasted,
        "fragmentation_ratio": ratio,
        "num_blocks": blocks,
    }


def model_weights_bytes(model: ModelConfig, precision: Precision) -> float:
    """Calculate memory required to store static model weights in VRAM."""
    return model.num_params * precision.bytes_per_element


def max_active_tokens_in_vram(
    model: ModelConfig,
    hardware: HardwareConfig,
    precision: Precision,
    reserved_headroom_gb: float = 2.0,
) -> int:
    """Calculate maximum number of concurrent KV-cache tokens that can fit in GPU VRAM.

    VRAM allocation:
        Total VRAM = Weights + KV_Cache + Runtime_Headroom (CUDA context, activations)
    """
    weights_mem = model_weights_bytes(model, precision)
    headroom_mem = reserved_headroom_gb * (1024**3)
    available_for_kv = hardware.vram_bytes - weights_mem - headroom_mem

    if available_for_kv <= 0:
        return 0

    bytes_per_token = kv_cache_bytes_per_token(model, precision)
    return int(available_for_kv // bytes_per_token)
