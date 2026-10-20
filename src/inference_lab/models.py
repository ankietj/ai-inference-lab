"""Model architecture specifications and documented provenance for LLM inference analysis.

All model configurations are derived from published peer-reviewed papers or official
open-source model configuration repositories.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Precision(str, Enum):
    """Numerical precision and byte footprint per parameter or activation."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT4 = "int4"

    @property
    def bytes_per_element(self) -> float:
        mapping = {
            Precision.FP32: 4.0,
            Precision.FP16: 2.0,
            Precision.BF16: 2.0,
            Precision.FP8: 1.0,
            Precision.INT4: 0.5,
        }
        return mapping[self]


@dataclass(frozen=True)
class ModelConfig:
    """Transformer model structural dimensions relevant to memory and inference rooflines."""
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    vocab_size: int
    max_seq_len: int
    num_params: int
    provenance: str

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def is_mha(self) -> bool:
        """True if standard Multi-Head Attention (n_kv_heads == n_heads)."""
        return self.n_kv_heads == self.n_heads

    @property
    def is_gqa(self) -> bool:
        """True if Grouped-Query Attention (1 < n_kv_heads < n_heads)."""
        return 1 < self.n_kv_heads < self.n_heads

    @property
    def is_mqa(self) -> bool:
        """True if Multi-Query Attention (n_kv_heads == 1)."""
        return self.n_kv_heads == 1

    @property
    def gqa_ratio(self) -> float:
        """Ratio of query heads to KV heads."""
        return self.n_heads / self.n_kv_heads


# Preconfigured models with documented sources:

LLAMA_3_8B = ModelConfig(
    name="llama-3-8b",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,  # GQA (4:1 query:kv ratio)
    d_model=4096,
    vocab_size=128256,
    max_seq_len=8192,
    num_params=8_030_000_000,
    provenance="Meta Llama 3 paper ('The Llama 3 Herd of Models', Meta 2024; HuggingFace meta-llama/Meta-Llama-3-8B/config.json)",
)

MISTRAL_7B_V01 = ModelConfig(
    name="mistral-7b-v0.1",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,  # GQA (4:1 query:kv ratio)
    d_model=4096,
    vocab_size=32000,
    max_seq_len=8192,
    num_params=7_240_000_000,
    provenance="Jiang et al., 'Mistral 7B', arXiv:2310.06825 (2023); HuggingFace mistralai/Mistral-7B-v0.1/config.json",
)

LLAMA_1_7B = ModelConfig(
    name="llama-1-7b",
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,  # Classic MHA baseline (1:1 ratio)
    d_model=4096,
    vocab_size=32000,
    max_seq_len=2048,
    num_params=6_738_000_000,
    provenance="Touvron et al., 'LLaMA: Open and Efficient Foundation Language Models', arXiv:2302.13971 (2023); HuggingFace huggyllama/llama-7b/config.json",
)

LLAMA_3_70B = ModelConfig(
    name="llama-3-70b",
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,  # GQA (8:1 query:kv ratio)
    d_model=8192,
    vocab_size=128256,
    max_seq_len=8192,
    num_params=70_600_000_000,
    provenance="Meta Llama 3 paper ('The Llama 3 Herd of Models', Meta 2024; HuggingFace meta-llama/Meta-Llama-3-70B/config.json)",
)

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama-3-8b": LLAMA_3_8B,
    "mistral-7b-v0.1": MISTRAL_7B_V01,
    "llama-1-7b": LLAMA_1_7B,
    "llama-3-70b": LLAMA_3_70B,
}


def get_model_config(name: str) -> ModelConfig:
    """Retrieve model configuration by registered canonical name."""
    canonical = name.lower().strip()
    if canonical not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[canonical]
