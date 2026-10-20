"""Inference Lab: Research-grade LLM inference engineering and scaling analysis."""

from inference_lab.hardware import (
    A100_SXM4_80GB,
    HARDWARE_REGISTRY,
    HardwareConfig,
    L4_24GB,
    T4_16GB,
    get_hardware_config,
)
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
from inference_lab.models import (
    LLAMA_1_7B,
    LLAMA_3_8B,
    LLAMA_3_70B,
    MISTRAL_7B_V01,
    MODEL_REGISTRY,
    ModelConfig,
    Precision,
    get_model_config,
)
from inference_lab.simulator import (
    Request,
    SimpleWorkloadSimulator,
    SimulationConfig,
    SimulationResult,
)

__all__ = [
    "Precision",
    "ModelConfig",
    "LLAMA_3_8B",
    "MISTRAL_7B_V01",
    "LLAMA_1_7B",
    "LLAMA_3_70B",
    "MODEL_REGISTRY",
    "get_model_config",
    "HardwareConfig",
    "A100_SXM4_80GB",
    "L4_24GB",
    "T4_16GB",
    "HARDWARE_REGISTRY",
    "get_hardware_config",
    "kv_cache_elements_per_token",
    "kv_cache_bytes_per_token",
    "kv_cache_total_bytes",
    "kv_cache_num_blocks",
    "kv_cache_blocked_bytes",
    "kv_cache_internal_fragmentation",
    "model_weights_bytes",
    "max_active_tokens_in_vram",
    "Request",
    "SimulationConfig",
    "SimulationResult",
    "SimpleWorkloadSimulator",
]


def main() -> None:
    from inference_lab.experiment import run_phase1_experiments
    from inference_lab.plots import generate_phase1_charts

    print("Running Phase 1 LLM Inference Scaling Experiments...")
    run_phase1_experiments()
    print("Generating figures...")
    chart_path = generate_phase1_charts()
    print(f"Done! Chart generated at: {chart_path}")
