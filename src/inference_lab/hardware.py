"""Hardware profile definitions and memory/compute bandwidth specifications.

Hardware performance characteristics are documented from official NVIDIA technical whitepapers
and datasheet specifications.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware capabilities relevant to inference roofline analysis."""
    name: str
    peak_fp16_tflops: float       # Dense FP16/BF16 Tensor Core peak TFLOPS (no sparsity)
    memory_bandwidth_gb_s: float  # Peak theoretical device memory bandwidth (GB/s)
    vram_gb: float                # Total device VRAM capacity (GB)
    default_compute_efficiency: float = 0.50  # Typical MFU / GEMM execution efficiency
    default_memory_efficiency: float = 0.70   # Empirical achievable bandwidth efficiency (e.g. 70-80%)
    provenance: str = ""

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gb * (1024**3))

    @property
    def effective_memory_bandwidth_bytes_s(self) -> float:
        """Effective memory bandwidth in bytes per second after efficiency penalty."""
        return self.memory_bandwidth_gb_s * 1e9 * self.default_memory_efficiency

    @property
    def effective_fp16_flops_s(self) -> float:
        """Effective FP16 FLOPs/s after efficiency penalty."""
        return self.peak_fp16_tflops * 1e12 * self.default_compute_efficiency


# Standard reference hardware profiles:

A100_SXM4_80GB = HardwareConfig(
    name="nvidia-a100-sxm4-80gb",
    peak_fp16_tflops=312.0,
    memory_bandwidth_gb_s=2039.0,
    vram_gb=80.0,
    provenance="NVIDIA A100 Tensor Core GPU Architecture Whitepaper (Table 1: SXM4 specifications)",
)

L4_24GB = HardwareConfig(
    name="nvidia-l4-24gb",
    peak_fp16_tflops=121.0,
    memory_bandwidth_gb_s=300.0,
    vram_gb=24.0,
    provenance="NVIDIA L4 Tensor Core GPU Datasheet (Ada Lovelace architecture)",
)

T4_16GB = HardwareConfig(
    name="nvidia-t4-16gb",
    peak_fp16_tflops=65.0,
    memory_bandwidth_gb_s=320.0,
    vram_gb=16.0,
    provenance="NVIDIA T4 Tensor Core GPU Datasheet (Turing architecture, free-tier Colab runtime)",
)

HARDWARE_REGISTRY: dict[str, HardwareConfig] = {
    "nvidia-a100-sxm4-80gb": A100_SXM4_80GB,
    "nvidia-l4-24gb": L4_24GB,
    "nvidia-t4-16gb": T4_16GB,
}


def get_hardware_config(name: str) -> HardwareConfig:
    """Retrieve hardware profile by canonical name."""
    canonical = name.lower().strip()
    if canonical not in HARDWARE_REGISTRY:
        raise KeyError(f"Hardware '{name}' not found. Available: {list(HARDWARE_REGISTRY.keys())}")
    return HARDWARE_REGISTRY[canonical]
