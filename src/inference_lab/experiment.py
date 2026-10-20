"""Reproducible experiment runner for Phase 1 scaling analysis (Corrected V2).

Executes controlled parameter sweeps across:
1. Prompt Length vs TTFT (decoupling Queue Waiting Time from Prefill Service Latency)
2. Output Length vs Turnaround Latency
3. Request Concurrency vs Throughput & TPOT
4. KV-Cache Memory comparison: MHA (LLaMA-1) vs GQA (LLaMA-3) in binary MiB units
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from inference_lab.hardware import A100_SXM4_80GB
from inference_lab.memory import kv_cache_total_bytes
from inference_lab.models import LLAMA_1_7B, LLAMA_3_8B, Precision
from inference_lab.simulator import SimpleWorkloadSimulator, SimulationConfig
from inference_lab.workload import WorkloadConfig, generate_workload


def run_phase1_experiments(output_dir: Path | str = "results/v2") -> dict[str, Any]:
    """Execute all Phase 1 scaling sweeps and return structured results with full metadata."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hardware = A100_SXM4_80GB
    model = LLAMA_3_8B
    precision = Precision.FP16
    base_seed = 42

    results: dict[str, Any] = {
        "metadata": {
            "experiment": "Phase 1: Analytical and Workload Simulation Scaling Laws (Corrected V2)",
            "model": model.name,
            "hardware": hardware.name,
            "precision": precision.value,
            "random_seed": base_seed,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware_specs": {
                "peak_fp16_tflops": hardware.peak_fp16_tflops,
                "memory_bandwidth_gb_s": hardware.memory_bandwidth_gb_s,
                "vram_gb": hardware.vram_gb,
                "compute_efficiency": hardware.default_compute_efficiency,
                "memory_efficiency": hardware.default_memory_efficiency,
            },
            "provenance_notes": [
                "Simulated with SimpleWorkloadSimulator based on hardware rooflines.",
                "Interleaved continuous batching with at most 1 prefill admitted per iteration.",
                "Prefill FLOPs includes both linear projections (2*P*S) and self-attention quadratic FLOPs (4*L*D*S^2).",
                "TPOT denominator corrected to count actual decode steps (output_length - 1).",
                "Memory units explicitly measured in binary MiB (1024^2 bytes) and GiB (1024^3 bytes).",
                "Not measured on physical GPUs. Serves as pre-experimental hypothesis for Colab/vLLM validation.",
            ],
            "omitted_factors": [
                "CUDA kernel launch jitter (constant 20us assumed).",
                "Host-to-device PCIe transfer and tokenizer overheads.",
                "Paged external fragmentation and memory bus contention.",
            ],
        },
        "prompt_length_sweep": [],
        "output_length_sweep": [],
        "concurrency_sweep": [],
        "kv_cache_comparison": [],
    }

    # -------------------------------------------------------------
    # Sweep 1: Prompt Length vs TTFT (Queue Time vs Prefill Service Time)
    # Fixed: Output Length = 128, Concurrency = 8, 24 requests
    # -------------------------------------------------------------
    prompt_lengths = [128, 256, 512, 1024, 2048, 4096]
    for plen in prompt_lengths:
        cfg = SimulationConfig(
            model=model,
            hardware=hardware,
            precision=precision,
            max_concurrency=8,
            random_seed=base_seed,
        )
        wl = WorkloadConfig(
            num_requests=24,
            prompt_length=plen,
            output_length=128,
            arrival_pattern="uniform",
            arrival_rate_rps=10.0,
            seed=base_seed,
        )
        reqs = generate_workload(wl)
        sim_res = SimpleWorkloadSimulator(cfg).run(reqs)
        results["prompt_length_sweep"].append({
            "prompt_length": plen,
            "avg_queue_ms": round(sim_res.avg_queue_time_s * 1000, 2),
            "p50_queue_ms": round(sim_res.p50_queue_time_s * 1000, 2),
            "avg_prefill_service_ms": round(sim_res.avg_prefill_service_time_s * 1000, 2),
            "p50_prefill_service_ms": round(sim_res.p50_prefill_service_time_s * 1000, 2),
            "avg_ttft_ms": round(sim_res.avg_ttft_s * 1000, 2),
            "p50_ttft_ms": round(sim_res.p50_ttft_s * 1000, 2),
            "p95_ttft_ms": round(sim_res.p90_ttft_s * 1000, 2),
            "avg_e2e_ms": round(sim_res.avg_e2e_latency_s * 1000, 2),
        })

    # -------------------------------------------------------------
    # Sweep 2: Output Length vs E2E Latency
    # Fixed: Prompt Length = 512, Concurrency = 8, 24 requests
    # -------------------------------------------------------------
    output_lengths = [64, 128, 256, 512, 1024]
    for olen in output_lengths:
        cfg = SimulationConfig(
            model=model,
            hardware=hardware,
            precision=precision,
            max_concurrency=8,
            random_seed=base_seed,
        )
        wl = WorkloadConfig(
            num_requests=24,
            prompt_length=512,
            output_length=olen,
            arrival_pattern="uniform",
            arrival_rate_rps=10.0,
            seed=base_seed,
        )
        reqs = generate_workload(wl)
        sim_res = SimpleWorkloadSimulator(cfg).run(reqs)
        results["output_length_sweep"].append({
            "output_length": olen,
            "avg_e2e_ms": round(sim_res.avg_e2e_latency_s * 1000, 2),
            "avg_tpot_ms": round(sim_res.avg_tpot_s * 1000, 3),
            "p50_tpot_ms": round(sim_res.p50_tpot_s * 1000, 3),
            "tokens_per_s": round(sim_res.tokens_per_second, 2),
        })

    # -------------------------------------------------------------
    # Sweep 3: Concurrency vs Throughput & TPOT
    # Fixed: Prompt Length = 512, Output Length = 256, 64 requests arriving in burst
    # -------------------------------------------------------------
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
    for c in concurrency_levels:
        cfg = SimulationConfig(
            model=model,
            hardware=hardware,
            precision=precision,
            max_concurrency=c,
            random_seed=base_seed,
        )
        wl = WorkloadConfig(
            num_requests=64,
            prompt_length=512,
            output_length=256,
            arrival_pattern="burst",
            arrival_rate_rps=0.0,
            seed=base_seed,
        )
        reqs = generate_workload(wl)
        sim_res = SimpleWorkloadSimulator(cfg).run(reqs)
        results["concurrency_sweep"].append({
            "concurrency": c,
            "throughput_tokens_per_s": round(sim_res.tokens_per_second, 2),
            "avg_tpot_ms": round(sim_res.avg_tpot_s * 1000, 3),
            "avg_queue_ms": round(sim_res.avg_queue_time_s * 1000, 2),
            "avg_prefill_service_ms": round(sim_res.avg_prefill_service_time_s * 1000, 2),
            "avg_ttft_ms": round(sim_res.avg_ttft_s * 1000, 2),
            "avg_e2e_ms": round(sim_res.avg_e2e_latency_s * 1000, 2),
            "peak_kv_cache_mib": round(sim_res.peak_kv_cache_bytes / (1024**2), 2),
        })

    # -------------------------------------------------------------
    # Sweep 4: Analytical KV-Cache Comparison: MHA vs GQA across Context Lengths
    # -------------------------------------------------------------
    context_lengths = [512, 1024, 2048, 4096, 8192]
    for ctx in context_lengths:
        mha_fp16_mib = kv_cache_total_bytes(LLAMA_1_7B, ctx, Precision.FP16) / (1024**2)
        gqa_fp16_mib = kv_cache_total_bytes(LLAMA_3_8B, ctx, Precision.FP16) / (1024**2)
        gqa_fp8_mib = kv_cache_total_bytes(LLAMA_3_8B, ctx, Precision.FP8) / (1024**2)
        results["kv_cache_comparison"].append({
            "context_length": ctx,
            "mha_fp16_mib": round(mha_fp16_mib, 2),
            "gqa_fp16_mib": round(gqa_fp16_mib, 2),
            "gqa_fp8_mib": round(gqa_fp8_mib, 2),
            "ratio_mha_to_gqa": round(mha_fp16_mib / gqa_fp16_mib, 1),
        })

    # Save to disk
    json_path = out_path / "phase1_results_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    data = run_phase1_experiments()
    print("Phase 1 V2 experiment complete. Data written to results/v2/phase1_results_v2.json")
