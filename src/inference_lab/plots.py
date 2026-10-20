"""Visualization module to generate reproducible charts from experimental results (V2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def generate_phase1_charts(
    results_path: Path | str = "results/v2/phase1_results_v2.json",
    output_image_path: Path | str = "results/v2/figures/phase1_scaling_analysis_v2.png",
) -> Path:
    """Generate a 4-panel publication-grade figure summarizing Phase 1 V2 scaling results."""
    with open(results_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    out_file = Path(output_image_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "LLM Inference Scaling Laws (Analytical & Workload Simulation - V2 Corrected)\n"
        "Model: LLaMA-3-8B | Target HW: NVIDIA A100-SXM4-80GB (Simulated)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # -------------------------------------------------------------
    # Panel 1: Prompt Length vs Prefill Service Time & TTFT
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    p_data = data["prompt_length_sweep"]
    p_lens = [d["prompt_length"] for d in p_data]
    p_prefill = [d["avg_prefill_service_ms"] for d in p_data]
    p_ttft = [d["avg_ttft_ms"] for d in p_data]

    ax1.plot(p_lens, p_prefill, marker="o", color="#1f77b4", linewidth=2.5, label="Prefill GPU Service Latency")
    ax1.plot(p_lens, p_ttft, marker="s", linestyle="--", color="#ff7f0e", linewidth=1.8, label="Total TTFT (Queue + Prefill)")
    ax1.set_title("1. Prompt Length vs. Prefill Service Time & TTFT", fontweight="bold")
    ax1.set_xlabel("Prompt Length (Tokens)")
    ax1.set_ylabel("Latency (milliseconds)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # -------------------------------------------------------------
    # Panel 2: Output Length vs E2E Latency
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    o_data = data["output_length_sweep"]
    o_lens = [d["output_length"] for d in o_data]
    o_e2e = [d["avg_e2e_ms"] / 1000.0 for d in o_data]

    ax2.plot(o_lens, o_e2e, marker="^", color="#2ca02c", linewidth=2.5, label="Mean Turnaround Latency")
    ax2.set_title("2. Output Length vs. Total Turnaround Latency", fontweight="bold")
    ax2.set_xlabel("Output Tokens Generated")
    ax2.set_ylabel("End-to-End Latency (seconds)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    # -------------------------------------------------------------
    # Panel 3: Concurrency vs Throughput & TPOT
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    c_data = data["concurrency_sweep"]
    concs = [d["concurrency"] for d in c_data]
    throughputs = [d["throughput_tokens_per_s"] for d in c_data]
    tpots = [d["avg_tpot_ms"] for d in c_data]

    color_tput = "#9467bd"
    color_tpot = "#d62728"

    ax3.set_title("3. Request Concurrency vs. Serving Throughput & TPOT", fontweight="bold")
    ax3.set_xlabel("Max Request Concurrency")
    ax3.set_ylabel("Throughput (Output Tokens / sec)", color=color_tput, fontweight="bold")
    line1 = ax3.plot(concs, throughputs, marker="o", color=color_tput, linewidth=2.5, label="Throughput (tokens/s)")
    ax3.tick_params(axis="y", labelcolor=color_tput)
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel("TPOT (ms / decode step)", color=color_tpot, fontweight="bold")
    line2 = ax3_twin.plot(concs, tpots, marker="x", linestyle="--", color=color_tpot, linewidth=2, label="TPOT (ms)")
    ax3_twin.tick_params(axis="y", labelcolor=color_tpot)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="center left")

    # -------------------------------------------------------------
    # Panel 4: KV-Cache Footprint: MHA vs GQA across Context Length
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    kv_data = data["kv_cache_comparison"]
    ctx_lens = [d["context_length"] for d in kv_data]
    mha_mib = [d["mha_fp16_mib"] for d in kv_data]
    gqa_fp16_mib = [d["gqa_fp16_mib"] for d in kv_data]
    gqa_fp8_mib = [d["gqa_fp8_mib"] for d in kv_data]

    ax4.plot(ctx_lens, mha_mib, marker="o", color="#d62728", linewidth=2.5, label="LLaMA-1 7B (MHA, FP16)")
    ax4.plot(ctx_lens, gqa_fp16_mib, marker="s", color="#1f77b4", linewidth=2.5, label="LLaMA-3 8B (GQA, FP16) - 4x smaller")
    ax4.plot(ctx_lens, gqa_fp8_mib, marker="d", linestyle="--", color="#2ca02c", linewidth=2, label="LLaMA-3 8B (GQA, FP8) - 8x smaller")
    ax4.set_title("4. KV-Cache Memory per Sequence (MHA vs. GQA)", fontweight="bold")
    ax4.set_xlabel("Context Length (Tokens)")
    ax4.set_ylabel("KV-Cache Footprint per Sequence (MiB)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


if __name__ == "__main__":
    generate_phase1_charts()
    print("V2 Charts successfully saved.")
