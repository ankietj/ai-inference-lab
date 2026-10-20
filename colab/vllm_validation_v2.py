"""Physical GPU validation harness V2 using vLLM for Google Colab / cloud GPU runtimes.

BENCHMARK V2 METHODOLOGY & BOUNDARIES:
--------------------------------------
1. Offline Batch Generation Boundaries:
   This benchmark uses vLLM's offline engine (`vllm.LLM.generate`).
   - Measures:
     * Offline Aggregate Output-Token Throughput (tokens/s): actual output tokens / batch wall-clock duration.
     * Offline Batch Duration (s): synchronized CUDA wall-clock time for batch completion.
     * Offline Prefill + 1-Token Latency (ms): isolated GPU prefill GEMM/attention + 1-token decode emission.
   - Does NOT measure:
     * Per-token Inter-Token Latency (ITL / TPOT): In offline batch generation, tokens are not streamed;
       per-token metrics require an online streaming server architecture (`vllm serve`).
     * Admission Queue Wait: All requests are submitted simultaneously at t=0; queue wait is N/A.
     * Online Serving TTFT: True TTFT includes network transit and dynamic server queueing.
2. Pre-Tokenization:
   Prompts are pre-tokenized on CPU before entering timed blocks. Timers measure pure GPU
   execution (CUDA synchronized), excluding HuggingFace CPU string tokenization.
3. Input Format Compatibility:
   Inputs are passed to `llm.generate()` using the standard vLLM structured prompt format:
   `prompts=[{"prompt_token_ids": token_ids}, ...]`.
4. Kernel Warm-Up & Triton JIT Mitigation:
   - Initial global engine priming runs dummy prefill and decode sequences to initialize CUDA contexts,
     cuBLAS/cuDNN handles, and base allocator pools.
   - Every parameter configuration executes at least 2 untimed warm-up passes with CUDA synchronization
     before measured trials.
   - Warm-up durations are recorded separately. Note: multi-pass warm-up mitigates initial Triton JIT
     compilation overheads but cannot theoretically guarantee that all dynamic kernel shapes are compiled.
5. Statistical Rigor (N >= 10 Trials):
   - Every individual trial's order index, timestamp, duration, token counts, and metric are preserved.
   - Token counts are explicitly verified; mismatches are flagged rather than silently ignored.
   - Deterministic summary statistics calculate mean, median, sample std (ddof=1), min, max,
     p25, p50, p75, p90, p95, p99 (linear interpolation), and IQR.
   - With N=10, p90, p95, and p99 are documented as exploratory rather than asymptotic tail estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def get_gpu_environment_metadata() -> dict[str, Any]:
    """Inspect and return verified physical GPU and software runtime metadata."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. Please run in an environment with PyTorch installed.")

    if not torch.cuda.is_available():
        raise SystemError(
            "CUDA is not available on this runtime.\n"
            "If running in Google Colab, go to 'Runtime' -> 'Change runtime type' "
            "and select a GPU accelerator (e.g., T4 GPU, L4 GPU, or A100 GPU)."
        )

    device_id = 0
    props = torch.cuda.get_device_properties(device_id)
    gpu_name = torch.cuda.get_device_name(device_id)
    total_vram_gib = props.total_memory / (1024**3)

    vllm_ver = "unknown"
    try:
        import vllm
        vllm_ver = getattr(vllm, "__version__", "unknown")
    except ImportError:
        pass

    return {
        "engine": "vllm",
        "benchmark_harness_version": "v2",
        "vllm_version": vllm_ver,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_vram_gib": round(total_vram_gib, 2),
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def select_default_model(total_vram_gib: float) -> str:
    """Recommend an open model fitting comfortably within detected GPU memory."""
    if total_vram_gib < 20.0:
        # Standard free Colab runtime: NVIDIA T4 (15-16 GiB VRAM)
        # 1.5B model takes ~3.0 GiB in FP16, leaving ~12 GiB for KV-cache blocks
        return "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        # High-memory runtime: NVIDIA L4 (24 GiB) or A100 (40/80 GiB)
        # Direct architectural match to our LLaMA-3-8B analytical model
        return "meta-llama/Meta-Llama-3-8B-Instruct"


def prepare_exact_tokens(
    tokenizer: Any,
    target_tokens: int,
    base_text: str = "Artificial intelligence and distributed high-performance computing systems architecture. ",
) -> list[int]:
    """Generate a token ID sequence of exact requested length using the model tokenizer."""
    multiplier = (target_tokens // 5) + 10
    full_text = base_text * multiplier
    token_ids = tokenizer.encode(full_text)
    if len(token_ids) < target_tokens:
        token_ids = token_ids * ((target_tokens // len(token_ids)) + 2)
    return token_ids[:target_tokens]


def format_vllm_token_inputs(batch_tokens: list[list[int]]) -> list[dict[str, list[int]]]:
    """Format batch of token ID sequences into the standard vLLM structured prompt format."""
    return [{"prompt_token_ids": seq} for seq in batch_tokens]


def compute_distribution_statistics(
    values: list[float],
    metric_name: str = "metric",
) -> dict[str, Any]:
    """Compute deterministic summary statistics from raw trial measurements.

    Statistical Conventions:
    - Sample standard deviation uses N - 1 degrees of freedom (Bessel's correction).
    - Percentiles use linear interpolation between order statistics (rank = p/100 * (N - 1)).
    - With N=10 trials, p90, p95, and p99 are labeled as exploratory rather than asymptotic tail estimates.
    """
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute statistics on an empty list.")

    sorted_vals = sorted(values)

    def linear_percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        rank = (p / 100.0) * (n - 1)
        k = int(rank)
        d = rank - k
        if k >= n - 1:
            return sorted_vals[-1]
        return sorted_vals[k] + d * (sorted_vals[k + 1] - sorted_vals[k])

    mean_val = sum(sorted_vals) / n
    median_val = linear_percentile(50.0)

    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / (n - 1)
        std_val = variance ** 0.5
    else:
        variance = 0.0
        std_val = 0.0

    p25 = linear_percentile(25.0)
    p75 = linear_percentile(75.0)
    p90 = linear_percentile(90.0)
    p95 = linear_percentile(95.0)
    p99 = linear_percentile(99.0)
    iqr = p75 - p25

    return {
        "metric_name": metric_name,
        "sample_count": n,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std_dev_sample": round(std_val, 4),
        "variance_sample": round(variance, 6),
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "p25": round(p25, 4),
        "p50": round(median_val, 4),
        "p75": round(p75, 4),
        "p90_exploratory": round(p90, 4),
        "p95_exploratory": round(p95, 4),
        "p99_exploratory": round(p99, 4),
        "iqr": round(iqr, 4),
        "statistical_conventions": {
            "std_dev_ddof": 1,
            "percentile_interpolation": "linear (rank = p/100 * (N - 1))",
            "tail_estimate_reliability_note": (
                "p90, p95, and p99 are exploratory when N=10; robust tail-latency estimates "
                "require large sample populations (N >= 1000) under online traffic."
            ),
        },
    }


def run_colab_vllm_benchmark_v2(
    model_id: str | None = None,
    output_path: str = "colab_vllm_measured_results_v2.json",
    num_trials: int = 10,
    num_warmup_passes: int = 2,
) -> dict[str, Any]:
    """Execute physical GPU validation benchmark V2 with rigorous warm-up and statistical recording."""
    import torch
    from vllm import LLM, SamplingParams

    if num_trials < 1:
        raise ValueError(f"num_trials must be at least 1, got {num_trials}")
    if num_warmup_passes < 2:
        print(f"Warning: Recommended warm-up passes is at least 2. Proceeding with {num_warmup_passes}.")

    env_meta = get_gpu_environment_metadata()
    total_vram = env_meta["gpu_vram_gib"]

    if model_id is None:
        model_id = select_default_model(total_vram)

    print("=" * 75)
    print("LLM INFERENCE LAB: PHYSICAL GPU BENCHMARK V2 (vLLM)")
    print(f"Device:      {env_meta['gpu_name']} ({total_vram:.2f} GiB VRAM)")
    print(f"PyTorch:     {env_meta['pytorch_version']} | CUDA: {env_meta['cuda_version']} | vLLM: {env_meta['vllm_version']}")
    print(f"Target:      {model_id}")
    print(f"Trials:      {num_trials} measured trials per configuration")
    print(f"Warm-Up:     {num_warmup_passes} untimed warm-up passes per configuration (+ global priming)")
    print("=" * 75)

    print(f"\nInitializing vLLM engine with {model_id}...")
    llm = LLM(
        model=model_id,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # ------------------------------------------------------------------
    # Global Priming Step
    # ------------------------------------------------------------------
    print("\n--- Priming vLLM Engine & CUDA Driver ---")
    prime_prompt_ids = prepare_exact_tokens(tokenizer, 128)
    prime_prompts = format_vllm_token_inputs([prime_prompt_ids, prime_prompt_ids])
    prime_sampling = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    torch.cuda.synchronize()
    _ = llm.generate(prompts=prime_prompts, sampling_params=prime_sampling, use_tqdm=False)
    torch.cuda.synchronize()
    print("Engine priming complete.")

    results: dict[str, Any] = {
        "benchmark_metadata": {
            **env_meta,
            "model_id": model_id,
            "precision": "float16",
            "num_trials": num_trials,
            "num_warmup_passes": num_warmup_passes,
            "metric_definitions": {
                "concurrency_sweep_metric": "offline_batch_throughput_tokens_per_s",
                "concurrency_sweep_description": (
                    "Offline aggregate output-token throughput under batch saturation. "
                    "Calculated as total measured output tokens divided by batch wall-clock duration. "
                    "Does NOT measure online serving throughput or dynamic queue delay."
                ),
                "prompt_length_sweep_metric": "offline_prefill_plus_1token_latency_ms",
                "prompt_length_sweep_description": (
                    "Single-request offline GPU prefill GEMM/attention + 1-token decode step. "
                    "Prompts are pre-tokenized on CPU before timing; excludes CPU string tokenization. "
                    "Does NOT measure online serving TTFT (which includes network transit and queue wait)."
                ),
                "serving_latency_disclaimer": (
                    "Offline batch generation (vllm.LLM.generate) submits all requests concurrently and returns "
                    "only when the entire batch completes. Inter-Token Latency (ITL/TPOT) and Queue Waiting Time "
                    "cannot be measured offline; they require an online streaming server architecture (e.g. vllm serve)."
                ),
                "warmup_and_jit_notes": (
                    f"Executed {num_warmup_passes} untimed warm-up passes per configuration to trigger Triton JIT compilation "
                    "and prime CUDA memory pools. This reduces JIT noise but does not guarantee that all dynamic kernel shapes "
                    "have completed compilation."
                ),
            },
        },
        "concurrency_sweep": [],
        "prompt_length_sweep": [],
    }

    # ------------------------------------------------------------------
    # Experiment 1: Concurrency Sweep (Offline Batch Throughput)
    # Fixed: Prompt length = 512 tokens, Output length = 128 tokens
    # ------------------------------------------------------------------
    print("\n--- Running Experiment 1: Concurrency Sweep (Offline Batch Throughput) ---")
    prompt_len = 512
    output_len = 128
    concurrency_levels = [1, 4, 16, 32]

    base_prompt_tokens = prepare_exact_tokens(tokenizer, prompt_len)
    assert len(base_prompt_tokens) == prompt_len, f"Expected {prompt_len} tokens, got {len(base_prompt_tokens)}"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    for c in concurrency_levels:
        batch_tokens = [base_prompt_tokens for _ in range(c)]
        prompts = format_vllm_token_inputs(batch_tokens)
        expected_total_tokens = c * output_len

        # 1. Multi-pass Warm-up (tracked separately)
        warmup_durations_s: list[float] = []
        for _ in range(num_warmup_passes):
            torch.cuda.synchronize()
            t_w0 = time.perf_counter()
            _ = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            t_w1 = time.perf_counter()
            warmup_durations_s.append(round(t_w1 - t_w0, 4))

        # 2. Measured trials
        trials: list[dict[str, Any]] = []
        durations_s: list[float] = []
        throughputs_tok_s: list[float] = []
        has_any_token_mismatch = False

        for trial_idx in range(1, num_trials + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            dur = t1 - t0
            actual_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            mismatch = (actual_tokens != expected_total_tokens)
            if mismatch:
                has_any_token_mismatch = True
                print(f"WARNING: Concurrency {c} Trial {trial_idx}: Token count mismatch! Expected {expected_total_tokens}, got {actual_tokens}")

            tput = actual_tokens / dur if dur > 0 else 0.0

            durations_s.append(dur)
            throughputs_tok_s.append(tput)

            trials.append({
                "trial_index": trial_idx,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "duration_s": round(dur, 6),
                "expected_output_tokens": expected_total_tokens,
                "actual_output_tokens": actual_tokens,
                "token_count_mismatch": mismatch,
                "offline_batch_throughput_tokens_per_s": round(tput, 2),
            })

        duration_stats = compute_distribution_statistics(durations_s, metric_name="duration_s")
        throughput_stats = compute_distribution_statistics(throughputs_tok_s, metric_name="offline_batch_throughput_tokens_per_s")

        entry = {
            "concurrency": c,
            "prompt_tokens_per_req": prompt_len,
            "output_tokens_per_req": output_len,
            "expected_total_output_tokens": expected_total_tokens,
            "has_token_count_mismatch": has_any_token_mismatch,
            "warmup_passes": len(warmup_durations_s),
            "warmup_durations_s": warmup_durations_s,
            "trials": trials,
            "summary_duration_s": duration_stats,
            "summary_throughput_tokens_per_s": throughput_stats,
            "per_token_itl_ms": "N/A (offline batch mode)",
            "queue_latency_ms": "N/A (offline batch mode)",
        }
        results["concurrency_sweep"].append(entry)
        print(
            f"Concurrency {c:2d}: {throughput_stats['median']:7.2f} tok/s (median), "
            f"Mean: {throughput_stats['mean']:7.2f} ± {throughput_stats['std_dev_sample']:5.2f} tok/s, "
            f"IQR: {throughput_stats['iqr']:5.2f} tok/s | "
            f"Warmups: {warmup_durations_s}s"
        )

    # ------------------------------------------------------------------
    # Experiment 2: Prompt Length Sweep (Offline Prefill + 1-Token Latency)
    # Fixed: Batch size = 1, Output length = 1 token
    # ------------------------------------------------------------------
    print("\n--- Running Experiment 2: Prompt Length Sweep (Prefill + 1-Token Latency) ---")
    prompt_lengths = [128, 256, 512, 1024, 2048]
    if total_vram >= 20.0:
        prompt_lengths.append(4096)

    sampling_params_1token = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    for plen in prompt_lengths:
        prompt_ids = prepare_exact_tokens(tokenizer, plen)
        assert len(prompt_ids) == plen, f"Expected {plen} tokens, got {len(prompt_ids)}"
        prompts_1token = format_vllm_token_inputs([prompt_ids])

        # 1. Multi-pass Warm-up
        warmup_durations_ms: list[float] = []
        for _ in range(num_warmup_passes):
            torch.cuda.synchronize()
            t_w0 = time.perf_counter()
            _ = llm.generate(prompts=prompts_1token, sampling_params=sampling_params_1token, use_tqdm=False)
            torch.cuda.synchronize()
            t_w1 = time.perf_counter()
            warmup_durations_ms.append(round((t_w1 - t_w0) * 1000.0, 2))

        # 2. Measured trials
        trials_prompt: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        has_any_token_mismatch = False

        for trial_idx in range(1, num_trials + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate(prompts=prompts_1token, sampling_params=sampling_params_1token, use_tqdm=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            actual_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            mismatch = (actual_tokens != 1)
            if mismatch:
                has_any_token_mismatch = True
                print(f"WARNING: Prompt length {plen} Trial {trial_idx}: Token count mismatch! Expected 1, got {actual_tokens}")

            latencies_ms.append(lat_ms)

            trials_prompt.append({
                "trial_index": trial_idx,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "latency_ms": round(lat_ms, 3),
                "expected_output_tokens": 1,
                "actual_output_tokens": actual_tokens,
                "token_count_mismatch": mismatch,
            })

        latency_stats = compute_distribution_statistics(latencies_ms, metric_name="offline_prefill_plus_1token_latency_ms")

        entry = {
            "prompt_tokens": plen,
            "output_tokens": 1,
            "has_token_count_mismatch": has_any_token_mismatch,
            "warmup_passes": len(warmup_durations_ms),
            "warmup_durations_ms": warmup_durations_ms,
            "trials": trials_prompt,
            "summary_latency_ms": latency_stats,
            "timing_scope": (
                "GPU prefill GEMM/attention + 1 token decode step. Prompts are pre-tokenized on CPU. "
                "Excludes CPU text tokenization. Does NOT measure online serving TTFT."
            ),
        }
        results["prompt_length_sweep"].append(entry)
        print(
            f"Prompt Tokens: {plen:5d} -> Prefill + 1tok: {latency_stats['median']:6.2f} ms (median), "
            f"Mean: {latency_stats['mean']:6.2f} ± {latency_stats['std_dev_sample']:5.2f} ms, "
            f"IQR: {latency_stats['iqr']:5.2f} ms | "
            f"Warmups: {warmup_durations_ms}ms"
        )

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"PHYSICAL BENCHMARK V2 COMPLETE. Results saved to: {output_path}")
    print("=" * 75)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run physical GPU vLLM inference validation benchmark V2.")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model ID (default: auto-detected by VRAM)")
    parser.add_argument("--output", type=str, default="colab_vllm_measured_results_v2.json", help="Path to write JSON output")
    parser.add_argument("--trials", type=int, default=10, help="Number of measured trials per configuration (default: 10)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of untimed warm-up passes per configuration (default: 2)")
    args = parser.parse_args()

    run_colab_vllm_benchmark_v2(
        model_id=args.model,
        output_path=args.output,
        num_trials=args.trials,
        num_warmup_passes=args.warmup,
    )
