"""Physical GPU validation harness using vLLM for Google Colab / cloud GPU runtimes.

DISCLAIMER & METHODOLOGY:
-------------------------
1. Offline Batch Generation Boundaries:
   This benchmark uses vLLM's offline engine (`vllm.LLM.generate`).
   - Measures: Aggregate output-token throughput (tokens/s), total batch duration (s),
     and isolated prefill + 1-token emission latency (ms).
   - Does NOT measure: Per-token Inter-Token Latency (ITL) or per-request queue wait.
     In offline batch mode, all requests are submitted simultaneously and return together;
     streaming ITL requires an online serving setup (`vllm serve` with streaming client).
2. Pre-Tokenization:
   Prompts are tokenized on CPU before entering timed blocks. Timers measure pure GPU
   execution (CUDA synchronized), excluding HuggingFace CPU string tokenization.
3. Repetitions & Statistical Rigor:
   Every configuration runs a warm-up step followed by at least 3 measured trials.
   Results report individual trial runs, median, and standard deviation.

PREREQUISITES IN GOOGLE COLAB:
------------------------------
    !pip install --quiet vllm
"""

from __future__ import annotations

import argparse
import json
import statistics
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
    # Repeat text to ensure sufficient length
    multiplier = (target_tokens // 5) + 10
    full_text = base_text * multiplier
    token_ids = tokenizer.encode(full_text)
    if len(token_ids) < target_tokens:
        token_ids = token_ids * ((target_tokens // len(token_ids)) + 2)
    return token_ids[:target_tokens]


def run_colab_vllm_benchmark(
    model_id: str | None = None,
    output_path: str = "colab_vllm_measured_results.json",
    num_trials: int = 3,
) -> dict[str, Any]:
    """Execute physical GPU validation benchmark and save structured measurements."""
    import torch
    from vllm import LLM, SamplingParams

    env_meta = get_gpu_environment_metadata()
    total_vram = env_meta["gpu_vram_gib"]

    if model_id is None:
        model_id = select_default_model(total_vram)

    print("=" * 70)
    print("LLM INFERENCE LAB: PHYSICAL GPU BENCHMARK (vLLM)")
    print(f"Device:    {env_meta['gpu_name']} ({total_vram:.2f} GiB VRAM)")
    print(f"PyTorch:   {env_meta['pytorch_version']} | CUDA: {env_meta['cuda_version']} | vLLM: {env_meta['vllm_version']}")
    print(f"Target:    {model_id}")
    print(f"Trials:    {num_trials} measured trials per configuration (+1 warm-up)")
    print("=" * 70)

    print(f"\nInitializing vLLM engine with {model_id}...")
    llm = LLM(
        model=model_id,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    results: dict[str, Any] = {
        "benchmark_metadata": {
            **env_meta,
            "model_id": model_id,
            "precision": "float16",
            "num_trials": num_trials,
            "measurement_boundaries": {
                "concurrency_sweep": "Measures total batch wall-clock duration and aggregate output token throughput. Per-token ITL and per-request queue delay are N/A in offline batch mode.",
                "prompt_length_sweep": "Measures single-request GPU prefill + 1 token sampling latency using pre-tokenized token IDs. Excludes CPU text tokenization.",
            },
        },
        "concurrency_sweep": [],
        "prompt_length_sweep": [],
    }

    # ------------------------------------------------------------------
    # Experiment 1: Concurrency Sweep (Batch Output Throughput)
    # Fixed: Prompt length = 512 tokens, Output length = 128 tokens
    # ------------------------------------------------------------------
    print("\n--- Running Experiment 1: Concurrency Sweep ---")
    prompt_len = 512
    output_len = 128
    concurrency_levels = [1, 4, 16, 32]

    # Pre-tokenize prompt once on CPU
    base_prompt_tokens = prepare_exact_tokens(tokenizer, prompt_len)
    assert len(base_prompt_tokens) == prompt_len, f"Expected {prompt_len} tokens, got {len(base_prompt_tokens)}"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,  # Force generating exactly output_len tokens
    )

    for c in concurrency_levels:
        batch_tokens = [base_prompt_tokens for _ in range(c)]
        expected_total_tokens = c * output_len

        # 1. Warm-up run (not timed)
        _ = llm.generate(prompt_token_ids=batch_tokens, sampling_params=sampling_params, use_tqdm=False)
        torch.cuda.synchronize()

        # 2. Measured trials
        durations: list[float] = []
        throughputs: list[float] = []

        for _ in range(num_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate(prompt_token_ids=batch_tokens, sampling_params=sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            dur = t1 - t0
            actual_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            tput = actual_tokens / dur if dur > 0 else 0.0

            durations.append(dur)
            throughputs.append(tput)

        median_dur = statistics.median(durations)
        std_dur = statistics.stdev(durations) if len(durations) > 1 else 0.0
        median_tput = statistics.median(throughputs)
        std_tput = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0

        entry = {
            "concurrency": c,
            "prompt_tokens_per_req": prompt_len,
            "output_tokens_per_req": output_len,
            "total_output_tokens": expected_total_tokens,
            "trial_durations_s": [round(d, 4) for d in durations],
            "trial_throughputs_tok_s": [round(tp, 2) for tp in throughputs],
            "median_duration_s": round(median_dur, 4),
            "std_duration_s": round(std_dur, 4),
            "median_throughput_tokens_per_s": round(median_tput, 2),
            "std_throughput_tokens_per_s": round(std_tput, 2),
            "per_token_itl_ms": "N/A (offline batch mode)",
            "queue_latency_ms": "N/A (offline batch mode)",
        }
        results["concurrency_sweep"].append(entry)
        print(
            f"Concurrency {c:2d}: {median_tput:7.2f} ± {std_tput:5.2f} tok/s "
            f"(Median Duration: {median_dur:5.2f}s across {num_trials} trials)"
        )

    # ------------------------------------------------------------------
    # Experiment 2: Prompt Length Sweep (Prefill + 1-Token Latency)
    # Fixed: Batch size = 1, Output length = 1 token
    # ------------------------------------------------------------------
    print("\n--- Running Experiment 2: Prompt Length Sweep (Prefill Latency) ---")
    prompt_lengths = [128, 256, 512, 1024, 2048]
    if total_vram >= 20.0:
        prompt_lengths.append(4096)

    sampling_params_1token = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    for plen in prompt_lengths:
        prompt_ids = prepare_exact_tokens(tokenizer, plen)
        assert len(prompt_ids) == plen, f"Expected {plen} tokens, got {len(prompt_ids)}"

        # 1. Warm-up
        _ = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sampling_params_1token, use_tqdm=False)
        torch.cuda.synchronize()

        # 2. Measured trials
        latencies_ms: list[float] = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sampling_params_1token, use_tqdm=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        median_lat = statistics.median(latencies_ms)
        std_lat = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

        entry = {
            "prompt_tokens": plen,
            "output_tokens": 1,
            "trial_latencies_ms": [round(l, 2) for l in latencies_ms],
            "median_prefill_plus_1token_ms": round(median_lat, 2),
            "std_prefill_plus_1token_ms": round(std_lat, 2),
            "timing_scope": "GPU prefill GEMM/attention + 1 token sampling. Excludes CPU text tokenization.",
        }
        results["prompt_length_sweep"].append(entry)
        print(f"Prompt Tokens: {plen:5d} -> Prefill + 1tok: {median_lat:6.2f} ± {std_lat:5.2f} ms")

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PHYSICAL BENCHMARK COMPLETE. Results saved to: {output_path}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run physical GPU vLLM inference validation benchmark.")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model ID (default: auto-detected by VRAM)")
    parser.add_argument("--output", type=str, default="colab_vllm_measured_results.json", help="Path to write JSON output")
    parser.add_argument("--trials", type=int, default=3, help="Number of measured trials per configuration")
    args = parser.parse_args()

    run_colab_vllm_benchmark(model_id=args.model, output_path=args.output, num_trials=args.trials)
