# Google Colab vLLM Hardware Validation Guide

This guide details how to execute physical GPU inference benchmarks against cloud hardware (Google Colab / cloud GPUs) using [vLLM](https://github.com/vllm-project/vllm) and compare empirical observations against the analytical predictions of our Phase 1 simulator.

---

## 1. Quick Start: Executing Benchmark V2 in Google Colab

### Prerequisites
1. Open a new notebook at [Google Colab](https://colab.research.google.com).
2. Attach a GPU accelerator: **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU** (free tier) or **L4 / A100** (Colab Pro).

### Execution Commands

```bash
# Cell 1: Install vLLM and dependencies
!pip install --quiet vllm
```

```bash
# Cell 2: Run Benchmark V2 (Recommended: 10 measured trials + 2 warm-up passes per configuration)
!python colab/vllm_validation_v2.py --trials 10 --warmup 2 --output colab_vllm_measured_results_v2.json
```

*(Optional)*: Specify a custom model:
```bash
!python colab/vllm_validation_v2.py --model meta-llama/Llama-3.2-1B-Instruct --trials 10 --warmup 2 --output colab_vllm_measured_results_v2.json
```

*(Legacy V1 execution)*:
```bash
!python colab/vllm_validation.py --trials 3 --output colab_vllm_measured_results.json
```

---

## 2. Model Selection by GPU VRAM Capacity

The validation script automatically detects GPU VRAM and selects an open model to safely maximize GPU saturation without triggering Out-Of-Memory (OOM) errors:

| GPU Hardware | VRAM | Architecture | Default Model | Quantization | Rationale |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **NVIDIA T4** (Free Colab) | ~14.5-16 GiB | Turing (Compute 7.5) | `Qwen/Qwen2.5-1.5B-Instruct` | Native FP16 | Weights occupy ~3.0 GiB, leaving ~11.5 GiB free for KV-cache blocks. Safely supports concurrency sweeps up to 32 streams without OOM. |
| **NVIDIA L4** (Colab Pro) | ~22-24 GiB | Ada Lovelace (Compute 8.9) | `meta-llama/Meta-Llama-3-8B-Instruct` | Native FP16 | Direct architectural match to our Phase 1 LLaMA-3-8B analytical model. |
| **NVIDIA A100** (Colab Pro) | 40/80 GiB | Ampere (Compute 8.0) | `meta-llama/Meta-Llama-3-8B-Instruct` | Native FP16 | Direct architectural match to our Phase 1 NVIDIA A100-SXM4 roofline profile. |

---

## 3. Benchmark V2 Methodology Improvements

### A. Triton JIT Compilation & Kernel Warm-Up Protocol
* **Investigation**: vLLM relies on custom attention kernels (Triton and CUDA PagedAttention) parameterized by batch size, head dimension, and sequence length. When a new batch size (e.g., concurrency 1, 4, 16, 32) is encountered, Triton JIT-compiles kernels from IR to PTX/Cubin on the host GPU. This causes an initial latency spike of hundreds of milliseconds to seconds.
* **V2 Mitigation**:
  1. **Global Engine Priming**: Before benchmark sweeps begin, a priming pass runs dummy prefill and decode sequences to initialize driver context, cuBLAS/cuDNN handles, and base memory pools.
  2. **Multi-Pass Warm-Up**: Every configuration executes **at least 2 untimed warm-up passes** with CUDA synchronization before starting measured trials.
  3. **Separate Tracking**: Warm-up durations are recorded separately in the output JSON.
  4. **Important Disclaimer**: While multi-pass warm-up mitigates initial JIT compilation and allocator overheads, it **does not guarantee** that all dynamic kernel shapes (e.g. sequence growth during decode) have completed compilation.

### B. Sample Size & Deterministic Statistics ($N \ge 10$ Trials)
* **Sample Count**: V2 increases measured trials from 3 to at least 10 per configuration.
* **Traceable Raw Records**: Every single trial records:
  `trial_index`, `timestamp_utc`, `duration_s`, `actual_output_tokens`, `expected_output_tokens`, `token_count_mismatch`, and measured metric.
* **Token Count Mismatch Validation**: Output tokens are verified against expected tokens. If early stopping or decoding truncates generation, `token_count_mismatch: True` is flagged and reported.
* **Deterministic Statistical Conventions**:
  * **Sample Standard Deviation**: Uses Bessel's correction ($N - 1$ degrees of freedom, `ddof=1`).
  * **Percentile Interpolation**: Computed via standard linear interpolation between order statistics ($\text{rank} = \frac{P}{100} \times (N - 1)$), matching NumPy/NIST standards.
  * **Exploratory Tail Estimates**: With $N=10$, $p_{90}, p_{95},$ and $p_{99}$ are explicitly labeled as **exploratory approximations** rather than asymptotic tail-latency estimates. Reliable tail estimation requires large sample sizes ($N \ge 1000$) under continuous traffic.

### C. Input Format Compatibility
* Prompts are pre-tokenized on CPU and formatted into the standard vLLM structured prompt format:
  `prompts=[{"prompt_token_ids": seq} for seq in batch_tokens]`.
  This prevents regression against vLLM versions that do not accept raw `prompt_token_ids` keyword arguments.

---

## 4. Measurement Boundaries & Metric Definitions

To ensure scientific integrity and prevent conflating offline batch processing with online serving:

| Metric | Measured in V2? | Timing Boundary & Definition |
| :--- | :---: | :--- |
| **Offline Aggregate Batch Throughput** | **YES** | $\sum \text{Actual Output Tokens} / \text{Batch Duration (s)}$. Measured with CUDA synchronization enclosing `llm.generate()`. Measures maximum sustained token generation under full batch saturation. |
| **Offline Batch Duration** | **YES** | Wall-clock time (s) from batch dispatch until all $C$ concurrent requests complete decoding. |
| **Offline Prefill + 1-Token Latency** | **YES** | Single-request latency (ms) for GPU prefill GEMM/attention + 1 token decode step ($S_{\text{gen}}=1$). **Excludes CPU text tokenization** (prompts are pre-tokenized on CPU before timing). |
| **Online Serving TTFT** | **N/A** | Real online Time-to-First-Token includes client-server network transit and dynamic admission queue wait. Marked as `N/A`. |
| **Per-Token ITL / TPOT** | **N/A** | Offline `LLM.generate()` returns only when the entire batch is complete; per-token inter-token streaming latency requires an online HTTP/gRPC server (`vllm serve`). Marked as `N/A`. |
| **Admission Queue Wait** | **N/A** | All $C$ requests are admitted simultaneously at $t=0$; queue delay is marked as `N/A`. |

---

## 5. Output Schemas & Artifact Separation

All V1 physical artifacts remain preserved and untouched in `results/physical_t4_v1/`.
V2 runs produce separate, isolated artifacts:
* Colab Output JSON: `colab_vllm_measured_results_v2.json`
* Downstream Summary Directory: `results/physical_t4_v2/`

### V2 JSON Schema Sample
```json
{
  "benchmark_metadata": {
    "engine": "vllm",
    "benchmark_harness_version": "v2",
    "vllm_version": "0.32.0",
    "pytorch_version": "2.13.0+cu130",
    "cuda_version": "13.0",
    "gpu_name": "Tesla T4",
    "gpu_vram_gib": 14.56,
    "gpu_compute_capability": "7.5",
    "timestamp_utc": "2026-10-20T...",
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "precision": "float16",
    "num_trials": 10,
    "num_warmup_passes": 2,
    "metric_definitions": { ... }
  },
  "concurrency_sweep": [
    {
      "concurrency": 4,
      "prompt_tokens_per_req": 512,
      "output_tokens_per_req": 128,
      "expected_total_output_tokens": 512,
      "has_token_count_mismatch": false,
      "warmup_passes": 2,
      "warmup_durations_s": [4.120, 3.845],
      "trials": [
        {
          "trial_index": 1,
          "timestamp_utc": "2026-10-20T...",
          "duration_s": 3.8421,
          "expected_output_tokens": 512,
          "actual_output_tokens": 512,
          "token_count_mismatch": false,
          "offline_batch_throughput_tokens_per_s": 133.26
        }
      ],
      "summary_duration_s": {
        "mean": 3.8415,
        "median": 3.8420,
        "std_dev_sample": 0.0032,
        "min": 3.8370,
        "max": 3.8470,
        "p25": 3.8395,
        "p75": 3.8440,
        "p90_exploratory": 3.8461,
        "iqr": 0.0045
      },
      "summary_throughput_tokens_per_s": {
        "mean": 133.28,
        "median": 133.26,
        "std_dev_sample": 0.11,
        "p25": 133.19,
        "p75": 133.35,
        "p90_exploratory": 133.42,
        "iqr": 0.16
      }
    }
  ],
  "prompt_length_sweep": [
    {
      "prompt_tokens": 512,
      "output_tokens": 1,
      "has_token_count_mismatch": false,
      "warmup_passes": 2,
      "warmup_durations_ms": [31.5, 27.8],
      "trials": [ ... ],
      "summary_latency_ms": {
        "mean": 27.85,
        "median": 27.80,
        "std_dev_sample": 0.15,
        "p25": 27.70,
        "p75": 27.95,
        "p90_exploratory": 28.05,
        "iqr": 0.25
      }
    }
  ]
}
```
