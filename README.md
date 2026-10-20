# LLM Inference Engineering Lab

A transparent, research-grade Python laboratory for studying LLM serving dynamics, KV-cache memory scaling, and latency/throughput trade-offs.

---

## 1. Research Epistemology: Distinguishing Results

In systems engineering research, conflating theoretical models, discrete simulations, and real-world engine measurements produces misleading claims. This project strictly enforces three distinct categories:

| Category | Definition | What It Captures | Limitations |
| :--- | :--- | :--- | :--- |
| **Analytical Estimates** | Pure closed-form arithmetic derived from hardware rooflines and model dimensions. | Theoretical upper bounds on FLOPs, memory transfer limits, and exact KV-cache tensor size per token. | Assumes zero queueing, zero OS/driver jitter, infinite bus efficiency, and ideal scheduling. |
| **Simulated Results** | Discrete workload simulation (`SimpleWorkloadSimulator`). | Request arrival queues, prefill compute bounds, decode weight-amortization across concurrent streams, and PagedAttention block paging. | **Does NOT predict physical vLLM performance**. Abstracted kernel launch delays; does not model CUDA stream synchronization, CPU-GPU PCIe transfers, or PyTorch runtime overheads. |
| **Measured Benchmarks (Real Server)** | Physical execution on real hardware (e.g. vLLM on NVIDIA GPUs). | Actual wall-clock time, kernel execution times, FlashAttention/PagedAttention kernel performance, memory fragmentation. | **Not yet executed locally**. Planned for Phase 2 on Google Colab with an NVIDIA GPU runtime. |

> **Explicit Disclaimer**: None of the simulated metrics in this repository are fabricated hardware measurements. They represent first-principles analytical simulations to generate falsifiable hypotheses for real hardware experiments.

---

## 2. Core First Principles: Prefill vs. Decode vs. Memory

LLM inference operates in two fundamentally different hardware regimes:

```text
[User Request Arrives]
         │
         ▼
┌───────────────────────────────────────────────────────────────────┐
│ 1. PREFILL PHASE (TTFT: Time To First Token)                      │
│ • Parallel computation over all S_prompt tokens                   │
│ • Compute-bound: High arithmetic intensity (FLOP/byte)            │
│ • Total FLOPs = 2 * P * S_prompt + 4 * L * d_model * S_prompt^2   │
│ • TTFT = Queue_Time + Prefill_Service_Time                        │
└───────────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────────────┐
│ 2. DECODE PHASE (TPOT: Time Per Output Token)                     │
│ • Autoregressive token-by-token generation                        │
│ • Memory-bandwidth bound: Low arithmetic intensity                │
│ • Must load full model weights M_weights + KV-cache per step      │
│ • Concurrency amortizes M_weights across active streams           │
│ • TPOT = (Finish_Time - Prefill_End_Time) / (Output_Tokens - 1)   │
└───────────────────────────────────────────────────────────────────┘
```

### Mathematical Formulas & Metric Definitions (V2 Corrected)

#### 1. Latency Decomposition & Identity
For every request, time intervals are strictly defined and verified:
- **Arrival Time** ($t_{\text{arr}}$): Client submission timestamp.
- **Start Time** ($t_{\text{start}}$): Timestamp when GPU begins executing prefill.
- **Prefill End Time** ($t_{\text{prefill\_end}}$): Timestamp when prefill completes and the first generated token is emitted.
- **Finish Time** ($t_{\text{finish}}$): Timestamp when the final output token is emitted.

Metrics are derived strictly as:
- **Queue Latency**: $T_{\text{queue}} = t_{\text{start}} - t_{\text{arr}}$
- **Prefill Service Latency**: $T_{\text{prefill\_service}} = t_{\text{prefill\_end}} - t_{\text{start}}$
- **Time To First Token (TTFT)**:
  $$\text{TTFT} = t_{\text{prefill\_end}} - t_{\text{arr}} = T_{\text{queue}} + T_{\text{prefill\_service}}$$
- **Time Per Output Token (TPOT)**:
  $$\text{TPOT} = \begin{cases} \frac{t_{\text{finish}} - t_{\text{prefill\_end}}}{S_{\text{gen}} - 1} & \text{if } S_{\text{gen}} > 1 \\ 0.0 & \text{if } S_{\text{gen}} \le 1 \end{cases}$$
  *(Explicit behavior: If $S_{\text{gen}} \le 1$, only the prefill token was generated. No decode intervals occurred; returns 0.0).*

#### 2. Prefill Roofline (Including Attention Quadratic FLOPs)
$$\text{FLOPs}_{\text{linear}} = 2 \times P \times S_{\text{prompt}}$$
$$\text{FLOPs}_{\text{attention}} = 4 \times n_{\text{layers}} \times d_{\text{model}} \times S_{\text{prompt}}^2$$
$$\text{FLOPs}_{\text{total}} = \text{FLOPs}_{\text{linear}} + \text{FLOPs}_{\text{attention}}$$
$$T_{\text{prefill\_compute}} = \frac{\text{FLOPs}_{\text{total}}}{\text{Peak TFLOPS} \times \eta_{\text{comp}}}$$
$$T_{\text{prefill\_mem}} = \frac{M_{\text{weights}} + M_{\text{KV}}(S_{\text{prompt}})}{\text{Peak Memory BW} \times \eta_{\text{mem}}}$$
$$T_{\text{prefill}} = \max(T_{\text{prefill\_compute}}, T_{\text{prefill\_mem}}) + T_{\text{kernel}}$$

#### 3. KV-Cache Memory per Token
For each token in the context, every transformer layer stores a Key vector and a Value vector:
$$\text{KV}_{\text{token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
Where $d_{\text{head}} = \frac{d_{\text{model}}}{n_{\text{heads}}}$.

- **Multi-Head Attention (MHA)**: $n_{\text{kv\_heads}} = n_{\text{heads}}$ (LLaMA-1 7B: 32 heads, 32 KV heads $\rightarrow 512\text{ KiB/token}$ in FP16).
- **Grouped-Query Attention (GQA)**: $n_{\text{kv\_heads}} \ll n_{\text{heads}}$ (LLaMA-3 8B: 32 query heads, 8 KV heads $\rightarrow 128\text{ KiB/token}$ in FP16).
- **Binary Unit Measurement**: Measured strictly in binary **Mebibytes (MiB, $1024^2$ bytes)** and **Gibibytes (GiB, $1024^3$ bytes)**.

#### 4. PagedAttention Block Allocation & Internal Fragmentation
In real serving engines (vLLM), KV cache is allocated in discrete blocks of $B_{\text{block}} = 16$ tokens:
$$N_{\text{blocks}}(L) = \left\lceil \frac{L}{B_{\text{block}}} \right\rceil, \quad L_{\text{allocated}} = N_{\text{blocks}}(L) \times B_{\text{block}}$$
$$\text{Fragmentation Ratio} = \frac{L_{\text{allocated}} - L}{L_{\text{allocated}}}$$

---

## 3. Simulator Scheduling Policy & Remaining Differences from vLLM

### Implemented Scheduling Policy (V2 Interleaved Batching)
- **Admission**: At most **one** pending request is admitted and prefilled per scheduling iteration.
- **Interleaving**: Once admitted, an active request decodes incrementally alongside other active streams rather than waiting for an entire batch of pending requests to execute serial prefills.
- **Queueing**: If active requests equal `max_concurrency` or active KV cache tokens exceed GPU VRAM, new arrivals wait in the FIFO admission queue.

### Documented Differences from Full vLLM Engine
1. **Chunked Prefill**: vLLM optionally supports chunked prefill (`--enable-chunked-prefill`), splitting large prompts across iterations to co-schedule prefill and decode in the same GEMM kernel. The current simulator runs full prompt prefill in a dedicated interval.
2. **CUDA Stream Synchronization & Memory Fragmentation**: vLLM manages a physical page table in GPU memory with dynamic block eviction/preemption. The simulator models block count and internal fragmentation, but does not simulate external memory fragmentation or page table lookups.
3. **Continuous Iteration Scheduling**: Real vLLM executes tokenization asynchronously and batches multiple incoming small prefills if `max_num_batched_tokens` permits.

---

## 4. Hardware Reference Profiles

Target hardware specs are derived from official NVIDIA architecture whitepapers:

| Profile | Peak Dense FP16 TFLOPS | Peak Memory Bandwidth (GB/s) | VRAM (GiB) | Memory Technology |
| :--- | :---: | :---: | :---: | :--- |
| **NVIDIA A100-SXM4-80GB** | 312.0 | 2039.0 | 80.0 | HBM2e |
| **NVIDIA L4-24GB** | 121.0 | 300.0 | 24.0 | GDDR6 |
| **NVIDIA T4-16GB** | 65.0 | 320.0 | 16.0 | GDDR6 |

---

## 5. Reproduction & Test Instructions

### Setup with `uv` and Python 3.12

```powershell
# 1. Sync project virtual environment
uv sync

# 2. Run unit tests (including hand-calculated numerical assertions)
uv run pytest -v

# 3. Run Phase 1 V2 scaling simulation and generate publication charts
uv run inference-lab
```

### Generated Artifacts
- **Original Results (Preserved)**: `results/phase1_results.json`, `results/figures/phase1_scaling_analysis.png`
- **Corrected V2 Results**: `results/v2/phase1_results_v2.json`, `results/v2/figures/phase1_scaling_analysis_v2.png`

---

## 6. Old vs. New Findings Comparison

Comparing the original baseline against the corrected V2 implementation for the same workload (LLaMA-3-8B on NVIDIA A100 rooflines):

| Workload Dimension | Metric | Original (V1) | Corrected (V2) | Engineering Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Prompt Length = 128** | Prefill Service Latency | *Not reported* | **$13.25\text{ ms}$** | Explicitly isolated GPU compute time from queue delay. |
| | Queue Delay | *Not reported* | **$758.91\text{ ms}$** | Quantifies admission queue wait under $\lambda = 10\text{ rps}$. |
| | Total TTFT | $772.88\text{ ms}$ | **$772.16\text{ ms}$** | Verified identity: $\text{TTFT} = 758.91 + 13.25 = 772.16\text{ ms}$. |
| **Prompt Length = 4096** | Prefill Service Latency | *Not reported* | **$478.08\text{ ms}$** | Accounts for quadratic attention FLOPs ($4 \cdot L \cdot D \cdot S^2$). |
| | Total TTFT | $5940.90\text{ ms}$ | **$6686.41\text{ ms}$** | Accurate attention compute increases prefill time and cascading queue wait. |
| **Output Length = 64** | TPOT (ms/token) | $16.273\text{ ms}$ | **$16.613\text{ ms}$** | Corrected off-by-one denominator: divided by 63 decode steps instead of 64. |
| **Concurrency = 64** | Peak KV Cache | $6144.0\text{ MB}$ | **$5944.0\text{ MiB}$** | Standardized binary memory notation (Mebibytes, $1024^2$ bytes). |
| | Scheduling | Serial block | **Interleaved** | Earlier requests begin decoding immediately between admissions. |
