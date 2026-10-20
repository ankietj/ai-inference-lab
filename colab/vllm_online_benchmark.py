"""Open-Loop Online Serving Benchmark Harness for vLLM (OpenAI-Compatible Streaming API).

RESEARCH OBJECTIVE:
-------------------
Evaluate the maximum sustainable request arrival rate (lambda) that satisfies predefined
Service Level Objectives (SLOs) for Time-to-First-Token (TTFT) and End-to-End latency (E2E),
under continuous batching and dynamic arrival conditions on physical GPU hardware.

METHODOLOGY & METRICS BOUNDARIES:
---------------------------------
1. Open-Loop vs Closed-Loop:
   - Uses an open-loop Poisson arrival process where scheduled dispatch times are determined
     independently of server turnaround latency: Delta t ~ Exponential(lambda).
   - Distinct from closed-loop concurrency (virtual user pool) and offline batch generation.
2. Timing Boundaries (Nanosecond Precision):
   - All relative intervals are measured using `time.perf_counter_ns()`.
   - All absolute timestamps record ISO 8601 UTC with microsecond resolution (`%Y-%m-%dT%H:%M:%S.%fZ`).
   - TTFT: Dispatch timestamp to arrival of the first content chunk.
   - E2E: Dispatch timestamp to completion of the final chunk/stream termination.
   - Inter-chunk intervals: Recorded as chunk-to-chunk arrival deltas. Clearly labeled as
     inter-chunk intervals (not guaranteed 1:1 token-level ITL unless verified).
   - Dispatch delay: Monotonic difference between actual dispatch time and scheduled target time.
3. Drain & Incomplete Stream Accounting:
   - When the scheduled arrival window ends, arrivals halt and a configurable drain window begins.
   - Streams terminating without [DONE] or a valid finish_reason are explicitly marked "incomplete".
   - Incomplete streams and timed-out requests are explicitly tracked and logged; no silent dropping.
4. Deterministic Statistics & Provisional SLO:
   - Sample standard deviation (ddof=1), NIST linear percentiles, IQR, min, max, mean, median.
   - Provisional SLO: TTFT p95 <= 150 ms, E2E p95 <= 3000 ms, zero observed errors.
   - Requires at least 20 completed requests to evaluate SLO; otherwise reports INCONCLUSIVE.
   - Disqualifies any stage where the circuit breaker tripped (reports FAIL).
   - p95 and p99 are marked as exploratory when sample sizes are small (N < 1000).
5. Output Token Accounting:
   - Extracts usage.completion_tokens from streaming response when provided by the server.
   - If usage metadata is unavailable, token counts are represented as null (None), not word counts.
   - Token throughput is calculated only from valid, observed token counts; otherwise null.
6. Environment & Colab Colocation Disclaimer:
   - Client and server run on the same virtual instance (localhost loopback).
   - Network latency is minimal (<0.5 ms); results reflect server engine scheduling and queueing,
     NOT internet-facing public transit latency or distributed production microservice routing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# Conditional import of httpx for live execution in Google Colab
try:
    import httpx
except ImportError:
    httpx = None


# ---------------------------------------------------------------------------
# Data Models & Configurations
# ---------------------------------------------------------------------------

@dataclass
class SLOConfig:
    """Provisional Service Level Objective thresholds."""
    max_ttft_p95_ms: float = 150.0
    max_e2e_p95_ms: float = 3000.0
    max_error_rate: float = 0.0  # Zero observed errors in the finite test window
    min_completed_requests: int = 20  # Minimum sample size required to certify an SLO pass


@dataclass
class RequestRecord:
    """Detailed telemetry record for an individual open-loop request."""
    request_id: int
    scheduled_offset_s: float
    scheduled_timestamp_utc: str
    actual_dispatch_timestamp_utc: str
    dispatch_delay_ms: float
    status: str  # "completed", "failed", "timed_out", "incomplete"
    http_status_code: int | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    actual_chunk_count: int = 0
    actual_output_tokens: int | None = None  # None if exact usage metadata unavailable
    mean_inter_chunk_interval_ms: float | None = None
    inter_chunk_intervals_ms: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Algorithmic Scheduling & Statistical Utilities
# ---------------------------------------------------------------------------

def schedule_poisson_arrivals(
    rate: float,
    duration_s: float,
    seed: int = 42,
) -> list[float]:
    """Generate deterministic scheduled arrival offsets (in seconds from t=0).
    
    Inter-arrival intervals: Delta t ~ Exponential(rate).
    Uses standard library random.Random(seed).expovariate(rate).
    """
    if rate <= 0:
        raise ValueError(f"Arrival rate must be positive, got {rate}")
    if duration_s <= 0:
        raise ValueError(f"Duration must be positive, got {duration_s}")

    rng = random.Random(seed)
    arrivals: list[float] = []
    current_time = 0.0

    while True:
        inter_arrival = rng.expovariate(rate)
        current_time += inter_arrival
        if current_time > duration_s:
            break
        arrivals.append(round(current_time, 9))

    return arrivals


def parse_sse_chunk(line: str) -> dict[str, Any] | None:
    """Parse a single Server-Sent Events (SSE) data line."""
    cleaned = line.strip()
    if not cleaned.startswith("data:"):
        return None
    data_str = cleaned[5:].strip()
    if data_str == "[DONE]":
        return {"done": True}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def compute_distribution_statistics(
    values: list[float],
    metric_name: str = "metric",
) -> dict[str, Any]:
    """Compute deterministic summary statistics from raw trial measurements.

    Statistical Conventions:
    - Sample standard deviation uses N - 1 degrees of freedom (Bessel's correction).
    - Percentiles use linear interpolation between order statistics (rank = p/100 * (N - 1)).
    - Tail estimates (p90, p95, p99) are labeled as exploratory for small sample sizes.
    """
    n = len(values)
    if n == 0:
        return {
            "metric_name": metric_name,
            "sample_count": 0,
            "mean": None,
            "median": None,
            "std_dev_sample": None,
            "min": None,
            "max": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90_exploratory": None,
            "p95_exploratory": None,
            "p99_exploratory": None,
            "iqr": None,
        }

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
                "p90, p95, and p99 are exploratory when N < 1000; robust tail-latency estimates "
                "require large sample populations under steady-state traffic."
            ),
        },
    }


def evaluate_slo(
    stage_summary: dict[str, Any],
    slo_config: SLOConfig,
) -> dict[str, Any]:
    """Evaluate whether an arrival rate stage meets the configured SLO criteria.
    
    Returns a formal status: 'PASS', 'FAIL', or 'INCONCLUSIVE'.
    """
    circuit_breaker_tripped = stage_summary.get("circuit_breaker_tripped", False)
    dispatched = stage_summary.get("dispatched_requests", 0)
    completed = stage_summary.get("completed_requests", 0)
    failed = stage_summary.get("failed_requests", 0)
    error_rate = stage_summary.get("error_rate", 1.0)

    ttft_stats = stage_summary.get("ttft_ms", {})
    e2e_stats = stage_summary.get("e2e_ms", {})
    ttft_p95 = ttft_stats.get("p95_exploratory")
    e2e_p95 = e2e_stats.get("p95_exploratory")

    min_reqs = slo_config.min_completed_requests

    # Disqualification 1: Circuit breaker tripped
    if circuit_breaker_tripped:
        return {
            "slo_status": "FAIL",
            "meets_slo": False,
            "reason": f"Disqualified: Circuit breaker tripped ({stage_summary.get('circuit_breaker_reason')}).",
            "ttft_p95_ms": ttft_p95,
            "e2e_p95_ms": e2e_p95,
            "error_rate": error_rate,
            "completed_requests": completed,
            "min_completed_required": min_reqs,
        }

    # Inconclusive 1: Insufficient sample size (< min_completed_requests)
    if completed < min_reqs or dispatched == 0:
        return {
            "slo_status": "INCONCLUSIVE",
            "meets_slo": False,
            "reason": f"Inconclusive: Insufficient completed requests ({completed} completed < {min_reqs} required).",
            "ttft_p95_ms": ttft_p95,
            "e2e_p95_ms": e2e_p95,
            "error_rate": error_rate,
            "completed_requests": completed,
            "min_completed_required": min_reqs,
        }

    # Inconclusive 2: Missing latency percentiles
    if ttft_p95 is None or e2e_p95 is None:
        return {
            "slo_status": "INCONCLUSIVE",
            "meets_slo": False,
            "reason": "Inconclusive: Latency percentiles could not be computed.",
            "ttft_p95_ms": ttft_p95,
            "e2e_p95_ms": e2e_p95,
            "error_rate": error_rate,
            "completed_requests": completed,
            "min_completed_required": min_reqs,
        }

    # Strict SLO evaluation
    ttft_pass = ttft_p95 <= slo_config.max_ttft_p95_ms
    e2e_pass = e2e_p95 <= slo_config.max_e2e_p95_ms
    error_pass = (error_rate <= slo_config.max_error_rate) and (failed == 0) and (completed == dispatched)

    if ttft_pass and e2e_pass and error_pass:
        status = "PASS"
        meets_slo = True
        reason = "All SLO criteria satisfied in finite test window."
    else:
        status = "FAIL"
        meets_slo = False
        reasons = []
        if not ttft_pass:
            reasons.append(f"TTFT p95 ({ttft_p95:.1f}ms > {slo_config.max_ttft_p95_ms}ms)")
        if not e2e_pass:
            reasons.append(f"E2E p95 ({e2e_p95:.1f}ms > {slo_config.max_e2e_p95_ms}ms)")
        if not error_pass:
            reasons.append(f"Errors observed ({failed} failed/incomplete requests, error rate {error_rate:.1%})")
        reason = "SLO violation: " + "; ".join(reasons)

    return {
        "slo_status": status,
        "meets_slo": meets_slo,
        "reason": reason,
        "ttft_p95_ms": ttft_p95,
        "ttft_p95_threshold_ms": slo_config.max_ttft_p95_ms,
        "ttft_pass": ttft_pass,
        "e2e_p95_ms": e2e_p95,
        "e2e_p95_threshold_ms": slo_config.max_e2e_p95_ms,
        "e2e_pass": e2e_pass,
        "error_rate": error_rate,
        "error_rate_threshold": slo_config.max_error_rate,
        "error_pass": error_pass,
        "completed_requests": completed,
        "min_completed_required": min_reqs,
        "notes": (
            "Zero observed errors represents the outcome of the finite test window. "
            "Tail percentiles (p95, p99) are exploratory approximations for finite samples (N < 1000)."
        ),
    }


def compute_stage_summary(
    records: list[RequestRecord],
    scheduled_rate: float,
    window_duration_s: float,
    total_elapsed_wall_clock_s: float,
    slo_config: SLOConfig,
) -> dict[str, Any]:
    """Aggregate per-request records into formal stage metrics and SLO evaluation."""
    dispatched = len(records)
    completed_records = [r for r in records if r.status == "completed"]
    failed_records = [r for r in records if r.status in ("failed", "timed_out", "incomplete")]

    completed_count = len(completed_records)
    failed_count = len(failed_records)
    error_rate = (failed_count / dispatched) if dispatched > 0 else 0.0

    # Collect successful latencies and dispatch delays
    ttft_list = [r.ttft_ms for r in completed_records if r.ttft_ms is not None]
    e2e_list = [r.e2e_ms for r in completed_records if r.e2e_ms is not None]
    dispatch_delay_list = [r.dispatch_delay_ms for r in records]
    all_inter_chunks = [ic for r in completed_records for ic in r.inter_chunk_intervals_ms]

    # Token counting: only if exact counts are available across all completed requests
    observed_tokens = [r.actual_output_tokens for r in completed_records if r.actual_output_tokens is not None]
    all_tokens_observed = (len(observed_tokens) == completed_count) and (completed_count > 0)
    total_tokens = sum(observed_tokens) if all_tokens_observed else None

    # Distinct throughput calculations:
    # 1. Scheduled arrival rate (offered load)
    # 2. Actual dispatch rate (dispatched / arrival window)
    # 3. Completed request throughput relative to arrival window
    # 4. Physical service throughput over total elapsed duration (including drain)
    offered_window_req_rate = round(dispatched / window_duration_s, 2) if window_duration_s > 0 else 0.0
    completed_window_req_rate = round(completed_count / window_duration_s, 2) if window_duration_s > 0 else 0.0
    physical_service_req_throughput = round(completed_count / total_elapsed_wall_clock_s, 2) if total_elapsed_wall_clock_s > 0 else 0.0

    if total_tokens is not None and total_elapsed_wall_clock_s > 0:
        physical_token_throughput = round(total_tokens / total_elapsed_wall_clock_s, 2)
    else:
        physical_token_throughput = None

    summary = {
        "scheduled_rate_req_s": scheduled_rate,
        "window_duration_s": window_duration_s,
        "total_elapsed_wall_clock_s": round(total_elapsed_wall_clock_s, 3),
        "drain_duration_s": round(max(0.0, total_elapsed_wall_clock_s - window_duration_s), 3),
        "dispatched_requests": dispatched,
        "completed_requests": completed_count,
        "failed_requests": failed_count,
        "error_rate": round(error_rate, 4),
        "scheduled_arrival_rate_req_s": scheduled_rate,
        "actual_dispatch_rate_req_s": offered_window_req_rate,
        "completed_request_rate_per_window_s": completed_window_req_rate,
        "physical_service_throughput_req_s": physical_service_req_throughput,
        "completed_token_throughput_tokens_s": physical_token_throughput,
        "total_generated_tokens": total_tokens,
        "token_accounting_note": (
            "Exact token count recorded from usage metadata when available; otherwise null. "
            "Word splitting is never used as a substitute for BPE tokens."
        ),
        "dispatch_delay_ms": compute_distribution_statistics(dispatch_delay_list, metric_name="dispatch_delay_ms"),
        "ttft_ms": compute_distribution_statistics(ttft_list, metric_name="ttft_ms"),
        "e2e_ms": compute_distribution_statistics(e2e_list, metric_name="e2e_ms"),
        "inter_chunk_intervals_ms": compute_distribution_statistics(all_inter_chunks, metric_name="inter_chunk_intervals_ms"),
    }

    slo_result = evaluate_slo(summary, slo_config)
    summary["slo_evaluation"] = slo_result
    return summary


# ---------------------------------------------------------------------------
# Asynchronous Load Execution
# ---------------------------------------------------------------------------

async def execute_streaming_request(
    client: Any,
    endpoint_url: str,
    payload: dict[str, Any],
    request_id: int,
    scheduled_offset_s: float,
    scheduled_target_ns: int,
    timeout_s: float = 60.0,
) -> RequestRecord:
    """Dispatch a single HTTP streaming request and record high-precision timing."""
    now_utc = datetime.now(timezone.utc).isoformat()
    t_disp_ns = time.perf_counter_ns()
    dispatch_delay_ms = max(0.0, (t_disp_ns - scheduled_target_ns) / 1_000_000.0)

    record = RequestRecord(
        request_id=request_id,
        scheduled_offset_s=scheduled_offset_s,
        scheduled_timestamp_utc=now_utc,
        actual_dispatch_timestamp_utc=now_utc,
        dispatch_delay_ms=round(dispatch_delay_ms, 3),
        status="initiated",
    )

    chunk_timestamps_ns: list[int] = []
    generated_text = ""
    finish_reason: str | None = None
    completion_tokens: int | None = None
    received_done_event = False

    try:
        async with client.stream("POST", endpoint_url, json=payload, timeout=timeout_s) as response:
            record.http_status_code = response.status_code

            if response.status_code != 200:
                record.status = "failed"
                record.error_message = f"HTTP {response.status_code}"
                return record

            async for line in response.aiter_lines():
                if not line:
                    continue
                t_chunk_ns = time.perf_counter_ns()
                parsed = parse_sse_chunk(line)
                if parsed is None:
                    continue
                if parsed.get("done"):
                    received_done_event = True
                    break

                # Extract usage metadata if present in stream chunk
                usage = parsed.get("usage")
                if usage and isinstance(usage, dict):
                    if "completion_tokens" in usage and usage["completion_tokens"] is not None:
                        completion_tokens = int(usage["completion_tokens"])

                choices = parsed.get("choices", [])
                if choices:
                    choice = choices[0]
                    text_delta = choice.get("text", "")
                    if text_delta:
                        chunk_timestamps_ns.append(t_chunk_ns)
                        generated_text += text_delta
                    if choice.get("finish_reason"):
                        finish_reason = choice.get("finish_reason")

            t_final_ns = time.perf_counter_ns()

            # Clean completion requires: content received AND (clean [DONE] event OR explicit finish_reason)
            is_cleanly_terminated = received_done_event or (finish_reason is not None)

            if is_cleanly_terminated and chunk_timestamps_ns:
                record.ttft_ms = (chunk_timestamps_ns[0] - t_disp_ns) / 1_000_000.0
                record.e2e_ms = (t_final_ns - t_disp_ns) / 1_000_000.0
                record.actual_chunk_count = len(chunk_timestamps_ns)
                record.actual_output_tokens = completion_tokens
                record.finish_reason = finish_reason
                record.status = "completed"

                if len(chunk_timestamps_ns) > 1:
                    intervals = [
                        (chunk_timestamps_ns[k] - chunk_timestamps_ns[k - 1]) / 1_000_000.0
                        for k in range(1, len(chunk_timestamps_ns))
                    ]
                    record.inter_chunk_intervals_ms = intervals
                    record.mean_inter_chunk_interval_ms = round(sum(intervals) / len(intervals), 3)
            else:
                record.status = "incomplete"
                record.actual_chunk_count = len(chunk_timestamps_ns)
                record.actual_output_tokens = completion_tokens
                record.finish_reason = finish_reason
                if not chunk_timestamps_ns:
                    record.error_message = "Stream ended with zero content chunks"
                else:
                    record.error_message = "Stream terminated abruptly without [DONE] or finish_reason"

    except asyncio.TimeoutError:
        record.status = "timed_out"
        record.error_message = f"Client request exceeded {timeout_s}s timeout"
    except asyncio.CancelledError:
        record.status = "incomplete"
        record.error_message = "Request cancelled before completion (drain timeout or circuit breaker)."
        return record
    except Exception as exc:
        record.status = "failed"
        record.error_message = str(exc)

    return record


async def run_stage_with_circuit_breaker(
    endpoint_url: str,
    model_id: str,
    prompt_text: str,
    rate: float,
    duration_s: float,
    drain_timeout_s: float = 30.0,
    seed: int = 42,
    max_consecutive_timeouts: int = 3,
    max_error_rate: float = 0.05,
    max_connections: int = 1000,
    max_keepalive_connections: int = 200,
) -> tuple[list[RequestRecord], bool, str | None, float]:
    """Execute a single load stage with scheduled Poisson arrivals and circuit breaker safety."""
    if httpx is None:
        raise ImportError("httpx is required. Install via: pip install httpx")

    arrival_offsets = schedule_poisson_arrivals(rate=rate, duration_s=duration_s, seed=seed)
    payload_base = {
        "model": model_id,
        "prompt": prompt_text,
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    records: list[RequestRecord] = []
    circuit_breaker_tripped = False
    trip_reason = None

    t0_perf_ns = time.perf_counter_ns()
    tasks: list[asyncio.Task[RequestRecord]] = []

    limits = httpx.Limits(
        max_keepalive_connections=max_keepalive_connections,
        max_connections=max_connections,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        for idx, offset in enumerate(arrival_offsets, start=1):
            target_ns = t0_perf_ns + int(offset * 1_000_000_000)
            now_ns = time.perf_counter_ns()
            sleep_ns = target_ns - now_ns

            if sleep_ns > 0:
                await asyncio.sleep(sleep_ns / 1_000_000_000.0)

            resolved_tasks = [t for t in tasks if t.done()]

            # Circuit breaker check 1: evaluate consecutive timeouts among resolved tasks
            recent_timeouts = 0
            for t in reversed(resolved_tasks):
                if t.result().status == "timed_out":
                    recent_timeouts += 1
                else:
                    break
            if recent_timeouts >= max_consecutive_timeouts:
                circuit_breaker_tripped = True
                trip_reason = f"Exceeded {max_consecutive_timeouts} consecutive request timeouts."
                print(f"CIRCUIT BREAKER TRIPPED: {trip_reason}")
                break

            # Circuit breaker check 2: evaluate running error rate across resolved tasks
            if len(resolved_tasks) >= 10:
                failed_so_far = sum(
                    1 for t in resolved_tasks
                    if t.result().status in ("failed", "timed_out", "incomplete")
                )
                running_err_rate = failed_so_far / len(resolved_tasks)
                if running_err_rate > max_error_rate:
                    circuit_breaker_tripped = True
                    trip_reason = (
                        f"Running error rate ({running_err_rate:.1%}) exceeded threshold ({max_error_rate:.1%}) "
                        f"across {len(resolved_tasks)} resolved requests."
                    )
                    print(f"CIRCUIT BREAKER TRIPPED: {trip_reason}")
                    break

            task = asyncio.create_task(
                execute_streaming_request(
                    client=client,
                    endpoint_url=endpoint_url,
                    payload=payload_base,
                    request_id=idx,
                    scheduled_offset_s=offset,
                    scheduled_target_ns=target_ns,
                    timeout_s=30.0,
                )
            )
            tasks.append(task)

        # Allow drain window for active in-flight requests (or immediately cancel if tripped)
        drain_wait = 0.0 if circuit_breaker_tripped else drain_timeout_s
        if not circuit_breaker_tripped:
            print(f"Arrivals halted for rate {rate} req/s. Entering drain window (up to {drain_timeout_s}s)...")
        else:
            print("Circuit breaker tripped: canceling remaining in-flight requests...")

        done, pending = await asyncio.wait(tasks, timeout=drain_wait)

        for t in done:
            records.append(t.result())

        if pending:
            for t in pending:
                t.cancel()
            cancelled_results = await asyncio.gather(*pending, return_exceptions=True)
            for res in cancelled_results:
                if isinstance(res, RequestRecord):
                    records.append(res)
                elif isinstance(res, Exception):
                    records.append(
                        RequestRecord(
                            request_id=len(records) + 1,
                            scheduled_offset_s=0.0,
                            scheduled_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            actual_dispatch_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            dispatch_delay_ms=0.0,
                            status="incomplete",
                            error_message=f"Abandoned in-flight request: {res}",
                        )
                    )

        records.sort(key=lambda r: r.request_id)

    t_end_perf_ns = time.perf_counter_ns()
    total_elapsed_wall_clock_s = (t_end_perf_ns - t0_perf_ns) / 1_000_000_000.0

    return records, circuit_breaker_tripped, trip_reason, total_elapsed_wall_clock_s


# ---------------------------------------------------------------------------
# Pre-Flight Health Check & Orchestration
# ---------------------------------------------------------------------------

async def wait_for_server_health(health_url: str, max_wait_s: int = 180) -> bool:
    """Poll vLLM server health endpoint until ready or timeout."""
    if httpx is None:
        raise ImportError("httpx is required. Install via: pip install httpx")

    print(f"Probing server health at {health_url} (timeout: {max_wait_s}s)...")
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < max_wait_s:
            try:
                resp = await client.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    print("Server is healthy and ready to accept requests.")
                    return True
            except Exception:
                pass
            await asyncio.sleep(3.0)

    print("ERROR: Server failed health check within timeout.")
    return False


async def run_online_serving_benchmark(
    base_url: str = "http://localhost:8000",
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    rates: list[float] | None = None,
    pilot_duration_s: float = 60.0,
    confirmation_duration_s: float = 120.0,
    output_path: str = "colab_vllm_online_results.json",
    seed: int = 42,
    max_error_rate: float = 0.05,
) -> dict[str, Any]:
    """Execute complete Phase 3 staged online serving benchmark."""
    if rates is None:
        rates = [0.5, 1.0, 2.5, 4.5, 6.0]

    health_url = f"{base_url}/health"
    completions_url = f"{base_url}/v1/completions"

    if not await wait_for_server_health(health_url):
        raise SystemError("Aborting benchmark: vLLM server is unreachable.")

    slo_config = SLOConfig(max_error_rate=0.0, min_completed_requests=20)
    prompt_text = (
        "Analyze the system performance and memory architecture of high-performance LLM serving systems, "
        "focusing on the interaction between continuous batching, PagedAttention block paging, "
        "and Tensor Core roofline execution. "
    ) * 4  # Approximately 512 characters/tokens

    results: dict[str, Any] = {
        "metadata": {
            "benchmark_version": "phase3_online_serving_v2",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_model": model_id,
            "server_endpoint": completions_url,
            "pilot_duration_s": pilot_duration_s,
            "confirmation_duration_s": confirmation_duration_s,
            "seed": seed,
            "circuit_breaker_max_error_rate": max_error_rate,
            "slo_thresholds": asdict(slo_config),
            "disclaimer": (
                "Benchmark measures client-side HTTP streaming latency on a collocated instance. "
                "Network latency is minimal loopback; metrics reflect server scheduling and queueing, "
                "not public internet transit latency."
            ),
        },
        "pilot_stages": [],
        "confirmation_stages": [],
        "highest_sustainable_rate_req_s": None,
        "overall_status": "PENDING",
    }

    # 1. Warm-up stage (10 requests at 1.0 req/s)
    print("\n" + "=" * 75)
    print("STEP 1: Server Warm-up (10 requests at 1.0 req/s)")
    print("=" * 75)
    _, _, _, _ = await run_stage_with_circuit_breaker(
        endpoint_url=completions_url,
        model_id=model_id,
        prompt_text=prompt_text,
        rate=1.0,
        duration_s=10.0,
        drain_timeout_s=15.0,
        seed=seed,
        max_error_rate=max_error_rate,
    )
    print("Warm-up complete.")

    # 2. Pilot Stage Sweep
    print("\n" + "=" * 75)
    print("STEP 2: Exploratory Pilot Sweeps (60s windows)")
    print("=" * 75)

    promising_rates: list[float] = []

    for rate in rates:
        print(f"\n--- Testing Pilot Arrival Rate: {rate} req/s ---")
        records, tripped, reason, total_elapsed_s = await run_stage_with_circuit_breaker(
            endpoint_url=completions_url,
            model_id=model_id,
            prompt_text=prompt_text,
            rate=rate,
            duration_s=pilot_duration_s,
            drain_timeout_s=30.0,
            seed=seed,
            max_error_rate=max_error_rate,
        )

        stage_summary = compute_stage_summary(
            records=records,
            scheduled_rate=rate,
            window_duration_s=pilot_duration_s,
            total_elapsed_wall_clock_s=total_elapsed_s,
            slo_config=slo_config,
        )
        stage_summary["circuit_breaker_tripped"] = tripped
        stage_summary["circuit_breaker_reason"] = reason
        stage_summary["raw_records"] = [asdict(r) for r in records]

        results["pilot_stages"].append(stage_summary)

        slo_status = stage_summary["slo_evaluation"]["slo_status"]
        ttft_p95 = stage_summary["ttft_ms"].get("p95_exploratory")
        e2e_p95 = stage_summary["e2e_ms"].get("p95_exploratory")
        err = stage_summary["error_rate"]

        print(
            f"Rate {rate:4.1f} req/s -> Completed: {stage_summary['completed_requests']}/{stage_summary['dispatched_requests']} | "
            f"TTFT p95: {ttft_p95} ms | E2E p95: {e2e_p95} ms | Err: {err:.1%} | "
            f"SLO: {slo_status}"
        )

        if slo_status == "PASS":
            promising_rates.append(rate)
        if tripped:
            print("Halting further pilot rates due to circuit breaker trip.")
            break

    # 3. Confirmation Stage for highest promising rate
    if promising_rates:
        best_candidate = max(promising_rates)
        print("\n" + "=" * 75)
        print(f"STEP 3: Confirmation Window for Highest Promising Rate: {best_candidate} req/s ({confirmation_duration_s}s)")
        print("=" * 75)

        conf_records, conf_tripped, conf_reason, conf_elapsed_s = await run_stage_with_circuit_breaker(
            endpoint_url=completions_url,
            model_id=model_id,
            prompt_text=prompt_text,
            rate=best_candidate,
            duration_s=confirmation_duration_s,
            drain_timeout_s=45.0,
            seed=seed + 100,  # Different seed for confirmation
            max_error_rate=max_error_rate,
        )

        conf_summary = compute_stage_summary(
            records=conf_records,
            scheduled_rate=best_candidate,
            window_duration_s=confirmation_duration_s,
            total_elapsed_wall_clock_s=conf_elapsed_s,
            slo_config=slo_config,
        )
        conf_summary["circuit_breaker_tripped"] = conf_tripped
        conf_summary["circuit_breaker_reason"] = conf_reason
        conf_summary["raw_records"] = [asdict(r) for r in conf_records]

        results["confirmation_stages"].append(conf_summary)

        conf_status = conf_summary["slo_evaluation"]["slo_status"]
        if conf_status == "PASS":
            results["highest_sustainable_rate_req_s"] = best_candidate
            results["overall_status"] = "CONFIRMED_PASS"
            print(f"CONFIRMED: Highest sustainable rate meeting SLO is {best_candidate} req/s.")
        else:
            results["overall_status"] = "CONFIRMATION_FAILED"
            print(f"Confirmation failed at {best_candidate} req/s under extended window (status: {conf_status}).")
    else:
        results["overall_status"] = "NO_QUALIFYING_RATES"
        print("No tested arrival rates met the provisional SLO criteria.")

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nPhase 3 Online Serving Benchmark complete. Results written to: {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 open-loop online serving benchmark for vLLM.")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Target model ID")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.5, 1.0, 2.5, 4.5, 6.0], help="Exploratory arrival rates (req/s)")
    parser.add_argument("--pilot-duration", type=float, default=60.0, help="Duration of each pilot stage (seconds)")
    parser.add_argument("--confirmation-duration", type=float, default=120.0, help="Duration of confirmation stage (seconds)")
    parser.add_argument("--output", type=str, default="colab_vllm_online_results.json", help="Path to write JSON results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Poisson arrival schedule")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Circuit breaker running error rate threshold (default: 0.05)")
    args = parser.parse_args()

    asyncio.run(
        run_online_serving_benchmark(
            base_url=args.base_url,
            model_id=args.model,
            rates=args.rates,
            pilot_duration_s=args.pilot_duration,
            confirmation_duration_s=args.confirmation_duration,
            output_path=args.output,
            seed=args.seed,
            max_error_rate=args.max_error_rate,
        )
    )


if __name__ == "__main__":
    main()
