"""Unit tests for Phase 3 online serving benchmark harness (non-GPU synthetic tests)."""

import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path("colab").resolve()))
from vllm_online_benchmark import (
    RequestRecord,
    SLOConfig,
    compute_distribution_statistics,
    compute_stage_summary,
    evaluate_slo,
    parse_sse_chunk,
    run_stage_with_circuit_breaker,
    schedule_poisson_arrivals,
)


def test_schedule_poisson_arrivals_reproducibility():
    """Verify that Poisson arrival scheduling is strictly reproducible with fixed seed."""
    schedule1 = schedule_poisson_arrivals(rate=2.0, duration_s=10.0, seed=42)
    schedule2 = schedule_poisson_arrivals(rate=2.0, duration_s=10.0, seed=42)
    schedule3 = schedule_poisson_arrivals(rate=2.0, duration_s=10.0, seed=99)

    assert schedule1 == schedule2
    assert schedule1 != schedule3
    assert len(schedule1) > 0
    # Strictly monotonically increasing
    for i in range(1, len(schedule1)):
        assert schedule1[i] > schedule1[i - 1]
    # Bound by duration
    assert all(t <= 10.0 for t in schedule1)


def test_schedule_poisson_arrivals_invalid_inputs():
    """Verify that non-positive arrival rates or durations raise ValueError."""
    with pytest.raises(ValueError, match="Arrival rate must be positive"):
        schedule_poisson_arrivals(rate=0.0, duration_s=10.0)

    with pytest.raises(ValueError, match="Duration must be positive"):
        schedule_poisson_arrivals(rate=2.0, duration_s=-5.0)


def test_parse_sse_chunk_standard_and_usage():
    """Verify SSE line parsing for standard payloads, [DONE] signal, and usage metadata."""
    # Standard chunk
    line_normal = 'data: {"id":"123","choices":[{"text":" hello","finish_reason":null}]}'
    parsed = parse_sse_chunk(line_normal)
    assert parsed is not None
    assert parsed["id"] == "123"
    assert parsed["choices"][0]["text"] == " hello"

    # Chunk with usage metadata
    line_usage = 'data: {"id":"123","choices":[],"usage":{"completion_tokens":128,"prompt_tokens":512,"total_tokens":640}}'
    parsed_usage = parse_sse_chunk(line_usage)
    assert parsed_usage is not None
    assert parsed_usage["usage"]["completion_tokens"] == 128

    # [DONE] line
    line_done = "data: [DONE]"
    assert parse_sse_chunk(line_done) == {"done": True}

    # Empty or comment lines
    assert parse_sse_chunk("") is None
    assert parse_sse_chunk(": ping") is None

    # Malformed JSON
    assert parse_sse_chunk("data: {invalid json}") is None


def test_compute_distribution_statistics_empty():
    """Verify that empty metric lists return safely without throwing exceptions."""
    stats = compute_distribution_statistics([], metric_name="test_empty")
    assert stats["sample_count"] == 0
    assert stats["mean"] is None
    assert stats["median"] is None


def test_evaluate_slo_pass():
    """Verify that a stage meeting all criteria passes the SLO with status PASS."""
    slo_config = SLOConfig(max_ttft_p95_ms=150.0, max_e2e_p95_ms=3000.0, max_error_rate=0.0, min_completed_requests=20)

    summary_pass = {
        "dispatched_requests": 30,
        "completed_requests": 30,
        "failed_requests": 0,
        "error_rate": 0.0,
        "circuit_breaker_tripped": False,
        "ttft_ms": {"p95_exploratory": 95.0},
        "e2e_ms": {"p95_exploratory": 2100.0},
    }

    result = evaluate_slo(summary_pass, slo_config)
    assert result["slo_status"] == "PASS"
    assert result["meets_slo"] is True
    assert result["ttft_pass"] is True
    assert result["e2e_pass"] is True
    assert result["error_pass"] is True


def test_evaluate_slo_circuit_breaker_disqualification():
    """Verify that circuit breaker activation immediately disqualifies a stage with status FAIL."""
    slo_config = SLOConfig()

    summary_tripped = {
        "dispatched_requests": 15,
        "completed_requests": 15,
        "failed_requests": 0,
        "error_rate": 0.0,
        "circuit_breaker_tripped": True,
        "circuit_breaker_reason": "Running error rate exceeded threshold",
        "ttft_ms": {"p95_exploratory": 50.0},
        "e2e_ms": {"p95_exploratory": 1500.0},
    }

    result = evaluate_slo(summary_tripped, slo_config)
    assert result["slo_status"] == "FAIL"
    assert result["meets_slo"] is False
    assert "Circuit breaker tripped" in result["reason"]


def test_evaluate_slo_inconclusive_sample_size():
    """Verify that stages with fewer than min_completed_requests return INCONCLUSIVE."""
    slo_config = SLOConfig(min_completed_requests=20)

    # Only 10 completed requests
    summary_small = {
        "dispatched_requests": 10,
        "completed_requests": 10,
        "failed_requests": 0,
        "error_rate": 0.0,
        "circuit_breaker_tripped": False,
        "ttft_ms": {"p95_exploratory": 80.0},
        "e2e_ms": {"p95_exploratory": 1900.0},
    }

    result = evaluate_slo(summary_small, slo_config)
    assert result["slo_status"] == "INCONCLUSIVE"
    assert result["meets_slo"] is False
    assert "Insufficient completed requests" in result["reason"]


def test_evaluate_slo_inconclusive_missing_latencies():
    """Verify that missing latency percentiles result in INCONCLUSIVE status."""
    slo_config = SLOConfig(min_completed_requests=20)

    summary_missing_lat = {
        "dispatched_requests": 25,
        "completed_requests": 25,
        "failed_requests": 0,
        "error_rate": 0.0,
        "circuit_breaker_tripped": False,
        "ttft_ms": {"p95_exploratory": None},
        "e2e_ms": {"p95_exploratory": 1900.0},
    }

    result = evaluate_slo(summary_missing_lat, slo_config)
    assert result["slo_status"] == "INCONCLUSIVE"
    assert result["meets_slo"] is False


def test_evaluate_slo_metric_failures():
    """Verify that SLO correctly identifies TTFT violations, E2E violations, and errors."""
    slo_config = SLOConfig(max_ttft_p95_ms=150.0, max_e2e_p95_ms=3000.0, max_error_rate=0.0, min_completed_requests=20)

    # TTFT violation
    summary_ttft_fail = {
        "dispatched_requests": 50,
        "completed_requests": 50,
        "failed_requests": 0,
        "error_rate": 0.0,
        "circuit_breaker_tripped": False,
        "ttft_ms": {"p95_exploratory": 185.0},
        "e2e_ms": {"p95_exploratory": 2100.0},
    }
    res1 = evaluate_slo(summary_ttft_fail, slo_config)
    assert res1["slo_status"] == "FAIL"
    assert res1["ttft_pass"] is False
    assert res1["e2e_pass"] is True

    # E2E violation
    summary_e2e_fail = {
        "dispatched_requests": 50,
        "completed_requests": 50,
        "failed_requests": 0,
        "error_rate": 0.0,
        "circuit_breaker_tripped": False,
        "ttft_ms": {"p95_exploratory": 105.0},
        "e2e_ms": {"p95_exploratory": 3500.0},
    }
    res2 = evaluate_slo(summary_e2e_fail, slo_config)
    assert res2["slo_status"] == "FAIL"
    assert res2["e2e_pass"] is False

    # Error rate violation
    summary_error_fail = {
        "dispatched_requests": 50,
        "completed_requests": 48,
        "failed_requests": 2,
        "error_rate": 0.04,
        "circuit_breaker_tripped": False,
        "ttft_ms": {"p95_exploratory": 90.0},
        "e2e_ms": {"p95_exploratory": 1800.0},
    }
    res3 = evaluate_slo(summary_error_fail, slo_config)
    assert res3["slo_status"] == "FAIL"
    assert res3["error_pass"] is False


def test_token_throughput_with_valid_and_missing_counts():
    """Verify that token throughput is computed only when exact token counts are observed."""
    # Case A: Valid token counts across all requests
    records_with_tokens = [
        RequestRecord(
            request_id=1,
            scheduled_offset_s=0.5,
            scheduled_timestamp_utc="2026-10-20T12:00:00.000000Z",
            actual_dispatch_timestamp_utc="2026-10-20T12:00:00.000000Z",
            dispatch_delay_ms=2.5,
            status="completed",
            http_status_code=200,
            ttft_ms=50.0,
            e2e_ms=1000.0,
            actual_output_tokens=128,
            actual_chunk_count=128,
        ),
        RequestRecord(
            request_id=2,
            scheduled_offset_s=1.0,
            scheduled_timestamp_utc="2026-10-20T12:00:01.000000Z",
            actual_dispatch_timestamp_utc="2026-10-20T12:00:01.000000Z",
            dispatch_delay_ms=3.1,
            status="completed",
            http_status_code=200,
            ttft_ms=60.0,
            e2e_ms=1000.0,
            actual_output_tokens=128,
            actual_chunk_count=128,
        ),
    ]

    summary_a = compute_stage_summary(
        records=records_with_tokens,
        scheduled_rate=2.0,
        window_duration_s=10.0,
        total_elapsed_wall_clock_s=12.0,
        slo_config=SLOConfig(min_completed_requests=2),
    )
    assert summary_a["total_generated_tokens"] == 256
    assert summary_a["completed_token_throughput_tokens_s"] == round(256 / 12.0, 2)

    # Case B: Missing token count on one request -> token throughput must be None
    records_missing_tokens = [
        records_with_tokens[0],
        RequestRecord(
            request_id=2,
            scheduled_offset_s=1.0,
            scheduled_timestamp_utc="2026-10-20T12:00:01.000000Z",
            actual_dispatch_timestamp_utc="2026-10-20T12:00:01.000000Z",
            dispatch_delay_ms=1.8,
            status="completed",
            http_status_code=200,
            ttft_ms=60.0,
            e2e_ms=1000.0,
            actual_output_tokens=None,  # Missing usage metadata
            actual_chunk_count=128,
        ),
    ]

    summary_b = compute_stage_summary(
        records=records_missing_tokens,
        scheduled_rate=2.0,
        window_duration_s=10.0,
        total_elapsed_wall_clock_s=12.0,
        slo_config=SLOConfig(min_completed_requests=2),
    )
    assert summary_b["total_generated_tokens"] is None
    assert summary_b["completed_token_throughput_tokens_s"] is None


def test_throughput_denominators_and_drain_accounting():
    """Verify proper distinction between window-rate and total wall-clock service throughput."""
    records = [
        RequestRecord(
            request_id=i,
            scheduled_offset_s=float(i),
            scheduled_timestamp_utc="2026-10-20T12:00:00.000000Z",
            actual_dispatch_timestamp_utc="2026-10-20T12:00:00.000000Z",
            dispatch_delay_ms=1.5,
            status="completed",
            http_status_code=200,
            ttft_ms=45.0,
            e2e_ms=1500.0,
            actual_output_tokens=100,
            actual_chunk_count=100,
        )
        for i in range(1, 21)
    ]

    window_s = 60.0
    total_elapsed_s = 75.0  # 15s drain

    summary = compute_stage_summary(
        records=records,
        scheduled_rate=0.33,
        window_duration_s=window_s,
        total_elapsed_wall_clock_s=total_elapsed_s,
        slo_config=SLOConfig(),
    )

    # Offered window rate: 20 / 60 = 0.33 req/s
    assert summary["completed_request_rate_per_window_s"] == 0.33
    # Total elapsed throughput: 20 / 75 = 0.27 req/s
    assert summary["physical_service_throughput_req_s"] == 0.27
    assert summary["drain_duration_s"] == 15.0
    assert summary["total_elapsed_wall_clock_s"] == 75.0


def test_reconciliation_of_request_states():
    """Verify request reconciliation across completed, failed, timed_out, and incomplete."""
    records = [
        RequestRecord(request_id=1, scheduled_offset_s=1.0, scheduled_timestamp_utc="", actual_dispatch_timestamp_utc="", dispatch_delay_ms=0.5, status="completed", ttft_ms=50.0, e2e_ms=1000.0),
        RequestRecord(request_id=2, scheduled_offset_s=2.0, scheduled_timestamp_utc="", actual_dispatch_timestamp_utc="", dispatch_delay_ms=0.8, status="completed", ttft_ms=55.0, e2e_ms=1100.0),
        RequestRecord(request_id=3, scheduled_offset_s=3.0, scheduled_timestamp_utc="", actual_dispatch_timestamp_utc="", dispatch_delay_ms=0.2, status="failed", error_message="HTTP 500"),
        RequestRecord(request_id=4, scheduled_offset_s=4.0, scheduled_timestamp_utc="", actual_dispatch_timestamp_utc="", dispatch_delay_ms=1.1, status="timed_out", error_message="Timeout"),
        RequestRecord(request_id=5, scheduled_offset_s=5.0, scheduled_timestamp_utc="", actual_dispatch_timestamp_utc="", dispatch_delay_ms=0.0, status="incomplete", error_message="Stream dropped"),
    ]

    summary = compute_stage_summary(
        records=records,
        scheduled_rate=1.0,
        window_duration_s=10.0,
        total_elapsed_wall_clock_s=15.0,
        slo_config=SLOConfig(min_completed_requests=2),
    )

    assert summary["dispatched_requests"] == 5
    assert summary["completed_requests"] == 2
    assert summary["failed_requests"] == 3
    assert summary["error_rate"] == 0.6
    assert summary["dispatched_requests"] == summary["completed_requests"] + summary["failed_requests"]
    assert summary["slo_evaluation"]["slo_status"] == "FAIL"


def test_incomplete_stream_counts_as_error():
    """Verify that an incomplete stream missing finish_reason/[DONE] fails the SLO."""
    incomplete_record = RequestRecord(
        request_id=1,
        scheduled_offset_s=1.0,
        scheduled_timestamp_utc="",
        actual_dispatch_timestamp_utc="",
        dispatch_delay_ms=0.5,
        status="incomplete",
        error_message="Stream terminated abruptly without [DONE] or finish_reason",
        actual_chunk_count=5,
    )

    records = [incomplete_record] + [
        RequestRecord(
            request_id=i,
            scheduled_offset_s=float(i),
            scheduled_timestamp_utc="",
            actual_dispatch_timestamp_utc="",
            dispatch_delay_ms=0.2,
            status="completed",
            ttft_ms=50.0,
            e2e_ms=1000.0,
            actual_output_tokens=100,
        )
        for i in range(2, 25)
    ]

    summary = compute_stage_summary(
        records=records,
        scheduled_rate=1.0,
        window_duration_s=25.0,
        total_elapsed_wall_clock_s=25.0,
        slo_config=SLOConfig(min_completed_requests=20),
    )

    assert summary["dispatched_requests"] == 24
    assert summary["completed_requests"] == 23
    assert summary["failed_requests"] == 1
    assert summary["error_rate"] > 0.0
    # SLO must fail due to non-zero error rate
    assert summary["slo_evaluation"]["slo_status"] == "FAIL"
    assert summary["slo_evaluation"]["error_pass"] is False


def test_circuit_breaker_trips_on_consecutive_timeouts(monkeypatch):
    """Verify that 3 consecutive timeouts immediately trip the circuit breaker and halt dispatch."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    mock_rec_timeout = RequestRecord(
        request_id=1,
        scheduled_offset_s=0.01,
        scheduled_timestamp_utc="",
        actual_dispatch_timestamp_utc="",
        dispatch_delay_ms=0.0,
        status="timed_out",
        error_message="Timed out",
    )

    async def mock_execute(*args, **kwargs):
        return mock_rec_timeout

    import vllm_online_benchmark

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_httpx = MagicMock()
    mock_httpx.Limits.return_value = MagicMock()
    mock_httpx.AsyncClient.return_value = mock_client

    monkeypatch.setattr(vllm_online_benchmark, "httpx", mock_httpx)
    monkeypatch.setattr(vllm_online_benchmark, "execute_streaming_request", mock_execute)

    async def run_test():
        records, tripped, reason, elapsed = await run_stage_with_circuit_breaker(
            endpoint_url="http://fake:8000/v1/completions",
            model_id="test-model",
            prompt_text="hello",
            rate=100.0,
            duration_s=2.0,
            drain_timeout_s=1.0,
            max_consecutive_timeouts=3,
            max_error_rate=0.5,
        )
        return records, tripped, reason

    records, tripped, reason = asyncio.run(run_test())
    assert tripped is True
    assert "consecutive request timeouts" in reason
    assert len(records) < 50


def test_circuit_breaker_trips_on_running_error_rate(monkeypatch):
    """Verify that exceeding 5% error rate across >= 10 resolved requests trips the circuit breaker."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    counter = 0

    async def mock_execute(*args, **kwargs):
        nonlocal counter
        counter += 1
        # Requests 1 to 8 succeed, 9 and 10 fail -> 2/10 = 20% > 5%
        status = "failed" if counter in (9, 10) else "completed"
        return RequestRecord(
            request_id=kwargs.get("request_id", counter),
            scheduled_offset_s=0.01,
            scheduled_timestamp_utc="",
            actual_dispatch_timestamp_utc="",
            dispatch_delay_ms=0.0,
            status=status,
            ttft_ms=50.0 if status == "completed" else None,
            e2e_ms=1000.0 if status == "completed" else None,
            actual_output_tokens=100 if status == "completed" else None,
            error_message="HTTP 500" if status == "failed" else None,
        )

    import vllm_online_benchmark

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_httpx = MagicMock()
    mock_httpx.Limits.return_value = MagicMock()
    mock_httpx.AsyncClient.return_value = mock_client

    monkeypatch.setattr(vllm_online_benchmark, "httpx", mock_httpx)
    monkeypatch.setattr(vllm_online_benchmark, "execute_streaming_request", mock_execute)

    async def run_test():
        records, tripped, reason, elapsed = await run_stage_with_circuit_breaker(
            endpoint_url="http://fake:8000/v1/completions",
            model_id="test-model",
            prompt_text="hello",
            rate=50.0,
            duration_s=2.0,
            drain_timeout_s=1.0,
            max_consecutive_timeouts=5,
            max_error_rate=0.05,
        )
        return records, tripped, reason

    records, tripped, reason = asyncio.run(run_test())
    assert tripped is True
    assert "Running error rate" in reason
    assert len(records) < 100

