"""Unit tests for workload simulator, corrected metrics, and scheduling behavior."""

import pytest
from inference_lab.hardware import A100_SXM4_80GB
from inference_lab.models import LLAMA_3_8B, Precision
from inference_lab.simulator import Request, SimpleWorkloadSimulator, SimulationConfig
from inference_lab.workload import WorkloadConfig, generate_workload


@pytest.fixture
def base_sim_config():
    return SimulationConfig(
        model=LLAMA_3_8B,
        hardware=A100_SXM4_80GB,
        precision=Precision.FP16,
        max_concurrency=8,
        random_seed=123,
    )


def test_simulation_reproducibility(base_sim_config):
    """Running identical workloads must yield identical metric results."""
    wl_config = WorkloadConfig(
        num_requests=10,
        prompt_length=256,
        output_length=64,
        arrival_pattern="poisson",
        arrival_rate_rps=5.0,
        seed=999,
    )

    reqs1 = generate_workload(wl_config)
    reqs2 = generate_workload(wl_config)

    sim1 = SimpleWorkloadSimulator(base_sim_config)
    res1 = sim1.run(reqs1)

    sim2 = SimpleWorkloadSimulator(base_sim_config)
    res2 = sim2.run(reqs2)

    assert res1.makespan_s == pytest.approx(res2.makespan_s, rel=1e-6)
    assert res1.tokens_per_second == pytest.approx(res2.tokens_per_second, rel=1e-6)
    assert res1.avg_ttft_s == pytest.approx(res2.avg_ttft_s, rel=1e-6)
    assert res1.avg_tpot_s == pytest.approx(res2.avg_tpot_s, rel=1e-6)


def test_ttft_identity_and_latency_decomposition(base_sim_config):
    """Verify that TTFT == queue_time + prefill_service_time holds for all requests."""
    wl_config = WorkloadConfig(
        num_requests=12,
        prompt_length=256,
        output_length=32,
        arrival_pattern="uniform",
        arrival_rate_rps=20.0,  # High arrival rate to induce queueing
        seed=42,
    )
    reqs = generate_workload(wl_config)
    sim = SimpleWorkloadSimulator(base_sim_config)
    res = sim.run(reqs)

    for r in res.requests:
        # Identity check within floating point tolerance
        assert r.ttft == pytest.approx(r.queue_time + r.prefill_service_time, abs=1e-9)
        assert r.prefill_service_time > 0.0
        assert r.e2e_latency == pytest.approx(
            r.queue_time + r.prefill_service_time + (r.finish_time - r.prefill_end_time),
            abs=1e-9,
        )

    # First request arriving at t=0 in isolation must have 0 queue time
    req0 = res.requests[0]
    assert req0.queue_time == 0.0
    assert req0.ttft == req0.prefill_service_time


def test_tpot_calculation_and_edge_cases(base_sim_config):
    """Verify TPOT counts actual decode intervals (output_length - 1) and handles edge cases."""
    sim = SimpleWorkloadSimulator(base_sim_config)

    # Case 1: output_length = 1 (no decode steps occurred; only prefill token emitted)
    req_single = [Request(request_id=0, arrival_time=0.0, prompt_length=128, output_length=1)]
    res_single = sim.run(req_single)
    assert res_single.requests[0].generated_tokens == 1
    assert res_single.requests[0].finish_time == res_single.requests[0].prefill_end_time
    assert res_single.requests[0].tpot == 0.0

    # Case 2: output_length = 2 (exactly 1 decode step)
    req_two = [Request(request_id=0, arrival_time=0.0, prompt_length=128, output_length=2)]
    res_two = sim.run(req_two)
    r2 = res_two.requests[0]
    assert r2.generated_tokens == 2
    decode_dur_2 = r2.finish_time - r2.prefill_end_time
    assert decode_dur_2 > 0.0
    # TPOT must equal exactly 1 decode step duration
    assert r2.tpot == pytest.approx(decode_dur_2 / 1, abs=1e-9)

    # Case 3: output_length = 5 (exactly 4 decode steps)
    req_five = [Request(request_id=0, arrival_time=0.0, prompt_length=128, output_length=5)]
    res_five = sim.run(req_five)
    r5 = res_five.requests[0]
    assert r5.generated_tokens == 5
    decode_dur_5 = r5.finish_time - r5.prefill_end_time
    assert r5.tpot == pytest.approx(decode_dur_5 / 4, abs=1e-9)


def test_hand_calculated_prefill_and_decode_values(base_sim_config):
    """Numerically verify prefill and decode formulas against independent hand calculation."""
    sim = SimpleWorkloadSimulator(base_sim_config)

    # Hand calculation for LLaMA-3-8B on A100-SXM4-80GB (FP16):
    # Prompt length = 128
    # Linear FLOPs: 2 * 8.03e9 * 128 = 2.05568e12
    # Quadratic attention FLOPs: 4 * 32 * 4096 * (128^2) = 8.589934592e9
    # Total FLOPs: 2.064269934592e12
    # Effective FLOPs/s: 312e12 * 0.50 = 1.56e14
    # Compute time: 2.064269934592e12 / 1.56e14 = 0.0132325 s (13.2325 ms)
    # Memory bytes: (8.03e9 * 2) + (128 * 131,072) = 1.6076777216e10 bytes
    # Effective Bandwidth: 2039e9 * 0.70 = 1.4273e12 B/s
    # Memory time: 1.6076777216e10 / 1.4273e12 = 0.01126377 s (11.2638 ms)
    # Roofline base: max(13.2325 ms, 11.2638 ms) = 13.2325 ms
    # Adding kernel launch overhead: 20e-6 s (0.02 ms) -> 0.0132525 s
    expected_prefill_128 = 0.0132525
    actual_prefill_128 = sim.compute_prefill_time(128)
    assert actual_prefill_128 == pytest.approx(expected_prefill_128, abs=1e-5)

    # Hand calculation for 1 decode step with batch size 1 at context length 512:
    # Memory bytes: (8.03e9 * 2) + (1 * 512 * 131,072) = 1.6127108864e10 bytes
    # Memory time: 1.6127108864e10 / 1.4273e12 = 0.011299 s (11.299 ms)
    # Compute FLOPs: 2 * 8.03e9 * 1 = 1.606e10
    # Compute time: 1.606e10 / 1.56e14 = 0.0001029 s (0.103 ms)
    # Roofline: max(11.299 ms, 0.103 ms) = 11.299 ms + 0.02 ms = 11.319 ms = 0.011319 s
    dummy_req = Request(request_id=0, arrival_time=0.0, prompt_length=512, output_length=10)
    dummy_req.generated_tokens = 0
    actual_decode_step = sim.compute_decode_step_time([dummy_req])
    assert actual_decode_step == pytest.approx(0.011319, abs=1e-5)


def test_interleaved_scheduling_event_ordering(base_sim_config):
    """Verify that requests decode incrementally rather than serializing all admissions upfront."""
    sim = SimpleWorkloadSimulator(base_sim_config)

    # Submit 2 burst requests at t=0 with prompt=128 and output=3
    reqs = [
        Request(request_id=0, arrival_time=0.0, prompt_length=128, output_length=3),
        Request(request_id=1, arrival_time=0.0, prompt_length=128, output_length=3),
    ]

    res = sim.run(reqs)
    r0, r1 = res.requests[0], res.requests[1]

    # In interleaved scheduling:
    # Req 0 prefill runs first (ends at t1)
    # Req 0 decodes 1 step (ends at t2). During this time, Req 1 has not finished prefill!
    # Req 1 prefill runs (starts at t2, ends at t3).
    # Therefore, Req 1 start time must be >= Req 0 prefill_end_time!
    assert r1.start_time >= r0.prefill_end_time
    # Both finish successfully
    assert r0.generated_tokens == 3
    assert r1.generated_tokens == 3
    # R0 finishes before R1
    assert r0.finish_time < r1.finish_time


def test_prompt_length_monotonicity_on_ttft(base_sim_config):
    """Increasing prompt length must monotonically increase prefill time and TTFT."""
    sim = SimpleWorkloadSimulator(base_sim_config)

    # Isolated requests arriving at t=0
    req_short = [Request(request_id=0, arrival_time=0.0, prompt_length=128, output_length=10)]
    req_med = [Request(request_id=0, arrival_time=0.0, prompt_length=1024, output_length=10)]
    req_long = [Request(request_id=0, arrival_time=0.0, prompt_length=4096, output_length=10)]

    res_short = sim.run(req_short)
    res_med = sim.run(req_med)
    res_long = sim.run(req_long)

    assert res_short.avg_prefill_service_time_s < res_med.avg_prefill_service_time_s < res_long.avg_prefill_service_time_s
    assert res_short.avg_ttft_s < res_med.avg_ttft_s < res_long.avg_ttft_s


def test_output_length_monotonicity_on_e2e_latency(base_sim_config):
    """Increasing output length must monotonically increase total turnaround latency."""
    sim = SimpleWorkloadSimulator(base_sim_config)

    req_short = [Request(request_id=0, arrival_time=0.0, prompt_length=256, output_length=32)]
    req_med = [Request(request_id=0, arrival_time=0.0, prompt_length=256, output_length=128)]
    req_long = [Request(request_id=0, arrival_time=0.0, prompt_length=256, output_length=512)]

    res_short = sim.run(req_short)
    res_med = sim.run(req_med)
    res_long = sim.run(req_long)

    assert res_short.avg_e2e_latency_s < res_med.avg_e2e_latency_s < res_long.avg_e2e_latency_s


def test_concurrency_throughput_scaling():
    """Higher concurrency must increase total token throughput by amortizing weight transfers."""
    req_factory = lambda: [
        Request(request_id=i, arrival_time=0.0, prompt_length=256, output_length=64)
        for i in range(16)
    ]

    cfg_c1 = SimulationConfig(
        model=LLAMA_3_8B,
        hardware=A100_SXM4_80GB,
        precision=Precision.FP16,
        max_concurrency=1,
    )
    cfg_c4 = SimulationConfig(
        model=LLAMA_3_8B,
        hardware=A100_SXM4_80GB,
        precision=Precision.FP16,
        max_concurrency=4,
    )
    cfg_c16 = SimulationConfig(
        model=LLAMA_3_8B,
        hardware=A100_SXM4_80GB,
        precision=Precision.FP16,
        max_concurrency=16,
    )

    res_c1 = SimpleWorkloadSimulator(cfg_c1).run(req_factory())
    res_c4 = SimpleWorkloadSimulator(cfg_c4).run(req_factory())
    res_c16 = SimpleWorkloadSimulator(cfg_c16).run(req_factory())

    # Throughput: C=16 > C=4 > C=1
    assert res_c16.tokens_per_second > res_c4.tokens_per_second > res_c1.tokens_per_second
    assert res_c16.makespan_s < res_c4.makespan_s < res_c1.makespan_s
