"""Integration test for Phase 1 V2 experiment pipeline and chart generation."""

from pathlib import Path
from inference_lab.experiment import run_phase1_experiments
from inference_lab.plots import generate_phase1_charts


def test_experiment_execution_and_chart_generation(tmp_path: Path):
    # Run experiments in isolated temporary directory
    results = run_phase1_experiments(output_dir=tmp_path)

    assert "metadata" in results
    assert "platform" in results["metadata"]
    assert "hardware_specs" in results["metadata"]
    assert len(results["prompt_length_sweep"]) == 6
    assert len(results["output_length_sweep"]) == 5
    assert len(results["concurrency_sweep"]) == 7
    assert len(results["kv_cache_comparison"]) == 5

    # Check sweep fields
    p0 = results["prompt_length_sweep"][0]
    assert "avg_queue_ms" in p0
    assert "avg_prefill_service_ms" in p0
    assert "avg_ttft_ms" in p0

    c0 = results["concurrency_sweep"][0]
    assert "peak_kv_cache_mib" in c0

    kv0 = results["kv_cache_comparison"][0]
    assert "mha_fp16_mib" in kv0

    json_file = tmp_path / "phase1_results_v2.json"
    assert json_file.exists()

    fig_file = tmp_path / "figures" / "test_chart_v2.png"
    out_fig = generate_phase1_charts(
        results_path=json_file,
        output_image_path=fig_file,
    )
    assert out_fig.exists()
    assert out_fig.stat().st_size > 50_000  # Generated PNG must be non-empty (>50KB)
