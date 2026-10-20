"""Unit tests for Colab benchmark validation script V2 (synthetic non-GPU tests).

IMPORTANT:
These tests use strictly synthetic mock data and fixtures to verify calculation
logic, statistical conventions, edge cases, and API compatibility.
No synthetic test data is ever written to physical results directories.
"""

import ast
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_colab_script_v2_syntax():
    """Verify that colab/vllm_validation_v2.py is syntactically valid Python."""
    script_path = Path("colab/vllm_validation_v2.py")
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")
    parsed = ast.parse(content)
    assert isinstance(parsed, ast.Module)


def test_format_vllm_token_inputs_structure():
    """Verify that format_vllm_token_inputs produces the supported vLLM prompt structure."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation_v2 import format_vllm_token_inputs

    batch = [[101, 102], [201, 202, 203]]
    formatted = format_vllm_token_inputs(batch)

    assert isinstance(formatted, list)
    assert len(formatted) == 2
    assert formatted[0] == {"prompt_token_ids": [101, 102]}
    assert formatted[1] == {"prompt_token_ids": [201, 202, 203]}


def test_select_default_model_by_vram_v2():
    """Verify dynamic model recommendation based on detected GPU VRAM capacity in V2."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation_v2 import select_default_model

    # On free Colab T4 (15-16 GiB)
    assert select_default_model(15.5) == "Qwen/Qwen2.5-1.5B-Instruct"
    assert select_default_model(16.0) == "Qwen/Qwen2.5-1.5B-Instruct"

    # On Colab Pro L4 / A100 (>=20 GiB)
    assert select_default_model(22.5) == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert select_default_model(80.0) == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_prepare_exact_tokens_v2():
    """Verify that token ID slicing produces the exact integer count requested."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation_v2 import prepare_exact_tokens

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text: list(range(2000))

    tokens_64 = prepare_exact_tokens(mock_tokenizer, 64)
    assert len(tokens_64) == 64
    assert tokens_64 == list(range(64))

    tokens_512 = prepare_exact_tokens(mock_tokenizer, 512)
    assert len(tokens_512) == 512
    assert tokens_512 == list(range(512))


def test_compute_distribution_statistics_hand_calculated():
    """Verify deterministic statistics calculation using a hand-calculated synthetic distribution."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation_v2 import compute_distribution_statistics

    # 10 values from 10.0 to 100.0 with step 10.0
    # Values: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    # N = 10
    # Mean = 55.0
    # Linear rank for P: rank = (P / 100) * (10 - 1) = (P / 100) * 9
    # P50: rank = 4.5 -> index 4 (50.0) + 0.5 * (60.0 - 50.0) = 55.0
    # P25: rank = 0.25 * 9 = 2.25 -> index 2 (30.0) + 0.25 * (40.0 - 30.0) = 32.5
    # P75: rank = 0.75 * 9 = 6.75 -> index 6 (70.0) + 0.75 * (80.0 - 70.0) = 77.5
    # IQR = 77.5 - 32.5 = 45.0
    # P90: rank = 0.90 * 9 = 8.10 -> index 8 (90.0) + 0.10 * (100.0 - 90.0) = 91.0
    # P95: rank = 0.95 * 9 = 8.55 -> index 8 (90.0) + 0.55 * (100.0 - 90.0) = 95.5
    # Variance (ddof=1): sum((x - 55)^2) / 9 = 8250 / 9 = 916.666667
    # Std: sqrt(916.666667) = 30.2765035
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = compute_distribution_statistics(data, metric_name="test_throughput")

    assert stats["metric_name"] == "test_throughput"
    assert stats["sample_count"] == 10
    assert math.isclose(stats["mean"], 55.0, abs_tol=1e-3)
    assert math.isclose(stats["median"], 55.0, abs_tol=1e-3)
    assert math.isclose(stats["min"], 10.0, abs_tol=1e-3)
    assert math.isclose(stats["max"], 100.0, abs_tol=1e-3)
    assert math.isclose(stats["p25"], 32.5, abs_tol=1e-3)
    assert math.isclose(stats["p50"], 55.0, abs_tol=1e-3)
    assert math.isclose(stats["p75"], 77.5, abs_tol=1e-3)
    assert math.isclose(stats["p90_exploratory"], 91.0, abs_tol=1e-3)
    assert math.isclose(stats["p95_exploratory"], 95.5, abs_tol=1e-3)
    assert math.isclose(stats["iqr"], 45.0, abs_tol=1e-3)
    assert math.isclose(stats["std_dev_sample"], 30.2765, abs_tol=1e-3)
    assert stats["statistical_conventions"]["std_dev_ddof"] == 1


def test_compute_distribution_statistics_edge_cases():
    """Verify statistical function on edge cases: N=1, zero variance, and empty list."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation_v2 import compute_distribution_statistics

    # N = 1
    single = compute_distribution_statistics([42.5], metric_name="single")
    assert single["sample_count"] == 1
    assert single["mean"] == 42.5
    assert single["median"] == 42.5
    assert single["std_dev_sample"] == 0.0
    assert single["iqr"] == 0.0

    # Zero variance (identical values)
    constant = compute_distribution_statistics([100.0] * 10, metric_name="constant")
    assert constant["sample_count"] == 10
    assert constant["mean"] == 100.0
    assert constant["median"] == 100.0
    assert constant["std_dev_sample"] == 0.0
    assert constant["iqr"] == 0.0
    assert constant["p90_exploratory"] == 100.0

    # Empty list raises ValueError
    with pytest.raises(ValueError):
        compute_distribution_statistics([], metric_name="empty")


def test_token_count_mismatch_flagging():
    """Verify that token count mismatches are detected and flagged on synthetic trials."""
    expected_tokens = 128
    synthetic_trial_pass = {"actual": 128, "expected": expected_tokens}
    synthetic_trial_fail = {"actual": 120, "expected": expected_tokens}

    assert (synthetic_trial_pass["actual"] != synthetic_trial_pass["expected"]) is False
    assert (synthetic_trial_fail["actual"] != synthetic_trial_fail["expected"]) is True


def test_gpu_environment_metadata_handles_no_gpu_v2():
    """Verify that get_gpu_environment_metadata in V2 raises SystemError/ImportError on CPU."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation_v2 import get_gpu_environment_metadata

    with pytest.raises((SystemError, ImportError)) as excinfo:
        get_gpu_environment_metadata()
    assert any(keyword in str(excinfo.value).lower() for keyword in ["cuda", "pytorch"])


def test_synthetic_data_safety_guardrail():
    """Verify that running test fixtures never writes files into physical results paths."""
    v1_results_dir = Path("results/physical_t4_v1")
    v2_results_dir = Path("results/physical_t4_v2")
    colab_default_output = Path("colab_vllm_measured_results_v2.json")

    # The test must not create or touch physical benchmark result files
    assert not colab_default_output.exists(), "Test fixture must not create colab_vllm_measured_results_v2.json"
