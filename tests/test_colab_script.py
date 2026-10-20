"""Unit tests for Colab benchmark validation script structure and helpers (no GPU required)."""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_colab_script_syntax():
    """Verify that colab/vllm_validation.py is syntactically valid Python."""
    script_path = Path("colab/vllm_validation.py")
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")
    parsed = ast.parse(content)
    assert isinstance(parsed, ast.Module)


def test_select_default_model_by_vram():
    """Verify dynamic model recommendation based on detected GPU VRAM capacity."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation import select_default_model

    # On free Colab T4 (15-16 GiB)
    assert select_default_model(15.5) == "Qwen/Qwen2.5-1.5B-Instruct"
    assert select_default_model(16.0) == "Qwen/Qwen2.5-1.5B-Instruct"

    # On Colab Pro L4 / A100 (>=20 GiB)
    assert select_default_model(22.5) == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert select_default_model(80.0) == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_prepare_exact_tokens_with_mock_tokenizer():
    """Verify that token ID slicing produces the exact integer count requested."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation import prepare_exact_tokens

    mock_tokenizer = MagicMock()
    # Mock tokenizer returns a list of token IDs
    mock_tokenizer.encode.side_effect = lambda text: list(range(1000))

    tokens_128 = prepare_exact_tokens(mock_tokenizer, 128)
    assert len(tokens_128) == 128
    assert tokens_128 == list(range(128))

    tokens_512 = prepare_exact_tokens(mock_tokenizer, 512)
    assert len(tokens_512) == 512
    assert tokens_512 == list(range(512))


def test_gpu_environment_metadata_handles_no_gpu():
    """Verify that get_gpu_environment_metadata raises SystemError when run on CPU-only machines."""
    import sys
    sys.path.insert(0, str(Path("colab").resolve()))
    from vllm_validation import get_gpu_environment_metadata

    # On standard non-GPU local machine, this must fail cleanly with a helpful error message
    with pytest.raises((SystemError, ImportError)) as excinfo:
        get_gpu_environment_metadata()
    assert any(keyword in str(excinfo.value).lower() for keyword in ["cuda", "pytorch"])
