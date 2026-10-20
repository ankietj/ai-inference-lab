"""Unit tests for model architecture configurations and documented provenance."""

import pytest
from inference_lab.models import (
    LLAMA_1_7B,
    LLAMA_3_8B,
    LLAMA_3_70B,
    MISTRAL_7B_V01,
    MODEL_REGISTRY,
    get_model_config,
)


def test_model_registry_keys():
    expected_models = {"llama-3-8b", "mistral-7b-v0.1", "llama-1-7b", "llama-3-70b"}
    assert set(MODEL_REGISTRY.keys()) == expected_models


def test_model_provenance_present():
    for name, model in MODEL_REGISTRY.items():
        assert model.provenance, f"Model {name} must have documented provenance"
        assert len(model.provenance) > 15


def test_head_dimension_consistency():
    for name, model in MODEL_REGISTRY.items():
        assert model.d_model % model.n_heads == 0, f"{name}: d_model must divide n_heads"
        assert model.d_head == 128, f"{name}: Expected standard d_head of 128"


def test_mha_vs_gqa_classification():
    # LLaMA-1 7B is classic MHA
    assert LLAMA_1_7B.is_mha
    assert not LLAMA_1_7B.is_gqa
    assert LLAMA_1_7B.gqa_ratio == 1.0

    # LLaMA-3 8B is GQA with 4:1 ratio (32 query heads, 8 KV heads)
    assert not LLAMA_3_8B.is_mha
    assert LLAMA_3_8B.is_gqa
    assert LLAMA_3_8B.gqa_ratio == 4.0

    # Mistral-7B-v0.1 is GQA with 4:1 ratio
    assert not MISTRAL_7B_V01.is_mha
    assert MISTRAL_7B_V01.is_gqa
    assert MISTRAL_7B_V01.gqa_ratio == 4.0

    # LLaMA-3 70B is GQA with 8:1 ratio (64 query heads, 8 KV heads)
    assert not LLAMA_3_70B.is_mha
    assert LLAMA_3_70B.is_gqa
    assert LLAMA_3_70B.gqa_ratio == 8.0


def test_get_model_config_invalid():
    with pytest.raises(KeyError):
        get_model_config("non-existent-model")
