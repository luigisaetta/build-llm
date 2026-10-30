"""
Author: L. Saetta
Date Last Modified: 2026-10-30
License: MIT
Description: Tests for exact model parameter counts and analytical GPT estimates.
"""

import pytest
import torch
from torch import nn

from build_llm.gpt2_weights import GPT2_SMALL_CONFIG
from build_llm.model_parameters import count_parameters, estimate_gpt_parameters


def test_count_parameters_handles_shared_and_frozen_parameters():
    """Shared weights count once; freezing only affects the optional filter."""
    model = nn.Module()
    model.embedding = nn.Embedding(7, 4)
    model.head = nn.Linear(4, 7, bias=True)
    model.head.weight = model.embedding.weight
    model.embedding.weight.requires_grad_(False)
    model.register_buffer("mask", torch.ones(10))

    assert count_parameters(model) == 35
    assert count_parameters(model, trainable_only=True) == 7
    assert count_parameters(nn.Identity()) == 0


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_estimate_matches_tiny_modules(use_bias, tied):
    """Compare the formula with independently constructed PyTorch components."""
    config = {
        "vocab_size": 11,
        "context_length": 8,
        "emb_dim": 4,
        "n_heads": 2,
        "n_layers": 2,
        "ff_hidden_dim": 9,
        "qkv_bias": use_bias,
        "attention_output_bias": use_bias,
        "mlp_bias": use_bias,
        "layer_norm_bias": use_bias,
        "output_bias": use_bias,
        "tie_embeddings": tied,
    }
    model = nn.Module()
    model.embedding = nn.Embedding(11, 4)
    model.position = nn.Embedding(8, 4)
    model.blocks = nn.ModuleList()
    for _ in range(2):
        # Only the parameter structure matters here, not a forward computation.
        block = nn.ModuleList(
            [
                *[nn.Linear(4, 4, bias=use_bias) for _ in range(4)],
                nn.Linear(4, 9, bias=use_bias),
                nn.Linear(9, 4, bias=use_bias),
                nn.LayerNorm(4, bias=use_bias),
                nn.LayerNorm(4, bias=use_bias),
            ]
        )
        model.blocks.append(block)
    model.final_norm = nn.LayerNorm(4, bias=use_bias)
    model.head = nn.Linear(4, 11, bias=use_bias)
    if tied:
        model.head.weight = model.embedding.weight

    counts = estimate_gpt_parameters(config)
    assert counts["total"] == count_parameters(model)
    assert counts["total"] == sum(
        value for key, value in counts.items() if key != "total"
    )


def test_standard_config_and_weight_tying():
    """The notebook configuration distinguishes untied weights from GPT-2 tying."""
    config = {**GPT2_SMALL_CONFIG, "qkv_bias": False}
    assert estimate_gpt_parameters(config)["total"] == 163009536
    config.update(tie_embeddings=True, qkv_bias=True)
    assert estimate_gpt_parameters(config)["total"] == 124439808


@pytest.mark.parametrize(
    "changes",
    [
        {"emb_dim": 0},
        {"vocab_size": True},
        {"n_layers": -1},
        {"ff_hidden_dim": 1.5},
        {"n_heads": 3},
        {"n_heads": 0},
        {"qkv_bias": "false"},
    ],
)
def test_estimate_rejects_invalid_configuration(changes):
    """Invalid dimensions and ambiguous flags fail before counting."""
    config = {"vocab_size": 7, "context_length": 8, "emb_dim": 4, "n_layers": 1}
    config.update(changes)
    with pytest.raises(ValueError):
        estimate_gpt_parameters(config)


def test_missing_configuration_field():
    """Missing dimensions produce an actionable error."""
    with pytest.raises(ValueError, match="vocab_size"):
        estimate_gpt_parameters({})


def test_zero_layers_and_head_count():
    """Zero blocks are supported; splitting a fixed width adds no parameters."""
    config = {"vocab_size": 7, "context_length": 8, "emb_dim": 4, "n_layers": 0}
    assert estimate_gpt_parameters(config)["total"] == 96
    config.update(n_layers=2, n_heads=1)
    first = estimate_gpt_parameters(config)
    config["n_heads"] = 2
    assert estimate_gpt_parameters(config) == first


def test_count_parameters_on_meta_device():
    """Shape-only modules can be counted without allocating weight storage."""
    with torch.device("meta"):
        model = nn.Linear(4, 3)
    assert count_parameters(model) == 15
