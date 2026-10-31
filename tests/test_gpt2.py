"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests for the GPT-2 building blocks in build_llm.gpt2.
"""

import pytest
import torch
import torch.nn.functional as F

from build_llm.gpt2 import (
    GELU,
    FeedForward,
    GPTModel,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
    generate_text_simple,
)
from build_llm.model_parameters import count_parameters, estimate_gpt_parameters

TINY_CONFIG = {
    "vocab_size": 50,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


@pytest.fixture(autouse=True)
def fixed_seed():
    """Make the random weights and inputs of every test reproducible."""
    torch.manual_seed(123)


def test_layer_norm_matches_pytorch():
    """With the initial scale and shift, the result equals F.layer_norm."""
    x = torch.randn(2, 5, 16)

    result = LayerNorm(16)(x)

    assert torch.allclose(result, F.layer_norm(x, (16,), eps=1e-5), atol=1e-6)


def test_gelu_matches_pytorch_tanh_approximation():
    """GELU equals the tanh approximation of PyTorch."""
    x = torch.linspace(-5, 5, 101)

    assert torch.allclose(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_multi_head_attention_matches_scaled_dot_product_attention():
    """The attention output equals PyTorch's causal scaled dot-product attention."""
    attention = MultiHeadAttention(16, 16, context_length=8, dropout=0.0, num_heads=4)
    x = torch.randn(2, 5, 16)

    def split_heads(tensor):
        return tensor.view(2, 5, 4, 4).transpose(1, 2)

    queries = split_heads(attention.W_query(x))
    keys = split_heads(attention.W_key(x))
    values = split_heads(attention.W_value(x))
    heads = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
    expected = attention.out_proj(heads.transpose(1, 2).reshape(2, 5, 16))

    assert torch.allclose(attention(x), expected, atol=1e-6)


def test_multi_head_attention_rejects_invalid_head_count():
    """d_out must be divisible by the number of heads."""
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(16, 16, context_length=8, dropout=0.0, num_heads=3)


def test_blocks_preserve_shape():
    """The feed-forward network and the transformer block keep the input shape."""
    x = torch.randn(2, 5, 16)

    assert FeedForward(TINY_CONFIG)(x).shape == x.shape
    assert TransformerBlock(TINY_CONFIG)(x).shape == x.shape


def test_gpt_model_returns_logits_per_position():
    """The model maps (batch_size, seq_len) token IDs to vocabulary logits."""
    model = GPTModel(TINY_CONFIG).eval()
    token_ids = torch.randint(0, 50, (2, 6))

    assert model(token_ids).shape == (2, 6, 50)


def test_gpt_model_is_causal():
    """Changing the last token does not change the logits of earlier positions."""
    model = GPTModel(TINY_CONFIG).eval()
    token_ids = torch.randint(0, 50, (1, 6))
    changed = token_ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 50

    original_logits = model(token_ids)
    changed_logits = model(changed)

    assert torch.allclose(original_logits[:, :-1], changed_logits[:, :-1], atol=1e-6)
    assert not torch.allclose(original_logits[:, -1], changed_logits[:, -1])


def test_gpt_model_rejects_inputs_longer_than_context():
    """An input longer than context_length raises a clear error."""
    model = GPTModel(TINY_CONFIG)

    with pytest.raises(ValueError, match="context length"):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_gpt_model_parameter_count_matches_estimate():
    """The parameters of the module equal the analytical estimate."""
    model = GPTModel(TINY_CONFIG)

    expected = estimate_gpt_parameters(TINY_CONFIG)["total"]

    assert count_parameters(model) == expected


def test_gpt_model_has_separate_output_weights_by_default():
    """Without tie_embeddings, the output head has its own matrix, as in the book."""
    model = GPTModel(TINY_CONFIG)

    assert model.out_head.weight is not model.tok_emb.weight


def test_tied_model_shares_one_weight_tensor():
    """With tie_embeddings, the output head and the embeddings share a Parameter."""
    model = GPTModel(TINY_CONFIG, tie_embeddings=True)

    assert model.out_head.weight is model.tok_emb.weight
    expected = estimate_gpt_parameters({**TINY_CONFIG, "tie_embeddings": True})
    assert count_parameters(model) == expected["total"]
    untied = count_parameters(GPTModel(TINY_CONFIG))
    assert untied - count_parameters(model) == 50 * 16  # vocab_size * emb_dim


def test_tied_weights_stay_shared_after_a_training_step():
    """One optimizer step updates the shared matrix, which stays shared."""
    model = GPTModel(TINY_CONFIG, tie_embeddings=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.tok_emb.weight.detach().clone()
    token_ids = torch.randint(0, 50, (2, 6))

    loss = F.cross_entropy(model(token_ids).flatten(0, 1), token_ids.flatten())
    loss.backward()
    optimizer.step()

    assert model.out_head.weight is model.tok_emb.weight
    assert not torch.equal(model.tok_emb.weight, before)


def test_generate_text_simple_appends_greedy_tokens():
    """Generation appends the argmax tokens and keeps the input as a prefix."""
    model = GPTModel(TINY_CONFIG).eval()
    start = torch.randint(0, 50, (2, 3))

    result = generate_text_simple(model, start, max_new_tokens=7, context_size=8)

    assert result.shape == (2, 10)
    assert torch.equal(result[:, :3], start)
    expected_first = model(start)[:, -1, :].argmax(dim=-1)
    assert torch.equal(result[:, 3], expected_first)


def test_generate_text_simple_crops_long_contexts():
    """Contexts longer than context_size are cropped, so generation can continue."""
    model = GPTModel(TINY_CONFIG).eval()
    start = torch.randint(0, 50, (1, 8))

    result = generate_text_simple(model, start, max_new_tokens=4, context_size=8)

    assert result.shape == (1, 12)
