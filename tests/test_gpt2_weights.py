"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests for downloading and loading the GPT-2 weights of Hugging Face.
"""

import hashlib
import io
import json
import urllib.request

import pytest
import torch
import torch.nn.functional as F

from build_llm import gpt2_weights
from build_llm.gpt2 import GPTModel
from build_llm.gpt2_weights import (
    WEIGHTS_FILE,
    convert_hf_gpt2_state_dict,
    download_gpt2_small_weights,
    load_hf_gpt2_weights,
)

TINY_CONFIG = {
    "vocab_size": 11,
    "context_length": 6,
    "emb_dim": 8,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def make_hf_state_dict(cfg):
    """Build random tensors with the names and shapes of a Hugging Face GPT-2."""
    generator = torch.Generator().manual_seed(0)
    d, v, c = cfg["emb_dim"], cfg["vocab_size"], cfg["context_length"]

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.2

    tensors = {"wte.weight": rand(v, d), "wpe.weight": rand(c, d)}
    tensors |= {"ln_f.weight": 1 + rand(d), "ln_f.bias": rand(d)}
    for i in range(cfg["n_layers"]):
        tensors |= {
            f"h.{i}.ln_1.weight": 1 + rand(d),
            f"h.{i}.ln_1.bias": rand(d),
            f"h.{i}.attn.c_attn.weight": rand(d, 3 * d),
            f"h.{i}.attn.c_attn.bias": rand(3 * d),
            f"h.{i}.attn.c_proj.weight": rand(d, d),
            f"h.{i}.attn.c_proj.bias": rand(d),
            f"h.{i}.ln_2.weight": 1 + rand(d),
            f"h.{i}.ln_2.bias": rand(d),
            f"h.{i}.mlp.c_fc.weight": rand(d, 4 * d),
            f"h.{i}.mlp.c_fc.bias": rand(4 * d),
            f"h.{i}.mlp.c_proj.weight": rand(4 * d, d),
            f"h.{i}.mlp.c_proj.bias": rand(d),
        }
    return tensors


def reference_gpt2_logits(sd, cfg, token_ids):  # pylint: disable=too-many-locals
    """Compute GPT-2 logits directly from the Hugging Face tensors.

    Conv1D layers compute x @ weight + bias, with weight of shape (in, out).
    """
    d, heads = cfg["emb_dim"], cfg["n_heads"]
    batch, tokens = token_ids.shape
    x = sd["wte.weight"][token_ids] + sd["wpe.weight"][:tokens]
    for i in range(cfg["n_layers"]):

        def p(name, layer=i):
            return sd[f"h.{layer}.{name}"]

        h = F.layer_norm(x, (d,), p("ln_1.weight"), p("ln_1.bias"), 1e-5)
        qkv = h @ p("attn.c_attn.weight") + p("attn.c_attn.bias")
        q, k, v = (
            t.view(batch, tokens, heads, d // heads).transpose(1, 2)
            for t in qkv.split(d, dim=-1)
        )
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, tokens, d)
        x = x + attention @ p("attn.c_proj.weight") + p("attn.c_proj.bias")
        h = F.layer_norm(x, (d,), p("ln_2.weight"), p("ln_2.bias"), 1e-5)
        h = F.gelu(h @ p("mlp.c_fc.weight") + p("mlp.c_fc.bias"), approximate="tanh")
        x = x + h @ p("mlp.c_proj.weight") + p("mlp.c_proj.bias")
    x = F.layer_norm(x, (d,), sd["ln_f.weight"], sd["ln_f.bias"], 1e-5)
    return x @ sd["wte.weight"].T


def test_loaded_model_matches_reference_gpt2():
    """After loading, GPTModel computes the same logits as GPT-2 itself."""
    hf_state_dict = make_hf_state_dict(TINY_CONFIG)
    model = GPTModel(TINY_CONFIG).eval()
    load_hf_gpt2_weights(model, hf_state_dict)
    token_ids = torch.tensor([[1, 5, 2, 9, 3], [4, 4, 0, 7, 10]])

    with torch.no_grad():
        logits = model(token_ids)

    expected = reference_gpt2_logits(hf_state_dict, TINY_CONFIG, token_ids)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_tied_model_loads_the_same_weights():
    """A model with tie_embeddings loads GPT-2 and computes the same logits."""
    hf_state_dict = make_hf_state_dict(TINY_CONFIG)
    model = GPTModel(TINY_CONFIG, tie_embeddings=True).eval()
    load_hf_gpt2_weights(model, hf_state_dict)
    token_ids = torch.tensor([[1, 5, 2, 9, 3]])

    with torch.no_grad():
        logits = model(token_ids)

    assert model.out_head.weight is model.tok_emb.weight
    expected = reference_gpt2_logits(hf_state_dict, TINY_CONFIG, token_ids)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_conversion_splits_and_transposes_the_fused_projection():
    """c_attn is split into query, key, and value, and transposed for nn.Linear."""
    hf_state_dict = make_hf_state_dict(TINY_CONFIG)

    converted = convert_hf_gpt2_state_dict(hf_state_dict, n_layers=2)

    fused = hf_state_dict["h.0.attn.c_attn.weight"]
    assert torch.equal(converted["trf_blocks.0.att.W_query.weight"], fused[:, :8].T)
    assert torch.equal(converted["trf_blocks.0.att.W_value.weight"], fused[:, 16:].T)
    assert torch.equal(converted["out_head.weight"], hf_state_dict["wte.weight"])


def test_conversion_accepts_the_transformer_prefix():
    """Checkpoints saved from GPT2LMHeadModel prefix the names with transformer."""
    prefixed = {
        f"transformer.{k}": t for k, t in make_hf_state_dict(TINY_CONFIG).items()
    }

    converted = convert_hf_gpt2_state_dict(prefixed, n_layers=2)

    assert "trf_blocks.1.norm2.shift" in converted


def test_missing_tensor_raises_value_error():
    """A checkpoint without a required tensor is rejected."""
    hf_state_dict = make_hf_state_dict(TINY_CONFIG)
    del hf_state_dict["h.1.mlp.c_fc.bias"]

    with pytest.raises(ValueError, match="h.1.mlp.c_fc.bias"):
        load_hf_gpt2_weights(GPTModel(TINY_CONFIG), hf_state_dict)


def test_model_without_qkv_bias_is_rejected():
    """GPT-2 has query, key, and value biases: the model needs qkv_bias=True."""
    model = GPTModel({**TINY_CONFIG, "qkv_bias": False})

    with pytest.raises(ValueError, match="unexpected"):
        load_hf_gpt2_weights(model, make_hf_state_dict(TINY_CONFIG))


def fake_urlopen(content):
    """Return a replacement for urllib.request.urlopen that serves content."""

    def urlopen(url, timeout):  # pylint: disable=unused-argument
        return io.BytesIO(content)

    return urlopen


def test_download_writes_and_verifies_the_file(tmp_path, monkeypatch):
    """The file is saved under its final name only after the checksum matches."""
    content = b"fake safetensors content"
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(content))

    path = download_gpt2_small_weights(
        tmp_path, expected_sha256=hashlib.sha256(content).hexdigest()
    )

    assert path == tmp_path / WEIGHTS_FILE
    assert path.read_bytes() == content
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_a_wrong_checksum(tmp_path, monkeypatch):
    """A corrupted download raises an error and leaves no file behind."""
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(b"corrupted"))

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download_gpt2_small_weights(tmp_path, expected_sha256="0" * 64)

    assert not list(tmp_path.iterdir())


def test_download_reuses_a_valid_file(tmp_path, monkeypatch):
    """A file with the expected checksum is not downloaded again."""
    content = b"already downloaded"
    (tmp_path / WEIGHTS_FILE).write_bytes(content)

    def no_network(url, timeout):
        raise AssertionError("The file must not be downloaded again.")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)

    path = download_gpt2_small_weights(
        tmp_path, expected_sha256=hashlib.sha256(content).hexdigest()
    )

    assert path.read_bytes() == content


@pytest.mark.network
def test_hugging_face_publishes_the_expected_file():
    """The pinned revision still lists the file with the expected size and checksum.

    It queries the metadata API of Hugging Face: the weights are not downloaded.
    """
    url = (
        f"https://huggingface.co/api/models/{gpt2_weights.HF_REPOSITORY}"
        f"/revision/{gpt2_weights.HF_REVISION}?blobs=true"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)

    files = {item["rfilename"]: item for item in metadata["siblings"]}
    weights = files[WEIGHTS_FILE]
    assert weights["size"] == gpt2_weights.WEIGHTS_SIZE
    assert weights["lfs"]["sha256"] == gpt2_weights.WEIGHTS_SHA256
