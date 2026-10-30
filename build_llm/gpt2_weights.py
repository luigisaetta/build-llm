"""
Author: L. Saetta
Date Last Modified: 2026-10-30
License: MIT
Description: Download the public GPT-2 Small weights from Hugging Face, in the
    safetensors format, and load them into build_llm.gpt2.GPTModel.

The weights come from https://huggingface.co/openai-community/gpt2. They are
read with the safetensors library, so neither TensorFlow nor the transformers
library is needed. The Hugging Face tensor names and layouts differ from those
of GPTModel: convert_hf_gpt2_state_dict documents the mapping.
"""

import hashlib
import shutil
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file

from build_llm.gpt2 import GPTModel

# Configuration of the pretrained GPT-2 Small, matching config.json of the
# Hugging Face model. It has the same dimensions as GPT_CONFIG_SMALL of the
# Chapter 3 notebooks, but qkv_bias is True: GPT-2 has these biases.
GPT2_SMALL_PRETRAINED_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

HF_REPOSITORY = "openai-community/gpt2"
WEIGHTS_FILE = "model.safetensors"
# A fixed revision of the repository, so that the downloaded file never changes.
HF_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHTS_URL = (
    f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}/{WEIGHTS_FILE}"
)
# Size and SHA-256 published by Hugging Face for this file (checked 2026-10-30).
WEIGHTS_SIZE = 548_105_171
WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"

_CHUNK_SIZE = 1024 * 1024


def default_weights_dir() -> Path:
    """Return the folder for the downloaded weights: models/gpt2-small.

    The folder is in the repository root and is ignored by Git.
    """
    return Path(__file__).resolve().parent.parent / "models" / "gpt2-small"


def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 checksum of a file, reading it in chunks.

    Args:
        path: The file to hash.

    Returns:
        The checksum as a lowercase hexadecimal string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_gpt2_small_weights(
    target_dir: Path | None = None,
    *,
    url: str = WEIGHTS_URL,
    expected_sha256: str = WEIGHTS_SHA256,
    timeout: float = 60.0,
) -> Path:
    """Download model.safetensors of GPT-2 Small, unless a valid copy exists.

    The file is first written with a .part suffix and renamed only after its
    checksum is verified, so an interrupted download never leaves a corrupted
    file under the final name.

    Args:
        target_dir: Destination folder; default_weights_dir() if None.
        url: Address of the file.
        expected_sha256: Checksum that the file must have.
        timeout: Timeout in seconds for the network connection.

    Returns:
        The path of the verified file.

    Raises:
        RuntimeError: If the downloaded file has a different checksum.
        urllib.error.URLError: If the download fails.
    """
    target_dir = default_weights_dir() if target_dir is None else Path(target_dir)
    target = target_dir / WEIGHTS_FILE
    if target.exists() and sha256_of_file(target) == expected_sha256:
        return target

    target_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with partial.open("wb") as file:
            shutil.copyfileobj(response, file, _CHUNK_SIZE)

    actual_sha256 = sha256_of_file(partial)
    if actual_sha256 != expected_sha256:
        partial.unlink()
        raise RuntimeError(
            f"Checksum mismatch for {url}: expected {expected_sha256}, "
            f"got {actual_sha256}."
        )
    partial.replace(target)
    return target


def load_hf_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Read a safetensors file into a dictionary of CPU tensors.

    Args:
        path: Path of the .safetensors file.

    Returns:
        The tensors, keyed by their Hugging Face names.
    """
    return load_file(str(path), device="cpu")


def _get(hf_state_dict: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return a tensor, accepting names with or without the transformer. prefix."""
    for key in (name, f"transformer.{name}"):
        if key in hf_state_dict:
            return hf_state_dict[key]
    raise ValueError(f"Missing tensor in the GPT-2 checkpoint: {name}.")


def convert_hf_gpt2_state_dict(
    hf_state_dict: Mapping[str, torch.Tensor], n_layers: int
) -> dict[str, torch.Tensor]:
    """Rename and reshape Hugging Face GPT-2 tensors for GPTModel.

    Three differences need a conversion:

    * Names: for example h.0.ln_1.weight becomes trf_blocks.0.norm1.scale.
    * Layout: GPT-2 uses Conv1D layers, with weights of shape
      (in_features, out_features); nn.Linear stores (out_features,
      in_features), so every projection weight is transposed.
    * Fused projections: c_attn holds the query, key, and value projections in
      one (emb_dim, 3 * emb_dim) matrix, which is split into three.

    GPT-2 ties the output head to the token embeddings, so the checkpoint has
    no separate output weight: wte.weight is used for both.

    Args:
        hf_state_dict: Tensors of the Hugging Face checkpoint.
        n_layers: Number of transformer blocks.

    Returns:
        A state dict with the keys of GPTModel, except the causal mask buffers.

    Raises:
        ValueError: If a required tensor is missing.
    """
    converted = {
        "tok_emb.weight": _get(hf_state_dict, "wte.weight"),
        "pos_emb.weight": _get(hf_state_dict, "wpe.weight"),
        "final_norm.scale": _get(hf_state_dict, "ln_f.weight"),
        "final_norm.shift": _get(hf_state_dict, "ln_f.bias"),
        # Weight tying in GPT-2: the output head reuses the token embeddings.
        "out_head.weight": _get(hf_state_dict, "wte.weight"),
    }

    for i in range(n_layers):
        hf = f"h.{i}."
        ours = f"trf_blocks.{i}."

        # Split the fused projection along the output features: q, k, v.
        q_w, k_w, v_w = _get(hf_state_dict, hf + "attn.c_attn.weight").chunk(3, dim=1)
        q_b, k_b, v_b = _get(hf_state_dict, hf + "attn.c_attn.bias").chunk(3, dim=0)
        converted |= {
            ours + "att.W_query.weight": q_w.T,
            ours + "att.W_key.weight": k_w.T,
            ours + "att.W_value.weight": v_w.T,
            ours + "att.W_query.bias": q_b,
            ours + "att.W_key.bias": k_b,
            ours + "att.W_value.bias": v_b,
            ours
            + "att.out_proj.weight": _get(hf_state_dict, hf + "attn.c_proj.weight").T,
            ours + "att.out_proj.bias": _get(hf_state_dict, hf + "attn.c_proj.bias"),
            ours + "ff.layers.0.weight": _get(hf_state_dict, hf + "mlp.c_fc.weight").T,
            ours + "ff.layers.0.bias": _get(hf_state_dict, hf + "mlp.c_fc.bias"),
            ours
            + "ff.layers.2.weight": _get(hf_state_dict, hf + "mlp.c_proj.weight").T,
            ours + "ff.layers.2.bias": _get(hf_state_dict, hf + "mlp.c_proj.bias"),
            ours + "norm1.scale": _get(hf_state_dict, hf + "ln_1.weight"),
            ours + "norm1.shift": _get(hf_state_dict, hf + "ln_1.bias"),
            ours + "norm2.scale": _get(hf_state_dict, hf + "ln_2.weight"),
            ours + "norm2.shift": _get(hf_state_dict, hf + "ln_2.bias"),
        }
    return converted


def load_hf_gpt2_weights(
    model: GPTModel, hf_state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Copy Hugging Face GPT-2 weights into a GPTModel, in place.

    The model must have the same configuration as the checkpoint, including
    qkv_bias=True. load_state_dict checks every shape and copies the values to
    the device of the model.

    Args:
        model: The model to fill.
        hf_state_dict: Tensors of the Hugging Face checkpoint.

    Raises:
        ValueError: If a tensor is missing, or the checkpoint does not cover
            all the parameters of the model.
        RuntimeError: If a tensor shape does not match the model.
    """
    converted = convert_hf_gpt2_state_dict(hf_state_dict, len(model.trf_blocks))
    result = model.load_state_dict(converted, strict=False)

    # Only the causal masks are not in the checkpoint: the model computes them.
    missing = [key for key in result.missing_keys if not key.endswith(".att.mask")]
    if missing or result.unexpected_keys:
        raise ValueError(
            f"Checkpoint and model do not match: missing {missing}, "
            f"unexpected {result.unexpected_keys}."
        )
