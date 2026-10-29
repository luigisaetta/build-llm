"""
Author: L. Saetta
Date Last Modified: 2026-10-29
License: MIT
Description: Exact parameter counts and configuration-based estimates for GPT models.
"""

from collections.abc import Mapping

from torch import nn


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count scalar parameters in any initialized PyTorch module.

    Shared Parameter objects are counted once, including tied embedding and
    output weights. Buffers, activations, and optimizer state are not parameters.
    This counts the model as implemented, including any empty placeholders.

    Args:
        model: An initialized module, on any device, including the meta device.
        trainable_only: Whether to include only parameters with requires_grad.

    Returns:
        The number of scalar parameters, rather than their memory size in bytes.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _integer(config: Mapping, name: str, *, minimum: int = 1) -> int:
    """Read a required integer dimension and reject invalid values."""
    if name not in config:
        raise ValueError(f"Missing configuration field: {name}.")
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _flag(config: Mapping, name: str, default: bool) -> bool:
    """Read an optional boolean architecture setting."""
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def estimate_gpt_parameters(config: Mapping[str, int | float | bool]) -> dict[str, int]:
    """Estimate a standard GPT architecture without allocating model tensors.

    Assumes learned token and absolute position embeddings, independent blocks,
    full-width Q/K/V projections plus an attention output projection, two linear
    feed-forward layers, two affine LayerNorms per block, and a final LayerNorm.
    It does not describe grouped-query attention, gated MLPs, RMSNorm, adapters,
    or parameter sharing across blocks. For other architectures use
    count_parameters on the actual module, optionally created on the meta device.

    Args:
        config: Required positive integer fields: vocab_size, context_length,
            emb_dim; n_layers is a nonnegative integer. Optional ff_hidden_dim
            defaults to 4 * emb_dim. Optional n_heads must divide emb_dim; at
            fixed emb_dim it does not change the standard attention count.
            Boolean options: qkv_bias (False), attention_output_bias (True),
            mlp_bias (True), layer_norm_bias (True), output_bias (False), and
            tie_embeddings (False). Tying reuses the input embedding Parameter
            as the output weight. Extra fields such as drop_rate are ignored.

    Returns:
        Counts for token_embeddings, position_embeddings, attention, feed_forward,
        block_norms, final_norm, output_head, and total. Block-related entries
        include all n_layers blocks. A tied output weight adds no parameters,
        but its optional output bias is still counted.

    Raises:
        ValueError: If a required field is absent or a supported field is invalid.
    """
    # Keep dimensions together: the formulas below mirror the component shapes.
    vocab = _integer(config, "vocab_size")
    context = _integer(config, "context_length")
    width = _integer(config, "emb_dim")
    layers = _integer(config, "n_layers", minimum=0)
    hidden = _integer(
        {"ff_hidden_dim": config.get("ff_hidden_dim", 4 * width)}, "ff_hidden_dim"
    )
    if "n_heads" in config and width % _integer(config, "n_heads") != 0:
        raise ValueError("emb_dim must be divisible by n_heads.")

    # Q/K/V each have (width, width) weights; the output projection adds a fourth.
    attention = 4 * width * width
    attention += 3 * width * _flag(config, "qkv_bias", False)
    attention += width * _flag(config, "attention_output_bias", True)
    # The MLP expands width -> hidden, then contracts hidden -> width.
    feed_forward = 2 * width * hidden
    feed_forward += (hidden + width) * _flag(config, "mlp_bias", True)
    # Each affine LayerNorm has a scale vector and an optional shift vector.
    norm = width * (1 + _flag(config, "layer_norm_bias", True))
    output_head = 0 if _flag(config, "tie_embeddings", False) else vocab * width
    output_head += vocab * _flag(config, "output_bias", False)

    counts = {
        "token_embeddings": vocab * width,
        "position_embeddings": context * width,
        "attention": layers * attention,
        "feed_forward": layers * feed_forward,
        "block_norms": layers * 2 * norm,
        "final_norm": norm,
        "output_head": output_head,
    }
    counts["total"] = sum(counts.values())
    return counts
