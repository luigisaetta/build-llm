"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: Apache-2.0
Description: Reusable building blocks of a GPT-2-like model: multi-head causal
    attention, layer normalization, GELU, feed-forward network, transformer
    block, the complete GPT model, and greedy text generation.

Origin: derived from the code of Chapters 3 and 4 of LLMs-from-scratch
(https://github.com/rasbt/LLMs-from-scratch), Copyright 2023-2026 Sebastian
Raschka, licensed under the Apache License 2.0 (see
LICENSES/Apache-2.0-LLMs-from-scratch.txt). Modified by L. Saetta (2026):
docstrings, type hints, and input validation added, comments corrected.

Addition not in the book: the tie_embeddings argument of GPTModel (default
False). When True, the output head reuses the token embedding matrix, as in
the original GPT-2, which removes vocab_size * emb_dim parameters. It is a
constructor argument, not a field of the configuration dictionary, so the
configurations of the book keep their meaning. With the default, GPTModel
behaves exactly as in the book.

With tie_embeddings=True, the token and position embeddings are also
initialized as in GPT-2, from a normal distribution with standard deviation
0.02 (initializer_range in the GPT-2 config.json). The default initialization
of nn.Embedding, standard deviation 1, suits an input table, but as the weight
of the output head it gives logits about 50 times too large: with
GPT_CONFIG_SMALL_256 the initial loss would be about 475 instead of about 11,
that is ln(vocab_size).

All the modules work on tensors of shape (batch_size, num_tokens, emb_dim),
except GPTModel, which maps token IDs to logits. The attribute names follow the
book (for example W_query and trf_blocks), so that the state_dict keys match
its code and the weight-loading functions built on it.
"""

from collections.abc import Mapping

import torch
from torch import nn

# Standard deviation of the GPT-2 embedding initialization (initializer_range
# in its config.json). Used by GPTModel when tie_embeddings is True.
EMBEDDING_INIT_STD = 0.02


# pylint: disable=invalid-name,too-many-instance-attributes
class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention, with all the heads computed at once.

    The queries, keys, and values are projected to d_out features, split into
    num_heads heads of d_out // num_heads features, and combined again by an
    output projection. A causal mask prevents each token from attending to the
    tokens after it.

    Args:
        d_in: Number of input features per token.
        d_out: Number of output features per token; must be divisible by
            num_heads.
        context_length: Maximum number of tokens; sets the size of the mask.
        dropout: Dropout probability applied to the attention weights.
        num_heads: Number of attention heads.
        qkv_bias: Whether the query, key, and value projections have a bias.

    Raises:
        ValueError: If d_out is not divisible by num_heads.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads.")

        self.d_out = d_out
        self.num_heads = num_heads
        # The size of each head: d_out is split evenly across the heads.
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Combines the outputs of the heads; GPT-2 has this layer too.
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # 1 above the diagonal marks the future tokens that must be hidden.
        # A buffer moves with the model to its device, but is not trained.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the context vectors.

        Args:
            x: Input of shape (b, num_tokens, d_in), with
                num_tokens <= context_length.

        Returns:
            Context vectors of shape (b, num_tokens, d_out).
        """
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the last dimension into heads:
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Move the heads forward, to compute them as a batch:
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Dot products of every query with every key, for each head:
        # (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Hide the future tokens: exp(-inf) = 0, so softmax gives them weight 0
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale by sqrt(head_dim), then normalize each row over the keys
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # Shape: (b, num_tokens, d_out)

        return context_vec


# pylint: enable=invalid-name,too-many-instance-attributes


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, with trainable scale and shift.

    Each token vector is normalized on its own to mean 0 and variance 1. The
    variance is the biased one (divided by n, not n - 1), as in GPT-2 and
    torch.nn.LayerNorm.

    Args:
        emb_dim: Size of the last dimension of the input.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5  # Avoids a division by zero for constant vectors
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input.

        Args:
            x: Input of shape (..., emb_dim).

        Returns:
            The normalized tensor, with the same shape as x.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU activation, with the tanh approximation used by the original GPT-2.

    It gives the same result as torch.nn.GELU(approximate="tanh").
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU element-wise.

        Args:
            x: Input tensor of any shape.

        Returns:
            A tensor with the same shape as x.
        """
        # tanh approximation of x * Phi(x), where Phi is the normal CDF
        coefficient = torch.sqrt(torch.tensor(2.0 / torch.pi))
        inner = coefficient * (x + 0.044715 * torch.pow(x, 3))
        return 0.5 * x * (1 + torch.tanh(inner))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear, GELU, Linear.

    Each token is processed on its own: the vector is expanded from emb_dim to
    4 * emb_dim features, and then contracted back to emb_dim.

    Args:
        cfg: Model configuration; uses emb_dim.
    """

    def __init__(self, cfg: Mapping):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand: D -> 4D
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Contract: 4D -> D
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network.

        Args:
            x: Input of shape (batch_size, num_tokens, emb_dim).

        Returns:
            A tensor of shape (batch_size, num_tokens, emb_dim).
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """A GPT-2 transformer block, with pre-LayerNorm and residual connections.

    The block computes:
        x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
        x = x + Dropout(FeedForward(LayerNorm(x)))

    Args:
        cfg: Model configuration; uses emb_dim, context_length, n_heads,
            drop_rate, and qkv_bias.
    """

    def __init__(self, cfg: Mapping):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input of shape (batch_size, num_tokens, emb_dim).

        Returns:
            A tensor of shape (batch_size, num_tokens, emb_dim).
        """
        # Shortcut (residual) connection around the attention sub-layer
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape: (batch_size, num_tokens, emb_dim)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut (residual) connection around the feed-forward sub-layer
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """A GPT-2-like model, from token IDs to next-token logits.

    Token and position embeddings, dropout, a stack of n_layers transformer
    blocks, a final layer normalization, and a linear output head without
    bias. By default the output head has its own weights, as in the book.

    Args:
        cfg: Model configuration with vocab_size, context_length, emb_dim,
            n_heads, n_layers, drop_rate, and qkv_bias.
        tie_embeddings: Addition to the book. If True, the output head and the
            token embeddings share one (vocab_size, emb_dim) weight tensor
            (weight tying, as in GPT-2): the model has vocab_size * emb_dim
            fewer parameters, and training updates both layers together.
            The token and position embeddings are then initialized with
            standard deviation EMBEDDING_INIT_STD, as in GPT-2.
    """

    def __init__(self, cfg: Mapping, tie_embeddings: bool = False):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        if tie_embeddings:
            # Both matrices have shape (vocab_size, emb_dim): nn.Embedding maps
            # token IDs to rows, and nn.Linear stores (out_features,
            # in_features) = (vocab_size, emb_dim). Assigning the Parameter
            # makes the two layers share one tensor; the separate weight
            # created by nn.Linear is discarded.
            self.out_head.weight = self.tok_emb.weight
            # The shared matrix now also produces the logits, so it needs the
            # small GPT-2 initialization; position embeddings get the same
            # scale, so that tokens and positions start balanced.
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=EMBEDDING_INIT_STD)
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=EMBEDDING_INIT_STD)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Compute the logits.

        Args:
            in_idx: Token IDs of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).

        Raises:
            ValueError: If seq_len is larger than context_length.
        """
        seq_len = in_idx.shape[1]
        context_length = self.pos_emb.num_embeddings
        if seq_len > context_length:
            raise ValueError(
                f"The input has {seq_len} tokens, but the context length "
                f"is {context_length}."
            )

        tok_embeds = self.tok_emb(in_idx)  # Shape: (batch_size, seq_len, emb_dim)
        # One vector per position, broadcast over the batch: (seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape: (batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # Shape: (batch_size, seq_len, vocab_size)
        return logits


def generate_text_simple(
    model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """Extend token sequences with greedy decoding.

    At each step, the model reads the last context_size tokens, and the token
    with the largest logit is appended. Gradients are disabled. The function
    does not change the mode of the model: call model.eval() first, otherwise
    dropout makes the output random.

    Args:
        model: A model that maps (batch_size, num_tokens) token IDs to
            (batch_size, num_tokens, vocab_size) logits, such as GPTModel.
        idx: Token IDs of the current context, of shape (batch_size, num_tokens),
            on the same device as the model.
        max_new_tokens: Number of tokens to generate.
        context_size: Maximum number of tokens that the model reads; at most the
            context_length of the model.

    Returns:
        Token IDs of shape (batch_size, num_tokens + max_new_tokens): the input
        followed by the generated tokens.
    """
    for _ in range(max_new_tokens):
        # Crop the context if it exceeds the supported size: for example, if the
        # model supports only 5 tokens and the context has 10, only the last 5
        # tokens are used
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last position, which predicts the next token:
        # (batch_size, num_tokens, vocab_size) -> (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # The token with the largest logit; softmax would not change the choice
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Append the new token to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens + 1)

    return idx
