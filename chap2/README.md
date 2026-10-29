# Chapter 2: understanding self-attention

Self-attention combines information from different tokens to produce one
context vector per token. This chapter builds the calculation in PyTorch,
starting with fixed input vectors, then introducing trainable query, key,
and value projections, and finally the causal mask, dropout, and batches
used in a GPT model.

The notebooks use a six-token sequence with three-dimensional illustrative
embeddings, so that intermediate tensors remain small enough to inspect.
They follow the attention examples in Chapter 3 of Sebastian Raschka's
*Build a Large Language Model (From Scratch)* and its
[companion notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb).
The chapter numbering in this repository differs from the book.

## Notebooks and reading order

1. **[Simplified self-attention](basic_attention.ipynb)**: use the input
   vectors directly as queries, keys, and values. Compute pairwise dot-product
   scores, normalize each row with softmax, and combine values into context
   vectors. Inspect one token's weighted sum and compare it with the matrix
   calculation. This introductory version has no trainable parameters and
   intentionally omits score scaling.
2. **[Self-attention with trainable weights](attention.ipynb)**: project the
   inputs into queries, keys, and values, scale scores by the square root of
   the key dimension, and compute context vectors. Compare explicit
   `nn.Parameter` matrices with `nn.Linear` layers, inspect tensor shapes,
   and verify that both versions agree when their weights match.
3. **[Causal attention and dropout](causal_attention.ipynb)**: build the
   causal mask step by step, so that each token attends only to itself and
   to earlier tokens, and apply dropout to the attention weights. Wrap
   everything in a `CausalAttention` class that processes batches, then
   check that future tokens do not influence earlier outputs, and compare
   training and evaluation mode.

## Run the notebooks

Follow the [Quickstart](../QUICKSTART.md) to create the `build-llm` Conda
environment and install the runtime and development dependencies, including
JupyterLab.

From the repository root:

```bash
conda activate build-llm
python -m jupyterlab
```

Open `chap2/basic_attention.ipynb`, then `chap2/attention.ipynb`, then
`chap2/causal_attention.ipynb`. Use the Python kernel from the `build-llm`
environment and run each notebook from top to bottom. Restart the kernel and
run all cells when checking changes. Each notebook is independent of the
others.

The examples run on CPU and require no datasets, downloaded weights, or
network access after the dependencies are installed.

## Follow the tensor shapes

In the trainable version, the input has one row per token. Each projection
maps `d_in` input features to `d_out` features:

| Tensor | Shape |
| --- | --- |
| Input embeddings | `(num_tokens, d_in)` |
| Queries, keys, values | `(num_tokens, d_out)` |
| Attention scores | `(num_tokens, num_tokens)` |
| Attention weights | `(num_tokens, num_tokens)` |
| Output context vectors | `(num_tokens, d_out)` |

Each score row belongs to one query and each column to one key. Softmax
normalizes across the keys, so each row of attention weights sums to one.
Multiplying those weights by the values produces one output vector per
token. In the simplified version, the projections are omitted and the input
embedding dimension is also the output dimension.

## Scope and next steps

The first two notebooks are single-head, unmasked examples for one sequence
without a batch dimension: a token can attend to later tokens.
`causal_attention.ipynb` adds the causal mask, dropout, and the batch
dimension, with inputs of shape `(b, num_tokens, d_in)`. All the examples are
single-head, and have no positional encoding or final output projection.

The projection matrices in `attention.ipynb` are trainable, but the notebook
does not train them: there is no loss, optimizer, or parameter update. The
displayed outputs come from randomly initialized weights. Attention weights
are computed from the current inputs and parameters; they are not separate
trainable parameters.

The next step is multi-head attention: several causal attention heads in
parallel, combined by a final linear projection.

## Development checks

From the repository root, with the `build-llm` environment active:

```bash
black .
pylint build_llm chap* tests
pytest
```

Black includes notebook formatting through the `black[jupyter]` development
dependency. The current pytest suite covers the existing package and chapter
scripts; it does not execute these notebooks. Verify the notebooks separately
with a fresh kernel and **Run All Cells**.
