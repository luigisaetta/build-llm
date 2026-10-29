# Chapter 3: the GPT model architecture

This chapter assembles the components of a GPT-2-like model. It starts from a
dummy model that already has the full data flow, from token IDs to logits,
with placeholders for the transformer block and the layer normalization. The
real components replace the placeholders in the next steps.

The notebook follows the dummy GPT example in Chapter 4 of Sebastian
Raschka's *Build a Large Language Model (From Scratch)* and its
[companion notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb).
The chapter numbering in this repository differs from the book.

## Notebooks

1. **[A dummy GPT model](dummy_gpt.ipynb)**: define the GPT-2 Small
   configuration and a `DummyGPTModel` with token and position embeddings,
   dropout, a stack of placeholder transformer blocks, a placeholder final
   normalization, and the output head. Then count the parameters actually
   present in the dummy model, and estimate those of the complete
   architecture with `build_llm.model_parameters`: about 163M parameters with
   separate input and output weights, and the familiar 124M of GPT-2 Small
   with tied embeddings and query, key, and value biases.

## Run the notebook

Follow the [Quickstart](../QUICKSTART.md) to create the `build-llm` Conda
environment, install the dependencies, and install the repository in editable
mode. The notebook imports `build_llm`, so the editable installation is
required.

From the repository root:

```bash
conda activate build-llm
```

```bash
python -m jupyterlab
```

Open `dummy_gpt.ipynb` with the Python kernel from the `build-llm`
environment, and run it from top to bottom. If the kernel cannot import
`build_llm`, restart it and run all cells again.

The model is created on PyTorch's `meta` device, which records tensor shapes
without allocating the weights, so counting the parameters of a 124M-class
model needs almost no memory. The notebook requires no datasets, downloaded
weights, or network access.

## Follow the tensor shapes

| Step | Shape |
| --- | --- |
| Input token IDs | `(batch_size, seq_len)` |
| Token embeddings | `(batch_size, seq_len, emb_dim)` |
| Position embeddings | `(seq_len, emb_dim)`, broadcast over the batch |
| Output of each transformer block | `(batch_size, seq_len, emb_dim)` |
| Logits | `(batch_size, seq_len, vocab_size)` |

`seq_len` cannot exceed `context_length`, the number of rows of the position
embedding table. The logits are scores, not probabilities: softmax turns the
logits at one position into the distribution of the next token.

## Scope and next steps

The transformer blocks and the final layer normalization are placeholders
that return their input, so the dummy model is not a working language model:
it has no attention and its weights are untrained.

The next steps replace the placeholders with real components: layer
normalization, the feed-forward network with GELU, residual connections, and
the transformer block built around the multi-head attention of
[Chapter 2](../chap2/README.md).

## Development checks

From the repository root, with the `build-llm` environment active:

```bash
black .
```

```bash
pylint build_llm chap* tests
```

```bash
pytest
```

The pytest suite tests `build_llm.model_parameters`; it does not execute the
notebook. Verify the notebook separately with a fresh kernel and
**Run All Cells**.
