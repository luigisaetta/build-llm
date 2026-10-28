# Questions and answers

Short answers to questions that come up while building the model, organized
by chapter. Each answer aims to fit in about half a page: it gives the key
idea, not the full theory.

## Contents

* [Chapter 1: text data and tokenization](#chapter-1-text-data-and-tokenization)
  * [Why do we need positional embeddings, if the tokens already arrive in order?](#why-do-we-need-positional-embeddings-if-the-tokens-already-arrive-in-order)
  * [Why does the notebook dataloader.ipynb use an embedding size of 256?](#why-does-the-notebook-dataloaderipynb-use-an-embedding-size-of-256)
* [Chapter 2: attention](#chapter-2-attention)
  * [Why are the attention weights normalized with softmax?](#why-are-the-attention-weights-normalized-with-softmax)
  * [Can the SelfAttention_v1 and SelfAttention_v2 classes be stacked as layers to build the LLM?](#can-the-selfattention_v1-and-selfattention_v2-classes-be-stacked-as-layers-to-build-the-llm)

## Chapter 1: text data and tokenization

### Why do we need positional embeddings, if the tokens already arrive in order?

The tokens are *stored* in order, but the transformer does not *use* that
order. The tensor of shape `(batch_size, num_tokens, emb_dim)` has an order of
rows, but no layer of the model reads the row index.

Unlike a recurrent network (RNN), a transformer does not read the tokens one
after the other. Self-attention processes all the tokens of a sequence at the
same time. For each token, it computes a weighted average of the vectors of
the other tokens, with weights that depend only on the *content* of the
vectors. Shuffle the input rows, and attention returns the same output
vectors, shuffled in the same way. The feed-forward layers and layer
normalization work on each token separately, so they do not see the order
either.

So, without information about position, "the dog bites the man" and "the man
bites the dog" contain the same tokens and look almost the same to the
model, even though their meaning is different.

Positional embeddings fix this. Each position `0, 1, 2, ...` has its own
vector, which is added to the token embedding:

```python
input_embeddings = token_embeddings + position_embeddings
```

After the sum, the same token has a different vector at position 3 and at
position 10, and the model can learn patterns such as "the token just
before" or "the subject usually comes before the verb".

Some details:

* GPT-2, and our model, use **learned absolute** positional embeddings: a
  table with one trainable vector per position, up to `context_length`.
  Newer models often use **relative** schemes such as RoPE, which encode the
  distance between tokens directly inside attention.
* The vectors are **added**, not concatenated, so the size stays `emb_dim`.
  In a high-dimensional space, the model can learn to keep the two kinds of
  information apart.
* The causal mask of a GPT model gives a weak, indirect hint about position,
  because each token sees only the tokens before it. Explicit positional
  information is still much more effective.

### Why does the notebook dataloader.ipynb use an embedding size of 256?

It is a teaching choice, not a property of the model. The embedding size
(`emb_dim`, also called the model dimension) is a **hyperparameter**: we pick
it when we design the model. 256 is large enough to show realistic tensor
shapes, and small enough to keep the examples fast and the printed tensors
readable.

Real models use larger values, which grow with the size of the model. The
four GPT-2 variants are a good example:

| GPT-2 variant | Parameters | Embedding size | Layers | Attention heads |
| --- | --- | --- | --- | --- |
| Small | 124M | 768 | 12 | 12 |
| Medium | 355M | 1,024 | 24 | 16 |
| Large | 774M | 1,280 | 36 | 20 |
| XL | 1.5B | 1,600 | 48 | 25 |

GPT-3, with 175B parameters, uses 12,288.

What does the size change?

* **Capacity.** Each token is represented by `emb_dim` numbers. More
  dimensions can store more information about the meaning and the role of a
  token, but need more data and more compute to train well.
* **Memory and speed.** The token embedding table alone has
  `vocab_size × emb_dim` parameters. With the GPT-2 vocabulary of 50,257
  tokens, that is about 12.9M parameters with 256 dimensions, and about
  38.6M with 768. The cost of the rest of the model grows with `emb_dim`
  too.
* **Consistency.** The same size runs through the whole model: the
  positional embeddings must have the same size as the token embeddings,
  because they are added together, and the attention layers must split
  `emb_dim` evenly across the heads (768 = 12 × 64 in GPT-2 Small).

Values are usually multiples of 64 (or powers of two), because GPU and MPS
kernels work most efficiently with these sizes.

To load the public GPT-2 weights later (see the roadmap), our GPT model will
need the same configuration as one of the variants, for example GPT-2 Small
with 768 dimensions.

## Chapter 2: attention

### Why are the attention weights normalized with softmax?

The attention scores are dot products between a query and the keys: they can
be any real number, positive or negative, small or large. To use them as the
weights of a weighted average of the value vectors, we need weights that are
**non-negative** and **sum to 1**. Softmax turns any vector of scores into
exactly that:

```text
softmax(s)_i = exp(s_i) / sum_j exp(s_j)
```

Why softmax, and not a simpler normalization such as `s_i / sum_j s_j`?

* **It handles negative scores.** `exp` is always positive, so every weight
  is between 0 and 1. Dividing raw scores by their sum fails with negative
  values, and can even divide by zero.
* **It keeps the ranking.** `exp` is increasing: a higher score always gives
  a higher weight.
* **It focuses attention.** The exponential amplifies the differences
  between scores, so the largest scores get most of the weight. Softmax is
  a smooth, "soft" version of picking the maximum (argmax).
* **It is differentiable everywhere.** Unlike a hard argmax, every token
  receives a gradient, so the model can learn to move its attention
  gradually during training.
* **It works well with masking.** Setting a score to `-inf` gives a weight of
  exactly 0, because `exp(-inf) = 0`. This is how the causal mask hides the
  future tokens.
* **It is numerically stable.** Adding the same constant to all the scores
  does not change the result, so implementations subtract the maximum score
  before computing `exp`, which avoids overflow.

Before softmax, the scores are divided by `sqrt(d_k)`, the square root of the
key size. Dot products grow with the vector size. Without this scaling,
softmax would saturate: one weight close to 1, the others close to 0, and
very small gradients.

Softmax is the standard choice, not the only one. Alternatives such as
sparsemax or sigmoid attention exist, but they are less common.

### Can the SelfAttention_v1 and SelfAttention_v2 classes be stacked as layers to build the LLM?

Not as they are. They are the right building block, but the LLM does not
stack bare attention layers.

Technically, stacking works: both classes are `nn.Module`s that map
`(num_tokens, d_in)` to `(num_tokens, d_out)`, so they can be chained in an
`nn.Sequential` when each `d_out` equals the next `d_in`. But a GPT model needs
more, at two levels.

**Inside the attention layer**, the chapter 2 classes are simplified:

* **No causal mask.** Every token sees the tokens after it. In next-token
  training, the model could read the answer instead of predicting it.
* **No batch dimension.** They use `keys.T`, which is correct only for a 2D
  matrix. With batches of shape `(batch_size, num_tokens, d_in)`, the code
  must use `keys.transpose(-2, -1)`.
* **A single head, and no output projection.** GPT uses multi-head
  attention: several heads in parallel, each looking at different relations
  between tokens, followed by a linear layer that combines their outputs.
* **Initialization.** `SelfAttention_v1` fills its matrices with
  `torch.rand`, which gives only positive values in `[0, 1)` and is a poor
  start for training. `SelfAttention_v2` uses `nn.Linear`, with a better
  initialization, and is the version to build on.

**Around the attention layer**, the unit that GPT stacks is the **transformer
block**, not attention alone:

```text
x = x + MultiHeadAttention(LayerNorm(x))   # tokens exchange information
x = x + FeedForward(LayerNorm(x))          # each token is processed on its own
```

* The **feed-forward network** adds the non-linearity. Attention mostly
  computes weighted averages of linear projections; the feed-forward layers
  transform each token's vector.
* The **residual connections** (`x + ...`) and **layer normalization** keep
  training stable when many blocks are stacked. Without residual
  connections, deep stacks of pure attention tend to make all the token
  vectors converge to the same value, a known problem called *rank collapse*.

GPT-2 Small stacks 12 of these blocks. The next steps of the chapter (causal
masking, batches, multiple heads) turn `SelfAttention_v2` into the multi-head
attention that goes inside each block.
