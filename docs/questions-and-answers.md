# Questions and answers

Short answers to questions that come up while building the model, organized
by chapter. Each answer aims to fit in about half a page: it gives the key
idea, not the full theory.

## Contents

* [Chapter 1: text data and tokenization](#chapter-1-text-data-and-tokenization)
  * [Why do we need positional embeddings, if the tokens already arrive in order?](#why-do-we-need-positional-embeddings-if-the-tokens-already-arrive-in-order)
* [Chapter 2: attention](#chapter-2-attention)
  * [Why are the attention weights normalized with softmax?](#why-are-the-attention-weights-normalized-with-softmax)

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
