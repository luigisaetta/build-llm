# Questions and answers

Short answers to questions that come up while building the model, organized
by chapter. Each answer aims to fit in about half a page: it gives the key
idea, not the full theory.

## Contents

* [Chapter 0: the environment](#chapter-0-the-environment)
  * [What is MPS?](#what-is-mps)
* [Chapter 1: text data and tokenization](#chapter-1-text-data-and-tokenization)
  * [Why do we need positional embeddings, if the tokens already arrive in order?](#why-do-we-need-positional-embeddings-if-the-tokens-already-arrive-in-order)
  * [Why does the notebook dataloader.ipynb use an embedding size of 256?](#why-does-the-notebook-dataloaderipynb-use-an-embedding-size-of-256)
* [Chapter 2: attention](#chapter-2-attention)
  * [Why are the attention weights normalized with softmax?](#why-are-the-attention-weights-normalized-with-softmax)
  * [Can the SelfAttention_v1 and SelfAttention_v2 classes be stacked as layers to build the LLM?](#can-the-selfattention_v1-and-selfattention_v2-classes-be-stacked-as-layers-to-build-the-llm)
  * [What are the advantages of SelfAttention_v2 over SelfAttention_v1?](#what-are-the-advantages-of-selfattention_v2-over-selfattention_v1)
* [Chapter 3: the GPT model architecture](#chapter-3-the-gpt-model-architecture)
  * [What are the logits produced by the GPT model?](#what-are-the-logits-produced-by-the-gpt-model)
* [Chapter 4: pretrained weights and training](#chapter-4-pretrained-weights-and-training)
  * [What is gradient clipping, and why does it help our training?](#what-is-gradient-clipping-and-why-does-it-help-our-training)
  * [Which learning rate schedules have we implemented, and why?](#which-learning-rate-schedules-have-we-implemented-and-why)

## Chapter 0: the environment

### What is MPS?

MPS (**Metal Performance Shaders**) is Apple's library of GPU-optimized
operations, built on Metal, Apple's GPU programming interface. PyTorch uses
it as a backend called `mps`: on a Mac with Apple Silicon (M1, M2, ...),
tensors and models placed on the `mps` device run on the GPU instead of the
CPU.

```python
device = torch.device("mps")
x = torch.rand(1024, 1024, device=device)  # this tensor lives on the GPU
```

In this project, `build_llm.device.get_device()` selects MPS when it is
available, then CUDA, then CPU, and `python -m chap0.check_environment`
checks that MPS computes the same results as the CPU.

Good to know:

* **Unified memory.** On Apple Silicon, CPU and GPU share the same memory, so
  there is no separate GPU memory to fill. Tensors must still be moved to the
  device with `.to(device)`.
* **Not every operation is supported.** Some PyTorch operations have no MPS
  version yet. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` runs them on the CPU
  instead of raising an error.
* **No 64-bit floats.** MPS does not support `float64`; use `float32` (the
  PyTorch default) or lower precision.
* **Requirements.** macOS 12.3 or later, and a PyTorch build with MPS
  support (the standard macOS builds have it).

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

### What are the advantages of SelfAttention_v2 over SelfAttention_v1?

Both classes in [`chap2/attention.ipynb`](../chap2/attention.ipynb) compute
the same attention: the notebook copies the v1 weights into v2 and gets the
same outputs. The advantages of v2 come from `nn.Linear`:

* **Better initialization.** v1 uses `torch.rand`: only positive values in
  `[0, 1)`, whatever the input size. With realistic sizes (768 dimensions),
  the scores become huge and softmax saturates: each token attends to one
  token only, and the gradients vanish. `nn.Linear` draws zero-centered
  values scaled by `1/sqrt(d_in)`, so attention starts smooth and can be
  trained.
* **Optional bias.** `qkv_bias=True` adds a bias to the projections, as GPT-2
  does: we need it to load the GPT-2 weights.
* **Standard building block.** The code is more idiomatic, and PyTorch tools
  (custom initialization, quantization, LoRA fine-tuning) recognize
  `nn.Linear` layers.

v1 is useful to see the matrix products; v2 is the base for the next steps.

## Chapter 3: the GPT model architecture

### What are the logits produced by the GPT model?

The **logits** are the raw scores that the output head computes for every
token of the vocabulary. In [`chap3/dummy_gpt.ipynb`](../chap3/dummy_gpt.ipynb),
the last layer is a linear layer from `emb_dim` to `vocab_size`:

```python
logits = self.out_head(x)  # (batch_size, seq_len, vocab_size)
```

So, for each position of each sequence, the model returns 50,257 numbers, one
per GPT-2 token. A higher logit means that the model considers that token a
more likely **next token** after the tokens seen so far.

Logits are **not probabilities**: they can be any real number, positive or
negative, and they do not sum to 1. Softmax turns the logits at one position
into the probability distribution of the next token:

```python
probabilities = torch.softmax(logits[:, -1, :], dim=-1)  # (batch_size, vocab_size)
```

Some details:

* **One prediction per position.** The logits at position `t` predict the
  token at position `t + 1`. During training, all the positions are used at
  once; during text generation, only the last one, `logits[:, -1, :]`, is
  needed to choose the next token.
* **Training uses the logits directly.** `torch.nn.functional.cross_entropy`
  takes the logits, not the probabilities: it applies log-softmax internally,
  which is more numerically stable.
* **Generation works on the logits too.** Greedy decoding picks the token with
  the largest logit (softmax does not change the ranking). Temperature divides
  the logits before softmax, and top-k sampling keeps only the `k` largest
  logits.
* **The name.** In statistics, the logit is the logarithm of the odds,
  `log(p / (1 - p))`, the inverse of the sigmoid function. In deep learning,
  the word means, more loosely, the unnormalized scores that go into
  softmax: up to a constant, logits are log-probabilities.

In the dummy model the weights are random and the transformer blocks do
nothing, so its logits are meaningless; the shape, however, is already the
final one.

## Chapter 4: pretrained weights and training

### What is gradient clipping, and why does it help our training?

At each step, the gradient tells the optimizer in which **direction** to move
every parameter, and its size tells **how far**. Gradient clipping limits the
size, not the direction: it computes one norm over all the gradients of the
model together, and if the norm is larger than a threshold, it scales all the
gradients down by the same factor.

```text
norm = sqrt(sum of the squares of all the gradients)
if norm > max_norm:
    every gradient = gradient * max_norm / norm
```

In [`chap4/first_training.py`](../chap4/first_training.py) it is one line,
between `loss.backward()` and `optimizer.step()`, with `max_norm = 1.0`
(`GRAD_CLIP_NORM`), the usual value for GPT models:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
```

Why it helps in our case:

* **Our batches are tiny.** Each step sees 4 sequences of 256 tokens, about
  1,000 tokens; GPT-2 used about half a million. The gradient of a small
  batch is noisy, and now and then an unusual batch (rare words, a long
  dialogue, the start of a new book) gives a gradient many times larger than
  usual.
* **One very large step can undo many small ones.** It moves the weights far
  from where they were improving, and the loss jumps up: a *loss spike*. In
  the worst case the training diverges and the loss becomes `nan`.
* **AdamW does not fully protect against it.** It divides each update by a
  running average of the squared gradients, but with `beta2 = 0.999` that
  average reacts slowly, so a sudden spike still passes through.
* **It changes almost nothing when the training is healthy.** Clipping acts
  only on the rare steps above the threshold.

Two notes. Clipping is a safety net, not a cure: if the norm is clipped at
almost every step, the learning rate is probably too high. And with gradient
accumulation, the clipping goes after the last micro-batch, on the
accumulated gradient, just before the optimizer step.

### Which learning rate schedules have we implemented, and why?

The learning rate is the size of the steps of the optimizer. At first
`chap4/first_training.py` kept it **constant**, at 4e-4, from the first step
to the last. [`build_llm/lr_schedule.py`](../build_llm/lr_schedule.py) adds
two schedules that change it during training; the script now uses the cosine
one. Both start with a **linear warmup**:

* **Warmup.** In the first steps (typically 1–2% of them) the learning rate
  grows linearly from almost 0 to its peak. At the start the weights are
  random, the gradients are large, and the AdamW running averages are not
  reliable yet: small steps avoid unstable updates.

After the warmup:

| Schedule | Shape | Total steps |
| --- | --- | --- |
| `"cosine"` | decreases along half a cosine wave, from the peak to the minimum at the last step | fixed from the start: they set the shape of the whole curve |
| `"wsd"` (warmup, stable, decay) | stays at the peak, then decreases linearly to the minimum in the last part of the steps (`decay_fraction`, for example 20%) | can still change until the decay starts |

Why decrease the learning rate at the end: large steps explore fast but
"bounce" around a good solution; small steps settle into it. Both schedules
usually reach a lower loss than a constant learning rate.

Why two schedules:

* **Cosine** is the classic choice, used for example by GPT-3 and in the
  appendix of the book on training improvements. It needs the length of the
  training from the start.
* **WSD** fits our situation better: on a laptop, an epoch takes about an
  hour, and we may decide to stop early or to train longer. With WSD the run
  can be extended while it is in the stable phase, and from any checkpoint of
  that phase a short decay gives a finished model.

Both schedules are a **pure function of the optimizer step**:

```python
lr = learning_rate(step, schedule="wsd", peak_lr=4e-4, min_lr=4e-5,
                   warmup_steps=1_000, total_steps=53_070)
```

There is no scheduler state to save: a training resumed from a checkpoint
gets exactly the same learning rates as an uninterrupted one, because the
checkpoint already stores the step number. The schedule parameters, and the
number of epochs, are part of the settings checked at resume, so that a run
cannot continue with a different curve by mistake.

Our choice for now is the **cosine**, with 6 epochs: the plan is fixed, so
its limit does not matter, and every end-of-epoch checkpoint is the best
model for that point of the training, which is useful to test it with
`chap4/generate_from_checkpoint.py`.
