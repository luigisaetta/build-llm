# How AI assistants helped build this repository

This repository is written and maintained by L. Saetta. Two AI coding
assistants were used along the way, to review and improve its material:

* **Codex**, by OpenAI;
* **Claude**, by Anthropic, through Claude Code.

This page explains how they were used, and what they did not do, so that
readers know how the material was produced.

## The author stays in charge

The assistants were tools, not authors. The author decided the plan of the
project, the topics, and the order of the chapters. The author also asked for
each change, reviewed the result, and chose what to keep. Every commit was
made by the author.

The assistants worked under the same rules as human contributors. These rules
are in [AGENTS.md](AGENTS.md): code style, tests, portability, licensing, and
attribution of third-party material. As AGENTS.md requires, no assistant is
listed as an author, co-author, or contributor in files or commit messages.

## What the assistants were used for

**Reviewing code and notebooks.** The assistants read the notebooks and the
code to find errors, imprecise statements, and comments that did not match
the code. For example, a comment said that the output layer produces "the
distribution" of the tokens, but it produces logits. A module header declared
the MIT License for code derived from Apache-2.0 material.

**Checking correctness.** The assistants did not rely on reading the code
alone; they also checked it with code that runs. They compared the
implementations with PyTorch's reference functions:

* layer normalization with `torch.nn.functional.layer_norm`;
* GELU with `torch.nn.functional.gelu`;
* multi-head attention with `scaled_dot_product_attention`.

They also checked that the attention is causal, that the parameter counts
match the analytical formulas, and that the results on the Apple Silicon GPU
(MPS) match the CPU. The pretrained GPT-2 weights loaded into our model were
compared with an independent implementation of GPT-2.

**Improving the explanations.** The assistants helped write and extend the
explanatory cells of the notebooks, the docstrings, and the short answers in
[docs/questions-and-answers.md](docs/questions-and-answers.md). The goal was
to explain each step, each tensor shape, and each non-obvious decision.

**Writing reusable code and tests.** Some building blocks of the `build_llm`
package and their `pytest` tests were written with the assistants, following
AGENTS.md. One example is the download and conversion of the GPT-2 weights
in `build_llm/gpt2_weights.py`.

**Documentation and consistency.** The assistants helped write the READMEs
and the Quickstart, and kept the roadmap, the chapter tables, and the lists of
derived files aligned with the code.

**Licensing and attribution.** The assistants checked that every file derived
from [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) states
its origin and keeps the Apache License 2.0. They also checked that each such
file is listed in [NOTICE](NOTICE).

**Figures.** The architecture diagram in `images/gpt-architecture.png` was
generated with matplotlib, from a script written with an assistant.

**Routine checks.** The assistants ran Black, Pylint, and pytest. They also
ran the notebooks again from a clean kernel after each change.

## What this means for the reader

* The code and the explanations were reviewed and tested, but they can still
  contain mistakes. AI assistants can be wrong in a confident tone, and so can
  humans. If you find an error, please report it: see
  [CONTRIBUTING.md](CONTRIBUTING.md).
* The assistants did not copy text or figures from the book. This repository
  does not include them.
* Material derived from LLMs-from-scratch is attributed to its author,
  Sebastian Raschka, whatever tool was used to adapt it.
