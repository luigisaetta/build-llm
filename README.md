# build-llm

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen.svg)](https://github.com/pylint-dev/pylint)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Build a small Large Language Model from scratch, one transparent step at a
> time, in plain Python and PyTorch.**

Large Language Models often feel like black boxes. This repository opens the
box. Starting from raw text, it builds every component needed to obtain a
working, GPT-style small LLM: the tokenizer, the data pipeline, the attention
mechanism, the transformer architecture, the pre-training loop, text
generation, evaluation, and fine-tuning.

No magic, no hidden frameworks: each piece is small, readable, tested, and
explained, so that you can understand *why* it works, not only *that* it works.

## Inspiration

This project is inspired by the book
[**Build a Large Language Model (From Scratch)**](https://www.manning.com/books/build-a-large-language-model-from-scratch)
by **Sebastian Raschka** (Manning, 2024), and by its companion repository
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

This is an independent learning project, not an official companion of the
book. The code here is written from scratch, organized as a reusable Python
package with tests, and may diverge from the book in structure and choices.
If you want the full story behind every concept, read the book.

## Why this repository

* **Step by step**: the project grows incrementally; each stage builds on the
  previous one and can be studied on its own.
* **Readable first**: clear names, small functions, docstrings, and comments
  that explain the non-obvious parts.
* **Tested**: every building block comes with `pytest` tests, so you can
  change things and immediately see what breaks.
* **Small and runnable**: the target is a model that you can train and run on
  a laptop (CPU, Apple Silicon MPS, or a single CUDA GPU).

## Target platform

The code is written to be portable: it runs on macOS, Linux, and Windows, on
CPU or on a CUDA GPU.

The reference environment, where the code is developed, tested, and
optimized, is a **MacBook with Apple Silicon**. When it is available, the
code uses PyTorch's
[MPS backend](https://pytorch.org/docs/stable/notes/mps.html) (Metal
Performance Shaders) for GPU acceleration. It falls back to CUDA or CPU on
other machines.

## Roadmap

The list below tracks the planned stages: ✅ implemented and tested,
🚧 in progress, ⬜ planned.

- ✅ **Text data and tokenization**: from raw text to token IDs with Byte
  Pair Encoding (BPE) and tiktoken, and the Tokenizer explorer demo
  ([Chapter 1](chap1/README.md)).
- ✅ **Data pipeline**: sliding-window datasets and data loaders for
  next-token prediction ([Chapter 1 notebook](chap1/dataloader.ipynb)).
- 🚧 **Embeddings**: token and positional embeddings.
- ⬜ **Attention**: self-attention, causal masking, multi-head attention.
- ⬜ **GPT architecture**: layer normalization, feed-forward blocks,
  transformer blocks, the full GPT model.
- ⬜ **Text generation**: greedy decoding, temperature, top-k sampling.
- ⬜ **Pre-training**: loss computation, training loop, checkpoints.
- ⬜ **Evaluation**: training and validation loss, perplexity.
- ⬜ **Loading pretrained weights**: load public GPT-2 weights into our model.
- ⬜ **Fine-tuning**: classification fine-tuning and instruction fine-tuning.

## Chapters

| Chapter | Topic | What you get |
| --- | --- | --- |
| [Chapter 0](chap0/README.md) | Environment | A check of Python, packages, and MPS acceleration |
| [Chapter 1](chap1/README.md) | Text data and tokenization | The Tokenizer explorer (web UI and CLI), and a notebook on the data loader for next-token prediction |

Questions that come up along the way, with short answers, are collected by
chapter in [Questions and answers](docs/questions-and-answers.md).

## Getting started

To set up the environment and run the chapters, follow the
[Quickstart](QUICKSTART.md). In short: create the `build-llm` Conda
environment with Python 3.11, install `requirements.txt` and
`requirements-dev.txt`, and run `python -m chap0.check_environment`.

## Repository layout

```text
build-llm/
├── build_llm/            # Reusable package with the LLM building blocks
├── chap0/                # Chapter 0: check of the development environment
├── chap1/                # Chapter 1: text data and tokenization
│   └── ui/               # Web UI of the tokenizer demo (Next.js)
├── docs/                 # Guides, such as the questions and answers by chapter
├── tests/                # pytest test suite
├── QUICKSTART.md         # Setup, environment check, and development checks
├── AGENTS.md             # Rules for contributors and coding agents
├── pyproject.toml        # Project metadata and tool configuration
├── requirements.txt      # Runtime dependencies
└── requirements-dev.txt  # Development tools and JupyterLab
```

## Contributing

Contributions and suggestions are welcome. Please read [AGENTS.md](AGENTS.md)
for the coding rules: code formatted with Black, linted with Pylint, and
tested with pytest.

## License

Released under the [MIT License](LICENSE).
