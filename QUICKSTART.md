# Quickstart

This guide sets up the development environment, checks it, and shows how to
run the chapters, the notebooks, and the development checks. For the purpose
of the project, see the [README](README.md).

## Prerequisites

* Python 3.11 or later.
* [Conda](https://docs.conda.io/) (Anaconda or Miniconda).
* Git.
* [Node.js](https://nodejs.org/) 20.9 or later, with npm, for the web UIs of
  the demos (for example the one in Chapter 1).

## Setup

Clone the repository:

```bash
git clone https://github.com/luigisaetta/build-llm.git
```

Create and activate the Conda environment:

```bash
conda create -n build-llm python=3.11
```

```bash
conda activate build-llm
```

From the repository root, install the runtime and development dependencies:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Install the local project in editable mode from the repository root:

```bash
python -m pip install --no-deps -e .
```

This makes `build_llm` importable from notebook kernels started in chapter
folders. Editable mode uses the source files directly, so code changes do not
require reinstalling the package. Dependencies remain managed through the two
requirements files above.

## Check the environment

From the repository root, run the check of [Chapter 0](chap0/README.md):

```bash
python -m chap0.check_environment
```

It checks the Python version, the required packages, and whether PyTorch can
use the Apple Silicon GPU (MPS). It exits with code 0 when the environment is
ready.

## Run the chapters

All the scripts run from the repository root as modules, with `python -m`.
For example:

```bash
python -m chap1.tokenize_text --text "Hello, world!"
```

Each chapter has its own README with the details. For example,
[Chapter 1](chap1/README.md) explains how to start the Tokenizer explorer,
with its Python backend and its web UI.

## Work with notebooks

Activate the `build-llm` environment and start JupyterLab from the repository
root:

```bash
python -m jupyterlab
```

Then open the notebooks from the chapter folders, for example
`chap1/dataloader.ipynb`.

Use a kernel from the `build-llm` environment. If a running kernel cannot import
`build_llm` after the editable installation, restart it and run all cells again.

## Development checks

Run these checks from the repository root before considering a change
complete. See [AGENTS.md](AGENTS.md) for all the coding rules.

Format the code:

```bash
black .
```

Lint the code:

```bash
pylint build_llm chap* tests
```

Run the tests:

```bash
pytest
```

Tests that need network access are excluded by default. Run them with:

```bash
pytest -m network
```
