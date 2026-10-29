# AGENTS.md

Rules for humans and coding agents working on this repository.

## Repository purpose

This repository builds, step by step, all the tools needed to create, train,
and evaluate a small GPT-style Large Language Model in Python and PyTorch. It
is inspired by the book "Build a Large Language Model (From Scratch)" by
Sebastian Raschka. The goal is learning: code must be easy to read,
understand, and run.

## Language

* Write all documentation, comments, docstrings, notebook narrative cells,
  CLI help, log messages, and program output in English.
* Conversation with the user may follow the user's language.

## Python environment

* Python 3.11 or later.
* The Conda environment is `build-llm`. Activate it with
  `conda activate build-llm`, or use `conda run -n build-llm ...`.
* Do not silently fall back to globally installed packages. If the
  environment is unavailable, report which checks could not run.
* Keep runtime dependencies in `requirements.txt` and development tools in
  `requirements-dev.txt`. When you add a dependency, install it in the
  `build-llm` environment as well.
* Add dependencies only when they serve a clear need of the project.

## Repository layout

* `build_llm/`: the package with reusable code, shared by all chapters.
* `chapN/` (`chap0/`, `chap1/`, ...): one folder per chapter, with a
  `README.md`, an `__init__.py`, and the chapter scripts. Chapter scripts
  import reusable code from `build_llm`; they never import from other
  chapters. When logic is useful beyond one chapter, move it to `build_llm`.
* `tests/`: the pytest test suite.

## Running scripts

Every Python script must run from the repository root as a module, with
`python -m`, for example:

```bash
python -m chap0.check_environment
```

Scripts define a `main()` function, call it under
`if __name__ == "__main__":`, and return a meaningful exit code. Do not
modify `sys.path` and do not rely on the current working directory: locate
repository files relative to `__file__`.

## Code conventions

Clarity and readability for a human reader come first: this is a learning
project. Prefer the simple, explicit solution over the clever or the
shortest one.

Every Python file must begin with this module header, using the actual
modification date. An executable script may place its shebang before the
header. `tests/test_headers.py` enforces it.

```python
"""
Author: L. Saetta
Date Last Modified: YYYY-MM-DD
License: MIT
Description: Brief description of this file's responsibilities.
"""
```

* Prefer readable names, small functions, explicit control flow, and type
  hints.
* Use Google-style docstrings for public functions and classes, including
  arguments, return values, and raised exceptions.
* Explain non-obvious decisions (for example tensor shapes, masking, and
  numerical tricks). Do not add comments that merely restate the code.
* Document tensor shapes in docstrings or comments, for example
  `(batch_size, num_tokens, emb_dim)`.
* Do not hard-code devices, machine-specific paths, or credentials.

## Portability and target platform

* Code must be portable across macOS, Linux, and Windows, and across CPU,
  CUDA, and MPS devices.
* The reference environment, where the code is developed, tested, and
  optimized, is a MacBook with Apple Silicon.
* Use PyTorch's MPS backend when it is available
  (`torch.backends.mps.is_available()`); otherwise fall back to CUDA, then
  CPU. Select the device in one place and pass it explicitly; never assume
  that a specific device exists.
* When an operation is not supported on MPS, or behaves differently there,
  handle it explicitly (for example with a documented CPU fallback) and
  explain why in a comment.
* Do not add mandatory macOS-only dependencies. Optimizations specific to
  Apple Silicon must not break the code on other platforms.
* Tests must pass on CPU. Tests that require MPS or CUDA are skipped
  automatically when the device is not available.
* Keep notebooks focused and restartable from a clean kernel; move reusable
  logic into the package.

## Formatting, linting, and testing

Before considering a change complete, run all three checks in the `build-llm`
environment and make them pass:

1. Format with Black, always: `black .`
2. Lint with Pylint: `pylint build_llm chap* tests`
3. Test with pytest: `pytest`

Testing rules:

* Every building block gets proportionate `pytest` tests: expected outputs,
  tensor shapes, edge cases, and failure paths.
* Tests must be fast and deterministic: fix random seeds and use tiny model
  configurations.
* Tests must not download data or model weights. Mark network-dependent
  tests with `@pytest.mark.network`: they are excluded from the default run,
  and run with `pytest -m network`.

Tool configuration lives in `pyproject.toml`.

## Web UIs of the demos

Some chapters have a web UI for their demo, in `chapN/ui/`.

* Use Next.js (App Router) with TypeScript and Tailwind CSS.
* The Python backend holds all the logic; the UI only displays it. The
  browser calls the routes of the Next.js app (`app/api/`), which forward the
  requests to the Python backend (FastAPI). Configure the backend address with
  the `BACKEND_URL` environment variable.
* Every Python backend must also be usable from the command line, through a
  `python -m` script that reuses the same `build_llm` code.
* Keep components small and readable, like the Python code.
* Before considering a UI change complete, run in `chapN/ui/`:
  `npm run lint` and `npm run build`.
* Keep `node_modules/` and `.next/` out of version control; commit
  `package-lock.json`.

## Data, models, and artifacts

* Keep datasets, downloaded weights, checkpoints, and generated artifacts out
  of version control.
* Document where data comes from and its license.

## Third-party material and attribution

* The original code and documentation are released under the MIT License.
* Material derived from other projects keeps its license. For example, the
  notebooks derived from
  [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) are under
  the Apache License 2.0 (copy in `LICENSES/`).
* When you add or adapt third-party material, in the same change:
  * state its origin, copyright, and license at the top of the file (for a
    notebook, in a markdown cell after the title), and say that it was
    modified;
  * list the file in `NOTICE`, with its source;
  * add a copy of the license in `LICENSES/`, if it is not there yet.
* Do not copy text or figures from the book: they are protected by copyright,
  and the open-source license of LLMs-from-scratch excludes them.

## Working rules

* Inspect the existing code and documentation before editing.
* Prefer small, coherent, reviewable changes over broad rewrites.
* Preserve user changes; never discard work or run destructive operations
  without explicit authorization.
* Do not create commits unless explicitly asked.
* Do not add AI tools or assistants as authors, co-authors, or contributors
  in files, commit messages, or pull requests.
* Keep `README.md` focused on purpose, setup, and usage, and keep it aligned
  with the implementation (for example the roadmap checklist).
