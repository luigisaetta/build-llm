# Chapter 0: check the environment

Before building anything, make sure that the development environment is
ready. The script in this folder checks:

* the operating system, the machine architecture, and the Python version
  (3.11 or later is required);
* that all the packages listed in `requirements.txt` and
  `requirements-dev.txt` are installed;
* whether PyTorch can use the Apple Silicon GPU through the MPS backend. On
  a Mac with Apple Silicon, it also compares the speed of a matrix
  multiplication on the CPU and on MPS.

## Run

From the repository root, with the `build-llm` environment active:

```bash
python -m chap0.check_environment
```

The script exits with code 0 when the environment is ready, and with code 1
when the Python version is not supported or a package is missing. MPS is not
required: on other machines the project falls back to CUDA or CPU, and the
MPS check is only informative.

## Reusable code

The checks are implemented in the `build_llm` package, so that later
chapters can reuse them:

* `build_llm/environment.py`: Python version and installed packages.
* `build_llm/device.py`: device selection (`get_device()`: MPS, then CUDA,
  then CPU) and the MPS checks.
