"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Checks the environment: Python version, required packages, and MPS.

Run it from the repository root:

    python -m chap0.check_environment

The exit code is 0 if all the required checks pass, 1 otherwise. MPS is not
required: on machines without Apple Silicon its check is only informative.
"""

import platform
import sys
from pathlib import Path

from build_llm.environment import (
    MIN_PYTHON_VERSION,
    check_packages,
    is_python_version_supported,
    python_version_string,
    read_requirement_names,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILES = ["requirements.txt", "requirements-dev.txt"]

OK = "[ OK ]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def print_title(title: str) -> None:
    """Print a section title."""
    print()
    print(title)
    print("-" * len(title))


def check_python() -> bool:
    """Print the platform details and check the Python version.

    Returns:
        True if the Python version is supported.
    """
    print_title("Environment")
    print(f"{INFO} Operating system: {platform.system()} {platform.release()}")
    print(f"{INFO} Machine:          {platform.machine()}")
    print(f"{INFO} Python path:      {sys.executable}")

    minimum = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    supported = is_python_version_supported()
    status = OK if supported else FAIL
    print(f"{status} Python version:   {python_version_string()} (>= {minimum})")
    return supported


def check_required_packages() -> bool:
    """Check that all the packages in the requirements files are installed.

    Returns:
        True if all the packages are installed.
    """
    print_title("Required packages")
    all_installed = True

    for file_name in REQUIREMENTS_FILES:
        names = read_requirement_names(REPOSITORY_ROOT / file_name)
        for package in check_packages(names):
            if package.is_installed:
                print(f"{OK} {package.name:<12} {package.installed_version}")
            else:
                print(f"{FAIL} {package.name:<12} not installed ({file_name})")
                all_installed = False

    if not all_installed:
        print()
        print("Install the missing packages from the repository root with:")
        print("    pip install -r requirements.txt -r requirements-dev.txt")
    return all_installed


def check_mps_acceleration() -> None:
    """Check whether MPS works, and compare its speed with the CPU."""
    # Imported here, not at the top of the file: build_llm.device needs
    # PyTorch, and this function runs only after PyTorch has been found.
    # pylint: disable=import-outside-toplevel
    import torch

    from build_llm.device import check_mps, get_device, time_matmul

    print_title("MPS acceleration (Apple Silicon GPU)")
    result = check_mps()
    print(f"{INFO} PyTorch built with MPS: {result.is_built}")
    print(f"{INFO} MPS device available:   {result.is_available}")

    if not result.is_available:
        print(f"{INFO} MPS is not available: the code will use CUDA or CPU.")
    elif result.works:
        print(f"{OK} A test computation on MPS matches the CPU result.")
        cpu_time = time_matmul(torch.device("cpu"))
        mps_time = time_matmul(get_device())
        print(f"{INFO} Matrix multiplication (2048 x 2048), average time:")
        print(f"{INFO}     CPU: {cpu_time * 1000:8.2f} ms")
        print(f"{INFO}     MPS: {mps_time * 1000:8.2f} ms")
        print(f"{INFO}     MPS speed-up: {cpu_time / mps_time:.1f}x")
    else:
        print(f"{FAIL} MPS is available, but the test computation failed.")
        if result.error:
            print(f"{INFO} Error: {result.error}")

    print(f"{INFO} Device selected for this project: {get_device()}")


def main() -> int:
    """Run all the checks and print a summary.

    Returns:
        The exit code: 0 if all the required checks pass, 1 otherwise.
    """
    python_ok = check_python()
    packages_ok = check_required_packages()

    if packages_ok:
        check_mps_acceleration()
    else:
        print()
        print(f"{INFO} MPS check skipped: install the missing packages first.")

    print_title("Summary")
    if python_ok and packages_ok:
        print(f"{OK} The environment is ready.")
        return 0
    print(f"{FAIL} The environment is not ready: see the failures above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
