"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Selection of the PyTorch device (MPS, CUDA, or CPU) and MPS checks.
"""

import time
from dataclasses import dataclass

import torch


def get_device() -> torch.device:
    """Return the best available device.

    The preference order is MPS (Apple Silicon GPU), then CUDA, then CPU.

    Returns:
        The selected torch.device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class MpsCheckResult:
    """Outcome of the MPS check.

    Attributes:
        is_built: PyTorch was built with MPS support.
        is_available: The MPS device can be used on this machine.
        computes_correctly: A test computation on MPS matched the CPU result.
        error: The error message, if the test computation failed.
    """

    is_built: bool
    is_available: bool
    computes_correctly: bool
    error: str | None = None

    @property
    def works(self) -> bool:
        """True if MPS is available and computes correctly."""
        return self.is_available and self.computes_correctly


def check_mps(size: int = 256) -> MpsCheckResult:
    """Check that MPS is available and produces correct results.

    The check multiplies two random matrices on the CPU and on MPS, and
    compares the results.

    Args:
        size: The size of the square test matrices.

    Returns:
        The outcome of the check.
    """
    is_built = torch.backends.mps.is_built()
    is_available = torch.backends.mps.is_available()
    if not is_available:
        return MpsCheckResult(is_built, is_available, computes_correctly=False)

    try:
        generator = torch.Generator().manual_seed(42)
        a = torch.rand(size, size, generator=generator)
        b = torch.rand(size, size, generator=generator)

        expected = a @ b
        result_on_mps = a.to("mps") @ b.to("mps")

        # The GPU may sum in a different order than the CPU, so the results
        # are compared with a tolerance, not for exact equality.
        computes_correctly = torch.allclose(
            result_on_mps.cpu(), expected, rtol=1e-3, atol=1e-3
        )
    except RuntimeError as error:
        return MpsCheckResult(
            is_built, is_available, computes_correctly=False, error=str(error)
        )

    return MpsCheckResult(is_built, is_available, computes_correctly)


def synchronize(device: torch.device) -> None:
    """Wait until all the queued operations on the device are completed.

    GPU operations run asynchronously: without this call, a timer would
    measure only the time needed to queue the work.

    Args:
        device: The device to synchronize.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_matmul(device: torch.device, size: int = 2048, repeats: int = 10) -> float:
    """Measure the average time of a matrix multiplication on a device.

    Args:
        device: The device to use.
        size: The size of the square matrices.
        repeats: The number of timed multiplications.

    Returns:
        The average time of one multiplication, in seconds.
    """
    a = torch.rand(size, size, device=device)
    b = torch.rand(size, size, device=device)

    # A first, untimed run: it includes one-time costs (for example kernel
    # compilation) that would distort the measure.
    _ = a @ b
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    synchronize(device)
    elapsed = time.perf_counter() - start

    return elapsed / repeats
