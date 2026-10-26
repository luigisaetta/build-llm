"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Tests of the device selection and MPS checks in build_llm.device.
"""

import pytest
import torch

from build_llm import device as device_module
from build_llm.device import check_mps, get_device, time_matmul

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)


def set_available_devices(monkeypatch, mps: bool, cuda: bool) -> None:
    """Pretend that only the given accelerators are available."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)


@pytest.mark.parametrize(
    "mps, cuda, expected",
    [
        (True, True, "mps"),
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_get_device_preference_order(monkeypatch, mps, cuda, expected):
    """MPS is preferred, then CUDA, then CPU."""
    set_available_devices(monkeypatch, mps=mps, cuda=cuda)

    assert get_device().type == expected


def test_check_mps_without_mps(monkeypatch):
    """Without MPS, the check reports it as not working, with no error."""
    monkeypatch.setattr(device_module.torch.backends.mps, "is_available", lambda: False)

    result = check_mps()

    assert not result.is_available
    assert not result.works
    assert result.error is None


@requires_mps
def test_check_mps_on_apple_silicon():
    """On Apple Silicon, MPS computes the same result as the CPU."""
    result = check_mps()

    assert result.is_built
    assert result.works


def test_time_matmul_on_cpu():
    """The timing is a positive number of seconds."""
    elapsed = time_matmul(torch.device("cpu"), size=32, repeats=2)

    assert elapsed > 0
