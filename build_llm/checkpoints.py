"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Save and restore training checkpoints, to resume an interrupted
    training at the end of the last completed epoch.

A checkpoint holds everything needed to continue as if the training had never
stopped: the model weights, the optimizer state (AdamW keeps two running
averages per parameter), the random number generator states (they decide the
batch order of the next epoch and the dropout masks), the progress counters,
the loss history, and the settings that must not change between runs.

Checkpoints are written atomically: first to a temporary file, then renamed,
so an interruption while saving never leaves a corrupted checkpoint under the
final name. They are read with torch.load(weights_only=True), which loads
tensors and plain Python values only and never executes code from the file.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Increase when the content of a checkpoint changes in an incompatible way
CHECKPOINT_VERSION = 1
_NAME_PREFIX = "epoch-"
_NAME_SUFFIX = ".pt"


def checkpoint_path(directory: Path, epoch: int) -> Path:
    """Return the path of the checkpoint saved at the end of an epoch.

    Args:
        directory: Folder of the checkpoints of one training run.
        epoch: One-based number of the completed epoch.

    Returns:
        A path such as directory/epoch-003.pt; zero padding keeps the files
        sorted by epoch.
    """
    return Path(directory) / f"{_NAME_PREFIX}{epoch:03d}{_NAME_SUFFIX}"


def list_checkpoints(directory: Path) -> list[Path]:
    """List the checkpoints of a directory, from the oldest epoch to the newest.

    Only files named like checkpoint_path(directory, epoch) are listed.

    Args:
        directory: Folder of the checkpoints of one training run.

    Returns:
        The checkpoint paths sorted by epoch; empty if the folder is missing.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    paths = []
    for path in directory.glob(f"{_NAME_PREFIX}*{_NAME_SUFFIX}"):
        number = path.name[len(_NAME_PREFIX) : -len(_NAME_SUFFIX)]
        if number.isdigit():
            paths.append((int(number), path))
    return [path for _, path in sorted(paths)]


def latest_checkpoint(directory: Path) -> Path | None:
    """Return the checkpoint of the most recent epoch, or None if there is none.

    Args:
        directory: Folder of the checkpoints of one training run.

    Returns:
        The path of the newest checkpoint, or None.
    """
    checkpoints = list_checkpoints(directory)
    return checkpoints[-1] if checkpoints else None


def _rng_states(device: torch.device) -> dict[str, torch.Tensor]:
    """Collect the random number generator states that affect training."""
    states = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        states["mps"] = torch.mps.get_rng_state()
    return states


def _restore_rng_states(
    states: Mapping[str, torch.Tensor], device: torch.device
) -> None:
    """Restore the generator states saved by _rng_states, for this device."""
    torch.set_rng_state(states["cpu"])
    if device.type == "cuda" and "cuda" in states:
        torch.cuda.set_rng_state(states["cuda"], device)
    elif device.type == "mps" and "mps" in states:
        torch.mps.set_rng_state(states["mps"])


def save_checkpoint(  # pylint: disable=too-many-arguments
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    training_state: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> None:
    """Save a checkpoint at the end of an epoch, atomically.

    Args:
        path: Destination file, usually checkpoint_path(directory, epoch).
        model: The model being trained.
        optimizer: Its optimizer.
        epoch: One-based number of the completed epoch.
        device: The device of the model, whose generator state is saved too.
        training_state: Progress counters and loss history, made of numbers,
            strings, lists, and dictionaries only.
        settings: The values that must be the same to resume, such as the
            model configuration and the batch size; see check_settings.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_states": _rng_states(device),
        "training_state": dict(training_state),
        "settings": dict(settings),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)  # Atomic on the same file system


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read a checkpoint file, without restoring it into a model.

    The file is memory-mapped: tensors are read from disk only when used, so
    reading a 1.5 GB checkpoint to take only the model weights does not load
    the optimizer state into memory.

    Args:
        path: The checkpoint file.

    Returns:
        The checkpoint, with "epoch", "model_state", "optimizer_state",
        "rng_states", "training_state", and "settings".

    Raises:
        ValueError: If the checkpoint has an unsupported version.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {checkpoint.get('version')} in "
            f"{path}; expected {CHECKPOINT_VERSION}."
        )
    return checkpoint


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint into a model and an optimizer, in place.

    Call it after moving the model to its device and creating the optimizer:
    the weights are copied to the device of the model, and the optimizer
    state follows its parameters. The generator states are restored too, so
    the next epoch uses the same batch order as an uninterrupted run.

    Args:
        path: The checkpoint file.
        model: A model with the same architecture as the saved one.
        optimizer: An optimizer of the same type, over model.parameters().
        device: The device of the model.
        expected_settings: If given, the settings of the current run: they are
            compared with the saved ones with check_settings before anything
            is restored, so an incompatible checkpoint changes nothing.

    Returns:
        The rest of the checkpoint: "epoch", "training_state", and "settings".

    Raises:
        ValueError: If the checkpoint has an unsupported version, or its
            settings differ from expected_settings.
        RuntimeError: If the saved weights do not fit the model.
    """
    checkpoint = read_checkpoint(path)
    if expected_settings is not None:
        check_settings(checkpoint["settings"], expected_settings)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _restore_rng_states(checkpoint["rng_states"], device)
    return {
        "epoch": checkpoint["epoch"],
        "training_state": checkpoint["training_state"],
        "settings": checkpoint["settings"],
    }


def check_settings(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Check that a training can resume with the current settings.

    Args:
        saved: The settings stored in the checkpoint.
        current: The settings of the current run.

    Raises:
        ValueError: If a setting differs, or is present on one side only; the
            message lists every difference.
    """
    differences = [
        f"{key}: checkpoint {saved.get(key)!r}, now {current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    if differences:
        raise ValueError(
            "The checkpoint was saved with different settings:\n  "
            + "\n  ".join(differences)
        )


def remove_old_checkpoints(directory: Path, keep: int) -> list[Path]:
    """Delete the oldest checkpoints of a directory, keeping the newest ones.

    Only files named like checkpoint_path(directory, epoch) are considered;
    no other file is touched.

    Args:
        directory: Folder of the checkpoints of one training run.
        keep: Number of most recent checkpoints to keep; at least 1.

    Returns:
        The deleted paths.

    Raises:
        ValueError: If keep is smaller than 1.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1.")
    old = list_checkpoints(directory)[:-keep]
    for path in old:
        path.unlink()
    return old
