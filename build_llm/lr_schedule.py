"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Learning rate schedules for training: linear warmup followed by
    a cosine decay, or by a stable phase and a final decay (WSD).

The learning rate is computed from the optimizer step alone, with a pure
function and no internal state. A training resumed from a checkpoint only
needs the step number to get exactly the same learning rate as an
uninterrupted run; there is no scheduler state to save or to restore.

Both schedules start with a linear warmup, from almost 0 to the peak learning
rate. At the start the weights are random, the gradients are large, and the
AdamW running averages are not reliable yet: small steps avoid unstable
updates. Then:

* "cosine": the learning rate decreases along half a cosine wave, from the
  peak to the minimum at the last step. The total number of steps must be
  known from the start, because it sets the shape of the whole curve.
* "wsd" (warmup, stable, decay): the learning rate stays at the peak, and
  decreases linearly to the minimum only in the last decay_fraction of the
  steps. Until the decay starts, the total number of steps can still change,
  so a run can be extended.

Small steps at the end, in both schedules, reduce the noise of the updates
and usually give a lower final loss than a constant learning rate.
"""

import math

import torch

SCHEDULES = ("cosine", "wsd")


def learning_rate(  # pylint: disable=too-many-arguments
    step: int,
    *,
    schedule: str,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    decay_fraction: float = 0.2,
) -> float:
    """Return the learning rate of an optimizer step.

    Args:
        step: Zero-based index of the optimizer step. Steps after the last one
            get min_lr.
        schedule: "cosine" or "wsd".
        peak_lr: The learning rate at the end of the warmup.
        min_lr: The learning rate at the last step; between 0 and peak_lr.
        warmup_steps: Number of warmup steps; 0 disables the warmup.
        total_steps: Number of optimizer steps of the whole training.
        decay_fraction: For "wsd" only, the fraction of total_steps used for
            the final decay, between 0 and 1. The decay never starts before
            the end of the warmup.

    Returns:
        The learning rate for this step.

    Raises:
        ValueError: If an argument is out of range or the schedule is unknown.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}.")
    if step < 0:
        raise ValueError("step must not be negative.")
    if not 0 <= min_lr <= peak_lr:
        raise ValueError("The learning rates must satisfy 0 <= min_lr <= peak_lr.")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps - 1.")
    if not 0 < decay_fraction <= 1:
        raise ValueError("decay_fraction must be in the interval (0, 1].")

    if step < warmup_steps:
        # Linear warmup: step 0 already gets a small, non-zero learning rate
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps - 1:
        return min_lr

    if schedule == "cosine":
        # Fraction of the decay done: 0 after the warmup, 1 at the last step
        progress = (step - warmup_steps) / (total_steps - 1 - warmup_steps)
        return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    decay_start = max(warmup_steps, total_steps - round(decay_fraction * total_steps))
    if step < decay_start:
        return peak_lr  # Stable phase
    # Linear decay, from the peak at decay_start to the minimum at the last step
    progress = (step - decay_start) / (total_steps - 1 - decay_start)
    return peak_lr + (min_lr - peak_lr) * progress


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    """Use a learning rate for the next optimizer steps, in every group.

    Call it before each optimizer.step(). It also overrides the learning rate
    that optimizer.load_state_dict restores from a checkpoint, so the
    schedule, not the saved value, always decides.

    Args:
        optimizer: The optimizer.
        value: The learning rate.
    """
    for group in optimizer.param_groups:
        group["lr"] = value
