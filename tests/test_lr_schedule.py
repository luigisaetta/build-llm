"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests for the cosine and WSD learning rate schedules.
"""

import pytest
import torch

from build_llm.lr_schedule import learning_rate, set_learning_rate

BASE = {"peak_lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 10, "total_steps": 110}


def curve(schedule, **overrides):
    """The learning rate of every step of a training."""
    arguments = {**BASE, **overrides}
    return [
        learning_rate(step, schedule=schedule, **arguments)
        for step in range(arguments["total_steps"])
    ]


@pytest.mark.parametrize("schedule", ["cosine", "wsd"])
def test_warmup_rises_linearly_to_the_peak(schedule):
    """The warmup starts small, grows by equal steps, and ends at the peak."""
    rates = curve(schedule)

    assert rates[0] == pytest.approx(1e-4)  # peak / warmup_steps
    assert rates[9] == pytest.approx(1e-3)
    steps = [b - a for a, b in zip(rates[:10], rates[1:10])]
    assert steps == pytest.approx([1e-4] * 9)


@pytest.mark.parametrize("schedule", ["cosine", "wsd"])
def test_both_schedules_end_at_the_minimum_and_never_rise(schedule):
    """After the warmup the learning rate only decreases, down to min_lr."""
    rates = curve(schedule)

    assert rates[-1] == pytest.approx(1e-4)
    after_warmup = rates[10:]
    assert all(b <= a + 1e-15 for a, b in zip(after_warmup, after_warmup[1:]))
    assert learning_rate(500, schedule=schedule, **BASE) == pytest.approx(1e-4)


def test_cosine_is_halfway_in_the_middle_of_the_decay():
    """Half way through the decay, the cosine is half way between the rates."""
    # 111 steps: the decay goes from step 10 to step 110, its middle is step 60
    rates = curve("cosine", total_steps=111)

    assert rates[60] == pytest.approx((1e-3 + 1e-4) / 2)


def test_wsd_is_stable_until_the_final_decay():
    """WSD keeps the peak until the last 20% of the steps, then decays linearly."""
    rates = curve("wsd")  # Decay starts at 110 - 22 = 88

    assert rates[10:89] == pytest.approx([1e-3] * 79)
    assert rates[89] < 1e-3
    decay = rates[88:]
    differences = [a - b for a, b in zip(decay, decay[1:])]
    assert differences == pytest.approx([differences[0]] * len(differences))


def test_wsd_can_be_extended_before_the_decay():
    """Before the decay, a longer run has the same learning rates so far."""
    short, longer = curve("wsd"), curve("wsd", total_steps=220)

    assert short[:88] == pytest.approx(longer[:88])


def test_no_warmup_starts_at_the_peak():
    """warmup_steps=0 gives the peak learning rate from the first step."""
    assert learning_rate(0, schedule="cosine", **{**BASE, "warmup_steps": 0}) == (
        pytest.approx(1e-3)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"schedule": "linear"},
        {"step": -1},
        {"min_lr": 2e-3},
        {"warmup_steps": 110},
        {"decay_fraction": 0.0},
    ],
)
def test_invalid_arguments_are_rejected(overrides):
    """Unknown schedules and out-of-range values raise ValueError."""
    arguments = {"step": 0, "schedule": "wsd", **BASE, **overrides}
    step = arguments.pop("step")

    with pytest.raises(ValueError):
        learning_rate(step, **arguments)


def test_set_learning_rate_updates_every_group():
    """The value is written in all the parameter groups of the optimizer."""
    first, second = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([{"params": [first]}, {"params": [second]}], lr=0.1)

    set_learning_rate(optimizer, 0.005)

    assert [group["lr"] for group in optimizer.param_groups] == [0.005, 0.005]
