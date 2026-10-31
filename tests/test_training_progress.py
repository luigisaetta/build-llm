"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests of terminal loss reporting and a tiny offline training run.
"""

import io

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from build_llm.gpt2 import GPTModel
from build_llm.training_progress import TrainingProgress, loss_trend
from chap4.first_training import TEXT_PATHS, train_model_simple


@pytest.mark.parametrize("unicode", [True, False])
def test_loss_trend(unicode):
    """Trends share a scale, tolerate missing values, and keep recent history."""
    levels = "▁▂▃▄▅▆▇█" if unicode else ".:-=+*#@"
    assert loss_trend([0, 7], (0, 7), unicode=unicode) == levels[0] + levels[-1]
    assert loss_trend([2, 2], (2, 2), unicode=unicode) == levels[0] * 2
    assert loss_trend([], (0, 1), unicode=unicode) == ""
    assert loss_trend([float("nan")], (0, 1), unicode=unicode) == "?"
    assert len(loss_trend([1] * 50, (0, 1), unicode=unicode)) == 40


def test_progress_best_validation_and_plain_output():
    """Only finite validation improvements are marked, without ANSI in logs."""
    stream = io.StringIO()
    progress = TrainingProgress(stream)
    progress.start_epoch(1, 2)
    for step, validation in enumerate([3.0, 4.0, 2.0, float("nan")], start=1):
        progress.update(step, 4, step, 1.0, validation)
    progress.finish_epoch(1.5, [2.0, 1.0], [3.0, 2.0])
    output = stream.getvalue()
    assert output.count("new best") == 2
    assert "100%" in output
    assert "non-finite loss" in output
    assert "Best validation: 2.000" in output
    assert "Scale: 1.000 to 3.000" in output
    assert "\033[" not in output


def test_color_can_be_disabled(monkeypatch):
    """Interactive output supports colors and respects NO_COLOR."""
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert TrainingProgress(stream).color
    monkeypatch.setenv("NO_COLOR", "")
    assert not TrainingProgress(stream).color


def test_books_are_located_in_books_directory():
    """Every training text remains accessible after moving the corpus."""
    assert len(TEXT_PATHS) == 12
    assert all(path.parent.name == "books" and path.is_file() for path in TEXT_PATHS)


def test_tiny_training_reports_final_step(capsys):
    """CPU training evaluates epoch boundaries even between scheduled checks."""
    torch.manual_seed(123)
    model = GPTModel(
        {
            "vocab_size": 8,
            "context_length": 4,
            "emb_dim": 8,
            "n_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    )
    inputs = torch.randint(0, 8, (3, 4))
    loader = DataLoader(TensorDataset(inputs, inputs.roll(-1, dims=1)), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    training, validation, tokens = train_model_simple(
        model,
        loader,
        loader,
        optimizer,
        torch.device("cpu"),
        num_epochs=2,
        eval_freq=10,
        eval_iter=1,
    )
    assert tokens == [4, 12, 24]
    assert len(training) == len(validation) == 3
    assert all(torch.isfinite(torch.tensor(training + validation)))
    output = capsys.readouterr().out
    assert "Epoch 2/2" in output
    assert output.count("100%") == 2
