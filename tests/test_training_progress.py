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
from chap4.first_training import train_model_simple
from chap4 import first_training


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


def test_progress_shows_the_learning_rate():
    """The learning rate column is filled when the value is given."""
    stream = io.StringIO()
    progress = TrainingProgress(stream)
    progress.start_epoch(1, 1)
    progress.update(1, 2, 1, 5.0, 5.5, learning_rate=4e-4)
    progress.update(2, 2, 2, 4.0, 4.5)

    output = stream.getvalue()
    rows = [line for line in output.splitlines() if line.lstrip().startswith("[")]
    assert "Learn. rate" in output
    assert "4.00e-04" in rows[0]
    assert "e-0" not in rows[1]  # No learning rate given: an empty column


def test_color_can_be_disabled(monkeypatch):
    """Interactive output supports colors and respects NO_COLOR."""
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert TrainingProgress(stream).color
    monkeypatch.setenv("NO_COLOR", "")
    assert not TrainingProgress(stream).color


def test_book_paths_follow_the_manifest(tmp_path, monkeypatch, write_manifest):
    """The books are the ones of the manifest, sorted by file name."""
    books_dir = tmp_path / "books"
    books_dir.mkdir()
    for name in ("zebra.txt", "apple.txt", "README.md"):
        (books_dir / name).write_text("Example", encoding="utf-8")
    (books_dir / "nested.txt").mkdir()  # A folder is not a book
    write_manifest(tmp_path / "manifest.json", books_dir, ["zebra.txt", "apple.txt"])
    monkeypatch.setattr(first_training, "BOOKS_DIR", books_dir)
    monkeypatch.setattr(first_training, "MANIFEST_PATH", tmp_path / "manifest.json")

    names = [path.name for path in first_training.find_book_paths()]

    assert names == ["apple.txt", "zebra.txt"]


@pytest.mark.parametrize("missing_directory", [False, True])
def test_missing_books_report_preparation_command(
    tmp_path, monkeypatch, capsys, write_manifest, missing_directory
):
    """Missing books fail before allocating a model, with the command to run."""
    (tmp_path / "book.txt").write_text("Example", encoding="utf-8")
    write_manifest(tmp_path / "manifest.json", tmp_path, ["book.txt"])
    (tmp_path / "book.txt").unlink()
    directory = tmp_path / "missing" if missing_directory else tmp_path
    monkeypatch.setattr(first_training, "BOOKS_DIR", directory)
    monkeypatch.setattr(first_training, "MANIFEST_PATH", tmp_path / "manifest.json")

    assert first_training.main([]) == 1
    error = capsys.readouterr().err
    assert "missing: book.txt" in error
    assert "python -m chap4.prepare_books" in error


def test_extra_books_are_refused(tmp_path, monkeypatch, capsys, write_manifest):
    """A text file that is not in the manifest stops the training."""
    (tmp_path / "book.txt").write_text("Example", encoding="utf-8")
    write_manifest(tmp_path / "manifest.json", tmp_path, ["book.txt"])
    (tmp_path / "extra.txt").write_text("Not chosen", encoding="utf-8")
    monkeypatch.setattr(first_training, "BOOKS_DIR", tmp_path)
    monkeypatch.setattr(first_training, "MANIFEST_PATH", tmp_path / "manifest.json")

    assert first_training.main([]) == 1
    assert "not in the manifest: extra.txt" in capsys.readouterr().err


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
    state = train_model_simple(
        model,
        loader,
        loader,
        optimizer,
        torch.device("cpu"),
        num_epochs=2,
        eval_freq=10,
        eval_iter=1,
    )
    assert state.track_tokens_seen == [4, 12, 24]
    assert len(state.train_losses) == len(state.val_losses) == 3
    assert all(torch.isfinite(torch.tensor(state.train_losses + state.val_losses)))
    output = capsys.readouterr().out
    assert "Epoch 2/2" in output
    assert output.count("100%") == 2
