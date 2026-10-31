"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests of the chapter 4 training script: checkpoints and resume.
"""

import pytest
import torch

from build_llm.checkpoints import checkpoint_path, list_checkpoints
from build_llm.training_log import read_metrics
from chap4 import first_training

TINY_CONFIG = {
    "vocab_size": 261,  # The 261 tokens of the tiny_encoding fixture
    "context_length": 8,
    "emb_dim": 8,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.1,  # Dropout on, so the generator state matters
    "qkv_bias": False,
}


@pytest.fixture(name="tiny_run")
def fixture_tiny_run(tmp_path, monkeypatch, tiny_encoding, write_manifest):
    """Make first_training fast and offline: tiny model, books, and tokenizer."""
    books_dir = tmp_path / "books"
    books_dir.mkdir()
    for name, word in (("a.txt", "hello world. "), ("b.txt", "the sea is calm. ")):
        (books_dir / name).write_text(word * 60, encoding="utf-8")

    write_manifest(tmp_path / "manifest.json", books_dir, ["a.txt", "b.txt"])
    monkeypatch.setattr(first_training, "BOOKS_DIR", books_dir)
    monkeypatch.setattr(first_training, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(first_training, "GPT_CONFIG_SMALL_256", TINY_CONFIG)
    monkeypatch.setattr(first_training, "BATCH_SIZE", 2)
    monkeypatch.setattr(first_training, "EVAL_FREQ", 1000)
    monkeypatch.setattr(first_training, "EVAL_ITER", 1)
    monkeypatch.setattr(first_training, "VAL_SAMPLE_SEQUENCES", 4)
    monkeypatch.setattr(first_training, "PROMPT_TOKENS", 3)
    monkeypatch.setattr(first_training, "NEW_TOKENS", 2)
    monkeypatch.setattr(first_training, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        first_training.tiktoken, "get_encoding", lambda name: tiny_encoding
    )
    return tmp_path


def final_weights(directory, epoch):
    """The model weights stored in the checkpoint of an epoch."""
    checkpoint = torch.load(
        checkpoint_path(directory, epoch), map_location="cpu", weights_only=True
    )
    return checkpoint["model_state"]


def stop_after_epoch(monkeypatch, last_epoch):
    """Simulate Ctrl-C right after the checkpoint of an epoch is saved."""
    real_saver = first_training.make_checkpoint_saver

    def saver_that_stops(*args, **kwargs):
        save = real_saver(*args, **kwargs)

        def save_then_stop(epoch, state):
            save(epoch, state)
            if epoch == last_epoch:
                raise KeyboardInterrupt

        return save_then_stop

    monkeypatch.setattr(first_training, "make_checkpoint_saver", saver_that_stops)


def test_resumed_training_matches_an_uninterrupted_one(tiny_run, monkeypatch):
    """Ctrl-C after epoch 1, then --resume, gives the weights and the metrics
    of an uninterrupted run, with the cosine schedule and gradient clipping."""
    monkeypatch.setattr(first_training, "KEEP_CHECKPOINTS", None)
    monkeypatch.setattr(first_training, "NUM_EPOCHS", 2)
    straight, split = tiny_run / "straight", tiny_run / "split"
    assert first_training.main(["--checkpoint-dir", str(straight)]) == 0

    with monkeypatch.context() as interrupted:
        stop_after_epoch(interrupted, last_epoch=1)
        assert first_training.main(["--checkpoint-dir", str(split)]) == 130
    # Rows written after the checkpoint, as by an interrupted epoch, are dropped
    with (split / "metrics.csv").open("a", encoding="utf-8") as file:
        file.write("999,2,0,0.1,1.0,1.0\n")
    assert first_training.main(["--checkpoint-dir", str(split), "--resume"]) == 0

    expected, resumed = final_weights(straight, 2), final_weights(split, 2)
    for name, tensor in expected.items():
        assert torch.equal(tensor, resumed[name]), name
    assert read_metrics(split / "metrics.csv") == read_metrics(straight / "metrics.csv")


def test_metrics_follow_the_cosine_schedule(tiny_run, monkeypatch):
    """The learning rate rises in the warmup and ends at the minimum."""
    monkeypatch.setattr(first_training, "NUM_EPOCHS", 2)
    monkeypatch.setattr(first_training, "WARMUP_FRACTION", 0.3)
    monkeypatch.setattr(first_training, "EVAL_FREQ", 2)
    directory = tiny_run / "run"

    assert first_training.main(["--checkpoint-dir", str(directory)]) == 0

    rows = read_metrics(directory / "metrics.csv")
    rates = [row["learning_rate"] for row in rows]
    assert rows[0]["step"] == 1 and rows[0]["epoch"] == 1
    assert rates[0] < first_training.LEARNING_RATE  # Still in the warmup
    assert max(rates) == pytest.approx(first_training.LEARNING_RATE, rel=0.05)
    assert rates[-1] == pytest.approx(first_training.MIN_LEARNING_RATE)
    assert [row["step"] for row in rows] == sorted(row["step"] for row in rows)


def test_resume_with_other_number_of_epochs_is_rejected(tiny_run, monkeypatch, capsys):
    """The cosine schedule depends on NUM_EPOCHS, so it cannot change."""
    directory = tiny_run / "run"
    monkeypatch.setattr(first_training, "NUM_EPOCHS", 2)
    with monkeypatch.context() as interrupted:
        stop_after_epoch(interrupted, last_epoch=1)
        assert first_training.main(["--checkpoint-dir", str(directory)]) == 130

    monkeypatch.setattr(first_training, "NUM_EPOCHS", 3)
    assert first_training.main(["--checkpoint-dir", str(directory), "--resume"]) == 1
    assert "num_epochs: checkpoint 2, now 3" in capsys.readouterr().err


def test_new_run_refuses_to_overwrite_checkpoints(tiny_run, monkeypatch, capsys):
    """A new run stops if the folder already holds checkpoints of another run."""
    directory = tiny_run / "run"
    monkeypatch.setattr(first_training, "NUM_EPOCHS", 1)
    assert first_training.main(["--checkpoint-dir", str(directory)]) == 0

    assert first_training.main(["--checkpoint-dir", str(directory)]) == 1
    assert "--resume" in capsys.readouterr().err


def test_resume_without_checkpoints_fails_clearly(tiny_run, capsys):
    """--resume with an empty folder exits with an error message."""
    directory = tiny_run / "empty"

    assert first_training.main(["--checkpoint-dir", str(directory), "--resume"]) == 1
    assert "No checkpoint found" in capsys.readouterr().err


def test_resume_with_other_settings_is_rejected(tiny_run, monkeypatch, capsys):
    """Changing the batch size between runs prevents the resume."""
    directory = tiny_run / "run"
    monkeypatch.setattr(first_training, "NUM_EPOCHS", 1)
    assert first_training.main(["--checkpoint-dir", str(directory)]) == 0

    monkeypatch.setattr(first_training, "BATCH_SIZE", 3)
    assert first_training.main(["--checkpoint-dir", str(directory), "--resume"]) == 1
    assert "batch_size: checkpoint 2, now 3" in capsys.readouterr().err


def test_only_the_newest_checkpoints_are_kept(tiny_run, monkeypatch):
    """With KEEP_CHECKPOINTS = 2, three epochs leave the last two checkpoints."""
    directory = tiny_run / "run"
    monkeypatch.setattr(first_training, "NUM_EPOCHS", 3)
    monkeypatch.setattr(first_training, "KEEP_CHECKPOINTS", 2)

    assert first_training.main(["--checkpoint-dir", str(directory)]) == 0

    names = [path.name for path in list_checkpoints(directory)]
    assert names == ["epoch-002.pt", "epoch-003.pt"]
