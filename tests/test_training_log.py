"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests for the CSV file of training metrics.
"""

import pytest

from build_llm.training_log import FIELDS, MetricsLog, read_metrics


def row(step, epoch=1):
    """A metrics row for an optimizer step."""
    return {
        "step": step,
        "epoch": epoch,
        "tokens_seen": 100 * step,
        "learning_rate": 4e-4,
        "train_loss": 5.0 - step / 10,
        "val_loss": 5.5 - step / 10,
    }


def test_rows_are_written_and_read_back(tmp_path):
    """Each appended row is on disk at once, with integers and floats restored."""
    log = MetricsLog(tmp_path / "run" / "metrics.csv")
    log.start()
    log.append(row(1))
    log.append(row(5))

    rows = read_metrics(tmp_path / "run" / "metrics.csv")

    assert rows == [row(1), row(5)]
    assert isinstance(rows[0]["step"], int)
    assert list(rows[0]) == list(FIELDS)


def test_a_new_run_starts_an_empty_file(tmp_path):
    """Starting without a resume step discards the rows of an older run."""
    log = MetricsLog(tmp_path / "metrics.csv")
    log.start()
    log.append(row(1))

    log.start()

    assert not read_metrics(tmp_path / "metrics.csv")


def test_resume_drops_the_rows_after_the_checkpoint(tmp_path):
    """Rows up to the checkpoint step are kept, later ones are removed."""
    log = MetricsLog(tmp_path / "metrics.csv")
    log.start()
    for step in (1, 50, 100, 150):
        log.append(row(step))

    log.start(resume_step=100)
    log.append(row(101, epoch=2))

    steps = [entry["step"] for entry in read_metrics(tmp_path / "metrics.csv")]
    assert steps == [1, 50, 100, 101]


def test_resume_without_a_file_starts_one(tmp_path):
    """Resuming a run whose metrics file is missing creates an empty file."""
    log = MetricsLog(tmp_path / "metrics.csv")

    log.start(resume_step=10)

    assert not read_metrics(tmp_path / "metrics.csv")


def test_rows_must_have_exactly_the_fields(tmp_path):
    """A row with a missing or an unknown field is rejected."""
    log = MetricsLog(tmp_path / "metrics.csv")
    log.start()
    incomplete = row(1)
    del incomplete["val_loss"]

    with pytest.raises(ValueError):
        log.append(incomplete)
    with pytest.raises(ValueError):
        log.append({**row(1), "extra": 1})
