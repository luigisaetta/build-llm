"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Record the losses of a training run in a CSV file, one row per
    evaluation, to plot them later; consistent with checkpoints and resume.

The file is written while training runs: each row is appended and flushed at
once, so the file can be read, and plotted, before the training ends. When a
training resumes from a checkpoint, the rows written after that checkpoint
(for example by an epoch that was interrupted) are removed, so the file always
describes one continuous run, exactly like the restored model.
"""

import csv
from collections.abc import Mapping
from pathlib import Path

# Columns of the file, in order
FIELDS = ("step", "epoch", "tokens_seen", "learning_rate", "train_loss", "val_loss")
_INTEGER_FIELDS = ("step", "epoch", "tokens_seen")


class MetricsLog:
    """A CSV file of training metrics, one row per evaluation.

    Args:
        path: The CSV file; its folder is created if needed.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def start(self, resume_step: int | None = None) -> None:
        """Prepare the file for a new or a resumed run.

        Args:
            resume_step: None for a new run, which starts an empty file. For a
                resumed run, the one-based number of the last optimizer step of
                the checkpoint: rows of later steps are removed, rows up to it
                are kept.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        kept = []
        if resume_step is not None and self.path.exists():
            kept = [
                row for row in read_metrics(self.path) if row["step"] <= resume_step
            ]
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(kept)

    def append(self, row: Mapping[str, float]) -> None:
        """Add one evaluation to the file, and write it to disk at once.

        Args:
            row: A value for each name in FIELDS.

        Raises:
            ValueError: If a field is missing or unknown.
        """
        if set(row) != set(FIELDS):
            raise ValueError(f"A metrics row needs exactly the fields {FIELDS}.")
        with self.path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDS).writerow(row)


def read_metrics(path: Path) -> list[dict[str, float]]:
    """Read a metrics file written by MetricsLog.

    Args:
        path: The CSV file.

    Returns:
        One dictionary per evaluation, with integer step, epoch, and
        tokens_seen, and float learning_rate, train_loss, and val_loss.
    """
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [
            {
                name: int(value) if name in _INTEGER_FIELDS else float(value)
                for name, value in row.items()
            }
            for row in csv.DictReader(file)
        ]
