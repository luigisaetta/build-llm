"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Readable terminal tables and loss trends for training loops.
"""

import math
import os
import sys
from typing import TextIO


def loss_trend(
    values: list[float], limits: tuple[float, float], *, unicode: bool
) -> str:
    """Render up to 40 recent losses with a shared vertical scale.

    Args:
        values: Loss measurements in chronological order.
        limits: Minimum and maximum loss for both training and validation.
        unicode: Whether the output supports Unicode block characters.

    Returns:
        A sparkline, with higher characters representing higher losses.
        Non-finite values appear as question marks; empty input yields an empty line.
    """
    levels = "▁▂▃▄▅▆▇█" if unicode else ".:-=+*#@"
    low, high = limits
    result = []
    for value in values[-40:]:
        if not math.isfinite(value):
            result.append("?")
            continue
        fraction = (value - low) / (high - low) if high > low else 0.0
        index = round(max(0.0, min(1.0, fraction)) * (len(levels) - 1))
        result.append(levels[index])
    return "".join(result)


class TrainingProgress:
    """Print aligned loss measurements, highlighting validation improvements.

    Args:
        stream: Destination for progress output; defaults to standard output.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.color = self.stream.isatty() and "NO_COLOR" not in os.environ
        self.best_validation = float("inf")

    def _style(self, text: str, code: str) -> str:
        """Apply ANSI styling only to interactive terminals that allow it."""
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def start_demo(self, title: str, details: list[tuple[str, str]]) -> None:
        """Print a demo title and its runtime configuration.

        Args:
            title: Short title for the whole demo.
            details: Configuration labels and their display values.
        """
        print(self._style("\n  " + "=" * 70, "36"), file=self.stream)
        print(self._style(f"  {title}", "1;36"), file=self.stream)
        print(
            "  Train a small GPT from scratch on public-domain books.", file=self.stream
        )
        print(self._style("  " + "=" * 70, "36"), file=self.stream)
        for label, value in details:
            print(f"  {label:<14} {self._style(value, '1')}", file=self.stream)
        print(file=self.stream, flush=True)

    def show_books(self, books: list[tuple[str, int, int]]) -> None:
        """Print the corpus as an aligned table with totals.

        Args:
            books: Book filenames, character counts, and token counts.
        """
        width = max([len("Book")] + [len(name) for name, _, _ in books])
        print(self._style(f"  Corpus | {len(books)} books", "1"), file=self.stream)
        print(
            f"  {'Book':<{width}}  {'Characters':>12}  {'Tokens':>12}",
            file=self.stream,
        )
        rule = "  " + "-" * (width + 28)
        print(rule, file=self.stream)
        for name, characters, tokens in books:
            print(
                f"  {name:<{width}}  {characters:>12,}  {tokens:>12,}",
                file=self.stream,
            )
        print(rule, file=self.stream)
        characters = sum(row[1] for row in books)
        tokens = sum(row[2] for row in books)
        print(
            self._style(f"  {'Total':<{width}}  {characters:>12,}  {tokens:>12,}", "1"),
            file=self.stream,
        )

    def show_split(self, datasets: list[tuple[str, int, int]]) -> None:
        """Print token and batch counts for each dataset split.

        Args:
            datasets: Split names, token counts, and batches per epoch.
                Token counts include any document separators.
        """
        print(
            self._style("\n  Dataset split | tokens include document separators", "1"),
            file=self.stream,
        )
        print(
            f"  {'Set':<14}  {'Tokens':>12}  {'Batches / epoch':>16}", file=self.stream
        )
        print("  " + "-" * 46, file=self.stream)
        for name, tokens, batches in datasets:
            print(f"  {name:<14}  {tokens:>12,}  {batches:>16,}", file=self.stream)
        print(file=self.stream, flush=True)

    def start_epoch(self, epoch: int, num_epochs: int) -> None:
        """Print an epoch title and column labels.

        Args:
            epoch: One-based epoch number.
            num_epochs: Total number of epochs.
        """
        print(self._style(f"\nEpoch {epoch}/{num_epochs}", "1"), file=self.stream)
        print(
            "  Progress                Step  Learn. rate     Train       Val       Gap"
            "  Validation",
            file=self.stream,
        )
        print("  " + "-" * 83, file=self.stream)

    def update(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        completed: int,
        total: int,
        step: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float | None = None,
    ) -> None:
        """Print an evaluation row with epoch progress and the generalization gap.

        Args:
            completed: Optimizer updates completed in the current epoch.
            total: Number of updates in this epoch, greater than zero.
            step: One-based optimizer step across all epochs.
            train_loss: Evaluated training cross-entropy.
            val_loss: Evaluated validation cross-entropy.
            learning_rate: Learning rate of the last optimizer step, if known.
        """
        fraction = completed / total
        filled = min(16, max(0, int(16 * fraction)))
        progress_bar = "#" * filled + "-" * (16 - filled)
        improved = math.isfinite(val_loss) and val_loss < self.best_validation
        if improved:
            self.best_validation = val_loss
        status = "new best" if improved else ""
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            status = "non-finite loss"
        training = self._style(f"{train_loss:8.3f}", "36")
        validation = self._style(f"{val_loss:8.3f}", "32" if improved else "33")
        rate = f"{learning_rate:11.2e}" if learning_rate is not None else " " * 11
        print(
            f"  [{progress_bar}] {fraction:4.0%}  {step:7d}  {rate}  {training}"
            f"  {validation}"
            f"  {val_loss - train_loss:+8.3f}  {status}",
            file=self.stream,
            flush=True,
        )

    def finish_epoch(
        self, seconds: float, train_losses: list[float], val_losses: list[float]
    ) -> None:
        """Print elapsed time, best validation loss, and two comparable trends.

        Args:
            seconds: Epoch duration, including evaluation.
            train_losses: All training measurements collected so far.
            val_losses: Validation measurements at the same steps.
        """
        recent = train_losses[-40:] + val_losses[-40:]
        finite = [value for value in recent if math.isfinite(value)]
        if finite:
            limits = min(finite), max(finite)
            encoding = self.stream.encoding or "ascii"
            try:
                "▁█".encode(encoding)
                supports_unicode = True
            except (UnicodeEncodeError, LookupError):
                supports_unicode = False
            print(
                "\n  Recent evaluations (left to right; shared loss scale):",
                file=self.stream,
            )
            for label, values, code in (
                ("Train", train_losses, "36"),
                ("Val  ", val_losses, "33"),
            ):
                trend = loss_trend(values, limits, unicode=supports_unicode)
                print(f"  {label}  {self._style(trend, code)}", file=self.stream)
            print(
                f"  Scale: {limits[0]:.3f} to {limits[1]:.3f} (lower is better)",
                file=self.stream,
            )
        best = (
            f"{self.best_validation:.3f}"
            if math.isfinite(self.best_validation)
            else "n/a"
        )
        print(
            f"  Completed in {seconds:.1f} s | Best validation: {best}\n",
            file=self.stream,
        )
