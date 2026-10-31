"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests for the fixed validation sample spread over every document.
"""

import pytest
import torch

from build_llm.validation_sample import (
    allocate_sequences,
    create_validation_sample_loader,
    evenly_spaced_indices,
)


class CharacterTokenizer:  # pylint: disable=too-few-public-methods
    """Offline stand-in for a tiktoken encoding: one token per character."""

    def encode(self, text, allowed_special=None):  # pylint: disable=unused-argument
        """Return the code point of each character."""
        return [ord(character) for character in text]


def test_evenly_spaced_indices_cover_the_whole_range():
    """The indices are the centers of equal slices, from start to end."""
    assert evenly_spaced_indices(10, 5) == [1, 3, 5, 7, 9]
    assert evenly_spaced_indices(4, 4) == [0, 1, 2, 3]
    assert evenly_spaced_indices(100, 1) == [50]


@pytest.mark.parametrize(("size", "count"), [(5, 0), (5, 6)])
def test_evenly_spaced_indices_reject_invalid_counts(size, count):
    """count must be between 1 and size."""
    with pytest.raises(ValueError):
        evenly_spaced_indices(size, count)


def test_allocation_is_proportional_with_one_per_document():
    """Large documents get more sequences, but every document gets at least one."""
    counts = allocate_sequences([1000, 100, 10, 2], total=50)

    assert sum(counts) == 50
    assert min(counts) >= 1
    assert counts[0] > counts[1] > counts[2]


def test_allocation_never_exceeds_the_available_sequences():
    """A request larger than the documents is capped at what they contain."""
    assert allocate_sequences([3, 2], total=10) == [3, 2]
    assert allocate_sequences([5, 5, 5], total=2) == [1, 1, 1]


def test_allocation_rejects_empty_or_invalid_sizes():
    """Sizes must be a non-empty list of positive numbers."""
    with pytest.raises(ValueError):
        allocate_sequences([], total=3)
    with pytest.raises(ValueError):
        allocate_sequences([4, 0], total=3)


def test_sample_loader_covers_every_document_and_is_fixed():
    """Every document is in the sample, and two passes yield the same batches."""
    # Documents of 41 and 401 tokens: 10 and 100 sequences of 4 tokens
    documents = ["a" * 41, "b" * 401]
    loader = create_validation_sample_loader(
        CharacterTokenizer(), documents, num_sequences=12, batch_size=4, max_length=4
    )

    first_pass = [inputs for inputs, _ in loader]
    second_pass = [inputs for inputs, _ in loader]
    tokens = torch.cat(first_pass).flatten().tolist()

    assert len(loader.dataset) == 12
    assert {ord("a"), ord("b")} <= set(tokens)
    assert all(torch.equal(x, y) for x, y in zip(first_pass, second_pass))
    assert first_pass[0].shape == (4, 4)
