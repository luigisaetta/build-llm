"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: A fixed validation sample spread over every document, for cheap
    and comparable validation losses during training.

Computing the validation loss on the whole validation set at every evaluation
is slow, so training loops evaluate a few batches only. Taking the first
batches of an unshuffled loader always measures the same small passage, the
beginning of the first document. This module chooses, once, sequences at
regular intervals inside every document, so that each evaluation covers all
the documents and uses exactly the same sequences as the previous one.
"""

from collections.abc import Sequence

import tiktoken
from torch.utils.data import ConcatDataset, DataLoader, Subset

from build_llm.dataloaders import GPTDatasetV1


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    """Choose count indices at regular intervals in range(size).

    Each index is the center of one of count equal slices of the range, so
    the beginning and the end of the range are covered alike.

    Args:
        size: Number of available items.
        count: Number of indices to choose; at most size.

    Returns:
        The indices, in increasing order and without repetitions.

    Raises:
        ValueError: If count is not between 1 and size.
    """
    if not 1 <= count <= size:
        raise ValueError(f"count must be between 1 and {size}, got {count}.")
    return [int((i + 0.5) * size / count) for i in range(count)]


def allocate_sequences(sizes: Sequence[int], total: int) -> list[int]:
    """Share total sequences among documents, in proportion to their sizes.

    Every document gets at least one sequence, so that short documents are
    represented too; the rest is shared in proportion to the sizes, with the
    largest-remainder method. No document gets more sequences than it has.

    Args:
        sizes: Number of available sequences of each document; all positive.
        total: Sequences to share. If it is smaller than the number of
            documents, every document still gets one.

    Returns:
        The number of sequences of each document, in the order of sizes. The
        sum is max(total, len(sizes)), capped at sum(sizes).

    Raises:
        ValueError: If sizes is empty or contains a value smaller than 1.
    """
    if not sizes or min(sizes) < 1:
        raise ValueError("sizes must be a non-empty list of positive integers.")

    total = min(max(total, len(sizes)), sum(sizes))
    # One sequence each, then the rest in proportion to the spare sequences
    spare = [size - 1 for size in sizes]
    remaining = total - len(sizes)
    if remaining == 0:
        return [1] * len(sizes)
    quotas = [remaining * s / sum(spare) for s in spare]
    counts = [1 + int(quota) for quota in quotas]
    # Largest remainders: the documents whose quota was rounded down the most
    # get one more sequence each. A quota never exceeds the spare sequences,
    # and only quotas with a positive remainder are rounded up, so no document
    # gets more sequences than it has.
    by_remainder = sorted(
        range(len(sizes)), key=lambda i: quotas[i] - int(quotas[i]), reverse=True
    )
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return counts


def create_validation_sample_loader(  # pylint: disable=too-many-arguments
    tokenizer: tiktoken.Encoding,
    documents: Sequence[str],
    *,
    num_sequences: int,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    """Create a loader of a fixed validation sample spread over every document.

    Each document is cut into non-overlapping sequences of max_length tokens,
    as the validation loader does. The num_sequences are shared among the
    documents in proportion to their length, at least one each, and chosen at
    regular intervals inside each document. The loader does not shuffle, so
    every pass yields the same batches in the same order.

    Args:
        tokenizer: A tiktoken encoding, such as tiktoken.get_encoding("gpt2").
        documents: The validation texts, one per document (for example one per
            book), not joined.
        num_sequences: Size of the sample, in sequences; for example 25 batches
            of 4 sequences are 100 sequences.
        batch_size: Number of sequences per batch.
        max_length: Number of tokens per sequence.

    Returns:
        A DataLoader that yields (inputs, targets) batches, each of shape
        (batch_size, max_length), except possibly the last one.

    Raises:
        ValueError: If a document has fewer than max_length + 1 tokens.
    """
    datasets = [
        GPTDatasetV1(document, tokenizer, max_length, stride=max_length)
        for document in documents
    ]
    counts = allocate_sequences([len(dataset) for dataset in datasets], num_sequences)
    sample = ConcatDataset(
        [
            Subset(dataset, evenly_spaced_indices(len(dataset), count))
            for dataset, count in zip(datasets, counts)
        ]
    )
    return DataLoader(sample, batch_size=batch_size, shuffle=False, drop_last=False)
