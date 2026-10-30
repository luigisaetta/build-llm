"""
Author: L. Saetta
Date Last Modified: 2026-10-30
License: Apache-2.0
Description: Dataset and data loader for next-token prediction: a text is split
    into input and target sequences with a sliding window. Also a per-document
    training and validation split, for datasets of several texts.

Origin: derived from the code of Chapter 2 of LLMs-from-scratch
(https://github.com/rasbt/LLMs-from-scratch), Copyright 2023-2026 Sebastian
Raschka, licensed under the Apache License 2.0 (see
LICENSES/Apache-2.0-LLMs-from-scratch.txt). Modified by L. Saetta (2026):
docstrings, type hints, and input validation added; keyword-only options;
split_documents added.
"""

from collections.abc import Sequence

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

# The GPT-2 token that marks the end of a document. Joining documents with it
# tells the model, as in GPT-2 pre-training, where one text ends and the next
# one begins, instead of presenting them as one continuous text.
END_OF_TEXT = "<|endoftext|>"


class GPTDatasetV1(Dataset):
    """Input and target token sequences for next-token prediction.

    The text is tokenized once, then a window of max_length tokens slides over
    it, moving by stride tokens each time. The target of each window is the
    same window shifted by one token: target[i] is the token that follows
    input[i]. With stride equal to max_length the windows do not overlap.

    Args:
        text: The training text.
        tokenizer: A tiktoken encoding, such as tiktoken.get_encoding("gpt2").
        max_length: Number of tokens per sequence.
        stride: Number of tokens between the starts of two windows.

    Raises:
        ValueError: If max_length or stride is not positive, or if the text has
            fewer than max_length + 1 tokens, so that not even one input and
            target pair fits.
    """

    def __init__(
        self, text: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ):
        if max_length < 1 or stride < 1:
            raise ValueError("max_length and stride must be positive integers.")

        # Tokenize the entire text; <|endoftext|> may separate documents
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        if len(token_ids) <= max_length:
            raise ValueError(
                f"The text has {len(token_ids)} tokens: at least {max_length + 1} "
                "are needed for one sequence of max_length tokens and its target."
            )

        self.input_ids = []
        self.target_ids = []
        # Each window needs max_length + 1 tokens: the inputs and, shifted by
        # one, the targets. The last start is len(token_ids) - max_length - 1.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        """Return the number of input and target pairs."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pair at index idx.

        Args:
            idx: Index of the pair.

        Returns:
            The input and target token IDs, each of shape (max_length,).
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(  # pylint: disable=too-many-arguments
    tokenizer: tiktoken.Encoding,
    text: str,
    *,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a data loader of input and target batches from a text.

    The options after text are keyword-only, so that a call cannot pass them
    in the wrong order.

    Args:
        tokenizer: A tiktoken encoding, such as tiktoken.get_encoding("gpt2").
        text: The text to split into sequences.
        batch_size: Number of sequences per batch.
        max_length: Number of tokens per sequence.
        stride: Number of tokens between the starts of two windows.
        shuffle: Whether to shuffle the sequences at every epoch; use True for
            training and False for validation.
        drop_last: Whether to drop the last batch if it is smaller than
            batch_size.
        num_workers: Number of worker processes; 0 loads the data in the main
            process.

    Returns:
        A DataLoader that yields (inputs, targets) pairs, each of shape
        (batch_size, max_length).

    Raises:
        ValueError: If the text is too short, or max_length or stride is not
            positive.
    """
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def split_documents(
    documents: Sequence[str], train_ratio: float, separator: str = END_OF_TEXT
) -> tuple[str, str]:
    """Split every document into a training and a validation part, then join them.

    Each document is cut at the same fraction of its characters: the first
    train_ratio goes to training, the rest to validation. So every document is
    represented in both sets, in proportion to its length. Splitting the
    concatenated text once would instead put only the end of the last document
    in the validation set.

    Args:
        documents: The texts, for example one string per book.
        train_ratio: Fraction of each document used for training, between 0
            and 1, both excluded.
        separator: Text placed between two documents; by default the GPT-2
            end-of-text token.

    Returns:
        The training text and the validation text.

    Raises:
        ValueError: If documents is empty, or train_ratio is not between 0 and 1.
    """
    if not documents:
        raise ValueError("documents must contain at least one text.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1, both excluded.")

    train_parts, val_parts = [], []
    for document in documents:
        split_idx = int(train_ratio * len(document))
        train_parts.append(document[:split_idx])
        val_parts.append(document[split_idx:])
    return separator.join(train_parts), separator.join(val_parts)
