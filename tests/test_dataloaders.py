"""
Author: L. Saetta
Date Last Modified: 2026-10-30
License: MIT
Description: Tests for the next-token prediction dataset and data loader.
"""

import pytest
import tiktoken
import torch

from build_llm.dataloaders import (
    END_OF_TEXT,
    GPTDatasetV1,
    create_dataloader_v1,
    split_documents,
)


class CharacterTokenizer:  # pylint: disable=too-few-public-methods
    """Offline stand-in for a tiktoken encoding: one token per character."""

    def encode(self, text, allowed_special=None):  # pylint: disable=unused-argument
        """Return the code point of each character."""
        return [ord(character) for character in text]


TEXT = "abcdefghij"  # Ten tokens: 97, 98, ..., 106


def test_targets_are_inputs_shifted_by_one():
    """Each target sequence is the input sequence moved one token ahead."""
    dataset = GPTDatasetV1(TEXT, CharacterTokenizer(), max_length=4, stride=4)

    inputs, targets = dataset[0]

    assert inputs.tolist() == [97, 98, 99, 100]
    assert targets.tolist() == [98, 99, 100, 101]
    assert inputs.dtype == torch.long


@pytest.mark.parametrize(
    ("max_length", "stride", "expected_pairs"),
    [(4, 4, 2), (4, 1, 6), (9, 1, 1)],
)
def test_number_of_windows(max_length, stride, expected_pairs):
    """A window needs max_length + 1 tokens; windows start every stride tokens."""
    dataset = GPTDatasetV1(TEXT, CharacterTokenizer(), max_length, stride)

    assert len(dataset) == expected_pairs


def test_text_too_short_raises():
    """A text with max_length tokens or fewer cannot form any pair."""
    with pytest.raises(ValueError, match="at least 11"):
        GPTDatasetV1(TEXT, CharacterTokenizer(), max_length=10, stride=1)


@pytest.mark.parametrize(("max_length", "stride"), [(0, 1), (4, 0)])
def test_invalid_window_raises(max_length, stride):
    """max_length and stride must be positive."""
    with pytest.raises(ValueError, match="positive"):
        GPTDatasetV1(TEXT, CharacterTokenizer(), max_length, stride)


def test_dataloader_batches_and_drop_last():
    """Batches have shape (batch_size, max_length); drop_last removes a partial one."""
    # 6 pairs with stride 1: batches of 4 give one full batch and one of 2
    kept = create_dataloader_v1(
        CharacterTokenizer(), TEXT, batch_size=4, max_length=4, stride=1, shuffle=False
    )
    dropped = create_dataloader_v1(
        CharacterTokenizer(),
        TEXT,
        batch_size=4,
        max_length=4,
        stride=1,
        shuffle=False,
        drop_last=False,
    )

    inputs, targets = next(iter(kept))
    assert inputs.shape == targets.shape == (4, 4)
    assert len(kept) == 1
    assert len(dropped) == 2


def test_dataloader_options_are_keyword_only():
    """Positional options are rejected, to avoid passing them in the wrong order."""
    # The wrong call is intentional: pylint flags it, as it should.
    # pylint: disable=too-many-function-args,missing-kwoa
    with pytest.raises(TypeError):
        create_dataloader_v1(CharacterTokenizer(), TEXT, 4, 4, 1)
    # pylint: enable=too-many-function-args,missing-kwoa


def test_split_documents_keeps_every_document_in_both_sets():
    """Each document contributes its start to training and its end to validation."""
    # 8 characters * 0.75 = 6 for training; 4 * 0.75 = 3
    train, val = split_documents(["aaaaaaAA", "bbbB"], train_ratio=0.75)

    assert train == "aaaaaa" + END_OF_TEXT + "bbb"
    assert val == "AA" + END_OF_TEXT + "B"


def test_split_documents_custom_separator():
    """The separator between documents can be chosen."""
    train, val = split_documents(["abcd", "efgh"], train_ratio=0.5, separator="|")

    assert (train, val) == ("ab|ef", "cd|gh")


@pytest.mark.network
def test_end_of_text_is_a_single_gpt2_token():
    """The separator is encoded as the GPT-2 end-of-text token, ID 50256.

    tiktoken may download the GPT-2 encoding the first time.
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    assert tokenizer.encode(END_OF_TEXT, allowed_special={END_OF_TEXT}) == [50256]


@pytest.mark.parametrize(
    ("documents", "train_ratio"), [([], 0.9), (["text"], 0.0), (["text"], 1.0)]
)
def test_split_documents_rejects_invalid_inputs(documents, train_ratio):
    """An empty list or a ratio outside (0, 1) raises an error."""
    with pytest.raises(ValueError):
        split_documents(documents, train_ratio)
