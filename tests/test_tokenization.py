"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Tests of build_llm.tokenization, with an offline tiny encoding.
"""

import pytest

from build_llm.tokenization import (
    available_encodings,
    bytes_to_token,
    describe_encoding,
    get_encoding,
    tokenize,
)
from tests.conftest import END_OF_TEXT, HELLO


def test_merged_word_is_one_token(tiny_encoding):
    """A word covered by the merges becomes a single token."""
    result = tokenize("hello", tiny_encoding)

    assert [token.id for token in result.tokens] == [HELLO]
    assert result.tokens[0].text == "hello"


def test_unmerged_text_falls_back_to_bytes(tiny_encoding):
    """Text without merges is split into one token per byte."""
    result = tokenize("abc", tiny_encoding)

    assert [token.id for token in result.tokens] == [ord("a"), ord("b"), ord("c")]


def test_statistics(tiny_encoding):
    """The counts describe the input text and its tokens."""
    result = tokenize("hello hello", tiny_encoding)

    assert result.encoding == "tiny"
    assert result.token_count == 3  # "hello", " ", "hello"
    assert result.character_count == 11
    assert result.word_count == 2
    assert result.characters_per_token == pytest.approx(3.67)


def test_tokens_rebuild_the_text(tiny_encoding):
    """Joining the token bytes gives back the original text."""
    text = "hello, world!\nhello\tagain"
    result = tokenize(text, tiny_encoding)

    all_bytes = bytes(value for token in result.tokens for value in token.byte_values)
    assert all_bytes.decode("utf-8") == text


def test_empty_text(tiny_encoding):
    """An empty text has no tokens, and no division by zero."""
    result = tokenize("", tiny_encoding)

    assert not result.tokens
    assert result.token_count == 0
    assert result.characters_per_token == 0.0


def test_special_token_is_encoded_as_plain_text(tiny_encoding):
    """A special token in user text is not turned into its control token."""
    result = tokenize("<|endoftext|>", tiny_encoding)

    assert END_OF_TEXT not in [token.id for token in result.tokens]
    assert "".join(token.text for token in result.tokens) == "<|endoftext|>"


def test_split_character_is_marked_as_partial(tiny_encoding):
    """Without merges, the 4 bytes of an emoji become 4 partial tokens."""
    result = tokenize("😀", tiny_encoding)

    assert result.token_count == 4
    assert all(token.is_partial_character for token in result.tokens)
    assert result.tokens[0].text == "�"


def test_bytes_to_token_with_valid_utf8():
    """Complete characters are decoded normally."""
    token = bytes_to_token(42, "è".encode("utf-8"))

    assert token.text == "è"
    assert token.byte_values == [0xC3, 0xA8]
    assert not token.is_partial_character


def test_get_encoding_rejects_unknown_name():
    """An unknown encoding name raises ValueError, listing the valid names."""
    with pytest.raises(ValueError, match="gpt2"):
        get_encoding("not-an-encoding")


def test_known_encodings():
    """The book's tokenizer is available and described."""
    assert "gpt2" in available_encodings()
    assert describe_encoding("gpt2")
    assert describe_encoding("not-an-encoding") == ""


@pytest.mark.network
def test_real_gpt2_encoding():
    """The real GPT-2 encoding gives the well-known IDs (may download it)."""
    result = tokenize("Hello, world", get_encoding("gpt2"))

    assert [token.id for token in result.tokens] == [15496, 11, 995]
