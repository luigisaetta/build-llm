"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Tokenization of text with the BPE encodings of the tiktoken library.
"""

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

DEFAULT_ENCODING = "gpt2"

# Short descriptions of the tiktoken encodings, shown to the user.
# Encodings missing from this table are still usable.
ENCODING_DESCRIPTIONS = {
    "gpt2": "GPT-2 (50,257 tokens), the tokenizer used in the book",
    "r50k_base": "GPT-3 models such as davinci (50,257 tokens)",
    "p50k_base": "Codex and text-davinci-002/003 (50,281 tokens)",
    "p50k_edit": "Edit models such as text-davinci-edit-001",
    "cl100k_base": "GPT-3.5 Turbo and GPT-4 (about 100,000 tokens)",
    "o200k_base": "GPT-4o and later models (about 200,000 tokens)",
    "o200k_harmony": "gpt-oss models (o200k_base plus chat format tokens)",
}


@dataclass
class Token:
    """One token of a tokenized text.

    Attributes:
        id: The token ID in the vocabulary of the encoding.
        text: The text of the token. If the token holds only part of a
            multi-byte UTF-8 character (for example a piece of an emoji),
            the missing part is shown as the replacement character U+FFFD.
        byte_values: The raw bytes of the token, as integers from 0 to 255.
        is_partial_character: True if the bytes are not valid UTF-8 on their
            own, because a character is split across tokens.
    """

    id: int
    text: str
    byte_values: list[int]
    is_partial_character: bool


@dataclass
class TokenizationResult:
    """The result of the tokenization of a text.

    Attributes:
        encoding: The name of the encoding used.
        vocabulary_size: The number of token IDs in the encoding.
        tokens: The tokens, in text order.
        token_count: The number of tokens.
        character_count: The number of characters of the input text.
        word_count: The number of whitespace-separated words.
        characters_per_token: The average number of characters per token,
            or 0.0 for an empty text.
    """

    encoding: str
    vocabulary_size: int
    tokens: list[Token]
    token_count: int
    character_count: int
    word_count: int
    characters_per_token: float


def available_encodings() -> list[str]:
    """Return the names of the encodings provided by tiktoken."""
    return tiktoken.list_encoding_names()


def describe_encoding(name: str) -> str:
    """Return a short description of an encoding, or "" if none is known."""
    return ENCODING_DESCRIPTIONS.get(name, "")


@lru_cache(maxsize=None)
def get_encoding(name: str) -> tiktoken.Encoding:
    """Load a tiktoken encoding, keeping it in memory for later calls.

    The first time an encoding is used, tiktoken downloads its vocabulary
    file and stores it in a local cache; later loads work offline.

    Args:
        name: The encoding name, for example "gpt2".

    Returns:
        The encoding.

    Raises:
        ValueError: If the encoding name is unknown.
    """
    if name not in available_encodings():
        known = ", ".join(available_encodings())
        raise ValueError(f"Unknown encoding '{name}'. Available encodings: {known}")
    return tiktoken.get_encoding(name)


def bytes_to_token(token_id: int, token_bytes: bytes) -> Token:
    """Build a Token from its ID and its raw bytes.

    Args:
        token_id: The token ID.
        token_bytes: The bytes that the token represents.

    Returns:
        The Token.
    """
    try:
        text = token_bytes.decode("utf-8")
        is_partial_character = False
    except UnicodeDecodeError:
        text = token_bytes.decode("utf-8", errors="replace")
        is_partial_character = True
    return Token(token_id, text, list(token_bytes), is_partial_character)


def tokenize(text: str, encoding: tiktoken.Encoding) -> TokenizationResult:
    """Split a text into tokens with the given encoding.

    Special tokens such as "<|endoftext|>" found in the text are encoded as
    plain text: they are not interpreted as control tokens.

    Args:
        text: The text to tokenize.
        encoding: The encoding to use, for example from get_encoding().

    Returns:
        The tokens of the text and some statistics.
    """
    # By default tiktoken raises an error when the text contains a special
    # token. An empty "disallowed_special" set encodes it as plain text,
    # which is the right behavior for text that comes from a user.
    token_ids = encoding.encode(text, disallowed_special=())
    tokens = [
        bytes_to_token(token_id, encoding.decode_single_token_bytes(token_id))
        for token_id in token_ids
    ]

    token_count = len(tokens)
    character_count = len(text)
    characters_per_token = character_count / token_count if token_count else 0.0

    return TokenizationResult(
        encoding=encoding.name,
        vocabulary_size=encoding.n_vocab,
        tokens=tokens,
        token_count=token_count,
        character_count=character_count,
        word_count=len(text.split()),
        characters_per_token=round(characters_per_token, 2),
    )
