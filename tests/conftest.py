"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Shared pytest fixtures, including a tiny offline BPE encoding.
"""

import pytest
import tiktoken

# Token IDs of the tiny encoding, after the 256 single-byte tokens.
HE, LL, HELL, HELLO, END_OF_TEXT = 256, 257, 258, 259, 260


@pytest.fixture
def tiny_encoding() -> tiktoken.Encoding:
    """A tiny BPE encoding that works without downloading a vocabulary.

    Every byte is a token (IDs 0-255), and four merges build "hello":
    "h"+"e" -> "he", "l"+"l" -> "ll", "he"+"ll" -> "hell", "hell"+"o" -> "hello".
    """
    mergeable_ranks = {bytes([value]): value for value in range(256)}
    mergeable_ranks[b"he"] = HE
    mergeable_ranks[b"ll"] = LL
    mergeable_ranks[b"hell"] = HELL
    mergeable_ranks[b"hello"] = HELLO
    return tiktoken.Encoding(
        name="tiny",
        pat_str=r"\s?\w+|\s?[^\w\s]+|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|endoftext|>": END_OF_TEXT},
    )
