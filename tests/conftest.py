"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Shared pytest fixtures: a tiny offline BPE encoding, and a
    writer of corpus manifests for test books.
"""

import hashlib
import json

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


@pytest.fixture
def write_manifest():
    """Return a function that writes a manifest listing books of a folder.

    The function takes the manifest path, the books folder, and the file names
    to list; each entry gets the real SHA-256 checksum of its file.
    """

    def write(manifest_path, books_dir, names):
        entries = [
            {
                "filename": name,
                "title": name,
                "author": "Test",
                "repository": f"test_{name}",
                "revision": "0" * 40,
                "sha256": hashlib.sha256((books_dir / name).read_bytes()).hexdigest(),
                "tokens": 1,
            }
            for name in names
        ]
        manifest_path.write_text(json.dumps(entries), encoding="utf-8")
        return entries

    return write
