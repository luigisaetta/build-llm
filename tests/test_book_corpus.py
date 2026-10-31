"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Tests for the corpus manifest and the check of the books folder.
"""

import json
from pathlib import Path

import pytest

from build_llm.book_corpus import (
    CorpusError,
    check_corpus,
    file_sha256,
    read_manifest,
)


@pytest.fixture(name="corpus")
def fixture_corpus(tmp_path, write_manifest):
    """A folder with two books and the manifest that lists them."""
    books_dir = tmp_path / "books"
    books_dir.mkdir()
    for name in ("b.txt", "a.txt"):
        (books_dir / name).write_text(f"Text of {name}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, books_dir, ["b.txt", "a.txt"])
    return manifest, books_dir


def test_a_matching_folder_gives_the_books_sorted(corpus):
    """The paths are sorted by file name, whatever the order of the manifest."""
    manifest, books_dir = corpus

    paths = check_corpus(read_manifest(manifest), books_dir)

    assert [path.name for path in paths] == ["a.txt", "b.txt"]


def test_every_problem_is_reported(corpus):
    """Missing, extra, and modified books are listed together."""
    manifest, books_dir = corpus
    (books_dir / "a.txt").unlink()
    (books_dir / "b.txt").write_text("Edited", encoding="utf-8")
    (books_dir / "c.txt").write_text("Extra", encoding="utf-8")
    (books_dir / "README.md").write_text("Not a book", encoding="utf-8")

    with pytest.raises(CorpusError) as error:
        check_corpus(read_manifest(manifest), books_dir)

    message = str(error.value)
    assert "missing: a.txt" in message
    assert "not in the manifest: c.txt" in message
    assert "modified (checksum mismatch): b.txt" in message
    assert "README.md" not in message


def test_a_missing_folder_means_missing_books(corpus):
    """Without the folder, all the books are missing."""
    manifest, books_dir = corpus

    with pytest.raises(CorpusError, match="missing: a.txt, b.txt"):
        check_corpus(read_manifest(manifest), books_dir / "absent")


def test_manifest_entries_are_validated(tmp_path):
    """Entries need the required fields and distinct file names."""
    entry = {
        "filename": "a.txt",
        "title": "A",
        "author": "Test",
        "repository": "test_a",
        "revision": "0" * 40,
        "sha256": "0" * 64,
    }
    path = tmp_path / "manifest.json"

    path.write_text(json.dumps([entry, entry]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)

    path.write_text(json.dumps([{"filename": "a.txt"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks title"):
        read_manifest(path)


def test_the_real_manifest_matches_the_committed_books():
    """The 12 books committed to the repository match their manifest checksums.

    The other books are downloaded data and may be absent from a fresh clone,
    so only the committed ones are checked here.
    """
    chapter = Path(__file__).resolve().parent.parent / "chap4"
    entries = read_manifest(chapter / "books_manifest.json")
    committed = [entry for entry in entries if entry.get("in_repository")]

    assert len(committed) == 12
    for entry in committed:
        path = chapter / "books" / entry["filename"]
        assert file_sha256(path) == entry["sha256"], entry["filename"]
