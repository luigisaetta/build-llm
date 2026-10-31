"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Read the manifest that defines a training corpus, and check that
    a folder holds exactly its books, unmodified.

The manifest is a JSON list with one entry per book. It is the single
definition of the corpus: which books, from which source and revision, and
the SHA-256 checksum of each converted text. Preparation scripts use it to
download or verify the books, and training scripts use it to know which files
to read. A book missing from the folder, a modified book, or an extra text
file that is not in the manifest would silently change the training data, so
check_corpus refuses all three.
"""

import hashlib
import json
from pathlib import Path

REQUIRED_FIELDS = ("filename", "title", "author", "repository", "revision", "sha256")
_CHUNK_SIZE = 1024 * 1024


class CorpusError(ValueError):
    """The books folder does not match the manifest."""


def read_manifest(path: Path) -> list[dict]:
    """Read and validate a corpus manifest.

    Args:
        path: The JSON manifest file.

    Returns:
        The entries, in the order of the file.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If an entry lacks a required field, or two entries have
            the same file name.
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    names = set()
    for number, entry in enumerate(entries, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"Manifest entry {number} lacks {', '.join(missing)}.")
        if entry["filename"] in names:
            raise ValueError(f"Duplicate manifest entry: {entry['filename']}")
        names.add(entry["filename"])
    return entries


def file_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum of a file, reading it in chunks.

    Args:
        path: The file to hash.

    Returns:
        The checksum as a lowercase hexadecimal string.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_corpus(entries: list[dict], books_dir: Path) -> list[Path]:
    """Check that a folder holds exactly the books of a manifest, unmodified.

    Args:
        entries: The manifest entries, from read_manifest.
        books_dir: The folder of the .txt books.

    Returns:
        The paths of the books, sorted by file name. The order is fixed, so
        the training data are the same at every run.

    Raises:
        CorpusError: If books are missing or modified, or the folder holds
            .txt files that are not in the manifest. The message lists all the
            problems.
    """
    books_dir = Path(books_dir)
    expected = {entry["filename"]: entry for entry in entries}
    present = (
        {path.name for path in books_dir.glob("*.txt") if path.is_file()}
        if books_dir.is_dir()
        else set()
    )

    problems = []
    missing = sorted(set(expected) - present)
    if missing:
        problems.append("missing: " + ", ".join(missing))
    extra = sorted(present - set(expected))
    if extra:
        problems.append("not in the manifest: " + ", ".join(extra))
    modified = sorted(
        name
        for name in set(expected) & present
        if file_sha256(books_dir / name) != expected[name]["sha256"]
    )
    if modified:
        problems.append("modified (checksum mismatch): " + ", ".join(modified))
    if problems:
        raise CorpusError(
            f"The books in {books_dir} do not match the manifest:\n  "
            + "\n  ".join(problems)
        )
    return [books_dir / name for name in sorted(expected)]
