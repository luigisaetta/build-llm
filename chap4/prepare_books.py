"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Download and verify the expanded Standard Ebooks training corpus.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import tiktoken

from build_llm.ebooks import convert_standard_ebook

CHAPTER_DIR = Path(__file__).resolve().parent
BOOKS_DIR = CHAPTER_DIR / "books"
MANIFEST_PATH = CHAPTER_DIR / "books_manifest.json"


def download(url: str) -> bytes:
    """Read an official source with a timeout and an identifying user agent.

    Args:
        url: HTTPS source URL.

    Returns:
        Response bytes.

    Raises:
        OSError: If the network request fails.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "build-llm-corpus/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def prepare_book(entry: dict) -> None:
    """Restore one missing book and verify its converted SHA-256 checksum.

    Args:
        entry: Manifest record with filename, repository, revision, and checksum.

    Raises:
        ValueError: If the downloaded conversion differs from the recorded text.
        OSError: If a source cannot be downloaded or a file cannot be written.
    """
    path = BOOKS_DIR / entry["filename"]
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {path.name}; existing file preserved."
            )
        print(f"Verified {path.name}")
        return
    archive = download(
        f"https://codeload.github.com/standardebooks/{entry['repository']}"
        f"/tar.gz/{entry['revision']}"
    )
    book = convert_standard_ebook(archive)
    data = book.text.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Converted checksum mismatch for {path.name}.")
    path.write_bytes(data)
    print(f"Prepared {path.name}")


def add_book(repository: str, tokenizer: tiktoken.Encoding) -> dict:
    """Download a new edition and return a pinned provenance record.

    Args:
        repository: Repository name within the official standardebooks organization.
        tokenizer: GPT-2 tokenizer used to measure the resulting training text.

    Returns:
        Source metadata, conversion details, token count, and output checksum.

    Raises:
        ValueError: If the repository name is invalid or would overwrite a book.
        OSError: If the source cannot be downloaded or written.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", repository) or "_" not in repository:
        raise ValueError(f"Invalid Standard Ebooks repository: {repository}")
    filename = repository.split("_")[1] + ".txt"
    path = BOOKS_DIR / filename
    if path.exists():
        raise ValueError(f"Refusing to overwrite {filename}.")
    commit = json.loads(
        download(
            f"https://api.github.com/repos/standardebooks/{repository}/commits/HEAD"
        )
    )
    revision = commit["sha"]
    archive = download(
        f"https://codeload.github.com/standardebooks/{repository}/tar.gz/{revision}"
    )
    book = convert_standard_ebook(archive)
    data = book.text.encode("utf-8")
    entry = {
        "filename": filename,
        "title": book.title,
        "author": book.author,
        "source": book.source,
        "repository": repository,
        "revision": revision,
        "license": "Public-domain source text; Standard Ebooks contributions CC0-1.0",
        "changes": (
            "Modified: bodymatter XHTML converted to plain text in spine order; "
            "headings kept; front/back matter, notes, note references, and "
            "Gutenberg boilerplate omitted; invisible formatting characters "
            "and typographic spaces normalized."
        ),
        "chapters": book.chapters,
        "characters": len(book.text),
        "tokens": len(tokenizer.encode(book.text)),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    path.write_bytes(data)
    print(f"Added {filename}: {entry['tokens']:,} tokens", flush=True)
    return entry


def main() -> int:
    """Prepare pinned books, optionally adding editions until a token target.

    Returns:
        Zero when preparation completes successfully.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("Description: ")[1].splitlines()[0]
    )
    parser.add_argument(
        "--add",
        nargs="+",
        default=[],
        metavar="REPOSITORY",
        help="Official Standard Ebooks repositories to add, in priority order.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=10_000_000,
        help="Stop adding books once the local corpus reaches this many GPT-2 tokens.",
    )
    args = parser.parse_args()
    BOOKS_DIR.mkdir(exist_ok=True)
    entries = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists()
        else []
    )
    for entry in entries:
        prepare_book(entry)
    tokenizer = tiktoken.get_encoding("gpt2")
    total = sum(
        len(tokenizer.encode(path.read_text(encoding="utf-8")))
        for path in BOOKS_DIR.glob("*.txt")
    )
    known = {entry["repository"] for entry in entries}
    for repository in args.add:
        if total >= args.target_tokens:
            break
        if repository in known:
            continue
        entry = add_book(repository, tokenizer)
        entries.append(entry)
        known.add(repository)
        MANIFEST_PATH.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        total += entry["tokens"]
        print(f"Corpus: {total:,} GPT-2 tokens", flush=True)
    num_books = len(list(BOOKS_DIR.glob("*.txt")))
    print(f"Ready: {total:,} GPT-2 tokens across {num_books} books.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
