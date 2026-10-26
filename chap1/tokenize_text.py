"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Command-line tool that tokenizes a text with a tiktoken encoding.

Run it from the repository root. Examples:

    python -m chap1.tokenize_text --text "Hello, world!"
    python -m chap1.tokenize_text --encoding cl100k_base --file notes.txt
    python -m chap1.tokenize_text --url https://example.com --format json
    echo "Hello" | python -m chap1.tokenize_text
    python -m chap1.tokenize_text --list-encodings

Exit codes: 0 on success, 1 if the text cannot be read or tokenized,
2 for invalid command-line arguments.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from build_llm.text_sources import TextSourceError, fetch_text_from_url
from build_llm.tokenization import (
    DEFAULT_ENCODING,
    TokenizationResult,
    available_encodings,
    describe_encoding,
    get_encoding,
    tokenize,
)

DEFAULT_TOKENS_SHOWN = 40


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        arguments: The arguments to parse. Defaults to sys.argv[1:].

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m chap1.tokenize_text",
        description=(
            "Tokenize a text with a tiktoken encoding. The text is read from "
            "--text, --file, --url, or, if none is given, from standard input."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="the text to tokenize")
    source.add_argument("--file", type=Path, help="a UTF-8 text file to tokenize")
    source.add_argument("--url", help="a web page or text URL to tokenize")
    source.add_argument(
        "--list-encodings",
        action="store_true",
        help="list the available encodings and exit",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        choices=available_encodings(),
        help=f"the tiktoken encoding to use (default: {DEFAULT_ENCODING})",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format: readable text or JSON (default: text)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=DEFAULT_TOKENS_SHOWN,
        metavar="N",
        help=(
            "in text format, the number of tokens to list "
            f"(default: {DEFAULT_TOKENS_SHOWN}; 0 lists all of them)"
        ),
    )
    return parser.parse_args(arguments)


def read_input_text(arguments: argparse.Namespace) -> str:
    """Read the text to tokenize from the source selected by the user.

    Args:
        arguments: The parsed command-line arguments.

    Returns:
        The text.

    Raises:
        TextSourceError: If the URL cannot be read.
        OSError: If the file cannot be read.
    """
    if arguments.text is not None:
        return arguments.text
    if arguments.file is not None:
        return arguments.file.read_text(encoding="utf-8")
    if arguments.url is not None:
        return fetch_text_from_url(arguments.url).text
    return sys.stdin.read()


def print_encodings() -> None:
    """Print the available encodings with their descriptions."""
    for name in available_encodings():
        print(f"{name:<15} {describe_encoding(name)}")


def print_result(result: TokenizationResult, tokens_shown: int) -> None:
    """Print a tokenization result in a readable form.

    Args:
        result: The tokenization result.
        tokens_shown: The number of tokens to list; 0 lists all of them.
    """
    print(f"Encoding:             {result.encoding}")
    print(f"Vocabulary size:      {result.vocabulary_size:,}")
    print(f"Characters:           {result.character_count:,}")
    print(f"Words:                {result.word_count:,}")
    print(f"Tokens:               {result.token_count:,}")
    print(f"Characters per token: {result.characters_per_token}")

    if not result.tokens:
        return

    shown = result.tokens if tokens_shown == 0 else result.tokens[:tokens_shown]
    print()
    print(f"{'#':>6}  {'ID':>7}  Text")
    for position, token in enumerate(shown):
        # repr() makes spaces, newlines, and tabs visible.
        note = "  (part of a character)" if token.is_partial_character else ""
        print(f"{position:>6}  {token.id:>7}  {token.text!r}{note}")

    hidden = result.token_count - len(shown)
    if hidden > 0:
        print(f"... and {hidden:,} more tokens (use --show 0 to list all of them)")


def main(arguments: list[str] | None = None) -> int:
    """Run the command-line tool.

    Args:
        arguments: The command-line arguments. Defaults to sys.argv[1:].

    Returns:
        The exit code.
    """
    arguments = parse_arguments(arguments)

    if arguments.list_encodings:
        print_encodings()
        return 0

    try:
        text = read_input_text(arguments)
        result = tokenize(text, get_encoding(arguments.encoding))
    except (TextSourceError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if arguments.format == "json":
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print_result(result, arguments.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
