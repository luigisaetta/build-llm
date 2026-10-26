"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Tests of the chap1.tokenize_text command-line tool.
"""

import io
import json

import pytest

from build_llm.text_sources import TextSourceError
from chap1 import tokenize_text
from tests.conftest import HELLO


@pytest.fixture(autouse=True)
def use_tiny_encoding(monkeypatch, tiny_encoding):
    """Replace the real encodings with the tiny offline one."""
    monkeypatch.setattr(tokenize_text, "get_encoding", lambda name: tiny_encoding)


def test_text_output(capsys):
    """The readable output lists the counts and the tokens."""
    exit_code = tokenize_text.main(["--text", "hello world"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Tokens:" in output
    assert f"{HELLO}  'hello'" in output


def test_json_output(capsys):
    """The JSON output contains all the tokens."""
    exit_code = tokenize_text.main(["--text", "hello", "--format", "json"])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["token_count"] == 1
    assert result["tokens"][0]["id"] == HELLO


def test_show_limits_the_listed_tokens(capsys):
    """Only the first N tokens are listed, with a note about the others."""
    tokenize_text.main(["--text", "abcdef", "--show", "2"])

    assert "and 4 more tokens" in capsys.readouterr().out


def test_reads_standard_input(monkeypatch, capsys):
    """Without a source option, the text is read from standard input."""
    monkeypatch.setattr("sys.stdin", io.StringIO("hello"))

    tokenize_text.main(["--format", "json"])

    assert json.loads(capsys.readouterr().out)["token_count"] == 1


def test_reads_a_file(tmp_path, capsys):
    """The --file option reads a UTF-8 file."""
    path = tmp_path / "input.txt"
    path.write_text("hello", encoding="utf-8")

    tokenize_text.main(["--file", str(path), "--format", "json"])

    assert json.loads(capsys.readouterr().out)["token_count"] == 1


def test_url_error_gives_exit_code_1(monkeypatch, capsys):
    """When the URL cannot be read, the tool prints an error and returns 1."""

    def failing_fetch(url):
        raise TextSourceError(f"Cannot read '{url}'")

    monkeypatch.setattr(tokenize_text, "fetch_text_from_url", failing_fetch)

    exit_code = tokenize_text.main(["--url", "https://x.test/"])

    assert exit_code == 1
    assert "Error: Cannot read" in capsys.readouterr().err


def test_unknown_encoding_is_rejected():
    """An encoding that is not in the list is an argument error (exit code 2)."""
    with pytest.raises(SystemExit) as exit_info:
        tokenize_text.main(["--text", "hello", "--encoding", "not-an-encoding"])

    assert exit_info.value.code == 2


def test_list_encodings(capsys):
    """The --list-encodings option prints the encodings."""
    exit_code = tokenize_text.main(["--list-encodings"])

    assert exit_code == 0
    assert "gpt2" in capsys.readouterr().out
