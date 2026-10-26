"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Checks that every Python file starts with the standard module header.
"""

import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# An optional shebang, then the docstring with the four required fields.
HEADER_PATTERN = re.compile(
    r'(#![^\n]*\n)?"""\n'
    r"Author: L\. Saetta\n"
    r"Date Last Modified: \d{4}-\d{2}-\d{2}\n"
    r"License: MIT\n"
    r"Description: +\S"
)


def find_python_files() -> list[Path]:
    """Return the Python files of the repository, skipping hidden folders."""
    files = []
    for path in sorted(REPOSITORY_ROOT.rglob("*.py")):
        relative_parts = path.relative_to(REPOSITORY_ROOT).parts
        if not any(part.startswith(".") for part in relative_parts):
            files.append(path)
    return files


@pytest.mark.parametrize(
    "path", find_python_files(), ids=lambda path: str(path.relative_to(REPOSITORY_ROOT))
)
def test_file_starts_with_header(path):
    """The file starts with the header defined in AGENTS.md."""
    content = path.read_text(encoding="utf-8")

    assert HEADER_PATTERN.match(content), f"Missing or invalid header in {path}"
