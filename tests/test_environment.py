"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Tests of the Python environment checks in build_llm.environment.
"""

import pytest

from build_llm.environment import (
    check_packages,
    is_python_version_supported,
    parse_requirement_name,
    read_requirement_names,
)


@pytest.mark.parametrize(
    "version, expected",
    [((3, 11), True), ((3, 12, 4), True), ((4, 0), True), ((3, 10, 9), False)],
)
def test_is_python_version_supported(version, expected):
    """Versions from 3.11 on are supported."""
    assert is_python_version_supported(version) is expected


def test_running_python_is_supported():
    """The interpreter running the tests satisfies the project requirement."""
    assert is_python_version_supported()


@pytest.mark.parametrize(
    "line, expected",
    [
        ("torch>=2.2", "torch"),
        ("tiktoken", "tiktoken"),
        ("black[jupyter]==24.1  # formatter", "black"),
        ("numpy ; python_version >= '3.11'", "numpy"),
        ("", None),
        ("   ", None),
        ("# a comment", None),
        ("-r requirements.txt", None),
    ],
)
def test_parse_requirement_name(line, expected):
    """The package name is extracted; blank lines, comments, options are skipped."""
    assert parse_requirement_name(line) == expected


def test_read_requirement_names(tmp_path):
    """All the package names of a requirements file are read, in order."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# Runtime\ntorch>=2.2\n\ntiktoken>=0.7\n", "utf-8")

    assert read_requirement_names(requirements) == ["torch", "tiktoken"]


def test_read_requirement_names_missing_file(tmp_path):
    """A missing requirements file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_requirement_names(tmp_path / "missing.txt")


def test_check_packages_installed_and_missing():
    """Installed packages report a version; missing packages report None."""
    statuses = check_packages(["pytest", "surely-not-a-real-package"])
    installed, missing = statuses[0], statuses[1]

    assert installed.is_installed
    assert installed.installed_version
    assert not missing.is_installed
    assert missing.installed_version is None
