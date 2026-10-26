"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Checks of the Python environment: Python version and installed packages.
"""

import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MIN_PYTHON_VERSION = (3, 11)

# Characters that end the package name in a requirement line,
# for example "torch>=2.2" or "black[jupyter]==24.1".
_REQUIREMENT_NAME_DELIMITERS = "<>=!~[;@ "


@dataclass
class PackageStatus:
    """Installation status of a Python package.

    Attributes:
        name: The distribution name, as written in the requirements file.
        installed_version: The installed version, or None if not installed.
    """

    name: str
    installed_version: str | None

    @property
    def is_installed(self) -> bool:
        """True if the package is installed."""
        return self.installed_version is not None


def python_version_string() -> str:
    """Return the running Python version, for example "3.11.0"."""
    return ".".join(str(part) for part in sys.version_info[:3])


def is_python_version_supported(
    current: tuple[int, ...] | None = None,
    minimum: tuple[int, int] = MIN_PYTHON_VERSION,
) -> bool:
    """Check that the Python version is at least the minimum required.

    Args:
        current: The version to check. Defaults to the running interpreter.
        minimum: The minimum supported (major, minor) version.

    Returns:
        True if the version is supported.
    """
    if current is None:
        current = tuple(sys.version_info[:2])
    return tuple(current[:2]) >= minimum


def parse_requirement_name(line: str) -> str | None:
    """Extract the package name from one line of a requirements file.

    Args:
        line: A line such as "torch>=2.2  # comment".

    Returns:
        The package name, or None for blank lines, comments, and pip options
        (lines starting with "-", such as "-r other.txt").
    """
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None

    for position, character in enumerate(line):
        if character in _REQUIREMENT_NAME_DELIMITERS:
            return line[:position]
    return line


def read_requirement_names(path: Path) -> list[str]:
    """Read the package names listed in a requirements file.

    Args:
        path: Path to a requirements file.

    Returns:
        The package names, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = parse_requirement_name(line)
        if name is not None:
            names.append(name)
    return names


def check_packages(names: list[str]) -> list[PackageStatus]:
    """Check which packages are installed, and their versions.

    Args:
        names: The distribution names to check.

    Returns:
        One PackageStatus for each name, in the same order.
    """
    statuses = []
    for name in names:
        try:
            installed_version = version(name)
        except PackageNotFoundError:
            installed_version = None
        statuses.append(PackageStatus(name, installed_version))
    return statuses
