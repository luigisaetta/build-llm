"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Smoke tests for the build_llm package.
"""

import build_llm


def test_package_exposes_version():
    """The package exposes a non-empty version string."""
    assert isinstance(build_llm.__version__, str)
    assert build_llm.__version__
