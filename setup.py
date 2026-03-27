"""Compatibility entry point for setuptools-based builds.

The package metadata lives in ``pyproject.toml``.
This file exists so legacy tooling that still expects ``setup.py`` continues
to work cleanly for the repository.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
