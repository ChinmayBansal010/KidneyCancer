"""Kidney cancer detection package."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("kidneycancer")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Chinmay Bansal"

__all__ = ["__version__"]
