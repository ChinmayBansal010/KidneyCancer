"""Shared logging helpers for scripts and package modules."""

from __future__ import annotations

import logging
from pathlib import Path


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    logger_name: str,
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create or reuse a consistently configured logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_files = {
            getattr(handler, "baseFilename", None) for handler in logger.handlers
        }
        if str(log_path.resolve()) not in existing_files:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
