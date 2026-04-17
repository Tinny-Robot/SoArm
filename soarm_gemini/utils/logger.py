"""
Structured logging utilities.

Provides a pre-configured logger factory and a dedicated GeminiLogger that
persists every LLM call (prompt + response) to timestamped files.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from soarm_gemini.config import GEMINI_LOG_DIR, LOG_DIR, LOG_LEVEL


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a logger with a consistent format and optional file handler.

    Args:
        name: Logger name (typically ``__name__``).
        level: Override log level string (e.g. "DEBUG").

    Returns:
        Configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    lvl = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(lvl)

    fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "soarm.log"), encoding="utf-8"
    )
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class GeminiLogger:
    """Persists every Gemini API call to a JSON-lines file for post-hoc debugging."""

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._dir = Path(log_dir or GEMINI_LOG_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = self._dir / f"gemini_{ts}.jsonl"
        self._seq = 0

    @property
    def path(self) -> Path:
        """Return the path to the current log file."""
        return self._path

    def log(self, prompt: str, response: str) -> None:
        """Append a prompt–response pair to the log file.

        Args:
            prompt: The user message sent to Gemini.
            response: The raw text returned by Gemini.
        """
        self._seq += 1
        record = {
            "seq": self._seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "response": response,
        }
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def configure_root_logger(level: Optional[str] = None) -> None:
    """One-shot configuration for the root logger (call once in main.py)."""
    lvl = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(lvl)

    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        ch.setFormatter(fmt)
        root.addHandler(ch)

        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(LOG_DIR, "soarm.log"), encoding="utf-8"
        )
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)
