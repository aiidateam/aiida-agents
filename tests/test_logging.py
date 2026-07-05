"""Unit tests for the trace-logging helpers in ``aiida_agents._logging``."""

from __future__ import annotations

import logging

from aiida_agents._logging import suppress_noisy_loggers


def test_suppress_noisy_loggers() -> None:
    """suppress_noisy_loggers sets noisy third-party loggers to WARNING level."""
    loggers = ["asyncio", "httpcore", "httpx", "openai", "chromadb", "markdown_it"]
    initial_levels = {name: logging.getLogger(name).level for name in loggers}

    try:
        for name in loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)

        suppress_noisy_loggers()

        for name in loggers:
            assert logging.getLogger(name).level == logging.WARNING
    finally:
        for name, level in initial_levels.items():
            logging.getLogger(name).setLevel(level)
