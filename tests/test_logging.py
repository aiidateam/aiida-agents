"""Unit tests for the logging helpers in ``aiida_agents._logging``."""

from __future__ import annotations

import logging
from pathlib import Path

from aiida_agents._logging import (
    TRACE_LOGGER_NAMES,
    _ConsoleFilter,
    _configure_logging,
    suppress_noisy_loggers,
)
from aiida_agents._settings import LoggingSettings


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


def _record(name: str) -> logging.LogRecord:
    return logging.LogRecord(name, logging.DEBUG, "path", 0, "msg", None, None)


def test_console_filter_blocks_trace_records() -> None:
    """The console filter drops trace-logger records and passes all others."""
    filt = _ConsoleFilter()
    assert filt.filter(_record("aiida_agents.cli")) is True
    assert filt.filter(_record("httpx")) is True
    for name in TRACE_LOGGER_NAMES:
        assert filt.filter(_record(name)) is False


def test_configure_logging_wires_handlers(tmp_path: Path) -> None:
    """With ``log_file`` set: a filtered console handler plus a file handler."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level

    try:
        _configure_logging(
            LoggingSettings(log_level="INFO", log_file=tmp_path / "agents.log")
        )

        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        other_handlers = [
            h for h in root.handlers if not isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert not any(
            isinstance(f, _ConsoleFilter) for h in file_handlers for f in h.filters
        )
        assert any(
            isinstance(f, _ConsoleFilter) for h in other_handlers for f in h.filters
        )
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            if handler not in saved_handlers:
                handler.close()
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
