"""Unit tests for the logging helpers in ``aiida_agents._logging``."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from aiida_agents._logging import (
    TRACE_LOGGER_NAMES,
    _ConsoleFilter,
    _configure_logging,
    suppress_noisy_loggers,
    trace_response,
)
from aiida_agents._settings import LoggingSettings, _LogLevel

_NOISY_LOGGERS = ["asyncio", "httpcore", "httpx", "openai", "chromadb", "markdown_it"]


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


@pytest.fixture
def _isolate_root_logging() -> Iterator[None]:
    """Snapshot and restore global logging state around a ``_configure_logging``.

    It runs ``basicConfig(force=True)`` and ``suppress_noisy_loggers``, both of
    which mutate process-wide state that would otherwise leak into later tests.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    saved_noisy = {n: logging.getLogger(n).level for n in _NOISY_LOGGERS}
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            if handler not in saved_handlers:
                handler.close()
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
        for name, level in saved_noisy.items():
            logging.getLogger(name).setLevel(level)


@pytest.mark.parametrize("with_file", [True, False], ids=["file", "console-only"])
@pytest.mark.usefixtures("_isolate_root_logging")
def test_configure_logging_wires_handlers(with_file: bool, tmp_path: Path) -> None:
    """The console handler is always present and filtered; the file handler
    appears only when ``log_file`` is set, and is unfiltered."""
    log_file = tmp_path / "agents.log" if with_file else None
    _configure_logging(LoggingSettings(log_level="INFO", log_file=log_file))

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    console_handlers = [
        h for h in root.handlers if not isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == (1 if with_file else 0)
    assert any(
        isinstance(f, _ConsoleFilter) for h in console_handlers for f in h.filters
    )
    assert not any(
        isinstance(f, _ConsoleFilter) for h in file_handlers for f in h.filters
    )


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        pytest.param("WARNING", logging.WARNING, id="warning"),
        pytest.param("INFO", logging.INFO, id="info-honoured"),
        pytest.param("DEBUG", logging.DEBUG, id="debug"),
    ],
)
@pytest.mark.usefixtures("_isolate_root_logging")
def test_console_handler_honours_log_level(log_level: _LogLevel, expected: int) -> None:
    """The console handler's level tracks ``log_level`` verbatim.

    Regression guard for the dropped INFO->WARNING special-case: ``log_level=INFO``
    must actually surface INFO on the console rather than be demoted to WARNING, so
    a re-introduced special-case or a changed mapping fails here. The WARNING
    default itself is pinned by ``test_settings`` (declared-defaults).
    """
    _configure_logging(LoggingSettings(log_level=log_level))
    console_handler = next(
        h
        for h in logging.getLogger().handlers
        if any(isinstance(f, _ConsoleFilter) for f in h.filters)
    )
    assert console_handler.level == expected


@pytest.mark.usefixtures("_isolate_root_logging")
def test_trace_reaches_file_at_info_level(tmp_path: Path) -> None:
    """A DEBUG trace lands in the log file even with the console level at INFO.

    End-to-end guard for the aa68795 fix: file logging must capture the
    agent-reply/tool-call traces regardless of ``AIIDA_AGENTS_LOG_LEVEL``. The
    handler-wiring and logger-level halves are pinned separately; this exercises
    the whole path (record -> propagation -> file).
    """
    log_file = tmp_path / "trace.log"
    _configure_logging(LoggingSettings(log_level="INFO", log_file=log_file))

    trace_response("hello from the agent")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "hello from the agent" in log_file.read_text()
