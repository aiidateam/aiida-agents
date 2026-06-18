"""Tests for the centralized settings (aiida_agents/_settings.py).

``ModelSettings``, ``OllamaSettings``, ``RagSettings``, and ``ServerSettings``
own the ``AIIDA_AGENTS_*`` configuration and replace the old ``python-dotenv`` +
``os.getenv`` reads. These tests pin the settings contract: declared defaults,
the conventional unprefixed ``OLLAMA_BASE_URL`` alias, ``.env`` file loading,
``str -> int``/``str -> Path`` coercion, case normalization, and failing fast
at load time on an unsupported provider / embedding backend / log level.

The tests are parametrized across the settings groups by behavior (defaults,
fail-fast, normalization, coercion) rather than duplicated per class. Each test
runs in an isolated working directory (``monkeypatch.chdir``) so a stray
``.env`` in the repo root can't leak into the assertions.
"""

from __future__ import annotations

import logging
import pathlib

import pytest
from pydantic import ValidationError

from aiida_agents._settings import (
    LoggingSettings,
    ModelSettings,
    OllamaSettings,
    RagSettings,
    ServerSettings,
    _Base,
    warn_on_unrecognized_settings,
)


@pytest.mark.parametrize(
    ("settings_cls", "clear_env", "expected"),
    [
        pytest.param(
            ModelSettings,
            ("AIIDA_AGENTS_PROVIDER", "AIIDA_AGENTS_MODEL", "AIIDA_AGENTS_BASE_URL"),
            {"provider": "ollama", "model": "qwen3.5:2b", "base_url": None},
            id="model",
        ),
        pytest.param(
            OllamaSettings,
            ("OLLAMA_BASE_URL",),
            {"base_url": "http://localhost:11434/v1"},
            id="ollama",
        ),
        pytest.param(
            RagSettings,
            (
                "AIIDA_AGENTS_EMBED_BACKEND",
                "AIIDA_AGENTS_EMBED_MODEL",
                "AIIDA_AGENTS_VECTOR_DB_PATH",
            ),
            {
                "embed_backend": "ollama",
                "embed_model": "mxbai-embed-large",
                "vector_db_path": pathlib.Path(".aiida_agents_vector_db"),
            },
            id="rag",
        ),
        pytest.param(
            ServerSettings,
            ("AIIDA_AGENTS_PORT",),
            {"port": 8000},
            id="server",
        ),
        pytest.param(
            LoggingSettings,
            ("AIIDA_AGENTS_LOG_LEVEL",),
            {"log_level": "INFO"},
            id="logging",
        ),
    ],
)
def test_defaults(
    settings_cls: type[_Base],
    clear_env: tuple[str, ...],
    expected: dict[str, object],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a clean environment, each group falls back to its declared defaults."""
    monkeypatch.chdir(tmp_path)  # empty dir, no .env to read
    for var in clear_env:
        monkeypatch.delenv(var, raising=False)
    settings = settings_cls()
    for field, value in expected.items():
        assert getattr(settings, field) == value


@pytest.mark.parametrize(
    ("settings_cls", "env_var", "bad_value", "match"),
    [
        pytest.param(
            ModelSettings,
            "AIIDA_AGENTS_PROVIDER",
            "no-such-provider",
            "literal_error",
            id="provider",
        ),
        pytest.param(
            RagSettings,
            "AIIDA_AGENTS_EMBED_BACKEND",
            "faiss",
            "literal_error",
            id="embed-backend",
        ),
        pytest.param(
            LoggingSettings,
            "AIIDA_AGENTS_LOG_LEVEL",
            "verbose",
            "literal_error",
            id="log-level",
        ),
    ],
)
def test_invalid_value_fails_fast(
    settings_cls: type[_Base],
    env_var: str,
    bad_value: str,
    match: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad value raises at load time, not deep inside a later call."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValidationError, match=match):
        settings_cls()


@pytest.mark.parametrize(
    ("settings_cls", "env_var", "raw", "field", "expected"),
    [
        pytest.param(
            ModelSettings,
            "AIIDA_AGENTS_PROVIDER",
            "Ollama",
            "provider",
            "ollama",
            id="provider-lowercased",
        ),
        pytest.param(
            RagSettings,
            "AIIDA_AGENTS_EMBED_BACKEND",
            "Ollama",
            "embed_backend",
            "ollama",
            id="embed-backend-lowercased",
        ),
        pytest.param(
            LoggingSettings,
            "AIIDA_AGENTS_LOG_LEVEL",
            "debug",
            "log_level",
            "DEBUG",
            id="log-level-uppercased",
        ),
    ],
)
def test_value_is_normalized(
    settings_cls: type[_Base],
    env_var: str,
    raw: str,
    field: str,
    expected: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed-case env values are normalized so they match the field's contract."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, raw)
    assert getattr(settings_cls(), field) == expected


@pytest.mark.parametrize(
    ("settings_cls", "env_var", "raw", "field", "expected"),
    [
        pytest.param(
            RagSettings,
            "AIIDA_AGENTS_VECTOR_DB_PATH",
            "/tmp/custom_db",
            "vector_db_path",
            pathlib.Path("/tmp/custom_db"),
            id="str-to-path",
        ),
        pytest.param(
            ServerSettings,
            "AIIDA_AGENTS_PORT",
            "9001",
            "port",
            9001,
            id="str-to-int",
        ),
    ],
)
def test_env_value_is_coerced(
    settings_cls: type[_Base],
    env_var: str,
    raw: str,
    field: str,
    expected: object,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A string env value is coerced to the field's declared type."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, raw)
    assert getattr(settings_cls(), field) == expected


# The remaining tests pin mechanisms specific to one group, so they stay
# individual rather than being forced into a shared parametrization.


def test_reads_dot_env_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config is read from a ``.env`` file in the CWD, replacing python-dotenv."""
    monkeypatch.delenv("AIIDA_AGENTS_PROVIDER", raising=False)
    monkeypatch.delenv("AIIDA_AGENTS_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "AIIDA_AGENTS_PROVIDER=openai\nAIIDA_AGENTS_MODEL=gpt-x\n"
    )
    monkeypatch.chdir(tmp_path)
    settings = ModelSettings()
    assert settings.provider == "openai"
    assert settings.model == "gpt-x"


def test_ollama_base_url_uses_unprefixed_alias(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint reads ``OLLAMA_BASE_URL``, no ``AIIDA_AGENTS_`` prefix."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIIDA_AGENTS_OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote:11434/v1")
    assert OllamaSettings().base_url == "http://remote:11434/v1"


def test_ollama_base_url_accepts_field_name_kwarg(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The aliased field is settable by its Python name (needs populate_by_name).

    Regression: with a ``validation_alias`` but no ``populate_by_name``, the
    Python field name is silently ignored on construction and the default is
    kept, which would quietly break programmatic construction.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert OllamaSettings(base_url="http://injected:1234/v1").base_url == (
        "http://injected:1234/v1"
    )


# Typo detection: ``extra="ignore"`` would otherwise drop an unknown
# ``AIIDA_AGENTS_*`` key silently, so ``warn_on_unrecognized_settings`` surfaces
# it at startup.


@pytest.mark.parametrize("source", ["process_env", "dot_env_file"])
def test_warns_on_unrecognized_prefixed_key(
    source: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A typo'd ``AIIDA_AGENTS_*`` key is flagged from either env or ``.env``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIIDA_AGENTS_PROVDER", raising=False)
    if source == "process_env":
        monkeypatch.setenv("AIIDA_AGENTS_PROVDER", "openai")
    else:
        (tmp_path / ".env").write_text("AIIDA_AGENTS_PROVDER=openai\n")

    with caplog.at_level(logging.WARNING, logger="aiida_agents._settings"):
        warn_on_unrecognized_settings()

    assert "AIIDA_AGENTS_PROVDER" in caplog.text
    assert "typo" in caplog.text.lower()


def test_does_not_warn_on_recognized_keys(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Declared keys, including the unprefixed ``OLLAMA_BASE_URL`` alias, are silent."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", "ollama")
    monkeypatch.setenv("AIIDA_AGENTS_PORT", "8000")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    with caplog.at_level(logging.WARNING, logger="aiida_agents._settings"):
        warn_on_unrecognized_settings()

    for key in ("AIIDA_AGENTS_PROVIDER", "AIIDA_AGENTS_PORT", "OLLAMA_BASE_URL"):
        assert key not in caplog.text
