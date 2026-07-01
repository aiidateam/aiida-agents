"""Typed, centralized configuration for aiida-agents.

Every ``AIIDA_AGENTS_*`` knob in the project lives here, one ``BaseSettings``
per subsystem (model, RAG, server) plus the cross-cutting ``OllamaSettings``
endpoint and ``LoggingSettings`` groups, all on a shared base so the loading
rules are defined once. Each subsystem imports only the group(s) it needs.

The local Ollama endpoint is shared infrastructure (both the chat model and
the RAG embeddings talk to the same server), so it lives in its own
``OllamaSettings`` group rather than being duplicated on each consumer. It is
the lone exception to the ``AIIDA_AGENTS_`` prefix: it keeps its conventional
unprefixed ``OLLAMA_BASE_URL`` name so an existing Ollama setup works out of
the box.

Cloud provider SDK keys (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
``OPENROUTER_API_KEY``) are read here too, under their conventional unprefixed
names (like ``OLLAMA_BASE_URL``), and the model factory hands the matching one
to the provider. Like the openai-compatible ``api_key`` they are secrets and
must never be persisted to a committed config file.

pydantic-settings reads both the process environment and a ``.env`` file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

logger = logging.getLogger(__name__)


class _Base(BaseSettings):
    """Shared loading rules for every settings group."""

    model_config = SettingsConfigDict(
        env_prefix="AIIDA_AGENTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        # A blank value (e.g. ``AIIDA_AGENTS_PROVIDER=``) means "unset", not the
        # empty string. Without this, the blank is read as "" and fails
        # validation (a literal_error for the constrained fields, a bad int for
        # the port), so uncommenting a key in .env but leaving it empty would
        # crash the group at startup instead of falling back to its default.
        env_ignore_empty=True,
        # The groups share one .env file, and pydantic-settings reads the
        # whole file; without this, each group would fail on the *other* groups'
        # AIIDA_AGENTS_* keys (e.g. ModelSettings choking on AIIDA_AGENTS_PORT).
        # Non-prefixed vars like PATH/HOME are already filtered out by env_prefix.
        extra="ignore",
        # Accept both the Python field name and an explicit ``validation_alias``
        # (e.g. ``OLLAMA_BASE_URL``) as constructor kwargs. Without this, a field
        # carrying an alias can only be set via that alias, so the natural
        # ``OllamaSettings(base_url=...)`` would silently drop the value.
        populate_by_name=True,
    )


# Constrained strings whose env value is case-folded before the ``Literal``
# check, so mixed-case input (``Ollama``, ``debug``) is accepted while a
# genuinely invalid value still fails fast with a ``literal_error``.
_Provider: TypeAlias = Annotated[
    Literal["ollama", "openai", "anthropic", "openrouter", "openai-compatible"],
    BeforeValidator(str.lower),
]
_EmbedBackend: TypeAlias = Annotated[
    Literal["ollama", "sentence-transformers"],
    BeforeValidator(str.lower),
]
_LogLevel: TypeAlias = Annotated[
    Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    BeforeValidator(str.upper),  # ``logging`` level names are case-sensitive
]


class ModelSettings(_Base):
    """LLM model/provider configuration (``AIIDA_AGENTS_*``)."""

    provider: _Provider = "ollama"
    model: str = "qwen3.5:2b"

    # OpenAI-compatible provider settings. ``api_key`` is the endpoint
    # credential (often a dummy for keyless servers); it stays on the env / .env
    # rail and must never be persisted to a committed config file.
    base_url: str | None = None
    api_key: str = "api-key-not-set"

    # Cloud provider SDK keys, read under their conventional unprefixed names
    # (not ``AIIDA_AGENTS_*``) so they work in ``.env`` as well as the real
    # environment. The model factory passes the active provider's key through;
    # ``None`` lets the SDK fall back to its own env lookup. Secrets: never
    # persist to a committed config file.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
    openrouter_api_key: str | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )

    # Output cap (all providers). Too small truncates long tool-calling runs.
    max_tokens: int = Field(default=8192, gt=0)

    # Ollama context window (``num_ctx``), sent per request; Ollama-only. ``None``
    # keeps Ollama's default. Larger windows cost more VRAM, so it is opt-in.
    context_length: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_token_budget(self) -> Self:
        """Reject an output budget too large for the context window.

        ``num_ctx`` bounds prompt and output together, so ``max_tokens`` >=
        ``context_length`` can never be met. Ollama-only; inert elsewhere.
        """
        if self.context_length is None:
            return self
        if self.provider != "ollama":
            logger.warning(
                "AIIDA_AGENTS_CONTEXT_LENGTH is set but only the 'ollama' provider "
                "uses it (cloud models have a fixed context window); it is ignored "
                "for provider %r.",
                self.provider,
            )
            return self
        if self.max_tokens >= self.context_length:
            msg = (
                f"max_tokens ({self.max_tokens}) must be smaller than context_length "
                f"({self.context_length}): the output budget has to fit inside the "
                f"context window, with room left for the prompt. Lower max_tokens or "
                f"raise context_length (AIIDA_AGENTS_CONTEXT_LENGTH)."
            )
            raise ValueError(msg)
        return self


class AgentSettings(_Base):
    """Agent behaviour configuration (``AIIDA_AGENTS_*``)."""

    # Max *consecutive* failed attempts at a single tool before the run is
    # aborted; any success resets the count (pydantic-ai tracks it per tool, not
    # per run). A small budget lets a hallucinating model recover from a bad or
    # wrong-type identifier without letting a genuinely broken tool retry forever,
    # which on a paid provider is unbounded cost. ``0`` disables retries (a tool
    # error aborts immediately); a negative value is meaningless, so reject it.
    tool_retries: int = Field(default=3, ge=0)


class OllamaSettings(_Base):
    """Local Ollama server endpoint.

    Shared infrastructure: both the chat model (``ModelSettings`` with
    ``provider='ollama'``) and the RAG embeddings (``RagSettings`` with
    ``embed_backend='ollama'``) point at the same server, so its location is
    defined once here and read by whichever consumer needs it.
    """

    # Conventional unprefixed name, no AIIDA_AGENTS_ prefix.
    base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias="OLLAMA_BASE_URL",
    )


class RagSettings(_Base):
    """RAG / documentation-retrieval configuration (``AIIDA_AGENTS_*``)."""

    # ``embed_model`` applies to the ``ollama`` backend; the
    # ``sentence-transformers`` backend uses its own fixed model.
    embed_backend: _EmbedBackend = "ollama"
    embed_model: str = "mxbai-embed-large"
    vector_db_path: Path = Path(".aiida_agents_vector_db")

    # The ``ollama`` backend's endpoint lives in ``OllamaSettings`` (shared with
    # the chat model), read by ``get_embedding_function`` where it builds the
    # client, not duplicated here.


class ServerSettings(_Base):
    """MCP server configuration (``AIIDA_AGENTS_*``)."""

    port: int = 8000


class LoggingSettings(_Base):
    """Process-wide logging configuration (``AIIDA_AGENTS_*``).

    Not MCP-server specific: every entry point (CLI, MCP server, RAG indexing)
    logs, so the level is a package-wide knob rather than part of
    ``ServerSettings``.
    """

    log_level: _LogLevel = "INFO"


# ---------------------------------------------------------------------------
# Typo detection
#
# ``extra="ignore"`` lets each group tolerate the other groups' keys in the
# shared ``.env``, but it also silently drops a *typo'd* key, leaving the
# setting at its default with no hint. The helpers below derive the recognised
# key names from the models themselves (so they never drift) and flag any
# stray ``AIIDA_AGENTS_*`` key at startup.
# ---------------------------------------------------------------------------

_SETTINGS_GROUPS: tuple[type[_Base], ...] = (
    ModelSettings,
    AgentSettings,
    OllamaSettings,
    RagSettings,
    ServerSettings,
    LoggingSettings,
)


def _known_env_var_names() -> frozenset[str]:
    """Every env var name the settings groups recognise, upper-cased.

    A field maps to ``<env_prefix><field_name>`` unless it carries an explicit
    ``validation_alias`` (e.g. ``OLLAMA_BASE_URL``), which is used verbatim.
    """
    names: set[str] = set()
    for cls in _SETTINGS_GROUPS:
        prefix = cls.model_config.get("env_prefix", "")
        for field_name, field in cls.model_fields.items():
            alias = field.validation_alias
            name = alias if isinstance(alias, str) else f"{prefix}{field_name}"
            names.add(name.upper())
    return frozenset(names)


def _present_prefixed_keys(env_file: Path) -> set[str]:
    """``AIIDA_AGENTS_*`` keys set in the process env or the ``.env`` file."""
    prefix = _Base.model_config.get("env_prefix", "").upper()
    present = {key.upper() for key in os.environ if key.upper().startswith(prefix)}
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key.startswith("export "):
                key = key.removeprefix("export ").strip()
            if key.upper().startswith(prefix):
                present.add(key.upper())
    return present


def warn_on_unrecognized_settings(env_file: Path | None = None) -> None:
    """Warn about any ``AIIDA_AGENTS_*`` variable no settings group declares.

    A typo'd key (e.g. ``AIIDA_AGENTS_PROVDER``) is otherwise dropped silently
    by ``extra="ignore"``, leaving the setting at its default. Call this once at
    startup so the typo is surfaced instead of quietly ignored. Only the
    ``AIIDA_AGENTS_`` namespace is checked; the unprefixed ``OLLAMA_BASE_URL``
    can't be told apart from unrelated environment variables.

    :param env_file: ``.env`` file to scan; defaults to ``.env`` in the current
        directory, matching what the settings groups load.
    """
    target = env_file if env_file is not None else Path(".env")
    unknown = _present_prefixed_keys(target) - _known_env_var_names()
    for key in sorted(unknown):
        logger.warning(
            "%s is set but is not a recognised aiida-agents setting; it will be "
            "ignored. Check for a typo.",
            key,
        )
