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

import difflib
import logging
import os
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, TypeAlias, get_args

from pydantic import BeforeValidator, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

logger = logging.getLogger(__name__)

# The env namespace and the .env file are the two anchors the settings groups
# load from. Named once here so the typo detector reuses them instead of
# re-hardcoding the strings (which risks the namespace check drifting from
# ``_Base``, or a ``""`` default silently matching every key).
_ENV_PREFIX = "AIIDA_AGENTS_"
_ENV_FILE = ".env"


class _Base(BaseSettings):
    """Shared loading rules for every settings group."""

    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_file=_ENV_FILE,
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
        # Use each field's attribute docstring (the string literal below it) as
        # its description, so one line documents the field for IDE hover/
        # autocomplete, ``config show``/``config init``, and JSON schema at once.
        use_attribute_docstrings=True,
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


def _choices(alias: object) -> tuple[str, ...]:
    """The allowed string values of an ``Annotated[Literal[...], ...]`` alias.

    Lets consumers (the CLI's ``--provider`` choices, drift guards) derive the
    option list from the single source of truth (the ``Literal``) instead of
    restating it.
    """
    literal, *_ = get_args(alias)  # unwrap Annotated -> the Literal
    return get_args(literal)  # the Literal's string members


# The ``--provider`` flag's choices come from ``_Provider`` so the CLI can never
# advertise a provider the settings don't accept (or omit a new one).
_PROVIDER_CHOICES = _choices(_Provider)

# Non-empty sentinel default for the openai-compatible ``api_key`` (so keyless
# local servers work); ``config show`` tells a set key from this placeholder by
# comparing against it, hence the shared constant.
_API_KEY_UNSET = "api-key-not-set"


class ModelSettings(_Base):
    """LLM model/provider configuration (``AIIDA_AGENTS_*``)."""

    provider: _Provider = "ollama"
    """Model backend: ollama | openai | anthropic | openrouter | openai-compatible."""

    model: str = "qwen3.5:2b"
    """Model name/tag for the chosen provider (an Ollama tag, or a cloud model id)."""

    base_url: str | None = None
    """Base URL for provider=openai-compatible (e.g. https://api.deepseek.com/v1)."""

    api_key: str = _API_KEY_UNSET
    """API key for provider=openai-compatible (with AIIDA_AGENTS_BASE_URL), or a
    dummy for keyless local servers. A secret: never commit it."""

    # Cloud provider SDK keys use their conventional unprefixed names (not the
    # AIIDA_AGENTS_ prefix) so an existing OPENAI_API_KEY etc. just works; the
    # factory passes the active provider's key through, and None lets the SDK
    # fall back to its own env lookup. Secrets: never commit them.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    """API key for provider=openai."""

    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
    """API key for provider=anthropic."""

    openrouter_api_key: str | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    """API key for provider=openrouter."""

    max_tokens: int = Field(default=8192, gt=0)
    """Output token cap (all providers). Too small truncates long tool-calling runs."""

    context_length: int | None = Field(default=None, gt=0)
    """Ollama context window (num_ctx), sent per request; Ollama-only. None keeps
    Ollama's default. Larger windows cost more VRAM, so it is opt-in."""

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

    tool_retries: int = Field(default=3, ge=0)
    """Max consecutive failed attempts at one tool before the run aborts; any
    success resets the count (tracked per tool). A small budget lets a
    hallucinating model recover from a bad identifier without a genuinely broken
    tool retrying forever (unbounded cost on a paid provider). 0 disables retries;
    negative is rejected."""


class SandboxSettings(_Base):
    """Where and how generated code is executed (``AIIDA_AGENTS_*``)."""

    sandbox_profile: str = "agents-sandbox"
    """AiiDA profile generated code runs against. Must be a disposable copy of
    the user's storage: ``aiida-agents sandbox init`` makes one, and ``sandbox
    check`` proves it shares nothing. Nothing downstream can tell a copy from
    the real thing, so this setting is the whole boundary around the user's
    AiiDA data. Outside the database it only narrows: a snippet that gets past
    the import guard can still read the filesystem and reach the network."""

    # Bounds and default mirror ``sandbox.runner``'s own, which clamps whatever
    # it is handed. Duplicated rather than imported: ``aiida_agents/__init__``
    # promises the settings depend on nothing beyond pydantic, and
    # ``test_snippet_timeout_matches_the_runner`` pins the two together.
    snippet_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    """Seconds one generated snippet may run before it is killed. A query over
    a large provenance graph is slow; an accidental ``while True`` is forever.
    Bounds only ``run_python_snippet``: ``sandbox init`` copies the user's storage
    and is deliberately allowed to take as long as that takes."""


class ReplSettings(_Base):
    """Interactive REPL configuration (``AIIDA_AGENTS_*``)."""

    history_max_turns: int = Field(default=10, ge=1)
    """Recent user turns the REPL replays as context per query. Capped on turn
    boundaries (never mid tool-call/return) so the window can't grow unbounded; at
    least 1."""

    vi_mode: bool = False
    """Use vi keybindings for REPL line editing instead of the default emacs bindings."""


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
    """Local Ollama endpoint (OpenAI-compatible /v1 URL), shared by the chat model
    and the RAG embeddings."""


def _default_vector_db_path() -> Path:
    """The per-user directory the vector store lives in by default.

    ``platformdirs`` rather than a hand-rolled ``~/.local/share``: it is what
    resolves correctly on macOS and Windows too, and it already honours
    ``XDG_DATA_HOME``.
    """
    from platformdirs import user_data_dir

    return Path(user_data_dir("aiida-agents")) / "vector_db"


class RagSettings(_Base):
    """RAG / documentation-retrieval configuration (``AIIDA_AGENTS_*``)."""

    embed_backend: _EmbedBackend = "ollama"
    """Embedding backend for RAG: ollama | sentence-transformers."""

    embed_model: str = "mxbai-embed-large"
    """Embedding model for the ollama backend (sentence-transformers uses its own
    fixed model)."""

    embed_timeout: int = Field(default=300, ge=1)
    """Seconds to wait for one embedding request before giving up.

    Embedding is the slow half of ``rag build``, and how slow depends entirely
    on the machine: a CPU-only Ollama can take minutes over a batch that a GPU
    finishes in seconds. Configurable (and defaulted generously) because the
    previous fixed 120s turned "this machine is slow" into an unrecoverable
    build failure with no knob to turn.
    """

    vector_db_path: Path = Field(default_factory=_default_vector_db_path)
    """Directory for the persisted ChromaDB vector store.

    A fixed per-user location, not a relative path: the index costs minutes to
    build, and a relative default silently resolved against the working
    directory, so running the CLI from anywhere but the build directory found an
    empty store and reported the index as missing. Override with
    ``AIIDA_AGENTS_VECTOR_DB_PATH`` to keep a per-project store deliberately.
    """

    # The ``ollama`` backend's endpoint lives in ``OllamaSettings`` (shared with
    # the chat model), read by ``get_embedding_function`` where it builds the
    # client, not duplicated here.


class ServerSettings(_Base):
    """MCP server configuration (``AIIDA_AGENTS_*``)."""

    port: int = 8000
    """Port the MCP server listens on."""


class LoggingSettings(_Base):
    """Process-wide logging configuration (``AIIDA_AGENTS_*``).

    Every entry point (CLI, MCP server, RAG indexing)
    logs, so the level is a package-wide knob rather than part of
    ``ServerSettings``.
    """

    log_level: _LogLevel = "WARNING"
    """Console log level: DEBUG | INFO | WARNING | ERROR | CRITICAL. Defaults to
    WARNING so an interactive session stays uncluttered; lower it to INFO or DEBUG
    to surface operational logs on the console."""

    log_file: Path | None = None
    """Optional log file; a path enables file logging. The file always captures the
    full tool-call/agent-reply traces (logged at DEBUG regardless of log_level);
    other records follow log_level."""


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` as concise ``field: reason`` lines.

    Drops pydantic's framing (the trailing docs URL that ``str(exc)`` appends,
    the ``Value error,`` prefix on custom-validator messages) so a CLI can tell a
    researcher what to fix without a wall of internals. A whole-model error (a
    cross-field ``model_validator``, which carries no field location) is shown as
    its bare message.
    """
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"])
        reason = err["msg"].removeprefix("Value error, ")
        lines.append(f"{location}: {reason}" if location else reason)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Typo detection
#
# ``extra="ignore"`` lets each group tolerate the other groups' keys in the
# shared ``.env``, but it also silently drops a *typo'd* key, leaving the
# setting at its default with no hint. The helpers below derive the recognised
# key names from the models themselves (so they never drift) and flag any
# stray ``AIIDA_AGENTS_*`` key at startup.
# ---------------------------------------------------------------------------

# Every ``_Base`` subclass in this module belongs here; a group left out is
# invisible to ``config show``, ``config init`` and the typo detector alike,
# and its correctly spelled env var is then reported as a typo.
_SETTINGS_GROUPS: tuple[type[_Base], ...] = (
    ModelSettings,
    AgentSettings,
    SandboxSettings,
    ReplSettings,
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


def _dotenv_keys(env_file: Path) -> set[str]:
    """Upper-cased keys assigned in a ``.env`` file.

    Comments, blank lines, and non-assignments are skipped, and a leading
    ``export `` is stripped. Shared by the typo detector here and ``config
    show``'s source column, so both read a ``.env`` the same way.
    """
    keys: set[str] = set()
    if not env_file.is_file():
        return keys
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        keys.add(key.upper())
    return keys


def _present_env_keys(env_file: Path) -> set[str]:
    """Upper-cased env var names set in the process env or the ``.env`` file."""
    return {key.upper() for key in os.environ} | _dotenv_keys(env_file)


# Similarity above which a stray, out-of-namespace key is treated as a typo of
# a real setting rather than an unrelated variable. Genuine near-misses (a one-
# or two-character slip in a 15+ character name, e.g. ``AIIDA_AGENS_PROVIDER``)
# score ~0.97; legitimate neighbours like ``OPENAI_API_BASE`` vs
# ``OPENAI_API_KEY`` sit around 0.8, so 0.9 separates them with headroom.
_TYPO_SIMILARITY = 0.9


class SettingProblem(NamedTuple):
    """A set env var that looks like a mistyped setting.

    ``suggestion`` is the closest real setting name for an out-of-namespace
    near-miss (``AIIDA_AGENS_PROVIDER`` -> ``AIIDA_AGENTS_PROVIDER``), or ``None``
    for an in-namespace key whose field simply isn't recognised
    (``AIIDA_AGENTS_PROVDER``). Unpacks as ``(key, suggestion)``.
    """

    key: str
    suggestion: str | None


def _classify_present_keys(present_keys: set[str]) -> list[SettingProblem]:
    """Classify each present-but-unrecognised env var as a likely typo.

    Pure (no I/O): ``present_keys`` are the upper-cased names already gathered
    from the process env and ``.env`` (see :func:`_present_env_keys`), so the
    prefix/difflib logic is unit-testable with a plain set. Two kinds are
    reported, both otherwise dropped silently by ``extra="ignore"`` (leaving the
    setting at its default):

    * a *field* typo inside the namespace (``AIIDA_AGENTS_PROVDER``): the prefix
      is right but no field matches, so the suggestion is ``None``;
    * a *prefix* typo that lands outside the namespace (``AIIDA_AGENS_PROVIDER``):
      the suggestion is the closest real setting name, reported only when the
      resemblance clears :data:`_TYPO_SIMILARITY` so unrelated variables stay out.

    Unprefixed but recognised names (``OLLAMA_BASE_URL``, the cloud SDK keys) are
    known and never reported.
    """
    known = _known_env_var_names()
    problems: list[SettingProblem] = []
    for key in sorted(present_keys - known):
        if key.startswith(_ENV_PREFIX):
            problems.append(SettingProblem(key, None))
        elif close := difflib.get_close_matches(
            key, sorted(known), n=1, cutoff=_TYPO_SIMILARITY
        ):
            problems.append(SettingProblem(key, close[0]))
    return problems


def find_unrecognized_settings(env_file: Path | None = None) -> list[SettingProblem]:
    """Return a :class:`SettingProblem` for each set variable that looks mistyped.

    The I/O boundary over :func:`_classify_present_keys`: gathers the keys set in
    the process env and ``env_file``, then classifies them.

    :param env_file: ``.env`` file to scan; defaults to ``.env`` in the current
        directory, matching what the settings groups load.
    """
    target = env_file if env_file is not None else Path(_ENV_FILE)
    return _classify_present_keys(_present_env_keys(target))


def warn_on_unrecognized_settings(env_file: Path | None = None) -> None:
    """Log a non-fatal warning for each mistyped setting.

    Detection lives in :func:`find_unrecognized_settings`; this is the
    warn-and-continue entry point used where stopping is undesirable (the MCP
    server). The CLI fails fast on the same findings instead.
    """
    for key, suggestion in find_unrecognized_settings(env_file):
        if suggestion is not None:
            logger.warning(
                "%s is set but is not a recognised aiida-agents setting; did you "
                "mean %s? It will be ignored.",
                key,
                suggestion,
            )
        else:
            logger.warning(
                "%s is set but is not a recognised aiida-agents setting; it will "
                "be ignored. Check for a typo.",
                key,
            )
