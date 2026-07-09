"""Agent lifecycle for the CLI: build it, run it, probe it.

Turns resolved settings into a ready agent (``_build_agent``), runs a one-shot
query (``ask``), and checks/warms the configured model (``_probe_model`` +
``_diagnose_probe_failure``). The heavy aiida / agent-stack imports stay local so
``--help`` and shell completion never load AiiDA.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import rich_click as click
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage

from aiida_agents._settings import ModelSettings, _format_validation_error
from aiida_agents.cli.ollama import _ensure_ollama_model, _ollama_pull
from aiida_agents.cli.output import _trace_tool_calls

logger = logging.getLogger(__name__)


async def ask(
    agent: Agent,
    question: str,
    message_history: list[ModelMessage] | None = None,
) -> Any:  # pragma: no cover
    """Run a single query through the agent, returning the result."""
    logger.info("agent query: %s", question)
    result = await agent.run(question, message_history=message_history)
    # Record the tool-call trace to the log file now (always); the console
    # render is the caller's job, done after any live spinner has stopped.
    _trace_tool_calls(result.new_messages())
    return result


def _resolve_model_settings(provider: str | None, model: str | None) -> ModelSettings:
    """Build ``ModelSettings`` with CLI overrides taking precedence.

    Only non-``None`` overrides are passed as constructor kwargs; pydantic-settings
    ranks init kwargs above the environment and ``.env`` (flag > env > file >
    default), so a ``--model`` flag wins over ``AIIDA_AGENTS_MODEL`` with no manual
    precedence handling.
    """
    # ``dict[str, Any]`` so the ``**`` splat's values stay assignable to each
    # typed field (``provider`` is a ``Literal``); pydantic validates them at
    # runtime, e.g. lower-casing and range-checking the provider string.
    overrides: dict[str, Any] = {
        key: value
        for key, value in (("provider", provider), ("model", model))
        if value is not None
    }
    return ModelSettings(**overrides)


def _resolve_settings_or_fail(provider: str | None, model: str | None) -> ModelSettings:
    """Resolve model settings, turning an invalid value into a clean CLI error.

    ``_resolve_model_settings`` raises pydantic's ``ValidationError`` (a raw
    traceback) on a bad ``AIIDA_AGENTS_*`` value. The read-only diagnostics
    commands (``check``/``warm``/``doctor``) build settings outside a ``try``, so
    this converts it to a ``ClickException`` the way ``_build_agent`` does for its
    other configuration errors.
    """
    try:
        return _resolve_model_settings(provider, model)
    except ValidationError as exc:
        msg = f"Invalid configuration:\n{_format_validation_error(exc)}"
        raise click.ClickException(msg) from exc


def _build_agent(
    provider: str | None, model: str | None, profile: str | None
) -> tuple[Agent, ModelSettings]:  # pragma: no cover
    """Load the profile and build the agent with CLI overrides applied.

    The aiida / agent-stack imports stay local so ``--help`` and shell completion
    don't pay for loading AiiDA. Expected configuration failures are surfaced as
    clean CLI errors instead of a traceback.
    """
    from aiida import load_profile
    from aiida_agents.agents import get_agent

    try:
        settings = _resolve_settings_or_fail(provider, model)
        _ensure_ollama_model(settings)
        load_profile(profile)
        agent = get_agent(model_settings=settings)
    except (UserError, ValueError) as exc:
        # UserError: pydantic-ai, for a missing cloud API key. ValueError:
        # get_model for an openai-compatible endpoint without base_url, and
        # pydantic's ValidationError (a subclass) for a bad provider/setting.
        # All are "fix your config", not bugs, so show a clean error.
        raise click.ClickException(str(exc)) from exc
    return agent, settings


def _probe_model(settings: ModelSettings) -> None:  # pragma: no cover
    """Fire a minimal generation against the configured model.

    A failure surfaces here (before a session), and for a local Ollama model the
    call loads it into memory so the first real query isn't a cold start. This is
    the heavy path behind ``warm``; ``check`` uses :func:`_check_reachable`, which
    never generates.
    """
    from aiida_agents.agents._models import get_model

    probe = Agent(get_model(model_settings=settings))
    asyncio.run(probe.run("Reply with the single word: ok."))


# A reachability check must never hang, so the model listing is bounded.
_REACHABILITY_TIMEOUT = 8.0


async def _list_model_ids(client: Any) -> set[str]:
    """Model ids the endpoint advertises, via a cheap listing under a timeout."""
    try:
        page = await asyncio.wait_for(
            client.models.list(), timeout=_REACHABILITY_TIMEOUT
        )
    except asyncio.TimeoutError as exc:
        # ``asyncio.TimeoutError``, not the builtin: on Python 3.10 (our minimum)
        # ``wait_for`` raises the asyncio one, a distinct class from the builtin
        # ``TimeoutError`` (they were only merged into aliases in 3.11), so
        # catching the builtin would let the timeout escape uncaught there.
        # Phrase the message so ``_diagnose_probe_failure`` routes it to the
        # "unreachable" branch (it matches on "connect").
        msg = f"could not connect within {_REACHABILITY_TIMEOUT:.0f}s"
        raise ConnectionError(msg) from exc
    return {item.id for item in page.data}


def _probe_reachable(settings: ModelSettings) -> tuple[str, int, bool]:
    """Reachability facts without a generation: ``(endpoint, n_models, model_ok)``.

    Builds the model (validating provider / base_url / key presence), then lists
    the endpoint's models (one cheap GET under a short timeout). ``model_ok`` is
    whether the configured model is among those advertised. Raises on an
    unreachable endpoint, bad key, or bad config. Shared by ``check`` and
    ``doctor``; never loads or runs the model.
    """
    from aiida_agents.agents._models import get_model

    model = get_model(model_settings=settings)
    # OpenAIChatModel / AnthropicModel both expose the underlying SDK client; the
    # base ``Model`` type does not, hence the ignore.
    client: Any = model.client  # type: ignore[attr-defined]
    ids = asyncio.run(_list_model_ids(client))
    # Ollama lists an untagged model as ``<name>:latest``; normalise so an
    # untagged configured name still matches. Cloud ids carry no such suffix.
    wanted = settings.model
    if settings.provider == "ollama" and ":" not in wanted:
        wanted = f"{wanted}:latest"
    model_ok = wanted in ids or settings.model in ids
    return str(client.base_url), len(ids), model_ok


def _check_reachable(settings: ModelSettings) -> None:  # pragma: no cover
    """Print ``check``'s reachability report; exit non-zero on a real problem.

    Reachability / auth failures raise for the caller to diagnose. A configured
    model the endpoint doesn't advertise is fatal for Ollama (its listing is
    authoritative, so it means "not pulled"), but only a warning for cloud or
    openai-compatible endpoints, whose ``/models`` listing may be partial.
    """
    endpoint, n_models, model_ok = _probe_reachable(settings)
    click.echo(f"Endpoint: {endpoint}")
    click.echo(f"✓ reachable ({n_models} models advertised)")
    if model_ok:
        click.echo(f"✓ model '{settings.model}' is available")
        return
    if settings.provider == "ollama":
        click.echo(f"✗ model '{settings.model}' is not pulled.", err=True)
        click.echo(f"  Pull it with: ollama pull {settings.model}", err=True)
        raise SystemExit(1)
    click.echo(
        f"! model '{settings.model}' is not in this endpoint's list "
        "(it may be partial; the model may still work).",
        err=True,
    )


def _diagnose_probe_failure(
    settings: ModelSettings, exc: Exception
) -> None:  # pragma: no cover
    """Turn a probe failure into an actionable message, offering an Ollama pull."""
    msg = str(exc).lower()
    if settings.provider == "ollama" and ("not found" in msg or "404" in msg):
        click.echo(f"✗ Ollama model '{settings.model}' is not pulled.", err=True)
        if click.confirm(f"Pull it now (ollama pull {settings.model})?", default=True):
            _ollama_pull(settings.model)
        return
    if ("api" in msg and "key" in msg) or "401" in msg or "403" in msg:
        if "not set" in msg or "environment variable" in msg or "set the" in msg:
            click.echo("✗ API key not set: set the provider's API key.", err=True)
        else:
            click.echo(
                "✗ Authentication failed: check the provider's API key.", err=True
            )
        return
    if "connect" in msg or "connection" in msg:
        click.echo("✗ Could not reach the endpoint. Is the server running?", err=True)
        return
    click.echo(f"✗ {exc}", err=True)
