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
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage

from aiida_agents._settings import ModelSettings, warn_on_unrecognized_settings
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

    warn_on_unrecognized_settings()
    try:
        settings = _resolve_model_settings(provider, model)
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
    call loads it into memory so the first real query isn't a cold start.
    """
    from aiida_agents.agents._models import get_model

    probe = Agent(get_model(model_settings=settings))
    asyncio.run(probe.run("Reply with the single word: ok."))


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
        click.echo("✗ Authentication failed — check the provider's API key.", err=True)
        return
    if "connect" in msg or "connection" in msg:
        click.echo("✗ Could not reach the endpoint — is the server running?", err=True)
        return
    click.echo(f"✗ {exc}", err=True)
