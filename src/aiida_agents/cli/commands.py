"""Root Click group for aiida-agents and its core model-facing commands.

The surface is split by concern: the ``config``, ``rag``, and ``mcp`` groups
and the ``doctor`` command live in their own modules and are registered onto
the root group at the bottom of this file. What stays here is the root group
(the global ``--provider`` / ``--model`` / ``--profile`` overrides and logging
setup) plus the small model-facing commands: ``chat``, ``ask``, ``check``, and
``warm``.
"""

from __future__ import annotations

import asyncio
import time

import rich_click as click
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents._logging import _configure_logging
from aiida_agents._settings import _PROVIDER_CHOICES, LoggingSettings
from aiida_agents.cli._guards import _needs_recognized_settings
from aiida_agents.cli.config import config
from aiida_agents.cli.doctor import doctor
from aiida_agents.cli.mcp import mcp
from aiida_agents.cli.output import (
    _format_duration,
    _print_reply,
    _render_tool_calls,
    console,
)
from aiida_agents.cli.rag import rag
from aiida_agents.cli.repl import _run_repl
from aiida_agents.cli.agent import (
    _build_agent,
    _check_reachable,
    _diagnose_probe_failure,
    _probe_model,
    _resolve_settings_or_fail,
    ask,
)


# ``-h`` as a help alias, everywhere: ``help_option_names`` is read from the
# context, which every subcommand inherits, so this covers the whole tree.
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(invoke_without_command=True, context_settings=_CONTEXT_SETTINGS)
@click.version_option(package_name="aiida-agents", message="%(prog)s %(version)s")
@click.option(
    "--provider",
    type=click.Choice(_PROVIDER_CHOICES, case_sensitive=False),
    default=None,
    help="Override the model provider.",
)
@click.option("--model", default=None, help="Override the model name.")
@click.option(
    "--profile",
    default=None,
    help="AiiDA profile to load (defaults to the default profile).",
)
@click.pass_context
def cli(
    ctx: click.Context, provider: str | None, model: str | None, profile: str | None
) -> None:
    """Natural-language, multi-agent interface to AiiDA."""
    # Configure logging once, at the entry point for every subcommand, so the
    # optional log file captures the full tool-call/agent-reply trace and trace
    # records stay off the console (see ``_configure_logging``). ``--help`` is an
    # eager option that exits before this runs, so help stays fast.
    _configure_logging(LoggingSettings())
    # A mistyped setting key is caught per command via ``_needs_recognized_settings``
    # (fail fast) or reported by ``config show`` (non-fatal), not here, so
    # ``--help`` and completion never trip over it.
    ctx.ensure_object(dict)
    ctx.obj["provider"] = provider
    ctx.obj["model"] = model
    ctx.obj["profile"] = profile
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)  # pragma: no cover


@cli.command()
@click.pass_context
@_needs_recognized_settings
def chat(ctx: click.Context) -> None:  # pragma: no cover
    """Start the interactive REPL (the default when no subcommand is given)."""
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    agent = _build_agent(settings, ctx.obj["profile"])
    _run_repl(agent, settings)


@cli.command("ask")
@click.argument("question")
@click.option(
    "--raw",
    is_flag=True,
    help="Print the raw Markdown reply instead of rendering it (for piping/copy).",
)
@click.pass_context
@_needs_recognized_settings
def ask_cmd(ctx: click.Context, question: str, raw: bool) -> None:
    """Answer a single question and exit (one-shot)."""
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    agent = _build_agent(settings, ctx.obj["profile"])
    result = asyncio.run(ask(agent, question))
    _render_tool_calls(result.new_messages(), console)  # debug-gated; no spinner here
    if isinstance(result.output, DeferredToolRequests):
        click.echo(
            "The agent proposed a write action, which needs interactive approval. "
            "Re-run it in `aiida-agents chat`.",
            err=True,
        )
        raise SystemExit(2)
    _print_reply(result.output, raw=raw)


@cli.command()
@click.pass_context
@_needs_recognized_settings
def check(ctx: click.Context) -> None:
    """Verify config, endpoint reachability, and model availability.

    Fast and read-only: it never loads or runs the model. Use `warm` to
    pre-load a local model before an interactive session.
    """
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo(f"Checking {settings.provider}:{settings.model} ...")
    try:
        _check_reachable(settings)
    except Exception as exc:
        _diagnose_probe_failure(settings, exc)
        raise SystemExit(1) from exc


@cli.command()
@click.pass_context
@_needs_recognized_settings
def warm(ctx: click.Context) -> None:
    """Warm the model with one tiny generation, so the first query isn't cold.

    Mainly useful for a local Ollama model before a session; for a cloud model
    it is just a round-trip that also confirms generation works.
    """
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo(f"Warming {settings.provider}:{settings.model} ...")
    start = time.monotonic()
    try:
        _probe_model(settings)
    except Exception as exc:
        _diagnose_probe_failure(settings, exc)
        raise SystemExit(1) from exc
    click.echo(f"✓ warmed in {_format_duration(time.monotonic() - start)}")


# Register the commands and groups that live in their own modules onto the root.
cli.add_command(config)
cli.add_command(rag)
cli.add_command(mcp)
cli.add_command(doctor)
