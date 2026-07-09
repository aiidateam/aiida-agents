"""Root Click group for aiida-agents and its core model-facing commands.

The surface is split by concern: the ``config``, ``rag``, and ``mcp`` groups
live in their own modules and are registered onto the root group at the bottom
of this file. What stays here is the root group (the global ``--provider`` /
``--model`` / ``--profile`` overrides and logging setup) plus the small
model-facing commands: ``chat``, ``ask``, ``check``, ``warm``, and ``doctor``.
"""

from __future__ import annotations

import asyncio
import time

import rich_click as click
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents._logging import _configure_logging
from aiida_agents._settings import LoggingSettings, ModelSettings
from aiida_agents.cli._guards import _needs_recognized_settings
from aiida_agents.cli.config import config
from aiida_agents.cli.mcp import mcp
from aiida_agents.cli.output import _format_duration, _render_tool_calls, console
from aiida_agents.cli.rag import _module_missing, rag
from aiida_agents.cli.repl import _run_repl
from aiida_agents.cli.session import (
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
    default=None,
    help="Override the model provider (ollama, openai, anthropic, openrouter, openai-compatible).",
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
    agent, settings = _build_agent(
        ctx.obj["provider"], ctx.obj["model"], ctx.obj["profile"]
    )
    _run_repl(agent, settings)


@cli.command("ask")
@click.argument("question")
@click.pass_context
@_needs_recognized_settings
def ask_cmd(ctx: click.Context, question: str) -> None:  # pragma: no cover
    """Answer a single QUESTION and exit (one-shot; scriptable)."""
    agent, _ = _build_agent(ctx.obj["provider"], ctx.obj["model"], ctx.obj["profile"])
    result = asyncio.run(ask(agent, question))
    _render_tool_calls(result.new_messages(), console)  # debug-gated; no spinner here
    if isinstance(result.output, DeferredToolRequests):
        click.echo(
            "The agent proposed a write action, which needs interactive approval. "
            "Re-run it in `aiida-agents chat`.",
            err=True,
        )
        raise SystemExit(2)
    click.echo(result.output)


@cli.command()
@click.pass_context
@_needs_recognized_settings
def check(ctx: click.Context) -> None:  # pragma: no cover
    """Verify config, endpoint reachability, and model availability.

    Fast and read-only: it never loads or runs the model. Use `warm` to
    pre-load a local model before an interactive session.
    """
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo(f"Checking {settings.provider}:{settings.model} …")
    try:
        _check_reachable(settings)
    except Exception as exc:
        _diagnose_probe_failure(settings, exc)
        raise SystemExit(1) from exc


@cli.command()
@click.pass_context
@_needs_recognized_settings
def warm(ctx: click.Context) -> None:  # pragma: no cover
    """Warm the model with one tiny generation, so the first query isn't cold.

    Mainly useful for a local Ollama model before a session; for a cloud model
    it is just a round-trip that also confirms generation works.
    """
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo(f"Warming {settings.provider}:{settings.model} …")
    start = time.monotonic()
    try:
        _probe_model(settings)
    except Exception as exc:
        _diagnose_probe_failure(settings, exc)
        raise SystemExit(1) from exc
    click.echo(f"✓ warmed in {_format_duration(time.monotonic() - start)}")


def _short_reason(exc: Exception) -> str:
    """First non-empty line of ``exc``'s message, truncated for a table cell.

    Guards the empty-message case (``"".splitlines()`` is ``[]``, so ``[0]``
    would raise): a health check that fails with a message-less error (a bare
    ``ValueError``, or ``asyncio.TimeoutError`` whose ``str`` is empty) must
    still yield a printable detail, never an ``IndexError`` that aborts the
    whole diagnostics report.
    """
    lines = [line for line in str(exc).splitlines() if line.strip()]
    return lines[0][:100] if lines else ""


def _run_diagnostics(
    settings: ModelSettings, profile: str | None
) -> list[tuple[str, bool, str]]:  # pragma: no cover
    """Run each health check, returning ``(label, ok, detail)`` rows.

    Every check is isolated in its own ``try`` so one failure (an unreachable
    model, an unloadable profile) never aborts the rest of the report. The model
    check uses the no-generation reachability probe, so ``doctor`` never warms
    the model.
    """
    from aiida_agents.cli.session import _probe_reachable

    rows: list[tuple[str, bool, str]] = []

    try:
        from aiida import load_profile

        loaded = load_profile(profile)
        rows.append(("AiiDA profile loads", True, loaded.name))
    except Exception as exc:
        rows.append(("AiiDA profile loads", False, _short_reason(exc)))

    model_label = f"Model reachable ({settings.provider}:{settings.model})"
    try:
        endpoint, _, model_ok = _probe_reachable(settings)
        if model_ok:
            rows.append((model_label, True, endpoint))
        elif settings.provider == "ollama":
            rows.append(
                (model_label, False, f"model not pulled (ollama pull {settings.model})")
            )
        else:
            rows.append(
                (model_label, True, "reachable; model not listed (may still work)")
            )
    except Exception as exc:
        rows.append((model_label, False, _short_reason(exc)))

    try:
        from aiida_agents.rag.store import index_status

        built = index_status().built
        rows.append(
            ("RAG index built", built, "" if built else "run `aiida-agents rag build`")
        )
    except Exception as exc:
        rows.append(("RAG index built", False, _short_reason(exc)))

    has_sphinx = not _module_missing("sphinx")
    rows.append(
        (
            "Docs toolchain (sphinx)",
            has_sphinx,
            "" if has_sphinx else "needed only for `rag build`",
        )
    )
    return rows


@cli.command()
@click.pass_context
@_needs_recognized_settings
def doctor(ctx: click.Context) -> None:  # pragma: no cover
    """Diagnose the setup: profile, model, RAG index, and docs toolchain."""
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo("Running diagnostics …\n")
    all_ok = True
    for label, ok, detail in _run_diagnostics(settings, ctx.obj["profile"]):
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        suffix = f" [dim]({detail})[/]" if detail else ""
        console.print(f"{mark} {label}{suffix}")
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit(1)


# Register the command groups that live in their own modules onto the root.
cli.add_command(config)
cli.add_command(rag)
cli.add_command(mcp)
