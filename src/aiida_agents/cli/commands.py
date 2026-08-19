"""Root Click group for aiida-agents and its core model-facing commands.

The surface is split by concern: the ``config``, ``rag``, and ``mcp`` groups
and the ``doctor`` command live in their own modules and are registered onto
the root group at the bottom of this file. What stays here is the root group
(the global ``--provider`` / ``--model`` / ``--profile`` overrides and logging
setup) plus the two model-facing commands: ``chat`` and ``ask``.
"""

from __future__ import annotations

import asyncio

import rich_click as click
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents._logging import _configure_logging
from aiida_agents.agents.handoff import node_references_from_messages
from aiida_agents._settings import _PROVIDER_CHOICES, LoggingSettings
from aiida_agents.cli._guards import _needs_recognized_settings
from aiida_agents.cli._position import PositionAwareGroup, accepts_root_options
from aiida_agents.cli.config import config
from aiida_agents.cli.doctor import doctor
from aiida_agents.cli.mcp import mcp
from aiida_agents.cli.output import (
    _warn_ungrounded,
    _print_reply,
    _render_tool_calls,
    console,
)
from aiida_agents.cli.rag import rag
from aiida_agents.cli.sandbox import sandbox
from aiida_agents.cli.repl import _run_repl
from aiida_agents.cli.agent import (
    _resolve_plan,
    _step_prompt,
    _StepResult,
    _AGENT_CHOICES,
    _build_agent,
    _resolve_settings_or_fail,
    ask,
)


# ``-h`` as a help alias, everywhere: ``help_option_names`` is read from the
# context, which every subcommand inherits, so this covers the whole tree.
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(
    cls=PositionAwareGroup,
    invoke_without_command=True,
    context_settings=_CONTEXT_SETTINGS,
)
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
@click.option(
    "--agent",
    "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Which agent runs `chat` / `ask`. `auto` routes each request to a "
        "specialist; naming one overrides that (switch mid-session with `/agent`)."
    ),
)
@click.pass_context
def cli(
    ctx: click.Context,
    provider: str | None,
    model: str | None,
    profile: str | None,
    agent: str,
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
    ctx.obj["agent"] = agent.lower()
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)  # pragma: no cover


@cli.command()
@accepts_root_options
@click.pass_context
@_needs_recognized_settings
def chat(ctx: click.Context) -> None:
    """Start the interactive REPL (the default when no subcommand is given)."""
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    agent_type = ctx.obj["agent"]
    # "auto" names a routing decision, not an agent. Each question is resolved
    # to a specialist by the REPL, which builds that specialist on first use, so
    # there is nothing to build up front -- and asking _build_agent for an agent
    # called "auto" is how the default entry point came to crash on startup.
    agent = (
        None
        if agent_type == "auto"
        else _build_agent(settings, ctx.obj["profile"], agent_type)
    )
    _run_repl(agent, settings, profile=ctx.obj["profile"], agent_type=agent_type)


@cli.command("ask")
@click.argument("question")
@click.option(
    "--raw",
    is_flag=True,
    help="Print the raw Markdown reply instead of rendering it (for piping/copy).",
)
@accepts_root_options
@click.pass_context
@_needs_recognized_settings
def ask_cmd(ctx: click.Context, question: str, raw: bool) -> None:
    """Answer a single question and exit (one-shot)."""
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    steps = _resolve_plan(ctx.obj["agent"], question, settings)

    previous: _StepResult | None = None
    for index, step in enumerate(steps, start=1):
        agent = _build_agent(settings, ctx.obj["profile"], step.specialist)
        prompt = _step_prompt(step, question, previous)
        try:
            result = asyncio.run(ask(agent, prompt))
        except KeyboardInterrupt:
            click.echo("(interrupted)", err=True)
            raise SystemExit(130) from None
        except Exception as exc:
            # Deliberately broad, and the same boundary the REPL already has: a
            # provider can fail in ways neither we nor pydantic-ai model (a free
            # router returning malformed tool-call JSON surfaces as
            # ModelHTTPError), and a one-shot command should report that, not
            # print a traceback.
            # Name the step only when there is more than one; on the common
            # single-step request "step 1 (analysis)" is noise in front of the
            # actual error.
            where = f" on step {index} ({step.specialist})" if len(steps) > 1 else ""
            raise click.ClickException(f"Agent run failed{where}: {exc}") from exc

        _render_tool_calls(result.new_messages(), console)  # debug-gated
        if isinstance(result.output, DeferredToolRequests):
            # A write needs a terminal. Stopping here rather than continuing is
            # the point: later steps would build on a submission that never
            # happened.
            click.echo(
                "The agent proposed a write action, which needs interactive "
                "approval. Re-run it in `aiida-agents chat`.",
                err=True,
            )
            raise SystemExit(2)

        if len(steps) > 1:
            console.print(
                f"[dim]— step {index}/{len(steps)} ({step.specialist}) —[/dim]"
            )
        _print_reply(result.output, raw=raw)
        _warn_ungrounded(result.output, result.all_messages(), prompt)
        previous = _StepResult(
            step.specialist,
            result.output,
            node_references_from_messages(result.all_messages()),
        )


# Register the commands and groups that live in their own modules onto the root.
cli.add_command(config)
cli.add_command(sandbox)
cli.add_command(rag)
cli.add_command(mcp)
cli.add_command(doctor)
