"""Click command surface for aiida-agents (``rich_click`` for pretty help)."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time

import rich_click as click
from pydantic_ai.tools import DeferredToolRequests
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from aiida_agents._logging import _configure_logging
from aiida_agents._settings import LoggingSettings, warn_on_unrecognized_settings
from aiida_agents.cli.session import (
    _build_agent,
    _diagnose_probe_failure,
    _probe_model,
    _resolve_model_settings,
    ask,
)
from aiida_agents.cli.ollama import _prompt_pull_ollama_model
from aiida_agents.cli.config import _config_rows
from aiida_agents.cli.output import _format_duration, console
from aiida_agents.cli.repl import _run_repl


# ``-h`` as a help alias, everywhere: ``help_option_names`` is read from the
# context, which every subcommand inherits, so this covers the whole tree.
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(invoke_without_command=True, context_settings=_CONTEXT_SETTINGS)
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
    ctx.ensure_object(dict)
    ctx.obj["provider"] = provider
    ctx.obj["model"] = model
    ctx.obj["profile"] = profile
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)  # pragma: no cover


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:  # pragma: no cover
    """Start the interactive REPL (the default when no subcommand is given)."""
    agent, settings = _build_agent(
        ctx.obj["provider"], ctx.obj["model"], ctx.obj["profile"]
    )
    _run_repl(agent, settings)


@cli.command("ask")
@click.argument("question")
@click.pass_context
def ask_cmd(ctx: click.Context, question: str) -> None:  # pragma: no cover
    """Answer a single QUESTION and exit (one-shot; scriptable)."""
    agent, _ = _build_agent(ctx.obj["provider"], ctx.obj["model"], ctx.obj["profile"])
    result = asyncio.run(ask(agent, question))
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
def check(ctx: click.Context) -> None:  # pragma: no cover
    """Verify the configured model is reachable, and warm it up."""
    warn_on_unrecognized_settings()
    settings = _resolve_model_settings(ctx.obj["provider"], ctx.obj["model"])
    click.echo(f"Checking {settings.provider}:{settings.model} …")
    start = time.monotonic()
    try:
        _probe_model(settings)
    except Exception as exc:
        _diagnose_probe_failure(settings, exc)
        raise SystemExit(1) from exc
    click.echo(f"✓ reachable, warmed in {_format_duration(time.monotonic() - start)}")


@cli.group()
def config() -> None:
    """Inspect effective configuration."""


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Print the effective settings and where each value comes from."""
    table = Table(title="aiida-agents configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_column("Env var", style="dim")
    table.add_column("Source", style="cyan")
    for setting, value, env_var, src in _config_rows(
        ctx.obj["provider"], ctx.obj["model"]
    ):
        table.add_row(setting, value, env_var, src)
    console.print(table)


def _module_missing(name: str) -> bool:  # pragma: no cover
    """True when importable module ``name`` is not available in this env."""
    import importlib.util

    return importlib.util.find_spec(name) is None


def _pip_install(spec: str) -> None:  # pragma: no cover
    """Install ``spec`` into the current environment (uv if present, else pip)."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable, spec]
    else:
        cmd = [sys.executable, "-m", "pip", "install", spec]
    click.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _ensure_docs_toolchain() -> None:  # pragma: no cover
    """Offer to install the docs toolchain if sphinx is missing.

    Building the RAG corpus shells out to sphinx (shipped by ``aiida-core[docs]``).
    Rather than fail with a manual install step, prompt to install it into the
    current environment. Declining, or a failed install, raises a clean error.
    """
    import importlib

    if not _module_missing("sphinx"):
        return
    click.echo("Building the RAG index needs the AiiDA docs toolchain (sphinx).")
    if not click.confirm(
        "Install aiida-core[docs] into the current environment now?", default=True
    ):
        raise click.ClickException(
            "Docs toolchain not installed. Run "
            "`uv pip install 'aiida-core[docs]'` and retry `aiida-agents rag build`."
        )
    try:
        _pip_install("aiida-core[docs]")
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"Install failed ({exc}). Install `aiida-core[docs]` manually and retry."
        ) from exc
    importlib.invalidate_caches()
    if _module_missing("sphinx"):
        raise click.ClickException(
            "Installed, but sphinx is still not importable here; "
            "install `aiida-core[docs]` manually and retry."
        )


@cli.group()
def rag() -> None:
    """Manage the AiiDA documentation RAG index."""


@rag.command("build")
@click.option("--force", is_flag=True, help="Rebuild even if an index already exists.")
def rag_build(force: bool) -> None:  # pragma: no cover
    """Build (or rebuild) the AiiDA docs RAG index."""
    from aiida_agents._settings import RagSettings
    from aiida_agents.rag import index_docs

    # Provision what the build needs before starting: the docs toolchain and,
    # for the default local embedder, the Ollama embedding model.
    _ensure_docs_toolchain()
    rag_cfg = RagSettings()
    if rag_cfg.embed_backend == "ollama":
        _prompt_pull_ollama_model(rag_cfg.embed_model)

    # The bar auto-refreshes on its own thread, so it animates while the
    # (blocking) build runs: indeterminate during the clone + sphinx phase, then
    # a real chunk count + ETA once embedding starts (via index_docs's callback).
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
            # Embedding a batch can take ~a minute on a weak GPU; the default 30s
            # speed window lets each sample age out before the next lands, so the
            # ETA never computes. Widen it to span several batches.
            speed_estimate_period=600.0,
        ) as bar:
            task = bar.add_task("Cloning docs, building text (sphinx)…", total=None)

            def _report(done: int, total: int) -> None:
                if done == 0:
                    # index_docs signals embed start with done=0; zero the timer
                    # here so the ETA reflects the embed rate, not the preceding
                    # clone/sphinx time.
                    bar.reset(task, total=total)
                bar.update(
                    task,
                    description=f"Embedding chunks {done}/{total}",
                    completed=done,
                    total=total,
                )

            index_docs(force=force, progress=_report)
    except (RuntimeError, OSError) as exc:
        # Any operational failure during the build becomes a clean CLI error
        # rather than a traceback: a failed sphinx build (RuntimeError), or the
        # embedder being unreachable / its model not pulled (urllib
        # HTTPError/URLError, both OSError subclasses).
        raise click.ClickException(f"RAG build failed: {exc}") from exc
    click.echo("✓ RAG index ready.")
