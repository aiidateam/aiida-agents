"""Click command surface for aiida-agents (``rich_click`` for pretty help)."""

from __future__ import annotations

import asyncio
import functools
import shutil
import subprocess
import sys
import time
from collections.abc import Callable

import rich_click as click
from pydantic_ai.tools import DeferredToolRequests
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from aiida_agents._logging import _configure_logging
from aiida_agents._settings import (
    LoggingSettings,
    ModelSettings,
    find_unrecognized_settings,
)
from aiida_agents.cli.session import (
    _build_agent,
    _check_reachable,
    _diagnose_probe_failure,
    _probe_model,
    _resolve_model_settings,
    ask,
)
from aiida_agents.cli.ollama import _prompt_pull_ollama_model
from aiida_agents.cli.config import _config_rows
from aiida_agents.cli.output import _format_duration, _render_tool_calls, console
from aiida_agents.cli.repl import _run_repl


# ``-h`` as a help alias, everywhere: ``help_option_names`` is read from the
# context, which every subcommand inherits, so this covers the whole tree.
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _report_setting_problems(problems: list[tuple[str, str | None]]) -> None:
    """Print one actionable line per key that looks like a mistyped setting."""
    for key, suggestion in problems:
        hint = (
            f"did you mean {suggestion}?"
            if suggestion is not None
            else "check for a typo."
        )
        click.echo(
            f"✗ {key} is not a recognised aiida-agents setting; {hint}", err=True
        )


def _needs_recognized_settings(command: Callable[..., None]) -> Callable[..., None]:
    """Fail fast, before the command runs, on a key that looks mistyped.

    Such a key is dropped silently by pydantic-settings, so the command would
    otherwise act on an unintended default (``AIIDA_AGENS_PROVIDER=openai`` runs
    with the ollama default). The check lives in the command body, so Click
    resolves ``--help`` / ``--version`` / completion before it can run and those
    always work. ``config`` commands are left undecorated: they exist to inspect
    and fix configuration, and flag the problem without stopping.
    """

    @functools.wraps(command)
    def wrapper(*args: object, **kwargs: object) -> None:
        problems = find_unrecognized_settings()
        if problems:
            _report_setting_problems(problems)
            click.echo("Fix or unset it and re-run.", err=True)
            raise SystemExit(2)
        command(*args, **kwargs)

    return wrapper


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
    settings = _resolve_model_settings(ctx.obj["provider"], ctx.obj["model"])
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
    settings = _resolve_model_settings(ctx.obj["provider"], ctx.obj["model"])
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
    settings = _resolve_model_settings(ctx.obj["provider"], ctx.obj["model"])
    click.echo("Running diagnostics …\n")
    all_ok = True
    for label, ok, detail in _run_diagnostics(settings, ctx.obj["profile"]):
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        suffix = f" [dim]({detail})[/]" if detail else ""
        console.print(f"{mark} {label}{suffix}")
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit(1)


@cli.group()
def config() -> None:
    """Inspect effective configuration."""


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Print the effective settings and where each value comes from."""
    # Flag (but don't fail on) a mistyped key: this is where you come to debug
    # configuration, so it must still render the table.
    _report_setting_problems(find_unrecognized_settings())
    table = Table(title="aiida-agents configuration")
    table.add_column("Group", style="dim")
    table.add_column("Setting", style="bold")
    # ``fold`` (not the table default ``ellipsis``) on the wide columns, so a
    # long value or env var wraps instead of hiding characters: both are read or
    # copied verbatim (a truncated URL, path, or ``AIIDA_AGENTS_MAX_TOK…`` is not
    # actionable). The short columns keep the default.
    table.add_column("Value", overflow="fold")
    table.add_column("Env var", style="dim", overflow="fold")
    table.add_column("Source", style="cyan")
    for group, setting, value, env_var, src in _config_rows(
        ctx.obj["provider"], ctx.obj["model"]
    ):
        table.add_row(group, setting, value, env_var, src)
    console.print(table)


@config.command("path")
def config_path() -> None:
    """Show where configuration is read from."""
    from pathlib import Path

    # click.echo (not console.print): the resolved path is long and must stay on
    # one line so it is copyable; rich would soft-wrap it to the terminal width.
    env_file = Path(".env").resolve()
    marker = "found" if env_file.is_file() else "not present"
    click.echo(f".env file: {env_file} ({marker})")
    click.echo(
        "Settings come from environment variables and this .env file. "
        "Run `aiida-agents config show` to see effective values and sources."
    )


@config.command("init")
@click.option(
    "--path",
    "path_str",
    type=click.Path(dir_okay=False),
    default=".env",
    show_default=True,
    help="Where to write the template.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def config_init(path_str: str, force: bool) -> None:
    """Write a commented ``.env`` template with the recognised settings."""
    from pathlib import Path

    from aiida_agents.cli.config import _env_template

    target = Path(path_str)
    if target.exists() and not force:
        raise click.ClickException(
            f"{target} already exists; pass --force to overwrite."
        )
    target.write_text(_env_template(), encoding="utf-8")
    click.echo(f"✓ Wrote {target}. Uncomment and edit the settings you want to change.")


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
@_needs_recognized_settings
def rag_build(force: bool) -> None:  # pragma: no cover
    """Build (or rebuild) the AiiDA docs RAG index."""
    from aiida_agents._settings import RagSettings
    from aiida_agents.rag import IndexOutcome, index_docs

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
            # Tear the bar down on exit so a fast no-op (an already-built index
            # returns before the first _report) leaves no stale, never-updated
            # "Cloning…" frame on screen. The permanent ✓ lines are console
            # prints above the live region, so they survive the teardown.
            transient=True,
            # Embedding a batch can take ~a minute on a weak GPU; the default 30s
            # speed window lets each sample age out before the next lands, so the
            # ETA never computes. Widen it to span several batches.
            speed_estimate_period=600.0,
        ) as bar:
            task = bar.add_task("Cloning docs, building text (sphinx)…", total=None)

            def _report(done: int, total: int) -> None:
                if done == 0:
                    # The corpus phase (clone + sphinx + chunk) has finished and
                    # embedding is starting. Leave a permanent line for it above
                    # the live bar so the completed step stays visible instead of
                    # being overwritten, then repurpose the task for embedding.
                    # Resetting also zeroes the timer, so the ETA reflects the
                    # embed rate, not the preceding clone/sphinx time.
                    console.print(f"✓ Text corpus ready ({total} chunks)")
                    bar.reset(task, total=total)
                bar.update(
                    task,
                    description=f"Embedding chunks {done}/{total}",
                    completed=done,
                    total=total,
                )

            outcome = index_docs(force=force, progress=_report)
    except (RuntimeError, OSError) as exc:
        # Any operational failure during the build becomes a clean CLI error
        # rather than a traceback: a failed sphinx build (RuntimeError), or the
        # embedder being unreachable / its model not pulled (urllib
        # HTTPError/URLError, both OSError subclasses).
        raise click.ClickException(f"RAG build failed: {exc}") from exc

    # Report what actually happened, so re-running on an up-to-date index reads
    # as a deliberate no-op rather than a fresh build.
    if outcome is IndexOutcome.ALREADY_PRESENT:
        click.echo("✓ RAG index already built (pass --force to rebuild).")
    elif outcome is IndexOutcome.EMPTY_CORPUS:
        click.echo("⚠ No documentation chunks were produced; index left unchanged.")
    else:
        click.echo("✓ RAG index ready.")


@rag.command("status")
@_needs_recognized_settings
def rag_status() -> None:
    """Show whether the docs index is built, and its details."""
    # Introspects the store on disk (no live embedder), so it reports the real
    # persisted state even when Ollama is down; see ``store.index_status``.
    from aiida_agents.rag.store import index_status

    status = index_status()
    console.print(f"[bold]Built:[/] {'yes' if status.built else 'no'}")
    console.print(
        f"[bold]Configured embedding:[/] {status.configured_embed_backend}"
        f" / {status.configured_embed_model}"
    )
    console.print(f"[bold]Docs version:[/] {status.configured_docs_version}")
    store_note = "" if status.store_exists else " (not created yet)"
    console.print(f"[bold]Store path:[/] {status.store_path}{store_note}")

    if not status.collections:
        console.print(
            "\nNo index built yet. Build it with [bold]aiida-agents rag build[/]."
        )
        return

    table = Table(title="\nCollections in the store")
    table.add_column("Chunks", justify="right")
    table.add_column("Docs version")
    table.add_column("Embedding")
    table.add_column("Collection", style="dim", overflow="fold")
    for info in status.collections:
        table.add_row(str(info.chunks), info.docs_version, info.embedding, info.name)
    console.print(table)


@rag.command("search")
@click.argument("query")
@click.option(
    "-n",
    "--limit",
    type=int,
    default=3,
    show_default=True,
    help="Maximum number of matches to show.",
)
@_needs_recognized_settings
def rag_search(query: str, limit: int) -> None:
    """Search the docs index directly for QUERY and print the top matches.

    A read-only shortcut to the same retrieval the agent uses, handy for
    checking what the index returns without starting a chat.
    """
    from aiida_agents.rag import query_docs
    from aiida_agents.rag.retriever import docs_index_available

    if not docs_index_available():
        raise click.ClickException(
            "No RAG index found. Build it with `aiida-agents rag build`."
        )
    results = query_docs(query, limit=limit)
    if not results:
        click.echo("No matching documentation found.")
        return
    for result in results:
        source = result.get("source", "unknown")
        section = result.get("section", "")
        title = f"{source}  §  {section}" if section else source
        # Text() renders the body literally: chunks contain bracketed headers
        # that rich's markup parser would otherwise swallow as style tags.
        console.print(
            Panel(Text(result.get("text", "")), title=title, border_style="cyan")
        )


@rag.command("clear")
@click.option("--yes", is_flag=True, help="Delete without the confirmation prompt.")
@_needs_recognized_settings
def rag_clear(yes: bool) -> None:
    """Delete the RAG store (vector index and cached text corpus)."""
    from aiida_agents._settings import RagSettings

    path = RagSettings().vector_db_path
    if not path.exists():
        click.echo("No RAG store to clear.")
        return
    if not yes and not click.confirm(f"Delete the RAG store at {path}?", default=False):
        click.echo("Cancelled.")
        return
    shutil.rmtree(path)
    click.echo(f"✓ Removed {path}.")


@cli.group()
def mcp() -> None:
    """Run the Model Context Protocol (MCP) server."""


@mcp.command("serve")
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to serve on (overrides AIIDA_AGENTS_PORT).",
)
@_needs_recognized_settings
def mcp_serve(port: int | None) -> None:  # pragma: no cover
    """Start the MCP server (streamable-http transport).

    The same server as ``python -m aiida_agents.mcp.server``, surfaced here so
    both entry points live under one command. Its lifespan loads the AiiDA
    profile and configures logging.
    """
    from aiida_agents._settings import ServerSettings
    from aiida_agents.mcp import server

    resolved = port if port is not None else ServerSettings().port
    click.echo(f"Starting MCP server on port {resolved} (Ctrl-C to stop) …")
    server.mcp.run(transport="streamable-http", port=resolved)
