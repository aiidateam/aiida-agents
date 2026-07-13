"""``aiida-agents rag`` command group: build, inspect, and query the docs index.

The heavy indexing/retrieval logic lives in the ``aiida_agents.rag`` package;
this module is the Click surface plus the docs-toolchain provisioning helpers
(``sphinx`` is an opt-in dependency, offered on demand before a build).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import rich_click as click
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

from aiida_agents.cli._guards import _needs_recognized_settings
from aiida_agents.cli.ollama import _prompt_pull_ollama_model
from aiida_agents.cli.output import console

if TYPE_CHECKING:
    from aiida_agents.rag import IndexOutcome


def _module_missing(name: str) -> bool:
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


def _ensure_docs_toolchain() -> None:
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


@click.group("rag")
def rag() -> None:
    """Manage the AiiDA documentation RAG index."""


def _build_with_progress(force: bool) -> IndexOutcome:  # pragma: no cover
    """Run ``index_docs`` behind a live progress bar, turning an operational
    build failure into a clean CLI error.

    The bar auto-refreshes on its own thread, so it animates while the (blocking)
    build runs: indeterminate during the clone + sphinx phase, then a real chunk
    count + ETA once embedding starts (via ``index_docs``'s callback).
    """
    from aiida_agents.rag import index_docs

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
            # Tear the bar down on exit so a fast no-op (an already-built index
            # returns before the first _report) leaves no stale, never-updated
            # "Cloning..." frame on screen. The permanent ✓ lines are console
            # prints above the live region, so they survive the teardown.
            transient=True,
            # Embedding a batch can take ~a minute on a weak GPU; the default 30s
            # speed window lets each sample age out before the next lands, so the
            # ETA never computes. Widen it to span several batches.
            speed_estimate_period=600.0,
        ) as bar:
            task = bar.add_task("Cloning docs, building text (sphinx)...", total=None)

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

            return index_docs(force=force, progress=_report)
    except (RuntimeError, OSError) as exc:
        # Any operational failure during the build becomes a clean CLI error
        # rather than a traceback: a failed sphinx build (RuntimeError), or the
        # embedder being unreachable / its model not pulled (urllib
        # HTTPError/URLError, both OSError subclasses).
        raise click.ClickException(f"RAG build failed: {exc}") from exc


def _report_build_outcome(outcome: IndexOutcome) -> None:
    """Print what the build actually did, so re-running on an up-to-date index
    reads as a deliberate no-op rather than a fresh build.
    """
    from aiida_agents.rag import IndexOutcome

    if outcome is IndexOutcome.ALREADY_PRESENT:
        click.echo("✓ RAG index already built (pass --force to rebuild).")
    elif outcome is IndexOutcome.EMPTY_CORPUS:
        click.echo("⚠ No documentation chunks were produced; index left unchanged.")
    else:
        click.echo("✓ RAG index ready.")


@rag.command("build")
@click.option("--force", is_flag=True, help="Rebuild even if an index already exists.")
@_needs_recognized_settings
def rag_build(force: bool) -> None:  # pragma: no cover
    """Build (or rebuild) the AiiDA docs RAG index."""
    from aiida_agents._settings import RagSettings

    # Provision what the build needs before starting: the docs toolchain and,
    # for the default local embedder, the Ollama embedding model.
    _ensure_docs_toolchain()
    rag_cfg = RagSettings()
    if rag_cfg.embed_backend == "ollama":
        _prompt_pull_ollama_model(rag_cfg.embed_model)

    _report_build_outcome(_build_with_progress(force))


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
@click.option(
    "--refs-only",
    is_flag=True,
    help="List only each match's source and section, not the full text.",
)
@_needs_recognized_settings
def rag_search(query: str, limit: int, refs_only: bool) -> None:
    """Search the docs index directly for QUERY and print the top matches.

    A read-only shortcut to the same retrieval the agent uses, handy for
    checking what the index returns without starting a chat. ``--refs-only``
    prints just the ``source § section`` of each match (one per line), for
    scanning or piping.
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
        if refs_only:
            click.echo(title)  # plain one-liner, so the list stays pipeable
            continue
        # Text() renders the body literally: chunks contain bracketed headers
        # that rich's markup parser would otherwise swallow as style tags.
        console.print(
            Panel(Text(result.get("text", "")), title=title, border_style="cyan")
        )


@rag.command("clear")
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Delete without the confirmation prompt.",
)
@_needs_recognized_settings
def rag_clear(force: bool) -> None:
    """Delete the RAG store (vector index and cached text corpus)."""
    from aiida_agents._settings import RagSettings

    path = RagSettings().vector_db_path
    if not path.exists():
        click.echo("No RAG store to clear.")
        return
    if not force and not click.confirm(
        f"Delete the RAG store at {path}?", default=False
    ):
        click.echo("Cancelled.")
        return
    shutil.rmtree(path)
    click.echo(f"✓ Removed {path}.")
