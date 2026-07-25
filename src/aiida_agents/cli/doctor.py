"""The ``aiida-agents doctor`` command and the health checks behind it.

Each subsystem (AiiDA profile, model reachability, RAG index, docs toolchain)
is probed in its own ``try`` so one failure never aborts the rest of the
report. Only ``doctor`` runs these checks (``check``/``warm`` probe the model
through ``agent.py``), so the command lives here with its logic rather than in
``commands.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import rich_click as click
from rich.markup import escape
from typing_extensions import assert_never

from aiida_agents._settings import ModelSettings
from aiida_agents.cli._guards import _needs_recognized_settings
from aiida_agents.cli.agent import _resolve_settings_or_fail
from aiida_agents.cli.output import console
from aiida_agents.cli.rag import _module_missing


class _DiagnosticRow(NamedTuple):
    """One ``doctor`` health-check result: a label, pass/fail, and a detail."""

    label: str
    ok: bool
    detail: str


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


def _check_profile(profile: str | None) -> _DiagnosticRow:
    """Whether the AiiDA profile loads."""
    label = "AiiDA profile loads"
    try:
        from aiida import load_profile

        loaded = load_profile(profile)
        return _DiagnosticRow(label, True, loaded.name)
    except Exception as exc:
        return _DiagnosticRow(label, False, _short_reason(exc))


def _check_model(settings: ModelSettings) -> _DiagnosticRow:
    """Whether the configured model is reachable and advertised (no generation).

    Uses the no-generation reachability probe, so ``doctor`` never warms the
    model. An unadvertised model is fatal for Ollama (its listing is
    authoritative) but only a note for a cloud endpoint (its listing may be
    partial).
    """
    from aiida_agents.cli.agent import _model_availability, _probe_reachable

    label = f"Model reachable ({settings.provider}:{settings.model})"
    try:
        reach = _probe_reachable(settings)
        availability = _model_availability(reach, settings.provider)
        if availability == "available":
            return _DiagnosticRow(label, True, reach.endpoint)
        if availability == "not_pulled":
            return _DiagnosticRow(
                label, False, f"model not pulled (ollama pull {settings.model})"
            )
        if availability == "unlisted":
            return _DiagnosticRow(
                label, True, "reachable; model not listed (may still work)"
            )
        assert_never(availability)  # pragma: no cover
    except Exception as exc:
        return _DiagnosticRow(label, False, _short_reason(exc))


def _check_rag_index() -> _DiagnosticRow:
    """Whether the RAG documentation index is built."""
    label = "RAG index built"
    try:
        from aiida_agents.rag.store import index_status

        status = index_status()
        if status.built:
            return _DiagnosticRow(label, True, "")
        # Distinguish "never built" from "built, but nothing a query can reach":
        # the second looks healthy in the store and is the more confusing one,
        # so name it and give the command that actually fixes it.
        if status.stale:
            return _DiagnosticRow(
                label,
                False,
                "stale index (built for a different docs version, corpus format, "
                "or embedding model); run `aiida-agents rag build --force`",
            )
        return _DiagnosticRow(label, False, "run `aiida-agents rag build`")
    except Exception as exc:
        return _DiagnosticRow(label, False, _short_reason(exc))


def _check_docs_toolchain() -> _DiagnosticRow:
    """Whether the sphinx docs toolchain (needed only for ``rag build``) is present."""
    has_sphinx = not _module_missing("sphinx")
    return _DiagnosticRow(
        "Docs toolchain (sphinx)",
        has_sphinx,
        "" if has_sphinx else "needed only for `rag build`",
    )


def _run_diagnostics(
    settings: ModelSettings, profile: str | None
) -> list[_DiagnosticRow]:
    """Run each health check, one :class:`_DiagnosticRow` per check.

    Each check catches its own failure and reports it as a failed row, so one
    broken check (an unreachable model, an unloadable profile) never aborts the
    rest of the report.
    """
    return [
        _check_profile(profile),
        _check_model(settings),
        _check_rag_index(),
        _check_docs_toolchain(),
    ]


@click.command()
@click.pass_context
@_needs_recognized_settings
def doctor(ctx: click.Context) -> None:
    """Diagnose the setup: profile, model, RAG index, and docs toolchain."""
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo("Running diagnostics ...\n")
    rows = _run_diagnostics(settings, ctx.obj["profile"])
    for row in rows:
        # escape() the dynamic label/detail (a model name, or an error message
        # from _short_reason) so a stray bracket can't be swallowed as Rich
        # markup or raise MarkupError; the ✓/✗ and [dim] tags stay intentional.
        mark = "[green]✓[/]" if row.ok else "[red]✗[/]"
        suffix = f" [dim]({escape(row.detail)})[/]" if row.detail else ""
        console.print(f"{mark} {escape(row.label)}{suffix}")
    if not all(row.ok for row in rows):
        raise SystemExit(1)
