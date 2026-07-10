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


def _run_diagnostics(
    settings: ModelSettings, profile: str | None
) -> list[_DiagnosticRow]:
    """Run each health check, returning a :class:`_DiagnosticRow` per check.

    Every check is isolated in its own ``try`` so one failure (an unreachable
    model, an unloadable profile) never aborts the rest of the report. The model
    check uses the no-generation reachability probe, so ``doctor`` never warms
    the model.
    """
    from aiida_agents.cli.agent import _probe_reachable

    rows: list[_DiagnosticRow] = []

    try:
        from aiida import load_profile

        loaded = load_profile(profile)
        rows.append(_DiagnosticRow("AiiDA profile loads", True, loaded.name))
    except Exception as exc:
        rows.append(_DiagnosticRow("AiiDA profile loads", False, _short_reason(exc)))

    model_label = f"Model reachable ({settings.provider}:{settings.model})"
    try:
        reach = _probe_reachable(settings)
        if reach.model_ok:
            rows.append(_DiagnosticRow(model_label, True, reach.endpoint))
        elif settings.provider == "ollama":
            rows.append(
                _DiagnosticRow(
                    model_label,
                    False,
                    f"model not pulled (ollama pull {settings.model})",
                )
            )
        else:
            rows.append(
                _DiagnosticRow(
                    model_label, True, "reachable; model not listed (may still work)"
                )
            )
    except Exception as exc:
        rows.append(_DiagnosticRow(model_label, False, _short_reason(exc)))

    try:
        from aiida_agents.rag.store import index_status

        built = index_status().built
        rows.append(
            _DiagnosticRow(
                "RAG index built",
                built,
                "" if built else "run `aiida-agents rag build`",
            )
        )
    except Exception as exc:
        rows.append(_DiagnosticRow("RAG index built", False, _short_reason(exc)))

    has_sphinx = not _module_missing("sphinx")
    rows.append(
        _DiagnosticRow(
            "Docs toolchain (sphinx)",
            has_sphinx,
            "" if has_sphinx else "needed only for `rag build`",
        )
    )
    return rows


@click.command()
@click.pass_context
@_needs_recognized_settings
def doctor(ctx: click.Context) -> None:
    """Diagnose the setup: profile, model, RAG index, and docs toolchain."""
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo("Running diagnostics ...\n")
    all_ok = True
    for label, ok, detail in _run_diagnostics(settings, ctx.obj["profile"]):
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        suffix = f" [dim]({detail})[/]" if detail else ""
        console.print(f"{mark} {label}{suffix}")
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit(1)
