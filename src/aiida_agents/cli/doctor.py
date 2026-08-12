"""The ``aiida-agents doctor`` command and the health checks behind it.

Each subsystem (AiiDA profile, model reachability, RAG index, codegen sandbox,
docs toolchain) is probed in its own ``try`` so one failure never aborts the
rest of the report.

This is the *only* diagnostic command. There were three, and telling them apart
was a puzzle nobody should have to solve: ``check`` probed config and model
reachability, ``doctor`` did that plus everything else, and ``warm`` was the
same model probe with a real generation instead of a listing. So ``check`` was
a strict subset of ``doctor``, and ``warm`` differed from its model row by one
boolean (issue #75).

They are now flags on the sweep. ``--only model`` is what ``check`` did,
``--only model --warm`` is what ``warm`` did, and plain ``doctor`` is still
everything --- the default stays the full picture, because this is the command
you reach for when something is already wrong.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import rich_click as click
from rich.markup import escape
from typing_extensions import assert_never

from aiida_agents._settings import ModelSettings
from aiida_agents.cli._guards import _needs_recognized_settings
from aiida_agents.cli.agent import _resolve_settings_or_fail, probe_failure_reason
from aiida_agents.cli.output import _format_duration, console
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


def _check_model(settings: ModelSettings, warm: bool = False) -> _DiagnosticRow:
    """Whether the configured model is reachable and advertised.

    Reachability is probed without generating, so the default sweep stays cheap.
    An unadvertised model is fatal for Ollama (its listing is authoritative) but
    only a note for a cloud endpoint (its listing may be partial).

    ``warm`` additionally fires one tiny generation. That is a different
    question --- "can it answer" rather than "is it there" --- and for a local
    model it has the side effect of loading it into memory, so the first real
    query is not a cold start. It is opt-in because it costs a round-trip, and
    on a cloud model, a token.
    """
    from aiida_agents.cli.agent import (
        _model_availability,
        _probe_model,
        _probe_reachable,
    )

    label = f"Model reachable ({settings.provider}:{settings.model})"
    try:
        reach = _probe_reachable(settings)
        availability = _model_availability(reach, settings.provider)
        if warm and availability != "not_pulled":
            # After the listing probe, so an unpulled Ollama model is reported
            # as unpulled rather than as whatever generating against it raises.
            started = time.monotonic()
            _probe_model(settings)
            elapsed = _format_duration(time.monotonic() - started)
            return _DiagnosticRow(label, True, f"{reach.endpoint}; warmed in {elapsed}")
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
        # `probe_failure_reason`, not `_short_reason`: a connection error should
        # say what to do about it. That mapping was the value `check` and `warm`
        # added over a raw traceback, so it moves here rather than dying with
        # them.
        return _DiagnosticRow(label, False, probe_failure_reason(settings, exc))


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


def _check_sandbox() -> _DiagnosticRow:
    """Whether the Codegen agent has a profile it can actually run code against.

    Reported here because the alternative is finding out from the answers. With
    no sandbox the agent still replies --- it writes the snippet, says it could
    not run it, and hands it over unverified --- which reads like a limitation
    of the model rather than a setup step nobody performed. One red row in
    ``doctor``, naming the command, is cheaper than that.

    A profile that exists is not enough. The question asked here is the one
    ``sandbox check`` asks: does it share storage with a real profile? A sandbox
    that does is a profile whose deletion destroys somebody's work, which is
    exactly how issue #73 cost a maintainer his database --- so it fails this
    row rather than passing it with a note.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents._settings import SandboxSettings
    from aiida_agents.sandbox.copy import profiles_sharing_storage

    label = "Codegen sandbox (disposable copy)"
    try:
        name = SandboxSettings().sandbox_profile
        config = get_config()
        profiles = {profile.name: profile for profile in config.profiles}
        if name not in profiles:
            return _DiagnosticRow(
                label,
                False,
                f"no profile {name!r}; run `aiida-agents sandbox init` "
                "(codegen writes code but cannot run it without one)",
            )

        sharing = profiles_sharing_storage(config, name)
        if sharing:
            return _DiagnosticRow(
                label,
                False,
                f"shares storage with {', '.join(sharing)}; deleting it would "
                "destroy that data. Rebuild with `aiida-agents sandbox refresh`",
            )
        return _DiagnosticRow(label, True, "own copy, shared with nothing")
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


#: Every check, by the name ``--only`` uses. Order is report order.
#:
#: One mapping rather than a list plus a set of choices: ``--only`` derives its
#: choices from these keys, so a check cannot exist without being selectable,
#: and a name cannot be offered that selects nothing.
_CHECKS: tuple[str, ...] = ("profile", "model", "rag", "sandbox", "docs")


def _run_diagnostics(
    settings: ModelSettings,
    profile: str | None,
    only: tuple[str, ...] = (),
    warm: bool = False,
) -> list[_DiagnosticRow]:
    """Run the selected health checks, one :class:`_DiagnosticRow` per check.

    Each check catches its own failure and reports it as a failed row, so one
    broken check (an unreachable model, an unloadable profile) never aborts the
    rest of the report.

    Args:
        settings: Resolved model/provider configuration.
        profile: AiiDA profile to load, or None for the default.
        only: Check names to run. Empty means all of them, which is the
            default because ``doctor`` is what you run when something is
            already wrong and you want the whole picture.
        warm: Also fire one real generation on the model check, loading a
            local model into memory instead of only probing reachability.
    """
    selected = only or _CHECKS
    runners = {
        "profile": lambda: _check_profile(profile),
        "model": lambda: _check_model(settings, warm=warm),
        "rag": _check_rag_index,
        "sandbox": _check_sandbox,
        "docs": _check_docs_toolchain,
    }
    return [runners[name]() for name in _CHECKS if name in selected]


@click.command()
@click.option(
    "--only",
    type=click.Choice(_CHECKS, case_sensitive=False),
    multiple=True,
    help=(
        "Run only these checks (repeatable). The default is all of them; "
        "`--only model` is the quick reachability probe on its own."
    ),
)
@click.option(
    "--warm",
    is_flag=True,
    help=(
        "Also generate once, so a local model is loaded into memory rather "
        "than only probed. Mainly useful before an interactive session."
    ),
)
@click.pass_context
@_needs_recognized_settings
def doctor(ctx: click.Context, only: tuple[str, ...], warm: bool) -> None:
    """Diagnose the setup: profile, model, RAG index, sandbox, docs toolchain.

    The one diagnostic command. `--only` narrows it; `--warm` makes the model
    check load the model instead of merely reaching it.

    Exits non-zero if any check that ran failed, so it can gate a script.
    """
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo("Running diagnostics ...\n")
    rows = _run_diagnostics(settings, ctx.obj["profile"], only=only, warm=warm)
    for row in rows:
        # escape() the dynamic label/detail (a model name, or an error message
        # from _short_reason) so a stray bracket can't be swallowed as Rich
        # markup or raise MarkupError; the ✓/✗ and [dim] tags stay intentional.
        mark = "[green]✓[/]" if row.ok else "[red]✗[/]"
        suffix = f" [dim]({escape(row.detail)})[/]" if row.detail else ""
        console.print(f"{mark} {escape(row.label)}{suffix}")
    if not all(row.ok for row in rows):
        raise SystemExit(1)
