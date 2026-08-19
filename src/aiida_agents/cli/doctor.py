"""The ``aiida-agents doctor`` command and the health checks behind it.

Each subsystem (AiiDA profile, daemon, model reachability, RAG index, codegen
sandbox, docs toolchain) is probed in its own ``try`` so one failure never
aborts the rest of the report. This is the only command that diagnoses a setup:
the model probes live in ``agent.py`` and are rendered here, so the logic sits
with the command rather than in ``commands.py``.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import rich_click as click
from pydantic import ValidationError
from rich.markup import escape
from typing_extensions import assert_never

from aiida_agents._settings import ModelSettings, _format_validation_error
from aiida_agents.cli._guards import _needs_recognized_settings
from aiida_agents.cli.agent import _resolve_settings_or_fail
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


def _probe_reason(settings: ModelSettings, exc: Exception) -> str:
    """A failed model probe as a row detail: the actionable hint, or the message.

    An unpulled model, a bad key and a dead endpoint are the failures a user can
    fix, and :func:`~aiida_agents.cli.agent._probe_failure_hint` phrases those as
    the fix. Anything else has no known remedy, so the exception speaks for
    itself rather than being dressed up as advice.
    """
    from aiida_agents.cli.agent import _probe_failure_hint

    hint = _probe_failure_hint(settings, exc)
    return hint if hint is not None else _short_reason(exc)


def _check_profile(profile: str | None) -> _DiagnosticRow:
    """Whether the AiiDA profile loads."""
    label = "AiiDA profile loads"
    try:
        from aiida import load_profile

        loaded = load_profile(profile)
        return _DiagnosticRow(label, True, loaded.name)
    except Exception as exc:
        return _DiagnosticRow(label, False, _short_reason(exc))


def _check_daemon() -> _DiagnosticRow:
    """Whether the AiiDA daemon is running and reachable.

    A loading profile is not enough: a submitted process only progresses if the
    daemon is up *and* its workers can be reached. The two differ, the circus
    endpoint can be briefly unreachable mid-restart while the daemon is
    technically alive, so this asks for the worker count to confirm reachability
    rather than trusting liveness alone.
    """
    label = "Daemon running and reachable"
    # Two ways to reach one condition (see the handler below), so one sentence.
    stopped = "not running; run `verdi daemon start`"
    try:
        from aiida.engine.daemon.client import (
            DaemonNotRunningException,
            DaemonStalePidException,
            DaemonTimeoutException,
            get_daemon_client,
        )

        client = get_daemon_client()
        if not client.is_daemon_running:
            return _DiagnosticRow(label, False, stopped)
        # Running is not reachable. get_numprocesses() round-trips to the daemon,
        # so a raise here means up-but-unreachable.
        try:
            response = client.get_numprocesses()
        except DaemonNotRunningException:
            # `is_daemon_running` only reads the PID file, so the daemon can stop
            # between the two calls. Same condition as the branch above, so it
            # gets that branch's sentence rather than a second one for one fault.
            return _DiagnosticRow(label, False, stopped)
        except DaemonStalePidException:
            # A stale PID file is how `is_daemon_running` says yes while the
            # round-trip fails, so it is this row's main non-trivial path. Named
            # here because AiiDA puts the remedy in a second sentence 100
            # characters in, past where `_short_reason` cuts the detail off.
            # `start` alone clears the file (AiiDA: "Either stop or start").
            return _DiagnosticRow(
                label,
                False,
                "stale PID file; run `verdi daemon start`, which clears it",
            )
        except DaemonTimeoutException:
            return _DiagnosticRow(
                label,
                False,
                "up but not answering; run `verdi daemon restart`",
            )
        workers = response.get("numprocesses")
        if workers is None:
            # circus reports a command-level failure in the payload rather than
            # by raising (``circus.commands.base.error``), and that payload
            # carries no count. Treating a missing count as "reachable" passed
            # the row for a daemon that had just answered with an error, which
            # is the one thing this check exists to catch.
            reason = str(response.get("reason") or response.get("status") or "no count")
            return _DiagnosticRow(
                label,
                False,
                f"answered without a worker count ({reason[:40]}); "
                "run `verdi daemon restart`",
            )
        if workers == 0:
            return _DiagnosticRow(
                label, False, "running but 0 workers; run `verdi daemon incr 1`"
            )
        return _DiagnosticRow(label, True, f"{workers} worker(s)")
    except Exception as exc:
        return _DiagnosticRow(label, False, _short_reason(exc))


def _check_model(settings: ModelSettings) -> _DiagnosticRow:
    """Whether the configured model is reachable and advertised (no generation).

    Uses the no-generation reachability probe, so the default report never
    generates; ``--warm`` adds the row that does. An unadvertised model is fatal
    for Ollama (its listing is authoritative) but only a note for a cloud
    endpoint (its listing may be partial).
    """
    from aiida_agents.cli.agent import (
        _model_availability,
        _not_pulled_detail,
        _probe_reachable,
    )

    label = f"Model reachable ({settings.provider}:{settings.model})"
    try:
        reach = _probe_reachable(settings)
        availability = _model_availability(reach, settings.provider)
        if availability == "available":
            return _DiagnosticRow(label, True, reach.endpoint)
        if availability == "not_pulled":
            return _DiagnosticRow(label, False, _not_pulled_detail(settings))
        if availability == "unlisted":
            return _DiagnosticRow(
                label, True, "reachable; model not listed (may still work)"
            )
        assert_never(availability)  # pragma: no cover
    except Exception as exc:
        return _DiagnosticRow(label, False, _probe_reason(settings, exc))


# Written twice, like the daemon label: by the check below, and by the skip that
# stands in for it when the model is unreachable.
_WARM_LABEL = "Model generates"


def _warm_model(settings: ModelSettings) -> _DiagnosticRow:
    """Whether the model actually generates, and how long the first call takes.

    Reachability is not generation: an endpoint can advertise a model it then
    fails to serve, and only a real call finds that out. For a local Ollama model
    the call also loads it into memory, so the first query of a session is not a
    cold start.

    The only check that spends tokens, which is why it is behind ``--warm``
    instead of part of the default report: running ``doctor`` against a paid
    provider should not cost anything.
    """
    from aiida_agents.cli.agent import _probe_model

    label = _WARM_LABEL
    start = time.monotonic()
    try:
        _probe_model(settings)
    except Exception as exc:
        return _DiagnosticRow(label, False, _probe_reason(settings, exc))
    elapsed = _format_duration(time.monotonic() - start)
    return _DiagnosticRow(label, True, f"warmed in {elapsed}")


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
            # No remedy named here: `check` is where the two cases are told
            # apart, and only one of them is the user's to fix.
            return _DiagnosticRow(
                label,
                False,
                "; ".join(entry.describe() for entry in sharing)
                + "; `aiida-agents sandbox check` spells out what to do",
            )
        return _DiagnosticRow(label, True, "own copy, shared with nothing")
    except ValidationError as exc:
        # Named before the broad except below, which would render this as "1
        # validation error for SandboxSettings": neither the setting nor the
        # fix. One red row rather than a raised error, so the rest of the
        # report still prints; diagnosing a broken setup is the whole job.
        reason = "; ".join(_format_validation_error(exc).splitlines())
        return _DiagnosticRow(label, False, reason)
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
    settings: ModelSettings, profile: str | None, *, warm: bool = False
) -> list[_DiagnosticRow]:
    """Run each health check, one :class:`_DiagnosticRow` per check.

    Each check catches its own failure and reports it as a failed row, so one
    broken check (an unreachable model, an unloadable profile) never aborts the
    rest of the report.

    ``warm`` appends the generation check. It is skipped when the model is not
    reachable in the first place: the call would fail for the reason already on
    the row above it, and one problem should not print as two.
    """
    model = _check_model(settings)
    rows = [
        _check_profile(profile),
        _check_daemon(),
        model,
        _check_rag_index(),
        _check_sandbox(),
        _check_docs_toolchain(),
    ]
    if warm:
        rows.append(
            _warm_model(settings)
            if model.ok
            else _DiagnosticRow(
                _WARM_LABEL, False, "not attempted; the model is unreachable"
            )
        )
    return rows


@click.command()
@click.option(
    "--warm",
    is_flag=True,
    help=(
        "Also send one tiny generation, which proves the model serves and "
        "pre-loads a local Ollama model so the first query is not a cold start. "
        "Costs a request on a paid provider, so it is off by default."
    ),
)
@click.pass_context
@_needs_recognized_settings
def doctor(ctx: click.Context, warm: bool) -> None:
    """Diagnose the setup: profile, daemon, model, RAG index, sandbox, docs toolchain.

    Read-only and free by default: nothing here generates. Pass `--warm` to
    also check that the model serves, and to pre-load a local one.
    """
    settings = _resolve_settings_or_fail(ctx.obj["provider"], ctx.obj["model"])
    click.echo("Running diagnostics ...\n")
    rows = _run_diagnostics(settings, ctx.obj["profile"], warm=warm)
    for row in rows:
        # escape() the dynamic label/detail (a model name, or an error message
        # from _short_reason) so a stray bracket can't be swallowed as Rich
        # markup or raise MarkupError; the ✓/✗ and [dim] tags stay intentional.
        mark = "[green]✓[/]" if row.ok else "[red]✗[/]"
        suffix = f" [dim]({escape(row.detail)})[/]" if row.detail else ""
        console.print(f"{mark} {escape(row.label)}{suffix}")
    if not all(row.ok for row in rows):
        raise SystemExit(1)
