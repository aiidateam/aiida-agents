"""Shared fail-fast guard for a mistyped settings key.

A stray ``AIIDA_AGENTS_*`` (or near-miss) key is dropped silently by
pydantic-settings, so a command would otherwise run on an unintended default.
Action commands wrap their body in :func:`_needs_recognized_settings` to stop
before doing work; ``config`` commands instead report the same problem without
stopping (they exist to inspect and fix configuration).

This is the CLI *adapter* over the surface-agnostic detection in
``_settings.find_unrecognized_settings`` (which just returns
``SettingProblem`` data): only the *reaction* to a problem lives here, turning
it into a Click error / ``SystemExit``. The detection stays click-free so the
MCP server and RAG indexing can reuse it without pulling in the CLI framework.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import rich_click as click

from aiida_agents._settings import SettingProblem, find_unrecognized_settings


def _report_setting_problems(problems: list[SettingProblem]) -> None:
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
