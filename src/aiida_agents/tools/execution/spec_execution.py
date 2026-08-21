"""Tool for submitting a process spec to the daemon.

Takes the ``SubmissionSpec`` the model built (``build_process_inputs`` or
``draft_process_inputs``), checks its shape and entry point, and delegates to
``submit_workflow`` behind the Human-In-The-Loop approval gate.
"""

from __future__ import annotations

import logging
import typing as t

from aiida.common.exceptions import MissingEntryPointError
from aiida.plugins.entry_point import load_entry_point
from pydantic import Field

from aiida_agents.tools._types import SubmitResult
from aiida_agents.tools.execution.submit import submit_workflow
from aiida_agents.tools.execution.schemas import SubmissionSpec

logger = logging.getLogger(__name__)

__all__ = ["submit_process_spec"]


def submit_process_spec(
    spec: t.Annotated[
        SubmissionSpec,
        Field(
            description="The exact SubmissionSpec dictionary to submit, containing 'entry_point' and 'inputs'."
        ),
    ],
) -> SubmitResult:
    """Submit a process spec to the AiiDA daemon.

    This tool takes a SubmissionSpec containing ``entry_point`` and ``inputs``,
    verifies the structure and entry point, and safely delegates to ``submit_workflow``.
    The daemon runs the process from there, so this returns as soon as it is
    accepted: follow up with ``get_process_status`` on the pk it reports.
    Because this writes nodes and hands work to the daemon, the tool requires
    Human-In-The-Loop confirmation (``requires_approval=True``).

    Args:
        spec: The SubmissionSpec dictionary (must contain 'entry_point'
            and 'inputs').

    Returns:
        Submission result dictionary containing the process PK and UUID or error
        details.
    """
    logger.debug("submit_process_spec(spec=%r)", spec)

    entry_point, inputs = _validate_spec(spec)

    # Delegate to our secure engine submission function
    logger.info("Delegating spec (%s) to submit_workflow", entry_point)
    return submit_workflow(entry_point=entry_point, inputs=inputs)


def _validate_spec(spec: SubmissionSpec) -> tuple[str, dict[str, t.Any]]:
    """Check a spec's shape and entry point, and return its two halves.

    Separated from :func:`submit_process_spec` so a batch can validate every
    spec *before* submitting any of them. Inlined, this check ran immediately
    before the submission it guarded, which is fine for one spec and wrong for
    twenty: the twentieth was checked after the nineteenth had already been
    handed to the daemon.

    Writes nothing and loads nothing but the entry point.

    Args:
        spec: A ``SubmissionSpec`` with ``entry_point`` and ``inputs``.

    Returns:
        ``(entry_point, inputs)``, ready for submission.

    Raises:
        TypeError: If ``spec`` is not a dictionary.
        ValueError: If either key is missing, or ``entry_point`` is not a known
            AiiDA entry point.
    """
    if not isinstance(spec, dict):
        msg = "spec must be a dictionary."
        raise TypeError(msg)

    entry_point = spec.get("entry_point")
    inputs = spec.get("inputs")

    if not entry_point or not isinstance(entry_point, str):
        msg = "spec is missing required 'entry_point' string parameter."
        raise ValueError(msg)

    if not inputs or not isinstance(inputs, dict):
        msg = "spec is missing required 'inputs' dictionary."
        raise ValueError(msg)

    for group in ("aiida.workflows", "aiida.calculations"):
        try:
            load_entry_point(group, entry_point)
            return entry_point, inputs
        except MissingEntryPointError:
            continue
        except Exception as exc:
            msg = f"Process {entry_point!r} failed to load: {exc}"
            raise ValueError(msg) from exc

    msg = f"Process {entry_point!r} is not a known AiiDA entry point."
    raise ValueError(msg)
