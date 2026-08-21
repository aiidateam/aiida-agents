"""Re-run past calculations, one approval for the whole set.

"Find my failed relaxations from this week and resubmit them with a higher
cutoff" is the shape of request that made batch operations worth building, and
it has two distinct problems in it. Neither is solved by looping.

**Reconstructing what a run actually used.** Resubmitting means rebuilding the
original inputs, and a model asked to do that from memory of the conversation
gets a plausible tree rather than the real one --- silently dropping the
optional port that mattered, or inventing a value that was never set.
``build_resubmission_spec`` reads them back off the node itself, so what gets
re-run is what ran, plus exactly the overrides asked for.

**Approving a set.** Twenty submissions could be twenty approval prompts, which
is not review --- it is a button people learn to hold down. So a batch is *one*
tool call carrying every spec, which the approval gate shows enumerated and the
user accepts or rejects as a unit. That is the honest granularity: the decision
being made is "re-run these twenty", not twenty separate decisions.

Two consequences of that choice, both deliberate.

The batch is capped. A prompt listing two hundred submissions is not read, and
an approval that is not read is not an approval. Past the cap the tool refuses
and says to narrow the query.

The batch is all-or-nothing at validation. If any spec fails to resolve, none
are submitted and the whole call comes back to the model with the offending
one named. A half-submitted batch leaves the user with a partial set they did
not agree to and no clear record of which half ran.
"""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from pydantic import Field
from typing_extensions import TypedDict

from aiida_agents.tools._orm import WrongNodeType, load_node
from aiida_agents.tools._types import Identifier, SubmitResult
from aiida_agents.tools.execution._spec import to_spec_value
from aiida_agents.tools.execution.schemas import SubmissionSpec

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_BATCH",
    "BatchResult",
    "build_resubmission_spec",
    "submit_process_batch",
]

#: Most submissions one batch may carry. Set by what a person will actually
#: read in an approval prompt, not by what the engine could take.
MAX_BATCH = 20

# AiiDA stores a nested input port as a flat link label joined by this
# separator, and rebuilding the nesting from it is what turns a node's links
# back into something ``submit_process_spec`` accepts.
_NAMESPACE_SEPARATOR = "__"


class BatchResult(TypedDict):
    """Return shape of ``submit_process_batch``.

    ``submitted`` is reported alongside the count the caller asked for, so a
    batch that was trimmed or partially rejected cannot be read as a batch that
    ran whole.
    """

    requested: int
    submitted: int
    results: list[SubmitResult]


def _nest(flat: dict[str, t.Any]) -> dict[str, t.Any]:
    """Rebuild nested namespaces from AiiDA's flat ``a__b__c`` link labels."""
    nested: dict[str, t.Any] = {}
    for label, value in flat.items():
        parts = label.split(_NAMESPACE_SEPARATOR)
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def _deep_merge(
    base: dict[str, t.Any], overrides: dict[str, t.Any]
) -> dict[str, t.Any]:
    """Overlay ``overrides`` onto ``base``, merging nested mappings rather than replacing them.

    Replacing would be the wrong rule here: an override of
    ``{"parameters": {"SYSTEM": {"ecutwfc": 80}}}`` is asking to change one
    cutoff, not to discard every other parameter the original run set.
    """
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def build_resubmission_spec(
    identifier: Identifier,
    overrides: t.Annotated[
        dict[str, t.Any] | None,
        Field(
            description=(
                "Changes to apply on top of the original inputs, in the same "
                "nested shape the workflow declares. Merged into the original "
                'rather than replacing it, so {"parameters": {"SYSTEM": '
                '{"ecutwfc": 80}}} changes one cutoff and leaves the rest of '
                "the parameters as they were."
            )
        ),
    ] = None,
) -> SubmissionSpec:
    """Rebuild the spec that would re-run a past process, with changes applied.

    Use this for anything of the form "run that again, but ...". It reads the
    inputs off the process node itself, so the resubmission carries exactly
    what the original ran --- including the optional ports nobody mentioned ---
    rather than a reconstruction from what was said in conversation.

    Check the returned spec before submitting it. The inputs come from a real
    run, so they are self-consistent, but a change you asked for may want a
    companion change the original did not need: raising ``ecutwfc`` usually
    means raising ``ecutrho`` with it, and ``check_cutoffs_against_pseudos`` will say
    whether the pair still matches the pseudopotentials.

    Args:
        identifier: The process to re-run, by pk or uuid. It does not have to
            have failed --- re-running a successful calculation with a tighter
            parameter is just as valid.
        overrides: Nested changes to merge into the original inputs.

    Returns:
        A ``SubmissionSpec`` for the same process type, ready for
        ``submit_process_spec`` or ``submit_process_batch``.

    Raises:
        WrongNodeType: If the identifier names a data node rather than a process.
        ValueError: If the process node records no entry point to re-run.
    """
    logger.debug(
        "build_resubmission_spec(identifier=%r, overrides=%r)", identifier, overrides
    )
    node = load_node(identifier)
    if not isinstance(node, orm.ProcessNode):
        msg = (
            f"Node {identifier} is not a process node (type {type(node).__name__}). "
            "Only a process can be resubmitted."
        )
        raise WrongNodeType(msg)

    process_type = node.process_type or ""
    # Stored as "aiida.workflows:core.arithmetic.multiply_add"; the entry point
    # is the half after the group.
    entry_point = process_type.split(":", 1)[1] if ":" in process_type else ""
    if not entry_point:
        msg = (
            f"Process {identifier} records no entry point (process_type="
            f"{process_type!r}), so there is nothing to resubmit it as. A "
            "calcfunction or a process run from a script cannot be rebuilt "
            "this way."
        )
        raise ValueError(msg)

    flat = {
        entry.link_label: to_spec_value(entry.node)
        for entry in node.base.links.get_incoming().all()
    }
    inputs = _nest(flat)
    if overrides:
        inputs = _deep_merge(inputs, overrides)

    return {
        "entry_point": entry_point,
        "inputs": inputs,
        "metadata": {"source": "resubmission", "resubmitted_from": node.pk},
    }


def submit_process_batch(
    specs: t.Annotated[
        list[SubmissionSpec],
        Field(
            description=(
                "The submissions to run, each in the same shape "
                "submit_process_spec accepts. All are approved together, or "
                f"none are. At most {MAX_BATCH}."
            )
        ),
    ],
) -> BatchResult:
    """Submit several workflows under a single approval.

    Use this for a request about a *set* --- "resubmit the ones that failed",
    "run all of these with a higher cutoff" --- rather than calling
    ``submit_process_spec`` in a loop, which would ask the user to approve
    each one separately and turn review into a formality.

    Build each spec with ``build_resubmission_spec`` when re-running something,
    so the inputs come from the original node rather than from memory. Say how
    many you are about to submit, and what they have in common, before you call
    this: the user is approving a set, and needs to know what is in it.

    The whole batch is validated before anything is written. If one spec is
    bad, nothing is submitted and the error names it — fix that spec and call
    again rather than dropping it silently.

    Args:
        specs: One ``SubmissionSpec`` per submission.

    Returns:
        How many were requested, how many were submitted, and each result's pk.

    Raises:
        ValueError: If the list is empty, over ``MAX_BATCH``, or not a list of
            specs.
    """
    logger.debug("submit_process_batch(%d specs)", len(specs) if specs else 0)
    if not isinstance(specs, list) or not specs:
        msg = "specs must be a non-empty list of SubmissionSpec dictionaries."
        raise ValueError(msg)
    if len(specs) > MAX_BATCH:
        msg = (
            f"A batch is capped at {MAX_BATCH} submissions and this one has "
            f"{len(specs)}. A longer list cannot be reviewed properly in one "
            "approval — narrow the query that produced it, or submit in "
            "batches the user can actually read."
        )
        raise ValueError(msg)

    from aiida_agents.tools.execution.spec_execution import _validate_spec
    from aiida_agents.tools.execution.submit import _prepare_submission, _run_submission

    # Every spec is resolved and validated before any of them is submitted.
    # The obvious loop --- `[submit_process_spec(spec) for spec in specs]` ---
    # validated and submitted each in turn, so a bad spec at position N left
    # 1..N-1 already on the daemon and then raised, returning no BatchResult at
    # all. The user was told "all are approved together, or none are", approved
    # twenty, and got fourteen plus a traceback naming none of them.
    prepared = []
    for position, spec in enumerate(specs, start=1):
        entry_point, inputs = _validate_spec(spec)
        try:
            process_class, resolved = _prepare_submission(entry_point, inputs)
        except Exception as exc:
            # Name the position as well as the reason: the specs were built
            # from a query, so "the ninth one" is how the user finds it.
            msg = (
                f"Spec {position} of {len(specs)} ({entry_point}) is not "
                f"submittable, so none of the batch was submitted: {exc}"
            )
            raise ValueError(msg) from exc
        prepared.append((entry_point, process_class, resolved))

    results = []
    for position, (entry_point, process_class, resolved) in enumerate(
        prepared, start=1
    ):
        try:
            results.append(_run_submission(entry_point, process_class, resolved))
        except Exception as exc:
            # Validation is all-or-nothing; submission cannot be. There is no
            # transaction across the daemon, so if the engine fails partway the
            # batch is genuinely split --- and the one thing that must not
            # happen then is losing the record of what did go out. Raising here
            # would discard the pks of everything already submitted, which is
            # how a user ends up unable to tell what is running.
            submitted = ", ".join(str(result.get("pk")) for result in results)
            msg = (
                f"Submitted {len(results)} of {len(specs)} before spec "
                f"{position} ({entry_point}) failed: {exc}. "
                f"Already running: {submitted or 'none'}. "
                "Do not resubmit those; check them with get_process_status."
            )
            raise RuntimeError(msg) from exc

    return BatchResult(requested=len(specs), submitted=len(results), results=results)
