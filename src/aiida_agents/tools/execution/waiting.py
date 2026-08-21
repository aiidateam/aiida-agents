"""Wait for a submitted process, and report what it produced.

Every submission so far has been terminal in practice: the agent submits, calls
``get_process_status`` once, reports "created" or "waiting", and stops. That is
the right shape for "run this", and it is why nothing here could ever do "relax
this structure *and then* compute its band structure" --- the second step needs
the first one's output structure, which does not exist until the first step has
finished.

This closes that. ``wait_for_process`` polls until the process terminates and
returns its outputs with their link labels, which is exactly the handle the
next submission needs: ``output_structure`` comes back as a pk, and that pk
goes straight into the next spec's ``structure`` input.

Two things it deliberately refuses to do.

**It does not wait indefinitely.** A relaxation can run for hours, and an agent
loop blocked for hours is worse than one that reports honestly. The wait is
bounded and a timeout returns ``terminated: false`` with the process's current
state --- a fact about the *wait*, not about the process, which is very likely
still running fine. The user can be told to check back, and ``get_process_status``
will answer later.

**It does not return outputs for a running process.** They are incomplete until
the process terminates, and a chained submission that consumes a half-written
result is a subtle, expensive way to be wrong. A non-terminated wait reports no
outputs at all rather than a partial set that reads like a complete one.

What this is *not*: constructing an AiiDA ``WorkChain``. It composes existing
workflows by running them in sequence, each with its own approval, which is a
different thing with a different provenance shape --- the steps are linked by
the data flowing between them rather than by a parent process. Where a plugin
already ships a work chain that does the whole sequence (``PwBandsWorkChain``
relaxes *and* computes bands), that is still the better answer and
``list_process_entry_points`` will find it.
"""

from __future__ import annotations

import logging
import time
import typing as t

from aiida import orm
from pydantic import Field

from aiida_agents.tools._orm import WrongNodeType, load_node
from aiida_agents.tools._types import Identifier, NodeLink, ProcessCompletion

logger = logging.getLogger(__name__)

__all__ = ["wait_for_process"]

# Seconds between polls. AiiDA's daemon updates a process node's state in the
# database, so this is a database read, not a scheduler query -- cheap enough
# to do often, and slow enough not to hammer the profile.
_POLL_SECONDS = 2.0

# Hard ceiling on any single wait, whatever the caller asks for. The agent loop
# has to stay responsive; a genuinely long run is reported as unfinished and
# checked again later, not sat on.
_MAX_TIMEOUT_SECONDS = 1800

_DEFAULT_TIMEOUT_SECONDS = 300


def _outputs_of(node: orm.ProcessNode) -> list[NodeLink]:
    """The process's outgoing links, in the shape ``get_node_outputs`` returns."""
    return [
        NodeLink(
            pk=t.cast(int, entry.node.pk),
            uuid=entry.node.uuid,
            node_type=entry.node.node_type,
            link_label=entry.link_label,
            link_type=entry.link_type.value,
        )
        for entry in node.base.links.get_outgoing().all()
    ]


def _completion(
    node: orm.ProcessNode, *, terminated: bool, waited: float
) -> ProcessCompletion:
    return ProcessCompletion(
        pk=t.cast(int, node.pk),
        process_label=t.cast(str, node.process_label),
        state=node.process_state.value if node.process_state else None,
        exit_status=node.exit_status,
        exit_message=node.exit_message,
        terminated=terminated,
        waited_seconds=round(waited, 1),
        outputs=_outputs_of(node) if terminated else [],
    )


def wait_for_process(
    identifier: Identifier,
    timeout_seconds: t.Annotated[
        int,
        Field(
            description=(
                "How long to wait before giving up and reporting the process as "
                f"unfinished. Capped at {_MAX_TIMEOUT_SECONDS}s. Giving up is not "
                "a failure: the process keeps running and can be checked again."
            )
        ),
    ] = _DEFAULT_TIMEOUT_SECONDS,
) -> ProcessCompletion:
    """Wait for a submitted process to finish, then report what it produced.

    Use this to run one thing *after* another: relax a structure, wait, then
    submit a band-structure calculation against the relaxed geometry. The
    returned ``outputs`` carry each result's link label and pk, so the next
    step's input is a pk you were given rather than one you guessed ---
    ``output_structure`` from a relaxation goes straight into the next spec's
    ``structure``.

    Check ``terminated`` before anything else:

    - **true** --- the process finished. ``exit_status`` says how (0 is success),
      and ``outputs`` is complete. If it failed, hand the pk to the Analysis
      agent's ``diagnose_process_failure`` rather than resubmitting blindly.
    - **false** --- the wait ran out, nothing more. The process is still running.
      Say so plainly, give the pk, and tell the user it can be checked later.
      **Do not** describe this as a failure, a hang, or a timeout of the
      calculation, and do not start the next step: its input does not exist yet.

    Before chaining two submissions by hand, check whether a single workflow
    already does the whole sequence --- ``PwBandsWorkChain`` relaxes and computes
    bands in one submission, with one approval and a cleaner provenance graph
    than two runs stitched together. Chain only when no such workflow exists.

    Args:
        identifier: The process's pk or uuid, as returned by
            ``submit_process_spec``.
        timeout_seconds: How long to wait before reporting it unfinished.

    Returns:
        The process's final (or current) status, with its outputs once it has
        terminated and an empty list while it has not.

    Raises:
        WrongNodeType: If the identifier names a data node rather than a process.
        ValueError: If ``timeout_seconds`` is not positive.
    """
    logger.debug(
        "wait_for_process(identifier=%r, timeout_seconds=%r)",
        identifier,
        timeout_seconds,
    )
    if timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {timeout_seconds}."
        raise ValueError(msg)
    budget = min(int(timeout_seconds), _MAX_TIMEOUT_SECONDS)

    node = load_node(identifier)
    if not isinstance(node, orm.ProcessNode):
        msg = (
            f"Node {identifier} is not a process node (type {type(node).__name__}). "
            "Only a process can be waited on."
        )
        raise WrongNodeType(msg)

    started = time.monotonic()
    while True:
        # Reloaded every poll: the daemon writes the state from another process,
        # and an already-loaded node can serve it from its own cache.
        node = t.cast("orm.ProcessNode", load_node(identifier))
        if node.is_terminated:
            waited = time.monotonic() - started
            logger.debug("pk=%s terminated after %.1fs", node.pk, waited)
            return _completion(node, terminated=True, waited=waited)

        waited = time.monotonic() - started
        if waited >= budget:
            logger.debug("pk=%s still running after %.1fs", node.pk, waited)
            return _completion(node, terminated=False, waited=waited)

        time.sleep(min(_POLL_SECONDS, budget - waited))
