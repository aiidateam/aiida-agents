"""Explain why a process failed, from what the plugin itself declares.

``get_process_status`` gives an exit code and ``get_process_report`` gives a
log, and between them a model was left to work out three things on its own:
which process in the tree actually failed, what its exit code means, and
whether the failure was one the workflow already knows how to fix. It guessed
at all three, because nothing here could tell it.

All three are recorded --- just not anywhere a tool looked.

**Which process failed.** A work chain's own exit code says a sub-process
failed, not why. The cause is a calculation somewhere below it, often two or
three levels down, and reading the top-level report to find it is exactly the
kind of tree-walking a model does badly. ``failure_chain`` walks it in code.

**What the exit code means.** Every process class declares its exit codes with
a message. That declaration is more diagnostic than the stored one when the
code is templated --- the run stores the formatted instance, the class carries
the general statement of the problem.

**What was already tried.** A restart work chain records every handler it
consulted, per iteration, in the ``considered_handlers`` extra: which ones
declined, which one fired, and what it decided. That is the difference between
"the calculation hit exit 410" and "the work chain recognised exit 410, applied
its remedy, restarted, and hit it again" --- which are the same failure with
opposite conclusions. A diagnosis that misses it recommends something already
attempted.

Nothing here knows anything about any particular code. The exit codes, the
handlers, and their descriptions all come from the installed plugin, so a
Quantum ESPRESSO convergence failure is explained by aiida-quantumespresso's
own handler docstrings, and a plugin nobody here has heard of is explained by
its own.

What this deliberately does *not* do is read the failed calculation's output
files. Those hold the code's own error text, and reaching them is
``list_retrieved_files`` / ``get_retrieved_file``; ``retrieved_files_pk`` names
the calculation to point them at.
"""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from aiida.engine import BaseRestartWorkChain, Process
from aiida.engine.processes.exit_code import ExitCode

from aiida_agents.tools._orm import WrongNodeType, load_node
from aiida_agents.tools._types import (
    FailedStep,
    FailureDiagnosis,
    HandlerAttempt,
    Identifier,
    RegisteredHandler,
)

logger = logging.getLogger(__name__)

__all__ = ["diagnose_process_failure"]

# Extra a BaseRestartWorkChain writes its per-iteration handler verdicts to.
# Read from the class rather than hardcoded, so a rename upstream follows.
_CONSIDERED_HANDLERS = BaseRestartWorkChain._considered_handlers_extra  # noqa: SLF001

# How deep the failure walk goes. Real nesting is three or four levels; this is
# only here so a cycle in the called graph cannot spin forever.
_MAX_DEPTH = 20

# Process states that mean "this did not complete", beyond a non-zero exit code.
_BROKEN_STATES = frozenset({"excepted", "killed"})


def _state_of(node: orm.ProcessNode) -> str | None:
    return node.process_state.value if node.process_state else None


def _has_failed(node: orm.ProcessNode) -> bool:
    """True if this process did not complete successfully.

    An exit status of 0 is success and ``None`` means it never got far enough
    to set one, so only a non-zero status counts --- plus the two states that
    fail without ever producing an exit code at all.
    """
    if _state_of(node) in _BROKEN_STATES:
        return True
    return bool(node.exit_status)


def _process_class_of(node: orm.ProcessNode) -> type[Process] | None:
    """The node's process class, or None if it cannot be imported.

    A node outlives the package that produced it: an archive imported from
    elsewhere routinely references plugins that are not installed here. That is
    a normal state of affairs, not an error, so it costs the class-derived
    fields and nothing else.
    """
    try:
        return node.process_class
    except Exception as exc:  # noqa: BLE001 - a missing/broken plugin, not our failure
        logger.debug("process class for pk=%s unavailable: %s", node.pk, exc)
        return None


def _exit_code_meaning(node: orm.ProcessNode) -> str | None:
    """What the process class declares this exit status to mean."""
    if node.exit_status is None:
        return None
    process_class = _process_class_of(node)
    if process_class is None:
        return None
    try:
        exit_codes = process_class.spec().exit_codes
    except Exception as exc:  # noqa: BLE001 - a broken spec is the plugin's problem
        logger.debug("spec for pk=%s unavailable: %s", node.pk, exc)
        return None
    for exit_code in exit_codes.values():
        if exit_code.status == node.exit_status:
            return t.cast("str | None", exit_code.message)
    return None


def _step_of(node: orm.ProcessNode) -> FailedStep:
    return FailedStep(
        pk=t.cast(int, node.pk),
        process_label=t.cast(str, node.process_label),
        node_type=type(node).__name__,
        state=_state_of(node),
        exit_status=node.exit_status,
        exit_message=node.exit_message,
        exit_code_meaning=_exit_code_meaning(node),
    )


def _failure_chain(node: orm.ProcessNode) -> list[orm.ProcessNode]:
    """Walk from ``node`` down to the deepest process that failed.

    At each level the *most recent* failed child is followed. That matters for
    a restart work chain, which leaves a trail of failed attempts behind it:
    the last one is the failure it gave up on, and the earlier ones are history
    that ``handling_attempted`` already accounts for.
    """
    chain = [node]
    seen = {node.pk}
    current = node

    for _ in range(_MAX_DEPTH):
        try:
            children = list(current.called)
        except Exception as exc:  # noqa: BLE001 - a broken link should not sink the walk
            logger.debug("children of pk=%s unavailable: %s", current.pk, exc)
            break
        failed = [c for c in children if c.pk not in seen and _has_failed(c)]
        if not failed:
            break
        current = max(failed, key=lambda n: n.ctime)
        seen.add(current.pk)
        chain.append(current)

    return chain


def _handler_exit_statuses(handler: t.Any) -> list[int] | None:
    """The exit statuses a handler declares it covers, if they can be read.

    ``process_handler`` validates ``exit_codes`` and then keeps it in its
    wrapper's closure; unlike ``priority`` and ``enabled`` it is never set on
    the function, so there is no supported way to ask a handler what it covers.
    Reading it out of the closure is the only route, and it is guarded to the
    same degree it is unofficial: any change to that decorator, or to the
    wrapper library under it, yields ``None`` --- "cannot tell, so assume it
    might apply" --- rather than an exception. The handler list is still
    returned in that case, only unfiltered.

    Cells are searched by shape rather than by position, so the answer does not
    depend on the order locals happen to be captured in.
    """
    wrapper = getattr(handler, "_self_wrapper", None)
    for cell in getattr(wrapper, "__closure__", None) or ():
        try:
            contents = cell.cell_contents
        except ValueError:  # an empty cell, not yet bound
            continue
        if (
            isinstance(contents, list)
            and contents
            and all(isinstance(item, ExitCode) for item in contents)
        ):
            return [item.status for item in contents]
    return None


def _description_of(handler: t.Any) -> str | None:
    """A handler's docstring, condensed to its first paragraph.

    That first paragraph is where a plugin states the remedy ("restart from the
    last charge density with a smaller mixing beta"); the rest is usually
    implementation detail that would crowd the diagnosis.
    """
    doc = getattr(handler, "__doc__", None)
    if not doc:
        return None
    paragraph = doc.strip().split("\n\n", 1)[0]
    return " ".join(paragraph.split())


def _restart_handlers(chain: t.Sequence[orm.ProcessNode]) -> list[t.Any]:
    """Every handler registered by a restart work chain in the failure chain.

    The handlers live on the work chain, but the failure they address belongs
    to the calculation *below* it, so the chain has to be searched rather than
    just the node the caller asked about.
    """
    handlers: list[t.Any] = []
    seen: set[str] = set()
    for node in chain:
        process_class = _process_class_of(node)
        if process_class is None or not issubclass(process_class, BaseRestartWorkChain):
            continue
        try:
            registered = process_class.get_process_handlers()
        except Exception as exc:  # noqa: BLE001 - a plugin's own introspection failing
            logger.debug("handlers for pk=%s unavailable: %s", node.pk, exc)
            continue
        for handler in registered:
            name = getattr(handler, "__name__", None)
            if name and name not in seen:
                seen.add(name)
                handlers.append(handler)
    return handlers


def _known_remedies(
    chain: t.Sequence[orm.ProcessNode], root_cause: orm.ProcessNode | None
) -> list[RegisteredHandler]:
    """Registered handlers, narrowed to the ones that cover the actual failure.

    A handler whose declared exit codes exclude the root cause's status is
    dropped: it is a remedy for a different problem, and offering it invites a
    fix that the work chain itself would never have applied. One whose codes
    cannot be determined is kept, since "might apply" is the safer error.
    """
    status = root_cause.exit_status if root_cause is not None else None
    remedies: list[RegisteredHandler] = []

    for handler in _restart_handlers(chain):
        statuses = _handler_exit_statuses(handler)
        if statuses is not None and status is not None and status not in statuses:
            continue
        remedies.append(
            RegisteredHandler(
                name=t.cast(str, getattr(handler, "__name__", "<unnamed>")),
                priority=getattr(handler, "priority", None),
                enabled=getattr(handler, "enabled", None),
                handles_exit_statuses=statuses,
                description=_description_of(handler),
            )
        )

    remedies.sort(key=lambda r: (-(r["priority"] or 0), r["name"]))
    return remedies


def _handling_attempted(
    chain: t.Sequence[orm.ProcessNode], handlers: t.Sequence[t.Any]
) -> list[HandlerAttempt]:
    """Flatten the ``considered_handlers`` extra into per-iteration verdicts.

    The extra is a list of iterations, each a list of ``[name, result]`` pairs;
    ``result`` is null when the handler declined and a mapping when it fired.
    Both are reported: a handler that declined has ruled its own remedy out,
    which is a finding rather than an absence of one.
    """
    described = {
        getattr(h, "__name__", None): _description_of(h)
        for h in handlers
        if getattr(h, "__name__", None)
    }
    attempts: list[HandlerAttempt] = []

    for node in chain:
        try:
            considered = node.base.extras.get(_CONSIDERED_HANDLERS, None)
        except Exception as exc:  # noqa: BLE001 - extras are not guaranteed readable
            logger.debug("extras for pk=%s unavailable: %s", node.pk, exc)
            continue
        if not isinstance(considered, list):
            continue

        for index, iteration in enumerate(considered, start=1):
            if not isinstance(iteration, list):
                continue
            for entry in iteration:
                # Stored as a two-element sequence; anything else is a shape
                # this version of aiida-core does not produce.
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    continue
                name, result = entry
                applied = isinstance(result, dict)
                attempts.append(
                    HandlerAttempt(
                        iteration=index,
                        handler=str(name),
                        applied=applied,
                        do_break=result.get("do_break") if applied else None,
                        exit_status=result.get("exit_status") if applied else None,
                        description=described.get(name),
                    )
                )

    return attempts


def _retrieved_files_pk(chain: t.Sequence[orm.ProcessNode]) -> int | None:
    """The deepest failed CalcJob, whose retrieved folder holds the code's own error."""
    for node in reversed(chain):
        if isinstance(node, orm.CalcJobNode):
            return t.cast(int, node.pk)
    return None


def diagnose_process_failure(identifier: Identifier) -> FailureDiagnosis:
    """Explain why a process failed: what broke, what it means, what was tried.

    Start here for any "why did this fail?" question, ahead of
    ``get_process_report``. A report is a log to read; this is the same failure
    already resolved into the calculation that actually broke, that exit code's
    declared meaning, and the remedies the work chain itself knows about ---
    including which of them it already attempted.

    Read the result in this order:

    1. ``root_cause`` --- the process that actually failed, which for a work
       chain is a calculation somewhere below the pk you asked about.
    2. ``handling_attempted`` --- what the restart machinery already tried. A
       remedy listed here as applied has been used; recommending it again
       repeats a failed attempt.
    3. ``known_remedies`` --- what the work chain could still do about this
       exit code, in the priority order it would consider them.
    4. ``retrieved_files_pk`` --- when the cause is a calculation, the pk to
       hand to ``list_retrieved_files`` / ``get_retrieved_file`` for the code's
       own error output. Do that whenever the declared meaning is generic:
       "the sub-process failed" names a symptom, and the real reason is in the
       output file.

    Every description here is written by the plugin that owns the workflow, not
    by this tool. Quote them rather than paraphrasing, and do not supplement
    them with a remedy of your own: a suggestion that no handler implements is
    one the work chain cannot carry out.

    Args:
        identifier: The process's pk or uuid. A top-level work chain is the
            usual thing to pass; a calculation works too.

    Returns:
        The diagnosis. ``failed`` is false for a process that succeeded or is
        still running, in which case there is nothing to explain --- though
        ``handling_attempted`` can still be non-empty, for a run that recovered
        from its own trouble. Say so when it is: a workflow that succeeded only
        after restarting three times is worth mentioning, not hiding behind
        "it worked".

    Raises:
        WrongNodeType: If the identifier names a data node rather than a process.
        NotExistent: If no node has that identifier.
    """
    logger.debug("diagnose_process_failure(identifier=%r)", identifier)
    node = load_node(identifier)
    if not isinstance(node, orm.ProcessNode):
        msg = (
            f"Node {identifier} is not a process node (type {type(node).__name__}). "
            "Use query_nodes() to explore data nodes."
        )
        raise WrongNodeType(msg)

    if not _has_failed(node):
        # A run that recovered is not a failure, but it is not the same as one
        # that never had trouble: "it succeeded" and "it succeeded once its
        # remedy had fired twice" answer different questions, and only the
        # handler trail distinguishes them. Report it. There is no failure to
        # chain, no root cause, and nothing left to remedy.
        return FailureDiagnosis(
            pk=t.cast(int, node.pk),
            process_label=t.cast(str, node.process_label),
            state=_state_of(node),
            failed=False,
            exit_status=node.exit_status,
            exit_message=node.exit_message,
            failure_chain=[],
            root_cause=None,
            handling_attempted=_handling_attempted([node], _restart_handlers([node])),
            known_remedies=[],
            retrieved_files_pk=None,
        )

    chain = _failure_chain(node)
    root_cause = chain[-1]
    handlers = _restart_handlers(chain)

    return FailureDiagnosis(
        pk=t.cast(int, node.pk),
        process_label=t.cast(str, node.process_label),
        state=_state_of(node),
        failed=True,
        exit_status=node.exit_status,
        exit_message=node.exit_message,
        failure_chain=[_step_of(step) for step in chain],
        root_cause=_step_of(root_cause),
        handling_attempted=_handling_attempted(chain, handlers),
        known_remedies=_known_remedies(chain, root_cause),
        retrieved_files_pk=_retrieved_files_pk(chain),
    )
