"""Pre-populate a workflow's inputs from its protocol builder, when it has one.

Many real-world workchains (mostly from ``aiida-quantumespresso`` and other
plugins following the AiiDA Common Workflows convention) expose a
``get_builder_from_protocol`` classmethod: given a structure (and usually a
code), it returns a ``ProcessBuilder`` already filled in with physically
sensible defaults for a named protocol ("fast", "moderate", "precise"), tuned
by people who have run the underlying simulations at scale. Building a
workchain's inputs from scratch -- which is all ``describe_process`` and
``submit_process_spec`` support -- throws that away and asks the model to
invent every physics parameter itself, for workchains whose input schema can
run into dozens of ports.

This tool calls the process's own protocol builder (introspecting its exact
keyword arguments rather than assuming ``structure=``/``code=``, since they
vary by workflow) and serialises the result back into the same
``{"pk"|"uuid"|"label"}``-reference / bare-value convention
``submit_process_spec`` already accepts -- so the agent's next step is
tweaking a handful of fields on the returned spec, not building it from zero.

Not every workflow has a protocol builder (most calculations and many older
workchains do not); ``describe_process``'s ``has_protocol_builder`` says so
up front, and this tool raises a clear, actionable error if called on one that
doesn't, pointing at ``draft_process_inputs`` --- which drafts the same spec
shape from the process's own ``Process.spec()`` instead.
"""

from __future__ import annotations

import inspect
import logging
import typing as t

from pydantic import Field

from aiida_agents.tools.execution._spec import (
    is_reference,
    load_process_class,
    resolve_reference,
    to_spec_value,
)
from aiida_agents.tools.execution.schemas import SubmissionSpec

logger = logging.getLogger(__name__)

__all__ = ["build_process_inputs"]

# Parameters every protocol builder implementation takes, so passing them
# through needs no per-workflow introspection.
_ALWAYS_ACCEPTED = frozenset({"protocol", "overrides"})


def _resolve_protocol_kwargs(kwargs: dict[str, t.Any]) -> dict[str, t.Any]:
    """Resolve any reference-shaped values, recursing into nested dicts.

    A multi-code workflow's protocol builder often takes a nested mapping
    (e.g. ``codes={"relax": {...}, "bands": {...}}``), so a reference is
    resolved wherever it appears, not only at the top level.
    """
    resolved: dict[str, t.Any] = {}
    for name, value in kwargs.items():
        if is_reference(value):
            resolved[name] = resolve_reference(value, name)
        elif isinstance(value, dict):
            resolved[name] = _resolve_protocol_kwargs(value)
        else:
            resolved[name] = value
    return resolved


def build_process_inputs(
    entry_point: t.Annotated[
        str,
        Field(
            description="The AiiDA process entry point, e.g. 'quantumespresso.pw.relax'."
        ),
    ],
    protocol: t.Annotated[
        str,
        Field(
            description="Named protocol to build from: typically 'fast', 'moderate', or 'precise'."
        ),
    ] = "fast",
    protocol_kwargs: t.Annotated[
        dict[str, t.Any] | None,
        Field(
            description=(
                "Keyword arguments for get_builder_from_protocol, beyond 'protocol' "
                "itself -- typically 'structure' and 'code' (or 'codes' for a "
                "multi-code workflow), whichever describe_process's "
                "'protocol_parameters' names as required. Node-valued arguments "
                "(structure, code, ...) are given as reference dicts: "
                '{"pk": N}, {"uuid": "..."}, or {"label": "name@computer"}. '
                "To change a physics parameter, pass it under 'overrides' (a "
                "nested mapping mirroring the workflow's input namespaces) rather "
                "than editing the returned inputs: the builder merges overrides "
                "into its own defaults, so dependent values stay consistent."
            )
        ),
    ] = None,
) -> SubmissionSpec:
    """Build a SubmissionSpec from a process's own protocol builder, if it has one.

    Calls ``entry_point``'s ``get_builder_from_protocol(protocol=..., **protocol_kwargs)``
    and serialises the result -- already populated with physically sensible
    defaults for the named protocol -- into the same spec shape
    ``submit_process_spec`` accepts. Use this instead of constructing
    ``inputs`` by hand whenever ``describe_process`` reports
    ``has_protocol_builder: true``: start from the returned spec and adjust
    only the handful of parameters that genuinely need it (per
    ``query_run_context``'s historical context, say), rather than
    inventing the full input tree.

    Args:
        entry_point: AiiDA entry point string.
        protocol: Named protocol, e.g. "fast", "moderate", "precise". Passed
            through only if the builder actually accepts it (nearly all do).
        protocol_kwargs: The builder's other keyword arguments -- see
            ``describe_process``'s ``protocol_parameters`` for exactly which
            ones this workflow needs and which are optional.

    Returns:
        A ``SubmissionSpec`` ready for ``submit_process_spec`` (or further
        adjustment first).

    Raises:
        ValueError: If the entry point is unknown, has no protocol builder,
            a required builder argument is missing, or the builder call itself
            fails (a bad reference, an invalid protocol name, ...).
    """
    logger.debug(
        "build_process_inputs(entry_point=%r, protocol=%r, protocol_kwargs=%r)",
        entry_point,
        protocol,
        protocol_kwargs,
    )
    process_class = load_process_class(entry_point)
    builder_from_protocol = getattr(process_class, "get_builder_from_protocol", None)
    if builder_from_protocol is None:
        msg = (
            f"{entry_point!r} has no get_builder_from_protocol -- call "
            f"draft_process_inputs({entry_point!r}) instead. It drafts the same "
            "spec shape from the process's own input ports, filling every default "
            "the spec declares and naming the required ports you still have to "
            "supply. Do not assemble the inputs by hand."
        )
        raise ValueError(msg)

    signature = inspect.signature(builder_from_protocol)
    resolved_kwargs = _resolve_protocol_kwargs(protocol_kwargs or {})

    call_kwargs: dict[str, t.Any] = {}
    if "protocol" in signature.parameters:
        call_kwargs["protocol"] = protocol
    for name, value in resolved_kwargs.items():
        if name not in signature.parameters and not _accepts_var_keyword(signature):
            msg = (
                f"{entry_point}.get_builder_from_protocol has no parameter "
                f"{name!r}. Its parameters are: "
                f"{_describe_parameters(signature)}."
            )
            raise ValueError(msg)
        call_kwargs[name] = value

    missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and name not in call_kwargs
        and name not in _ALWAYS_ACCEPTED
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if missing:
        msg = (
            f"{entry_point}.get_builder_from_protocol needs {missing} in "
            f"'protocol_kwargs', which {'was' if len(missing) == 1 else 'were'} not "
            f"provided. Its parameters are: {_describe_parameters(signature)}."
        )
        raise ValueError(msg)

    try:
        builder = builder_from_protocol(**call_kwargs)
    except Exception as exc:
        msg = f"{entry_point}.get_builder_from_protocol({call_kwargs!r}) failed: {exc}"
        raise ValueError(msg) from exc

    populated = to_spec_value(builder._inputs(prune=True))  # noqa: SLF001 - AiiDA's own public-in-practice accessor
    return {
        "entry_point": entry_point,
        "inputs": populated,
        "metadata": {"protocol": protocol, "source": "get_builder_from_protocol"},
    }


def _accepts_var_keyword(signature: inspect.Signature) -> bool:
    """True if the callable accepts arbitrary ``**kwargs``."""
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )


def _describe_parameters(signature: inspect.Signature) -> str:
    """A short, readable list of a signature's parameters and their defaults."""
    parts = []
    for name, param in signature.parameters.items():
        if name in ("cls", "self") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            parts.append(name)
        else:
            parts.append(f"{name}={param.default!r}")
    return ", ".join(parts)
