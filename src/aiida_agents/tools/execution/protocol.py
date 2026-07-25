"""Pre-populate a workflow's inputs from its protocol builder, when it has one.

Many real-world workchains (mostly from ``aiida-quantumespresso`` and other
plugins following the AiiDA Common Workflows convention) expose a
``get_builder_from_protocol`` classmethod: given a structure (and usually a
code), it returns a ``ProcessBuilder`` already filled in with physically
sensible defaults for a named protocol ("fast", "moderate", "precise"), tuned
by people who have run the underlying simulations at scale. Building a
workchain's inputs from scratch -- which is all ``describe_workflow`` and
``execute_workflow_spec`` support -- throws that away and asks the model to
invent every physics parameter itself, for workchains whose input schema can
run into dozens of ports.

This tool calls the process's own protocol builder (introspecting its exact
keyword arguments rather than assuming ``structure=``/``code=``, since they
vary by workflow) and serialises the result back into the same
``{"pk"|"uuid"|"label"}``-reference / bare-value convention
``execute_workflow_spec`` already accepts -- so the agent's next step is
tweaking a handful of fields on the returned spec, not building it from zero.

Not every workflow has a protocol builder (most calculations and many older
workchains do not); ``describe_workflow``'s ``has_protocol_builder`` says so
up front, and this tool raises a clear, actionable error if called on one that
doesn't, telling the agent to build the spec directly instead.
"""

from __future__ import annotations

import inspect
import logging
import typing as t

from aiida import orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine.processes.builder import ProcessBuilderNamespace
from aiida.orm.nodes.data.base import BaseType
from aiida.plugins.entry_point import load_entry_point
from pydantic import Field

from aiida_agents.tools.execution.schemas import WorkflowSpec

logger = logging.getLogger(__name__)

__all__ = ["build_workflow_inputs"]

# Parameters every protocol builder implementation takes, so passing them
# through needs no per-workflow introspection.
_ALWAYS_ACCEPTED = frozenset({"protocol", "overrides"})


def _load_process_class(entry_point: str) -> type:
    """Load a process class by entry point, checking both process groups."""
    for group in ("aiida.workflows", "aiida.calculations"):
        try:
            return t.cast("type", load_entry_point(group, entry_point))
        except MissingEntryPointError:
            continue
    msg = f"Entry point not found: {entry_point!r}"
    raise ValueError(msg)


def _is_reference(value: t.Any) -> bool:
    """True for a bare ``{"pk"|"uuid"|"label"}`` node-reference dict."""
    return isinstance(value, dict) and bool({"pk", "uuid", "label"} & value.keys())


def _resolve_reference(ref: dict[str, t.Any], context: str) -> orm.Node:
    """Resolve a ``{"pk"|"uuid"|"label"}`` reference to an existing node.

    Same reference convention as ``execute_workflow_spec``'s inputs, so a
    structure or code the model already has a handle on plugs in unchanged.
    """
    if "pk" in ref:
        try:
            return orm.load_node(ref["pk"])
        except Exception as exc:
            raise ValueError(
                f"No node found with pk={ref['pk']} for {context!r}."
            ) from exc
    if "uuid" in ref:
        try:
            return orm.load_node(ref["uuid"])
        except Exception as exc:
            raise ValueError(
                f"No node found with uuid={ref['uuid']!r} for {context!r}."
            ) from exc
    if "label" in ref:
        try:
            return orm.load_code(ref["label"])
        except Exception as exc:
            raise ValueError(
                f"No Code found with label={ref['label']!r} for {context!r}."
            ) from exc
    msg = f"Reference {ref!r} for {context!r} must contain 'pk', 'uuid', or 'label'."
    raise ValueError(msg)


def _resolve_protocol_kwargs(kwargs: dict[str, t.Any]) -> dict[str, t.Any]:
    """Resolve any reference-shaped values, recursing into nested dicts.

    A multi-code workflow's protocol builder often takes a nested mapping
    (e.g. ``codes={"relax": {...}, "bands": {...}}``), so a reference is
    resolved wherever it appears, not only at the top level.
    """
    resolved: dict[str, t.Any] = {}
    for name, value in kwargs.items():
        if _is_reference(value):
            resolved[name] = _resolve_reference(value, name)
        elif isinstance(value, dict):
            resolved[name] = _resolve_protocol_kwargs(value)
        else:
            resolved[name] = value
    return resolved


def _unwrap(value: t.Any) -> t.Any:
    """Serialise one populated builder value back to the WorkflowSpec convention.

    Mirrors ``execute_workflow_spec``'s input convention exactly, so the
    returned spec plugs straight back in: ``Dict``/``List`` unwrap to their
    plain contents, primitive-backed nodes (``BaseType``: Int/Float/Str/Bool)
    to their bare ``.value``, and any other *stored* node (Code, StructureData,
    ...) to a ``{"pk": ...}`` reference. A protocol builder can also leave an
    *unstored* non-primitive node in place (e.g. a freshly built ``KpointsData``
    mesh with no pk yet); ``submit_workflow`` already accepts a pre-built AiiDA
    node as-is, so that is passed straight through rather than forced into a
    reference it cannot have yet.
    """
    if isinstance(value, ProcessBuilderNamespace):
        return {k: _unwrap(v) for k, v in dict(value).items() if v is not None}
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items() if v is not None}
    if isinstance(value, orm.Dict):
        return value.get_dict()  # type: ignore[no-untyped-call]
    if isinstance(value, orm.List):
        return value.get_list()  # type: ignore[no-untyped-call]
    if isinstance(value, BaseType):
        return value.value
    if isinstance(value, orm.Node):
        return {"pk": value.pk} if value.is_stored else value
    return value


def build_workflow_inputs(
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
                "multi-code workflow), whichever describe_workflow's "
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
) -> WorkflowSpec:
    """Build a WorkflowSpec from a process's own protocol builder, if it has one.

    Calls ``entry_point``'s ``get_builder_from_protocol(protocol=..., **protocol_kwargs)``
    and serialises the result -- already populated with physically sensible
    defaults for the named protocol -- into the same spec shape
    ``execute_workflow_spec`` accepts. Use this instead of constructing
    ``inputs`` by hand whenever ``describe_workflow`` reports
    ``has_protocol_builder: true``: start from the returned spec and adjust
    only the handful of parameters that genuinely need it (per
    ``query_analysis_agent``'s historical context, say), rather than
    inventing the full input tree.

    Args:
        entry_point: AiiDA entry point string.
        protocol: Named protocol, e.g. "fast", "moderate", "precise". Passed
            through only if the builder actually accepts it (nearly all do).
        protocol_kwargs: The builder's other keyword arguments -- see
            ``describe_workflow``'s ``protocol_parameters`` for exactly which
            ones this workflow needs and which are optional.

    Returns:
        A ``WorkflowSpec`` ready for ``execute_workflow_spec`` (or further
        adjustment first).

    Raises:
        ValueError: If the entry point is unknown, has no protocol builder,
            a required builder argument is missing, or the builder call itself
            fails (a bad reference, an invalid protocol name, ...).
    """
    logger.debug(
        "build_workflow_inputs(entry_point=%r, protocol=%r, protocol_kwargs=%r)",
        entry_point,
        protocol,
        protocol_kwargs,
    )
    process_class = _load_process_class(entry_point)
    builder_from_protocol = getattr(process_class, "get_builder_from_protocol", None)
    if builder_from_protocol is None:
        msg = (
            f"{entry_point!r} has no get_builder_from_protocol -- build its inputs "
            "directly instead (see describe_workflow's required_inputs/optional_inputs)."
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

    populated = _unwrap(builder._inputs(prune=True))  # noqa: SLF001 - AiiDA's own public-in-practice accessor
    return {
        "workflow_type": entry_point,
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
