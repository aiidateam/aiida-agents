"""Shared reading of a process spec, and the ``SubmissionSpec`` value convention.

Two tools now build a :class:`~aiida_agents.tools.execution.schemas.SubmissionSpec`
--- ``build_process_inputs`` from a workflow's own protocol builder, and
``draft_process_inputs`` from its ``Process.spec()`` --- and both have to speak
the input convention ``submit_process_spec`` accepts: node-valued inputs as
``{"pk"|"uuid"|"label"}`` reference dicts, everything else as a bare value.

Keeping the conversions here means the two builders cannot drift apart on what a
spec looks like. If they did, a spec produced by one and adjusted through the
other would submit differently depending on which tool touched it last.
"""

from __future__ import annotations

import typing as t

from aiida import orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine import Process
from aiida.engine.processes.builder import ProcessBuilderNamespace
from aiida.orm.nodes.data.base import BaseType
from aiida.plugins.entry_point import load_entry_point

__all__ = [
    "is_reference",
    "load_process_class",
    "port_accepts_node",
    "resolve_reference",
    "to_spec_value",
    "valid_type_names",
]


def load_process_class(entry_point: str) -> type[Process]:
    """Load a process class by entry point, checking both process groups."""
    for group in ("aiida.workflows", "aiida.calculations"):
        try:
            return t.cast("type[Process]", load_entry_point(group, entry_point))
        except MissingEntryPointError:
            continue
    msg = f"Entry point not found: {entry_point!r}"
    raise ValueError(msg)


def is_reference(value: t.Any) -> bool:
    """True for a bare ``{"pk"|"uuid"|"label"}`` node-reference dict.

    Shape only. Whether the value *means* a reference also depends on where it
    sits, so pair this with :func:`port_accepts_node` wherever a port is known.
    """
    return isinstance(value, dict) and bool({"pk", "uuid", "label"} & value.keys())


def _port_valid_types(port: t.Any) -> tuple[type, ...]:
    """A port's ``valid_type`` as a tuple, with ``NoneType`` dropped.

    ``valid_type`` is a single type, a tuple of them, or ``None``; ``NoneType``
    appears for an optional port and says nothing about what to supply.
    """
    valid_type = getattr(port, "valid_type", None)
    if isinstance(valid_type, (tuple, list)):
        return tuple(vt for vt in valid_type if vt is not type(None))
    if valid_type is None or valid_type is type(None):
        return ()
    return (valid_type,)


def port_accepts_node(port: t.Any) -> bool:
    """Whether a node can go on this port, so a reference dict means something.

    The missing half of :func:`is_reference`: a ``{"pk"|"uuid"|"label"}`` dict
    names a node only where a node can be supplied. In a dict-*valued* port such
    as ``metadata.options.environment_variables`` a key called ``label`` is the
    user's own data, and reading it as a reference reports a missing *code* the
    user never mentioned. An unconstrained port counts as accepting one, since a
    reference is the only way to name a node there.
    """
    valid_types = _port_valid_types(port)
    if not valid_types:
        return True
    return any(isinstance(vt, type) and issubclass(vt, orm.Node) for vt in valid_types)


def resolve_reference(ref: dict[str, t.Any], context: str) -> orm.Node:
    """Resolve a ``{"pk"|"uuid"|"label"}`` reference to an existing node.

    Same reference convention as ``submit_process_spec``'s inputs, so a
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


def to_spec_value(value: t.Any) -> t.Any:
    """Serialise one AiiDA-side value back to the ``SubmissionSpec`` convention.

    Mirrors ``submit_process_spec``'s input convention exactly, so a
    serialised value plugs straight back in: ``Dict``/``List`` unwrap to their
    plain contents, primitive-backed nodes (``BaseType``: Int/Float/Str/Bool)
    to their bare ``.value``, and any other *stored* node (Code, StructureData,
    ...) to a ``{"pk": ...}`` reference. An *unstored* non-primitive node can
    also turn up (a protocol builder's freshly built ``KpointsData`` mesh with
    no pk yet, or a port default that constructs a node); ``submit_workflow``
    already accepts a pre-built AiiDA node as-is, so that is passed straight
    through rather than forced into a reference it cannot have yet.
    """
    if isinstance(value, ProcessBuilderNamespace):
        return {k: to_spec_value(v) for k, v in dict(value).items() if v is not None}
    if isinstance(value, dict):
        return {k: to_spec_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, orm.Dict):
        return value.get_dict()  # type: ignore[no-untyped-call]
    if isinstance(value, orm.List):
        return value.get_list()  # type: ignore[no-untyped-call]
    if isinstance(value, BaseType):
        return value.value
    if isinstance(value, orm.Node):
        return {"pk": value.pk} if value.is_stored else value
    return value


def valid_type_names(port: t.Any) -> list[str]:
    """The names of the node types a port accepts, ``["Any"]`` if unconstrained."""
    names = [
        vt.__name__ if hasattr(vt, "__name__") else str(vt)
        for vt in _port_valid_types(port)
    ]
    return names or ["Any"]
