"""MCP tool for submitting AiiDA workflows."""

from __future__ import annotations

import logging
from typing import Any, cast

from aiida import orm
from aiida.engine import submit, run_get_node
from aiida.manage import get_manager
from aiida.plugins.entry_point import load_entry_point
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from aiida_agents.mcp._types import SubmitResult


logger = logging.getLogger(__name__)

# Mapping from Python primitives to AiiDA data node types
_PRIMITIVE_TO_NODE: dict[type, type] = {
    int: orm.Int,
    float: orm.Float,
    str: orm.Str,
    bool: orm.Bool,
    list: orm.List,
    dict: orm.Dict,
}


def _load_process_class(entry_point: str) -> Any:
    """Load an AiiDA process class from its entry point string."""
    for group in ("aiida.calculations", "aiida.workflows"):
        try:
            return load_entry_point(group, entry_point)
        except Exception:
            continue
    msg = f"Entry point not found: {entry_point!r}"
    raise ToolError(msg)


def _resolve_node_reference(ref: dict[str, Any], port_name: str) -> orm.Node:
    """Resolve an explicit node reference dict to an AiiDA node.

    Supported reference forms:
        {"pk": 42}                  — load by PK
        {"uuid": "abc-..."}         — load by UUID
        {"label": "bash@localhost"} — load Code by label (Code ports only)

    Args:
        ref: The reference dict from the user.
        port_name: Port name, used only for error messages.

    Returns:
        The loaded AiiDA node.

    Raises:
        ToolError: If the reference form is unrecognised or the node is not found.
    """
    if "pk" in ref:
        try:
            return orm.load_node(ref["pk"])
        except Exception as exc:
            raise ToolError(
                f"No node found with pk={ref['pk']!r} for input {port_name!r}"
            ) from exc

    if "uuid" in ref:
        try:
            return orm.load_node(ref["uuid"])
        except Exception as exc:
            raise ToolError(
                f"No node found with uuid={ref['uuid']!r} for input {port_name!r}"
            ) from exc

    if "label" in ref:
        try:
            return orm.load_code(ref["label"])
        except Exception as exc:
            raise ToolError(
                f"No Code found with label={ref['label']!r} for input {port_name!r}. "
                f"Use the format 'name@computer', e.g. 'bash@localhost'."
            ) from exc

    raise ToolError(
        f"Unrecognised node reference for input {port_name!r}: {ref!r}. "
        f'Use one of: {{"pk": N}}, {{"uuid": "..."}}, {{"label": "name@computer"}}.'
    )


def _is_reference_type(expected_types: tuple[type, ...]) -> bool:
    """Return True if the port expects a node type that cannot be created from a bare primitive.

    Ports expecting types like Code, AbstractCode, or StructureData require an
    explicit node reference (pk/uuid/label) rather than a plain Python value.
    """
    _REFERENCE_ONLY_TYPES = (orm.AbstractCode,)
    return any(
        issubclass(t, _REFERENCE_ONLY_TYPES)
        for t in expected_types
        if isinstance(t, type)
    )


def _resolve_inputs(entry_point: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Resolve user-supplied values to AiiDA nodes using the process port spec.

    Input value conventions
    -----------------------
    Bare primitive (``{"x": 2}``)
        Always treated as the *value* 2 — wraps in ``orm.Int(2)``.
        Never interpreted as a node PK.

    Reference dict (``{"code": {"pk": 42}}``)
        Loads an existing node explicitly. Three forms are supported:

        * ``{"pk": N}``                  — load by integer PK
        * ``{"uuid": "abc-..."}``        — load by UUID string
        * ``{"label": "bash@localhost"}``— load Code by ``name@computer`` label

    Already an AiiDA node
        Passed through as-is.

    Reference-only ports (e.g. ``code``)
        Ports whose ``valid_type`` is ``AbstractCode`` or similar non-wrappable
        types *require* an explicit reference dict. Passing a bare primitive
        raises a clear ``ToolError``.

    Args:
        entry_point: AiiDA entry point string.
        inputs: Dict mapping port names to values or reference dicts.

    Returns:
        Dict mapping port names to resolved, *unstored* AiiDA nodes.
        Nodes are stored by AiiDA automatically during ``submit()`` /
        ``run_get_node()``.

    Raises:
        ToolError: If a reference cannot be resolved or a reference-only port
            receives a bare primitive.
    """
    process_class = _load_process_class(entry_point)
    spec = process_class.spec()
    resolved: dict[str, Any] = {}

    for name, value in inputs.items():
        # Already an AiiDA node — use as-is
        if isinstance(value, orm.Node):
            resolved[name] = value
            continue

        # Explicit node reference dict — resolve and use directly
        if isinstance(value, dict) and {"pk", "uuid", "label"} & value.keys():
            resolved[name] = _resolve_node_reference(value, name)
            continue

        # Get the expected type from the port spec
        port = spec.inputs.get(name)
        valid_type = getattr(port, "valid_type", None) if port else None

        # Normalise valid_type to a tuple, stripping NoneType
        if valid_type is None:
            expected_types: tuple[type, ...] = ()
        elif isinstance(valid_type, tuple):
            expected_types = tuple(t for t in valid_type if t is not type(None))
        else:
            expected_types = (valid_type,) if valid_type is not type(None) else ()

        # Reference-only ports require an explicit reference dict
        if expected_types and _is_reference_type(expected_types):
            raise ToolError(
                f"Input {name!r} expects a node reference, not a plain value. "
                f'Use one of: {{"pk": N}}, {{"uuid": "..."}}, '
                f'{{"label": "name@computer"}}.'
            )

        # Auto-wrap primitive in the first matching expected AiiDA type.
        # Nodes are intentionally NOT stored here — storage happens during
        # submit()/run_get_node() so that validation failures leave no
        # orphaned nodes in the database (ADR-07).
        python_type = type(value)
        if python_type in _PRIMITIVE_TO_NODE and expected_types:
            for expected in expected_types:
                if isinstance(expected, type) and issubclass(expected, orm.Data):
                    resolved[name] = expected(value)
                    break
            else:
                # Fallback: use the generic primitive → node mapping
                node_class = _PRIMITIVE_TO_NODE.get(python_type)
                resolved[name] = node_class(value) if node_class else value
        else:
            resolved[name] = value

    return resolved


def _format_resolved_inputs(resolved: dict[str, Any]) -> str:
    """Format resolved AiiDA nodes for human-readable display in the HITL prompt.

    For each resolved input, shows:
    - Stored nodes: their type, PK, and value (if available)
    - Unstored nodes: their type and value

    Args:
        resolved: Dict of port names to resolved AiiDA nodes or plain values.

    Returns:
        A formatted multi-line string for display.
    """
    lines = []
    for name, node in resolved.items():
        if isinstance(node, orm.Node):
            node_type = type(node).__name__
            if node.is_stored:
                # Existing node loaded by pk/uuid/label
                value = node.value if hasattr(node, "value") else repr(node)
                lines.append(f"   {name}: {node_type}(pk={node.pk}, value={value!r})")
            else:
                # Newly wrapped primitive — not yet in DB
                value = node.value if hasattr(node, "value") else repr(node)
                lines.append(f"   {name}: {node_type}(value={value!r})  [new]")
        else:
            lines.append(f"   {name}: {node!r}")
    return "\n".join(lines)


def submit_workflow(entry_point: str, inputs: dict[str, Any]) -> SubmitResult:
    """Submit an AiiDA workflow or calculation.

    Resolves user-supplied values to AiiDA nodes automatically using the
    process port spec. Three input conventions are supported:

    * **Bare primitive** — ``{"x": 2}`` always means the *value* 2 and wraps
      it in ``orm.Int(2)``. It is never treated as a node PK.
    * **Reference dict** — pass ``{"pk": N}``, ``{"uuid": "..."}``, or
      ``{"label": "bash@localhost"}`` to reuse an existing node.
    * **AiiDA node** — passed through unchanged.

    Validates all inputs before submission. No nodes are written to the
    database until validation passes (ADR-07). The caller (CLI) must obtain
    user confirmation (HITL) before calling this tool.

    Args:
        entry_point: AiiDA entry point string, e.g. ``"core.arithmetic.add"``.
        inputs: Dict mapping port names to values or reference dicts.

    Returns:
        A ``SubmitResult`` with the new process PK, UUID, and initial state.

    Raises:
        ToolError: If validation fails or the entry point is not found.
    """
    # Lazy import — breaks the circular dependency:
    # analysis -> mcp.tools.submit -> agents.validator -> agents.__init__ -> analysis
    from aiida_agents.agents.validator import ValidationError, validate

    try:
        resolved = _resolve_inputs(entry_point, inputs)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to resolve inputs: {exc}") from exc

    try:
        validate(entry_point, resolved)
    except ValidationError as exc:
        raise ToolError(
            f"Validation failed for {entry_point!r}:\n" + "\n".join(exc.errors)
        ) from exc

    process_class = _load_process_class(entry_point)

    try:
        manager = get_manager()
        profile = manager.get_profile()
        if profile and profile.process_control_backend:
            node = submit(process_class, **resolved)
        else:
            # Brokerless profile (sqlite_dos) — run in-process
            _, node = run_get_node(process_class, **resolved)
    except Exception as exc:
        raise ToolError(f"Submission failed: {exc}") from exc

    logger.info("submitted %s → pk=%d", entry_point, node.pk)

    return {
        "pk": cast(int, node.pk),
        "uuid": node.uuid,
        "workflow": entry_point,
        "state": node.process_state.value if node.process_state else "created",
    }


def register(mcp: FastMCP) -> None:
    """Register submit tools on the MCP server."""
    mcp.tool()(submit_workflow)
