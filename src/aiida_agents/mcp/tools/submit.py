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

from aiida_agents.agents.validator import ValidationError, validate
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


def _resolve_inputs(entry_point: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Resolve Python primitives to AiiDA nodes using the process port spec.

    For each input:
    - If it's already an AiiDA node, use it as-is.
    - If it's a primitive (int, float, str, bool, list, dict):
      - Check the port's valid_type from the spec.
      - If the primitive is an int and a node with that PK exists and matches
        the expected type, reuse that node (explicit PK reference).
      - Otherwise, wrap the primitive in the expected AiiDA node type.
    """
    process_class = _load_process_class(entry_point)
    spec = process_class.spec()
    resolved: dict[str, Any] = {}

    for name, value in inputs.items():
        # Already an AiiDA node — use as-is
        if isinstance(value, orm.Node):
            resolved[name] = value
            continue

        # Get the expected type from the port spec
        port = spec.inputs.get(name)
        valid_type = getattr(port, "valid_type", None) if port else None

        # Normalise valid_type to a tuple
        if valid_type is None:
            expected_types: tuple[type, ...] = ()
        elif isinstance(valid_type, tuple):
            expected_types = valid_type
        else:
            expected_types = (valid_type,)

        # Filter out NoneType
        expected_types = tuple(t for t in expected_types if t is not type(None))

        # For integers: try PK lookup if the expected type matches
        if isinstance(value, int) and expected_types:
            try:
                node = orm.load_node(value)
                if isinstance(node, expected_types):
                    resolved[name] = node
                    continue
            except Exception:
                pass

        # Auto-wrap primitive in the first matching expected AiiDA type
        python_type = type(value)
        if python_type in _PRIMITIVE_TO_NODE and expected_types:
            for expected in expected_types:
                if issubclass(expected, orm.Data):
                    resolved[name] = expected(value).store()
                    break
            else:
                # Fallback: wrap using the generic primitive mapping
                node_class = _PRIMITIVE_TO_NODE.get(python_type)
                if node_class:
                    resolved[name] = node_class(value).store()
                else:
                    resolved[name] = value
        else:
            resolved[name] = value

    return resolved


def submit_workflow(entry_point: str, inputs: dict[str, Any]) -> SubmitResult:
    """Submit an AiiDA workflow or calculation.

    Resolves Python primitives to AiiDA nodes automatically using the process
    port spec — pass ``{"x": 2, "y": 3}`` and the tool wraps them in
    ``orm.Int`` as required. Integer values that match an existing node PK of
    the correct type are reused rather than duplicated.

    Validates all inputs before submission. The caller (CLI) must obtain user
    confirmation (HITL) before calling this tool.

    Args:
        entry_point: AiiDA entry point string, e.g. ``"core.arithmetic.add"``.
        inputs: Dict mapping port names to values. Primitives are wrapped
            automatically; AiiDA nodes are used as-is.

    Returns:
        A ``SubmitResult`` with the new process PK, UUID, and initial state.

    Raises:
        ToolError: If validation fails or the entry point is not found.
    """
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
            # brokerless profile (sqlite_dos) — run in-process
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
