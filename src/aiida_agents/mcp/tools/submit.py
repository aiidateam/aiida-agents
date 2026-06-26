"""MCP tool for submitting AiiDA workflows."""

from __future__ import annotations

import logging
from typing import Any, cast

from aiida import orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine import Process, run_get_node, submit
from aiida.manage import get_manager
from aiida.plugins.entry_point import load_entry_point
from fastmcp.exceptions import ToolError

from aiida_agents.mcp._types import SubmitResult


logger = logging.getLogger(__name__)

# Node types a bare Python value may be wrapped into. A value is wrapped only
# into a type it can faithfully represent: ``int`` widens to ``Float`` (an int
# is a valid float), but ``float`` does not narrow to ``Int`` -- a float for an
# Int-only port is left raw so ``spec.inputs.validate()`` rejects it instead of
# silently truncating it into ``Int(2)``.
_COMPATIBLE_NODES: dict[type, tuple[type, ...]] = {
    int: (orm.Int, orm.Float),
    float: (orm.Float,),
    str: (orm.Str,),
    bool: (orm.Bool,),
    list: (orm.List,),
    dict: (orm.Dict,),
}


def _load_process_class(entry_point: str) -> type[Process]:
    """Load an AiiDA process class from its entry point string.

    Tries the calculation group, then the workflow group. Only a genuinely
    *missing* entry point falls through to the next group; a registered entry
    point that fails to import (a broken plugin) raises its own error rather
    than being masked as "not found".
    """
    for group in ("aiida.calculations", "aiida.workflows"):
        try:
            return cast("type[Process]", load_entry_point(group, entry_point))
        except MissingEntryPointError:
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

    Only top-level ports are resolved. A nested input namespace (e.g.
    ``pw.parameters`` on a real workchain) is passed through unchanged, so this
    handles flat-input processes (the arithmetic add / multiply_add demos) but
    not workflows whose inputs live under nested namespaces.

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

        # Auto-wrap a bare primitive in the first port-accepted node type the
        # value can faithfully represent (see _COMPATIBLE_NODES). A value
        # incompatible with every accepted type is left raw, so the spec
        # validator reports the type error rather than the wrap silently
        # coercing it. Nodes are intentionally NOT stored here -- storage
        # happens during submit()/run_get_node() so that validation failures
        # leave no orphaned nodes in the database (ADR-07).
        compatible_nodes = _COMPATIBLE_NODES.get(type(value), ())
        for expected in expected_types:
            if isinstance(expected, type) and any(
                issubclass(node, expected) for node in compatible_nodes
            ):
                resolved[name] = expected(value)
                break
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


class SubmissionInputError(Exception):
    """Inputs could not be resolved or failed the process spec.

    Carries a model-facing message and is raised *before* anything is written
    to the database, so the caller can route it back to the agent as a
    correctable error (or show it to the user) without leaving orphan nodes.
    """


def _prepare_submission(
    entry_point: str, inputs: dict[str, Any]
) -> tuple[type[Process], dict[str, Any]]:
    """Resolve JSON inputs to unstored nodes and validate against the spec.

    Resolution turns the agent's JSON (bare values, reference dicts) into AiiDA
    nodes; validation is delegated to AiiDA's own ``spec().inputs.validate()``,
    which runs the full required/type/nested-namespace check pre-submit and
    writes nothing. This is the single seam every submission passes through, and
    the place to repoint at aiida-restapi once it grows a write endpoint
    (ADR-02, docs/adr/02-mcp-tools-wrap-aiida-restapi.md).

    Args:
        entry_point: AiiDA entry point string, e.g. ``"core.arithmetic.add"``.
        inputs: Port names mapped to bare values, reference dicts, or nodes.

    Returns:
        The loaded process class and the resolved, *unstored* inputs.

    Raises:
        SubmissionInputError: If resolution fails or the spec rejects the
            inputs. The message is safe to show the agent or the user.
    """
    try:
        resolved = _resolve_inputs(entry_point, inputs)
    except ToolError as exc:
        raise SubmissionInputError(str(exc)) from exc
    except Exception as exc:
        raise SubmissionInputError(f"Could not build inputs: {exc}") from exc

    process_class = _load_process_class(entry_point)
    error = process_class.spec().inputs.validate(resolved)
    if error is not None:
        raise SubmissionInputError(str(error))
    return process_class, resolved


def submit_workflow(entry_point: str, inputs: dict[str, Any]) -> SubmitResult:
    """Submit an AiiDA workflow or calculation.

    Resolves user-supplied values to AiiDA nodes automatically using the
    process port spec. Three input conventions are supported:

    * **Bare primitive** — ``{"x": 2}`` always means the *value* 2 and wraps
      it in ``orm.Int(2)``. It is never treated as a node PK.
    * **Reference dict** — pass ``{"pk": N}``, ``{"uuid": "..."}``, or
      ``{"label": "bash@localhost"}`` to reuse an existing node.
    * **AiiDA node** — passed through unchanged.

    Inputs are resolved and validated against the process spec before
    submission; nothing is written to the database unless they pass. The
    caller (CLI) must obtain user confirmation (HITL) before calling this tool.
    Only top-level inputs are resolved (see ``_resolve_inputs``).

    With a process-control broker the workflow is submitted and this returns
    immediately, with ``state`` the initial state. On a brokerless profile
    (e.g. ``core.sqlite_dos``) it runs in-process and *blocks* until the run
    finishes, with ``state`` the final state.

    Args:
        entry_point: AiiDA entry point string, e.g. ``"core.arithmetic.add"``.
        inputs: Dict mapping port names to values or reference dicts.

    Returns:
        A ``SubmitResult`` with the new process PK, UUID, and initial state.

    Raises:
        ToolError: If inputs fail to resolve/validate or submission fails.
    """
    try:
        process_class, resolved = _prepare_submission(entry_point, inputs)
    except SubmissionInputError as exc:
        raise ToolError(str(exc)) from exc

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
