"""Surface-agnostic channel-1 tools for introspecting AiiDA workflows and calculations.

These tools read directly from AiiDA's entry-point registry and `Process.spec()`,
enabling the agent to discover and inspect installed plugin workflows automatically
without vendoring static tables or plugin code into this repository.
"""

from __future__ import annotations

import inspect
import logging
import typing as t

from aiida.common.exceptions import MissingEntryPointError
from aiida.plugins.entry_point import get_entry_points, load_entry_point
from plumpy.ports import PortNamespace
from pydantic import Field

from aiida_agents.tools.execution._spec import valid_type_names

logger = logging.getLogger(__name__)

__all__ = ["describe_process", "list_process_entry_points"]


def list_process_entry_points(
    group: t.Annotated[
        str | None,
        Field(
            description="The entry-point group to list ('aiida.workflows' or 'aiida.calculations'). If None, lists both."
        ),
    ] = None,
) -> dict[str, t.Any]:
    """List available AiiDA process entry points registered in the environment.

    Queries AiiDA's entry-point registry for available workflows and/or calculations.
    When a plugin package (e.g., aiida-quantumespresso) is installed via pip, its
    workflows automatically appear here without any extra configuration.

    Args:
        group: Optional entry-point group name. If not provided, both
            'aiida.workflows' and 'aiida.calculations' are listed.

    Returns:
        Dictionary mapping entry-point groups to dictionaries of entry point name
        -> module path/class name.
    """
    logger.debug("list_process_entry_points(group=%r)", group)
    groups_to_check = (group,) if group else ("aiida.workflows", "aiida.calculations")
    results: dict[str, dict[str, str]] = {}
    total_count = 0

    for grp in groups_to_check:
        group_entries: dict[str, str] = {}
        try:
            for ep in get_entry_points(grp):
                group_entries[ep.name] = ep.value
        except Exception as exc:
            logger.debug("Could not inspect entry-point group %r: %s", grp, exc)
            continue
        results[grp] = group_entries
        total_count += len(group_entries)

    return {
        "query_type": "list_process_entry_points",
        "group": group,
        "total_entry_points": total_count,
        "entry_points": results,
    }


def _inspect_port_namespace(
    namespace: PortNamespace, prefix: str = ""
) -> dict[str, t.Any]:
    """Recursively inspect an AiiDA PortNamespace (`spec.inputs`)."""
    ports_info: dict[str, t.Any] = {}
    for name, port in namespace.items():
        full_name = f"{prefix}{name}" if prefix else name
        if isinstance(port, PortNamespace):
            ports_info[full_name] = {
                "type": "namespace",
                "required": getattr(port, "required", False),
                "ports": _inspect_port_namespace(port, prefix=f"{full_name}."),
            }
        else:
            vt_names = valid_type_names(port)

            has_def = port.has_default() if hasattr(port, "has_default") else False
            def_val = None
            if has_def and hasattr(port, "default"):
                try:
                    raw_def = port.default() if callable(port.default) else port.default
                    def_val = repr(raw_def)
                except Exception:
                    def_val = "<default available>"

            ports_info[full_name] = {
                "type": "port",
                "required": getattr(port, "required", False),
                "valid_types": vt_names,
                "has_default": has_def,
                "default": def_val if has_def else None,
                "help": getattr(port, "help", None),
            }
    return ports_info


def _collect_flat_ports(
    namespace: dict[str, t.Any], prefix: str = ""
) -> tuple[list[str], list[str]]:
    """Helper to collect flat lists of required vs optional input port names."""
    required: list[str] = []
    optional: list[str] = []
    for name, info in namespace.items():
        if info["type"] == "namespace":
            sub_req, sub_opt = _collect_flat_ports(info["ports"])
            required.extend(sub_req)
            optional.extend(sub_opt)
        else:
            if info["required"] and not info["has_default"]:
                required.append(name)
            else:
                optional.append(name)
    return required, optional


def describe_process(
    entry_point: t.Annotated[
        str,
        Field(
            description="The AiiDA process entry point to describe, e.g. 'core.arithmetic.multiply_add' or 'quantumespresso.pw.relax'."
        ),
    ],
) -> dict[str, t.Any]:
    """Introspect `Process.spec()` for a specific workflow or calculation entry point.

    Loads the process class and inspects its schema, returning required/optional ports,
    accepted node types, default values, exit codes, and whether a protocol builder
    (`get_builder_from_protocol`) is available.

    Args:
        entry_point: Full or short entry point string, e.g. 'core.arithmetic.multiply_add'.

    Returns:
        Dictionary detailing the process class schema and parameters.
    """
    logger.debug("describe_process(entry_point=%r)", entry_point)
    if not entry_point or not isinstance(entry_point, str):
        msg = "entry_point must be a non-empty string."
        raise ValueError(msg)

    process_class = None
    resolved_group = None
    for group in ("aiida.workflows", "aiida.calculations"):
        try:
            process_class = load_entry_point(group, entry_point)
            resolved_group = group
            break
        except MissingEntryPointError:
            continue
        except Exception as exc:
            msg = (
                f"Entry point {entry_point!r} in group {group!r} failed to load: {exc}"
            )
            raise ValueError(msg) from exc

    if process_class is None:
        msg = (
            f"Entry point {entry_point!r} not found in 'aiida.workflows' "
            f"or 'aiida.calculations'."
        )
        raise ValueError(msg)

    spec = process_class.spec()
    inputs_schema = _inspect_port_namespace(spec.inputs)
    required_inputs, optional_inputs = _collect_flat_ports(inputs_schema)

    outputs_list = [name for name in spec.outputs.keys()]
    exit_codes_list = [
        {"status": ec.status, "message": ec.message} for ec in spec.exit_codes.values()
    ]

    builder_from_protocol = getattr(process_class, "get_builder_from_protocol", None)

    return {
        "query_type": "describe_process",
        "entry_point": entry_point,
        "group": resolved_group,
        "process_class": f"{process_class.__module__}:{process_class.__name__}",
        "has_protocol_builder": builder_from_protocol is not None,
        "protocol_parameters": _describe_protocol_parameters(builder_from_protocol),
        "required_inputs": sorted(required_inputs),
        "optional_inputs": sorted(optional_inputs),
        "inputs_schema": inputs_schema,
        "outputs": sorted(outputs_list),
        "exit_codes": sorted(exit_codes_list, key=lambda x: x["status"]),
    }


def _describe_protocol_parameters(
    builder_from_protocol: t.Callable[..., t.Any] | None,
) -> list[dict[str, t.Any]] | None:
    """The exact keyword arguments ``get_builder_from_protocol`` takes, if any.

    Signatures vary by workflow -- some need only ``structure``, others also
    ``code`` or a ``codes`` mapping for a multi-code workflow -- so this is read
    from the callable itself rather than assumed, telling
    ``build_process_inputs``'s caller exactly what to put in its
    ``protocol_kwargs``.
    """
    if builder_from_protocol is None:
        return None
    parameters = []
    for name, param in inspect.signature(builder_from_protocol).parameters.items():
        if name in ("cls", "self") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        has_default = param.default is not inspect.Parameter.empty
        parameters.append(
            {
                "name": name,
                "required": not has_default,
                "default": repr(param.default) if has_default else None,
            }
        )
    return parameters
