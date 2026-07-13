"""Tool for retrieving workflow templates and parameter definitions.

Allows the agent to dynamically introspect available workflow definitions,
required/optional inputs, parameter bounds, and example successful runs.
"""

from __future__ import annotations

import logging
import typing as t

from pydantic import Field

from aiida_agents.tools.workflows.schemas import KNOWN_WORKFLOWS, PARAMETER_SCHEMA

logger = logging.getLogger(__name__)

__all__ = ["get_workflow_templates"]


def _sanitize_for_json(obj: t.Any) -> t.Any:
    """Recursively convert non-serializable objects (like class 'type') to strings."""
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_json(x) for x in obj)
    return obj


def get_workflow_templates(
    workflow_type: t.Annotated[
        str | None,
        Field(
            description="Optional full entry point string (e.g. 'aiida.workflows:PwRelaxWorkChain' or 'aiida.workflows:VaspRelaxWorkChain'). If None, returns all known workflow templates."
        ),
    ] = None,
) -> dict[str, t.Any]:
    """Get workflow templates, schema definitions, parameter bounds, and usage examples.

    This tool lets the Execution Agent check exact input requirements, parameter validation
    ranges (`min`, `max`, `typical`), and example structures before or during specification generation.

    Args:
        workflow_type: Optional specific workflow to inspect. If omitted, lists all available workflows.

    Returns:
        Dictionary detailing the requested workflow schema or all known workflow templates.
    """
    logger.debug("get_workflow_templates(workflow_type=%r)", workflow_type)

    if workflow_type:
        if workflow_type not in KNOWN_WORKFLOWS:
            available = list(KNOWN_WORKFLOWS.keys())
            return {
                "error": f"Unknown workflow_type {workflow_type!r}",
                "available_workflows": available,
                "suggestion": f"Choose one of: {', '.join(available)}",
            }

        info = KNOWN_WORKFLOWS[workflow_type].copy()

        # Include relevant parameters schema
        relevant_params: dict[str, t.Any] = {}
        if "Vasp" in workflow_type:
            for k in ["ENCUT", "EDIFF", "EDIFFG", "PREC", "ISIF", "IBRION"]:
                if k in PARAMETER_SCHEMA:
                    relevant_params[k] = _sanitize_for_json(PARAMETER_SCHEMA[k])
        else:
            for k in [
                "ecutwfc",
                "ecutrho",
                "conv_thr",
                "conv_thr_forces",
                "kpoints_distance",
                "ion_dynamics",
            ]:
                if k in PARAMETER_SCHEMA:
                    relevant_params[k] = _sanitize_for_json(PARAMETER_SCHEMA[k])

        return t.cast(
            dict[str, t.Any],
            _sanitize_for_json(
                {
                    "workflow_type": workflow_type,
                    "description": info.get("description", ""),
                    "required_inputs": info.get("required_inputs", []),
                    "optional_inputs": info.get("optional_inputs", []),
                    "for_structure_types": info.get("for_structure_types", ["all"]),
                    "typical_walltime_hours": info.get("typical_walltime_hours"),
                    "parameters_schema": relevant_params,
                }
            ),
        )

    # If no workflow_type provided, return catalog of all known workflows
    catalog = {}
    for name, data in KNOWN_WORKFLOWS.items():
        catalog[name] = {
            "description": data.get("description", ""),
            "required_inputs": data.get("required_inputs", []),
        }

    return t.cast(
        dict[str, t.Any],
        _sanitize_for_json(
            {
                "query_type": "all_workflow_templates",
                "total_workflows": len(catalog),
                "workflows": catalog,
                "note": "Use get_workflow_templates(workflow_type='...') to see full parameter bounds and typical values.",
            }
        ),
    )
