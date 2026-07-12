"""Tool for executing a validated workflow specification.

Takes a validated workflow dictionary produced by validate_workflow_spec,
extracts the entry point and inputs, and delegates to submit_workflow with Human-In-The-Loop approval.
"""

from __future__ import annotations

import logging
import typing as t

from pydantic import Field

from aiida_agents.tools.submit import submit_workflow
from aiida_agents.tools.workflows.schemas import KNOWN_WORKFLOWS

logger = logging.getLogger(__name__)

__all__ = ["execute_workflow_spec"]


def execute_workflow_spec(
    validated_spec: t.Annotated[
        dict[str, t.Any],
        Field(
            description="The exact dictionary produced by generate_workflow_spec and verified by validate_workflow_spec."
        ),
    ],
) -> dict[str, t.Any]:
    """Execute a validated workflow specification by submitting it to the AiiDA engine.

    This tool takes a validated spec containing `workflow_type` and `inputs`, verifies the structure,
    and safely delegates to `submit_workflow`. Because this triggers actual calculation execution and database
    creation, this tool requires Human-In-The-Loop confirmation (`requires_approval=True`).

    Args:
        validated_spec: The workflow spec dictionary (must contain 'workflow_type' and 'inputs').

    Returns:
        Submission result dictionary containing the process PK and UUID or error details.
    """
    logger.debug("execute_workflow_spec(validated_spec=%r)", validated_spec)

    if not isinstance(validated_spec, dict):
        raise TypeError("validated_spec must be a dictionary.")

    workflow_type = validated_spec.get("workflow_type")
    inputs = validated_spec.get("inputs")

    if not workflow_type or not isinstance(workflow_type, str):
        raise ValueError(
            "validated_spec is missing required 'workflow_type' string parameter."
        )

    if not inputs or not isinstance(inputs, dict):
        raise ValueError("validated_spec is missing required 'inputs' dictionary.")

    if workflow_type not in KNOWN_WORKFLOWS:
        raise ValueError(
            f"Workflow {workflow_type!r} is not known. Must be validated first."
        )

    # Delegate to our secure engine submission function
    logger.info("Delegating validated_spec (%s) to submit_workflow", workflow_type)
    return submit_workflow(entry_point=workflow_type, inputs=inputs)
