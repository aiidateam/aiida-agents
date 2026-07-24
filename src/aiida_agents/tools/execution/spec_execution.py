"""Tool for executing a validated workflow specification.

Takes a validated WorkflowSpec produced by validate_workflow_spec, extracts the
entry point and inputs, and delegates to submit_workflow with Human-In-The-Loop
approval.
"""

from __future__ import annotations

import logging
import typing as t

from aiida.common.exceptions import MissingEntryPointError
from aiida.plugins.entry_point import load_entry_point
from pydantic import Field

from aiida_agents.tools._types import SubmitResult
from aiida_agents.tools.execution.submit import submit_workflow
from aiida_agents.tools.execution.schemas import WorkflowSpec

logger = logging.getLogger(__name__)

__all__ = ["execute_workflow_spec"]


def execute_workflow_spec(
    validated_spec: t.Annotated[
        WorkflowSpec,
        Field(
            description="The exact WorkflowSpec dictionary to execute, containing 'workflow_type' and 'inputs'."
        ),
    ],
) -> SubmitResult:
    """Execute a validated workflow specification by submitting it to the AiiDA engine.

    This tool takes a WorkflowSpec containing ``workflow_type`` and ``inputs``,
    verifies the structure and entry point, and safely delegates to ``submit_workflow``.
    Because this triggers actual calculation execution and database creation, this
    tool requires Human-In-The-Loop confirmation (``requires_approval=True``).

    Args:
        validated_spec: The WorkflowSpec dictionary (must contain 'workflow_type'
            and 'inputs').

    Returns:
        Submission result dictionary containing the process PK and UUID or error
        details.
    """
    logger.debug("execute_workflow_spec(validated_spec=%r)", validated_spec)

    if not isinstance(validated_spec, dict):
        msg = "validated_spec must be a dictionary."
        raise TypeError(msg)

    workflow_type = validated_spec.get("workflow_type")
    inputs = validated_spec.get("inputs")

    if not workflow_type or not isinstance(workflow_type, str):
        msg = "validated_spec is missing required 'workflow_type' string parameter."
        raise ValueError(msg)

    if not inputs or not isinstance(inputs, dict):
        msg = "validated_spec is missing required 'inputs' dictionary."
        raise ValueError(msg)

    found = False
    for group in ("aiida.workflows", "aiida.calculations"):
        try:
            load_entry_point(group, workflow_type)
            found = True
            break
        except MissingEntryPointError:
            continue
        except Exception as exc:
            msg = f"Workflow {workflow_type!r} failed to load: {exc}"
            raise ValueError(msg) from exc

    if not found:
        msg = f"Workflow {workflow_type!r} is not a known AiiDA entry point."
        raise ValueError(msg)

    # Delegate to our secure engine submission function
    logger.info("Delegating validated_spec (%s) to submit_workflow", workflow_type)
    return submit_workflow(entry_point=workflow_type, inputs=inputs)
