"""Tools owned by the Execution agent: discovery, protocol-based input
building, and workflow submission.

Public API
----------
list_workflows, describe_workflow
    Read-only introspection of installed process plugins, read straight from
    AiiDA's entry-point registry and ``Process.spec()``.
build_workflow_inputs
    Pre-populate a workflow's inputs from its own protocol builder
    (``get_builder_from_protocol``), when it has one.
draft_workflow_inputs
    Draft the same spec shape from a process's own input ports, for the
    processes that have no protocol builder --- most calculations, and any
    plugin that never adopted the convention.
query_run_context
    Query the Analysis agent for historical context (past successful
    workflows, optimal parameters, structure classifications).
execute_workflow_spec
    Validate a ``WorkflowSpec`` and submit it (HITL-gated, ADR-08).

The write tool ``submit_workflow`` stays an explicit ``tools.execution.submit``
import, so the database-writing tool is not grabbed as casually as a read; the
Execution agent registers ``execute_workflow_spec`` instead, which delegates to
it after building and validating a spec.
"""

from __future__ import annotations

from aiida_agents.tools.run_context import query_run_context
from aiida_agents.tools.execution.drafting import draft_workflow_inputs
from aiida_agents.tools.execution.introspection import describe_workflow, list_workflows
from aiida_agents.tools.execution.protocol import build_workflow_inputs
from aiida_agents.tools.execution.schemas import (
    ValidationError,
    ValidationResult,
    WorkflowSpec,
)
from aiida_agents.tools.execution.spec_execution import execute_workflow_spec

__all__ = [
    "ValidationError",
    "ValidationResult",
    "WorkflowSpec",
    "build_workflow_inputs",
    "describe_workflow",
    "draft_workflow_inputs",
    "execute_workflow_spec",
    "list_workflows",
    "query_run_context",
]
