"""Tools owned by the Execution agent: discovery, protocol-based input
building, and workflow submission.

Public API
----------
list_process_entry_points, describe_process
    Read-only introspection of installed process plugins, read straight from
    AiiDA's entry-point registry and ``Process.spec()``.
build_process_inputs
    Pre-populate a workflow's inputs from its own protocol builder
    (``get_builder_from_protocol``), when it has one.
draft_process_inputs
    Draft the same spec shape from a process's own input ports, for the
    processes that have no protocol builder --- most calculations, and any
    plugin that never adopted the convention.
check_cutoffs_against_pseudos
    Compare a spec's cutoffs against what its pseudopotential family was
    converged for, before it is submitted.
query_run_context
    Query the Analysis agent for historical context (past successful
    workflows, optimal parameters, structure classifications).
execute_workflow_spec
    Validate a ``WorkflowSpec`` and submit it (HITL-gated, ADR-08).
wait_for_process
    Wait for a submitted process to finish and report its outputs, so one
    submission can be run against another's result.

The write tool ``submit_workflow`` stays an explicit ``tools.execution.submit``
import, so the database-writing tool is not grabbed as casually as a read; the
Execution agent registers ``execute_workflow_spec`` instead, which delegates to
it after building and validating a spec.
"""

from __future__ import annotations

from aiida_agents.tools.run_context import query_run_context
from aiida_agents.tools.execution.drafting import draft_process_inputs
from aiida_agents.tools.execution.introspection import (
    describe_process,
    list_process_entry_points,
)
from aiida_agents.tools.execution.protocol import build_process_inputs
from aiida_agents.tools.execution.ranges import check_cutoffs_against_pseudos
from aiida_agents.tools.execution.resubmission import build_resubmission_spec
from aiida_agents.tools.execution.schemas import (
    ValidationError,
    ValidationResult,
    WorkflowSpec,
)
from aiida_agents.tools.execution.spec_execution import execute_workflow_spec
from aiida_agents.tools.execution.waiting import wait_for_process

__all__ = [
    "ValidationError",
    "ValidationResult",
    "WorkflowSpec",
    "build_resubmission_spec",
    "build_process_inputs",
    "check_cutoffs_against_pseudos",
    "describe_process",
    "draft_process_inputs",
    "execute_workflow_spec",
    "list_process_entry_points",
    "query_run_context",
    "wait_for_process",
]
