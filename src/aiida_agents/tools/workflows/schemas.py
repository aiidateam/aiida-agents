"""Data structures and metadata for workflow specification and execution.

Defines:
- WorkflowSpec: The structured JSON spec that the model generates and passes to execution.
- ValidationResult / ValidationError: Standard error structures.
"""

from __future__ import annotations

import typing as t

from typing_extensions import TypedDict

__all__ = [
    "WorkflowSpec",
    "ValidationResult",
    "ValidationError",
]


class WorkflowSpec(TypedDict, total=False):
    """A structured workflow specification that the model generates and executes."""

    workflow_type: str
    """Full or short workflow identifier, e.g. 'core.arithmetic.multiply_add'"""

    inputs: dict[str, t.Any]
    """Workflow inputs dictionary mapping port names to values or reference dicts."""

    metadata: dict[str, t.Any]
    """Optional metadata: description, estimated walltime, etc."""


class ValidationError(TypedDict):
    """A single validation error with context."""

    error: str
    parameter: str | None
    suggestion: str | None


class ValidationResult(TypedDict, total=False):
    """Return shape of validation checks."""

    valid: bool
    errors: list[ValidationError]
    warnings: list[str]
    suggestions: list[str]
