"""Data structures and metadata for process specification and submission.

Defines:
- SubmissionSpec: The structured JSON spec that the model generates and submits.
- ValidationResult / ValidationError: Standard error structures.
"""

from __future__ import annotations

import typing as t

from typing_extensions import TypedDict

__all__ = [
    "SubmissionSpec",
    "ValidationResult",
    "ValidationError",
]


class SubmissionSpec(TypedDict, total=False):
    """A structured process specification that the model generates and submits."""

    entry_point: str
    """Entry point of the process to run, e.g. 'core.arithmetic.multiply_add'"""

    inputs: dict[str, t.Any]
    """Process inputs dictionary mapping port names to values or reference dicts."""

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
