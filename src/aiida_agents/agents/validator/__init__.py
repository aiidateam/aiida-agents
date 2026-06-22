"""Validator for AiiDA workflow submissions.

Sits between the agent's decision to submit and the actual submission.
All tiers must pass before a workflow reaches the write path.

Public API
----------
validate(workflow_entry_point, inputs)
    Run all validation tiers. Raises ``ValidationError`` on failure.
"""

from __future__ import annotations

from typing import Any

from aiida_agents.agents.validator._schema import validate_schema

__all__ = ["ValidationError", "validate"]


class ValidationError(Exception):
    """Raised when workflow inputs fail validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def validate(workflow_entry_point: str, inputs: dict[str, Any]) -> None:
    """Validate workflow inputs against all tiers.

    Args:
        workflow_entry_point: AiiDA entry point string, e.g.
            ``"core.arithmetic.add"`` or ``"core.arithmetic.multiply_add"``.
        inputs: Dict mapping port names to AiiDA nodes or plain Python values.

    Raises:
        ValidationError: If any tier finds invalid inputs.
    """
    errors: list[str] = []
    errors.extend(validate_schema(workflow_entry_point, inputs))
    if errors:
        raise ValidationError(errors)
