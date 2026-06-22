"""Range and physics tier validation — placeholder for Weeks 7-8.

This tier will enforce sensible value ranges and physics constraints on
workflow inputs, e.g. positive k-point meshes, non-negative energies,
reasonable wallclock limits. Kept separate from schema checks so each
tier can evolve independently.
"""

from __future__ import annotations

from typing import Any


def validate_ranges(entry_point: str, inputs: dict[str, Any]) -> list[str]:  # noqa: ARG001
    """Check inputs against known physical and numerical constraints.

    Not yet implemented — returns an empty list (no errors) until
    workflow-specific range rules are defined.
    """
    return []
