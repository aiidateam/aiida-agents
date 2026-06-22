"""Schema-tier validation — type and presence checks against AiiDA port specs.

Uses the process class's own ``spec().inputs`` to determine which ports are
required and what types they accept. No hardcoded workflow knowledge here;
the spec is the source of truth.
"""

from __future__ import annotations

import logging
from typing import Any

from aiida.plugins.entry_point import load_entry_point

logger = logging.getLogger(__name__)


def _load_process_class(entry_point: str) -> Any:
    """Load an AiiDA process class from its entry point string.

    Tries ``aiida.calculations`` first, then ``aiida.workflows``.

    Raises:
        ValueError: If the entry point is not found in either group.
    """
    for group in ("aiida.calculations", "aiida.workflows"):
        try:
            return load_entry_point(group, entry_point)
        except Exception:
            continue

    msg = (
        f"No AiiDA process found for entry point {entry_point!r}. "
        "Check 'verdi plugin list aiida.calculations' and "
        "'verdi plugin list aiida.workflows'."
    )
    raise ValueError(msg)


def validate_schema(entry_point: str, inputs: dict[str, Any]) -> list[str]:
    """Check inputs against the process spec. Returns a list of error strings.

    Two checks per non-metadata port:
    - Required ports must be present.
    - Provided values must be instances of the port's ``valid_type``.

    Metadata ports are skipped — scheduler options validated by AiiDA at
    submit time.
    """
    try:
        process_class = _load_process_class(entry_point)
    except ValueError as exc:
        return [str(exc)]

    spec = process_class.spec()
    errors: list[str] = []

    # CalcJob processes always require a 'code' input to be submitted
    from aiida.engine import CalcJob

    if issubclass(process_class, CalcJob) and "code" not in inputs:
        errors.append(
            "Missing required input: 'code' (required for CalcJob submission)."
        )

    for name, port in spec.inputs.items():
        if name == "metadata":
            continue

        # Skip metadata ports (e.g. inside options namespace)
        if getattr(port, "is_metadata", False):
            continue

        required = getattr(port, "required", False)
        valid_type = getattr(port, "valid_type", None)

        if required and name not in inputs:
            errors.append(f"Missing required input: '{name}'")
            continue

        if name not in inputs:
            continue

        if valid_type is None:
            continue

        value = inputs[name]
        # valid_type can be a single type or a tuple of types
        expected = valid_type if isinstance(valid_type, tuple) else (valid_type,)
        # Filter out NoneType — missing inputs are handled above
        expected = tuple(t for t in expected if t is not type(None))

        if expected and not isinstance(value, expected):
            type_names = ", ".join(t.__name__ for t in expected)
            errors.append(
                f"Input '{name}' has type {type(value).__name__!r}, "
                f"expected one of: {type_names}."
            )

    return errors
