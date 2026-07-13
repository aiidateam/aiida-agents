"""Tool for validating workflow specifications against AiiDA schema and best practices."""

from __future__ import annotations

import logging
import typing as t

from pydantic import Field

from aiida_agents.tools.workflows.schemas import (
    KNOWN_WORKFLOWS,
    ValidationError,
    ValidationResult,
    WorkflowSpec,
    check_parameter_compatibility,
    validate_parameter,
)

logger = logging.getLogger(__name__)

__all__ = ["validate_workflow_spec"]


def validate_workflow_spec(
    spec: t.Annotated[
        WorkflowSpec,
        Field(
            description="The workflow spec to validate (output from generate_workflow_spec)"
        ),
    ],
    structure_type: t.Annotated[
        str,
        Field(description="Structure type: metallic, insulator, or semiconductor"),
    ],
) -> ValidationResult:
    """Validate a workflow specification against known AiiDA schema and best practices.

    This tool checks:
    1. Workflow type is known and submittable.
    2. All required inputs are present (``None`` values produce a warning, not an error,
       because the model cannot know the user's structure reference at generation time).
    3. Parameter types and ranges are valid.
    4. Parameter compatibility (ecutrho ≈ 8×ecutwfc, etc.).
    5. K-point density sensible for structure type.

    Args:
        spec: The WorkflowSpec to validate.
        structure_type: What kind of structure (metallic/insulator/semiconductor).

    Returns:
        ValidationResult with valid=True or False, plus errors/suggestions.
    """
    logger.debug(
        "validate_workflow_spec(workflow_type=%r, structure_type=%r)",
        spec.get("workflow_type"),
        structure_type,
    )

    errors: list[ValidationError] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    # ========================================================================
    # Check 1: Workflow type is known
    # ========================================================================

    workflow_type = spec.get("workflow_type")
    if not workflow_type:
        errors.append(
            {
                "error": "Missing workflow_type in spec",
                "parameter": None,
                "suggestion": "Generate spec with a valid workflow_type",
            }
        )
        return {
            "valid": False,
            "errors": errors,
            "suggestions": suggestions,
        }

    if workflow_type not in KNOWN_WORKFLOWS:
        known = ", ".join(KNOWN_WORKFLOWS.keys())
        errors.append(
            {
                "error": f"Unknown workflow_type: {workflow_type}",
                "parameter": "workflow_type",
                "suggestion": f"Use one of: {known}",
            }
        )

    # ========================================================================
    # Check 2: Required inputs are present
    # A required input that is completely *absent* is an error.
    # A required input that is explicitly ``None`` is a warning — the model
    # cannot supply the user's structure reference at generation time; the user
    # must fill it in before submission.
    # ========================================================================

    if workflow_type in KNOWN_WORKFLOWS:
        workflow_def = KNOWN_WORKFLOWS[workflow_type]
        required = workflow_def.get("required_inputs", [])
        inputs = spec.get("inputs", {})

        for req_input in required:
            if req_input not in inputs:
                errors.append(
                    {
                        "error": f"Missing required input: {req_input}",
                        "parameter": req_input,
                        "suggestion": f"Provide a value for {req_input}",
                    }
                )
            elif inputs[req_input] is None:
                warnings.append(
                    f"Required input '{req_input}' is null (placeholder). "
                    f"The user must supply the actual reference before submission "
                    f'(e.g. {{"pk": <node_pk>}} or {{"label": "name@computer"}}).'
                )

    # ========================================================================
    # Check 3: Validate parameters (type, range)
    # ========================================================================

    inputs = spec.get("inputs", {})
    parameters = inputs.get("parameters", {})
    flat_params: dict[
        str, t.Any
    ] = {}  # Flatten nested parameters for compatibility checks

    if isinstance(parameters, dict):
        for param_name, param_value in parameters.items():
            # Flatten nested parameters (system.ecutwfc → ecutwfc for schema lookup)
            if isinstance(param_value, dict):
                for nested_name, nested_value in param_value.items():
                    is_valid, error_msg = validate_parameter(nested_name, nested_value)
                    if not is_valid:
                        errors.append(
                            {
                                "error": error_msg or "Unknown validation error",
                                "parameter": f"{param_name}.{nested_name}",
                                "suggestion": _suggest_parameter_fix(
                                    nested_name, nested_value, structure_type
                                ),
                            }
                        )
                    flat_params[nested_name] = nested_value
            else:
                # Flat parameter
                is_valid, error_msg = validate_parameter(param_name, param_value)
                if not is_valid:
                    errors.append(
                        {
                            "error": error_msg or "Unknown validation error",
                            "parameter": param_name,
                            "suggestion": _suggest_parameter_fix(
                                param_name, param_value, structure_type
                            ),
                        }
                    )
                flat_params[param_name] = param_value

    # ========================================================================
    # Check 4: Parameter compatibility
    # ========================================================================

    compat_warnings = check_parameter_compatibility(flat_params, structure_type)
    warnings.extend(compat_warnings)

    # ========================================================================
    # Check 5: K-points distance sensibility
    # ========================================================================

    kpoints_distance = inputs.get("kpoints_distance")
    if kpoints_distance is not None:
        if not isinstance(kpoints_distance, (int, float)):
            errors.append(
                {
                    "error": f"kpoints_distance: expected float, got {type(kpoints_distance).__name__}",
                    "parameter": "kpoints_distance",
                    "suggestion": "Use a float value like 0.2",
                }
            )
        elif kpoints_distance < 0.05 or kpoints_distance > 1.0:
            warnings.append(
                f"kpoints_distance={kpoints_distance} is outside typical range [0.05, 1.0]. "
                "Standard is 0.2 (denser = smaller value, slower but more accurate)"
            )

    # ========================================================================
    # Check 6: Metallic-specific checks
    # ========================================================================

    if structure_type == "metallic" and flat_params.get("scf_kpoints"):
        if flat_params["scf_kpoints"] < 20:
            warnings.append(
                f"Metallic structure with only {flat_params['scf_kpoints']} k-points may be insufficient. "
                "Consider 30+ points for accurate electronic structure."
            )

    # ========================================================================
    # Check 7: Pseudopotential family existence in the active AiiDA profile
    # ========================================================================

    pseudo_family = inputs.get("pseudo_family")
    if pseudo_family and isinstance(pseudo_family, str):
        _pseudo_ok, _pseudo_note = _check_pseudo_family_exists(pseudo_family)
        if not _pseudo_ok:
            warnings.append(
                f"Pseudopotential family {pseudo_family!r} is not installed in the active AiiDA profile. "
                f"{_pseudo_note} "
                "Run: aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency — "
                "this must be done before submitting."
            )

    # ========================================================================
    # Return result
    # ========================================================================

    if errors:
        suggestions.extend(
            [
                "Review the errors above and regenerate the spec with corrections",
                "If unsure about parameter values, call query_analysis_agent() "
                "to see past successful runs on similar structures",
            ]
        )

    logger.debug(
        "validate_workflow_spec: valid=%s, %d errors, %d warnings",
        len(errors) == 0,
        len(errors),
        len(warnings),
    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
    }


# ============================================================================
# Helper: suggest fixes for specific parameter failures
# ============================================================================


def _suggest_parameter_fix(
    param_name: str, param_value: t.Any, structure_type: str
) -> str:
    """Generate a suggestion for fixing a parameter."""
    suggestions = {
        "ecutwfc": (
            f"Try ecutwfc={65 if structure_type == 'metallic' else 50 if structure_type == 'insulator' else 55}. "
            f"{structure_type.capitalize()} structures typically use this range."
        ),
        "ecutrho": "Set ecutrho = 8 × ecutwfc. For ecutwfc=65, use ecutrho=520.",
        "conv_thr": "Use scientific notation: 1e-8 for standard, 1e-9 for high accuracy",
        "conv_thr_forces": "For geometry optimization, 1e-4 Ry/Bohr is typical",
        "kpoints_distance": "Typical range: 0.15-0.25 Å^-1. Use 0.2 for standard accuracy",
        "ion_dynamics": "Use 'bfgs' for geometry optimization (standard choice)",
        "scf_maxiter": "Typical: 100 iterations. Increase if convergence is slow.",
        "mixing_beta": "For difficult SCF: 0.3-0.5. For easy: 0.7. Typical: 0.7",
    }
    return suggestions.get(param_name, "Check the parameter value and try again")


def _check_pseudo_family_exists(pseudo_family: str) -> tuple[bool, str]:
    """Check whether a pseudopotential family label exists in the active AiiDA profile.

    Returns:
        (True, \"\") if found, or (False, note_message) if not found or AiiDA unavailable.
    """
    try:
        from aiida import orm

        # First: exact label match (most reliable)
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={"label": pseudo_family})
        if qb.count() > 0:
            return True, ""

        # Second: list only real pseudo family groups (type_string from aiida-pseudo)
        qb2 = orm.QueryBuilder()
        qb2.append(
            orm.Group,
            filters={"type_string": {"like": "aiida_pseudo%"}},
            project=["label"],
        )
        pseudo_labels = [row[0] for row in qb2.all()]

        if pseudo_labels:
            available = ", ".join(pseudo_labels[:5])
            return (
                False,
                f"Family {pseudo_family!r} not found. Installed pseudo families: {available}.",
            )
        return False, (
            "No pseudopotential families are installed in the active AiiDA profile. "
            "Install one first: aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency"
        )
    except Exception as exc:
        # AiiDA not loaded (e.g. in tests) — skip the check rather than blocking
        logger.debug("Pseudo family existence check skipped: %s", exc)
        return True, ""
