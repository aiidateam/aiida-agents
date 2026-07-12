"""Tool for validating workflow specifications against AiiDA schema.

Enhanced version with parameter compatibility checking.
"""

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
    spec: WorkflowSpec = Field(
        description="The workflow spec to validate (output from generate_workflow_spec)"
    ),
    structure_type: str = Field(
        description="Structure type: metallic, insulator, or semiconductor"
    ),
) -> ValidationResult:
    """Validate a workflow specification against known AiiDA schema and best practices.

    This tool checks:
    1. Workflow type is known
    2. All required inputs are present
    3. Parameter types and ranges are valid
    4. Parameter compatibility (ecutrho ≈ 8×ecutwfc, etc.)
    5. K-point density sensible for structure type

    Args:
        spec: The WorkflowSpec to validate
        structure_type: What kind of structure (metallic/insulator/semiconductor)

    Returns:
        ValidationResult with valid=True or False, plus errors/suggestions
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
    # ========================================================================

    if workflow_type in KNOWN_WORKFLOWS:
        workflow_def = KNOWN_WORKFLOWS[workflow_type]
        required = workflow_def.get("required_inputs", [])
        inputs = spec.get("inputs", {})

        for req_input in required:
            if req_input not in inputs or inputs[req_input] is None:
                errors.append(
                    {
                        "error": f"Missing required input: {req_input}",
                        "parameter": req_input,
                        "suggestion": f"Provide a value for {req_input}",
                    }
                )

    # ========================================================================
    # Check 3: Validate parameters (type, range)
    # ========================================================================

    inputs = spec.get("inputs", {})
    parameters = inputs.get("parameters", {})
    flat_params = {}  # Flatten nested parameters for compatibility checks

    if isinstance(parameters, dict):
        for param_name, param_value in parameters.items():
            # Flatten nested parameters (system.ecutwfc -> ecutwfc for schema lookup)
            if isinstance(param_value, dict):
                # Nested structure like {"system": {"ecutwfc": 65}}
                for nested_name, nested_value in param_value.items():
                    is_valid, error_msg = validate_parameter(nested_name, nested_value)
                    if not is_valid:
                        errors.append(
                            {
                                "error": error_msg,
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
                            "error": error_msg,
                            "parameter": param_name,
                            "suggestion": _suggest_parameter_fix(
                                param_name, param_value, structure_type
                            ),
                        }
                    )
                flat_params[param_name] = param_value

    # ========================================================================
    # Check 4: Parameter compatibility (NEW)
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
        "ecutwfc": f"Try ecutwfc={65 if structure_type == 'metallic' else 50 if structure_type == 'insulator' else 55}. "
        f"{structure_type.capitalize()} structures typically use this range.",
        "ecutrho": "Set ecutrho = 8 × ecutwfc. For ecutwfc=65, use ecutrho=520.",
        "conv_thr": "Use scientific notation: 1e-8 for standard, 1e-9 for high accuracy",
        "conv_thr_forces": "For geometry optimization, 1e-4 Ry/Bohr is typical",
        "kpoints_distance": "Typical range: 0.15-0.25 Å^-1. Use 0.2 for standard accuracy",
        "ion_dynamics": "Use 'bfgs' for geometry optimization (standard choice)",
        "scf_maxiter": "Typical: 100 iterations. Increase if convergence is slow.",
        "mixing_beta": "For difficult SCF: 0.3-0.5. For easy: 0.7. Typical: 0.7",
    }

    return suggestions.get(param_name, "Check the parameter value and try again")
