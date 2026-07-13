"""Data structures and metadata for workflow specification and validation.

Defines:
- WorkflowSpec: The structured JSON spec that the model generates
- ValidationResult: The return type of validate_workflow_spec
- KNOWN_WORKFLOWS: Expanded Phase 1 workflow templates (6+ workflows)
- PARAMETER_SCHEMA: Complete parameter definitions and ranges
"""

from __future__ import annotations

import typing as t

from typing_extensions import TypedDict

__all__ = [
    "WorkflowSpec",
    "ValidationResult",
    "ValidationError",
    "KNOWN_WORKFLOWS",
    "PARAMETER_SCHEMA",
    "PSEUDO_FAMILIES",
    "validate_parameter",
]


class WorkflowSpec(TypedDict, total=False):
    """A structured workflow specification that the model generates."""

    workflow_type: str
    """Full workflow identifier, e.g. 'aiida.workflows:PwRelaxWorkChain'"""

    inputs: dict[str, t.Any]
    """Workflow inputs: structure, codes, pseudo_family, parameters, k-points"""

    metadata: dict[str, t.Any]
    """Optional metadata: description, estimated walltime, etc."""


class ValidationError(TypedDict):
    """A single validation error with context."""

    error: str
    parameter: str | None
    suggestion: str | None


class ValidationResult(TypedDict, total=False):
    """Return shape of ``validate_workflow_spec``."""

    valid: bool
    errors: list[ValidationError]
    warnings: list[str]
    suggestions: list[str]


# ============================================================================
# Phase 1: Expanded hardcoded workflow templates (6+ workflows)
# ============================================================================

KNOWN_WORKFLOWS: dict[str, dict[str, t.Any]] = {
    "aiida.workflows:PwRelaxWorkChain": {
        "description": "Geometry relaxation using Quantum ESPRESSO pw.x",
        "required_inputs": ["pw_code_label", "structure", "pseudo_family"],
        "optional_inputs": ["parameters", "kpoints_distance", "electronic_type"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 4,
        "common_failure_modes": [
            "Convergence not met (ecutwfc too low)",
            "SCF non-convergence",
            "Geometry optimization did not converge",
        ],
    },
    "aiida.workflows:PwBandsWorkChain": {
        "description": "Electronic band structure using SCF + bands calculation",
        "required_inputs": [
            "pw_code_label",
            "structure",
            "pseudo_family",
            "scf_kpoints_distance",
        ],
        "optional_inputs": ["parameters", "bands_kpoints_distance"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 8,
        "note": "Requires converged structure from PwRelaxWorkChain",
    },
    "aiida.workflows:PwDosWorkChain": {
        "description": "Density of states calculation",
        "required_inputs": ["pw_code_label", "structure", "pseudo_family"],
        "optional_inputs": ["parameters", "kpoints_distance", "dos_kpoints_distance"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 6,
    },
    "aiida.workflows:PwCalculation": {
        "description": "Single point SCF calculation (non-self-consistent)",
        "required_inputs": ["pw_code_label", "structure", "pseudo_family"],
        "optional_inputs": ["parameters", "kpoints_distance"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 2,
    },
    "aiida.workflows:PhRelaxWorkChain": {
        "description": "Phonon calculation with ionic relaxation",
        "required_inputs": [
            "pw_code_label",
            "ph_code_label",
            "structure",
            "pseudo_family",
        ],
        "optional_inputs": ["parameters", "kpoints_distance"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 12,
        "note": "Requires both pw and ph codes",
    },
    "aiida.workflows:PwSecondOrderWorkChain": {
        "description": "Second-order elastic constants calculation",
        "required_inputs": ["pw_code_label", "structure", "pseudo_family"],
        "optional_inputs": ["parameters", "kpoints_distance", "displacement"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 8,
    },
    "aiida.workflows:VaspWorkChain": {
        "description": "Single point SCF calculation using VASP",
        "required_inputs": ["code_label", "structure", "potential_family"],
        "optional_inputs": ["parameters", "kpoints_distance"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 3,
    },
    "aiida.workflows:VaspRelaxWorkChain": {
        "description": "Geometry optimization using VASP",
        "required_inputs": ["code_label", "structure", "potential_family"],
        "optional_inputs": ["parameters", "kpoints_distance"],
        "for_structure_types": ["all"],
        "typical_walltime_hours": 6,
    },
}

# ============================================================================
# Phase 1: Expanded parameter metadata and ranges
# ============================================================================

PARAMETER_SCHEMA: dict[str, dict[str, t.Any]] = {
    # VASP Cutoff Energy (eV)
    "ENCUT": {
        "description": "VASP plane wave cutoff energy (eV)",
        "type": "float",
        "min": 100.0,
        "max": 1500.0,
        "typical_metallic": 520.0,
        "typical_insulator": 400.0,
        "typical_semiconductor": 450.0,
        "unit": "eV",
    },
    # VASP Energy Convergence Threshold (eV)
    "EDIFF": {
        "description": "VASP SCF electronic convergence threshold (eV)",
        "type": "float",
        "min": 1e-9,
        "max": 1e-3,
        "typical": 1e-6,
        "unit": "eV",
    },
    # VASP Force/Energy Convergence Threshold for Geometry Optimization
    "EDIFFG": {
        "description": "VASP ionic relaxation convergence threshold (negative for forces in eV/Å)",
        "type": "float",
        "min": -1.0,
        "max": 0.1,
        "typical": -0.01,
        "unit": "eV/Å",
    },
    # VASP Precision Setting
    "PREC": {
        "description": "VASP calculation precision mode",
        "type": "str",
        "allowed_values": ["Low", "Medium", "Normal", "High", "Accurate"],
        "typical": "Accurate",
    },
    # VASP Stress Tensor and Relaxation Degrees of Freedom
    "ISIF": {
        "description": "VASP degrees of freedom for ionic relaxation (e.g., 2=ions only, 3=ions+volume+shape)",
        "type": "int",
        "min": 0,
        "max": 7,
        "typical": 3,
    },
    # VASP Ionic Relaxation Algorithm
    "IBRION": {
        "description": "VASP ionic optimization algorithm (1=RMM-DIIS, 2=CG)",
        "type": "int",
        "min": -1,
        "max": 8,
        "typical": 2,
    },
    # Wavefunction cutoff
    "ecutwfc": {
        "description": "Plane wave cutoff for wavefunctions (Ry)",
        "type": "float",
        "min": 20,
        "max": 200,
        "typical_metallic": 65,
        "typical_insulator": 50,
        "typical_semiconductor": 55,
        "unit": "Ry",
    },
    # Charge density cutoff
    "ecutrho": {
        "description": "Charge density cutoff (Ry), typically 8x ecutwfc",
        "type": "float",
        "min": 100,
        "max": 800,
        "rule": "8x ecutwfc (±10% tolerance)",
        "unit": "Ry",
    },
    # SCF convergence threshold (energy)
    "conv_thr": {
        "description": "SCF convergence threshold (Ry)",
        "type": "float",
        "min": 1e-10,
        "max": 1e-6,
        "typical": 1e-8,
        "note": "Use scientific notation (1e-8, not 0.00000001)",
    },
    # Force convergence threshold (for geometry optimization)
    "conv_thr_forces": {
        "description": "Force convergence threshold for ionic relaxation (Ry/Bohr)",
        "type": "float",
        "min": 1e-6,
        "max": 1e-3,
        "typical": 1e-4,
        "unit": "Ry/Bohr",
    },
    # K-point density
    "kpoints_distance": {
        "description": "Target k-point grid density (Å^-1)",
        "type": "float",
        "min": 0.05,
        "max": 1.0,
        "typical": 0.2,
        "note": "Smaller = denser, more accurate but slower",
        "unit": "Å^-1",
    },
    # Ionic dynamics (relaxation algorithm)
    "ion_dynamics": {
        "description": "Ionic optimization algorithm",
        "type": "str",
        "allowed_values": ["bfgs", "damp", "beeman"],
        "typical": "bfgs",
    },
    # Maximum number of SCF iterations
    "scf_maxiter": {
        "description": "Maximum number of SCF iterations",
        "type": "int",
        "min": 1,
        "max": 500,
        "typical": 100,
    },
    # Electronic mixer beta (mixing parameter)
    "mixing_beta": {
        "description": "Mixing parameter for SCF (0-1, lower = more stable but slower)",
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "typical": 0.7,
    },
    # Maximum ionic steps (geometry optimization)
    "ion_relax_maxsteps": {
        "description": "Maximum number of geometry optimization steps",
        "type": "int",
        "min": 1,
        "max": 1000,
        "typical": 100,
    },
    # Smearing type (for metallic structures)
    "smearing": {
        "description": "Smearing type for electronic occupancies",
        "type": "str",
        "allowed_values": ["gaussian", "fermi-dirac", "methfessel-paxton"],
        "typical": "gaussian",
        "note": "Used for metallic structures",
    },
    # Smearing width
    "degauss": {
        "description": "Smearing width (Ry)",
        "type": "float",
        "min": 0.001,
        "max": 0.5,
        "typical": 0.01,
        "unit": "Ry",
    },
    # Celldm (cell shape parameters)
    "celldm": {
        "description": "Cell lattice parameters",
        "type": "dict",
        "note": "Structure-dependent; usually not modified",
    },
    # Nosé thermostat temperature
    "temp_w": {
        "description": "Thermostat temperature (K) for MD",
        "type": "float",
        "min": 0,
        "max": 10000,
        "typical": 300,
        "unit": "K",
    },
}

# ============================================================================
# Pseudopotential families (common in AiiDA profiles)
# ============================================================================

PSEUDO_FAMILIES: list[str] = [
    "SSSP/1.3/PBE/efficiency",
    "SSSP/1.3/PBE/precision",
    "SSSP/1.3/PBE/precision_stringent",
    "SSSP/1.2/PBE/efficiency",
    "SSSP/1.2/PBE/precision",
    "PAW",
    "GBRV",
    "LDA",
]

# ============================================================================
# Parameter compatibility rules
# ============================================================================

PARAMETER_COMPAT_RULES: list[dict[str, t.Any]] = [
    {
        "rule": "ecutrho ≈ 8 × ecutwfc (±10% tolerance)",
        "tolerance": 0.1,
        "params": ["ecutwfc", "ecutrho"],
        "check": lambda vals: (
            0.9 * vals.get("ecutwfc", 65) * 8
            <= vals.get("ecutrho", 520)
            <= 1.1 * vals.get("ecutwfc", 65) * 8
        ),
    },
    {
        "rule": "If smearing is used, degauss must be set",
        "params": ["smearing", "degauss"],
        "check": lambda vals: (
            not vals.get("smearing") or vals.get("degauss") is not None
        ),
    },
    {
        "rule": "For metallic structures, smearing should be gaussian or fermi-dirac",
        "structure_type": "metallic",
        "params": ["smearing"],
        "check": lambda vals: vals.get("smearing") in [None, "gaussian", "fermi-dirac"],
    },
    # NEW RULES:
    {
        "rule": "k_points_distance for metallic structures should be < 0.3 Å^-1 (finer than standard 0.2)",
        "structure_type": "metallic",
        "params": ["kpoints_distance"],
        "check": lambda vals: (
            vals.get("kpoints_distance") is None or vals.get("kpoints_distance") <= 0.3
        ),
    },
    {
        "rule": "High accuracy (conv_thr < 1e-8) should use scf_maxiter >= 150",
        "params": ["conv_thr", "scf_maxiter"],
        "check": lambda vals: (
            vals.get("conv_thr", 1e-8) >= 1e-8 or vals.get("scf_maxiter", 100) >= 150
        ),
    },
    {
        "rule": "For geometry optimization, set both conv_thr (energy) and conv_thr_forces",
        "params": ["conv_thr", "conv_thr_forces"],
        "check": lambda vals: (
            not vals.get("conv_thr") or vals.get("conv_thr_forces") is not None
        ),
    },
]

# More specific parameter value suggestions based on structure type
PARAMETER_SUGGESTIONS_BY_STRUCTURE = {
    "metallic": {
        "ecutwfc": {
            "standard": 65,
            "high_accuracy": 70,
            "range": (60, 80),
            "reason": "Metallic structures need higher cutoff for accurate electronic structure",
        },
        "kpoints_distance": {
            "standard": 0.18,
            "high_accuracy": 0.12,
            "range": (0.10, 0.25),
            "reason": "Metallic structures need dense k-grids for accurate DOS/bands",
        },
        "mixing_beta": {
            "standard": 0.7,
            "high_accuracy": 0.5,
            "reason": "Metallic SCF can be tricky; reduce beta for convergence",
        },
    },
    "insulator": {
        "ecutwfc": {
            "standard": 50,
            "high_accuracy": 60,
            "range": (40, 70),
            "reason": "Insulators converge easily; moderate cutoff sufficient",
        },
        "kpoints_distance": {
            "standard": 0.20,
            "high_accuracy": 0.15,
            "range": (0.15, 0.30),
            "reason": "Insulators don't need as dense k-grids as metals",
        },
    },
    "semiconductor": {
        "ecutwfc": {
            "standard": 55,
            "high_accuracy": 65,
            "range": (50, 75),
            "reason": "Semiconductors need careful parameter tuning",
        },
        "kpoints_distance": {
            "standard": 0.19,
            "high_accuracy": 0.14,
            "range": (0.12, 0.28),
            "reason": "Semiconductors need gap-dependent k-point sampling",
        },
    },
}

# ============================================================================
# Helper: validate parameter value against schema
# ============================================================================


def validate_parameter(param_name: str, param_value: t.Any) -> tuple[bool, str | None]:
    """Check if a parameter value matches its schema definition.

    Returns:
        (is_valid, error_message)
    """
    if param_name not in PARAMETER_SCHEMA:
        return False, f"Unknown parameter: {param_name}"

    schema = PARAMETER_SCHEMA[param_name]
    expected_type = schema.get("type")

    # Type check
    if expected_type:
        type_name = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else str(expected_type)
        )
        if type_name == "float" or expected_type is float:
            valid_type = isinstance(param_value, (int, float)) and not isinstance(
                param_value, bool
            )
        elif type_name == "int" or expected_type is int:
            valid_type = isinstance(param_value, int) and not isinstance(
                param_value, bool
            )
        elif type_name == "str" or expected_type is str:
            valid_type = isinstance(param_value, str)
        elif type_name == "dict" or expected_type is dict:
            valid_type = isinstance(param_value, dict)
        elif type_name == "bool" or expected_type is bool:
            valid_type = isinstance(param_value, bool)
        elif isinstance(expected_type, type):
            valid_type = isinstance(param_value, expected_type)
        else:
            valid_type = type(param_value).__name__ == type_name

        if not valid_type:
            return False, (
                f"{param_name}: expected {type_name}, got {type(param_value).__name__}"
            )

    # Range check (for numeric types)
    if isinstance(param_value, (int, float)):
        min_val = schema.get("min")
        max_val = schema.get("max")

        if min_val is not None and param_value < min_val:
            return False, (f"{param_name}: {param_value} is below minimum {min_val}")
        if max_val is not None and param_value > max_val:
            return False, (f"{param_name}: {param_value} exceeds maximum {max_val}")

    # Enum check (for allowed_values)
    allowed = schema.get("allowed_values")
    if allowed and param_value not in allowed:
        return False, (f"{param_name}: {param_value!r} not in allowed values {allowed}")

    return True, None


def check_parameter_compatibility(
    parameters: dict[str, t.Any], structure_type: str | None = None
) -> list[str]:
    """Check if parameters are mutually compatible.

    Returns:
        List of compatibility warnings (empty if all OK)
    """
    warnings = []

    for rule in PARAMETER_COMPAT_RULES:
        # Skip if rule is for specific structure type and doesn't match
        if "structure_type" in rule and rule["structure_type"] != structure_type:
            continue

        try:
            if not rule["check"](parameters):
                warnings.append(rule["rule"])
        except (KeyError, TypeError, ValueError):
            # One of the required parameters is missing or malformed, skip compatibility check
            pass

    return warnings
