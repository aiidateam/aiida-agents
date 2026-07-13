"""Tool for generating structured workflow specifications."""

from __future__ import annotations

import logging
import typing as t

from pydantic import Field

from aiida_agents.tools.workflows.schemas import (
    KNOWN_WORKFLOWS,
    PSEUDO_FAMILIES,
    WorkflowSpec,
)

logger = logging.getLogger(__name__)

__all__ = ["generate_workflow_spec"]


def _get_default_pw_code() -> str:
    """Query the active AiiDA profile for the first installed quantumespresso.pw code.

    Returns the code label if found, otherwise a placeholder that will be
    caught by validate_workflow_spec and reported to the agent.
    """
    try:
        from aiida import orm

        qb = orm.QueryBuilder()
        qb.append(
            orm.Code,
            filters={"attributes.input_plugin": "quantumespresso.pw"},
            project=["label"],
        )
        results = qb.all(flat=True)
        if results:
            return str(results[0])
    except Exception as exc:
        logger.debug("Could not query pw codes from AiiDA: %s", exc)
    return "<no-pw-code-found>"


def generate_workflow_spec(
    description: t.Annotated[
        str,
        Field(
            description="What calculation the user wants (e.g. 'Geometry optimization of CoV')"
        ),
    ],
    workflow_type: t.Annotated[
        str,
        Field(
            description=(
                "Which workflow to run (e.g. 'aiida.workflows:PwRelaxWorkChain'). "
                "Choose from: PwRelaxWorkChain, PwBandsWorkChain, PwDosWorkChain, "
                "PwCalculation, PhRelaxWorkChain, VaspWorkChain, VaspRelaxWorkChain"
            )
        ),
    ],
    structure_type: t.Annotated[
        str,
        Field(
            description="What kind of structure: 'metallic', 'insulator', 'semiconductor'"
        ),
    ],
    optimization_level: t.Annotated[
        str,
        Field(
            description=(
                "Calculation accuracy level: 'standard' (faster) or "
                "'high_accuracy' (slower, more accurate)"
            )
        ),
    ],
) -> WorkflowSpec:
    """Generate a structured workflow specification for execution.

    This tool produces a JSON specification describing which workflow to run,
    what inputs it needs, and what parameters to use.  Falls back to hardcoded
    sensible defaults when no historical context is available.

    The ``structure`` field in the returned spec is set to ``None`` as a
    placeholder — the model cannot know the user's AiiDA structure reference.
    Before calling ``validate_workflow_spec``, the user must supply the
    structure as a reference dict (e.g. ``{"pk": 12345}``).

    Args:
        description: What the user is trying to calculate.
        workflow_type: Which AiiDA workflow to use.
        structure_type: metallic/insulator/semiconductor.
        optimization_level: standard or high_accuracy.

    Returns:
        WorkflowSpec: A JSON-like dict specifying workflow, inputs, parameters.
    """
    logger.debug(
        "generate_workflow_spec(description=%r, workflow_type=%r, "
        "structure_type=%r, optimization_level=%r)",
        description,
        workflow_type,
        structure_type,
        optimization_level,
    )

    if optimization_level not in ("standard", "high_accuracy"):
        msg = (
            f"optimization_level must be 'standard' or 'high_accuracy', "
            f"got {optimization_level!r}"
        )
        raise ValueError(msg)

    if structure_type not in ("metallic", "insulator", "semiconductor"):
        msg = (
            f"structure_type must be 'metallic', 'insulator', or 'semiconductor', "
            f"got {structure_type!r}"
        )
        raise ValueError(msg)

    if workflow_type not in KNOWN_WORKFLOWS:
        known = ", ".join(KNOWN_WORKFLOWS.keys())
        msg = f"Unknown workflow_type: {workflow_type!r}. Known workflows: {known}"
        raise ValueError(msg)

    # ========================================================================
    # Build the spec based on workflow type and user parameters
    # ========================================================================

    spec: WorkflowSpec = {
        "workflow_type": workflow_type,
        "inputs": {},
        "metadata": {"description": description},
    }

    if workflow_type == "aiida.workflows:PwRelaxWorkChain":
        spec["inputs"] = {
            "pw_code_label": _get_default_pw_code(),
            "pseudo_family": PSEUDO_FAMILIES[0],
            "structure": None,
            "parameters": _build_relax_parameters(structure_type, optimization_level),
            "kpoints_distance": (
                0.15 if optimization_level == "high_accuracy" else 0.2
            ),
        }
        spec["metadata"]["estimated_walltime_hours"] = 4

    elif workflow_type == "aiida.workflows:PwBandsWorkChain":
        spec["inputs"] = {
            "pw_code_label": _get_default_pw_code(),
            "pseudo_family": PSEUDO_FAMILIES[0],
            "structure": None,
            "scf_kpoints_distance": 0.2,
            "bands_kpoints_distance": 0.1,
            "parameters": _build_scf_parameters(structure_type, optimization_level),
        }
        spec["metadata"]["estimated_walltime_hours"] = 8

    elif workflow_type == "aiida.workflows:PwDosWorkChain":
        spec["inputs"] = {
            "pw_code_label": _get_default_pw_code(),
            "pseudo_family": PSEUDO_FAMILIES[0],
            "structure": None,
            "parameters": _build_scf_parameters(structure_type, optimization_level),
            "kpoints_distance": 0.15,
            "dos_kpoints_distance": 0.1,
        }
        spec["metadata"]["estimated_walltime_hours"] = 6

    elif workflow_type == "aiida.workflows:PwCalculation":
        spec["inputs"] = {
            "pw_code_label": _get_default_pw_code(),
            "pseudo_family": PSEUDO_FAMILIES[0],
            "structure": None,
            "parameters": _build_scf_parameters(structure_type, optimization_level),
            "kpoints_distance": 0.2,
        }
        spec["metadata"]["estimated_walltime_hours"] = 2

    elif workflow_type == "aiida.workflows:PhRelaxWorkChain":
        spec["inputs"] = {
            "pw_code_label": _get_default_pw_code(),
            "ph_code_label": "<no-ph-code-found>",
            "pseudo_family": PSEUDO_FAMILIES[0],
            "structure": None,
            "parameters": _build_relax_parameters(structure_type, optimization_level),
            "kpoints_distance": 0.2,
        }
        spec["metadata"]["estimated_walltime_hours"] = 12

    elif workflow_type in (
        "aiida.workflows:VaspWorkChain",
        "aiida.workflows:VaspRelaxWorkChain",
    ):
        encut = (
            520.0
            if structure_type == "metallic"
            else (400.0 if structure_type == "insulator" else 450.0)
        )
        if optimization_level == "high_accuracy":
            encut += 50.0
        vasp_params: dict[str, t.Any] = {
            "ENCUT": encut,
            "EDIFF": 1e-6 if optimization_level != "high_accuracy" else 1e-7,
            "PREC": "Accurate",
        }
        if "Relax" in workflow_type:
            vasp_params["IBRION"] = 2
            vasp_params["ISIF"] = 3
            vasp_params["EDIFFG"] = -0.01
        spec["inputs"] = {
            "code_label": "vasp-6.3",
            "potential_family": "PBE.54",
            "structure": None,
            "parameters": vasp_params,
            "kpoints_distance": 0.15 if optimization_level == "high_accuracy" else 0.2,
        }
        spec["metadata"]["estimated_walltime_hours"] = (
            6 if "Relax" in workflow_type else 3
        )

    logger.debug("generate_workflow_spec: returning spec for %s", workflow_type)

    return spec


def _build_relax_parameters(
    structure_type: str, optimization_level: str
) -> dict[str, t.Any]:
    """Build relaxation parameters based on structure type and accuracy level."""
    base_params: dict[str, t.Any] = {
        "system": {},
        "electrons": {"conv_thr": 1e-8},
        "ions": {"ion_dynamics": "bfgs"},
    }

    if structure_type == "metallic":
        ecutwfc = 70 if optimization_level == "high_accuracy" else 65
    elif structure_type == "insulator":
        ecutwfc = 60 if optimization_level == "high_accuracy" else 50
    else:  # semiconductor
        ecutwfc = 65 if optimization_level == "high_accuracy" else 55

    base_params["system"]["ecutwfc"] = ecutwfc
    base_params["system"]["ecutrho"] = ecutwfc * 8

    # Force-convergence parameters for geometry optimisation
    base_params["ions"]["ion_relax_maxsteps"] = 100
    base_params["ions"]["conv_thr_forces"] = 1e-4

    return base_params


def _build_scf_parameters(
    structure_type: str, optimization_level: str
) -> dict[str, t.Any]:
    """Build SCF parameters for band structure / DOS calculations."""
    base_params: dict[str, t.Any] = {
        "system": {},
        "electrons": {},
    }

    if structure_type == "metallic":
        ecutwfc = 70 if optimization_level == "high_accuracy" else 65
        base_params["electrons"]["conv_thr"] = (
            1e-9 if optimization_level == "high_accuracy" else 1e-8
        )
    elif structure_type == "insulator":
        ecutwfc = 60 if optimization_level == "high_accuracy" else 50
        base_params["electrons"]["conv_thr"] = 1e-8
    else:  # semiconductor
        ecutwfc = 65 if optimization_level == "high_accuracy" else 55
        base_params["electrons"]["conv_thr"] = 1e-8

    base_params["system"]["ecutwfc"] = ecutwfc
    base_params["system"]["ecutrho"] = ecutwfc * 8

    base_params["electrons"]["scf_maxiter"] = (
        100 if optimization_level == "standard" else 150
    )

    return base_params
