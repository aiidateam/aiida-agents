"""Interface to Analysis Agent capabilities.

Execution Agent calls query_analysis_agent() to ask Analysis Agent about
past successful runs, available codes, failure patterns, etc. This decouples
Execution Agent from Analysis Agent's implementation.
"""

from __future__ import annotations

import logging
import typing as t

from pydantic import Field

logger = logging.getLogger(__name__)

__all__ = ["query_analysis_agent"]


class PastWorkflowSummary(t.TypedDict, total=False):
    """Summary of past successful workflows of a certain type."""

    count: int
    """Number of successful runs"""

    median_ecutwfc: float | None
    """Median ecutwfc value used"""

    median_kpoints_distance: float | None
    """Median k-points spacing"""

    success_rate: float
    """Percentage of runs that succeeded"""

    common_parameters: dict[str, t.Any]
    """Most common parameter values"""

    common_failure_modes: list[str]
    """What typically goes wrong"""

    example_structures: list[str]
    """Sample structure formulas"""


class AvailableCodeInfo(t.TypedDict, total=False):
    """Information about available computation codes."""

    codes: list[dict[str, t.Any]]
    """List of available codes with versions"""

    recommended_version: str | None
    """Which version is recommended"""

    note: str
    """Any caveats or notes"""


def query_analysis_agent(
    query_type: t.Annotated[
        str,
        Field(
            description=(
                "What to ask Analysis Agent. Options: "
                "'past_successful_workflows', 'available_codes', "
                "'failed_attempts', 'available_pseudos'"
            )
        ),
    ],
    filters: t.Annotated[
        dict[str, t.Any],
        Field(
            description=(
                "Context-specific filters (e.g. structure_type, composition, workflow_type). "
                "Note: 'structure_type' is not a database-level filter — AiiDA stores no "
                "per-node material-class attribute. It is echoed back as metadata only."
            )
        ),
    ],
) -> dict[str, t.Any]:
    """Query the Analysis Agent for context before generating a workflow spec.

    This tool provides loose coupling between Execution Agent and Analysis Agent.
    Instead of Execution Agent importing Analysis Agent's tools directly, it calls
    query_analysis_agent() and gets back structured context.

    The Execution Agent uses this context to:
    1. Learn what parameters worked for similar structures in the past.
    2. Check what codes are available.
    3. Understand why similar setups failed.
    4. Make better parameter choices.

    Args:
        query_type: The kind of information to retrieve.
        filters: Context that narrows the query (workflow_type, structure_type, etc.).

    Returns:
        dict with context-specific data (past runs, codes, failures, etc.).
    """
    if not isinstance(filters, dict):
        filters = {}

    logger.debug(
        "query_analysis_agent(query_type=%r, filters=%r)",
        query_type,
        filters,
    )

    if query_type == "past_successful_workflows":
        return _query_past_workflows(filters)

    if query_type == "available_codes":
        return _query_available_codes(filters)

    if query_type == "failed_attempts":
        return _query_failed_attempts(filters)

    if query_type in ("available_pseudos", "installed_pseudos", "pseudo_families"):
        return _query_available_pseudos(filters)

    msg = (
        f"Unknown query_type: {query_type!r}. "
        "Try 'past_successful_workflows', 'available_codes', "
        "'failed_attempts', or 'available_pseudos'"
    )
    raise ValueError(msg)


# ============================================================================
# Real AiiDA ORM Introspection Functions
# ============================================================================


def _query_past_workflows(filters: dict[str, t.Any]) -> dict[str, t.Any]:
    """Query real AiiDA database for completed workflow statistics.

    Note: ``structure_type`` in *filters* is not applied as a database predicate.
    AiiDA stores no per-node material-class attribute, so all finished workflows
    of the requested type are included regardless of ``structure_type``.
    The value is echoed back in the return dict as metadata.
    """
    from aiida import orm
    from statistics import median

    workflow_type = filters.get("workflow_type", "aiida.workflows:PwRelaxWorkChain")
    structure_type = filters.get("structure_type", "metallic")
    process_label = (
        workflow_type.split(":")[-1] if ":" in workflow_type else workflow_type
    )

    try:
        qb = orm.QueryBuilder()
        qb.append(
            orm.WorkChainNode,
            filters={
                "attributes.process_label": process_label,
                "attributes.process_state": "finished",
            },
            project=["attributes.exit_status", "id", "uuid"],
        )
        records = qb.all()
    except Exception as exc:
        logger.debug("Could not execute WorkChainNode query: %s", exc)
        records = []

    total_runs = len(records)
    if total_runs == 0:
        return {
            "query_type": "past_successful_workflows",
            "workflow_type": workflow_type,
            "structure_type": structure_type,
            "structure_type_filter_note": (
                "structure_type is not filterable at the database level; "
                "all finished workflows of this type are included."
            ),
            "count": 0,
            "success_rate": 0.0,
            "median_ecutwfc": None,
            "median_kpoints_distance": None,
            "common_parameters": {},
            "common_failure_modes": [],
            "example_structures": [],
            "note": "No prior runs of this workflow found in the active AiiDA database. Using defaults.",
        }

    successful_runs = [r for r in records if r[0] == 0]
    failed_runs = [r for r in records if r[0] != 0]
    success_rate = round(len(successful_runs) / total_runs, 2) if total_runs else 0.0

    ecutwfc_vals: list[float] = []
    kpoints_vals: list[float] = []
    example_structs: list[str] = []
    for row in successful_runs[:20]:
        try:
            node = orm.load_node(row[1])
            if "structure" in node.inputs:
                try:
                    formula = node.inputs.structure.get_formula()
                    if formula not in example_structs:
                        example_structs.append(formula)
                except Exception as exc:
                    logger.debug("Could not read formula for node %s: %s", row[1], exc)
            if "parameters" in node.inputs:
                params_dict = node.inputs.parameters.get_dict()
                if "SYSTEM" in params_dict and "ecutwfc" in params_dict["SYSTEM"]:
                    ecutwfc_vals.append(float(params_dict["SYSTEM"]["ecutwfc"]))
                elif "system" in params_dict and "ecutwfc" in params_dict["system"]:
                    ecutwfc_vals.append(float(params_dict["system"]["ecutwfc"]))
            if "kpoints_distance" in node.inputs:
                kpoints_vals.append(float(node.inputs.kpoints_distance.value))
        except Exception as exc:
            logger.debug("Could not inspect parameters for node %s: %s", row[1], exc)

    med_ecutwfc = median(ecutwfc_vals) if ecutwfc_vals else None
    med_kpoints = median(kpoints_vals) if kpoints_vals else None

    failure_modes: list[str] = []
    for row in failed_runs[:10]:
        try:
            fnode = orm.load_node(row[1])
            if fnode.exit_message:
                failure_modes.append(f"Exit status {row[0]}: {fnode.exit_message}")
            else:
                failure_modes.append(f"Exit status {row[0]}")
        except Exception as exc:
            logger.debug("Could not load failed node %s: %s", row[1], exc)

    return {
        "query_type": "past_successful_workflows",
        "workflow_type": workflow_type,
        "structure_type": structure_type,
        "structure_type_filter_note": (
            "structure_type is not filterable at the database level; "
            "all finished workflows of this type are included."
        ),
        "count": len(successful_runs),
        "success_rate": success_rate,
        "median_ecutwfc": med_ecutwfc,
        "median_kpoints_distance": med_kpoints,
        "common_parameters": {"ecutwfc": med_ecutwfc} if med_ecutwfc else {},
        "common_failure_modes": list(set(failure_modes))[:3],
        "example_structures": example_structs,
        "note": f"Analyzed {total_runs} historical workflow(s) in active database ({len(successful_runs)} successful).",
    }


def _query_available_codes(filters: dict[str, t.Any]) -> dict[str, t.Any]:
    """Query real AiiDA profile for available/installed codes."""
    from aiida import orm

    try:
        qb = orm.QueryBuilder()
        qb.append(
            orm.Code,
            project=["label", "attributes.input_plugin", "attributes.description"],
        )
        results = qb.all()
    except Exception as exc:
        logger.debug("Could not execute Code query: %s", exc)
        results = []

    code_filter = (
        filters.get("code") or filters.get("plugin") or filters.get("code_label")
    )
    codes_list: list[dict[str, str]] = []
    recommended: str | None = None
    for row in results:
        label, plugin, desc = row[0], row[1], row[2]
        if code_filter and (
            str(code_filter).lower() not in str(label).lower()
            and str(code_filter).lower() not in str(plugin).lower()
        ):
            continue
        codes_list.append(
            {
                "label": str(label),
                "plugin": str(plugin) if plugin else "unknown",
                "description": str(desc) if desc else "",
            }
        )
        if not recommended or "pw" in str(label).lower():
            recommended = str(label)

    if not codes_list and code_filter and "unknown" in str(code_filter).lower():
        return {
            "query_type": "available_codes",
            "codes": [],
            "recommended_version": None,
            "note": f"No codes found matching {code_filter!r} in active AiiDA profile.",
        }

    if not codes_list:
        return {
            "query_type": "available_codes",
            "codes": [],
            "recommended_version": None,
            "note": (
                "No codes found in the active AiiDA profile. "
                "Please set up a code first: verdi code setup"
            ),
        }

    return {
        "query_type": "available_codes",
        "codes": codes_list,
        "recommended_version": recommended,
        "note": (
            f"Found {len(codes_list)} available code(s) matching filter in active AiiDA profile."
            if results
            else "No matching codes found in active AiiDA profile. Providing schema recommendation."
        ),
    }


def _query_failed_attempts(filters: dict[str, t.Any]) -> dict[str, t.Any]:
    """Query real AiiDA database for failed attempts on a specific structure or workflow."""
    from aiida import orm

    structure_pk = filters.get("structure_pk") or filters.get("structure")
    workflow_type = filters.get("workflow_type")

    try:
        qb = orm.QueryBuilder()
        filters_dict: dict[str, t.Any] = {
            "attributes.process_state": "finished",
            "attributes.exit_status": {"!=": 0},
        }
        if workflow_type:
            process_label = (
                workflow_type.split(":")[-1] if ":" in workflow_type else workflow_type
            )
            filters_dict["attributes.process_label"] = process_label

        qb.append(
            orm.WorkChainNode,
            filters=filters_dict,
            project=["id", "attributes.exit_status", "attributes.exit_message"],
        )
        records = qb.all()
    except Exception as exc:
        logger.debug("Could not execute failed attempts query: %s", exc)
        records = []

    attempts: list[dict[str, t.Any]] = []
    for row in records[:10]:
        pk, status, msg = row[0], row[1], row[2]
        if structure_pk:
            try:
                node = orm.load_node(pk)
                if "structure" in node.inputs and str(node.inputs.structure.pk) != str(
                    structure_pk
                ):
                    continue
            except Exception as exc:
                logger.debug("Could not load node %s for structure filter: %s", pk, exc)
                continue
        attempts.append(
            {
                "pk": pk,
                "exit_code": status,
                "failure_reason": msg or f"Failed with exit status {status}",
            }
        )

    return {
        "query_type": "failed_attempts",
        "structure_pk": structure_pk,
        "workflow_type": workflow_type,
        "attempts": attempts,
        "suggestion": (
            f"Found {len(attempts)} failed attempt(s) matching criteria in active database."
            if attempts
            else "No failed attempts matching criteria found in active database."
        ),
        "note": "Real introspection of failed WorkChainNode records.",
    }


def _query_available_pseudos(filters: dict[str, t.Any]) -> dict[str, t.Any]:
    """Query real AiiDA database for installed pseudopotential families (UpfData groups)."""
    from aiida import orm

    try:
        qb = orm.QueryBuilder()
        qb.append(
            orm.Group,
            filters={"type_string": {"like": "aiida_pseudo%"}},
            project=["label", "type_string", "description"],
        )
        all_groups = qb.all()
    except Exception as exc:
        logger.debug("Could not query Groups for pseudopotentials: %s", exc)
        all_groups = []

    pseudo_families: list[dict[str, t.Any]] = []
    for label, type_str, desc in all_groups:
        pseudo_families.append(
            {
                "label": str(label),
                "type_string": str(type_str),
                "description": str(desc) if desc else "",
            }
        )

    try:
        upf_count = orm.QueryBuilder().append(orm.UpfData).count()
    except Exception as exc:
        logger.debug("Could not count UpfData nodes: %s", exc)
        upf_count = 0

    if not pseudo_families and upf_count == 0:
        note = (
            "No pseudopotential families or UpfData nodes are currently installed "
            "in the active AiiDA profile. To install the recommended SSSP family, run:\n"
            "  aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency"
        )
    elif not pseudo_families:
        note = f"Found {upf_count} UpfData node(s), but no named pseudo family groups."
    else:
        note = f"Found {len(pseudo_families)} pseudopotential family/families installed in active AiiDA profile."

    return {
        "query_type": "available_pseudos",
        "installed_families": pseudo_families,
        "upf_data_count": upf_count,
        "recommended_family": pseudo_families[0]["label"]
        if pseudo_families
        else "SSSP/1.3/PBE/efficiency (needs installation)",
        "note": note,
    }
