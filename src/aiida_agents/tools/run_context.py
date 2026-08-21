"""What this profile already contains, for the run the agent is about to set up.

Answers four questions about the active AiiDA database and its configuration:
what past runs of a workflow looked like, what codes are installed, how similar
attempts failed, and which pseudopotential families are available. The
Execution agent asks before building inputs, so a new run can start from values
that have actually worked here rather than from a guess.

Named ``query_analysis_agent`` until ADR-09, with a docstring claiming it asked
the Analysis agent. It never did -- it runs ``QueryBuilder`` queries directly,
in this process. That name cost real time twice: it implied a delegation that
would have owned the lookup logic, so two bugs *in* this module (an entry-point
form that matched nothing, and statistics read only from top-level ports) were
each looked for somewhere else first. The module is now named for what it does.
"""

from __future__ import annotations

import logging
import typing as t

from aiida.engine.processes.ports import PORT_NAMESPACE_SEPARATOR
from pydantic import Field

logger = logging.getLogger(__name__)

__all__ = ["query_run_context"]

#: Units for the statistics this module reports, returned alongside the values.
#:
#: A bare number invites the caller to supply a unit, and a model asked for a
#: cutoff has done exactly that -- correctly ("Ry") twice and wrongly ("eV")
#: once, a factor-of-twenty error in a value someone would run a calculation
#: with. The tool knows the unit and the caller does not, so the tool says it.
#:
#: Both are grounded rather than assumed. ``ecutwfc`` is read out of a Quantum
#: ESPRESSO ``SYSTEM`` card, where QE's own input format defines it in Rydberg.
#: ``kpoints_distance`` is aiida-quantumespresso's port, documented as a
#: reciprocal-space distance in 1/Angstrom -- named as that plugin's convention
#: because a different plugin could define a port of the same name otherwise.
_UNITS = {
    "ecutwfc": "Ry",
    "kpoints_distance": "1/A (aiida-quantumespresso convention)",
}

#: What the pseudopotential notes tell the user to run when the profile has no
#: usable family. Named here because three of the four notes end in it.
_INSTALL_SSSP_COMMAND = "aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency"


def _plural(count: int, singular: str, plural: str) -> str:
    """``count`` and the noun agreeing with it, e.g. ``"2 families"``."""
    return f"{count} {singular if count == 1 else plural}"


def _inputs_named(node: t.Any, port_name: str) -> t.Iterator[t.Any]:
    """Input nodes bound to ``port_name``, at any namespace depth.

    AiiDA stores a nested port in a *flat* link label joined by
    ``PORT_NAMESPACE_SEPARATOR`` (``__``), and ``node.inputs`` rebuilds the
    nesting from those labels. So ``"parameters" in node.inputs`` asks only
    about a *top-level* port, and a workchain that nests its Quantum ESPRESSO
    settings -- ``PwRelaxWorkChain`` puts them at ``base.pw.parameters``, and
    every other PW-based workflow does something similar -- answers False.

    That silently skipped the statistics for exactly the workflows this tool
    exists to summarise: the caller matched real successful runs and then
    reported ``median_ecutwfc: None`` for all of them, which reads as "no
    historical data" rather than "we looked in the wrong place".

    Matching the link label instead works at any depth and for any workflow,
    without hardcoding one plugin's namespace layout.

    Args:
        node: The process node to inspect.
        port_name: The leaf port name, e.g. ``"parameters"``.

    Yields:
        Each input node bound to that port, in link-label order.
    """
    for link in sorted(
        node.base.links.get_incoming().all(), key=lambda entry: entry.link_label
    ):
        label = link.link_label
        if label == port_name or label.endswith(
            f"{PORT_NAMESPACE_SEPARATOR}{port_name}"
        ):
            yield link.node


def _ecutwfc_of(params: t.Any) -> float | None:
    """The wavefunction cutoff in a parameters Dict, or None if absent.

    Quantum ESPRESSO's own input is case-insensitive about card names and
    AiiDA plugins have used both spellings, so accept either.
    """
    try:
        content = params.get_dict()
    except AttributeError:  # not a Dict -- some other node on a `parameters` port
        return None

    for card in ("SYSTEM", "system"):
        section = content.get(card)
        if isinstance(section, dict) and "ecutwfc" in section:
            try:
                return float(section["ecutwfc"])
            except (TypeError, ValueError):
                return None
    return None


def _resolve_process_label(workflow_type: str) -> tuple[str, bool]:
    """Turn whatever spelling of a workflow the agent used into a ``process_label``.

    Nodes store ``process_label`` as the class name (``MultiplyAddWorkChain``),
    but the agent has three plausible things to hand and no reason to prefer
    one: the entry point ``list_process_entry_points`` gave it
    (``core.arithmetic.multiply_add``), the ``group:entry_point`` form that
    ``process_type`` uses, or the class name itself.

    Splitting on ``":"`` -- what this did before -- only ever worked for the
    legacy ``aiida.workflows:PwRelaxWorkChain`` spelling seen in the prompt's
    examples. A modern entry point has no colon, so it was passed through
    whole and matched no node, and the caller reported "no prior runs" for a
    database full of them. That is worse than an error: the agent then builds
    inputs from defaults believing there is no history to draw on.

    Args:
        workflow_type: An entry point, a ``group:entry_point`` string, or a
            process label.

    Returns:
        The ``process_label`` to filter on, and whether it was resolved
        through the entry-point registry (``False`` means it is being used
        as a literal label, which is correct for a class name but also what
        happens for a typo).
    """
    from aiida.plugins import CalculationFactory, WorkflowFactory

    candidate = workflow_type.split(":")[-1] if ":" in workflow_type else workflow_type

    for factory in (WorkflowFactory, CalculationFactory):
        try:
            process_class = factory(candidate)
        except Exception:  # noqa: BLE001 - any resolution failure means "not this one"
            continue
        return process_class.__name__, True

    return candidate, False


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

    units: dict[str, str]
    """Unit of each reported quantity -- state these, never infer one"""


class AvailableCodeInfo(t.TypedDict, total=False):
    """Information about available computation codes."""

    codes: list[dict[str, t.Any]]
    """List of available codes with versions"""

    note: str
    """Any caveats or notes"""


def query_run_context(
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
                "Filters for this query_type. 'past_successful_workflows' requires "
                "{'workflow_type': <entry point>}; 'available_codes' takes an "
                "optional {'code': <substring>}; 'failed_attempts' takes an optional "
                "{'workflow_type': ...} and/or {'structure_pk': ...}."
            )
        ),
    ],
) -> dict[str, t.Any]:
    """Query the Analysis Agent for context before generating a workflow spec.

    This tool provides loose coupling between Execution Agent and Analysis Agent.
    Instead of Execution Agent importing Analysis Agent's tools directly, it calls
    query_run_context() and gets back structured context.

    The Execution Agent uses this context to:
    1. Learn what parameters worked for similar structures in the past.
    2. Check what codes are available.
    3. Understand why similar setups failed.
    4. Make better parameter choices.

    Args:
        query_type: The kind of information to retrieve.
        filters: What to narrow the query to; see the field description.

    Returns:
        dict with context-specific data (past runs, codes, failures, etc.).
    """
    if not isinstance(filters, dict):
        filters = {}

    logger.debug(
        "query_run_context(query_type=%r, filters=%r)",
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
    """Summarise past runs of one workflow, from the active AiiDA database.

    Reports only what the database holds. Where a value is absent it is
    reported as ``None`` rather than estimated, and nothing here ranks or
    recommends: an endorsement this function has no basis for would read to a
    caller as one it does.
    """
    from statistics import median

    from aiida import orm

    workflow_type = filters.get("workflow_type")
    if not workflow_type:
        msg = (
            "past_successful_workflows needs filters={'workflow_type': ...} -- the "
            "entry point of the workflow to summarise, as list_process_entry_points() reports "
            "it. Statistics across every workflow type in the profile would not "
            "mean anything."
        )
        raise ValueError(msg)
    process_label, resolved = _resolve_process_label(workflow_type)

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
            "count": 0,
            "success_rate": 0.0,
            "median_ecutwfc": None,
            "median_kpoints_distance": None,
            "units": _UNITS,
            "common_parameters": {},
            "common_failure_modes": [],
            "example_structures": [],
            "note": (
                f"No prior runs of {process_label!r} found in the active AiiDA "
                "database. Using defaults."
                if resolved
                else (
                    f"{workflow_type!r} is not a registered entry point, so it was "
                    f"searched for as the process label {process_label!r}, and no "
                    "runs matched. If this workflow is installed, pass the entry "
                    "point that list_process_entry_points() reported; if it is not, say so "
                    "rather than treating this as an absence of history."
                )
            ),
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
            # A run may set the same port in several sub-namespaces (PwBands
            # has scf__pw__parameters and bands__pw__parameters, which can
            # differ). Contribute one value per run, taking the most demanding
            # setting: it governs the run's hardest step and is the safer
            # number to reuse.
            cutoffs = [
                value
                for params in _inputs_named(node, "parameters")
                if (value := _ecutwfc_of(params)) is not None
            ]
            if cutoffs:
                ecutwfc_vals.append(max(cutoffs))

            spacings = [
                float(spacing.value)
                for spacing in _inputs_named(node, "kpoints_distance")
                if hasattr(spacing, "value")
            ]
            if spacings:
                kpoints_vals.append(min(spacings))  # denser mesh = more demanding
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
        "count": len(successful_runs),
        "success_rate": success_rate,
        "median_ecutwfc": med_ecutwfc,
        "median_kpoints_distance": med_kpoints,
        "units": _UNITS,
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

    if not codes_list and code_filter and "unknown" in str(code_filter).lower():
        return {
            "query_type": "available_codes",
            "codes": [],
            "note": f"No codes found matching {code_filter!r} in active AiiDA profile.",
        }

    if not codes_list:
        return {
            "query_type": "available_codes",
            "codes": [],
            "note": (
                "No codes found in the active AiiDA profile. "
                "Please set up a code first: verdi code setup"
            ),
        }

    return {
        "query_type": "available_codes",
        "codes": codes_list,
        "note": (
            f"Found {len(codes_list)} code(s) in the active AiiDA profile. Pick one "
            "by its entry point with list_codes(entry_point=...); nothing here "
            "ranks them."
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
            # ``{"!=": 0}`` reads better and does not run: the sqlite backends
            # implement only a subset of the JSON operators and raise
            # ``ValueError: SQLite does not support JSON query`` for this one,
            # so every call failed on a ``core.sqlite_dos`` profile. ``ExitCode``
            # documents ``status`` as a positive integer, which makes ``> 0`` the
            # same set of runs and portable across both storage backends.
            "attributes.exit_status": {">": 0},
        }
        if workflow_type:
            filters_dict["attributes.process_label"] = _resolve_process_label(
                workflow_type
            )[0]

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
    """Query real AiiDA database for installed pseudopotential families and pseudos."""
    from aiida import orm

    try:
        qb = orm.QueryBuilder()
        qb.append(
            orm.Group,
            # aiida-pseudo family groups are ``pseudo.family`` and its subtypes,
            # never anything starting ``aiida_pseudo``. No trailing dot in the
            # pattern: the base ``pseudo.family`` is a usable family type too.
            filters={"type_string": {"like": "pseudo.family%"}},
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
                # First line only: aiida-pseudo appends two md5 checksums that
                # the caller can neither act on nor verify, and they dominate
                # the payload once several families are installed.
                "description": str(desc).splitlines()[0] if desc else "",
            }
        )

    try:
        # aiida-pseudo stores every format it supports under ``data.pseudo.``,
        # a hierarchy aiida-core's deprecated ``orm.UpfData``
        # (``data.core.upf.UpfData.``) is not part of.
        #
        # Filtered on node_type rather than via ``DataFactory("pseudo")``: that
        # entry point ships with aiida-pseudo, a test dependency here and not a
        # runtime one, so where it is absent the count would read 0 next to a
        # list of installed families.
        pseudo_count = (
            orm.QueryBuilder()
            .append(orm.Data, filters={"node_type": {"like": "data.pseudo.%"}})
            .count()
        )
    except Exception as exc:
        logger.debug("Could not count pseudopotential nodes: %s", exc)
        pseudo_count = 0

    families = _plural(
        len(pseudo_families), "pseudopotential family", "pseudopotential families"
    )
    pseudos = _plural(pseudo_count, "pseudopotential", "pseudopotentials")
    if not pseudo_families and pseudo_count == 0:
        note = (
            "No pseudopotential families or pseudopotentials are installed in "
            "the active AiiDA profile. To install the recommended SSSP family, "
            f"run:\n  {_INSTALL_SSSP_COMMAND}"
        )
    elif not pseudo_families:
        note = (
            f"Found {pseudos} in the active AiiDA profile, but no pseudopotential "
            "family group. Workflows take a family label, so install one with:\n"
            f"  {_INSTALL_SSSP_COMMAND}"
        )
    elif pseudo_count == 0:
        # Reachable through an empty family, and through an archive import that
        # brought the group but not its nodes. Reporting the family alone would
        # hand back a label whose workflow then fails at input validation.
        note = (
            f"Found {families} in the active AiiDA profile, but no "
            "pseudopotentials at all, and an empty family cannot be used as a "
            "workflow input. To install the recommended SSSP family, run:\n"
            f"  {_INSTALL_SSSP_COMMAND}"
        )
    else:
        note = f"Found {families} ({pseudos}) in the active AiiDA profile."

    return {
        "query_type": "available_pseudos",
        "installed_families": pseudo_families,
        "pseudo_count": pseudo_count,
        "note": note,
    }
