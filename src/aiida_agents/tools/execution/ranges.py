"""Check a spec's physics parameters against what the pseudopotentials expect.

ADR-07 built the schema tier and deferred this one. Since then every write has
been guarded by code except the question that decides whether the calculation
is worth running: a cutoff of 12 Ry passes schema validation, passes
``spec.inputs.validate()``, submits cleanly, and produces a converged,
authoritative, wrong number. The only thing standing between that and an HPC
queue was a human reading a diff and knowing the right value by heart.

There is an authority for that value, and it is not this package. A
pseudopotential family ships the cutoffs its authors converged it against, and
``aiida-pseudo`` exposes them per element. So the check compares against what
the family recommends for *this structure's elements* and reports the
difference; it does not carry a table of its own, and where no family can be
identified it reports that rather than guessing a number.

**This warns, it does not block.** A low cutoff is a legitimate choice for a
smoke test and a high one for a converged production run, so refusing the
submission would be wrong about as often as it was right. What was actually
missing is the fact reaching the person approving, at the moment they approve.
The finding is therefore surfaced at the approval prompt and offered to the
agent as a tool --- the same belt-and-braces the grounding check uses, where
the prompt asks the model to check and the code checks regardless.

**Only cutoffs, deliberately.** ``kpoints_distance`` has no equivalent
authority: no installed package ships a recommended k-spacing per structure,
so any bound here would be a number this file invented, presented with the
authority of a validator. That is the failure mode the whole project is built
against, so it is left alone and said to be left alone.
"""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from pydantic import Field
from typing_extensions import TypedDict

from aiida_agents.tools.execution._spec import is_reference, resolve_reference
from aiida_agents.tools.execution.schemas import SubmissionSpec

logger = logging.getLogger(__name__)

__all__ = ["RangeFinding", "check_cutoffs_against_pseudos"]

# Quantum ESPRESSO's input is case-insensitive about card names and AiiDA
# plugins have used both spellings; ``run_context`` accepts either for the same
# reason, and the two must not disagree about where a cutoff lives.
_SYSTEM_CARDS = ("SYSTEM", "system")

# The cutoff keys, and the aiida-pseudo accessor each is compared against.
_CUTOFF_KEYS = ("ecutwfc", "ecutrho")

# Fraction of the recommended cutoff below which a value is called out. Not a
# physics constant: a value at 95% of the recommendation is not worth
# interrupting an approval for, and one at half of it is.
_LOW_FRACTION = 0.9

# Multiple of the recommended cutoff above which a value is called out as
# unusually expensive. Cost in a plane-wave code grows steeply with the cutoff,
# so a 3x cutoff is a real bill, not a rounding error.
_HIGH_MULTIPLE = 3.0


class RangeFinding(TypedDict):
    """One parameter that differs from what its pseudopotentials expect.

    ``recommended`` and ``source`` are always populated together: a finding
    exists because an authority disagreed with the spec, and naming the
    authority is what separates this from an opinion.
    """

    parameter: str
    """Dotted path to the value in the spec, e.g. ``'base.pw.parameters.SYSTEM.ecutwfc'``."""

    value: float
    """What the spec proposes."""

    recommended: float
    """What the pseudopotential family recommends for these elements."""

    unit: str
    """Unit of both, as reported by the family."""

    severity: t.Literal["below_recommended", "far_above_recommended"]
    """Which direction, and therefore which risk: wrong answer, or wasted time."""

    source: str
    """The family and elements the recommendation came from."""

    message: str
    """A one-line statement of the difference, ready to show a user."""


def _walk(inputs: t.Any, path: str = "") -> t.Iterator[tuple[str, t.Any]]:
    """Yield every ``(dotted_path, value)`` in a nested mapping."""
    if not isinstance(inputs, dict):
        return
    for key, value in inputs.items():
        here = f"{path}.{key}" if path else str(key)
        yield here, value
        yield from _walk(value, here)


def _find_leaf(inputs: dict[str, t.Any], name: str) -> t.Iterator[tuple[str, t.Any]]:
    """Every value bound to a leaf called ``name``, at any depth.

    Matching the leaf rather than a fixed path is the same choice
    ``run_context._inputs_named`` makes, and for the same reason: a cutoff
    lives at ``base.pw.parameters`` on a ``PwRelaxWorkChain`` and somewhere
    else on the next workflow, and hardcoding one plugin's layout silently
    skips every other.
    """
    for path, value in _walk(inputs):
        if path.split(".")[-1] == name:
            yield path, value


def _cutoffs_in(inputs: dict[str, t.Any]) -> dict[str, tuple[str, float]]:
    """Cutoffs found anywhere in the spec, keyed by name.

    Returns the first of each: a spec that sets ``ecutwfc`` in two namespaces
    is describing two sub-calculations, and reporting the same finding twice
    would clutter an approval prompt without adding anything.
    """
    found: dict[str, tuple[str, float]] = {}
    for params_path, params in _find_leaf(inputs, "parameters"):
        if not isinstance(params, dict):
            continue
        for card in _SYSTEM_CARDS:
            section = params.get(card)
            if not isinstance(section, dict):
                continue
            for key in _CUTOFF_KEYS:
                if key in found or key not in section:
                    continue
                try:
                    found[key] = (f"{params_path}.{card}.{key}", float(section[key]))
                except (TypeError, ValueError):
                    logger.debug("non-numeric %s at %s", key, params_path)
    return found


def _structure_in(inputs: dict[str, t.Any]) -> orm.StructureData | None:
    """The structure the spec will run on, resolved from its reference."""
    for _, value in _find_leaf(inputs, "structure"):
        if is_reference(value):
            try:
                node = resolve_reference(value, "structure")
            except ValueError:
                continue
            if isinstance(node, orm.StructureData):
                return node
        if isinstance(value, orm.StructureData):
            return value
    return None


def _family_in(inputs: dict[str, t.Any]) -> t.Any | None:
    """The pseudopotential family the spec names, if it names one.

    Two conventions are in use: a ``pseudo_family`` label, and a ``pseudos``
    namespace holding the pseudopotential nodes themselves. The label is
    checked first because it identifies the family directly; the nodes only
    identify it by the group they belong to.
    """
    for _, value in _find_leaf(inputs, "pseudo_family"):
        label = value.value if isinstance(value, orm.Str) else value
        if isinstance(label, str):
            try:
                return orm.load_group(label)
            except Exception as exc:  # noqa: BLE001 - a label that names no group
                logger.debug("pseudo_family %r did not load: %s", label, exc)

    for _, value in _find_leaf(inputs, "pseudos"):
        if not isinstance(value, dict):
            continue
        for entry in value.values():
            node = entry
            if is_reference(entry):
                try:
                    node = resolve_reference(entry, "pseudos")
                except ValueError:
                    continue
            groups = _groups_of(node)
            if groups:
                return groups[0]
    return None


def _groups_of(node: t.Any) -> list[t.Any]:
    """Pseudopotential families a node belongs to, newest first."""
    if not isinstance(node, orm.Data):
        return []
    try:
        builder = orm.QueryBuilder()
        builder.append(orm.Node, filters={"id": node.pk}, tag="pseudo")
        builder.append(orm.Group, with_node="pseudo")
        return [group for (group,) in builder.all() if hasattr(group, "get_cutoffs")]
    except Exception as exc:  # noqa: BLE001 - a query failure is not a finding
        logger.debug(
            "group lookup for pk=%s failed: %s", getattr(node, "pk", None), exc
        )
        return []


def _recommended(
    family: t.Any, structure: orm.StructureData
) -> tuple[dict[str, float], str] | None:
    """The family's recommended cutoffs for this structure, and their unit."""
    try:
        wfc, rho = family.get_recommended_cutoffs(structure=structure)
        unit = family.get_cutoffs_unit()
    except Exception as exc:  # noqa: BLE001 - an element the family does not cover
        logger.debug("no recommended cutoffs from %r: %s", family, exc)
        return None
    return {"ecutwfc": float(wfc), "ecutrho": float(rho)}, str(unit)


def check_cutoffs_against_pseudos(
    spec: t.Annotated[
        SubmissionSpec,
        Field(
            description=(
                "The SubmissionSpec to check, exactly as it would go to "
                "submit_process_spec. It must name a structure and a "
                "pseudopotential family (or carry resolved pseudos), since the "
                "recommendation is per element."
            )
        ),
    ],
) -> list[RangeFinding]:
    """Check a spec's cutoffs against what its pseudopotentials were converged for.

    Call this before ``submit_process_spec`` whenever you have set or changed
    a cutoff --- particularly one you took from historical runs, which may have
    used a different pseudopotential family than this spec does.

    An empty list means either that nothing disagreed or that nothing could be
    compared: no cutoffs in the spec, no structure, or no identifiable family.
    It does **not** mean the parameters were verified. Say which of the two you
    are reporting, and never present an empty result as a clean bill of health.

    A finding is a fact about a disagreement, not a verdict. ``below_recommended``
    means the calculation may not be converged --- the expensive kind of wrong,
    because it completes and produces a plausible number. ``far_above_recommended``
    means it will cost considerably more than the family's authors found
    necessary. Report the finding with its ``source``, and let the user decide;
    a deliberately cheap smoke test is a legitimate thing to run.

    Only cutoffs are checked. There is no authority for a recommended k-point
    spacing, so this reports nothing about ``kpoints_distance`` rather than
    inventing a bound for it.

    Args:
        spec: The process spec about to be submitted.

    Returns:
        One finding per cutoff that differs materially from the recommendation,
        empty when nothing does or nothing could be compared.
    """
    inputs = spec.get("inputs") if isinstance(spec, dict) else None
    if not isinstance(inputs, dict):
        return []

    cutoffs = _cutoffs_in(inputs)
    if not cutoffs:
        return []

    structure = _structure_in(inputs)
    family = _family_in(inputs)
    if structure is None or family is None:
        logger.debug(
            "cutoffs present but not comparable (structure=%s, family=%s)",
            structure is not None,
            family is not None,
        )
        return []

    resolved = _recommended(family, structure)
    if resolved is None:
        return []
    recommended, unit = resolved

    elements = ", ".join(sorted(structure.get_symbols_set()))  # type: ignore[no-untyped-call]
    source = f"{family.label} recommended cutoffs for {elements}"

    findings: list[RangeFinding] = []
    for key, (path, value) in sorted(cutoffs.items()):
        expected = recommended.get(key)
        if not expected:
            continue
        if value < expected * _LOW_FRACTION:
            severity: t.Literal["below_recommended", "far_above_recommended"] = (
                "below_recommended"
            )
            message = (
                f"{key} is {value:g} {unit}, below the {expected:g} {unit} "
                f"{source} — the calculation may not be converged."
            )
        elif value > expected * _HIGH_MULTIPLE:
            severity = "far_above_recommended"
            message = (
                f"{key} is {value:g} {unit}, more than {_HIGH_MULTIPLE:g}x the "
                f"{expected:g} {unit} {source} — this will be expensive."
            )
        else:
            continue

        findings.append(
            RangeFinding(
                parameter=path,
                value=value,
                recommended=expected,
                unit=unit,
                severity=severity,
                source=source,
                message=message,
            )
        )

    return findings
