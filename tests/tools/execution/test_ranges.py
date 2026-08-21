"""Tests for ``check_cutoffs_against_pseudos``.

Built on a real ``SsspFamily`` rather than a stub, because the whole claim of
this check is that the number it compares against comes from an authority
rather than from this package. A mocked ``get_recommended_cutoffs`` would test
that the comparison arithmetic works while quietly removing the only part worth
doubting --- that aiida-pseudo is being asked correctly, for the right
elements, in the right unit.

The family is assembled locally from a one-line UPF and given cutoffs by hand,
so it exercises the real API without downloading several hundred megabytes of
pseudopotentials. Silicon is given ``cutoff_wfc: 30 Ry`` / ``cutoff_rho:
240 Ry`` --- close enough to SSSP's real values that the fixture reads like
the thing it stands in for.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import pytest
from aiida import orm
from aiida_pseudo.groups.family.sssp import SsspFamily

from aiida_agents.tools.execution.ranges import check_cutoffs_against_pseudos
from aiida_agents.tools.execution.schemas import SubmissionSpec

FAMILY_LABEL = "SSSP/1.3/PBE/efficiency"
RECOMMENDED_WFC = 30.0
RECOMMENDED_RHO = 240.0

# Enough of a UPF for aiida-pseudo to parse an element out of the filename and
# store it; none of the physics is read by anything under test.
_UPF = """<UPF version="2.0.1">
<PP_HEADER element="Si" pseudo_type="NC" z_valence="4.0"/>
</UPF>
"""


@pytest.fixture(scope="module")
def pseudo_family(tmp_path_factory: pytest.TempPathFactory) -> SsspFamily:
    """A real SSSP-shaped family carrying silicon's recommended cutoffs."""
    dirpath: Path = tmp_path_factory.mktemp("pseudos")
    (dirpath / "Si.upf").write_text(_UPF)
    family = SsspFamily.create_from_folder(dirpath, FAMILY_LABEL)
    family.set_cutoffs(
        {"Si": {"cutoff_wfc": RECOMMENDED_WFC, "cutoff_rho": RECOMMENDED_RHO}},
        "efficiency",
        unit="Ry",
    )
    return family


def _spec(
    structure: orm.StructureData,
    ecutwfc: t.Any = RECOMMENDED_WFC,
    *,
    nested: bool = False,
    family_label: str | None = FAMILY_LABEL,
    card: str = "SYSTEM",
) -> SubmissionSpec:
    """A spec in the shape ``submit_process_spec`` accepts."""
    parameters = {card: {"ecutwfc": ecutwfc, "ecutrho": RECOMMENDED_RHO}}
    inner: dict[str, t.Any] = {"parameters": parameters}
    inputs: dict[str, t.Any] = {"structure": {"pk": structure.pk}}
    if family_label is not None:
        inputs["pseudo_family"] = family_label
    if nested:
        inputs["base"] = {"pw": inner}
    else:
        inputs.update(inner)
    spec: SubmissionSpec = {
        "entry_point": "quantumespresso.pw.relax",
        "inputs": inputs,
    }
    return spec


class TestFindingsAgainstTheFamily:
    """The comparison itself, against a real recommendation."""

    def test_a_cutoff_below_the_recommendation_is_reported(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        findings = check_cutoffs_against_pseudos(_spec(silicon_structure, ecutwfc=12.0))

        wfc = next(f for f in findings if f["parameter"].endswith("ecutwfc"))
        assert wfc["severity"] == "below_recommended"
        assert wfc["value"] == 12.0
        assert wfc["recommended"] == RECOMMENDED_WFC
        assert wfc["unit"] == "Ry"

    def test_the_finding_names_the_family_and_elements_it_came_from(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        """Without the source it is an opinion, not a check."""
        findings = check_cutoffs_against_pseudos(_spec(silicon_structure, ecutwfc=12.0))

        wfc = next(f for f in findings if f["parameter"].endswith("ecutwfc"))
        assert FAMILY_LABEL in wfc["source"]
        assert "Si" in wfc["source"]
        assert "12" in wfc["message"] and "30" in wfc["message"]

    def test_a_cutoff_at_the_recommendation_is_not_reported(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        assert check_cutoffs_against_pseudos(_spec(silicon_structure)) == []

    def test_a_cutoff_slightly_under_is_not_worth_interrupting_for(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        """29 Ry against a 30 Ry recommendation is noise, not a finding."""
        findings = check_cutoffs_against_pseudos(_spec(silicon_structure, ecutwfc=29.0))

        assert [f for f in findings if f["parameter"].endswith("ecutwfc")] == []

    def test_a_far_higher_cutoff_is_reported_as_expensive(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        """The other direction costs core-hours rather than correctness."""
        findings = check_cutoffs_against_pseudos(
            _spec(silicon_structure, ecutwfc=200.0)
        )

        wfc = next(f for f in findings if f["parameter"].endswith("ecutwfc"))
        assert wfc["severity"] == "far_above_recommended"


class TestWhereItLooks:
    """Finding the values at all, across the shapes real specs take."""

    def test_a_nested_cutoff_is_found_and_reported_by_its_path(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        """PwRelaxWorkChain puts them at base.pw.parameters, not the top level."""
        findings = check_cutoffs_against_pseudos(
            _spec(silicon_structure, ecutwfc=12.0, nested=True)
        )

        wfc = next(f for f in findings if f["parameter"].endswith("ecutwfc"))
        assert wfc["parameter"] == "base.pw.parameters.SYSTEM.ecutwfc"

    def test_the_lowercase_card_spelling_is_accepted(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        """Plugins have written both; run_context accepts either, so must this."""
        findings = check_cutoffs_against_pseudos(
            _spec(silicon_structure, ecutwfc=12.0, card="system")
        )

        assert any(f["parameter"].endswith("ecutwfc") for f in findings)

    def test_the_family_can_be_identified_from_the_pseudos_namespace(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        """A protocol builder resolves pseudos to nodes and drops the label."""
        pseudo = pseudo_family.get_pseudo(element="Si")
        spec = _spec(silicon_structure, ecutwfc=12.0, family_label=None)
        spec["inputs"]["pseudos"] = {"Si": {"pk": pseudo.pk}}

        findings = check_cutoffs_against_pseudos(spec)

        assert any(f["parameter"].endswith("ecutwfc") for f in findings)


class TestWhenItCannotCompare:
    """Silence has two meanings, and neither is 'verified'."""

    def test_a_spec_with_no_family_yields_nothing(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        findings = check_cutoffs_against_pseudos(
            _spec(silicon_structure, ecutwfc=12.0, family_label=None)
        )

        assert findings == []

    def test_a_spec_with_no_structure_yields_nothing(
        self, pseudo_family: SsspFamily
    ) -> None:
        spec: SubmissionSpec = {
            "entry_point": "quantumespresso.pw.relax",
            "inputs": {
                "pseudo_family": FAMILY_LABEL,
                "parameters": {"SYSTEM": {"ecutwfc": 12.0}},
            },
        }

        assert check_cutoffs_against_pseudos(spec) == []

    def test_an_unresolvable_structure_reference_yields_nothing(
        self, pseudo_family: SsspFamily
    ) -> None:
        """A bad pk is submit_workflow's error to raise, not this one's."""
        spec: SubmissionSpec = {
            "entry_point": "quantumespresso.pw.relax",
            "inputs": {
                "structure": {"pk": 999999},
                "pseudo_family": FAMILY_LABEL,
                "parameters": {"SYSTEM": {"ecutwfc": 12.0}},
            },
        }

        assert check_cutoffs_against_pseudos(spec) == []

    def test_a_family_label_naming_no_group_yields_nothing(
        self, silicon_structure: orm.StructureData
    ) -> None:
        findings = check_cutoffs_against_pseudos(
            _spec(silicon_structure, ecutwfc=12.0, family_label="no/such/family")
        )

        assert findings == []


class TestMalformedInput:
    """A check that raises during an approval is worse than one that abstains."""

    def test_a_non_numeric_cutoff_is_skipped(
        self, pseudo_family: SsspFamily, silicon_structure: orm.StructureData
    ) -> None:
        findings = check_cutoffs_against_pseudos(
            _spec(silicon_structure, ecutwfc="not a number")
        )

        assert [f for f in findings if f["parameter"].endswith("ecutwfc")] == []

    @pytest.mark.parametrize(
        "spec",
        [
            {},
            {"entry_point": "x"},
            {"entry_point": "x", "inputs": None},
            {"entry_point": "x", "inputs": []},
        ],
    )
    def test_a_spec_with_no_usable_inputs_yields_nothing(self, spec: t.Any) -> None:
        assert check_cutoffs_against_pseudos(spec) == []
