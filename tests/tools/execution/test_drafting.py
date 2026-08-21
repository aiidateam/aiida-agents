"""Tests for ``draft_process_inputs``.

Two kinds of test here, deliberately.

The behavioural rules --- which defaults get filled, which required ports get
reported, what happens to an optional namespace nobody touched --- are pinned
against a synthetic ``WorkChain`` declared in this module. It mixes the port
shapes real workflows mix (a default, a required port, a required and an
optional namespace, a dynamic one) in a single process, which no installed
core process does, and it will not drift when aiida-core changes an unrelated
spec.

The round trip is pinned against a real one: a draft of
``core.arithmetic.add``, completed and fed straight through ``_resolve_inputs``
and the engine. That is the claim that actually matters --- that what this tool
returns is submittable, not merely well-shaped --- and only a real process can
carry it.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm
from aiida.calculations.arithmetic.add import ArithmeticAddCalculation
from aiida.engine import WorkChain, run_get_node

from aiida_agents.tools.execution import drafting
from aiida_agents.tools.execution.drafting import draft_process_inputs
from aiida_agents.tools.execution.schemas import WorkflowSpec
from aiida_agents.tools.execution.submit import _resolve_inputs

ADD_EP = "core.arithmetic.add"
MULTIPLY_ADD_EP = "core.arithmetic.multiply_add"
SYNTHETIC_EP = "tests.drafting.synthetic"


class SyntheticWorkChain(WorkChain):
    """A process that mixes every port shape the drafter has to handle."""

    @classmethod
    def define(cls, spec: t.Any) -> None:
        super().define(spec)
        spec.input("required_port", valid_type=orm.Int, help="Needs a value.")
        spec.input("cutoff", valid_type=orm.Float, default=lambda: orm.Float(30.0))
        spec.input("scheme", valid_type=orm.Str, default=lambda: orm.Str("bfgs"))
        spec.input("optional_port", valid_type=orm.Str, required=False)
        spec.input("structure", valid_type=orm.StructureData, required=False)
        spec.input("required_ns.inner", valid_type=orm.Int, help="Inner requirement.")
        spec.input_namespace("optional_ns", required=False)
        spec.input("optional_ns.inner", valid_type=orm.Int)
        spec.input("optional_ns.other", valid_type=orm.Int, required=False)
        spec.input_namespace("pseudos", dynamic=True, required=False)


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve ``SYNTHETIC_EP`` to the synthetic process class.

    The class cannot be reached through AiiDA's entry-point registry without
    installing a package, and entry-point resolution is not what these tests
    are about --- ``test_unknown_entry_point_is_rejected`` covers that.
    """
    monkeypatch.setattr(
        drafting,
        "load_process_class",
        lambda entry_point: (
            SyntheticWorkChain
            if entry_point == SYNTHETIC_EP
            else pytest.fail(f"unexpected entry point {entry_point!r}")
        ),
    )


def _missing_ports(spec: WorkflowSpec) -> list[str]:
    return [entry["port"] for entry in spec["metadata"]["missing_required"]]


class TestDefaults:
    """What the drafter fills in on the caller's behalf."""

    def test_declared_defaults_are_filled_in(self, synthetic: None) -> None:
        spec = draft_process_inputs(SYNTHETIC_EP)

        # Serialised to bare values, not left as the Float/Str nodes the
        # defaults construct -- that is the WorkflowSpec convention.
        assert spec["inputs"]["cutoff"] == 30.0
        assert spec["inputs"]["scheme"] == "bfgs"

    def test_a_supplied_value_wins_over_the_default(self, synthetic: None) -> None:
        spec = draft_process_inputs(SYNTHETIC_EP, {"cutoff": 80.0})

        assert spec["inputs"]["cutoff"] == 80.0

    def test_optional_ports_without_defaults_stay_absent(self, synthetic: None) -> None:
        spec = draft_process_inputs(SYNTHETIC_EP)

        assert "optional_port" not in spec["inputs"]
        assert "structure" not in spec["inputs"]

    def test_a_default_that_cannot_be_evaluated_is_not_invented(
        self, synthetic: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising default leaves the port unfilled rather than guessed."""

        def _explode() -> orm.Float:
            raise RuntimeError("needs context the draft does not have")

        port = SyntheticWorkChain.spec().inputs["cutoff"]
        monkeypatch.setattr(port, "_default", _explode)

        spec = draft_process_inputs(SYNTHETIC_EP)

        assert "cutoff" not in spec["inputs"]


class TestMissingRequired:
    """What the drafter refuses to guess, and reports instead."""

    def test_required_ports_are_reported_with_types_and_help(
        self, synthetic: None
    ) -> None:
        spec = draft_process_inputs(SYNTHETIC_EP)

        entry = next(
            e
            for e in spec["metadata"]["missing_required"]
            if e["port"] == "required_port"
        )
        assert entry["valid_types"] == ["Int"]
        assert entry["help"] == "Needs a value."
        assert spec["metadata"]["ready_to_submit"] is False

    def test_a_port_inside_a_required_namespace_is_reported_by_full_path(
        self, synthetic: None
    ) -> None:
        spec = draft_process_inputs(SYNTHETIC_EP)

        assert "required_ns.inner" in _missing_ports(spec)

    def test_an_untouched_optional_namespace_demands_nothing(
        self, synthetic: None
    ) -> None:
        """Its inner port is required *of the namespace*, not of the process."""
        spec = draft_process_inputs(SYNTHETIC_EP)

        assert "optional_ns.inner" not in _missing_ports(spec)
        assert "optional_ns" not in spec["inputs"]

    def test_populating_an_optional_namespace_activates_its_requirements(
        self, synthetic: None
    ) -> None:
        """Once populated, AiiDA validates it -- so the draft must too.

        Supplying only the namespace's *optional* port is what makes this
        visible: the required sibling now has to be reported, where a moment
        ago it did not.
        """
        spec = draft_process_inputs(SYNTHETIC_EP, {"optional_ns": {"other": 1}})

        assert "optional_ns.inner" in _missing_ports(spec)
        assert spec["inputs"]["optional_ns"]["other"] == 1

    def test_filling_every_requirement_marks_the_draft_ready(
        self, synthetic: None
    ) -> None:
        spec = draft_process_inputs(
            SYNTHETIC_EP,
            {"required_port": 1, "required_ns": {"inner": 2}},
        )

        assert spec["metadata"]["missing_required"] == []
        assert spec["metadata"]["ready_to_submit"] is True

    def test_a_real_workchains_requirements_are_read_from_its_spec(
        self, aiida_profile: t.Any
    ) -> None:
        """No synthetic class involved: the four inputs MultiplyAdd declares."""
        spec = draft_process_inputs(MULTIPLY_ADD_EP)

        assert set(_missing_ports(spec)) == {"x", "y", "z", "code"}


class TestRejectedInput:
    """The errors that catch an invented input while it is still cheap."""

    def test_an_undeclared_port_is_rejected_and_the_real_ones_named(
        self, synthetic: None
    ) -> None:
        with pytest.raises(ValueError, match="Unknown input port") as excinfo:
            draft_process_inputs(SYNTHETIC_EP, {"ecutwfc": 60.0})

        message = str(excinfo.value)
        assert "'ecutwfc'" in message
        assert "required_port" in message  # the ports it could have meant

    def test_an_undeclared_nested_port_names_its_namespace(
        self, synthetic: None
    ) -> None:
        with pytest.raises(ValueError, match="required_ns") as excinfo:
            draft_process_inputs(SYNTHETIC_EP, {"required_ns": {"nope": 1}})

        assert "'nope'" in str(excinfo.value)

    def test_a_dynamic_namespace_accepts_undeclared_keys(self, synthetic: None) -> None:
        """Its keys are open by declaration, so the spec is not the authority."""
        spec = draft_process_inputs(SYNTHETIC_EP, {"pseudos": {"Si": "SSSP/Si"}})

        assert spec["inputs"]["pseudos"] == {"Si": "SSSP/Si"}

    def test_a_namespace_given_a_scalar_is_rejected(self, synthetic: None) -> None:
        with pytest.raises(ValueError, match="is a namespace"):
            draft_process_inputs(SYNTHETIC_EP, {"required_ns": 5})

    def test_an_unresolvable_reference_is_rejected(
        self, synthetic: None, aiida_profile: t.Any
    ) -> None:
        """Caught while the agent can still fix it, not at submission."""
        with pytest.raises(ValueError, match="No node found with pk=999999"):
            draft_process_inputs(SYNTHETIC_EP, {"structure": {"pk": 999999}})

    def test_unknown_entry_point_is_rejected(self, aiida_profile: t.Any) -> None:
        with pytest.raises(ValueError, match="Entry point not found"):
            draft_process_inputs("not.a.real.process")

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_entry_point_must_be_a_non_empty_string(
        self, bad: t.Any, aiida_profile: t.Any
    ) -> None:
        with pytest.raises(ValueError, match="entry_point must be"):
            draft_process_inputs(bad)

    def test_inputs_must_be_a_mapping(self, aiida_profile: t.Any) -> None:
        with pytest.raises(ValueError, match="inputs must be a mapping"):
            draft_process_inputs(ADD_EP, [("x", 2)])  # type: ignore[arg-type]


class TestReferences:
    """Node references survive the draft in the form submission expects."""

    def test_a_resolvable_reference_is_kept_as_a_reference(
        self, silicon_structure: orm.StructureData, synthetic: None
    ) -> None:
        """Resolved to prove it exists, then handed back unchanged.

        The spec stays a plain printable document -- which is what the approval
        prompt shows the user before anything is written.
        """
        ref = {"pk": silicon_structure.pk}

        spec = draft_process_inputs(SYNTHETIC_EP, {"structure": ref})

        assert spec["inputs"]["structure"] == ref

    def test_a_reference_key_inside_a_dict_valued_port_is_data(self) -> None:
        """``environment_variables`` takes a plain dict, so no node can go there
        and nothing inside it can be a node reference. Reading its ``label`` key
        as one failed the draft with a missing-*code* error naming a code the
        user never mentioned.
        """
        environment = {"label": "run-42", "OMP_NUM_THREADS": "4"}

        spec = draft_process_inputs(
            ADD_EP, {"metadata": {"options": {"environment_variables": environment}}}
        )

        assert spec["inputs"]["metadata"]["options"]["environment_variables"] == (
            environment
        )

    def test_a_reference_on_a_node_port_still_resolves_eagerly(self) -> None:
        """The gate narrows *where* a reference is looked for, not whether an
        unresolvable one is caught: a bad reference on a node port must still
        fail during drafting, while the agent can act on it.
        """
        with pytest.raises(ValueError, match=r"No node found with pk=.*'code'"):
            draft_process_inputs(ADD_EP, {"code": {"pk": 10**9}})


class TestRoundTrip:
    """The claim that matters: a completed draft actually submits."""

    def test_a_completed_draft_runs_through_the_engine(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        spec = draft_process_inputs(
            ADD_EP,
            {"x": 2, "y": 3, "code": {"pk": arithmetic_add_code.pk}},
        )
        assert spec["metadata"]["ready_to_submit"] is True

        resolved = _resolve_inputs(spec["workflow_type"], spec["inputs"])
        _, node = run_get_node(ArithmeticAddCalculation, **resolved)

        assert node.is_finished_ok
        assert node.outputs.sum.value == 5

    def test_the_draft_is_addressed_to_the_process_it_was_asked_about(
        self, aiida_profile: t.Any
    ) -> None:
        spec = draft_process_inputs(ADD_EP)

        assert spec["workflow_type"] == ADD_EP
        assert spec["metadata"]["source"] == "process_spec"
