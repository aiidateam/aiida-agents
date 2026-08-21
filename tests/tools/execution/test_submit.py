"""Tests for ``submit_workflow`` input resolution and the write path.

Covers the value/reference convention in ``_resolve_inputs``:

- a bare primitive is wrapped as the literal *value*, never read as a node PK;
- an existing node is loaded via an explicit ``{"pk"}``/``{"uuid"}``/``{"label"}``
  reference;
- wrapped nodes are *not* stored during resolution, so a later validation
  failure leaves no orphan in the database;
- a reference-only port (``code`` → ``AbstractCode``) rejects a bare value;
- a port that names no node (the ``metadata.options`` settings) takes its plain
  value as given, and a namespace dict is never mistaken for a reference.

plus the resolve → validate → (submit-only) path.

All tests run real AiiDA nodes against the session ``aiida_profile`` (brokerless,
``core.sqlite_dos``). ``submit_workflow`` is submit-only, so on this brokerless
profile it raises a clear "no broker" error; the run-to-completion tests instead
drive the engine through ``run_get_node`` (the daemonless local path) on the
resolved inputs, proving our resolution feeds a real run without a daemon.
"""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_agents.tools.execution.submit import (
    SubmissionError,
    SubmissionInputError,
    _format_resolved_inputs,
    _is_reference_type,
    _resolve_inputs,
    _prepare_submission,
    submit_workflow,
)

MULTIPLY_ADD_EP = "core.arithmetic.multiply_add"


class TestBareValueResolution:
    def test_bare_int_wraps_as_literal_value(self) -> None:
        """A bare int becomes an unstored ``Int(value)``."""
        node = _resolve_inputs(MULTIPLY_ADD_EP, {"x": 7})["x"]
        assert isinstance(node, orm.Int)
        assert node.value == 7
        assert not node.is_stored

    def test_bare_int_is_not_interpreted_as_pk(self) -> None:
        """A bare int equal to an existing PK resolves to the *value*, not that
        node. An earlier version loaded the decoy via load_node instead.
        """
        decoy = orm.Int(99999).store()
        node = _resolve_inputs(MULTIPLY_ADD_EP, {"x": decoy.pk})["x"]
        assert isinstance(node, orm.Int)
        assert node.value == decoy.pk
        assert not node.is_stored  # a fresh node, not the stored decoy
        assert node.uuid != decoy.uuid

    @pytest.mark.parametrize(
        "node_fixture", ["arithmetic_add_code", "silicon_structure"]
    )
    def test_existing_node_is_passed_through(
        self, request: pytest.FixtureRequest, node_fixture: str
    ) -> None:
        """An AiiDA node supplied directly is used as-is, whatever its type."""
        node = request.getfixturevalue(node_fixture)
        assert _resolve_inputs(MULTIPLY_ADD_EP, {"x": node})["x"] is node

    def test_float_for_int_port_is_not_silently_truncated(self) -> None:
        """A float for an ``Int``-only port is left unwrapped, not coerced into
        ``Int(2)``. Wrapping would hide the type error; leaving the raw value
        lets ``spec.inputs.validate()`` reject it.
        """
        resolved = _resolve_inputs(MULTIPLY_ADD_EP, {"z": 2.5})
        assert not isinstance(resolved["z"], orm.Int)
        assert resolved["z"] == 2.5


class TestReferenceResolution:
    @pytest.mark.parametrize("ref_key", ["pk", "uuid", "label"])
    def test_reference_loads_existing_code(
        self, arithmetic_add_code: orm.InstalledCode, ref_key: str
    ) -> None:
        """Every reference form resolves to the same existing Code."""
        ref_value = {
            "pk": arithmetic_add_code.pk,
            "uuid": arithmetic_add_code.uuid,
            "label": arithmetic_add_code.full_label,
        }[ref_key]
        resolved = _resolve_inputs(MULTIPLY_ADD_EP, {"code": {ref_key: ref_value}})[
            "code"
        ]
        assert isinstance(resolved, orm.InstalledCode)
        assert resolved.uuid == arithmetic_add_code.uuid

    @pytest.mark.parametrize(
        "bad_ref, match",
        [
            pytest.param(
                {"pk": 10**9}, r"No node found with pk=.*input 'code'", id="missing-pk"
            ),
            pytest.param(
                {"uuid": "00000000-0000-0000-0000-000000000000"},
                r"No node found with uuid=.*input 'code'",
                id="missing-uuid",
            ),
            pytest.param(
                {"label": "nope@nowhere"},
                r"No Code found with label=.*input 'code'",
                id="missing-label",
            ),
        ],
    )
    def test_unresolvable_reference_raises(
        self, bad_ref: dict[str, object], match: str
    ) -> None:
        """A reference to a non-existent node names the form and the port."""
        with pytest.raises(SubmissionInputError, match=match):
            _resolve_inputs(MULTIPLY_ADD_EP, {"code": bad_ref})

    def test_reference_still_resolves_on_an_unconstrained_port(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """Where nothing constrains the port, a reference dict is still the only
        way to name a node, so it keeps resolving. Reference resolution is
        otherwise gated on the port accepting a node, to stop a plain dict whose
        contents happen to carry a ``label`` key being read as one.
        """
        resolved = _resolve_inputs(
            MULTIPLY_ADD_EP, {"undeclared": {"pk": arithmetic_add_code.pk}}
        )
        assert resolved["undeclared"].uuid == arithmetic_add_code.uuid


class TestReferenceOnlyPorts:
    def test_code_port_rejects_bare_value(self) -> None:
        """``code`` (AbstractCode) cannot be built from a plain value."""
        with pytest.raises(
            SubmissionInputError,
            match=r"Input 'code' expects a node reference, not a plain value",
        ):
            _resolve_inputs(MULTIPLY_ADD_EP, {"code": 1})

    @pytest.mark.parametrize(
        "valid_types, needs_reference",
        [
            pytest.param((orm.Int,), False, id="int"),
            pytest.param((orm.Str,), False, id="str"),
            pytest.param((orm.Int, orm.Float), False, id="int-or-float"),
            pytest.param((orm.AbstractCode,), True, id="code"),
            pytest.param((orm.StructureData,), True, id="structure"),
            pytest.param((orm.RemoteData,), True, id="remote-data"),
            pytest.param((orm.Int, orm.StructureData), False, id="mixed-has-primitive"),
            pytest.param((), False, id="unconstrained"),
            # A port naming no node at all takes its value directly: the plain
            # Python types of metadata.options are the common case, and
            # orm.Computer is a non-node entity that lands here too.
            pytest.param((int,), False, id="python-int-option"),
            pytest.param((bool,), False, id="python-bool-option"),
            pytest.param((dict,), False, id="python-dict-option"),
            pytest.param((orm.Computer,), False, id="computer-entity"),
        ],
    )
    def test_reference_needed_only_for_unwrappable_node_ports(
        self, valid_types: tuple[type, ...], needs_reference: bool
    ) -> None:
        """A port needs an explicit reference iff a node type is among its valid
        types and none of them is wrappable. A wrappable type (``orm.Int``) wraps
        a bare value; a non-wrappable node (``StructureData``/``Code``) gets the
        clean "expects a reference" error; a port naming no node (a
        ``metadata.options`` setting, ``metadata.computer``) is left to spec
        validation.
        """
        assert _is_reference_type(valid_types) is needs_reference


class TestPlainValuePorts:
    """Ports whose ``valid_type`` names no node take the value as given.

    Without this the whole ``metadata.options`` namespace, which the protocol
    builder fills and which the execution agent is told to set for a real
    cluster, is unsubmittable: every option is rejected as needing a node
    reference.

    ``core.arithmetic.add`` rather than ``MULTIPLY_ADD_EP`` throughout, because
    only a CalcJob declares ``metadata.options`` at all: the ``multiply_add``
    WorkChain's ``metadata`` namespace has no ``options`` in it.
    """

    @pytest.mark.parametrize(
        "option, value",
        [
            pytest.param("max_wallclock_seconds", 3600, id="int"),
            pytest.param("withmpi", True, id="bool"),
            pytest.param("resources", {"num_machines": 1}, id="dict"),
            pytest.param("additional_retrieve_list", ["aiida.out"], id="list"),
            pytest.param("queue_name", "debug", id="str"),
        ],
    )
    def test_option_keeps_its_python_type(self, option: str, value: object) -> None:
        """The value comes back as itself, not wrapped in a node.

        Asserting the exact type matters: ``orm.Int(3600) == 3600`` is true, so an
        equality check alone would still pass if the option were silently wrapped
        in a node, and the scheduler would only choke on it much later.
        """
        resolved = _resolve_inputs(
            "core.arithmetic.add", {"metadata": {"options": {option: value}}}
        )
        resolved_option = resolved["metadata"]["options"][option]
        assert resolved_option == value
        assert type(resolved_option) is type(value)

    def test_metadata_label_is_a_label_not_a_code_reference(self) -> None:
        """``metadata.label`` is a declared ``str`` port, so a namespace dict
        containing ``label`` is namespace contents, not a ``{"label": ...}`` node
        reference. Reading it as one made every labelled submission fail with a
        missing-code error naming a code the user never mentioned.
        """
        resolved = _resolve_inputs(
            "core.arithmetic.add", {"metadata": {"label": "my run"}}
        )
        assert resolved["metadata"]["label"] == "my run"

    def test_dict_valued_option_may_contain_a_reference_key(self) -> None:
        """A key called ``label`` inside a dict-*valued* port is the user's data.
        ``environment_variables`` takes a plain dict, so no node can go there and
        nothing in it can be a node reference.
        """
        environment = {"label": "run-42", "OMP_NUM_THREADS": "4"}
        resolved = _resolve_inputs(
            "core.arithmetic.add",
            {"metadata": {"options": {"environment_variables": environment}}},
        )
        assert resolved["metadata"]["options"]["environment_variables"] == environment


class TestNoStoreDuringResolution:
    def test_wrapped_nodes_are_unstored(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """Resolution wraps primitives without storing; the referenced code,
        which already exists, comes back stored.
        """
        resolved = _resolve_inputs(
            MULTIPLY_ADD_EP,
            {"x": 2, "y": 3, "z": 4, "code": {"pk": arithmetic_add_code.pk}},
        )
        assert [resolved[k].is_stored for k in ("x", "y", "z")] == [False, False, False]
        assert resolved["code"].is_stored


class TestFormatResolvedInputs:
    def test_stored_and_unstored_rendered_distinctly(self) -> None:
        """The prompt marks newly-wrapped nodes ``[new]`` and shows the pk of
        existing ones, so the human sees what is being created vs reused.
        """
        stored = orm.Int(5).store()
        text = _format_resolved_inputs({"new": orm.Int(7), "existing": stored})
        assert text == (
            f"   new: Int(value=7)  [new]\n   existing: Int(pk={stored.pk}, value=5)"
        )


class TestPrepareSubmission:
    """The seam that resolves inputs and delegates validation to the spec."""

    def test_valid_inputs_return_class_and_resolved(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        from aiida.plugins import WorkflowFactory

        process_class, resolved = _prepare_submission(
            MULTIPLY_ADD_EP,
            {"x": 2, "y": 3, "z": 4, "code": {"pk": arithmetic_add_code.pk}},
        )
        assert process_class is WorkflowFactory(MULTIPLY_ADD_EP)
        assert isinstance(resolved["x"], orm.Int) and resolved["x"].value == 2
        assert resolved["code"].uuid == arithmetic_add_code.uuid

    def test_option_defaults_are_applied_before_validation(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """A CalcJob option that is required but carries a spec default (here
        ``metadata.options.resources``) must not force the user to supply it:
        pre-processing fills it exactly as the engine does at submit time, so a
        local submission validates without the user knowing the nested path. The
        returned inputs stay limited to what the user gave (no metadata
        boilerplate leaking into the HITL preview).
        """
        from aiida.plugins import CalculationFactory

        process_class, resolved = _prepare_submission(
            "core.arithmetic.add",
            {"x": 2, "y": 3, "code": {"pk": arithmetic_add_code.pk}},
        )
        assert process_class is CalculationFactory("core.arithmetic.add")
        assert "metadata" not in resolved

    def test_calcjob_requires_a_code(self) -> None:
        """Agent-scope policy: a compute CalcJob must be given a code. AiiDA makes
        ``code`` optional on the base CalcJob on purpose (import/parse jobs ingest
        a RemoteData and run no code), but the agent only submits compute jobs, so
        require one and fail loudly rather than submit a job that cannot run.
        """
        with pytest.raises(SubmissionInputError, match=r"needs a code"):
            _prepare_submission("core.arithmetic.add", {"x": 5, "y": 8})

    @pytest.mark.parametrize(
        "entry_point, inputs, match",
        [
            pytest.param(
                MULTIPLY_ADD_EP, {"x": 1, "y": 2}, r"'z'", id="missing-required"
            ),
            pytest.param(
                "core.does.not.exist", {}, r"Entry point not found", id="unknown-ep"
            ),
        ],
    )
    def test_invalid_inputs_raise_submission_input_error(
        self, entry_point: str, inputs: dict[str, object], match: str
    ) -> None:
        """Both resolution and validation failures surface as one error type,
        which is what the CLI triage catches to deny back to the model.
        """
        with pytest.raises(SubmissionInputError, match=match):
            _prepare_submission(entry_point, inputs)


class TestSubmitWorkflow:
    def test_workchain_resolution_runs_to_completion(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """The bare-value convention resolves through a WorkChain and runs:
        ``(2 * 3) + 4 == 10``. ``submit_workflow`` is submit-only (it needs a
        broker + daemon), so this drives the engine via ``run_get_node`` -- the
        daemonless local path AiiDA points to -- to prove the resolved inputs run.
        """
        from aiida.engine import run_get_node

        process_class, resolved = _prepare_submission(
            MULTIPLY_ADD_EP,
            {"x": 2, "y": 3, "z": 4, "code": {"pk": arithmetic_add_code.pk}},
        )
        _, node = run_get_node(process_class, **resolved)
        assert node.is_finished_ok
        assert node.outputs.result.value == 10

    def test_submit_requires_a_broker(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """Submit-only: on a brokerless profile (the test profile) the tool
        refuses with a clear, actionable error instead of running in-process.
        """
        with pytest.raises(SubmissionError, match=r"no broker"):
            submit_workflow(
                "core.arithmetic.add",
                {"x": 5, "y": 8, "code": {"pk": arithmetic_add_code.pk}},
            )

    def test_calcjob_resolution_runs_without_user_supplied_resources(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """A CalcJob runs to completion with no resources in the inputs: the
        engine fills ``metadata.options.resources`` from the spec default, so
        ``5 + 8 == 13`` without the user knowing the option exists. Driven via
        ``run_get_node`` since ``submit_workflow`` is submit-only.
        """
        from aiida.engine import run_get_node

        process_class, resolved = _prepare_submission(
            "core.arithmetic.add",
            {"x": 5, "y": 8, "code": {"pk": arithmetic_add_code.pk}},
        )
        _, node = run_get_node(process_class, **resolved)
        assert node.is_finished_ok
        assert node.outputs.sum.value == 13

    def test_calcjob_resolution_runs_with_user_supplied_options(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """The companion case, and the point of resolving plain-value ports: the
        same CalcJob runs with scheduler options the user set, and they reach the
        node as the plain values they were given rather than the spec defaults.
        Resolution and validation are covered above; this is the engine agreeing.
        """
        from aiida.engine import run_get_node

        options = {
            "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
            "max_wallclock_seconds": 120,
            "withmpi": False,
        }
        process_class, resolved = _prepare_submission(
            "core.arithmetic.add",
            {
                "x": 5,
                "y": 8,
                "code": {"pk": arithmetic_add_code.pk},
                "metadata": {"options": options},
            },
        )
        _, node = run_get_node(process_class, **resolved)
        assert node.is_finished_ok
        assert node.outputs.sum.value == 13
        assert node.get_option("max_wallclock_seconds") == 120
        assert node.get_option("resources") == options["resources"]
        assert node.get_option("withmpi") is False

    def test_validation_failure_writes_no_orphans(self) -> None:
        """Invalid inputs raise before any node is stored, so the wrapped
        primitives leave no orphan behind. AiiDA's spec validator reports the
        first missing port ('z' here); the point is that nothing was written.
        """
        sentinel = 987654321  # distinctive value to detect a leaked node
        with pytest.raises(SubmissionInputError, match=r"'z'"):
            submit_workflow(MULTIPLY_ADD_EP, {"x": sentinel, "y": 2})  # missing z, code

        leaked = (
            orm.QueryBuilder()
            .append(orm.Int, filters={"attributes.value": sentinel})
            .count()
        )
        assert leaked == 0

    def test_invalid_inputs_never_reach_the_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validation runs before the engine, so a bad submission never calls
        ``submit`` (the ADR-08 write guarantee).
        """
        from aiida_agents.tools.execution import submit as submit_mod

        called: list[str] = []
        monkeypatch.setattr(
            submit_mod, "submit", lambda *a, **k: called.append("submit")
        )

        with pytest.raises(SubmissionInputError):
            submit_workflow("core.arithmetic.add", {})

        assert called == []


class TestDeeplyNestedNamespaceResolution:
    """Verify recursive port/namespace resolution across 5+ levels of nesting."""

    def test_five_level_nested_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure _resolve_inputs and _format_resolved_inputs handle deeply nested namespaces smoothly."""
        from aiida.engine.processes.ports import InputPort, PortNamespace
        from aiida_agents.tools.execution.submit import _resolve_namespace_inputs

        # Construct 5 levels of nested namespaces: l1 -> l2 -> l3 -> l4 -> l5 -> cutoff
        ns1 = PortNamespace("l1")
        ns2 = ns1.create_port_namespace("l2")
        ns3 = ns2.create_port_namespace("l3")
        ns4 = ns3.create_port_namespace("l4")
        ns5 = ns4.create_port_namespace("l5")
        ns5["cutoff"] = InputPort("cutoff", valid_type=orm.Float)

        raw_inputs = {"l1": {"l2": {"l3": {"l4": {"l5": {"cutoff": 65.0}}}}}}

        resolved = _resolve_namespace_inputs(raw_inputs["l1"], ns1)
        # Check deep nesting structure
        val_node = resolved["l2"]["l3"]["l4"]["l5"]["cutoff"]
        assert isinstance(val_node, orm.Float)
        assert val_node.value == 65.0

        # Check hierarchical formatting displays deep path clearly
        formatted = _format_resolved_inputs({"l1": resolved})
        assert "l1.l2.l3.l4.l5.cutoff: Float(value=65.0)  [new]" in formatted
