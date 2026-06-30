"""Tests for ``submit_workflow`` input resolution and the write path.

Covers the value/reference convention in ``_resolve_inputs``:

- a bare primitive is wrapped as the literal *value*, never read as a node PK;
- an existing node is loaded via an explicit ``{"pk"}``/``{"uuid"}``/``{"label"}``
  reference;
- wrapped nodes are *not* stored during resolution, so a later validation
  failure leaves no orphan in the database;
- a reference-only port (``code`` → ``AbstractCode``) rejects a bare value.

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

from aiida_agents.tools.submit import (
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
        ],
    )
    def test_only_non_primitive_ports_need_a_reference(
        self, valid_types: tuple[type, ...], needs_reference: bool
    ) -> None:
        """A port needs an explicit reference iff none of its valid types can be
        built from a bare primitive. The rule is an allow-list of wrappable node
        types, not a block-list of one (``AbstractCode``), so a bare value to a
        ``StructureData``/``RemoteData`` port gets the clean "expects a reference"
        error rather than a confusing ``StructureData(value)`` failure.
        """
        assert _is_reference_type(valid_types) is needs_reference


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
        from aiida_agents.tools import submit as submit_mod

        called: list[str] = []
        monkeypatch.setattr(
            submit_mod, "submit", lambda *a, **k: called.append("submit")
        )

        with pytest.raises(SubmissionInputError):
            submit_workflow("core.arithmetic.add", {})

        assert called == []
