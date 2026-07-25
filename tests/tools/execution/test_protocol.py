"""Tests for ``build_workflow_inputs``.

No installed process in this environment has a real ``get_builder_from_protocol``
(it is an aiida-quantumespresso / ACWF convention, not aiida-core's), so these
attach a synthetic one to ``ArithmeticAddCalculation`` via monkeypatch -- a real
process class and a real ``ProcessBuilder``, just with a hand-written protocol
method, so the unwrapping logic runs against genuine AiiDA nodes rather than
mocks. One test also feeds the returned spec straight back through
``_resolve_inputs`` and runs it, proving the round-trip actually works, not just
that the shape looks right.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm
from aiida.calculations.arithmetic.add import ArithmeticAddCalculation
from aiida.engine import run_get_node

from aiida_agents.tools.execution.submit import _resolve_inputs
from aiida_agents.tools.execution.introspection import describe_workflow
from aiida_agents.tools.execution.protocol import build_workflow_inputs

ADD_EP = "core.arithmetic.add"


def _fake_get_builder_from_protocol(
    cls: type[ArithmeticAddCalculation],
    code: orm.AbstractCode,
    structure: orm.StructureData | None = None,
    protocol: str = "fast",
    overrides: dict[str, t.Any] | None = None,
) -> t.Any:
    """Stands in for a real ACWF-style protocol builder.

    ``code`` is required (no default) so tests can pin down the "missing
    required kwarg" behaviour; ``structure``/``overrides`` are accepted but
    unused, matching how many real calculations ignore ``structure``.
    """
    builder = cls.get_builder()
    builder.code = code
    builder.x = 2
    builder.y = 3
    return builder


@pytest.fixture
def with_fake_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach the synthetic protocol builder to ``ArithmeticAddCalculation``."""
    monkeypatch.setattr(
        ArithmeticAddCalculation,
        "get_builder_from_protocol",
        classmethod(_fake_get_builder_from_protocol),
        raising=False,
    )


class TestDescribeWorkflowProtocolParameters:
    def test_no_protocol_builder_reports_false_and_none(self) -> None:
        """A process without get_builder_from_protocol reports it plainly."""
        desc = describe_workflow(ADD_EP)
        assert desc["has_protocol_builder"] is False
        assert desc["protocol_parameters"] is None

    @pytest.mark.usefixtures("with_fake_protocol")
    def test_protocol_builder_reports_its_exact_signature(self) -> None:
        """protocol_parameters names each kwarg, whether required, and its default.

        Signatures vary by workflow (not every one takes 'structure', some need a
        'codes' mapping instead of 'code'), so this must read the real signature.
        """
        desc = describe_workflow(ADD_EP)
        assert desc["has_protocol_builder"] is True
        by_name = {p["name"]: p for p in desc["protocol_parameters"]}
        assert by_name["code"] == {"name": "code", "required": True, "default": None}
        assert by_name["structure"]["required"] is False
        assert by_name["protocol"]["required"] is False
        assert "cls" not in by_name  # the classmethod's own first argument


class TestBuildWorkflowInputs:
    @pytest.mark.usefixtures("with_fake_protocol")
    def test_builds_a_spec_from_the_protocol_builder(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """The populated builder serialises to a ready-to-use WorkflowSpec."""
        spec = build_workflow_inputs(
            ADD_EP,
            protocol="fast",
            protocol_kwargs={"code": {"label": arithmetic_add_code.full_label}},
        )
        assert spec["workflow_type"] == ADD_EP
        assert spec["inputs"]["x"] == 2
        assert spec["inputs"]["y"] == 3
        assert spec["inputs"]["code"] == {"pk": arithmetic_add_code.pk}
        assert spec["metadata"]["protocol"] == "fast"

    @pytest.mark.usefixtures("with_fake_protocol")
    def test_reference_by_pk_and_uuid_also_resolve(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """The same {"pk"|"uuid"|"label"} convention as execute_workflow_spec."""
        for ref in (
            {"pk": arithmetic_add_code.pk},
            {"uuid": arithmetic_add_code.uuid},
        ):
            spec = build_workflow_inputs(ADD_EP, protocol_kwargs={"code": ref})
            assert spec["inputs"]["code"] == {"pk": arithmetic_add_code.pk}

    def test_no_protocol_builder_raises_a_clear_error(self) -> None:
        """A process without get_builder_from_protocol fails loudly, naming the
        fallback (build inputs directly) rather than a raw AttributeError."""
        with pytest.raises(ValueError, match="has no get_builder_from_protocol"):
            build_workflow_inputs(ADD_EP)

    @pytest.mark.usefixtures("with_fake_protocol")
    def test_missing_required_kwarg_names_it(self) -> None:
        """Omitting a required protocol_kwarg (here: code) is a clear, actionable
        error, not a raw TypeError from the builder call."""
        with pytest.raises(ValueError, match=r"needs \['code'\]"):
            build_workflow_inputs(ADD_EP)

    @pytest.mark.usefixtures("with_fake_protocol")
    def test_unknown_protocol_kwarg_is_rejected(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """A protocol_kwarg the builder doesn't accept is caught before the call,
        naming the real parameters instead of a generic TypeError."""
        with pytest.raises(ValueError, match="has no parameter 'not_a_real_kwarg'"):
            build_workflow_inputs(
                ADD_EP,
                protocol_kwargs={
                    "code": {"label": arithmetic_add_code.full_label},
                    "not_a_real_kwarg": 1,
                },
            )

    @pytest.mark.usefixtures("with_fake_protocol")
    def test_unresolvable_reference_names_the_kwarg(self) -> None:
        """A bad node reference in protocol_kwargs fails with the same clarity as
        execute_workflow_spec's own reference resolution."""
        with pytest.raises(ValueError, match=r"No node found with pk=.*'code'"):
            build_workflow_inputs(ADD_EP, protocol_kwargs={"code": {"pk": 10**9}})

    def test_multi_code_style_nested_reference_resolves(
        self, monkeypatch: pytest.MonkeyPatch, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """A reference nested inside a dict (the 'codes' shape a multi-code
        workflow's protocol builder often takes) resolves too, not just a
        top-level 'code'/'structure'."""

        def fake_multi_code(
            cls: type[ArithmeticAddCalculation],
            codes: dict[str, orm.AbstractCode],
            protocol: str = "fast",
        ) -> t.Any:
            builder = cls.get_builder()
            builder.code = codes["add"]
            builder.x = 1
            builder.y = 1
            return builder

        monkeypatch.setattr(
            ArithmeticAddCalculation,
            "get_builder_from_protocol",
            classmethod(fake_multi_code),
            raising=False,
        )
        spec = build_workflow_inputs(
            ADD_EP,
            protocol_kwargs={
                "codes": {"add": {"label": arithmetic_add_code.full_label}}
            },
        )
        assert spec["inputs"]["code"] == {"pk": arithmetic_add_code.pk}

    @pytest.mark.usefixtures("with_fake_protocol")
    def test_spec_round_trips_through_submission(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """The returned spec is not just shaped right -- fed back through the
        existing resolution/submission pipeline, it actually runs and produces
        the correct result. Pins the contract between build_workflow_inputs and
        execute_workflow_spec/submit_workflow, not just build_workflow_inputs
        in isolation.
        """
        spec = build_workflow_inputs(
            ADD_EP,
            protocol_kwargs={"code": {"label": arithmetic_add_code.full_label}},
        )
        resolved = _resolve_inputs(spec["workflow_type"], spec["inputs"])
        _, node = run_get_node(ArithmeticAddCalculation, **resolved)
        assert node.outputs.sum.value == 5


class TestOverridesReachTheBuilder:
    """``overrides`` is the builder's own merge mechanism, so it must arrive
    exactly as written -- the reference resolver walks nested dicts looking for
    pk/uuid/label references, and must leave a physics-parameter tree alone.
    """

    def test_nested_overrides_pass_through_unmangled(
        self, monkeypatch: pytest.MonkeyPatch, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        seen: dict[str, t.Any] = {}

        def _recording_builder(
            cls: type[ArithmeticAddCalculation],
            code: orm.AbstractCode,
            protocol: str = "fast",
            overrides: dict[str, t.Any] | None = None,
        ) -> t.Any:
            seen["overrides"] = overrides
            builder = cls.get_builder()
            builder.code = code
            builder.x = 2
            builder.y = 3
            return builder

        monkeypatch.setattr(
            ArithmeticAddCalculation,
            "get_builder_from_protocol",
            classmethod(_recording_builder),
            raising=False,
        )

        overrides = {"base": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": 60.0}}}}}
        build_workflow_inputs(
            ADD_EP,
            protocol_kwargs={
                "code": {"label": arithmetic_add_code.full_label},
                "overrides": overrides,
            },
        )

        assert seen["overrides"] == overrides

    def test_reference_inside_overrides_is_still_resolved(
        self, monkeypatch: pytest.MonkeyPatch, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """A genuine node reference nested in overrides is still loaded.

        Some workflows take a node (a parent folder, a reference structure)
        through overrides, so the resolver must reach into it -- passing the
        raw dict through would hand the builder ``{"pk": N}`` instead of a node.
        """
        seen: dict[str, t.Any] = {}

        def _recording_builder(
            cls: type[ArithmeticAddCalculation],
            code: orm.AbstractCode,
            protocol: str = "fast",
            overrides: dict[str, t.Any] | None = None,
        ) -> t.Any:
            seen["overrides"] = overrides
            builder = cls.get_builder()
            builder.code = code
            builder.x = 2
            builder.y = 3
            return builder

        monkeypatch.setattr(
            ArithmeticAddCalculation,
            "get_builder_from_protocol",
            classmethod(_recording_builder),
            raising=False,
        )

        build_workflow_inputs(
            ADD_EP,
            protocol_kwargs={
                "code": {"label": arithmetic_add_code.full_label},
                "overrides": {"base": {"parent_code": {"pk": arithmetic_add_code.pk}}},
            },
        )

        assert seen["overrides"]["base"]["parent_code"].pk == arithmetic_add_code.pk
