"""Tests for ``submit_workflow`` input resolution and the write path.

Covers the value/reference convention in ``_resolve_inputs``:

- a bare primitive is wrapped as the literal *value*, never read as a node PK;
- an existing node is loaded via an explicit ``{"pk"}``/``{"uuid"}``/``{"label"}``
  reference;
- wrapped nodes are *not* stored during resolution, so a later validation
  failure leaves no orphan in the database;
- a reference-only port (``code`` → ``AbstractCode``) rejects a bare value.

plus the full resolve → validate → submit path end to end.

All tests run real AiiDA nodes against the session ``aiida_profile`` (brokerless
and ``core.sqlite_dos``), so ``submit_workflow`` takes the in-process
``run_get_node`` branch and a real workflow actually executes.
"""

from __future__ import annotations

import pytest
from aiida import orm
from fastmcp.exceptions import ToolError

from aiida_agents.mcp.tools.submit import (
    _format_resolved_inputs,
    _resolve_inputs,
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
        with pytest.raises(ToolError, match=match):
            _resolve_inputs(MULTIPLY_ADD_EP, {"code": bad_ref})


class TestReferenceOnlyPorts:
    def test_code_port_rejects_bare_value(self) -> None:
        """``code`` (AbstractCode) cannot be built from a plain value."""
        with pytest.raises(
            ToolError, match=r"Input 'code' expects a node reference, not a plain value"
        ):
            _resolve_inputs(MULTIPLY_ADD_EP, {"code": 1})


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


class TestSubmitWorkflow:
    def test_submit_runs_workflow_end_to_end(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """The bare-value convention flows through validation into a real run:
        ``(2 * 3) + 4 == 10``.
        """
        result = submit_workflow(
            MULTIPLY_ADD_EP,
            {"x": 2, "y": 3, "z": 4, "code": {"pk": arithmetic_add_code.pk}},
        )
        node = orm.load_node(result["pk"])
        assert result == {
            "pk": node.pk,
            "uuid": node.uuid,
            "workflow": MULTIPLY_ADD_EP,
            "state": "finished",
        }
        assert node.is_finished_ok
        assert node.outputs.result.value == 10

    def test_validation_failure_writes_no_orphans(self) -> None:
        """Invalid inputs raise before any node is stored, so the wrapped
        primitives leave no orphan behind.
        """
        sentinel = 987654321  # distinctive value to detect a leaked node
        with pytest.raises(ToolError, match="Validation failed") as exc_info:
            submit_workflow(MULTIPLY_ADD_EP, {"x": sentinel, "y": 2})  # missing z, code

        # the error names every missing port, not just the first
        message = str(exc_info.value)
        assert "'z'" in message and "'code'" in message

        leaked = (
            orm.QueryBuilder()
            .append(orm.Int, filters={"attributes.value": sentinel})
            .count()
        )
        assert leaked == 0
