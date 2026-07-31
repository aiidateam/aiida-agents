"""Tests for the typed message one specialist passes to the next.

The point of the structured half is that a pk survives the trip exactly. These
pin that: what gets extracted, what does not, and that the rendered message
tells the receiving agent to use the identifiers rather than re-read them out
of the prose.
"""

from __future__ import annotations

from aiida_agents.agents.handoff import (
    MAX_REFERENCES,
    Handoff,
    NodeReference,
    node_references_in,
)


class TestExtraction:
    """Which nodes a step is considered to have identified."""

    def test_a_pk_in_a_tool_return_is_captured(self) -> None:
        refs = node_references_in([{"pk": 334407, "process_label": "PwCalculation"}])

        assert refs == (NodeReference(pk=334407, label="PwCalculation"),)

    def test_a_pk_nested_in_a_result_is_found(self) -> None:
        """diagnose_process_failure buries the root cause two levels down."""
        refs = node_references_in(
            [{"failed": True, "root_cause": {"pk": 334407, "process_label": "PwCalc"}}]
        )

        assert [r.pk for r in refs] == [334407]

    def test_every_row_of_a_list_is_captured_in_order(self) -> None:
        refs = node_references_in(
            [[{"pk": 3, "node_type": "a"}, {"pk": 1, "node_type": "b"}]]
        )

        assert [r.pk for r in refs] == [3, 1]

    def test_a_pk_seen_twice_is_reported_once(self) -> None:
        refs = node_references_in([{"pk": 7, "process_label": "X"}, {"pk": 7}])

        assert len(refs) == 1

    def test_the_first_label_wins_over_a_later_bare_mention(self) -> None:
        """A record that names the node is the one that returned it."""
        refs = node_references_in([{"pk": 7, "process_label": "PwBase"}, {"pk": 7}])

        assert refs[0].label == "PwBase"

    def test_a_node_with_no_usable_label_is_still_carried(self) -> None:
        """The pk is the point; the label is the courtesy."""
        refs = node_references_in([{"pk": 9}])

        assert refs == (NodeReference(pk=9, label=""),)

    def test_a_non_integer_pk_is_ignored(self) -> None:
        """Some tools echo a pk back as the string the model sent."""
        assert node_references_in([{"pk": "not a pk"}]) == ()

    def test_extraction_is_capped(self) -> None:
        """A query matching hundreds must not crowd out the findings."""
        refs = node_references_in([[{"pk": n} for n in range(200)]])

        assert len(refs) == MAX_REFERENCES

    def test_text_output_yields_nothing(self) -> None:
        """search_aiida_docs returns prose; there is no node in it."""
        assert node_references_in(["some documentation excerpt"]) == ()


class TestRendering:
    """What the receiving agent is actually handed."""

    def _handoff(self, **kwargs: object) -> Handoff:
        base: dict[str, object] = {
            "from_specialist": "analysis",
            "to_specialist": "execution",
            "task": "resubmit it with a longer wallclock",
            "findings": "pk 334599 failed with exit 501.",
        }
        base.update(kwargs)
        return Handoff(**base)  # type: ignore[arg-type]

    def test_the_task_comes_first(self) -> None:
        rendered = self._handoff().render()

        assert rendered.startswith("resubmit it with a longer wallclock")

    def test_the_findings_are_carried_verbatim(self) -> None:
        """Paraphrasing would be a third party rewriting a conclusion."""
        rendered = self._handoff().render()

        assert "pk 334599 failed with exit 501." in rendered

    def test_the_producing_specialist_is_named(self) -> None:
        rendered = self._handoff().render()

        assert "analysis agent" in rendered

    def test_references_are_listed_with_what_they_are(self) -> None:
        rendered = self._handoff(
            node_references=(NodeReference(pk=334407, label="PwCalculation"),)
        ).render()

        assert "pk 334407 — PwCalculation" in rendered

    def test_the_consumer_is_told_to_use_the_pks_directly(self) -> None:
        """Without this the structured half is decoration."""
        rendered = self._handoff(
            node_references=(NodeReference(pk=1, label="X"),)
        ).render()

        assert "Use these pks directly" in rendered

    def test_no_reference_section_appears_when_there_are_none(self) -> None:
        """An empty heading reads as "it found nothing", which is different."""
        rendered = self._handoff().render()

        assert "Nodes that step identified" not in rendered

    def test_the_no_assumptions_instruction_survives(self) -> None:
        """Carried over from the free-text handoff it replaces."""
        rendered = self._handoff().render()

        assert "rather than proceeding on an assumption" in rendered
