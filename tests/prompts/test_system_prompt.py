"""Sanity checks for the agent system prompt.

The prompt is loaded from ``system_prompt.md`` via ``importlib.resources``
and exposed as the ``SYSTEM_PROMPT`` string.  These tests are intentionally
lightweight — they guard against accidental erasure or breakage of the
prompt file, not semantic correctness (which is evaluated manually).
"""

from __future__ import annotations

from aiida_agents.prompts import SYSTEM_PROMPT


class TestSystemPrompt:
    def test_is_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)

    def test_non_empty(self) -> None:
        assert SYSTEM_PROMPT.strip()

    def test_contains_aiida_terminology(self) -> None:
        """Core AiiDA terms must be present — guards against loading the wrong file."""
        lower = SYSTEM_PROMPT.lower()
        for term in ("aiida", "provenance", "process"):
            assert term in lower, f"SYSTEM_PROMPT missing expected term: {term!r}"

    def test_mentions_core_tools(self) -> None:
        """Every tool the agent exposes should be referenced in the prompt."""
        lower = SYSTEM_PROMPT.lower()
        for tool in (
            "get_process_status",
            "list_processes",
            "query_nodes",
            "get_node_inputs",
            "get_node_outputs",
            "search_structures",
        ):
            assert tool in lower, f"SYSTEM_PROMPT does not mention tool: {tool!r}"

    def test_has_output_rules(self) -> None:
        """Prompt must include output formatting guidance."""
        assert "output rules" in SYSTEM_PROMPT.lower()
