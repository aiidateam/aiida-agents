"""Unit tests for the ``search_aiida_docs`` Pydantic AI tool.

Tests cover output formatting, the no-results path, and that the tool
delegates correctly to ``query_docs``.  No Ollama or ChromaDB required.
"""

from __future__ import annotations

from unittest.mock import patch

from aiida_agents.rag import search_aiida_docs


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestSearchAiidaDocs:
    def test_no_results_message(self) -> None:
        """A built index that simply has no match reports "no relevant docs"."""
        with (
            patch("aiida_agents.rag.query_docs", return_value=[]),
            patch("aiida_agents.rag.docs_index_available", return_value=True),
        ):
            result = search_aiida_docs("xyzzy unknown term")
        assert "No relevant" in result

    def test_unbuilt_index_message(self) -> None:
        """No results *and* no index tells the user to build it, not "no match"."""
        with (
            patch("aiida_agents.rag.query_docs", return_value=[]),
            patch("aiida_agents.rag.docs_index_available", return_value=False),
        ):
            result = search_aiida_docs("what is a CalcJobNode")
        assert "has not been built" in result
        assert "No relevant" not in result

    def test_formats_source_and_section(self) -> None:
        fake = [
            {
                "source": "topics/data_types.txt",
                "section": "KpointsData",
                "text": "KpointsData represents a grid of k-points.",
            }
        ]
        with patch("aiida_agents.rag.query_docs", return_value=fake):
            result = search_aiida_docs("What is KpointsData?")

        assert "topics/data_types.txt" in result
        assert "KpointsData" in result
        assert "grid of k-points" in result

    def test_multiple_results_separated_by_rule(self) -> None:
        fake = [
            {"source": "a.txt", "section": "A", "text": "Text A."},
            {"source": "b.txt", "section": "B", "text": "Text B."},
        ]
        with patch("aiida_agents.rag.query_docs", return_value=fake):
            result = search_aiida_docs("query")

        assert "---" in result

    def test_delegates_to_query_docs_with_limit_3(self) -> None:
        with (
            patch("aiida_agents.rag.query_docs", return_value=[]) as mock_qd,
            patch("aiida_agents.rag.docs_index_available", return_value=True),
        ):
            search_aiida_docs("test query")
        mock_qd.assert_called_once_with("test query", limit=3)

    def test_result_without_section_omits_section_header(self) -> None:
        fake = [{"source": "misc.txt", "section": "", "text": "Some content."}]
        with patch("aiida_agents.rag.query_docs", return_value=fake):
            result = search_aiida_docs("query")

        # Should use the source-only header format
        assert "[misc.txt]" in result
        assert "§" not in result
