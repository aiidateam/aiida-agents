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
                "source": "topics/data_types",
                "section": "KpointsData",
                "text": "KpointsData represents a grid of k-points.",
            }
        ]
        with patch("aiida_agents.rag.query_docs", return_value=fake):
            result = search_aiida_docs("What is KpointsData?")

        assert "topics/data_types" in result
        assert "KpointsData" in result
        assert "grid of k-points" in result

    def test_multiple_results_separated_by_rule(self) -> None:
        fake = [
            {"source": "topics/a", "section": "A", "text": "Text A."},
            {"source": "topics/b", "section": "B", "text": "Text B."},
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
        fake = [{"source": "misc", "section": "", "text": "Some content."}]
        with patch("aiida_agents.rag.query_docs", return_value=fake):
            result = search_aiida_docs("query")

        # Should use the source-only header format
        assert "[misc]" in result
        assert "§" not in result

    def test_core_corpus_result_omits_corpus_prefix(self) -> None:
        """A result explicitly attributed to the core docs reads exactly like
        one with no 'corpus' key at all -- no visual noise for the common case."""
        fake = [
            {
                "source": "topics/data_types",
                "section": "KpointsData",
                "text": "...",
                "corpus": "aiida-core",
            }
        ]
        with patch("aiida_agents.rag.query_docs", return_value=fake):
            result = search_aiida_docs("What is KpointsData?")

        assert "[topics/data_types  §  KpointsData]" in result
        assert "aiida-core" not in result

    def test_plugin_corpus_result_is_attributed_in_the_header(self) -> None:
        """A hit from a plugin's own corpus names that plugin, so the model
        (and a reader) can tell whose documentation answered."""
        fake = [
            {
                "source": "topics/pseudopotentials",
                "section": "Choosing a pseudo",
                "text": "Use SSSP for most cases.",
                "corpus": "quantumespresso",
            }
        ]
        with patch("aiida_agents.rag.query_docs", return_value=fake):
            result = search_aiida_docs("which pseudopotential should I use?")

        assert (
            "[quantumespresso: topics/pseudopotentials  §  Choosing a pseudo]" in result
        )
        assert "Use SSSP for most cases." in result
