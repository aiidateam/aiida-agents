"""Unit tests for the ``search_aiida_docs`` Pydantic AI tool.

Tests cover output formatting, the no-results path, and that the tool
delegates correctly to ``query_docs``.  No Ollama or ChromaDB required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


from aiida_agents.rag import search_aiida_docs


class _FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeClient:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def list_collections(self) -> list[_FakeCollection]:
        return [_FakeCollection(n) for n in self._names]


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


class TestUnavailableIndexMessage:
    """What the tool tells the model when it cannot search.

    Two distinct causes with two different fixes: never built (rebuild) versus
    built under a different embedding model (start Ollama). Conflating them
    sent users to rebuild, which only creates a *second* unreachable index.
    """

    def test_never_built_says_to_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiida_agents.rag import _unavailable_message

        monkeypatch.setattr(
            "aiida_agents.rag.store._get_client", lambda settings=None: _FakeClient([])
        )
        monkeypatch.setattr(
            "aiida_agents.rag.embeddings.configured_embedding_name",
            lambda *a, **k: "ollama/mxbai-embed-large",
        )

        msg = _unavailable_message()

        assert "has not been built" in msg
        assert "rag build" in msg

    def test_embedding_mismatch_says_not_to_rebuild(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An index built with the configured model, queried with the fallback."""
        from aiida_agents.rag import _unavailable_message
        from aiida_agents.rag.store import _core_name_for

        configured = "ollama/mxbai-embed-large"
        monkeypatch.setattr(
            "aiida_agents.rag.store._get_client",
            lambda settings=None: _FakeClient([_core_name_for(configured)]),
        )
        monkeypatch.setattr(
            "aiida_agents.rag.embeddings.configured_embedding_name",
            lambda *a, **k: configured,
        )

        msg = _unavailable_message()

        assert "Ollama" in msg
        assert "NOT the fix" in msg

    def test_every_unavailable_message_forbids_answering_from_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fabricated-API failure mode: the instruction must always be there.

        Observed in end-to-end testing -- with the index unavailable the agent
        invented a confident, entirely fictional AiiDA API.
        """
        from aiida_agents.rag import _unavailable_message

        monkeypatch.setattr(
            "aiida_agents.rag.store._get_client", lambda settings=None: _FakeClient([])
        )
        monkeypatch.setattr(
            "aiida_agents.rag.embeddings.configured_embedding_name",
            lambda *a, **k: "ollama/mxbai-embed-large",
        )
        assert "do NOT answer this from memory" in _unavailable_message()

        # And on the degraded path, where diagnosis itself failed.
        monkeypatch.setattr(
            "aiida_agents.rag.store._get_client",
            lambda settings=None: (_ for _ in ()).throw(RuntimeError("store gone")),
        )
        assert "do NOT answer this from memory" in _unavailable_message()
