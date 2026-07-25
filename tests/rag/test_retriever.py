"""Unit tests for the RAG retriever (query path)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiida_agents.plugins import LoadedPlugin, RagCorpus
from aiida_agents.rag.retriever import docs_index_available, query_docs
from aiida_agents.rag.store import _collection_name, _plugin_collection_name


def _fake_embed(name: str = "fake/model") -> MagicMock:
    embed = MagicMock()
    embed.name.return_value = name
    embed.embed_query.return_value = [[0.1, 0.2, 0.3]]
    return embed


class TestQueryDocs:
    def test_returns_empty_list_when_collection_missing(self) -> None:
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []  # nothing indexed yet

        fake_embed = MagicMock()
        fake_embed.name.return_value = "fake/model"

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.retriever.get_embedding_function",
                return_value=fake_embed,
            ),
            patch("aiida_agents.rag.retriever.discover_plugins", return_value=()),
        ):
            results = query_docs("anything")

        assert results == []

    def test_returns_results_with_expected_keys(self) -> None:
        fake_embed = MagicMock()
        fake_embed.name.return_value = "fake/model"
        fake_embed.embed_query.return_value = [[0.1, 0.2, 0.3]]
        # The query path looks for the collection keyed by version + model.
        name = _collection_name(fake_embed)

        mock_col = MagicMock()
        mock_col.name = name
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_col]
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc text"]],
            "metadatas": [[{"source": "topics/foo", "section": "Foo"}]],
            "distances": [[0.05]],
        }
        mock_client.get_collection.return_value = mock_collection

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.retriever.get_embedding_function",
                return_value=fake_embed,
            ),
            patch("aiida_agents.rag.retriever.discover_plugins", return_value=()),
        ):
            results = query_docs("What is Foo?", limit=1)

        assert results == [
            {
                "text": "doc text",
                "source": "topics/foo",
                "section": "Foo",
                "corpus": "aiida-core",
            }
        ]

    def test_merges_core_and_plugin_corpus_results_by_distance(self) -> None:
        """A plugin's corpus is searched alongside the core docs, and the
        closer hit wins regardless of which collection it came from."""
        embed = _fake_embed()
        core_name = _collection_name(embed)
        plugin_name = _plugin_collection_name(embed, "qe", "1.0")

        mock_core_col = MagicMock(name=core_name)
        mock_core_col.name = core_name
        mock_plugin_col = MagicMock(name=plugin_name)
        mock_plugin_col.name = plugin_name

        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_core_col, mock_plugin_col]

        core_collection = MagicMock()
        core_collection.query.return_value = {
            "documents": [["core hit"]],
            "metadatas": [[{"source": "topics/core", "section": "Core"}]],
            "distances": [[0.5]],  # farther
        }
        plugin_collection = MagicMock()
        plugin_collection.query.return_value = {
            "documents": [["qe hit"]],
            "metadatas": [[{"source": "topics/qe", "section": "QE"}]],
            "distances": [[0.1]],  # closer
        }

        def _get_collection(name: str, embedding_function: object) -> MagicMock:
            return core_collection if name == core_name else plugin_collection

        mock_client.get_collection.side_effect = _get_collection

        plugins = (
            LoadedPlugin(
                name="quantumespresso",
                corpora=(RagCorpus(name="qe", version="1.0", text_dir=None),),
            ),
        )

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.retriever.get_embedding_function", return_value=embed
            ),
            patch("aiida_agents.rag.retriever.discover_plugins", return_value=plugins),
        ):
            results = query_docs("anything", limit=2)

        # The plugin hit is closer (smaller distance), so it ranks first, and
        # each result is attributed to the corpus it actually came from.
        assert results == [
            {
                "text": "qe hit",
                "source": "topics/qe",
                "section": "QE",
                "corpus": "qe",
            },
            {
                "text": "core hit",
                "source": "topics/core",
                "section": "Core",
                "corpus": "aiida-core",
            },
        ]

    def test_searches_plugin_corpus_even_when_core_index_missing(self) -> None:
        """No core index yet, but an installed plugin's corpus is searchable."""
        embed = _fake_embed()
        plugin_name = _plugin_collection_name(embed, "qe", "1.0")

        mock_plugin_col = MagicMock(name=plugin_name)
        mock_plugin_col.name = plugin_name
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_plugin_col]

        plugin_collection = MagicMock()
        plugin_collection.query.return_value = {
            "documents": [["qe hit"]],
            "metadatas": [[{"source": "topics/qe", "section": "QE"}]],
            "distances": [[0.2]],
        }
        mock_client.get_collection.return_value = plugin_collection

        plugins = (
            LoadedPlugin(
                name="quantumespresso",
                corpora=(RagCorpus(name="qe", version="1.0", text_dir=None),),
            ),
        )

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.retriever.get_embedding_function", return_value=embed
            ),
            patch("aiida_agents.rag.retriever.discover_plugins", return_value=plugins),
        ):
            results = query_docs("anything")

        assert results == [
            {"text": "qe hit", "source": "topics/qe", "section": "QE", "corpus": "qe"}
        ]


class TestDocsIndexAvailable:
    def test_false_when_nothing_populated(self) -> None:
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.retriever.get_embedding_function",
                return_value=_fake_embed(),
            ),
            patch("aiida_agents.rag.retriever.discover_plugins", return_value=()),
        ):
            assert docs_index_available() is False

    def test_true_when_only_a_plugin_corpus_is_populated(self) -> None:
        """The core docs were never built, but a plugin's corpus was --
        documentation search must still report itself as available."""
        embed = _fake_embed()
        plugin_name = _plugin_collection_name(embed, "qe", "1.0")

        mock_col = MagicMock(name=plugin_name)
        mock_col.name = plugin_name
        mock_col.count.return_value = 5
        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_col]

        plugins = (
            LoadedPlugin(
                name="quantumespresso",
                corpora=(RagCorpus(name="qe", version="1.0", text_dir=None),),
            ),
        )

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.retriever.get_embedding_function", return_value=embed
            ),
            patch("aiida_agents.rag.retriever.discover_plugins", return_value=plugins),
        ):
            assert docs_index_available() is True
