"""Unit tests for the RAG retriever (query path)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiida_agents.rag.retriever import query_docs
from aiida_agents.rag.store import _collection_name


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
            "metadatas": [[{"source": "topics/foo.txt", "section": "Foo"}]],
        }
        mock_client.get_collection.return_value = mock_collection

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.retriever.get_embedding_function",
                return_value=fake_embed,
            ),
        ):
            results = query_docs("What is Foo?", limit=1)

        assert results == [
            {"text": "doc text", "source": "topics/foo.txt", "section": "Foo"}
        ]
