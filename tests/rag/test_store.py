"""Unit tests for the RAG store helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiida_agents.rag.store import _DOCS_TAG, _collection_name


def test_collection_name_keys_by_version_and_model() -> None:
    embed = MagicMock()
    embed.name.return_value = "ollama/mxbai-embed-large"
    # Keyed by the pinned docs version + the model, with "/" sanitised to "_".
    assert (
        _collection_name(embed) == f"aiida_docs__{_DOCS_TAG}__ollama_mxbai-embed-large"
    )
