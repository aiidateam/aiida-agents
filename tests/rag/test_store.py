"""Unit tests for the RAG store helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import chromadb

from aiida_agents.rag.store import (
    _CORPUS_FORMAT,
    _DOCS_TAG,
    _collection_name,
    _collection_populated,
)


def test_collection_populated_treats_empty_stub_as_absent() -> None:
    """Absent and empty collections both read as "not populated"; a filled one does not.

    Pins the guard the atomic indexer relies on: an empty stub left by an
    interrupted build must not be mistaken for a finished index.
    """
    client = chromadb.EphemeralClient()
    name = "aiida_docs_populated_probe"

    assert _collection_populated(client, name) is False  # absent

    collection = client.create_collection(name)
    assert _collection_populated(client, name) is False  # exists but empty

    collection.add(ids=["a"], embeddings=[[0.1, 0.2]], documents=["x"])
    assert _collection_populated(client, name) is True  # populated


def test_collection_name_keys_by_version_format_and_model() -> None:
    embed = MagicMock()
    embed.name.return_value = "ollama/mxbai-embed-large"
    # Keyed by the pinned docs version + corpus format + the model, with "/"
    # sanitised to "_". Any of the three changing must resolve to a new
    # collection so a stale or incompatible index is never silently reused.
    assert (
        _collection_name(embed)
        == f"aiida_docs__{_DOCS_TAG}__{_CORPUS_FORMAT}__ollama_mxbai-embed-large"
    )
