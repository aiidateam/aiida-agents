"""Unit tests for the RAG store helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import chromadb

from aiida_agents.rag.store import (
    _CORPUS_FORMAT,
    _DOCS_TAG,
    _collection_name,
    _collection_populated,
    _plugin_collection_name,
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


def test_plugin_collection_name_keys_by_corpus_name_version_format_and_model() -> None:
    embed = MagicMock()
    embed.name.return_value = "ollama/mxbai-embed-large"
    # A plugin corpus is keyed by its own name + version too, so it never
    # collides with the core docs or with another plugin's corpus, and a
    # version bump on just this corpus rebuilds only its own collection.
    assert (
        _plugin_collection_name(embed, "quantumespresso", "4.5.0")
        == "aiida_agents_plugin_docs__quantumespresso__4.5.0"
        "__fenced1__ollama_mxbai-embed-large"
    )


def test_plugin_collection_name_sanitises_special_characters() -> None:
    embed = MagicMock()
    embed.name.return_value = "sentence-transformers/all-MiniLM-L6-v2"
    # Corpus name/version land in a collection name verbatim, so characters
    # ChromaDB collection names reject must be sanitised the same way the
    # embedding model name already is.
    name = _plugin_collection_name(embed, "my plugin!", "v1.0/rc1")
    assert " " not in name
    assert "!" not in name
    assert "/" not in name
