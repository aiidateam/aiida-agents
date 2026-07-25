"""Unit tests for the RAG store helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import chromadb
import pytest

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


class TestStaleCollectionsDoNotCountAsBuilt:
    """A collection a query cannot reach must not read as a built index.

    Regression from end-to-end testing: ``built`` was "any populated
    collection", but retrieval looks a collection up by exact name. A store
    holding only a collection from an older docs version / corpus format /
    embedding model therefore reported healthy in ``rag status`` and
    ``doctor`` while ``search_aiida_docs`` said no index existed -- sending the
    user to look in entirely the wrong place.
    """

    @staticmethod
    def _seed(path: str, name: str) -> None:
        from aiida_agents.rag.store import _get_client

        client = _get_client()
        client.create_collection(
            name,
            metadata={
                "docs_version": _DOCS_TAG,
                "embedding": "ollama/mxbai-embed-large",
            },
        ).add(ids=["1"], documents=["x"], embeddings=[[0.1, 0.2, 0.3]])

    def test_stale_collection_is_not_built_and_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from aiida_agents.rag.store import index_status

        monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path / "vdb"))
        # Built before the corpus-format tag existed: on disk, full, unreachable.
        stale = f"aiida_docs__{_DOCS_TAG}__ollama_mxbai-embed-large"
        self._seed(str(tmp_path), stale)

        status = index_status()

        assert status.built is False
        assert [c.name for c in status.stale] == [stale]
        assert all(c.usable is False for c in status.collections)

    def test_current_collection_is_built_and_usable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The name the retrieval path would look for counts as built."""
        from aiida_agents.rag.store import _core_name_for, index_status

        monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path / "vdb"))
        monkeypatch.setenv("AIIDA_AGENTS_EMBED_BACKEND", "ollama")
        monkeypatch.setenv("AIIDA_AGENTS_EMBED_MODEL", "mxbai-embed-large")
        current = _core_name_for("ollama/mxbai-embed-large")
        self._seed(str(tmp_path), current)

        status = index_status()

        assert status.built is True
        assert status.stale == ()

    def test_status_name_matches_what_retrieval_looks_up(self) -> None:
        """The offline-derived name and the live-embedder name must agree.

        These are computed by different paths -- configuration vs a constructed
        embedder -- and their divergence is the whole bug, so pin them together.
        """
        from aiida_agents.rag.embeddings import configured_embedding_name
        from aiida_agents.rag.store import _collection_name, _core_name_for

        embed = MagicMock()
        embed.name.return_value = "ollama/mxbai-embed-large"

        assert _core_name_for(configured_embedding_name()) == _collection_name(embed)
