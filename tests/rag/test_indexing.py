"""Unit tests for the RAG indexing pipeline.

The corpus half (git clone + sphinx-build) is integration-level and is exercised
by ``dev/rag/test_rag.py`` and manual dogfooding. The indexing half — the atomic
build: skip only a *populated* collection, rebuild an empty stub, drop a partial
collection on failure, and leave the prior index alone when the corpus is empty —
is unit-tested here with a real in-memory ChromaDB client and a fake embedder, so
no Ollama/network is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import chromadb
import pytest

from aiida_agents.rag.indexing import _clone_and_build_text, index_docs
from aiida_agents.rag.store import _collection_name, _collection_populated


def test_build_requires_docs_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sphinx -> RuntimeError with the install hint, before any clone."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, package=None: None,
    )
    with pytest.raises(RuntimeError, match=r"aiida-core\[docs\]"):
        _clone_and_build_text("/tmp/unused")


class _FakeEmbed:
    """A ChromaDB-compatible embedding function returning fixed vectors.

    Satisfies the ``EmbeddingFunction`` protocol (``__call__``/``name``/
    ``embed_query``) so it can key a collection name without a live backend.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def name(self) -> str:
        return "ollama/mxbai-embed-large"


def _wire_index_docs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    client: Any,
    embed: _FakeEmbed,
    chunks: list[dict[str, str]],
    clone: MagicMock | None = None,
) -> MagicMock:
    """Point index_docs at an in-memory client + fake embedder, stubbing the
    (slow, network-bound) corpus build. Returns the corpus-build mock."""
    clone_mock = clone if clone is not None else MagicMock()
    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path))
    monkeypatch.setattr(
        "aiida_agents.rag.indexing._get_client", lambda cfg=None: client
    )
    monkeypatch.setattr(
        "aiida_agents.rag.indexing.get_embedding_function", lambda cfg=None: embed
    )
    monkeypatch.setattr("aiida_agents.rag.indexing._clone_and_build_text", clone_mock)
    monkeypatch.setattr("aiida_agents.rag.indexing._load_docs", lambda text_dir: chunks)
    return clone_mock


def test_index_docs_drops_partial_collection_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An embed failure mid-build leaves no half-filled stub behind (the atomic
    try/finally); the next run therefore rebuilds rather than skipping."""
    client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))

    class _BoomEmbed(_FakeEmbed):
        def __call__(self, input: list[str]) -> list[list[float]]:
            raise RuntimeError("embed backend down")

    embed = _BoomEmbed()
    _wire_index_docs(
        monkeypatch,
        tmp_path,
        client=client,
        embed=embed,
        chunks=[{"text": "x", "source": "a.txt", "section": "A"}],
    )
    with pytest.raises(RuntimeError, match="embed backend down"):
        index_docs()
    assert _collection_populated(client, _collection_name(embed)) is False


def test_index_docs_skips_when_already_populated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A populated collection is not rebuilt and the slow corpus step is skipped."""
    client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
    embed = _FakeEmbed()
    name = _collection_name(embed)
    client.create_collection(name, embedding_function=embed).add(
        ids=["seed"], documents=["seed"], metadatas=[{"source": "s", "section": "S"}]
    )
    clone = _wire_index_docs(
        monkeypatch,
        tmp_path,
        client=client,
        embed=embed,
        chunks=[{"text": "new", "source": "a.txt", "section": "A"}],
    )
    index_docs(force=False)
    clone.assert_not_called()
    assert client.get_collection(name).count() == 1  # untouched


def test_index_docs_rebuilds_empty_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty stub is rebuilt, not skipped as 'already exists'."""
    client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
    embed = _FakeEmbed()
    name = _collection_name(embed)
    client.create_collection(name, embedding_function=embed)  # empty stub
    clone = _wire_index_docs(
        monkeypatch,
        tmp_path,
        client=client,
        embed=embed,
        chunks=[{"text": "c", "source": "a.txt", "section": "A"}],
    )
    index_docs(force=False)
    clone.assert_called_once()
    assert _collection_populated(client, name) is True


def test_index_docs_empty_corpus_leaves_existing_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rebuild that yields no chunks leaves the prior index untouched."""
    client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
    embed = _FakeEmbed()
    name = _collection_name(embed)
    client.create_collection(name, embedding_function=embed).add(
        ids=["seed"], documents=["seed"], metadatas=[{"source": "s", "section": "S"}]
    )
    _wire_index_docs(monkeypatch, tmp_path, client=client, embed=embed, chunks=[])
    index_docs(force=True)
    assert client.get_collection(name).count() == 1  # untouched
