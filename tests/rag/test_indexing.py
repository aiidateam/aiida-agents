"""Unit tests for the RAG indexing pipeline.

The corpus half (git clone + sphinx-build) is integration-level and is exercised
by ``dev/rag/test_rag.py`` and manual dogfooding. The indexing half is unit-tested
here with a real ChromaDB client (on a per-test temp dir) and a fake embedder, so
no Ollama or network is needed: skip only a *populated* collection, rebuild an
empty stub, drop a partial collection on failure, leave a prior index alone when
the corpus is empty, and re-clone under ``force``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import chromadb
import pytest

from aiida_agents.plugins import LoadedPlugin, RagCorpus
from aiida_agents.rag.indexing import (
    IndexOutcome,
    _clone_and_build_text,
    index_docs,
    index_plugin_corpora,
    index_plugin_corpus,
)
from aiida_agents.rag.store import (
    _CORPUS_FORMAT,
    _collection_name,
    _collection_populated,
    _plugin_collection_name,
)


def test_build_requires_docs_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sphinx -> RuntimeError with the install hint, before any clone."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, package=None: None,
    )
    with pytest.raises(RuntimeError, match=r"aiida-agents\[rag-build\]"):
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
    name = _collection_name(embed)
    _wire_index_docs(
        monkeypatch,
        tmp_path,
        client=client,
        embed=embed,
        chunks=[{"text": "x", "source": "a.txt", "section": "A"}],
    )
    with pytest.raises(RuntimeError, match="embed backend down"):
        index_docs()
    # The partial collection must be *gone*, not merely empty: a failed add
    # leaves the collection present at count 0, so a `_collection_populated`
    # (count > 0) check would pass even if the `finally` drop never ran.
    assert name not in {c.name for c in client.list_collections()}


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
    assert index_docs(force=False) is IndexOutcome.ALREADY_PRESENT
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
    assert index_docs(force=False) is IndexOutcome.BUILT
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
    assert index_docs(force=True) is IndexOutcome.EMPTY_CORPUS
    assert client.get_collection(name).count() == 1  # untouched


def test_index_docs_force_reclones_and_rebuilds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """force=True re-clones even with a cached corpus on disk, and rebuilds a
    populated collection with the fresh chunks.

    Pins the ``force`` term of the clone guard: without it, an existing corpus
    dir (and populated collection) would skip both the clone and the rebuild.
    """
    client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
    embed = _FakeEmbed()
    name = _collection_name(embed)
    client.create_collection(name, embedding_function=embed).add(
        ids=["old"], documents=["old"], metadatas=[{"source": "s", "section": "S"}]
    )
    # A cached corpus on disk: without ``force`` the clone would be skipped.
    corpus = tmp_path / f"aiida_text_corpus__{_CORPUS_FORMAT}"
    corpus.mkdir(parents=True)
    (corpus / "cached.txt").write_text("cached")

    clone = _wire_index_docs(
        monkeypatch,
        tmp_path,
        client=client,
        embed=embed,
        chunks=[
            {"text": f"new {i}", "source": "a.txt", "section": "A"} for i in range(3)
        ],
    )
    assert index_docs(force=True) is IndexOutcome.BUILT
    clone.assert_called_once()  # force re-clones despite the cached corpus
    assert client.get_collection(name).count() == 3  # rebuilt with the new chunks


def test_index_status_empty_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A store that was never created reports not-built, with no collections."""
    from aiida_agents.rag.store import index_status

    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path / "absent"))
    status = index_status()
    assert status.built is False
    assert status.collections == ()
    assert status.store_exists is False


def test_index_status_reports_built_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A populated collection surfaces with its chunk count and build metadata."""
    from aiida_agents._settings import RagSettings
    from aiida_agents.rag.store import _get_client, index_status

    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path / "vdb"))
    embed = _FakeEmbed()
    client = _get_client(RagSettings())
    client.create_collection(
        "aiida_docs__v2.8.0__fenced1__ollama_mxbai-embed-large",
        embedding_function=embed,
        metadata={"docs_version": "v2.8.0", "embedding": "ollama/mxbai-embed-large"},
    ).add(ids=["a"], documents=["x"], metadatas=[{"source": "s", "section": "S"}])

    status = index_status()
    assert status.built is True
    assert len(status.collections) == 1
    info = status.collections[0]
    assert info.chunks == 1
    assert info.docs_version == "v2.8.0"
    assert info.embedding == "ollama/mxbai-embed-large"


# ---------------------------------------------------------------------------
# Plugin-contributed corpora
# ---------------------------------------------------------------------------


def _wire_plugin_indexing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    client: Any,
    embed: _FakeEmbed,
    chunks: list[dict[str, str]],
    clone: MagicMock | None = None,
) -> MagicMock:
    """Like _wire_index_docs, for the plugin-corpus indexing functions."""
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


class TestIndexPluginCorpus:
    def test_no_source_raises(self) -> None:
        """A corpus naming neither text_dir nor docs_repo cannot be built."""
        corpus = RagCorpus(name="qe", version="1.0")
        with pytest.raises(RuntimeError, match="no text_dir and no docs_repo"):
            index_plugin_corpus(corpus)

    def test_builds_directly_from_text_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A corpus with a pre-rendered text_dir is indexed with no clone."""
        client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        embed = _FakeEmbed()
        text_dir = tmp_path / "prebuilt_docs"
        text_dir.mkdir()
        corpus = RagCorpus(name="qe", version="1.0", text_dir=text_dir)
        clone = _wire_plugin_indexing(
            monkeypatch,
            tmp_path,
            client=client,
            embed=embed,
            chunks=[{"text": "x", "source": "a.txt", "section": "A"}],
        )

        outcome = index_plugin_corpus(corpus)

        assert outcome is IndexOutcome.BUILT
        clone.assert_not_called()
        name = _plugin_collection_name(embed, "qe", "1.0")
        assert _collection_populated(client, name) is True
        assert client.get_collection(name).metadata["corpus"] == "qe"

    def test_missing_text_dir_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A text_dir that does not exist on disk is a clean error, not a
        confusing empty-corpus result."""
        client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        embed = _FakeEmbed()
        corpus = RagCorpus(name="qe", version="1.0", text_dir=tmp_path / "nope")
        _wire_plugin_indexing(
            monkeypatch, tmp_path, client=client, embed=embed, chunks=[]
        )

        with pytest.raises(RuntimeError, match="does not exist"):
            index_plugin_corpus(corpus)

    def test_clones_docs_repo_with_the_corpus_own_ref_and_subdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A repo-backed corpus clones with its own repo/ref/subdir, not the
        core docs' pinned defaults."""
        client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        embed = _FakeEmbed()
        corpus = RagCorpus(
            name="qe",
            version="1.0",
            docs_repo="https://github.com/aiidateam/aiida-quantumespresso.git",
            docs_ref="v4.5.0",
            docs_subdir="docs",
        )
        clone = _wire_plugin_indexing(
            monkeypatch,
            tmp_path,
            client=client,
            embed=embed,
            chunks=[{"text": "x", "source": "a.txt", "section": "A"}],
        )

        assert index_plugin_corpus(corpus) is IndexOutcome.BUILT
        clone.assert_called_once()
        _, kwargs = clone.call_args
        assert kwargs["docs_repo"] == corpus.docs_repo
        assert kwargs["docs_tag"] == "v4.5.0"
        assert kwargs["docs_subdir"] == "docs"

    def test_skips_when_already_populated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A populated plugin collection is reused, like the core docs."""
        client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        embed = _FakeEmbed()
        corpus = RagCorpus(
            name="qe", version="1.0", docs_repo="https://example.invalid/qe.git"
        )
        name = _plugin_collection_name(embed, "qe", "1.0")
        client.create_collection(name, embedding_function=embed).add(
            ids=["seed"],
            documents=["seed"],
            metadatas=[{"source": "s", "section": "S"}],
        )
        clone = _wire_plugin_indexing(
            monkeypatch, tmp_path, client=client, embed=embed, chunks=[]
        )

        assert index_plugin_corpus(corpus, force=False) is IndexOutcome.ALREADY_PRESENT
        clone.assert_not_called()


class TestIndexPluginCorpora:
    def test_empty_when_no_plugins_discovered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("aiida_agents.rag.indexing.discover_plugins", lambda: ())
        assert index_plugin_corpora() == {}

    def test_indexes_every_discovered_corpus(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Every plugin's every corpus is indexed, keyed by plugin/corpus name."""
        client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        embed = _FakeEmbed()
        _wire_plugin_indexing(
            monkeypatch,
            tmp_path,
            client=client,
            embed=embed,
            chunks=[{"text": "x", "source": "a.txt", "section": "A"}],
        )
        qe_docs = tmp_path / "qe_docs"
        qe_docs.mkdir()
        plugins = (
            LoadedPlugin(
                name="quantumespresso",
                corpora=(RagCorpus(name="qe", version="1.0", text_dir=qe_docs),),
            ),
            LoadedPlugin(
                name="siesta",
                corpora=(
                    RagCorpus(name="siesta-docs", version="2.0", text_dir=qe_docs),
                ),
            ),
        )
        monkeypatch.setattr(
            "aiida_agents.rag.indexing.discover_plugins", lambda: plugins
        )

        outcomes = index_plugin_corpora()

        assert outcomes == {
            "quantumespresso/qe": IndexOutcome.BUILT,
            "siesta/siesta-docs": IndexOutcome.BUILT,
        }

    def test_isolates_one_corpus_failure_from_the_rest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One plugin's corpus failing to build must not stop the others."""
        client: Any = chromadb.PersistentClient(path=str(tmp_path / "chroma_db"))
        embed = _FakeEmbed()
        _wire_plugin_indexing(
            monkeypatch,
            tmp_path,
            client=client,
            embed=embed,
            chunks=[{"text": "x", "source": "a.txt", "section": "A"}],
        )
        good_docs = tmp_path / "good_docs"
        good_docs.mkdir()
        plugins = (
            LoadedPlugin(
                name="broken",
                corpora=(
                    RagCorpus(
                        name="broken-corpus",
                        version="1.0",
                        text_dir=tmp_path / "does_not_exist",
                    ),
                ),
            ),
            LoadedPlugin(
                name="good",
                corpora=(
                    RagCorpus(name="good-corpus", version="1.0", text_dir=good_docs),
                ),
            ),
        )
        monkeypatch.setattr(
            "aiida_agents.rag.indexing.discover_plugins", lambda: plugins
        )

        outcomes = index_plugin_corpora()

        assert outcomes["broken/broken-corpus"] is IndexOutcome.FAILED
        assert outcomes["good/good-corpus"] is IndexOutcome.BUILT
