"""ChromaDB client, persistence path, and collection naming for the RAG index.

Shared by the indexing and retrieval paths so they agree on where the store
lives and how a collection is named. The persistence path defaults to
``.aiida_agents_vector_db/`` and is overridable via
``AIIDA_AGENTS_VECTOR_DB_PATH`` (read through ``RagSettings``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb

from aiida_agents._settings import RagSettings
from aiida_agents.rag.embeddings import EmbeddingFunction

_COLLECTION_PREFIX = "aiida_docs"
_DOCS_TAG = "v2.8.0"  # pinned aiida-core docs version; part of the index identity

# Corpus rendering format; bump when the text-build output changes shape
# (e.g. "fenced1" = code blocks carry ```lang fences, see rag._textbuild), so
# existing indexes and cached corpora are rebuilt instead of silently reused.
_CORPUS_FORMAT = "fenced1"


def _get_client(settings: RagSettings | None = None) -> Any:
    cfg = settings if settings is not None else RagSettings()
    path = cfg.vector_db_path
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collection_populated(client: Any, name: str) -> bool:
    """True if a collection named ``name`` exists and holds at least one entry.

    An empty collection counts as absent: a failed or interrupted build can leave
    a zero-row stub, and both the indexer (skip guard) and the retriever (missing
    -index message) must treat that stub as "not built", not as a finished index.
    """
    for collection in client.list_collections():
        if collection.name == name:
            count: int = collection.count()
            return count > 0
    return False


@dataclass(frozen=True)
class CollectionInfo:
    """A persisted collection's user-facing facts (from its build-time metadata)."""

    name: str
    chunks: int
    docs_version: str
    embedding: str


@dataclass(frozen=True)
class IndexStatus:
    """Network-free snapshot of the RAG store, for ``aiida-agents rag status``."""

    store_path: str
    store_exists: bool
    configured_docs_version: str
    configured_embed_backend: str
    configured_embed_model: str
    collections: tuple[CollectionInfo, ...]

    @property
    def built(self) -> bool:
        """True if any populated collection is present (a usable index exists)."""
        return any(c.chunks > 0 for c in self.collections)


def index_status(settings: RagSettings | None = None) -> IndexStatus:
    """Introspect the persisted store directly, without a live embedder.

    Reads the collections and their build-time metadata straight from disk, so
    it reports the real persisted state regardless of whether Ollama is
    reachable (constructing the live embedding function would probe the network
    and could silently fall back to a different backend, and thus a different
    collection name).
    """
    cfg = settings if settings is not None else RagSettings()
    if not cfg.vector_db_path.exists():
        collections: tuple[CollectionInfo, ...] = ()
    else:
        client = _get_client(cfg)
        collections = tuple(
            CollectionInfo(
                name=c.name,
                chunks=c.count(),
                docs_version=(c.metadata or {}).get("docs_version", "unknown"),
                embedding=(c.metadata or {}).get("embedding", "unknown"),
            )
            for c in client.list_collections()
        )
    return IndexStatus(
        store_path=str(cfg.vector_db_path),
        store_exists=cfg.vector_db_path.exists(),
        configured_docs_version=_DOCS_TAG,
        configured_embed_backend=str(cfg.embed_backend),
        configured_embed_model=cfg.embed_model,
        collections=collections,
    )


def _collection_name(embed_fn: EmbeddingFunction) -> str:
    """Collection name keyed by docs version, corpus format, and embedding model.

    Index- and query-time embeddings must use the same model, and therefore
    the same vector dimension, so the collection is keyed by both ``_DOCS_TAG``
    and ``embed_fn.name()``. A docs-version bump, a corpus-format change, or an
    embedding-backend change resolves to a different collection name, which
    triggers a rebuild rather than silently serving a stale or
    dimension-incompatible index.
    """
    model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", embed_fn.name())
    return f"{_COLLECTION_PREFIX}__{_DOCS_TAG}__{_CORPUS_FORMAT}__{model_slug}"
