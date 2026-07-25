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
from aiida_agents.rag.embeddings import EmbeddingFunction, configured_embedding_name

_COLLECTION_PREFIX = "aiida_docs"
_PLUGIN_COLLECTION_PREFIX = "aiida_agents_plugin_docs"
_DOCS_TAG = "v2.8.0"  # pinned aiida-core docs version; part of the index identity

# Corpus rendering format; bump when the text-build output changes shape
# (e.g. "fenced1" = code blocks carry ```lang fences, see rag._textbuild), so
# existing indexes and cached corpora are rebuilt instead of silently reused.
_CORPUS_FORMAT = "fenced1"


def _slug(value: str) -> str:
    """Sanitise a value for use inside a ChromaDB collection name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _core_name_for(embedding_name: str) -> str:
    """The core-docs collection name implied by an embedding's name.

    Takes the name rather than the embedder, so callers that must stay offline
    (``rag status``, ``doctor``) can derive it from configuration alone while
    still going through this one formatter. Two places building the name by
    hand is what let the reported and queried names drift apart.
    """
    return (
        f"{_COLLECTION_PREFIX}__{_DOCS_TAG}__{_CORPUS_FORMAT}__{_slug(embedding_name)}"
    )


def _plugin_name_for(embedding_name: str, corpus_name: str, corpus_version: str) -> str:
    """The plugin-corpus collection name implied by an embedding's name."""
    return (
        f"{_PLUGIN_COLLECTION_PREFIX}__{_slug(corpus_name)}__{_slug(corpus_version)}"
        f"__{_CORPUS_FORMAT}__{_slug(embedding_name)}"
    )


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
    corpus: str = "aiida-core"
    usable: bool = True
    """Whether a query would actually find this collection.

    Retrieval looks a collection up by exact name (``_collection_name`` /
    ``_plugin_collection_name``), so one built under a different docs version,
    corpus format, or embedding model is inert: still on disk, still full of
    chunks, and never returned. Reporting it as a plain collection is what let
    ``rag status`` call the index "built" while ``search_aiida_docs`` said it
    did not exist.
    """


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
        """True if a populated collection exists that a query would actually hit.

        Deliberately not "any populated collection": retrieval looks a
        collection up by exact name, so a stale one (older docs version, corpus
        format, or embedding model) is never returned. Counting it here is what
        made ``rag status`` and ``doctor`` report a healthy index while
        ``search_aiida_docs`` reported none.
        """
        return any(c.chunks > 0 and c.usable for c in self.collections)

    @property
    def stale(self) -> tuple[CollectionInfo, ...]:
        """Populated collections no query can reach, so they can be reported."""
        return tuple(c for c in self.collections if c.chunks > 0 and not c.usable)


def _reachable_collection_names(settings: RagSettings | None = None) -> frozenset[str]:
    """Every collection name a query could currently resolve to.

    The core docs plus each installed plugin's contributed corpus, named exactly
    as the retrieval path names them. Derived from configuration only -- no live
    embedder, no network -- so ``rag status`` and ``doctor`` keep reporting the
    persisted truth even with Ollama down.

    Plugin discovery is error-isolated already; a failure here would only
    mislabel a corpus as unreachable, so it degrades to the core name rather
    than breaking status output.
    """
    cfg = settings if settings is not None else RagSettings()
    embedding_name = configured_embedding_name(cfg)
    names = {_core_name_for(embedding_name)}

    try:
        from aiida_agents.plugins import discover_plugins

        for plugin in discover_plugins():
            for corpus in plugin.corpora:
                names.add(_plugin_name_for(embedding_name, corpus.name, corpus.version))
    except Exception:  # pragma: no cover - defensive; see docstring
        pass
    return frozenset(names)


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
        # The names retrieval would look for, derived from configuration rather
        # than a live embedder, so this stays offline (see the docstring).
        reachable = _reachable_collection_names(cfg)
        collections = tuple(
            CollectionInfo(
                name=c.name,
                chunks=c.count(),
                docs_version=(c.metadata or {}).get("docs_version", "unknown"),
                embedding=(c.metadata or {}).get("embedding", "unknown"),
                corpus=(c.metadata or {}).get("corpus", "aiida-core"),
                usable=c.name in reachable,
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
    return _core_name_for(embed_fn.name())


def _plugin_collection_name(
    embed_fn: EmbeddingFunction, corpus_name: str, corpus_version: str
) -> str:
    """Collection name for a plugin-contributed :class:`.spec.RagCorpus`.

    Keyed the same way as the core docs (corpus format + embedding model), plus
    the corpus's own name and version, so a plugin's corpus never collides with
    the core docs or with another plugin's, and a version bump on just that
    corpus rebuilds only its own collection.
    """
    return _plugin_name_for(embed_fn.name(), corpus_name, corpus_version)
