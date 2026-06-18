"""ChromaDB client, persistence path, and collection naming for the RAG index.

Shared by the indexing and retrieval paths so they agree on where the store
lives and how a collection is named. The persistence path defaults to
``.aiida_agents_vector_db/`` and is overridable via
``AIIDA_AGENTS_VECTOR_DB_PATH`` (read through ``RagSettings``).
"""

from __future__ import annotations

import re
from typing import Any

import chromadb

from aiida_agents._settings import RagSettings
from aiida_agents.rag.embeddings import EmbeddingFunction

_COLLECTION_PREFIX = "aiida_docs"
_DOCS_TAG = "v2.8.0"  # pinned aiida-core docs version; part of the index identity


def _get_client(settings: RagSettings | None = None) -> Any:
    cfg = settings if settings is not None else RagSettings()
    path = cfg.vector_db_path
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collection_name(embed_fn: EmbeddingFunction) -> str:
    """Collection name keyed by docs version and embedding model.

    Index- and query-time embeddings must use the same model, and therefore
    the same vector dimension, so the collection is keyed by both ``_DOCS_TAG``
    and ``embed_fn.name()``. A docs-version bump or an embedding-backend change
    resolves to a different collection name, which triggers a rebuild rather
    than silently serving a stale or dimension-incompatible index.
    """
    model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", embed_fn.name())
    return f"{_COLLECTION_PREFIX}__{_DOCS_TAG}__{model_slug}"
