"""Semantic retrieval over the indexed AiiDA documentation.

Read-only counterpart to ``indexing``: embed the question and return the
closest chunks from the ChromaDB collection built by ``index_docs``. The
collection is keyed by docs version and embedding model (see ``_store``), so a
query only ever hits an index built with the same embedding model.
"""

from __future__ import annotations

import logging

from aiida_agents.rag.store import _DOCS_TAG, _collection_name, _get_client
from aiida_agents.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


def query_docs(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Query the AiiDA docs with a natural language string.

    The query is embedded with the mxbai query prefix (added inside
    ``OllamaEmbedding.embed_query``).

    Args:
        query: Natural language question.
        limit: Number of results to return.

    Returns:
        List of dicts with 'text', 'source', and 'section' keys,
        ordered by relevance.
    """
    client = _get_client()
    embed_fn = get_embedding_function()
    name = _collection_name(embed_fn)
    existing = [c.name for c in client.list_collections()]

    if name not in existing:
        logger.warning(
            "no index for docs %s + embedding '%s' — run `aiida-agents rag init`",
            _DOCS_TAG,
            embed_fn.name(),
        )
        return []

    collection = client.get_collection(name=name, embedding_function=embed_fn)

    # ChromaDB's query_texts path would embed via __call__ (the document side,
    # no prefix). We embed the query ourselves via embed_query (which adds the
    # mxbai query prefix) and pass the vector directly through query_embeddings.
    query_vector = embed_fn.embed_query([query])[0]
    results = collection.query(query_embeddings=[query_vector], n_results=limit)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "section": meta.get("section", ""),
        }
        for doc, meta in zip(docs, metas)
    ]
