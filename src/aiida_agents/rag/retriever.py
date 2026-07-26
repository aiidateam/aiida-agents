"""Semantic retrieval over the indexed AiiDA documentation.

Read-only counterpart to ``indexing``: embed the question and return the
closest chunks across the core docs collection and every installed plugin's
contributed corpus (see ``aiida_agents.plugins``), built by ``index_docs`` and
``index_plugin_corpora`` respectively. A collection is keyed by docs version,
corpus identity, and embedding model (see ``_store``), so a query only ever
hits an index built with the same embedding model.
"""

from __future__ import annotations

import logging
from typing import Any

from aiida_agents.plugins import discover_plugins
from aiida_agents.rag.store import (
    _DOCS_TAG,
    _collection_name,
    _collection_populated,
    _core_name_for,
    _get_client,
    _plugin_collection_name,
)
from aiida_agents.rag.embeddings import (
    EmbeddingFunction,
    configured_embedding_name,
    get_embedding_function,
)

logger = logging.getLogger(__name__)


def docs_index_available() -> bool:
    """True if a populated docs index exists -- the core corpus or any
    installed plugin's contributed corpus.

    Lets callers tell "the index was never built (or is empty)" apart from
    "queried, but nothing matched", so an unbuilt index can be reported instead
    of silently returning no results.
    """
    client = _get_client()
    embed_fn = get_embedding_function()
    if _collection_populated(client, _collection_name(embed_fn)):
        return True
    return any(
        _collection_populated(
            client, _plugin_collection_name(embed_fn, corpus.name, corpus.version)
        )
        for plugin in discover_plugins()
        for corpus in plugin.corpora
    )


def _query_one(
    client: Any,
    existing: set[str],
    name: str,
    embed_fn: EmbeddingFunction,
    query_vector: list[float],
    limit: int,
    corpus_label: str,
) -> list[dict[str, Any]]:
    """Query one collection by name, if it is among ``existing``.

    Each hit carries its raw ``_distance`` (cosine, lower is closer) so
    ``query_docs`` can rank hits from different collections against each other
    -- valid because every collection here is embedded with the same
    ``embed_fn``, so their vectors share one space.
    """
    if name not in existing:
        return []
    collection = client.get_collection(name=name, embedding_function=embed_fn)
    results = collection.query(query_embeddings=[query_vector], n_results=limit)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "section": meta.get("section", ""),
            "corpus": corpus_label,
            "_distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def _warn_missing_core_index(embed_fn: EmbeddingFunction, existing: set[str]) -> None:
    """Explain *why* the index is missing, not merely that it is.

    The confusing case is an index that was built and then became invisible:
    ``get_embedding_function`` silently falls back to sentence-transformers when
    Ollama fails a short health check, and the collection name encodes the
    embedding model, so a momentary blip makes a perfectly good index
    unreachable. Telling the user to rebuild there is wrong advice -- the
    rebuild would just produce a second index under the other name.
    """
    active = embed_fn.name()
    configured = configured_embedding_name()
    if configured != active and _core_name_for(configured) in existing:
        logger.warning(
            "an index exists for the configured embedding %r, but %r is active, "
            "so searches cannot reach it. This usually means Ollama was "
            "unreachable and the embedder fell back. Start Ollama (or set "
            "AIIDA_AGENTS_EMBED_BACKEND) rather than rebuilding.",
            configured,
            active,
        )
        return
    logger.warning(
        "no index for docs %s + embedding '%s'. Build it by running: "
        "aiida-agents rag build",
        _DOCS_TAG,
        active,
    )


#: Cosine-distance ceiling (mxbai-embed-large) above which a hit is treated as
#: noise rather than a real match. Calibrated empirically: genuinely relevant
#: hits against the aiida-core corpus land at 0.21-0.31, while queries with no
#: real answer in the corpus (off-topic entirely) land at 0.54-0.64. A query
#: that is AiiDA-domain-adjacent but about a concept the corpus doesn't cover
#: (e.g. a plugin-specific parameter with no plugin corpus indexed) can still
#: land in the 0.30-0.36 range and slip under this ceiling -- this threshold
#: catches "nothing in the corpus is even topically close", not "the closest
#: thing found isn't specific enough". The latter is why corpus coverage
#: (installed plugin corpora) matters as much as this cutoff does.
_MAX_RELEVANT_DISTANCE = 0.45


def query_docs(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Query the AiiDA docs with a natural language string.

    Searches the core aiida-core corpus and every installed plugin's
    contributed corpus in one pass, then returns the ``limit`` closest hits
    overall, ranked by embedding distance regardless of which corpus they came
    from. Each result carries a ``corpus`` key ("aiida-core" for the core
    docs, or the contributing plugin's corpus name), so a caller can attribute
    a plugin's docs as its own.

    Hits farther than :data:`_MAX_RELEVANT_DISTANCE` from the query are
    dropped before ``limit`` is applied: a distant "closest available" hit is
    noise, not a weak signal, and formatting it identically to a strong match
    invites a model to answer from it anyway (see ``search_aiida_docs``).

    The query is embedded with the mxbai query prefix (added inside
    ``OllamaEmbedding.embed_query``).

    Args:
        query: Natural language question.
        limit: Number of results to return.

    Returns:
        List of dicts with 'text', 'source', 'section', and 'corpus' keys,
        ordered by relevance across all searched corpora.
    """
    client = _get_client()
    embed_fn = get_embedding_function()
    existing = {c.name for c in client.list_collections()}

    core_name = _collection_name(embed_fn)
    if core_name not in existing:
        _warn_missing_core_index(embed_fn, existing)

    query_vector = embed_fn.embed_query([query])[0]
    candidates = _query_one(
        client, existing, core_name, embed_fn, query_vector, limit, "aiida-core"
    )
    for plugin in discover_plugins():
        for corpus in plugin.corpora:
            name = _plugin_collection_name(embed_fn, corpus.name, corpus.version)
            candidates.extend(
                _query_one(
                    client, existing, name, embed_fn, query_vector, limit, corpus.name
                )
            )

    candidates.sort(key=lambda c: c["_distance"])
    return [
        {
            "text": c["text"],
            "source": c["source"],
            "section": c["section"],
            "corpus": c["corpus"],
        }
        for c in candidates[:limit]
        if c["_distance"] <= _MAX_RELEVANT_DISTANCE
    ]
