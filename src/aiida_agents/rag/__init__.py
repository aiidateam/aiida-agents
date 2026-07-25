"""RAG pipeline for AiiDA documentation retrieval.

Public API
----------
index_docs()
    Build (or rebuild) the ChromaDB vector index from the sphinx text corpus.
    Call once before querying; subsequent calls are no-ops unless ``force=True``.

index_plugin_corpora()
    Build (or rebuild) every installed plugin's contributed
    :class:`~aiida_agents.plugins.spec.RagCorpus`, each into its own collection.
    Error-isolated per corpus; see ``aiida_agents.rag.indexing``.

IndexOutcome
    Enum returned by ``index_docs``/``index_plugin_corpora`` (``BUILT`` /
    ``ALREADY_PRESENT`` / ``EMPTY_CORPUS`` / ``FAILED``) so a caller can report
    what the build actually did.

query_docs(query, limit=3)
    Low-level semantic search across the core docs and every installed
    plugin's corpus — returns raw result dicts with ``text``, ``source``,
    ``section``, and ``corpus`` keys.

search_aiida_docs(query)
    Pydantic AI tool — wraps ``query_docs`` with formatted output suitable for
    an LLM context window. Register this directly in an ``Agent(tools=[...])``.
"""

from __future__ import annotations

from aiida_agents.rag.indexing import IndexOutcome, index_docs, index_plugin_corpora
from aiida_agents.rag.retriever import docs_index_available, query_docs

__all__ = [
    "IndexOutcome",
    "index_docs",
    "index_plugin_corpora",
    "query_docs",
    "search_aiida_docs",
]


def search_aiida_docs(query: str) -> str:
    """Search the AiiDA documentation for conceptual knowledge.

    Use this tool for questions about *what* AiiDA concepts are or *how* AiiDA
    works — e.g. what a CalcJobNode is, how to set up a WorkChain, what the
    provenance graph tracks, what KpointsData represents. The corpus is the
    official sphinx-built prose docs pinned to v2.8, so prefer it over general
    knowledge for AiiDA-specific questions. Any installed plugin's own
    documentation corpus is searched alongside it, when the plugin contributes
    one (see ``aiida_agents.plugins``).

    Do **not** use this for queries about specific processes or nodes in the
    user's own database — use the live DB tools for those.

    Args:
        query: A natural language question or keyword string.

    Returns:
        Up to three documentation excerpts, each prefixed with its source file
        and section heading (and, for a plugin's own corpus, the plugin's
        corpus name), separated by horizontal rules.
    """
    results = query_docs(query, limit=3)
    if not results:
        if not docs_index_available():
            return (
                "The AiiDA documentation index has not been built yet, so "
                "documentation search is unavailable. It must be built once by "
                "running `aiida-agents rag build` in a shell. Tell the user to "
                "run that; do not answer AiiDA documentation questions from "
                "memory in the meantime."
            )
        return "No relevant AiiDA documentation found for this query."

    formatted = []
    for r in results:
        source = r.get("source", "unknown")
        section = r.get("section", "")
        text = r.get("text", "")
        corpus = r.get("corpus", "aiida-core")
        location = f"{source}  §  {section}" if section else source
        header = (
            f"[{location}]" if corpus == "aiida-core" else f"[{corpus}: {location}]"
        )
        formatted.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted)
