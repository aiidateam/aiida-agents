"""RAG pipeline for AiiDA documentation retrieval.

Public API
----------
index_docs()
    Build (or rebuild) the ChromaDB vector index from the sphinx text corpus.
    Call once before querying; subsequent calls are no-ops unless ``force=True``.

query_docs(query, limit=3)
    Low-level semantic search — returns raw result dicts with ``text``,
    ``source``, and ``section`` keys.

search_aiida_docs(query)
    Pydantic AI tool — wraps ``query_docs`` with formatted output suitable for
    an LLM context window.  Register this directly in an ``Agent(tools=[...])``.
"""

from __future__ import annotations

from aiida_agents.rag.retriever import index_docs, query_docs

__all__ = ["index_docs", "query_docs", "search_aiida_docs"]


def search_aiida_docs(query: str) -> str:
    """Search the AiiDA v2.8 documentation for conceptual knowledge.

    Use this tool for questions about *what* AiiDA concepts are or *how* AiiDA
    works — e.g. what a CalcJobNode is, how to set up a WorkChain, what the
    provenance graph tracks, what KpointsData represents.  The corpus is the
    official sphinx-built prose docs pinned to v2.8, so prefer it over general
    knowledge for AiiDA-specific questions.

    Do **not** use this for queries about specific processes or nodes in the
    user's own database — use the live DB tools for those.

    Args:
        query: A natural language question or keyword string.

    Returns:
        Up to three documentation excerpts, each prefixed with its source file
        and section heading, separated by horizontal rules.
    """
    results = query_docs(query, limit=3)
    if not results:
        return "No relevant AiiDA documentation found for this query."

    formatted = []
    for r in results:
        source = r.get("source", "unknown")
        section = r.get("section", "")
        text = r.get("text", "")
        header = f"[{source}  §  {section}]" if section else f"[{source}]"
        formatted.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted)
