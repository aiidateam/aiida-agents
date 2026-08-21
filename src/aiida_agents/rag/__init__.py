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
    "search_aiida_examples",
]


_NO_MEMORY = (
    "Tell the user, and do NOT answer this from memory: without the docs you "
    "would be guessing at AiiDA's API, and a plausible-looking invented "
    "function name is worse for them than no answer."
)


def _unavailable_message() -> str:
    """Why documentation search is unavailable, and what actually fixes it.

    Distinguishes "never built" from "built, but the active embedding model
    differs from the one it was built with" -- the second is the confusing one,
    and telling the user to rebuild there is wrong advice: it would just create
    a second index under the other name while the first stays unreachable.
    """
    from aiida_agents.rag.embeddings import configured_embedding_name
    from aiida_agents.rag.store import _core_name_for, _get_client

    # Both of these are offline: the configured name comes from settings, and
    # listing collections only reads the store. Naming the *active* embedder
    # would need a live probe, so it is attempted separately and only to make
    # the message more specific -- never to decide which message to give.
    try:
        configured = configured_embedding_name()
        existing = {c.name for c in _get_client().list_collections()}
    except Exception:  # pragma: no cover - diagnosis is best-effort
        return (
            "The AiiDA documentation index is unavailable, so documentation "
            "search cannot be used. It may need building with `aiida-agents "
            f"rag build`. {_NO_MEMORY}"
        )

    if _core_name_for(configured) in existing:
        return (
            f"An AiiDA documentation index exists, built with the embedding "
            f"model '{configured}', but a different embedder is active, so it "
            "cannot be searched. This usually means the local Ollama server is "
            "not running and the embedder fell back. Tell the user to start "
            "Ollama (or set AIIDA_AGENTS_EMBED_BACKEND to match) -- rebuilding "
            f"the index is NOT the fix here. {_NO_MEMORY}"
        )

    return (
        "The AiiDA documentation index has not been built yet, so "
        "documentation search is unavailable. It must be built once by "
        "running `aiida-agents rag build` in a shell. Tell the user to run "
        f"that. {_NO_MEMORY}"
    )


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
        corpus name), then the URL of the page it came from, separated by
        horizontal rules.

        **Include that URL in your answer** whenever you use an excerpt --- as
        a Markdown link on the concept it supports, not a bare list at the end.
        Finding the right page is the hardest part of these docs for most
        users, and the link is the part of the answer they can act on. Only
        link a URL this tool returned: do not construct one yourself, and do
        not cite a page you were not given.
    """
    results = query_docs(query, limit=3)
    if not results:
        if not docs_index_available():
            return _unavailable_message()
        return (
            "No relevant AiiDA documentation found for this query -- the "
            "index was searched, but nothing in it is a close enough match, "
            "which usually means this is covered by a plugin's docs that "
            "aren't indexed here, or isn't documented at all. "
            f"{_NO_MEMORY}"
        )

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
        url = r.get("url", "")
        formatted.append(f"{header}\n{url}\n{text}" if url else f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted)


#: Fence languages that mark a block as Python. The corpus is rendered by
#: ``rag._textbuild``, which carries each literal block's sphinx language
#: through into the fence, so this is a reliable marker rather than a guess.
_PYTHON_FENCES = ("```python", "```py\n", "```pycon", "```default")

#: How many hits to pull before filtering down to the ones carrying code. The
#: corpus is mostly prose, so the closest three chunks to "how do I query"
#: are often all explanation; casting wider and then filtering finds the
#: worked example that actually answers a code request.
_CODE_SEARCH_POOL = 12
_CODE_SEARCH_LIMIT = 4


def _has_python(text: str) -> bool:
    return any(fence in text for fence in _PYTHON_FENCES)


def search_aiida_examples(task: str) -> str:
    """Find real AiiDA code examples for a task, to write code from.

    Use this **before writing any Python that uses AiiDA**, and prefer it over
    ``search_aiida_docs`` when the user wants a snippet, a script, or "how do I
    do X in code". It returns worked examples from the official documentation
    at the pinned version, so the APIs in them exist and have the signatures
    shown.

    Write your code from what this returns. Do **not** write AiiDA code from
    memory: a plausible-looking method name that does not exist costs the user
    a traceback and their trust, and it is the single most common way this
    fails. If nothing here covers what you were asked, say so and show the
    closest thing it did return rather than inventing the rest.

    Args:
        task: What the code needs to do, in natural language --- e.g.
            "query for finished PwCalculations ordered by creation time".

    Returns:
        Up to four documentation excerpts that contain Python, each with its
        source, section and a link to the page it came from.
    """
    results = [
        r
        for r in query_docs(task, limit=_CODE_SEARCH_POOL)
        if _has_python(r.get("text", ""))
    ]
    if not results:
        if not docs_index_available():
            return _unavailable_message()
        return (
            "No AiiDA code examples found for this task -- the documentation "
            "was searched, but nothing matching it contains a code example. "
            "Say so, and do NOT write the code from memory: an invented API "
            "is worse for the user than telling them the docs do not cover "
            "this. Suggest `search_aiida_docs` for the concept, or point them "
            "at the API reference."
        )

    formatted = []
    for r in results[:_CODE_SEARCH_LIMIT]:
        source = r.get("source", "unknown")
        section = r.get("section", "")
        corpus = r.get("corpus", "aiida-core")
        location = f"{source}  §  {section}" if section else source
        header = (
            f"[{location}]" if corpus == "aiida-core" else f"[{corpus}: {location}]"
        )
        url = r.get("url", "")
        body = (
            f"{header}\n{url}\n{r.get('text', '')}"
            if url
            else (f"{header}\n{r.get('text', '')}")
        )
        formatted.append(body)

    return "\n\n---\n\n".join(formatted)
