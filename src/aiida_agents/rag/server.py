"""A small read-only HTTP API over the documentation RAG index.

Exposes :func:`aiida_agents.rag.query_docs` as a JSON endpoint so a public site
(the AiiDA website) can offer a "search the docs" box. This tier is retrieval
only: no model, no API key, no AiiDA profile, retrieval hits ChromaDB and the
configured embedder and nothing else. Generating a natural-language answer is a
separate, heavier tier (an LLM call) and is deliberately not done here.

Run it with ``aiida-agents rag serve``. It is meant to sit behind a proxy or
rate limiter for public traffic; nothing here does authentication or throttling.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """A documentation search query."""

    query: str = Field(min_length=1, description="Natural-language question.")
    limit: int = Field(
        default=3, ge=1, le=20, description="Maximum number of matches to return."
    )


class SearchHit(BaseModel):
    """One retrieved documentation chunk, matching a :func:`query_docs` result."""

    text: str
    source: str
    section: str
    corpus: str
    url: str


class SearchResponse(BaseModel):
    """The ranked matches for a query, closest first."""

    query: str
    results: list[SearchHit]


def create_app(cors_origins: list[str] | None = None) -> FastAPI:
    """Build the retrieval-only RAG API.

    :param cors_origins: Browser origins allowed to call the API directly (e.g.
        ``["https://aiida.net"]``). Omit when the API is only reached server-side
        (through a proxy), since a same-origin caller needs no CORS grant.
    """
    app = FastAPI(
        title="aiida-agents docs search",
        summary="Read-only semantic search over the AiiDA documentation.",
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe: cheap, and never touches the index."""
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        """Return the closest documentation chunks for ``request.query``.

        A 503 (not an empty result) distinguishes "the index was never built"
        from "nothing in the index matched", so a caller can surface a setup
        error rather than a silent no-op.
        """
        from aiida_agents.rag import query_docs
        from aiida_agents.rag.retriever import docs_index_available

        if not docs_index_available():
            raise HTTPException(
                status_code=503,
                detail="Documentation index is not built. Run `aiida-agents rag build`.",
            )
        hits = query_docs(request.query, limit=request.limit)
        return SearchResponse(
            query=request.query, results=[SearchHit(**hit) for hit in hits]
        )

    return app
