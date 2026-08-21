"""Unit tests for the retrieval-only RAG HTTP API.

The app is exercised through FastAPI's ``TestClient`` with ``query_docs`` and
``docs_index_available`` monkeypatched, so no ChromaDB store or embedder is
needed, this tests the HTTP contract, not retrieval itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aiida_agents.rag.server import create_app

_SAMPLE_HIT = {
    "text": "To restart a work chain, ...",
    "source": "howto/restart.txt",
    "section": "Restarting",
    "corpus": "aiida-core",
    "url": "https://aiida.readthedocs.io/en/v2.8.0/howto/restart.html#restarting",
}


def test_health_never_touches_the_index() -> None:
    resp = TestClient(create_app()).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_returns_ranked_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aiida_agents.rag.retriever.docs_index_available", lambda: True)
    monkeypatch.setattr(
        "aiida_agents.rag.query_docs", lambda query, limit=3: [_SAMPLE_HIT]
    )

    resp = TestClient(create_app()).post(
        "/search", json={"query": "how do I restart a work chain?", "limit": 3}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "how do I restart a work chain?"
    assert body["results"] == [_SAMPLE_HIT]


def test_search_reports_an_unbuilt_index_as_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing index is 503, distinct from a 200 with no matches, so the
    caller can tell "not set up" apart from "nothing matched"."""
    monkeypatch.setattr(
        "aiida_agents.rag.retriever.docs_index_available", lambda: False
    )
    resp = TestClient(create_app()).post("/search", json={"query": "anything"})
    assert resp.status_code == 503
    assert "rag build" in resp.json()["detail"]


def test_search_rejects_an_empty_query() -> None:
    """The ``min_length=1`` constraint is enforced by the request schema."""
    resp = TestClient(create_app()).post("/search", json={"query": ""})
    assert resp.status_code == 422
