"""Unit tests for the RAG embedding providers.

All tests run without a live Ollama server — HTTP calls are patched so CI
stays fast and dependency-free.  The contract tested here is:

* ``OllamaEmbedding.__call__`` batches documents without a query prefix.
* ``OllamaEmbedding.embed_query`` prepends the mxbai query prefix.
* ``OllamaEmbedding._embed`` sub-batches internally and returns one vector
  per input text.
* ``get_embedding_function`` returns ``OllamaEmbedding`` when Ollama is
  reachable and ``SentenceTransformerEmbedding`` when it is not.
* The health-check URL stripping handles the ``/v1`` suffix correctly.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# White-box test of the embedding backends: the concrete classes and the
# query-prefix constant are internal (`_`-prefixed), aliased here to keep the
# test body readable.
from aiida_agents.rag.embeddings import (
    _MXBAI_QUERY_PREFIX as MXBAI_QUERY_PREFIX,
    _OllamaEmbedding as OllamaEmbedding,
    _SentenceTransformerEmbedding as SentenceTransformerEmbedding,
    get_embedding_function,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_VECTOR = [0.1, 0.2, 0.3]


def _make_urlopen_mock(vectors: list[list[float]]) -> MagicMock:
    """Return a context-manager mock that yields the given embedding vectors."""
    response_body = json.dumps({"embeddings": vectors}).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(
        return_value=MagicMock(read=MagicMock(return_value=response_body))
    )
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# OllamaEmbedding — initialisation
# ---------------------------------------------------------------------------


class TestOllamaEmbeddingInit:
    def test_default_base_url(self) -> None:
        emb = OllamaEmbedding()
        assert emb.base_url == "http://localhost:11434"

    def test_strips_v1_suffix(self) -> None:
        emb = OllamaEmbedding(base_url="http://localhost:11434/v1")
        assert emb.base_url == "http://localhost:11434"

    def test_trailing_slash_stripped(self) -> None:
        emb = OllamaEmbedding(base_url="http://localhost:11434/")
        assert emb.base_url == "http://localhost:11434"

    def test_custom_model(self) -> None:
        emb = OllamaEmbedding(model="nomic-embed-text")
        assert emb.model == "nomic-embed-text"

    def test_name(self) -> None:
        assert OllamaEmbedding().name() == "ollama/mxbai-embed-large"


# ---------------------------------------------------------------------------
# OllamaEmbedding — _embed (sub-batching)
# ---------------------------------------------------------------------------


class TestOllamaEmbeddingEmbed:
    def test_single_text_returns_one_vector(self) -> None:
        emb = OllamaEmbedding()
        mock_cm = _make_urlopen_mock([_FAKE_VECTOR])
        with patch("urllib.request.urlopen", return_value=mock_cm):
            result = emb._embed(["hello"])
        assert result == [_FAKE_VECTOR]

    def test_multiple_texts_returns_matching_vectors(self) -> None:
        texts = [f"text {i}" for i in range(5)]
        vectors = [[float(i)] * 3 for i in range(5)]
        # With _SUB_BATCH=10 all 5 fit in one call
        mock_cm = _make_urlopen_mock(vectors)
        emb = OllamaEmbedding()
        with patch("urllib.request.urlopen", return_value=mock_cm):
            result = emb._embed(texts)
        assert len(result) == 5

    def test_sub_batching_makes_multiple_requests(self) -> None:
        """15 texts with _SUB_BATCH=10 should produce 2 HTTP requests."""
        texts = [f"t{i}" for i in range(15)]
        call_count = 0

        def _fake_urlopen(req: Any, timeout: Any) -> Any:  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            n = len(json.loads(req.data)["input"])
            return _make_urlopen_mock([[0.0] * 3] * n)

        emb = OllamaEmbedding()
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = emb._embed(texts)

        assert call_count == 2
        assert len(result) == 15

    def test_uses_api_embed_endpoint(self) -> None:
        emb = OllamaEmbedding()
        captured: list[str] = []

        def _fake_urlopen(req: Any, timeout: Any) -> Any:  # noqa: ANN001
            captured.append(req.full_url)
            return _make_urlopen_mock([[0.1]])

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            emb._embed(["test"])

        assert captured[0].endswith("/api/embed")

    def test_payload_uses_input_field(self) -> None:
        emb = OllamaEmbedding()
        captured_payloads: list[dict[str, Any]] = []

        def _fake_urlopen(req: Any, timeout: Any) -> Any:  # noqa: ANN001
            captured_payloads.append(json.loads(req.data))
            return _make_urlopen_mock([[0.1]])

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            emb._embed(["hello"])

        assert "input" in captured_payloads[0]
        assert "prompt" not in captured_payloads[0]


# ---------------------------------------------------------------------------
# OllamaEmbedding — __call__ vs embed_query (prefix handling)
# ---------------------------------------------------------------------------


class TestOllamaEmbeddingPrefixes:
    def test_call_sends_text_without_prefix(self) -> None:
        emb = OllamaEmbedding()
        captured: list[dict[str, Any]] = []

        def _fake_urlopen(req: Any, timeout: Any) -> Any:  # noqa: ANN001
            captured.append(json.loads(req.data))
            return _make_urlopen_mock([[0.1]])

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            emb(["my document"])

        assert captured[0]["input"] == ["my document"]

    def test_embed_query_adds_mxbai_prefix(self) -> None:
        emb = OllamaEmbedding()
        captured: list[dict[str, Any]] = []

        def _fake_urlopen(req: Any, timeout: Any) -> Any:  # noqa: ANN001
            captured.append(json.loads(req.data))
            return _make_urlopen_mock([[0.1]])

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            emb.embed_query(["my question"])

        assert captured[0]["input"] == [MXBAI_QUERY_PREFIX + "my question"]


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedding
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbedding:
    def test_call_and_embed_query_return_same_result(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[[0.5, 0.6]])
        )

        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            emb = SentenceTransformerEmbedding()
            r1 = emb(["text"])
            r2 = emb.embed_query(["text"])

        assert r1 == r2

    def test_name(self) -> None:
        mock_model = MagicMock()
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            emb = SentenceTransformerEmbedding()
        assert emb.name() == "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# get_embedding_function — backend selection
# ---------------------------------------------------------------------------


class TestGetEmbeddingFunction:
    def test_returns_ollama_when_reachable(self) -> None:
        with patch("urllib.request.urlopen"):
            fn = get_embedding_function()
        assert isinstance(fn, OllamaEmbedding)

    def test_falls_back_to_sentence_transformers_when_ollama_unreachable(
        self,
    ) -> None:
        mock_model = MagicMock()
        with (
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        ):
            fn = get_embedding_function()
        assert isinstance(fn, SentenceTransformerEmbedding)

    def test_health_check_url_strips_v1_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rstrip('/v1') is a bug — verify the fixed endswith+slice logic."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        captured: list[str] = []

        def _fake_urlopen(url: Any, timeout: Any) -> Any:  # noqa: ANN001
            captured.append(url)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            get_embedding_function()

        # Must be the clean base URL, not a mangled one
        assert captured[0] == "http://localhost:11434"

    def test_sentence_transformers_backend_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIIDA_AGENTS_EMBED_BACKEND", "sentence-transformers")
        mock_model = MagicMock()
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            fn = get_embedding_function()
        assert isinstance(fn, SentenceTransformerEmbedding)
