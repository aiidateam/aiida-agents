"""Embedding providers for the RAG pipeline.

Primary: mxbai-embed-large via Ollama (local, 1024-dim).
Fallback: sentence-transformers/all-MiniLM-L6-v2 (CPU, no server needed).

Backend selection (AIIDA_AGENTS_EMBED_BACKEND):
  ollama                — local Ollama server (default)
  sentence-transformers — HuggingFace sentence-transformers
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Protocol, cast

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# mxbai-embed-large query prefix (documents need no prefix)
MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingFunction(Protocol):
    """Minimal interface expected by ChromaDB for a custom embedding function."""

    def __call__(self, input: list[str]) -> list[list[float]]: ...
    def name(self) -> str: ...
    def embed_query(self, input: list[str]) -> list[list[float]]: ...


class OllamaEmbedding:
    """Embeddings via a locally running Ollama server.

    Uses mxbai-embed-large by default.
    - __call__()    → index time, no prefix needed
    - embed_query() → query time, adds mxbai query prefix

    Uses the modern /api/embed endpoint (not the deprecated /api/embeddings),
    which supports batching, returns float32, and performs L2 normalisation.
    """

    def __init__(
        self, model: str = "mxbai-embed-large", base_url: str | None = None
    ) -> None:
        self.model = model
        raw = (
            base_url
            if base_url is not None
            else os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )
        # Strip the /v1 suffix if present (we call /api/embed directly)
        base = raw[:-3] if raw.endswith("/v1") else raw
        self.base_url = base.rstrip("/") or "http://localhost:11434"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama /api/embed, sub-batching to avoid timeouts."""
        _SUB_BATCH = 10
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _SUB_BATCH):
            sub = texts[i : i + _SUB_BATCH]
            payload = json.dumps({"model": self.model, "input": sub}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                # /api/embed returns {"embeddings": [[...], [...]]}
                # (the old /api/embeddings returned {"embedding": [...]})
                all_embeddings.extend(data["embeddings"])

        return all_embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed documents for indexing — no prefix needed for mxbai."""
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed queries — adds mxbai query prefix."""
        prefixed = [MXBAI_QUERY_PREFIX + t for t in input]
        return self._embed(prefixed)

    def name(self) -> str:
        return f"ollama/{self.model}"


class SentenceTransformerEmbedding:
    """Embeddings via sentence-transformers (CPU, no server required)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return cast(list[list[float]], self._model.encode(input).tolist())

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def name(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function() -> EmbeddingFunction:
    """Return the configured embedding function.

    Checks AIIDA_AGENTS_EMBED_BACKEND (ollama | sentence-transformers).
    If Ollama is unreachable, falls back to sentence-transformers.
    """
    backend = os.getenv("AIIDA_AGENTS_EMBED_BACKEND", "ollama").lower()

    if backend == "ollama":
        try:
            # Strip /v1 suffix correctly — rstrip("/v1") is wrong because
            # rstrip treats its argument as a set of characters, not a suffix,
            # and will mangle URLs like "http://localhost:11434" by stripping
            # trailing chars from {/, v, 1}.
            raw_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            base = raw_url[:-3] if raw_url.endswith("/v1") else raw_url
            base = base.rstrip("/") or "http://localhost:11434"

            urllib.request.urlopen(base, timeout=2)
            model = os.getenv("AIIDA_AGENTS_EMBED_MODEL", "mxbai-embed-large")
            logger.debug("embedding backend: ollama (%s)", model)
            return OllamaEmbedding(model=model)
        except Exception:
            logger.warning("Ollama unreachable — falling back to sentence-transformers")

    logger.debug("embedding backend: sentence-transformers (all-MiniLM-L6-v2)")
    return SentenceTransformerEmbedding()
