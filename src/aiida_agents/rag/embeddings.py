"""Embedding providers for the RAG pipeline.

Primary: mxbai-embed-large via Ollama (local, 1024-dim).
Fallback: sentence-transformers/all-MiniLM-L6-v2 (CPU, no server needed).

Backend selection (``AIIDA_AGENTS_EMBED_BACKEND`` via ``RagSettings``):
  ollama                — local Ollama server (default)
  sentence-transformers — HuggingFace sentence-transformers
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Protocol, cast

from aiida_agents._settings import OllamaSettings, RagSettings

logger = logging.getLogger(__name__)

# mxbai-embed-large query prefix (documents need no prefix)
_MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingFunction(Protocol):
    """Minimal interface expected by ChromaDB for a custom embedding function."""

    def __call__(self, input: list[str]) -> list[list[float]]: ...
    def name(self) -> str: ...
    def embed_query(self, input: list[str]) -> list[list[float]]: ...


class _OllamaEmbedding:
    """Embeddings via a locally running Ollama server.

    Uses mxbai-embed-large by default.
    - __call__()    → index time, no prefix needed
    - embed_query() → query time, adds mxbai query prefix

    Uses the modern /api/embed endpoint (not the deprecated /api/embeddings),
    which supports batching, returns float32, and performs L2 normalisation.
    """

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        self.model = model
        # Strip the /v1 suffix correctly: we call /api/embed directly, not the
        # OpenAI-style /v1 routes. Use endswith+slice, not rstrip("/v1"): rstrip
        # treats its argument as a *set* of characters and would mangle a URL
        # like "http://localhost:11434" by stripping trailing chars from {/, v, 1}.
        base = base_url[:-3] if base_url.endswith("/v1") else base_url
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
        prefixed = [_MXBAI_QUERY_PREFIX + t for t in input]
        return self._embed(prefixed)

    def name(self) -> str:
        return f"ollama/{self.model}"


class _SentenceTransformerEmbedding:
    """Embeddings via sentence-transformers (CPU, no server required)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        # Imported lazily: the default backend is Ollama, and importing
        # sentence-transformers eagerly would force torch (and its CUDA wheel
        # stack) onto every install, even ones that never touch this fallback.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "The sentence-transformers fallback embedder is not installed. "
                "Either run a local Ollama server (the default 'ollama' backend) "
                "or install the optional extra: uv pip install 'aiida-agents[rag-fallback]'."
            )
            raise ImportError(msg) from exc
        self._model = SentenceTransformer(model)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return cast(list[list[float]], self._model.encode(input).tolist())

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def name(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function(
    rag_settings: RagSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
) -> EmbeddingFunction:
    """Return the configured embedding function.

    Args:
        rag_settings: RAG configuration. Read from env / ``.env`` if not given.
        ollama_settings: Ollama endpoint configuration, consulted only for the
            ``ollama`` backend. Read from env / ``.env`` if not given.

    Reads ``embed_backend`` (ollama | sentence-transformers). If Ollama is
    unreachable, falls back to sentence-transformers.
    """
    cfg = rag_settings if rag_settings is not None else RagSettings()

    if cfg.embed_backend == "ollama":
        ollama_cfg = (
            ollama_settings if ollama_settings is not None else OllamaSettings()
        )
        embedding = _OllamaEmbedding(
            model=cfg.embed_model, base_url=ollama_cfg.base_url
        )
        try:
            # Health-check the bare server root. ``embedding.base_url`` already
            # has the /v1 suffix stripped (the embedder calls /api/embed), so the
            # probe and the embedder agree on the endpoint rather than stripping
            # the URL in two places.
            urllib.request.urlopen(embedding.base_url, timeout=2)
            logger.debug("embedding backend: ollama (%s)", cfg.embed_model)
            return embedding
        except Exception:
            logger.warning("Ollama unreachable — falling back to sentence-transformers")

    logger.debug("embedding backend: sentence-transformers (all-MiniLM-L6-v2)")
    return _SentenceTransformerEmbedding()
