# ADR-05: RAG over AiiDA docs — local embeddings, offline-first

> Seed — direction confirmed 2026-05-25; chunking strategy, collection schema, and retriever integration still to be specified during implementation.

## Context

The MCP tools (ADR-02) let the agent query database records, but they cannot answer conceptual questions:
*"What is the difference between a CalcJobNode and a WorkChainNode?"*
or *"How do I set up a KpointsData node?"*

The Diagnostic Agent (ADR-04) needs to map AiiDA exit codes and error messages to documented causes and remedies — knowledge that lives in the AiiDA documentation, not in the provenance graph.

The project targets local models (ADR-03) running on HPC clusters where outbound internet access is restricted or absent.
Any documentation retrieval pipeline must therefore be fully offline-capable.
Cloud embedding APIs (OpenAI, Cohere) and cloud-hosted vector stores are ruled out.

## Decision

Build a local, offline-first RAG pipeline over the official AiiDA documentation with the following components.

### Document source

Fetch and parse the AiiDA documentation from the `aiida-core` repository (`docs/source/`, MyST/Sphinx format).
Attach source path and section heading as metadata to each chunk so the agent can cite the origin page.

### Chunking

Recursive character splitting at ~500 characters with ~50-character overlap.
Overlap preserves sentence context at chunk boundaries without duplicating large blocks.
The target chunk size is deliberately small to fit within the context windows of the local models we target.

### Embedding model

Primary: `nomic-embed-text` running locally via Ollama.
It produces 768-dimensional embeddings, runs entirely on CPU, and is free.
Fallback: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace for environments without Ollama.
The embedding backend is selected at runtime from the `AIIDA_AGENTS_EMBED_BACKEND` environment variable (`ollama` / `sentence-transformers`), defaulting to `ollama`.

### Vector store

ChromaDB with a persistent local client.
The persistence path defaults to `.aiida_agent_vector_db/` under the repo root and is overridable via `AIIDA_AGENTS_VECTOR_DB_PATH`.
The directory is added to `.gitignore` — binary vector indices are not version-controlled.
ChromaDB runs in-process; no server setup or cloud account is required.

### Retriever integration

The RAG retriever is exposed as an MCP tool (`search_docs`) so the Diagnostic Agent can call it alongside the existing database tools.
It accepts a natural-language query, returns the top-k most similar chunks with their source metadata, and leaves synthesis to the agent.

## Consequences

- Documentation retrieval works fully offline, matching HPC cluster constraints.
- The embedding backend is swappable without changing agent code.
- The vector DB is rebuilt from source on first run and cached locally; rebuilds are triggered manually or by a CLI command, not on every agent startup.
- ChromaDB's in-process model means no separate service to manage, but also no concurrent multi-process access — acceptable for the single-user, local-dev target.
- Chunk quality directly affects retrieval quality; the chunking strategy will need empirical tuning against real queries.

## Alternatives considered

- **pgvector (Postgres extension).**
  Rejected: requires superuser privileges to install the extension, which is not available on shared HPC clusters.
  AiiDA already uses Postgres, so the dependency exists, but the extension install barrier is too high for a developer-friendly setup.
- **FAISS.**
  Rejected: in-memory only with no built-in persistence or metadata filtering; requires manual serialisation and a separate metadata store.
  ChromaDB provides both out of the box.
- **Qdrant.**
  Rejected: requires a running server (Docker or binary); adds operational overhead for a local-dev prototype.
  Revisit if multi-user or high-throughput retrieval becomes a requirement.
- **Weaviate.**
  Rejected: same server-dependency concern as Qdrant; heavier than needed for this scale.
- **OpenAI / Cohere embeddings.**
  Rejected: violates the offline-first requirement and introduces subscription costs.
- **No RAG; rely on model training data alone.**
  Rejected: local small models (ADR-03) have limited AiiDA-specific knowledge and hallucinate on domain-specific questions; grounding with retrieved documentation is the direct mitigation.