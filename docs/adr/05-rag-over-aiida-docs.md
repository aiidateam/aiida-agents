# ADR-05: RAG over AiiDA docs — local embeddings, offline-first

> Status: accepted (core approach). Implementation is landing in [#5](https://github.com/aiidateam/aiida-agents/pull/5); the index is built on demand via the CLI (`aiida-agents rag init`), with curation and version-keying still to implement (see *Build and versioning* and *Consequences*).

## Context

The MCP tools (ADR-02) let the agent query live database records, but they
cannot answer conceptual questions such as:

- *"What is the difference between a CalcJobNode and a WorkChainNode?"*
- *"How do I set up a KpointsData node?"*
- *"What does exit code 305 mean?"*

The Analysis Agent (ADR-04) needs to ground its answers in the official AiiDA
documentation rather than relying on the base model's general knowledge, which
is limited and prone to hallucination on AiiDA-specific terminology.

The project supports local, self-hosted LLMs (ADR-03) so users are not forced
onto a commercial API vendor (Anthropic, OpenAI, ...) for cost, privacy, and
control reasons. The retrieval pipeline follows the same principle: the
embedding model and vector store run locally rather than through a paid cloud
API, so RAG needs no third-party key or subscription. (This is about avoiding
a required cloud vendor, not about running air-gapped; building the corpus may
fetch the public docs over the network.)

## Decision

Build a local, offline-first RAG pipeline over the official AiiDA documentation
with the following components.

### Document source

**Sphinx text-builder output, not raw RST/MD.**

Early prototypes parsed the raw RST and Markdown files from `aiida-core`
directly. This produced noisy chunks full of `:ref:` directives, unresolved
`.. include::` references, and RST markup that embedding models cannot
understand. Retrieval quality was poor.

The correct approach, confirmed with the project mentor, is to run
`sphinx-build -b text` on the `docs/source/` directory. This produces clean
`.txt` files — plain prose with `Note:`/`Warning:` labels preserved, all
directives resolved, all includes expanded. The text corpus is version-matched
to `aiida-core v2.8.0` (pinned via `--branch v2.8.0` sparse clone).

Notebook execution is disabled during the build (`-D nb_execution_mode=off`)
so the corpus can be rebuilt quickly without a full AiiDA installation.

The text corpus (`.txt` files) is cached separately from the vector index, so
switching embedding models only requires re-embedding from the cached corpus,
not re-cloning and re-building Sphinx.

**What is excluded:**

- AiiDA tutorials repo — version drift and materials-science specificity;
  to be archived in favour of new tutorial modules.
- Plugin documentation — deferred to plugin-specific agents.
- API reference and Python source stubs — NL-trained embeddings match
  English questions against bare signatures poorly. API/source knowledge
  is better served by a read-only MCP introspection tool using `inspect`
  and `stubgen` (future work, noted in ADR-04).
- Release notes / changelog (`reference/_changelog.txt`): high-volume,
  low-conceptual-value text that outranks concept docs (empirically it
  surfaced at rank 1 for "What is a CalcJobNode?"). The corpus build must
  drop `reference/` and `_changelog`.

### Chunking

Section-based chunking over the Sphinx text output. Each `.txt` file is
parsed into `(heading, body)` pairs using the underline-based heading
convention that Sphinx text output preserves. Long sections are recursively
split at natural boundaries (paragraph → line → sentence) up to
`~2000 characters` per chunk.

Each chunk is prefixed with a breadcrumb:

```
KpointsData — AiiDA Topics > Data Types
<body text>
```

The section title appears first so the embedding is dominated by the topic
rather than the file path. Short sections below 150 characters are discarded
as noise.

The corpus produced 1,111 chunks from the v2.8 docs at an average of
~1,100 characters per chunk.

### Embedding model

**Primary: `mxbai-embed-large` via Ollama (local, 1024-dim).**

`nomic-embed-text` was the original choice. In practice it produced
near-random retrieval for AiiDA-specific terminology — `nomic` scored 0.67
for "KpointsData" while unrelated results scored 0.77. The root cause was
missing task prefixes (`search_document:` / `search_query:`), but even after
fixing the prefixes, domain-specific retrieval remained weak.

`mxbai-embed-large` (mixedbread-ai) outperforms `nomic-embed-text` on
technical domain retrieval and is SOTA on the MTEB benchmark. Its prefix
convention differs: no prefix at index time; at query time the prefix
`"Represent this sentence for searching relevant passages: "` is added.
This is handled transparently by `OllamaEmbedding.embed_query()`.

**Fallback: `sentence-transformers/all-MiniLM-L6-v2`** for environments
without Ollama (CI, offline dev). Selected automatically when Ollama is
unreachable.

The backend is selected at runtime from `AIIDA_AGENTS_EMBED_BACKEND`
(`ollama` / `sentence-transformers`), defaulting to `ollama`.

**Implementation note — Ollama API migration:**

The original implementation called the deprecated `/api/embeddings` endpoint
with a `"prompt"` field. Ollama ≥ 0.4.0 broke this with HTTP 500 errors. The
correct endpoint is `/api/embed` with an `"input"` field (list), which also
supports batching natively. A secondary bug in the health-check URL stripping
(`rstrip("/v1")` treats its argument as a character set, not a suffix, mangling
URLs) was fixed by replacing it with `endswith("/v1")` + slice. Embedding calls
are sub-batched at 10 texts per request to avoid timeouts on CPU hardware.

### Vector store

ChromaDB with a persistent local client. The persistence path defaults to
`.aiida_agents_vector_db/` and is overridable via `AIIDA_AGENTS_VECTOR_DB_PATH`.
The vector index is excluded from version control (`.gitignore`); only the
text corpus is cached. ChromaDB runs in-process with cosine similarity
(`hnsw:space = cosine`).

### Build and versioning

The index is built on demand by a one-time CLI step, `aiida-agents rag init`,
not shipped. It runs the full pipeline: clone the docs at the pinned tag,
`sphinx-build -b text`, curate, chunk, embed, and persist. The build needs the
AiiDA docs toolchain, installed explicitly with `uv pip install 'aiida-core[docs]'`; `rag init` checks for it and errors with that hint rather
than installing it silently. The clone +
`sphinx-build` runs under the active interpreter (`sys.executable -m sphinx`)
so it uses the environment that has `aiida` installed, not a stray system
`sphinx-build`.

The result is embedded into a ChromaDB collection keyed by
`(aiida-docs-version, embedding-model)` (e.g. `aiida_docs__v2.8.0__mxbai`).
Keying on both is required because index-time and query-time embeddings must
match: a collection built with mxbai (1024-dim) cannot be queried with MiniLM
(384-dim). A version bump or a backend change therefore triggers a rebuild
rather than silently serving a stale or dimension-incompatible index. The
build runs once and is cached; ordinary queries never rebuild.

Not chosen, for now: pre-building the corpus once in CI per `aiida-core`
release and shipping the ~1 MB chunked records in the wheel. That would spare
every user the clone + Sphinx build and give a byte-identical corpus across
users, at the cost of CI machinery and a committed artifact. Revisit if
first-run build friction becomes a real problem.

### Retriever integration

The RAG retriever is exposed as `search_aiida_docs(query)` — a plain Python
function registered directly in the Analysis agent's `tools=[]` list (ADR-04).
It lives in `aiida_agents/rag/__init__.py` alongside `index_docs()` and
`query_docs()`, keeping the RAG package's public API cohesive.

At query time, the question is embedded with the mxbai query prefix via
`embed_query()` and passed directly to `collection.query(query_embeddings=...)`
— bypassing ChromaDB's default `query_texts` path which would call `__call__`
(the document prefix) instead of `embed_query` (the query prefix).

## Consequences

- Retrieval runs entirely on local models and a local store, with no required
  cloud-vendor API or key. (`rag init` does need the network once, to clone
  the docs.)
- The text corpus is model-agnostic: re-embedding for a new model requires
  no re-clone or re-build.
- The embedding backend is swappable via environment variable without
  changing agent code.
- `aiida-agents rag init` builds and caches the index once; re-embedding is
  triggered by `index_docs(force=True)` or by a version/model change, not by
  ordinary queries.
- The exclusion list above is the *intended* corpus; the build must enforce
  it. As of this writing the changelog and API pages still leak into the
  index, to be filtered in the corpus build.
- ChromaDB's in-process model means no separate service to manage, acceptable
  for single-user local dev.
- Chunk quality directly affects retrieval quality; the section-based strategy
  outperformed fixed-size splitting on the AiiDA docs in manual evaluation.

## Alternatives considered

- **Raw RST/MD parsing.** Rejected: noisy output, unresolved includes,
  directive markup degrades embedding quality significantly.
- **`nomic-embed-text`.** Tried and rejected: weak domain-specific retrieval
  even with correct task prefixes; replaced by `mxbai-embed-large`.
- **pgvector.** Rejected: requires superuser privileges to install the
  extension, unavailable on shared HPC clusters.
- **FAISS.** Rejected: in-memory only, no built-in persistence or metadata
  filtering; requires manual serialisation.
- **Qdrant / Weaviate.** Rejected: require a running server, adding
  operational overhead for a local-dev prototype.
- **OpenAI / Cohere embeddings.** Rejected: would require a commercial API key
  and subscription, the vendor dependency we want to avoid; local embedding
  models meet the goal.
- **No RAG; rely on model training data.** Rejected: local small models have
  limited AiiDA-specific knowledge and hallucinate on domain terminology;
  retrieved documentation is the direct mitigation.
- **API reference in RAG.** Rejected: NL-trained embeddings match English
  questions against bare signatures poorly. Deferred to an `inspect`-based
  MCP introspection tool (future work).
