"""ChromaDB-backed retriever for AiiDA documentation.

Workflow:
  1. Sparse-clone aiida-core (v2.8 tag) — only the docs/ directory.
  2. Run `sphinx-build -b text` to produce clean plain-text files.
     This strips all RST/MD directives, resolves includes, and keeps
     Note:/Warning: labels in the output.
  3. Parse the .txt files into section-based chunks using the plain
     underline headings that Sphinx text output preserves.
  4. Embed each chunk with the Nomic "search_document:" prefix and
     store in ChromaDB.
  5. At query time, embed the question with "search_query:" prefix
     and return the closest chunks.

Why sphinx-build -b text instead of raw RST:
  - Resolves all .. include:: directives (missing content in raw clone)
  - Strips :ref:, :py:class:, .. code-block::, .. note:: markup
  - Produces clean prose the embedding model can actually understand
  - Version-matched: we pin to aiida-core v2.8

Why Nomic task prefixes matter:
  nomic-embed-text was contrastively trained with "search_document:"
  and "search_query:" prefixes. Without them retrieval is near-random.
  The prefixes are added by OllamaEmbedding — NOT added here manually.

DB path defaults to .aiida_agent_vector_db/, overridable via
AIIDA_AGENTS_VECTOR_DB_PATH.
"""

from __future__ import annotations

import chromadb
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from aiida_agents.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "aiida_docs"
_TARGET_CHUNK_CHARS = 2000  # ~512 tokens; 2026 benchmark sweet spot
_MIN_CHUNK_CHARS = 150  # discard stubs shorter than this

_DOCS_REPO = "https://github.com/aiidateam/aiida-core.git"
_DOCS_TAG = "v2.8.0"  # pin to v2.8
_DOCS_SUBDIR = "docs"  # need full docs/ for sphinx-build, not just source/


def _get_db_path() -> str:
    return os.getenv("AIIDA_AGENTS_VECTOR_DB_PATH", ".aiida_agent_vector_db")


def _get_client() -> Any:
    path = _get_db_path()
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


# ---------------------------------------------------------------------------
# Sphinx text build
# ---------------------------------------------------------------------------


def _clone_and_build_text(target_dir: str) -> None:
    """Clone aiida-core docs and run sphinx-build -b text.

    Produces .txt files in target_dir — clean prose, no RST markup.
    Notebook execution is disabled (NB_EXECUTION_MODE=off) for faster builds.
    """

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir = tmp_path / "aiida-core"

        # Step 1: sparse-clone only docs/ at the v2.8 tag
        logger.info("cloning aiida-core %s (sparse, docs/ only)…", _DOCS_TAG)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                _DOCS_TAG,
                "--filter=blob:none",
                "--sparse",
                _DOCS_REPO,
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", _DOCS_SUBDIR],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
        )

        docs_dir = repo_dir / "docs"
        text_out = tmp_path / "text_build"

        # Step 2: require the docs toolchain in THIS interpreter. We do not
        # pip-install at runtime: that mutates the user's environment and fails
        # on uv venvs (no bundled pip). It is an opt-in dependency; fail early
        # with guidance if it is missing.
        try:
            import sphinx  # noqa: F401
        except ModuleNotFoundError as exc:
            msg = (
                "Building the docs corpus needs the AiiDA docs toolchain, which "
                "is not installed in this environment. Install it with:\n"
                "    uv pip install 'aiida-core[docs]'\n"
                "then re-run indexing."
            )
            raise RuntimeError(msg) from exc

        # Step 3: run the sphinx text builder under THIS interpreter
        # (sys.executable), so it uses the environment that has aiida installed,
        # not a stray system sphinx-build on PATH.
        # -E = don't use cached environment (fresh build)
        # -D nb_execution_mode=off = don't execute notebooks
        # -q = quiet (suppress most output)
        logger.info("running sphinx -b text (this takes ~30–60s)…")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "text",
                "-E",
                "-q",
                "-D",
                "nb_execution_mode=off",
                "-D",
                "jupyter_execute_notebooks=off",
                str(docs_dir / "source"),
                str(text_out),
            ],
            cwd=str(docs_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Log warnings but don't abort — partial builds still produce
            # useful output even when some pages fail
            logger.warning(
                "sphinx-build finished with warnings/errors (rc=%d):\n%s",
                result.returncode,
                result.stderr[-2000:],  # last 2000 chars
            )
        else:
            logger.info("sphinx text build complete")

        # Step 4: copy the .txt output to our persistent location. Guard against
        # a build that produced nothing, so the failure surfaces here with the
        # sphinx error rather than as an opaque FileNotFoundError from copytree.
        if not text_out.exists():
            msg = (
                f"sphinx produced no output (rc={result.returncode}); "
                f"the docs build failed. Last stderr:\n{result.stderr[-2000:]}"
            )
            raise RuntimeError(msg)
        if Path(target_dir).exists():
            shutil.rmtree(target_dir)
        shutil.copytree(str(text_out), target_dir)

    logger.info("text corpus ready at %s", target_dir)


# ---------------------------------------------------------------------------
# Parsing sphinx text output
# ---------------------------------------------------------------------------


def _extract_text_sections(text: str) -> list[tuple[str, str]]:
    """Parse a Sphinx .txt file into (heading, body) pairs.

    Sphinx text output uses the same underline convention as RST but
    the file is already clean prose — no directives, no :ref: noise.

    A heading looks like:
        KpointsData
        ===========
        Body text here...

    Returns list of (title, body) tuples.
    The first tuple has title='' for content before the first heading.
    """
    heading_chars = set("=-~^+#*")
    lines = text.splitlines(keepends=True)

    # Find all heading positions: line i is a heading title if line i+1
    # is an underline of the same length (or longer)
    heading_positions: list[int] = []  # index of the TITLE line
    for i in range(len(lines) - 1):
        title_line = lines[i].rstrip("\n")
        next_line = lines[i + 1].rstrip("\n")
        if (
            title_line.strip()  # title is non-empty
            and len(next_line) >= len(title_line.strip())  # underline is long enough
            and len(next_line) >= 3
            and all(c in heading_chars for c in next_line.strip())
            and len(set(next_line.strip())) == 1  # all same char
        ):
            heading_positions.append(i)

    if not heading_positions:
        return [("", text.strip())]

    sections: list[tuple[str, str]] = []

    # Content before first heading
    if heading_positions[0] > 0:
        pre = "".join(lines[: heading_positions[0]]).strip()
        if pre:
            sections.append(("", pre))

    for pos, title_idx in enumerate(heading_positions):
        title = lines[title_idx].strip()
        # Body starts 2 lines after title (title + underline)
        body_start = title_idx + 2
        # Body ends at the next heading title line (or EOF)
        body_end = (
            heading_positions[pos + 1]
            if pos + 1 < len(heading_positions)
            else len(lines)
        )
        body = "".join(lines[body_start:body_end]).strip()
        sections.append((title, body))

    return sections


def _split_large_text(text: str, max_chars: int) -> list[str]:
    """Recursively split text that exceeds max_chars.

    Tries to split at natural boundaries:
      1. Double newline (paragraph)
      2. Single newline (line)
      3. Sentence end
      4. Hard character split as last resort
    """
    if len(text) <= max_chars:
        return [text]

    for sep in ["\n\n", "\n", ". ", "? ", "! "]:
        parts = text.split(sep)
        if len(parts) < 2:
            continue

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > max_chars:
                    chunks.extend(_split_large_text(part, max_chars))
                    current = ""
                else:
                    current = part.strip()
        if current:
            chunks.append(current)

        if len(chunks) > 1:
            return [c for c in chunks if c.strip()]

    # Last resort: hard split
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _chunk_text(text: str, source: str) -> list[dict[str, str]]:
    """Split a sphinx .txt file into embeddable chunks.

    Each chunk gets a breadcrumb prefix:
        "KpointsData — AiiDA Topics > Data Types"

    The section title comes FIRST so the embedding is dominated by
    the topic, not the file path. The Nomic "search_document:" prefix
    is added by OllamaEmbedding.__call__(), not here.
    """
    sections = _extract_text_sections(text)
    chunks: list[dict[str, str]] = []

    # Build a readable topic label from the file path
    parts = Path(source).with_suffix("").parts  # drop .txt
    topic_label = (
        " > ".join(
            p.replace("_", " ").replace("-", " ").title()
            for p in parts[:-1]  # skip the filename itself
        )
        or "AiiDA"
    )

    for title, body in sections:
        if not body or len(body) < _MIN_CHUNK_CHARS:
            continue

        breadcrumb = (
            f"{title} — AiiDA {topic_label}" if title else f"AiiDA {topic_label}"
        )

        for sub in _split_large_text(body, _TARGET_CHUNK_CHARS):
            if len(sub.strip()) < _MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    "text": breadcrumb + "\n" + sub.strip(),
                    "source": source,
                    "section": title or "(preamble)",
                }
            )

    # Fallback for files with no headings
    if not chunks and len(text.strip()) >= _MIN_CHUNK_CHARS:
        chunks.append(
            {
                "text": f"AiiDA {topic_label}\n{text.strip()}",
                "source": source,
                "section": "(full file)",
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Load docs from the text corpus
# ---------------------------------------------------------------------------


def _load_docs(text_dir: str) -> list[dict[str, str]]:
    """Walk text_dir and return all chunks from .txt files."""
    chunks: list[dict[str, str]] = []
    for path in Path(text_dir).rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = str(path.relative_to(text_dir))
            chunks.extend(_chunk_text(text, source=rel))
        except Exception as exc:
            logger.warning("skipping %s: %s", path, exc)

    avg = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
    logger.info(
        "loaded %d chunks from %s  (avg %.0f chars/chunk)",
        len(chunks),
        text_dir,
        avg,
    )
    return chunks


# ---------------------------------------------------------------------------
# Index and query
# ---------------------------------------------------------------------------


def index_docs(force: bool = False) -> None:
    """Build or rebuild the ChromaDB collection from aiida-core docs.

    Args:
        force: If True, delete and rebuild even if the collection exists.
    """
    client = _get_client()
    existing = [c.name for c in client.list_collections()]

    if _COLLECTION_NAME in existing and not force:
        logger.info("collection '%s' already exists — skipping index", _COLLECTION_NAME)
        return

    if _COLLECTION_NAME in existing:
        client.delete_collection(_COLLECTION_NAME)

    embed_fn = get_embedding_function()
    collection = client.create_collection(
        name=_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    text_dir = os.path.join(_get_db_path(), "aiida_text_corpus")
    # Skip clone if text corpus already exists
    if not os.path.exists(text_dir) or not list(Path(text_dir).rglob("*.txt")):
        _clone_and_build_text(text_dir)
    else:
        logger.info("text corpus already exists at %s — skipping clone", text_dir)
    chunks = _load_docs(text_dir)

    if not chunks:
        logger.warning("no chunks loaded — collection will be empty")
        return

    ids = [f"doc_{i}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "section": c["section"]} for c in chunks]

    batch = 50
    for i in range(0, len(texts), batch):
        collection.add(
            ids=ids[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
        logger.debug(
            "indexed batch %d/%d",
            i // batch + 1,
            -(-len(texts) // batch),
        )

    logger.info("indexed %d chunks into '%s'", len(texts), _COLLECTION_NAME)


def query_docs(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Query the AiiDA docs with a natural language string.

    The query is embedded with the "search_query:" Nomic prefix
    (handled inside OllamaEmbedding.embed_query).

    Args:
        query: Natural language question.
        limit: Number of results to return.

    Returns:
        List of dicts with 'text', 'source', and 'section' keys,
        ordered by relevance.
    """
    client = _get_client()
    existing = [c.name for c in client.list_collections()]

    if _COLLECTION_NAME not in existing:
        logger.warning("collection not found — run index_docs() first")
        return []

    embed_fn = get_embedding_function()
    collection = client.get_collection(
        name=_COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    # IMPORTANT: ChromaDB's query_texts always calls __call__ (search_document prefix).
    # We must manually embed with embed_query (search_query prefix) and pass the
    # vector directly via query_embeddings so Nomic gets the correct task signal.
    query_vector = embed_fn.embed_query([query])[0]
    results = collection.query(query_embeddings=[query_vector], n_results=limit)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "section": meta.get("section", ""),
        }
        for doc, meta in zip(docs, metas)
    ]
