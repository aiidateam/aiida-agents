"""Build the RAG corpus and index it into ChromaDB.

Runs the one-time :func:`index_docs` pipeline: sparse-clone the pinned aiida-core
docs, render them to fenced text (``rag._textbuild``), chunk, embed, and persist.
Heavy and network-bound, so it is a deliberate one-shot, not part of querying.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from aiida_agents._settings import RagSettings
from aiida_agents.rag.store import (
    _CORPUS_FORMAT,
    _DOCS_TAG,
    _collection_name,
    _collection_populated,
    _get_client,
)
from aiida_agents.rag.chunking import _load_docs
from aiida_agents.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

_DOCS_REPO = "https://github.com/aiidateam/aiida-core.git"
_DOCS_SUBDIR = "docs"  # need full docs/ for sphinx-build, not just source/


def _clone_and_build_text(target_dir: str) -> None:
    """Clone aiida-core docs and run sphinx-build -b text.

    Produces .txt files in target_dir — clean prose, no RST markup.
    Notebook execution is disabled (NB_EXECUTION_MODE=off) for faster builds.
    """
    # Fail fast, before the (slow) clone: the docs toolchain must be present in
    # THIS interpreter. We do not pip-install at runtime (it mutates the user's
    # env and fails on uv venvs with no bundled pip); it is an opt-in dependency.
    if importlib.util.find_spec("sphinx") is None:
        msg = (
            "Building the docs corpus needs the AiiDA docs toolchain, which "
            "is not installed in this environment. Install it with:\n"
            "    uv pip install 'aiida-core[docs]'\n"
            "then re-run indexing."
        )
        raise RuntimeError(msg)

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

        # Step 2: run the fenced text builder (sphinx text output with
        # ```lang fences around code blocks; see rag._textbuild) under THIS
        # interpreter (sys.executable), so it uses the environment that has
        # aiida installed, not a stray system sphinx-build on PATH.
        logger.info("running sphinx text build…")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aiida_agents.rag._textbuild",
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

        # Step 3: copy the .txt output to our persistent location. Guard against
        # a build that produced nothing, so the failure surfaces here with the
        # sphinx error rather than as an opaque FileNotFoundError from copytree.
        if not text_out.exists():
            msg = (
                f"sphinx produced no output (rc={result.returncode}); "
                f"the docs build failed. Last stderr:\n{result.stderr[-2000:]}"
            )
            raise RuntimeError(msg)
        # Publish atomically: copy into a sibling staging dir (a crash here
        # leaves any previous corpus untouched), then swap it into place with a
        # single rename. Copying straight into target_dir is not atomic: an
        # interrupted copytree would leave a truncated corpus that later runs
        # reuse as if it were complete.
        staging = f"{target_dir}.staging"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        shutil.copytree(str(text_out), staging)
        if Path(target_dir).exists():
            shutil.rmtree(target_dir)
        os.replace(staging, target_dir)

    logger.info("text corpus ready at %s", target_dir)


def index_docs(force: bool = False) -> None:
    """Build or rebuild the ChromaDB collection from aiida-core docs.

    The collection is keyed by docs version and embedding model (see
    :func:`aiida_agents.rag.store._collection_name`), so different versions or
    backends never share an index.

    The build is all-or-nothing: the corpus is rendered and chunked *before* the
    collection is (re)created, and any failure or interruption mid-embed drops
    the partially filled collection. The store therefore never keeps a silent
    empty or half-built stub that the next run would mistake for a finished index.

    The collection swap is delete-then-create, though: a ``force`` rebuild of an
    existing index that fails mid-embed leaves no collection at all (the drop
    covers the partial, and the old one was already replaced). It self-heals on
    the next successful run; a non-``force`` build never deletes a populated
    collection, so a good index is only ever at risk under ``force``.

    Args:
        force: If True, re-render the corpus and rebuild the collection even if a
            populated one already exists. A rebuild that yields no chunks leaves
            the existing index untouched.
    """
    cfg = RagSettings()
    client = _get_client(cfg)
    embed_fn = get_embedding_function(cfg)
    name = _collection_name(embed_fn)

    if not force and _collection_populated(client, name):
        logger.info("collection '%s' already exists — skipping index", name)
        return

    # Render and chunk the corpus BEFORE touching the collection, so a failed
    # clone/build or an empty corpus never leaves an empty collection behind. The
    # corpus format is part of the directory name, so a format bump regenerates
    # the corpus instead of reusing a stale cached rendering.
    text_dir = str(cfg.vector_db_path / f"aiida_text_corpus__{_CORPUS_FORMAT}")
    # ``force`` re-renders the corpus too: a cached corpus may itself be stale or
    # (before the atomic swap above) truncated, and a forced reindex must be able
    # to repair it, not just rebuild the collection from the same bad corpus.
    if force or not os.path.exists(text_dir) or not list(Path(text_dir).rglob("*.txt")):
        _clone_and_build_text(text_dir)
    else:
        logger.info("text corpus already exists at %s — skipping clone", text_dir)
    chunks = _load_docs(text_dir)

    if not chunks:
        logger.warning("no chunks loaded; leaving any existing index untouched")
        return

    # Only now that there are chunks to add do we replace any prior (possibly
    # empty or stale) collection.
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)
    collection = client.create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={
            "hnsw:space": "cosine",
            "docs_version": _DOCS_TAG,
            "embedding": embed_fn.name(),
        },
    )

    ids = [f"doc_{i}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "section": c["section"]} for c in chunks]

    batch = 50
    completed = False
    try:
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
        completed = True
    finally:
        # Ctrl-C or an embed error mid-loop would otherwise leave a partial
        # collection the next run treats as complete; drop it to stay atomic.
        if not completed:
            logger.warning(
                "indexing did not complete; removing partial collection '%s'", name
            )
            client.delete_collection(name)

    logger.info("indexed %d chunks into '%s'", len(texts), name)
