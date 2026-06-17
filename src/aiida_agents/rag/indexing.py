"""Build the RAG corpus and index it into ChromaDB.

Runs the one-time pipeline behind ``aiida-agents rag init``: sparse-clone the
pinned aiida-core docs, ``sphinx-build -b text`` them, chunk, embed, and
persist. Heavy and network-bound, so it is a deliberate one-shot, not part of
querying.
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

from aiida_agents.rag.store import (
    _DOCS_TAG,
    _collection_name,
    _get_client,
    _get_db_path,
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

        # Step 2: run the sphinx text builder under THIS interpreter
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

        # Step 3: copy the .txt output to our persistent location. Guard against
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


def index_docs(force: bool = False) -> None:
    """Build or rebuild the ChromaDB collection from aiida-core docs.

    The collection is keyed by docs version and embedding model (see
    :func:`aiida_agents.rag.store._collection_name`), so different versions or
    backends never share an index.

    Args:
        force: If True, delete and rebuild even if the collection exists.
    """
    client = _get_client()
    embed_fn = get_embedding_function()
    name = _collection_name(embed_fn)
    existing = [c.name for c in client.list_collections()]

    if name in existing and not force:
        logger.info("collection '%s' already exists — skipping index", name)
        return

    if name in existing:
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

    logger.info("indexed %d chunks into '%s'", len(texts), name)
