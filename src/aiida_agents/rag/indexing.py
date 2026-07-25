"""Build the RAG corpus and index it into ChromaDB.

Runs the one-time pipeline behind ``aiida-agents rag build``: sparse-clone the
pinned aiida-core docs, render them to fenced text (``rag._textbuild``), chunk,
embed, and persist. Heavy and network-bound, so it is a deliberate one-shot, not
part of querying.

``index_plugin_corpus``/``index_plugin_corpora`` run the same pipeline for a
plugin-contributed :class:`~aiida_agents.plugins.spec.RagCorpus`, each into its
own collection (see ``rag.store._plugin_collection_name``), so a plugin's docs
never share an index with the core docs or with another plugin's.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from aiida_agents._settings import RagSettings
from aiida_agents.plugins import discover_plugins
from aiida_agents.plugins.spec import RagCorpus
from aiida_agents.rag.store import (
    _CORPUS_FORMAT,
    _DOCS_TAG,
    _collection_name,
    _collection_populated,
    _get_client,
    _plugin_collection_name,
)
from aiida_agents.rag.chunking import _load_docs
from aiida_agents.rag.embeddings import EmbeddingFunction, get_embedding_function

logger = logging.getLogger(__name__)

_DOCS_REPO = "https://github.com/aiidateam/aiida-core.git"
_DOCS_SUBDIR = "docs"  # need full docs/ for sphinx-build, not just source/


class IndexOutcome(Enum):
    """What an index build did, so a caller can report it precisely.

    Only ``BUILT`` (re)created the collection; ``ALREADY_PRESENT`` reused a
    populated one, ``EMPTY_CORPUS`` produced no chunks, and ``FAILED`` hit an
    error partway through (see ``index_plugin_corpora``) -- all three leave any
    existing index exactly as it was.
    """

    BUILT = "built"
    ALREADY_PRESENT = "already_present"
    EMPTY_CORPUS = "empty_corpus"
    FAILED = "failed"


def _clone_and_build_text(
    target_dir: str,
    *,
    docs_repo: str = _DOCS_REPO,
    docs_tag: str = _DOCS_TAG,
    docs_subdir: str = _DOCS_SUBDIR,
) -> None:
    """Clone a sphinx docs repo and run sphinx-build -b text.

    Produces .txt files in target_dir — clean prose, no RST markup.
    Notebook execution is disabled (NB_EXECUTION_MODE=off) for faster builds.

    Defaults to the pinned aiida-core docs; ``index_plugin_corpus`` calls this
    with a plugin's own ``docs_repo``/``docs_ref``/``docs_subdir`` instead.
    ``docs_subdir`` is assumed to contain a ``source/`` directory that is the
    actual sphinx source root (aiida-core's own layout: the full ``docs/`` tree
    is fetched, not just ``docs/source``, because its build needs files
    alongside ``source/`` too). A plugin whose docs don't fit that shape should
    ship pre-rendered text via ``RagCorpus.text_dir`` instead.
    """
    # Fail fast, before the (slow) clone: the docs toolchain must be present in
    # THIS interpreter. We do not pip-install at runtime (it mutates the user's
    # env and fails on uv venvs with no bundled pip); it is an opt-in dependency.
    if importlib.util.find_spec("sphinx") is None:
        msg = (
            "Building the docs corpus needs the AiiDA docs toolchain, which "
            "is not installed in this environment. Install it with:\n"
            "    uv pip install 'aiida-agents[rag-build]'\n"
            "then re-run indexing."
        )
        raise RuntimeError(msg)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir = tmp_path / "repo"

        # Step 1: sparse-clone only docs_subdir/ at the pinned tag/ref
        logger.info(
            "cloning %s %s (sparse, %s/ only)...", docs_repo, docs_tag, docs_subdir
        )
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                docs_tag,
                "--filter=blob:none",
                "--sparse",
                docs_repo,
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
            ["git", "sparse-checkout", "set", docs_subdir],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
        )

        docs_dir = repo_dir / docs_subdir
        text_out = tmp_path / "text_build"

        # Step 2: run the fenced text builder (sphinx text output with
        # ```lang fences around code blocks; see rag._textbuild) under THIS
        # interpreter (sys.executable), so it uses the environment that has
        # aiida installed, not a stray system sphinx-build on PATH.
        logger.info("running sphinx text build...")
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


def _build_collection(
    client: Any,
    name: str,
    chunks: list[dict[str, str]],
    embed_fn: EmbeddingFunction,
    metadata: dict[str, Any],
    progress: Callable[[int, int], None] | None,
) -> IndexOutcome:
    """Build collection ``name`` from ``chunks`` without risking the existing one.

    Shared by the core-docs and plugin-corpus indexers. The build goes into a
    staging collection and is swapped in only once every chunk is embedded, so
    a failure or Ctrl-C leaves whatever was already indexed exactly as it was.
    That matters most under ``force``: the previous shape deleted the good
    collection first and then spent minutes embedding, so a rebuild that timed
    out partway left the user with no index at all and no way back.

    Only the swap itself (drop the old, rename the new) is unprotected, and it
    is two metadata operations with no embedding between them. An empty
    ``chunks`` leaves any existing collection untouched.
    """
    if not chunks:
        logger.warning("no chunks loaded; leaving any existing index untouched")
        return IndexOutcome.EMPTY_CORPUS

    staging = f"{name}__building"
    # A previous run killed mid-build can leave staging behind; it is by
    # definition incomplete, so never reuse it.
    if staging in {c.name for c in client.list_collections()}:
        client.delete_collection(staging)
    collection = client.create_collection(
        name=staging,
        embedding_function=embed_fn,
        metadata=metadata,
    )

    ids = [f"doc_{i}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "section": c["section"]} for c in chunks]

    total = len(texts)
    if progress is not None:
        progress(0, total)  # signal embed start (with the total) so a UI can init
    batch = 50
    completed = False
    try:
        for i in range(0, total, batch):
            collection.add(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )
            if progress is not None:
                progress(min(i + batch, total), total)
            logger.debug("indexed batch %d/%d", i // batch + 1, -(-total // batch))
        completed = True
    finally:
        # Drop the partial build, never the live collection: a half-filled
        # staging collection the next run mistook for finished is the thing
        # this guards, and the existing index must survive the failure.
        if not completed:
            logger.warning(
                "indexing did not complete; removing partial build '%s' "
                "(any existing '%s' is left untouched)",
                staging,
                name,
            )
            client.delete_collection(staging)

    # Everything is embedded; now the swap.
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)
    collection.modify(name=name)

    logger.info("indexed %d chunks into '%s'", len(texts), name)
    return IndexOutcome.BUILT


def index_docs(
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> IndexOutcome:
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
        progress: Optional ``(done, total)`` callback invoked after each embed
            batch, so a caller (e.g. the CLI) can render a progress bar.

    Returns:
        An :class:`IndexOutcome`: ``BUILT`` when the collection was (re)created,
        ``ALREADY_PRESENT`` when a populated collection was reused, or
        ``EMPTY_CORPUS`` when the corpus yielded no chunks and any existing index
        was left untouched.
    """
    cfg = RagSettings()
    client = _get_client(cfg)
    embed_fn = get_embedding_function(cfg)
    name = _collection_name(embed_fn)

    if not force and _collection_populated(client, name):
        logger.info("collection '%s' already exists — skipping index", name)
        return IndexOutcome.ALREADY_PRESENT

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

    return _build_collection(
        client,
        name,
        chunks,
        embed_fn,
        metadata={
            "hnsw:space": "cosine",
            "docs_version": _DOCS_TAG,
            "embedding": embed_fn.name(),
            "corpus": "aiida-core",
        },
        progress=progress,
    )


def index_plugin_corpus(
    corpus: RagCorpus,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> IndexOutcome:
    """Build or rebuild the ChromaDB collection for one plugin-contributed corpus.

    Mirrors :func:`index_docs`'s skip-if-populated and all-or-nothing build, but
    sources its text from ``corpus.text_dir`` directly if given, or by cloning
    ``corpus.docs_repo`` at ``corpus.docs_ref`` and rendering
    ``corpus.docs_subdir`` the same way the core docs are built.

    Args:
        corpus: The plugin's corpus declaration. Must name a source (see
            :attr:`RagCorpus.has_source`); a plugin's corpus is validated for
            this already by ``discover_plugins``, but a directly-called corpus
            is checked here too.
        force: If True, re-render and rebuild even if a populated collection
            already exists.
        progress: Optional ``(done, total)`` embed-batch callback.

    Returns:
        The same :class:`IndexOutcome` vocabulary as :func:`index_docs`.
    """
    if not corpus.has_source:
        msg = f"plugin corpus {corpus.name!r} names no text_dir and no docs_repo to build from."
        raise RuntimeError(msg)

    cfg = RagSettings()
    client = _get_client(cfg)
    embed_fn = get_embedding_function(cfg)
    name = _plugin_collection_name(embed_fn, corpus.name, corpus.version)

    if not force and _collection_populated(client, name):
        logger.info(
            "collection '%s' already exists — skipping index for plugin corpus %r",
            name,
            corpus.name,
        )
        return IndexOutcome.ALREADY_PRESENT

    if corpus.text_dir is not None:
        text_dir = str(corpus.text_dir)
        if not os.path.exists(text_dir):
            msg = (
                f"plugin corpus {corpus.name!r} names text_dir {text_dir!r}, "
                "which does not exist."
            )
            raise RuntimeError(msg)
    else:
        assert corpus.docs_repo is not None  # has_source guarantees one of the two
        cache_dir = str(
            cfg.vector_db_path
            / f"plugin_text_corpus__{corpus.name}__{corpus.version}__{_CORPUS_FORMAT}"
        )
        if (
            force
            or not os.path.exists(cache_dir)
            or not list(Path(cache_dir).rglob("*.txt"))
        ):
            _clone_and_build_text(
                cache_dir,
                docs_repo=corpus.docs_repo,
                docs_tag=corpus.docs_ref or "HEAD",
                docs_subdir=corpus.docs_subdir or "docs",
            )
        else:
            logger.info("text corpus already exists at %s — skipping clone", cache_dir)
        text_dir = cache_dir

    chunks = _load_docs(text_dir)
    return _build_collection(
        client,
        name,
        chunks,
        embed_fn,
        metadata={
            "hnsw:space": "cosine",
            "docs_version": corpus.version,
            "embedding": embed_fn.name(),
            "corpus": corpus.name,
        },
        progress=progress,
    )


def index_plugin_corpora(
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, IndexOutcome]:
    """Build or rebuild every installed plugin's contributed corpora.

    Discovers plugins fresh on each call and indexes each corpus in isolation:
    a clone, build, or embed failure for one corpus is logged and reported as
    ``FAILED`` rather than raised, so it never stops the corpora that follow --
    the same error-isolation the plugin channel itself guarantees at
    agent-build time (see ``aiida_agents.plugins.discovery``).

    Returns:
        A mapping of ``"{plugin_name}/{corpus_name}"`` to the
        :class:`IndexOutcome` for that corpus. Empty if no installed plugin
        contributes a corpus.
    """
    outcomes: dict[str, IndexOutcome] = {}
    for plugin in discover_plugins():
        for corpus in plugin.corpora:
            key = f"{plugin.name}/{corpus.name}"
            try:
                outcomes[key] = index_plugin_corpus(
                    corpus, force=force, progress=progress
                )
            except Exception:
                # Deliberately broad: a plugin's docs repo can fail to clone, its
                # sphinx build can error, or its text_dir can vanish -- none of
                # that should cost the other installed plugins their index.
                logger.exception(
                    "plugin %r: indexing corpus %r failed; skipping",
                    plugin.name,
                    corpus.name,
                )
                outcomes[key] = IndexOutcome.FAILED
    return outcomes
