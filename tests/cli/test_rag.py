"""Tests for cli/rag.py: the rag command group (build/status/search/clear)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import rich_click
from click.testing import CliRunner

from aiida_agents.cli import cli


def _raise_called_process_error(spec: str) -> None:
    raise subprocess.CalledProcessError(1, "pip")


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError("sphinx build failed"), id="sphinx"),
        pytest.param(OSError("embedder unreachable"), id="embedder"),
    ],
)
def test_rag_build_reports_index_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """A build failure (sphinx, or the embedder over the network) surfaces as a
    clean CLI error, not a traceback."""
    monkeypatch.setattr("aiida_agents.cli.rag._module_missing", lambda name: False)
    # No real Ollama pull during the test if the embed model is absent.
    monkeypatch.setattr(
        "aiida_agents.cli.rag._prompt_pull_ollama_model", lambda model: None
    )

    def _boom(force: bool, progress: object = None) -> None:
        raise exc

    monkeypatch.setattr("aiida_agents.rag.index_docs", _boom)
    result = CliRunner().invoke(cli, ["rag", "build"])
    assert result.exit_code == 1
    assert "RAG build failed" in result.output
    # Converted, not leaked as an uncaught traceback.
    assert not isinstance(result.exception, (RuntimeError, OSError))


def test_rag_build_declining_toolchain_install_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sphinx plus a declined install gives a clean, actionable error."""
    monkeypatch.setattr(
        "aiida_agents.cli.rag._module_missing", lambda name: name == "sphinx"
    )
    result = CliRunner().invoke(cli, ["rag", "build"], input="n\n")
    assert result.exit_code == 1
    assert "not installed" in result.output
    assert "aiida-core[docs]" in result.output


@pytest.mark.parametrize(
    "outcome_name, expected, unexpected",
    [
        pytest.param(
            "ALREADY_PRESENT", "already built", "index ready", id="already-present"
        ),
        pytest.param(
            "EMPTY_CORPUS", "left unchanged", "index ready", id="empty-corpus"
        ),
        pytest.param("BUILT", "RAG index ready", "already built", id="built"),
    ],
)
def test_rag_build_reports_outcome(
    monkeypatch: pytest.MonkeyPatch, outcome_name: str, expected: str, unexpected: str
) -> None:
    """The final line reflects what ``index_docs`` actually did, so re-running on
    an up-to-date index reads as a no-op, not a fresh build."""
    from aiida_agents.rag import IndexOutcome

    monkeypatch.setattr("aiida_agents.cli.rag._module_missing", lambda name: False)
    monkeypatch.setattr(
        "aiida_agents.cli.rag._prompt_pull_ollama_model", lambda model: None
    )
    outcome = getattr(IndexOutcome, outcome_name)
    monkeypatch.setattr(
        "aiida_agents.rag.index_docs", lambda force, progress=None: outcome
    )
    result = CliRunner().invoke(cli, ["rag", "build"])
    assert result.exit_code == 0
    assert expected in result.output
    assert unexpected not in result.output


def test_rag_build_reports_plugin_corpus_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin's corpus outcome is reported alongside the core docs' own,
    including a failure -- which must not turn the command itself into an
    error, since it is isolated per-corpus."""
    from aiida_agents.rag import IndexOutcome

    monkeypatch.setattr("aiida_agents.cli.rag._module_missing", lambda name: False)
    monkeypatch.setattr(
        "aiida_agents.cli.rag._prompt_pull_ollama_model", lambda model: None
    )
    monkeypatch.setattr(
        "aiida_agents.rag.index_docs",
        lambda force, progress=None: IndexOutcome.ALREADY_PRESENT,
    )
    monkeypatch.setattr(
        "aiida_agents.rag.index_plugin_corpora",
        lambda force=False: {
            "quantumespresso/qe": IndexOutcome.BUILT,
            "broken/corpus": IndexOutcome.FAILED,
        },
    )
    result = CliRunner().invoke(cli, ["rag", "build"])
    assert result.exit_code == 0
    assert "Plugin corpus 'quantumespresso/qe' indexed." in result.output
    assert "Plugin corpus 'broken/corpus' failed to index" in result.output


def test_rag_search_errors_cleanly_without_an_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rag search` with no index is a clean CLI error, not a traceback."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    monkeypatch.setattr(
        "aiida_agents.rag.retriever.docs_index_available", lambda: False
    )
    result = CliRunner().invoke(cli, ["rag", "search", "what is a calcjob"])
    assert result.exit_code == 1
    assert "No RAG index" in result.output


@pytest.mark.parametrize(
    "results, needle",
    [
        pytest.param(
            [{"source": "howto/run", "section": "Run", "text": "use verdi run"}],
            "verdi run",
            id="renders-a-match",
        ),
        pytest.param([], "No matching documentation", id="empty-is-clean"),
    ],
)
def test_rag_search_renders_results(
    monkeypatch: pytest.MonkeyPatch,
    results: list[dict[str, str]],
    needle: str,
) -> None:
    """With an index, matches render (source and body shown); no matches is a
    clean message, not an error.
    """
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    monkeypatch.setattr("aiida_agents.rag.retriever.docs_index_available", lambda: True)
    monkeypatch.setattr("aiida_agents.rag.query_docs", lambda query, limit: results)

    result = CliRunner().invoke(cli, ["rag", "search", "how to run"])

    assert result.exit_code == 0
    assert needle in result.output


def test_rag_search_refs_only_lists_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--refs-only` prints each match's source/section, not the body text."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    monkeypatch.setattr("aiida_agents.rag.retriever.docs_index_available", lambda: True)
    monkeypatch.setattr(
        "aiida_agents.rag.query_docs",
        lambda query, limit: [
            {"source": "howto/query", "section": "Shortcuts", "text": "use verdi run"}
        ],
    )

    result = CliRunner().invoke(cli, ["rag", "search", "qb", "--refs-only"])

    assert result.exit_code == 0
    assert "howto/query" in result.output
    assert "Shortcuts" in result.output
    assert "verdi run" not in result.output  # the body is omitted


def test_rag_status_reports_unbuilt_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no store on disk, `rag status` reports not-built and points to `rag build`."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path / "absent"))
    result = CliRunner().invoke(cli, ["rag", "status"])
    assert result.exit_code == 0
    assert "Built: no" in result.output
    assert "No index built yet" in result.output


def test_rag_status_reports_built_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A populated store surfaces as built, rendering its collection and metadata."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    store = tmp_path / "vdb"
    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(store))
    # A real chroma collection with an entry (count > 0). Explicit embeddings keep
    # the setup network-free; `rag status` only reads the count and metadata.
    import chromadb

    from aiida_agents.rag.store import _core_name_for

    # Named exactly as retrieval would look it up: an arbitrary name is now
    # correctly reported stale, which is a different test (see below).
    name = _core_name_for("ollama/mxbai-embed-large")
    collection = chromadb.PersistentClient(path=str(store)).create_collection(
        name,
        metadata={"docs_version": "v2.9.0", "embedding": "ollama/mxbai-embed-large"},
    )
    collection.add(ids=["1"], documents=["hello"], embeddings=[[0.1, 0.2, 0.3]])

    result = CliRunner().invoke(cli, ["rag", "status"])
    assert result.exit_code == 0
    assert "Built: yes" in result.output
    # The collections table rendered the persisted row's name and build metadata.
    # A narrow terminal may wrap the (long) embedding name across lines, so check
    # a prefix short enough to always land on one line rather than the full string.
    assert "aiida_docs__" in result.output
    assert "v2.9.0" in result.output
    assert "ollama/mxbai" in result.output
    # No "corpus" metadata was set on this collection; it reads as the core docs.
    assert "aiida-core" in result.output


def test_rag_status_attributes_a_plugin_corpus_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plugin-contributed corpus's collection shows its own corpus name in
    the status table, not lumped in with the core docs."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    store = tmp_path / "vdb"
    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(store))
    import chromadb

    client = chromadb.PersistentClient(path=str(store))
    from aiida_agents.rag.store import _plugin_name_for

    client.create_collection(
        _plugin_name_for("ollama/mxbai-embed-large", "qe", "1.0"),
        metadata={
            "docs_version": "1.0",
            "embedding": "ollama/mxbai-embed-large",
            "corpus": "quantumespresso",
        },
    ).add(ids=["1"], documents=["hello"], embeddings=[[0.1, 0.2, 0.3]])

    result = CliRunner().invoke(cli, ["rag", "status"])

    assert result.exit_code == 0
    # A narrow terminal folds the corpus name across lines, so match a prefix
    # short enough to stay on one (same reason as the embedding check above).
    assert "quantumespre" in result.output
    # No such plugin is installed here, so its collection is genuinely
    # unreachable and must say so rather than counting as a built index.
    assert "stale" in result.output
    assert "Built: no" in result.output


def test_rag_clear_no_store_is_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`rag clear` with nothing to delete is a clean no-op, not an error."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path / "absent"))
    result = CliRunner().invoke(cli, ["rag", "clear"])
    assert result.exit_code == 0
    assert "No RAG store to clear" in result.output


@pytest.mark.parametrize(
    ("args", "stdin", "marker", "removed"),
    [
        pytest.param(
            ["rag", "clear", "--force"], None, "Removed", True, id="force-flag"
        ),
        pytest.param(["rag", "clear"], "y\n", "Removed", True, id="confirm-yes"),
        pytest.param(["rag", "clear"], "n\n", "Cancelled", False, id="confirm-no"),
    ],
)
def test_rag_clear_honours_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: list[str],
    stdin: str | None,
    marker: str,
    removed: bool,
) -> None:
    """`--force` deletes without prompting; an interactive `y` deletes, `n` keeps."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    store = tmp_path / "vdb"
    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(store))
    store.mkdir(parents=True)
    (store / "chroma.sqlite3").write_text("x")  # something for rmtree to remove

    result = CliRunner().invoke(cli, args, input=stdin)
    assert result.exit_code == 0
    assert marker in result.output
    assert store.exists() == (not removed)


@pytest.mark.parametrize(
    "name, missing",
    [
        pytest.param("os", False, id="stdlib-present"),
        pytest.param("aiida_agents", False, id="own-package-present"),
        pytest.param("definitely_not_a_real_module_zzz", True, id="absent"),
    ],
)
def test_module_missing(name: str, missing: bool) -> None:
    """Reports a module's absence; the docs-toolchain check depends on this
    True-means-missing sense (easy to get backwards).
    """
    from aiida_agents.cli.rag import _module_missing

    assert _module_missing(name) is missing


def test_ensure_docs_toolchain_noop_when_sphinx_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With sphinx already importable, nothing is prompted or installed."""
    from aiida_agents.cli import rag

    asked: list[int] = []

    def _confirm(*args: object, **kwargs: object) -> bool:
        asked.append(1)
        return True

    monkeypatch.setattr(rag, "_module_missing", lambda name: False)
    monkeypatch.setattr(rich_click, "confirm", _confirm)

    rag._ensure_docs_toolchain()  # returns without raising

    assert asked == []


def test_ensure_docs_toolchain_declined_raises_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining the install gives a clean error naming the manual command.

    Asserts the whole command, because it has to name what the accept path
    installs (``aiida-core[docs]``, pinned below); the two drifted apart once
    before.
    """
    from aiida_agents.cli import rag

    monkeypatch.setattr(rag, "_module_missing", lambda name: True)
    monkeypatch.setattr(rich_click, "confirm", lambda *a, **k: False)

    with pytest.raises(rich_click.ClickException) as exc_info:
        rag._ensure_docs_toolchain()

    assert "uv pip install 'aiida-core[docs]'" in str(exc_info.value)


def test_ensure_docs_toolchain_installs_on_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting installs the docs extra and, once importable, returns cleanly."""
    from aiida_agents.cli import rag

    # Missing at the first check, importable after the install runs.
    checks = iter([True, False])
    installed: list[str] = []
    monkeypatch.setattr(rag, "_module_missing", lambda name: next(checks))
    monkeypatch.setattr(rich_click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(rag, "_pip_install", lambda spec: installed.append(spec))

    rag._ensure_docs_toolchain()  # returns without raising

    assert installed == ["aiida-core[docs]"]


@pytest.mark.parametrize(
    "pip_install, needle",
    [
        pytest.param(
            lambda spec: None, "still not importable", id="install-did-not-take"
        ),
        pytest.param(
            _raise_called_process_error, "Install failed", id="install-command-failed"
        ),
    ],
)
def test_ensure_docs_toolchain_install_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
    pip_install: object,
    needle: str,
) -> None:
    """A failed install, or one that leaves sphinx still unimportable, is a clean
    error telling the user to install it manually.
    """
    from aiida_agents.cli import rag

    monkeypatch.setattr(rag, "_module_missing", lambda name: True)
    monkeypatch.setattr(rich_click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(rag, "_pip_install", pip_install)

    with pytest.raises(rich_click.ClickException) as exc_info:
        rag._ensure_docs_toolchain()

    assert needle in str(exc_info.value)
