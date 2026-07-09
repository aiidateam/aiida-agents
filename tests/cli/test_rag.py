"""Tests for cli/rag.py: the rag command group (build/status/search/clear)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from aiida_agents.cli import cli


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

    collection = chromadb.PersistentClient(path=str(store)).create_collection(
        "aiida_docs__test",
        metadata={"docs_version": "v2.8.0", "embedding": "ollama/mxbai-embed-large"},
    )
    collection.add(ids=["1"], documents=["hello"], embeddings=[[0.1, 0.2, 0.3]])

    result = CliRunner().invoke(cli, ["rag", "status"])
    assert result.exit_code == 0
    assert "Built: yes" in result.output
    # The collections table rendered the persisted row's name and build metadata.
    assert "aiida_docs__test" in result.output
    assert "v2.8.0" in result.output
    assert "ollama/mxbai-embed-large" in result.output


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
        pytest.param(["rag", "clear", "--yes"], None, "Removed", True, id="yes-flag"),
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
    """`--yes` deletes without prompting; an interactive `y` deletes, `n` keeps."""
    monkeypatch.setattr("aiida_agents.cli._guards.find_unrecognized_settings", list)
    store = tmp_path / "vdb"
    monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(store))
    store.mkdir(parents=True)
    (store / "chroma.sqlite3").write_text("x")  # something for rmtree to remove

    result = CliRunner().invoke(cli, args, input=stdin)
    assert result.exit_code == 0
    assert marker in result.output
    assert store.exists() == (not removed)
