"""Tests for cli/ollama.py: listing parse, the pull decision, and provisioning."""

from __future__ import annotations

import subprocess

import pytest
import rich_click

from aiida_agents._settings import _Provider
from aiida_agents.cli.ollama import _ollama_lists_model, _ollama_model_names


@pytest.mark.parametrize(
    "output, expected",
    [
        pytest.param("NAME  ID  SIZE  MODIFIED\n", set(), id="header-only-nothing"),
        pytest.param(
            "NAME                                            ID    SIZE\n"
            "qwen3.5:9b                                      abc   6.6 GB\n"
            "MichelRosselli/apertus:8b-instruct-2509-q4_k_m  def   5.1 GB\n",
            {"qwen3.5:9b", "MichelRosselli/apertus:8b-instruct-2509-q4_k_m"},
            id="multi-model-first-column",
        ),
    ],
)
def test_ollama_model_names_parses_first_column(
    output: str, expected: set[str]
) -> None:
    """Names come from the first column, header row skipped (empty when none)."""
    assert _ollama_model_names(output) == expected


@pytest.mark.parametrize(
    "model, present",
    [
        pytest.param("mxbai-embed-large", True, id="untagged-matches-latest"),
        pytest.param("qwen3.5:9b", True, id="tagged-exact"),
        pytest.param("not-there", False, id="absent"),
    ],
)
def test_ollama_lists_model_normalizes_latest_tag(model: str, present: bool) -> None:
    """An untagged configured name matches the ':latest' `ollama list` shows."""
    out = (
        "NAME                        ID   SIZE\n"
        "mxbai-embed-large:latest    abc  669 MB\n"
        "qwen3.5:9b                  def  6.6 GB\n"
    )
    assert _ollama_lists_model(model, out) is present


def test_should_skip_ollama_pull_reflects_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `ollama list` runs, the decision follows whether the model is listed."""
    from types import SimpleNamespace

    from aiida_agents.cli import ollama

    listing = "NAME              ID   SIZE\nqwen3.5:9b        abc  6.6 GB\n"
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=listing)
    )
    assert ollama._should_skip_ollama_pull("qwen3.5:9b") is True
    assert ollama._should_skip_ollama_pull("absent") is False


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError("no ollama on PATH"), id="not-installed"),
        pytest.param(subprocess.CalledProcessError(1, "ollama"), id="server-down"),
    ],
)
def test_should_skip_ollama_pull_fails_open(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """A check that can't run (no ollama on PATH, server down) skips the pull
    prompt rather than blocking the session with a doomed download.
    """
    from aiida_agents.cli import ollama

    def _raise(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(subprocess, "run", _raise)
    assert ollama._should_skip_ollama_pull("anything") is True


@pytest.mark.parametrize(
    "skip, accept, pulled",
    [
        pytest.param(True, False, [], id="present-no-prompt"),
        pytest.param(False, True, ["m"], id="missing-accepted-pulls"),
        pytest.param(False, False, [], id="missing-declined-no-pull"),
    ],
)
def test_prompt_pull_ollama_model(
    monkeypatch: pytest.MonkeyPatch, skip: bool, accept: bool, pulled: list[str]
) -> None:
    """A present model prompts nothing; a missing one pulls only on confirmation."""
    from aiida_agents.cli import ollama

    ran: list[str] = []
    monkeypatch.setattr(ollama, "_should_skip_ollama_pull", lambda model: skip)
    monkeypatch.setattr(rich_click, "confirm", lambda *a, **k: accept)
    monkeypatch.setattr(ollama, "_ollama_pull", lambda model: ran.append(model))

    ollama._prompt_pull_ollama_model("m")

    assert ran == pulled


@pytest.mark.parametrize(
    "provider, prompts",
    [
        pytest.param("ollama", ["qwen3.5:2b"], id="ollama-prompts"),
        pytest.param("openrouter", [], id="cloud-skips"),
    ],
)
def test_ensure_ollama_model_only_for_ollama_provider(
    monkeypatch: pytest.MonkeyPatch, provider: _Provider, prompts: list[str]
) -> None:
    """Only the Ollama provider offers a pull; cloud providers pass through."""
    from aiida_agents._settings import ModelSettings
    from aiida_agents.cli import ollama

    prompted: list[str] = []
    monkeypatch.setattr(
        ollama, "_prompt_pull_ollama_model", lambda model: prompted.append(model)
    )

    ollama._ensure_ollama_model(ModelSettings(provider=provider, model="qwen3.5:2b"))

    assert prompted == prompts
