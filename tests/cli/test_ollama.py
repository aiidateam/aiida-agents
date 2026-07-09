"""Tests for cli/ollama.py: parsing `ollama list` output."""

from __future__ import annotations

import pytest

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
