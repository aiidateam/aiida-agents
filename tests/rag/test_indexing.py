"""Unit tests for the RAG indexing pipeline.

The full pipeline (git clone + sphinx-build + embed) is integration-level and
is exercised by ``dev/test_rag.py`` and manual dogfooding, not here. This
covers the cheap, deterministic guard: a missing docs toolchain fails fast,
before any clone, with actionable guidance.
"""

from __future__ import annotations

import pytest

from aiida_agents.rag.indexing import _clone_and_build_text


def test_build_requires_docs_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sphinx -> RuntimeError with the install hint, before any clone."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, package=None: None,
    )
    with pytest.raises(RuntimeError, match=r"aiida-core\[docs\]"):
        _clone_and_build_text("/tmp/unused")
