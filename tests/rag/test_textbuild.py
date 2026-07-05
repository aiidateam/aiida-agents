"""Integration test for the fenced sphinx text builder (``rag._textbuild``).

Builds a minimal real sphinx project, so it needs the docs toolchain; skipped
when sphinx is not installed (it is an opt-in dependency, see
``indexing._clone_and_build_text``).
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from aiida_agents.rag._textbuild import main as textbuild_main

requires_sphinx = pytest.mark.skipif(
    find_spec("sphinx") is None,
    reason="sphinx not installed (opt-in docs toolchain)",
)


@requires_sphinx
def test_code_blocks_come_out_fenced_with_language(tmp_path: Path) -> None:
    """RST code-block directives survive as ```lang fences; prose stays flat.

    Regression for the core corpus defect: the stock text builder renders
    code as indented prose, losing both the code/prose distinction and the
    language.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "conf.py").write_text('project = "t"\n')
    (src / "index.rst").write_text(
        "Title\n"
        "=====\n"
        "\n"
        "Some prose before.\n"
        "\n"
        ".. code-block:: python\n"
        "\n"
        "   from aiida.engine import submit\n"
        "\n"
        "Some prose after.\n"
    )
    out = tmp_path / "out"

    assert textbuild_main([str(src), str(out)]) == 0

    rendered = (out / "index.txt").read_text()
    assert "```python" in rendered
    assert "from aiida.engine import submit" in rendered
    assert rendered.count("```") == 2
    # Prose is not fenced and headings keep the text builder's underline style
    # (the chunker's section detection depends on it).
    assert "Some prose before." in rendered
    assert "*****" in rendered
