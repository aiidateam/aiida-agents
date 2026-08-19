"""Sphinx text builder with fenced code blocks.

Run as ``python -m aiida_agents.rag._textbuild <srcdir> <outdir>`` in place of
``python -m sphinx -b text -E -q <srcdir> <outdir>``: the output is identical
except that every ``literal_block`` (RST ``.. code-block::`` directives and
MyST fences alike) is wrapped in markdown-style fences carrying the block's
language. The stock text builder flattens code to indented prose,
indistinguishable from block quotes and stripped of its language, which is
exactly what the RAG corpus must not lose (see ``chunking``, which keeps
fenced blocks atomic).
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Build ``srcdir`` into ``outdir`` with the fenced text translator.

    :param argv: ``[srcdir, outdir]``; defaults to ``sys.argv[1:]``.
    :return: The sphinx status code (0 on success).
    """
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        sys.stderr.write(
            "Usage: python -m aiida_agents.rag._textbuild <srcdir> <outdir>\n"
        )
        return 2
    srcdir, outdir = args

    # Imported here: sphinx is an indexing-only dependency, shipped by the
    # `rag` extra (see the check in ``indexing._clone_and_build_text``), so
    # importing this module never requires the docs toolchain.
    from docutils import nodes
    from sphinx.application import Sphinx
    from sphinx.writers.text import TextTranslator

    def _inside_table(node: nodes.Element) -> bool:
        """True if the block sits in a table cell, where the writer's column
        layout would wrap the fence lines into cell fragments."""
        parent = node.parent
        while parent is not None:
            if isinstance(parent, nodes.entry):
                return True
            parent = parent.parent
        return False

    class FencedTextTranslator(TextTranslator):
        """Stock text output, plus ```lang fences around literal blocks."""

        def visit_literal_block(self, node: nodes.Element) -> None:
            if _inside_table(node):
                super().visit_literal_block(node)
                return
            language = node.get("language") or ""
            # 'default'/'none' are sphinx's fillers for "no language given".
            if language in {"default", "none"}:
                language = ""
            self.new_state(0)
            self.add_text(f"```{language}")
            # Inner state at indent 0: the code sits flush with its fences,
            # as in a regular markdown block (the stock builder indents by 3).
            self.new_state(0)

        def depart_literal_block(self, node: nodes.Element) -> None:
            if _inside_table(node):
                super().depart_literal_block(node)
                return
            self.end_state(wrap=False)
            self.add_text("```")
            self.end_state(wrap=False)

    app = Sphinx(
        srcdir=srcdir,
        confdir=srcdir,
        outdir=outdir,
        doctreedir=f"{outdir}/.doctrees",
        buildername="text",
        confoverrides={
            # Don't execute notebooks; same as the -D flags the plain
            # sphinx-build invocation used.
            "nb_execution_mode": "off",
            "jupyter_execute_notebooks": "off",
        },
        status=None,  # -q
        freshenv=True,  # -E
    )
    app.set_translator("text", FencedTextTranslator, override=True)
    app.build()
    return app.statuscode


if __name__ == "__main__":
    raise SystemExit(main())
