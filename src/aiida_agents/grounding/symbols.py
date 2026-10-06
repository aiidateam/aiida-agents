"""Checks for generated Python snippets and imported AiiDA names."""

from __future__ import annotations

import re

__all__ = [
    "labelled_python_blocks",
    "python_blocks",
    "syntax_errors",
    "ungrounded_symbols",
]

# ---------------------------------------------------------------------------
# Code
#
# The quantity checks above ask whether a number in an answer came from a tool.
# A generated snippet raises the same question about names: an AiiDA class or
# function that exists in no retrieved example is the code equivalent of an
# invented cutoff, and it is worse, because the user finds out by running it.
#
# Two checks, deliberately narrow. Anything broader produces false positives,
# and this module's whole premise is that a warning nobody believes protects
# nothing -- a check that fires on `qb.append` because lists also have
# `append` would train people to ignore the one that matters.
# ---------------------------------------------------------------------------

#: A fenced block that could be Python: labelled as such, or unlabelled. A
#: fence labelled with another language is not matched at all.
_PY_BLOCK = re.compile(
    r"```(?P<lang>python|py|pycon)?[ \t]*\n(?P<code>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def python_blocks(answer: str) -> list[str]:
    """Every fenced block in ``answer`` that could be Python.

    Includes unlabelled fences: models routinely omit the language, and a
    snippet the user will paste into ``verdi shell`` is Python whether or not
    it said so. Safe for the symbol check, where a block that is not Python
    simply parses to nothing and contributes no imports.
    """
    return [m.group("code") for m in _PY_BLOCK.finditer(answer)]


def labelled_python_blocks(answer: str) -> list[str]:
    """Only the fences that explicitly say they are Python.

    The stricter list, and the one :func:`syntax_errors` uses. Live testing
    showed why: an answer that illustrates a point with an unlabelled block of
    ``verdi`` commands, YAML or pseudocode produced "the Python above does not
    parse" every time, because none of it is Python and none of it was ever
    meant to be.

    That is the failure this module exists to avoid --- a warning that fires on
    correct output teaches people to ignore it, and then it is not protecting
    the case it was built for. Losing the check on unlabelled blocks is the
    cheaper mistake: the Codegen agent is told to fence its snippets as
    ``python``, so the blocks that matter carry the label.
    """
    return [m.group("code") for m in _PY_BLOCK.finditer(answer) if m.group("lang")]


def syntax_errors(answer: str) -> list[str]:
    """Blocks in ``answer`` that are not parseable Python, with the reason.

    Cheap, deterministic, and zero false positives on real code: if CPython
    cannot parse it, no user can run it. Worth checking separately from the
    symbol check because it catches the failure that wastes the most of a
    user's time for the least excuse.

    Only blocks *labelled* as Python are checked --- see
    :func:`labelled_python_blocks`. Console transcripts are skipped on top of
    that, since a ``python``-labelled block of ``>>>`` prompts is a session
    rather than a module.
    """
    import ast

    problems = []
    for block in labelled_python_blocks(answer):
        stripped = block.strip()
        if not stripped or stripped.startswith((">>>", "$", "verdi ")):
            continue
        try:
            ast.parse(stripped)
        except SyntaxError as exc:
            problems.append(f"{exc.msg} (line {exc.lineno})")
    return problems


def _imported_aiida_names(code: str) -> set[str]:
    """Names the snippet imports from AiiDA, e.g. ``QueryBuilder``.

    Only imports, and only from an ``aiida`` module. This is the narrowest
    check that still catches the failure worth catching: a hallucinated class
    or helper is almost always *imported*, and an import naming something that
    does not exist fails at the very first line. Attribute and method names are
    deliberately not checked --- ``.append``, ``.get``, ``.all`` belong to
    builtins as often as to AiiDA, and flagging them would drown the signal.
    """
    import ast

    names: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names  # reported by syntax_errors, not here

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "aiida" or module.startswith("aiida."):
                names.update(alias.name for alias in node.names)
    return names


def ungrounded_symbols(answer: str, evidence: str) -> set[str]:
    """AiiDA names imported in ``answer``'s code that no tool output mentions.

    Args:
        answer: The reply shown to the user, code blocks included.
        evidence: Everything the tools returned during the run.

    Returns:
        Imported AiiDA names absent from every tool return. Empty means each
        one appeared in something the agent actually retrieved.
    """
    imported: set[str] = set()
    for block in python_blocks(answer):
        imported |= _imported_aiida_names(block)
    return {name for name in imported if name not in evidence}
