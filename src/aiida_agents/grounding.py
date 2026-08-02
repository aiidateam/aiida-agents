"""Is a number in an answer traceable to something a tool returned?

The agents are asked what cutoff to use, and a wrong answer is not an
inconvenience -- it configures a calculation that burns real CPU hours. The
failure that actually occurs is not a refusal or an obvious error: it is a
plausible number, often attached to a real label the model did retrieve
("42 for gold", where the gold came from a query and the 42 came from
nowhere). That reads as sourced and cannot be checked by eye.

Prompt rules asking the model not to do this have a measurable failure rate:
across one session's testing, an explicit instruction was ignored in five runs
out of five, and fabrication recurred intermittently on every model tried. So
this check does not ask. It reads the answer after the fact and reports which
quantities no tool produced, which works whether or not the model cooperated.

Deliberately narrow. Only numbers carrying a unit, or bound to a named
simulation parameter, count as claims about physics -- "step 2 of 3" and
"found 12 structures" are prose. Values are normalised, so 60 in the answer
matches 60.0 in a tool return. A check that cried wolf would be switched off,
and then it would protect nothing.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["quantities_in", "tool_output_text", "ungrounded_quantities"]


#: Units whose presence makes the preceding number a physics claim. ASCII
#: spellings are included because tool output and terminals both use them --
#: ``query_run_context`` reports "1/A", not "1/Angstrom-with-a-ring".
_UNITS = (
    r"Ry|Rydberg|eV|meV|Ha|Hartree|Bohr|bohr|GPa|kbar|K"
    r"|Å\^?-1|Å⁻¹|1/Å|Å|A\^-1|A-1|1/A|Ang|angstrom"
)

#: Parameter names whose values are physics claims regardless of unit.
_PARAMS = (
    r"ecutwfc|ecutrho|conv_thr|degauss|kpoints?_distance|"
    r"k-?point spacing|mixing_beta|smearing|etot_conv_thr|forc_conv_thr|"
    r"press_conv_thr|nbnd|electron_maxstep|cutoff"
)

_NUMBER = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

#: A number carrying a unit -- "60 Ry", "0.15 1/A" -- anywhere in the text.
_WITH_UNIT = re.compile(rf"(?P<num>{_NUMBER})\s*(?:{_UNITS})\b")

#: A sentence that names a simulation parameter.
#:
#: Proximity was the first attempt and was wrong: requiring the number within
#: ~20 characters of the parameter name missed the fabrication this check was
#: built for -- "ecutwfc varies widely (e.g., 42 for gold, 8 for the alkali
#: metals)" puts 22 characters between them, and the 8 much further. Prose does
#: not keep a value next to its name. Scoping to the sentence matches how the
#: claim is actually written.
_PARAM_SENTENCE = re.compile(rf"(?:{_PARAMS})", re.IGNORECASE)

#: A percentage, which needs its own pattern rather than a place in _UNITS:
#: ``\b`` after a non-word character like ``%`` never matches, so "40% more"
#: would slip straight through the units regex.
#:
#: Percentages are worth catching because they are how a fabricated claim
#: usually arrives -- "~40% more resources", "converges 30% faster" reads as
#: quantified and authoritative, and nothing in a profile ever produced it.
_PERCENT = re.compile(rf"(?P<num>{_NUMBER})\s*(?:%|percent\b)")

#: A unit written on its own, for removal before counting bare numbers.
_UNIT_TOKEN = re.compile(rf"(?:{_UNITS})")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


def _numbers_in(text: str) -> set[str]:
    """Every numeric literal in ``text``, normalised for comparison."""
    return {_normalise(raw) for raw in re.findall(_NUMBER, text)}


def _normalise(number: str) -> str:
    """Canonicalise a numeric literal so 60, 60.0 and 6e1 compare equal.

    Without this the check reports false fabrications constantly: a tool
    returns ``60.0`` and the model writes "60 Ry", which is the same claim
    correctly repeated.
    """
    try:
        return repr(float(number))
    except ValueError:  # pragma: no cover - regex only yields parseable numbers
        return number


def _percentages_in(text: str) -> set[str]:
    """Every number written as a percentage, normalised for comparison."""
    return {_normalise(m.group("num")) for m in _PERCENT.finditer(text)}


def quantities_in(text: str) -> set[str]:
    """Physical quantities asserted in ``text``, as normalised numbers.

    A number counts when it carries a unit, when it is written as a
    percentage, or when it appears in a sentence naming a simulation
    parameter. Bare numbers in ordinary prose ("step 2 of 3", "found 12
    structures") do not: flagging those would bury the signal, and a warning
    nobody believes protects nothing.
    """
    found = {_normalise(m.group("num")) for m in _WITH_UNIT.finditer(text)}
    found |= _percentages_in(text)

    for sentence in _SENTENCE_SPLIT.split(text):
        if not _PARAM_SENTENCE.search(sentence):
            continue
        # Strip the number+unit pairs already collected above, then any bare
        # unit token, before scanning for unattached numbers. Some unit
        # spellings contain a digit of their own -- "1/A", "A-1" -- and
        # reporting that 1 as an invented quantity is exactly the kind of false
        # positive that gets a warning ignored.
        stripped = _UNIT_TOKEN.sub(" ", _WITH_UNIT.sub(" ", sentence))
        found |= _numbers_in(stripped)

    return found


def ungrounded_quantities(answer: str, evidence: str, prompt: str = "") -> set[str]:
    """Physical quantities in ``answer`` that appear in neither source.

    Args:
        answer: The reply shown to the user.
        evidence: Everything the tools returned during the run.
        prompt: The user's own turn, so a value they supplied is not reported
            as invented when the agent repeats it back.

    Returns:
        Normalised numeric literals with no source. Empty means every physics
        number in the answer is traceable to a tool or to the question.
    """
    grounded = _numbers_in(evidence) | _numbers_in(prompt)
    missing = quantities_in(answer) - grounded

    # A percentage is also grounded by its fraction. ``query_run_context``
    # reports ``success_rate`` as 0.67, and an answer saying "67% succeeded" is
    # repeating that correctly rather than inventing it -- flagging the
    # conversion would train people to ignore the warning that matters.
    for value in _percentages_in(answer) & missing:
        if _normalise(repr(float(value) / 100)) in grounded:
            missing.discard(value)

    return missing


def tool_output_text(messages: list[Any]) -> str:
    """Everything the tools returned during a run, flattened for searching.

    Takes pydantic-ai messages rather than a prepared string so a caller can
    hand over ``result.all_messages()`` without knowing the shape of a tool
    return -- which is a dict or a dataclass as often as text.
    """
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                content = part.content
                chunks.append(content if isinstance(content, str) else repr(content))
    return "\n".join(chunks)


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
