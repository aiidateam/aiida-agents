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

from aiida_agents.plugins.spec import GroundingVocabulary

__all__ = [
    "asserted_quantities",
    "evidence_quantities",
    "quantities_in",
    "tool_output_text",
    "ungrounded_quantities",
]


_NUMBER = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

#: A percentage, which needs its own pattern rather than a place in _UNITS:
#: ``\b`` after a non-word character like ``%`` never matches, so "40% more"
#: would slip straight through the units regex.
#:
#: Percentages are worth catching because they are how a fabricated claim
#: usually arrives -- "~40% more resources", "converges 30% faster" reads as
#: quantified and authoritative, and nothing in a profile ever produced it.
_PERCENT = re.compile(rf"(?P<num>{_NUMBER})\s*(?:%|percent\b)")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


def _vocabulary_patterns(
    vocabulary: GroundingVocabulary | None,
) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None, re.Pattern[str] | None]:
    """Build escaped matchers from plugin-provided domain vocabulary."""
    if vocabulary is None:
        return None, None, None

    units = sorted(set(vocabulary.units), key=len, reverse=True)
    parameters = sorted(set(vocabulary.parameters), key=len, reverse=True)
    unit_alternatives = "|".join(re.escape(item) for item in units)
    parameter_alternatives = "|".join(re.escape(item) for item in parameters)

    # Lookarounds handle tokens containing punctuation or Unicode symbols,
    # where ``\b`` is not a reliable boundary (for example ``Å⁻¹`` or ``1/A``).
    unit_match = (
        re.compile(rf"(?P<num>{_NUMBER})\s*(?<!\w)(?:{unit_alternatives})(?!\w)")
        if unit_alternatives
        else None
    )
    unit_token = (
        re.compile(rf"(?<!\w)(?:{unit_alternatives})(?!\w)")
        if unit_alternatives
        else None
    )
    parameter_match = (
        re.compile(
            rf"(?<![A-Za-z0-9])(?:{parameter_alternatives})(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        if parameter_alternatives
        else None
    )
    return unit_match, unit_token, parameter_match


def _parameter_value_pattern(
    vocabulary: GroundingVocabulary | None,
) -> re.Pattern[str] | None:
    if vocabulary is None:
        return None
    parameters = sorted(set(vocabulary.parameters), key=len, reverse=True)
    alternatives = "|".join(re.escape(item) for item in parameters)
    if not alternatives:
        return None
    return re.compile(
        rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])"
        rf"\D{{0,24}}?(?P<num>{_NUMBER})",
        re.IGNORECASE,
    )


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


def asserted_quantities(
    text: str, vocabulary: GroundingVocabulary | None = None
) -> set[str]:
    """Numbers ``text`` states *as measurements*: with a unit, or as a percentage.

    The strong half of :func:`quantities_in`. A number written "60 Ry" or
    "40%" is a claim about physics however it got there, so it is held to a
    stricter standard of evidence than a bare number that merely shares a
    sentence with a parameter name.
    """
    unit_match, _, _ = _vocabulary_patterns(vocabulary)
    with_units = (
        {_normalise(m.group("num")) for m in unit_match.finditer(text)}
        if unit_match is not None
        else set()
    )
    return with_units | _percentages_in(text)


def evidence_quantities(
    evidence: str, vocabulary: GroundingVocabulary | None = None
) -> set[str]:
    """Numbers the tool output actually presents as quantities.

    The fix for the failure this whole check exists to catch. Grounding used to
    accept a claim if its bare number appeared *anywhere* in the evidence, and
    real tool output is dense with incidental integers --- pks, exit codes,
    counts, ports, timestamps. So a fabricated "60 Ry" was certified the moment
    some unrelated node happened to have pk 60, which is precisely the class of
    invention the check was built to find (issue #81).

    A number counts here when the evidence carries it with a unit, writes it as
    a percentage, or puts it next to a parameter name --- ``'ecutwfc': 60.0``.
    Adjacency is the right rule on this side and the wrong one on the answer
    side, because structured output keeps a key beside its value and prose does
    not.
    """
    parameter_value = _parameter_value_pattern(vocabulary)
    values = (
        {_normalise(m.group("num")) for m in parameter_value.finditer(evidence)}
        if parameter_value is not None
        else set()
    )
    return asserted_quantities(evidence, vocabulary) | values


def quantities_in(text: str, vocabulary: GroundingVocabulary | None = None) -> set[str]:
    """Physical quantities asserted in ``text``, as normalised numbers.

    A number counts when it carries a unit, when it is written as a
    percentage, or when it appears in a sentence naming a simulation
    parameter. Bare numbers in ordinary prose ("step 2 of 3", "found 12
    structures") do not: flagging those would bury the signal, and a warning
    nobody believes protects nothing.
    """
    found = asserted_quantities(text, vocabulary)
    unit_match, unit_token, parameter_match = _vocabulary_patterns(vocabulary)

    for sentence in _SENTENCE_SPLIT.split(text):
        if parameter_match is None or not parameter_match.search(sentence):
            continue
        # Strip the number+unit pairs already collected above, then any bare
        # unit token, before scanning for unattached numbers. Some unit
        # spellings contain a digit of their own -- "1/A", "A-1" -- and
        # reporting that 1 as an invented quantity is exactly the kind of false
        # positive that gets a warning ignored.
        stripped = sentence
        if unit_match is not None:
            stripped = unit_match.sub(" ", stripped)
        if unit_token is not None:
            stripped = unit_token.sub(" ", stripped)
        found |= _numbers_in(stripped)

    return found


def ungrounded_quantities(
    answer: str,
    evidence: str,
    prompt: str = "",
    vocabulary: GroundingVocabulary | None = None,
) -> set[str]:
    """Physical quantities in ``answer`` that appear in neither source.

    Args:
        answer: The reply shown to the user.
        evidence: Everything the tools returned during the run.
        prompt: The user's own turn, so a value they supplied is not reported
            as invented when the agent repeats it back.
        vocabulary: Domain terms contributed by installed plugins. With no
            plugin vocabulary, only domain-independent percentages are checked.

    Returns:
        Normalised numeric literals with no source. Empty means every physics
        number in the answer is traceable to a tool or to the question.

    Two classes of claim, held to different standards, because one bar cannot
    serve both without failing in one direction or the other (issue #81).

    A number the answer writes **with a unit or as a percentage** is an
    unambiguous measurement, and is grounded only if the evidence also presents
    it as one. Accepting any matching digit is how "60 Ry" passed on the
    strength of an unrelated pk.

    A **bare number sharing a sentence with a parameter name** is a guess about
    what the sentence means; the sentence scope is deliberately generous
    because prose separates a value from its name. Holding those to the strict
    bar would flag "I checked 3 relaxations and the cutoff was fine", and a
    warning that fires on correct answers is one people learn to scroll past.
    They stay grounded by any number the evidence contains.
    """
    strict = evidence_quantities(evidence, vocabulary) | evidence_quantities(
        prompt, vocabulary
    )
    loose = _numbers_in(evidence) | _numbers_in(prompt)

    asserted = asserted_quantities(answer, vocabulary)
    missing = (asserted - strict) | (
        (quantities_in(answer, vocabulary) - asserted) - loose
    )
    grounded = loose

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
