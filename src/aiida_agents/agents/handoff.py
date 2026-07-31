"""The message one specialist passes to the next.

The proposal asked for an agent-to-agent communication protocol. What existed
was a formatted string: the CLI pasted step one's prose answer into step two's
prompt under a heading. It worked, and it lost the thing that mattered most.

A diagnosis says "the failure is in PwCalculation pk 334407". The next step
needs *334407*, and with a free-text handoff the only way to get it is for a
second model to read the sentence and pull the number back out --- re-deriving,
from prose, a fact the first step held exactly. That is a transcription step
between two language models, which is precisely where a digit goes missing and
a resubmission targets the wrong node.

So the handoff is a structured message with the node references carried
alongside the prose, extracted from what the tools actually returned rather
than from what the agent wrote about them. The receiving agent is told to use
those pks directly.

This is a protocol in the sense that matters here --- a defined message with a
producer, a consumer, and typed content --- and deliberately not a transport.
Both specialists run in one interpreter against one profile, so a channel
between processes would add moving parts and buy nothing; and routing a
specialist's output through anything other than the CLI would break the
approval gate, for the reason set out in ADR-09.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

__all__ = [
    "Handoff",
    "NodeReference",
    "node_references_from_messages",
    "node_references_in",
]

#: Most references to carry. A query can match hundreds of nodes, and pasting
#: them all would crowd out the findings they are supposed to support; the next
#: step can always query again for the full set.
MAX_REFERENCES = 20

#: Keys whose value describes what a referenced node *is*, best first. A pk
#: alone is not usable context -- "pk 334407" tells the next agent nothing about
#: whether it is the thing they were looking for.
_LABEL_KEYS = ("process_label", "link_label", "filename", "formula", "node_type")


@dataclass(frozen=True)
class NodeReference:
    """One AiiDA node a step identified, and what it is."""

    pk: int
    label: str

    def __str__(self) -> str:
        return f"pk {self.pk} — {self.label}" if self.label else f"pk {self.pk}"


@dataclass(frozen=True)
class Handoff:
    """What one specialist passes to the next, as a typed message.

    ``findings`` is the producing agent's own answer, kept verbatim: it carries
    the reasoning, and paraphrasing it here would be a third party rewriting a
    conclusion it did not reach. ``node_references`` is the machine-readable
    half --- the nodes the step's *tools* returned, so the consumer works from
    identifiers rather than from a sentence about them.
    """

    from_specialist: str
    to_specialist: str
    task: str
    findings: str
    node_references: tuple[NodeReference, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """The message as the receiving agent's prompt."""
        parts = [
            self.task,
            "",
            f"Context from the previous step ({self.from_specialist} agent):",
            '"""',
            self.findings,
            '"""',
        ]
        if self.node_references:
            parts += [
                "",
                "Nodes that step identified, from its own tool output:",
                *(f"  - {reference}" for reference in self.node_references),
                "",
                "Use these pks directly. They are what the tools returned, so "
                "they are correct as given -- do not re-read them out of the "
                "text above, and do not substitute a pk of your own.",
            ]
        parts += [
            "",
            "Use that context as the findings it is. Do not restate values it "
            "does not contain, and if it does not give you what this step "
            "needs, say so rather than proceeding on an assumption.",
        ]
        return "\n".join(parts)


def _label_for(record: dict[str, t.Any]) -> str:
    """The most descriptive name a record offers for its node."""
    for key in _LABEL_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _walk(value: t.Any, found: dict[int, str]) -> None:
    """Collect every ``pk`` in a tool return, with the best label beside it.

    First label wins: a record naming the node is usually the one that returned
    it, and a later bare mention should not overwrite that with nothing.
    """
    if isinstance(value, dict):
        pk = value.get("pk")
        if isinstance(pk, int) and pk not in found:
            found[pk] = _label_for(value)
        for nested in value.values():
            _walk(nested, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk(item, found)


def node_references_from_messages(
    messages: t.Iterable[t.Any],
) -> tuple[NodeReference, ...]:
    """The nodes a run's tools returned, taken from pydantic-ai messages.

    A convenience over :func:`node_references_in` so a caller can hand over
    ``result.all_messages()`` without knowing the shape of a tool return.
    """
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    contents = [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    return node_references_in(contents)


def node_references_in(tool_returns: t.Iterable[t.Any]) -> tuple[NodeReference, ...]:
    """The nodes a step's tools returned, in the order they first appeared.

    Taken from tool output rather than from the answer on purpose: a pk the
    agent wrote down may be a typo of one it was given, and the point of the
    structured half of a handoff is that it cannot be.

    Args:
        tool_returns: The content of each tool return in the run.

    Returns:
        Up to :data:`MAX_REFERENCES` references, deduplicated by pk.
    """
    found: dict[int, str] = {}
    for content in tool_returns:
        _walk(content, found)
        if len(found) >= MAX_REFERENCES:
            break
    return tuple(
        NodeReference(pk=pk, label=label)
        for pk, label in list(found.items())[:MAX_REFERENCES]
    )
