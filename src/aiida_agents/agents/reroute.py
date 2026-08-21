"""The cross-agent hand-off tool, and the signal the REPL re-routes on.

The planner routes each turn, but a mis-route (or its fallback to the read-only
analysis agent) can land a request on a specialist that has no tool for it. Any
specialist can then call :func:`hand_off_to` to pass the turn to the right one;
the REPL re-runs the turn there once, carrying a handoff. This is the recovery
net for routing, not a substitute for it: an agent hands off only when it
genuinely lacks the tools for what is asked, never to avoid answering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from aiida_agents.agents.planner import Specialist

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

#: The specialists a turn can be handed to. Kept beside the tool so its
#: ``Specialist`` argument and the REPL's routing agree on the same set.
_KNOWN: tuple[Specialist, ...] = ("analysis", "execution", "codegen")


def hand_off_to(specialist: Specialist, reason: str) -> str:
    """Hand the current request to another specialist instead of answering.

    Call this the moment the request needs something you have no tool for:
    ``execution`` to submit / run / import / delete, ``codegen`` to answer with
    custom Python that no fixed tool expresses, ``analysis`` to explore existing
    data. ``reason`` restates what the user wants, with the specifics (pks,
    workflow, any changes), so the receiving agent can act without re-asking.
    The CLI re-runs the turn on that agent automatically; do not tell the user to
    switch, and never claim to have done work that belongs to the other agent.
    """
    return f"Handed off to the {specialist} agent: {reason}"


@dataclass(frozen=True)
class HandoffRequest:
    """A parsed ``hand_off_to`` call: which specialist to run next, and why."""

    specialist: Specialist
    reason: str


def handoff_request(messages: list[ModelMessage]) -> HandoffRequest | None:
    """The first ``hand_off_to`` call in a turn's new messages, as a request.

    Returns ``None`` if the turn made no such call, or named a specialist the
    REPL cannot run (the tool's ``Specialist`` type should already prevent that,
    but a router must never dispatch on a value it cannot honour).
    """
    from pydantic_ai.messages import ToolCallPart

    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name == hand_off_to.__name__:
                args = part.args_as_dict()
                specialist = str(args.get("specialist", "")).strip()
                if specialist not in _KNOWN:
                    continue
                reason = str(args.get("reason", "")).strip()
                return HandoffRequest(
                    cast("Specialist", specialist), reason or "(no detail given)"
                )
    return None
