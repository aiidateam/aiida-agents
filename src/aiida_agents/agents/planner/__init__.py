"""Planner — decides which specialist does what, and in what order.

ADR-09's orchestration layer. It was scheduled as two increments and this is
both: routing a request to one specialist is the degenerate case of planning, a
plan of length one. So there is one component rather than a router with a
coordinator bolted on top, and a simple request still costs one model call.

It is deliberately *not* a coordinator that wraps the specialists. The
human-in-the-loop mechanism (ADR-08) works because a specialist's run returns
``DeferredToolRequests`` as its output and the CLI intercepts it, then resumes
*that same agent* with the approved results. A specialist running inside
another agent's tool call would hand its ``DeferredToolRequests`` back as a
tool result the CLI never sees, and the approval loop would resume the wrong
agent -- so wrapping would break the guarantee that nothing is written without
confirmation.

Planning avoids that entirely. This module only *names* steps; the CLI runs
each specialist at the top level exactly as it does for a single-agent turn,
feeding each answer into the next step's prompt. The approval path is untouched
by construction rather than by care.

The module is named for what it does. ``query_analysis_agent`` is the
cautionary tale: a name promising delegation, wrapping a plain database query,
which hid two bugs because the name was believed over the code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from pydantic_ai import Agent

from aiida_agents._settings import ModelSettings, OllamaSettings
from aiida_agents.agents._models import get_model

logger = logging.getLogger(__name__)

__all__ = ["MAX_STEPS", "Specialist", "Step", "get_planner", "plan"]

#: The specialists a step can be assigned to.
Specialist = Literal["analysis", "execution"]

#: Same names as values, for matching a reply against.
_SPECIALISTS: tuple[Specialist, ...] = ("analysis", "execution")

#: Longest plan accepted. Beyond a few steps a plan is speculation about what
#: earlier steps will find, and the planner has seen none of them. A request
#: needing more is better served by planning the first steps and letting the
#: user continue from what they have learned.
MAX_STEPS = 3

_SYSTEM_PROMPT = (
    files(__package__).joinpath("prompt.md").read_text(encoding="utf-8").strip()
)


@dataclass(frozen=True)
class Step:
    """One specialist, and what it should do.

    An empty ``task`` means "the user's request, as they wrote it" -- the
    one-step case, where rephrasing it would only risk losing detail.
    """

    specialist: Specialist
    task: str


def get_planner(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
) -> Agent[None, str]:
    """Build the planning agent.

    It has no tools, so planning is one cheap round-trip that cannot touch the
    database.

    The output is plain text rather than a constrained type on purpose. A
    constrained type makes pydantic-ai request structured output, and this
    project has already met free-tier backends that cannot reliably emit valid
    tool-call JSON. A plan is the last thing that should fail to parse, so it
    is asked for as lines and parsed strictly (:func:`_parse_plan`).

    Args:
        model_settings: Model/provider config. Read from env/.env if not given.
        ollama_settings: Ollama endpoint config. Read from env/.env if not given.

    Returns:
        An agent that replies with one ``specialist: task`` line per step.
    """
    return Agent(
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        system_prompt=_SYSTEM_PROMPT,
        output_type=str,
    )


#: Leading list markers a model adds when it cannot resist formatting.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

#: A fenced code block's delimiter, likewise.
_FENCE = re.compile(r"^\s*```")


def _clean_line(line: str) -> str:
    """Strip formatting a model may wrap a step in, leaving the content."""
    return _LIST_MARKER.sub("", line).strip().strip("`").strip()


def _as_specialist(name: str) -> Specialist:
    """Narrow an already-validated name to the literal type."""
    return "execution" if name.strip().lower() == "execution" else "analysis"


def _parse_plan(reply: str) -> list[Step] | None:
    """Read a plan out of the reply, or ``None`` if any part is unusable.

    Tolerant of *formatting* -- numbering, bullets, code fences, casing, blank
    lines -- and strict about *content*. A line naming no specialist, naming
    one that does not exist, or carrying no task makes the whole plan unusable
    rather than being skipped. A plan silently missing a step is worse than no
    plan: the steps that do run are then acting on a premise nobody
    established.

    A bare specialist name on its own is accepted as a one-step plan over the
    user's original request. That was this agent's entire output format before
    it could plan, and a model answering that way is being unambiguous rather
    than wrong.

    Returns:
        The steps, or ``None`` -- meaning fall back rather than guess.
    """
    lines = [
        cleaned
        for raw in reply.strip().splitlines()
        if not _FENCE.match(raw) and (cleaned := _clean_line(raw))
    ]
    if not lines:
        return None

    if len(lines) == 1 and lines[0].lower() in _SPECIALISTS:
        return [Step(_as_specialist(lines[0]), "")]

    steps: list[Step] = []
    for line in lines:
        name, separator, task = line.partition(":")
        if (
            not separator
            or name.strip().lower() not in _SPECIALISTS
            or not task.strip()
        ):
            logger.debug("unusable plan line: %r", line)
            return None
        steps.append(Step(_as_specialist(name), task.strip()))

    if len(steps) > MAX_STEPS:
        logger.debug("plan of %d steps exceeds the cap of %d", len(steps), MAX_STEPS)
        return None
    return steps


def plan(
    question: str,
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
) -> list[Step]:
    """Plan how to serve one request.

    Falls back to a single read-only step if the call fails or the reply cannot
    be parsed. That degrades to the specialist which cannot write, so a
    planning failure costs the user a redirect rather than putting them in
    front of a submission flow they never asked for. The fallback is logged,
    not silent: a planner that has quietly stopped planning should be visible
    rather than inferred from the answers looking odd.

    Args:
        question: The user's request.
        model_settings: Model/provider config. Read from env/.env if not given.
        ollama_settings: Ollama endpoint config. Read from env/.env if not given.

    Returns:
        One to :data:`MAX_STEPS` steps.
    """
    try:
        reply = get_planner(model_settings, ollama_settings).run_sync(question).output
    except Exception as exc:  # noqa: BLE001 - planning must never be fatal
        logger.warning(
            "planning failed (%s); falling back to the read-only analysis agent", exc
        )
        return [Step("analysis", "")]

    steps = _parse_plan(reply)
    if steps is None:
        logger.warning(
            "planner replied %r, which is not a usable plan; "
            "falling back to the read-only analysis agent",
            reply[:200],
        )
        return [Step("analysis", "")]

    logger.debug("planned: %s", [(s.specialist, s.task) for s in steps])
    return steps
