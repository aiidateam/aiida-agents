"""Router — picks which specialist agent should answer a request.

ADR-09 decides the orchestration layer arrives in two increments: routing
first, multi-step coordination second. This is the first.

It is deliberately *not* a coordinator that wraps the specialists. The
human-in-the-loop mechanism (ADR-08) works because a specialist's run returns
``DeferredToolRequests`` as its output and the CLI intercepts it, then resumes
*that same agent* with the approved results. A specialist running inside
another agent's tool call would hand its ``DeferredToolRequests`` back as a
tool result the CLI never sees, and the approval loop would resume the wrong
agent -- so wrapping would break the guarantee that nothing is written without
confirmation.

Routing avoids that entirely: the router only names a specialist, and the CLI
runs it exactly as it does today. The approval path is untouched. Multi-step
coordination, which does need a specialist's result to feed the next step, is
the increment that has to solve that plumbing first.

The module is named for what it does. ``query_analysis_agent`` is the
cautionary tale: a name promising delegation, wrapping a plain database query,
which hid two bugs because the name was believed over the code.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

from pydantic_ai import Agent

from aiida_agents._settings import ModelSettings, OllamaSettings
from aiida_agents.agents._models import get_model

#: The specialists the router chooses between.
Specialist = Literal["analysis", "execution"]

#: Same names as values, for matching a reply against.
_SPECIALISTS: tuple[Specialist, ...] = ("analysis", "execution")

_SYSTEM_PROMPT = (
    files(__package__).joinpath("prompt.md").read_text(encoding="utf-8").strip()
)


def get_router(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
) -> Agent[None, str]:
    """Build the routing agent.

    It has no tools, so a routing call is one cheap round-trip that cannot
    touch the database.

    The output is plain text rather than a constrained type on purpose. A
    constrained output makes pydantic-ai ask the model for structured output,
    and this project has already seen free-tier routers whose backends cannot
    reliably emit valid tool-call JSON -- which would fail routing for a
    one-word answer. Asking for a word and parsing it strictly
    (:func:`_parse_choice`) works on any model that can follow an instruction.

    Args:
        model_settings: Model/provider config. Read from env/.env if not given.
        ollama_settings: Ollama endpoint config. Read from env/.env if not given.

    Returns:
        An agent that replies with the name of the specialist to run.
    """
    return Agent(
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        system_prompt=_SYSTEM_PROMPT,
        output_type=str,
    )


def _parse_choice(reply: str) -> Specialist | None:
    """Read a specialist name out of the router's reply, or None if unclear.

    A bare word is the expected case. A chatty model may wrap it in a sentence,
    which is accepted only when exactly one specialist is named: "not
    execution, use analysis" mentions both, and resolving that by whichever
    name is checked first would be a guess dressed as a decision.
    """
    text = reply.strip().strip(".'\"` \n").lower()
    if text in _SPECIALISTS:
        return text

    named = [name for name in _SPECIALISTS if name in text]
    return named[0] if len(named) == 1 else None


def route(
    question: str,
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
) -> Specialist:
    """Choose the specialist for one request.

    Falls back to ``"analysis"`` if the call fails or the reply cannot be read.
    That is the read-only specialist, so a failed routing decision degrades to
    an agent that cannot write rather than to no answer at all -- and never to
    one that can submit. The fallback is logged rather than silent: a router
    that has quietly stopped routing should be visible, not inferred from the
    answers looking odd.

    Args:
        question: The user's request.
        model_settings: Model/provider config. Read from env/.env if not given.
        ollama_settings: Ollama endpoint config. Read from env/.env if not given.

    Returns:
        ``"analysis"`` or ``"execution"``.
    """
    import logging

    logger = logging.getLogger(__name__)
    try:
        reply = get_router(model_settings, ollama_settings).run_sync(question).output
    except Exception as exc:  # noqa: BLE001 - routing must never be fatal
        logger.warning(
            "routing failed (%s); falling back to the read-only analysis agent", exc
        )
        return "analysis"

    choice = _parse_choice(reply)
    if choice is None:
        logger.warning(
            "router replied %r, which names no single specialist; "
            "falling back to the read-only analysis agent",
            reply[:200],
        )
        return "analysis"
    logger.debug("routed to %s", choice)
    return choice
