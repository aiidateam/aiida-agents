"""Agent lifecycle for the CLI: build it, run it, probe it.

Turns resolved settings into a ready agent (``_build_agent``), runs a one-shot
query (``ask``), and probes the configured model for ``doctor``
(``_probe_reachable``, ``_probe_model``, ``_probe_failure_hint``). The heavy
aiida / agent-stack imports stay local so ``--help`` and shell completion never
load AiiDA.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, NamedTuple, TypeAlias

import rich_click as click
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage

from aiida_agents._settings import ModelSettings, _Provider, _format_validation_error
from aiida_agents.agents.planner import Specialist, Step, _as_specialist
from aiida_agents.cli.ollama import _ensure_ollama_model
from aiida_agents.agents.handoff import Handoff, NodeReference
from aiida_agents.cli.output import _trace_tool_calls, console


class _StepResult(NamedTuple):
    """What one executed step produced, for the next step to build on.

    ``node_references`` comes from the step's tool output, not its prose, so
    the next step works from identifiers the tools returned rather than from a
    number a second model read back out of a sentence.
    """

    specialist: Specialist
    answer: str
    node_references: tuple[NodeReference, ...] = ()


logger = logging.getLogger(__name__)

# The agents the CLI can launch. ``analysis`` (read-only exploration) and
# ``execution`` (the workflow generate/validate/submit pipeline) are the two
# specialists; ``auto`` (the default) picks between them per request with the
# router (ADR-09), so a user need not know the taxonomy. Naming a specialist
# explicitly overrides the router.
# The tuple is the single source for the ``--agent`` choice and the REPL's
# ``/agent`` switch, so the two never drift apart.
_AGENT_CHOICES = ("auto", "analysis", "execution", "codegen")
#: The specialists a request can actually be run by -- ``auto`` is a decision,
#: not an agent, so it is never passed to ``get_agent``.
_SPECIALISTS = ("analysis", "execution", "codegen")


async def ask(
    agent: Agent,
    question: str,
    message_history: list[ModelMessage] | None = None,
) -> Any:  # pragma: no cover
    """Run a single query through the agent, returning the result."""
    # DEBUG, not INFO: the raw prompt can carry secrets or private project detail,
    # so it only reaches an (opt-in) log sink when the user asks for debug output.
    logger.debug("agent query: %s", question)
    result = await agent.run(question, message_history=message_history)
    # Record the tool-call trace to the log file now (always); the console
    # render is the caller's job, done after any live spinner has stopped.
    _trace_tool_calls(result.new_messages())
    return result


def _resolve_model_settings(provider: str | None, model: str | None) -> ModelSettings:
    """Build ``ModelSettings`` with CLI overrides taking precedence.

    Only non-``None`` overrides are passed as constructor kwargs; pydantic-settings
    ranks init kwargs above the environment and ``.env`` (flag > env > file >
    default), so a ``--model`` flag wins over ``AIIDA_AGENTS_MODEL`` with no manual
    precedence handling.
    """
    # ``dict[str, Any]`` so the ``**`` splat's values stay assignable to each
    # typed field (``provider`` is a ``Literal``); pydantic validates them at
    # runtime, e.g. lower-casing and range-checking the provider string.
    overrides: dict[str, Any] = {
        key: value
        for key, value in (("provider", provider), ("model", model))
        if value is not None
    }
    return ModelSettings(**overrides)


def _resolve_settings_or_fail(provider: str | None, model: str | None) -> ModelSettings:
    """Resolve model settings, turning an invalid value into a clean CLI error.

    ``_resolve_model_settings`` raises pydantic's ``ValidationError`` (a raw
    traceback) on a bad ``AIIDA_AGENTS_*`` value. ``doctor`` builds settings
    outside a ``try``, so this converts it to a ``ClickException`` the way
    ``_build_agent`` does for its other configuration errors.
    """
    try:
        return _resolve_model_settings(provider, model)
    except ValidationError as exc:
        msg = f"Invalid configuration:\n{_format_validation_error(exc)}"
        raise click.ClickException(msg) from exc


def _build_agent(
    settings: ModelSettings, profile: str | None, agent_type: str = "analysis"
) -> Agent:  # pragma: no cover
    """Load the profile, open its storage, and build the requested agent.

    ``agent_type`` selects which agent to build (``"analysis"`` or
    ``"execution"``); the value is already constrained by the ``--agent`` choice
    and the REPL's ``/agent`` switch. ``"auto"`` is a routing decision rather
    than an agent, so callers resolve it with :func:`_resolve_plan` before
    getting here. The aiida / agent-stack imports stay local
    so ``--help`` and shell completion don't pay for loading AiiDA. Expected
    configuration failures are surfaced as clean CLI errors instead of a
    traceback.
    """
    from aiida import load_profile
    from aiida.common.exceptions import AiidaException

    from aiida_agents._profile import open_profile_storage
    from aiida_agents.agents import get_agent

    if agent_type not in _SPECIALISTS:
        msg = (
            f"{agent_type!r} is not a runnable agent. Resolve it with "
            "_resolve_plan() first."
        )
        raise ValueError(msg)

    try:
        _ensure_ollama_model(settings)
        load_profile(profile)
        # The agent's tools run on worker threads, so the storage has to be open
        # before the first one does; see ``open_profile_storage``.
        open_profile_storage()
        agent = get_agent(agent_type=agent_type, model_settings=settings)
    except (UserError, ValueError, AiidaException) as exc:
        # Every one of these is "fix your config", not a bug, so the message is
        # worth far more in the error box than at the bottom of a traceback.
        # UserError: pydantic-ai, for a missing cloud API key. ValueError:
        # get_model for an openai-compatible endpoint without a base_url (a bad
        # provider/setting value is already caught upstream at resolution).
        # AiidaException: a profile that does not exist, a storage still on an
        # older schema, an unreachable database -- AiiDA's own message names the
        # command that fixes each.
        raise click.ClickException(str(exc)) from exc
    return agent


def _probe_model(settings: ModelSettings) -> None:  # pragma: no cover
    """Fire a minimal generation against the configured model.

    A failure surfaces here (before a session), and for a local Ollama model the
    call loads it into memory so the first real query isn't a cold start. This is
    the heavy path behind ``doctor --warm``; the default report uses
    :func:`_probe_reachable`, which never generates.
    """
    from aiida_agents.agents._models import get_model

    probe = Agent(get_model(model_settings=settings))
    asyncio.run(probe.run("Reply with the single word: ok."))


# A reachability check must never hang, so the model listing is bounded.
_REACHABILITY_TIMEOUT = 8.0


# ``client`` is the provider's async SDK client (openai.AsyncOpenAI /
# anthropic.AsyncAnthropic), reached through ``model.client`` which the base
# pydantic-ai ``Model`` doesn't type; ``Any`` avoids a faithful-but-heavy
# Protocol for an object we only poke dynamically.
async def _list_model_ids(client: Any) -> set[str]:
    """Model ids the endpoint advertises, via a cheap listing under a timeout."""
    try:
        page = await asyncio.wait_for(
            fut=client.models.list(), timeout=_REACHABILITY_TIMEOUT
        )
    except asyncio.TimeoutError as exc:
        # ``asyncio.TimeoutError``, not the builtin: on Python 3.10 (our minimum)
        # ``wait_for`` raises the asyncio one, a distinct class from the builtin
        # ``TimeoutError`` (they were only merged into aliases in 3.11), so
        # catching the builtin would let the timeout escape uncaught there.
        # Phrase the message so ``_probe_failure_hint`` routes it to the
        # "unreachable" branch (it matches on "connect").
        msg = f"could not connect within {_REACHABILITY_TIMEOUT:.0f}s"
        raise ConnectionError(msg) from exc
    return {item.id for item in page.data}


class _Reachability(NamedTuple):
    """What a no-generation reachability probe learns about the endpoint."""

    endpoint: str
    n_models: int
    model_ok: bool


def _probe_reachable(settings: ModelSettings) -> _Reachability:
    """Reachability facts without a generation: ``(endpoint, n_models, model_ok)``.

    Builds the model (validating provider / base_url / key presence), then lists
    the endpoint's models (one cheap GET under a short timeout). ``model_ok`` is
    whether the configured model is among those advertised. Raises on an
    unreachable endpoint, bad key, or bad config. Behind ``doctor``'s model
    row; never loads or runs the model.
    """
    from aiida_agents.agents._models import get_model

    model = get_model(model_settings=settings)
    # OpenAIChatModel / AnthropicModel both expose the underlying SDK client; the
    # base ``Model`` type does not, hence the ignore.
    client: Any = model.client  # type: ignore[attr-defined]
    ids = asyncio.run(_list_model_ids(client))
    # Ollama lists an untagged model as ``<name>:latest``; normalise so an
    # untagged configured name still matches. Cloud ids carry no such suffix.
    wanted = settings.model
    if settings.provider == "ollama" and ":" not in wanted:
        wanted = f"{wanted}:latest"
    model_ok = wanted in ids or settings.model in ids
    return _Reachability(
        endpoint=str(client.base_url), n_models=len(ids), model_ok=model_ok
    )


# How a probed model's availability maps to a single policy decision, kept apart
# from the rendering so the "is it usable?" rule lives in one place.
_ModelAvailability: TypeAlias = Literal["available", "not_pulled", "unlisted"]


def _model_availability(
    reach: _Reachability, provider: _Provider
) -> _ModelAvailability:
    """Classify whether the configured model is usable at the probed endpoint.

    Ollama's ``/models`` listing is authoritative, so a model absent from it is
    genuinely not pulled; a cloud (or openai-compatible) endpoint's listing may be
    partial, so an absent model is only "unlisted" and may still work. ``doctor``
    renders the outcome as a row.
    """
    if reach.model_ok:
        return "available"
    return "not_pulled" if provider == "ollama" else "unlisted"


def _not_pulled_detail(settings: ModelSettings) -> str:
    """The one sentence for "this Ollama model is not installed".

    Reached two ways a user cannot tell apart and should not have to: the
    endpoint's listing not carrying the model, and the probe raising a 404 for
    it. Both render this, so the report does not word one condition twice.
    """
    return f"model not pulled (ollama pull {settings.model})"


def _probe_failure_hint(settings: ModelSettings, exc: Exception) -> str | None:
    """A model-probe failure phrased as what to do about it, or ``None``.

    The three failures a user can actually fix (an unpulled Ollama model, a bad
    or absent API key, a server that is not up) each surface as provider-specific
    SDK wording that says what broke and never what to do. Classification is by
    message text because that is the only thing those exceptions have in common.

    ``None`` means the failure is not one of the three, so the caller should fall
    back to the exception's own message rather than print a guess.
    """
    msg = str(exc).lower()
    if settings.provider == "ollama" and ("not found" in msg or "404" in msg):
        return _not_pulled_detail(settings)
    if ("api" in msg and "key" in msg) or "401" in msg or "403" in msg:
        if "not set" in msg or "environment variable" in msg or "set the" in msg:
            return "API key not set; set the provider's API key"
        return "authentication failed; check the provider's API key"
    if "connect" in msg or "connection" in msg:
        return "could not reach the endpoint; is the server running?"
    return None


def _resolve_plan(
    agent_type: str, question: str, settings: ModelSettings
) -> list[Step]:  # pragma: no cover
    """Turn the ``--agent`` value and the request into the steps to run.

    ``auto`` asks the planner (ADR-09). An explicitly named specialist is
    honoured as a single step, so ``-a execution`` remains a way to bypass
    planning entirely -- for debugging, and for a user who already knows what
    they want.

    The plan is printed rather than executed silently. Which specialist
    answered changes what the answer can be, and a planner that has quietly
    stopped planning should be visible in the output rather than inferred from
    odd replies. It is shown, not confirmed: every write is already gated by
    ``requires_approval``, so a plan can only *read* before the user is asked,
    and a second prompt stacked on the approval prompt would train people to
    dismiss both.
    """
    from aiida_agents.agents.planner import Step, plan

    if agent_type != "auto":
        return [Step(_as_specialist(agent_type), "")]

    steps = plan(question, settings)
    if len(steps) == 1:
        console.print(f"[dim]→ {steps[0].specialist} agent[/dim]")
    else:
        console.print("[dim]→ plan:[/dim]")
        for index, step in enumerate(steps, start=1):
            console.print(f"[dim]   {index}. {step.specialist}: {step.task}[/dim]")
    return steps


#: How an earlier step's answer is handed to the next one.
#:
#: Explicit text rather than replayed message history: the specialists hold
#: different tools, so one agent's history references tools the other does not
#: have. Labelled with its source so the receiving agent can weigh it as a
#: finding rather than as the user's own words -- and so a user reading the
#: transcript can see exactly what was carried forward.
def _step_prompt(step: Step, question: str, previous: _StepResult | None) -> str:
    """The prompt for one step: its task, plus what the last step found.

    A step with no task of its own runs the user's request verbatim -- the
    single-step case, where rephrasing could only lose detail.

    With a previous step, the prompt is a rendered :class:`Handoff`: its prose
    findings *and* the node references its tools produced. See
    ``agents/handoff.py`` for why the second half exists.
    """
    task = step.task or question
    if previous is None:
        return task
    return Handoff(
        from_specialist=previous.specialist,
        to_specialist=step.specialist,
        task=task,
        findings=previous.answer,
        node_references=previous.node_references,
    ).render()
