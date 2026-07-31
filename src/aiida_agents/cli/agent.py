"""Agent lifecycle for the CLI: build it, run it, probe it.

Turns resolved settings into a ready agent (``_build_agent``), runs a one-shot
query (``ask``), and checks/warms the configured model (``_probe_model`` +
``_diagnose_probe_failure``). The heavy aiida / agent-stack imports stay local so
``--help`` and shell completion never load AiiDA.
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
from typing_extensions import assert_never

from aiida_agents._settings import ModelSettings, _Provider, _format_validation_error
from aiida_agents.agents.planner import Specialist, Step
from aiida_agents.cli.ollama import _ensure_ollama_model, _ollama_pull
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
_AGENT_CHOICES = ("auto", "analysis", "execution")
#: The specialists a request can actually be run by -- ``auto`` is a decision,
#: not an agent, so it is never passed to ``get_agent``.
_SPECIALISTS = ("analysis", "execution")

# Colored status glyphs so ``check`` marks success/failure the same green/red as
# ``doctor`` (which renders its rows through rich). ``click.echo`` strips the
# color automatically when the output is not a terminal.
_OK = click.style("✓", fg="green")
_FAIL = click.style("✗", fg="red")
_WARN = click.style("!", fg="yellow")


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
    traceback) on a bad ``AIIDA_AGENTS_*`` value. The read-only diagnostics
    commands (``check``/``warm``/``doctor``) build settings outside a ``try``, so
    this converts it to a ``ClickException`` the way ``_build_agent`` does for its
    other configuration errors.
    """
    try:
        return _resolve_model_settings(provider, model)
    except ValidationError as exc:
        msg = f"Invalid configuration:\n{_format_validation_error(exc)}"
        raise click.ClickException(msg) from exc


def _build_agent(
    settings: ModelSettings, profile: str | None, agent_type: str = "analysis"
) -> Agent:  # pragma: no cover
    """Load the profile and build the requested agent from resolved settings.

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
        agent = get_agent(agent_type=agent_type, model_settings=settings)
    except (UserError, ValueError) as exc:
        # UserError: pydantic-ai, for a missing cloud API key. ValueError:
        # get_model for an openai-compatible endpoint without a base_url. Both
        # are "fix your config", not bugs (a bad provider/setting value is
        # already caught upstream at resolution), so show a clean error.
        raise click.ClickException(str(exc)) from exc
    return agent


def _probe_model(settings: ModelSettings) -> None:  # pragma: no cover
    """Fire a minimal generation against the configured model.

    A failure surfaces here (before a session), and for a local Ollama model the
    call loads it into memory so the first real query isn't a cold start. This is
    the heavy path behind ``warm``; ``check`` uses :func:`_check_reachable`, which
    never generates.
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
        # Phrase the message so ``_diagnose_probe_failure`` routes it to the
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
    unreachable endpoint, bad key, or bad config. Shared by ``check`` and
    ``doctor``; never loads or runs the model.
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


# How a probed model's availability maps to a single policy decision, shared by
# ``check`` and ``doctor`` so the "is it usable?" rule lives in one place.
_ModelAvailability: TypeAlias = Literal["available", "not_pulled", "unlisted"]


def _model_availability(
    reach: _Reachability, provider: _Provider
) -> _ModelAvailability:
    """Classify whether the configured model is usable at the probed endpoint.

    Ollama's ``/models`` listing is authoritative, so a model absent from it is
    genuinely not pulled; a cloud (or openai-compatible) endpoint's listing may be
    partial, so an absent model is only "unlisted" and may still work. Callers
    render this as they need (``check`` echoes and exits; ``doctor`` builds a row).
    """
    if reach.model_ok:
        return "available"
    return "not_pulled" if provider == "ollama" else "unlisted"


def _check_reachable(settings: ModelSettings) -> None:
    """Print ``check``'s reachability report; exit non-zero on a real problem.

    Reachability / auth failures raise for the caller to diagnose. A configured
    model the endpoint doesn't advertise is fatal for Ollama (its listing is
    authoritative, so it means "not pulled"), but only a warning for cloud or
    openai-compatible endpoints, whose ``/models`` listing may be partial.
    """
    reach = _probe_reachable(settings)
    click.echo(f"Endpoint: {reach.endpoint}")
    click.echo(f"{_OK} reachable ({reach.n_models} models advertised)")
    availability = _model_availability(reach, settings.provider)
    if availability == "available":
        click.echo(f"{_OK} model '{settings.model}' is available")
        return
    if availability == "not_pulled":
        click.echo(f"{_FAIL} model '{settings.model}' is not pulled.", err=True)
        click.echo(f"  Pull it with: ollama pull {settings.model}", err=True)
        raise SystemExit(1)
    if availability == "unlisted":
        click.echo(
            f"{_WARN} model '{settings.model}' is not in this endpoint's list "
            "(it may be partial; the model may still work).",
            err=True,
        )
        return
    assert_never(availability)  # pragma: no cover


def _diagnose_probe_failure(settings: ModelSettings, exc: Exception) -> None:
    """Turn a probe failure into an actionable message, offering an Ollama pull."""
    msg = str(exc).lower()
    if settings.provider == "ollama" and ("not found" in msg or "404" in msg):
        click.echo(f"{_FAIL} Ollama model '{settings.model}' is not pulled.", err=True)
        if click.confirm(f"Pull it now (ollama pull {settings.model})?", default=True):
            _ollama_pull(settings.model)
        return
    if ("api" in msg and "key" in msg) or "401" in msg or "403" in msg:
        if "not set" in msg or "environment variable" in msg or "set the" in msg:
            click.echo(
                f"{_FAIL} API key not set: set the provider's API key.", err=True
            )
        else:
            click.echo(
                f"{_FAIL} Authentication failed: check the provider's API key.",
                err=True,
            )
        return
    if "connect" in msg or "connection" in msg:
        click.echo(
            f"{_FAIL} Could not reach the endpoint. Is the server running?", err=True
        )
        return
    click.echo(f"{_FAIL} {exc}", err=True)


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
        return [Step(_as_specialist_literal(agent_type), "")]

    steps = plan(question, settings)
    if len(steps) == 1:
        console.print(f"[dim]→ {steps[0].specialist} agent[/dim]")
    else:
        console.print("[dim]→ plan:[/dim]")
        for index, step in enumerate(steps, start=1):
            console.print(f"[dim]   {index}. {step.specialist}: {step.task}[/dim]")
    return steps


def _as_specialist_literal(name: str) -> Specialist:
    """Narrow a ``--agent`` value the Click choice has already constrained."""
    return "execution" if name == "execution" else "analysis"


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
