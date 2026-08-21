"""Codegen Agent — writes Python against the user's data, runs it, shows both.

The third specialist, and the first one added since the split was drawn. It
earns its place on the two grounds ``docs/extending.md`` sets out, and it is
worth stating them rather than leaving the reader to infer them.

**Its own tool surface.** Two tools, one of which executes code. Adding that to
the Analysis agent would widen a twelve-tool read-only surface with the single
capability whose blast radius is different in kind from all the others, and
would make "the read-only agent" a name that needed a footnote.

**Its own prompt.** Analysis explains what is in a database. This one writes a
program, runs it, reads its own traceback and tries again. That loop is the
whole job, and folding it into a prompt about explaining findings produced an
agent that did neither well.

What it does *not* have is a write tool, and that is deliberate. Executing
generated code sounds like the most dangerous thing in this project, and it
would be if it ran against the user's live profile. It does not: it runs
against a disposable copy of their storage, so the danger is handled by what
the code can reach rather than by asking a user to vet Python at a prompt.
Approving twenty lines of unexecuted code is a weak check. Reading what the
code actually returned is a real one --- which is why this agent is not gated
and every tool that genuinely writes still is, on the Execution agent, where
the preview is a resolved input a researcher can judge in seconds.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from aiida_agents._settings import AgentSettings, ModelSettings, OllamaSettings
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents._models import get_model
from aiida_agents.agents.reroute import hand_off_to
from aiida_agents.rag import search_aiida_docs, search_aiida_examples
from aiida_agents.tools.codegen import run_python_snippet

#: Everything this agent can do. ``search_aiida_examples`` comes first because the
#: prompt requires it before any Python is written: the failure this agent is
#: most prone to is a confident call to a method that does not exist.
_TOOLS: list[Any] = [
    search_aiida_examples,  # Worked examples to write from (read-only)
    run_python_snippet,  # Run the snippet against the sandbox copy (read-only)
    search_aiida_docs,  # What a concept means, when the examples are not enough (read-only)
]

_SYSTEM_PROMPT = (
    files(__package__).joinpath("prompt.md").read_text(encoding="utf-8").strip()
)


def get_agent(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    agent_settings: AgentSettings | None = None,
) -> Agent:
    """Build and return the Codegen Agent.

    Registers no approval-gated tool, because it holds none that can write ---
    see the module docstring. ``output_type`` is a plain ``str`` for the same
    reason: with nothing deferrable, a run always ends in an answer, and
    declaring ``DeferredToolRequests`` would advertise a pause that can never
    happen.

    Args:
        model_settings: Model/provider config. Read from env/.env if not given.
        ollama_settings: Ollama endpoint config. Read from env/.env if not given.
        agent_settings: Agent behaviour (retry budget). Read from env/.env if
            not given.

    Returns:
        Agent: Ready-to-use Codegen Agent instance.
    """
    cfg = agent_settings if agent_settings is not None else AgentSettings()

    # Retry-wrapped like every other read surface: a tool that fails comes back
    # to the model as a recoverable error rather than aborting the run. That
    # matters more here than elsewhere -- fixing its own snippet from the error
    # is the agent's normal working loop, not an exceptional path.
    toolset = RetryOnToolError(FunctionToolset(_TOOLS + [hand_off_to]))

    return Agent(
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        toolsets=[toolset],
        retries=cfg.tool_retries,
        system_prompt=_SYSTEM_PROMPT,
    )
