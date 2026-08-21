"""Capture and assert on what an agent *did*, not just what it said.

Every grounding defect found in this project so far was found by a human
running the CLI and noticing the answer contained a number nobody had
retrieved. The unit suite could not have caught any of them: it exercises
tools in isolation, and the defects live in the agent's behaviour --- which
tool it reached for, and whether its prose is traceable to what that tool
returned.

This module turns that manual read-through into assertions. It records a run
into a :class:`RunTrace` (tool calls in order, what each returned, and the
final text), and provides the two checks that correspond to the two ways an
agent has actually gone wrong here:

``assert_consulted_docs``
    Did it look anything up before answering? The Execution agent once routed
    every workflow-shaped question through a fixed tool progression that had
    no documentation step, so it answered parameter questions from memory
    without ever consulting the docs.

``ungrounded_quantities``
    Does every physical quantity in the answer appear in something a tool
    returned? The detection itself lives in ``aiida_agents.grounding``, which
    the CLI also runs on every reply -- one implementation, so the check that
    guards a shipped answer and the check that guards this suite cannot drift
    apart. A model that retrieves correctly can still garnish the answer
    with a plausible cutoff or k-point spacing it invented. That reads as
    authoritative, is unverifiable by the user, and quietly configures a wrong
    calculation.

The harness is model-agnostic on purpose: the same assertions run against a
scripted ``FunctionModel`` in CI (see ``test_harness.py``, which proves the
checks actually fire) and against a real model in the opt-in tier (see
``test_grounding.py``).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiida_agents.grounding import quantities_in as quantities_in  # re-exported
from aiida_agents.grounding import ungrounded_quantities as _ungrounded
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


def project_env_file() -> Path:
    """Path the developer's ``.env`` would occupy, whether or not it exists."""
    return Path(__file__).resolve().parents[2] / ".env"


def copy_project_env(destination: Path) -> Path | None:
    """Put the developer's ``.env`` where ``ModelSettings`` will find it.

    ``tests/conftest.py`` chdirs every test into an empty temp directory, on
    purpose, so that a developer's ``.env`` cannot leak into tests that are
    supposed to be hermetic. This tier needs the opposite: it exists to run
    against whatever model the developer actually configured, and pydantic-
    settings reads ``.env`` relative to the working directory.

    Without this the eval resolved to ``ModelSettings``' bare defaults --
    ``ollama``/``qwen3.5:2b`` -- and reported grounding failures for a model
    that had never been contacted. The conftest docstring prescribes the way
    out ("a test that needs a ``.env`` writes its own and chdirs to it"); this
    is that, using the real one.

    Exported shell variables survive the chdir and are unaffected either way;
    they still win over the copied file, as pydantic-settings intends.

    Args:
        destination: Directory to copy into -- normally the test's cwd.

    Returns:
        The written path, or ``None`` when the project has no ``.env``.
    """
    source = project_env_file()
    if not source.is_file():
        return None
    target = destination / ".env"
    shutil.copy(source, target)
    return target


@dataclass
class ToolCall:
    """One tool invocation and what it returned."""

    name: str
    args: Any
    output: str = ""


@dataclass
class RunTrace:
    """What an agent did during a single run.

    Attributes:
        calls: Tool invocations in the order the model made them.
        answer: The final text the user would see.
        messages: The raw message list, for assertions this class doesn't cover.
    """

    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    messages: list[ModelMessage] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        """Names of the tools called, in order, including repeats."""
        return [c.name for c in self.calls]

    def called(self, name: str) -> bool:
        return name in self.tool_names

    def outputs_for(self, name: str) -> list[str]:
        return [c.output for c in self.calls if c.name == name]

    @property
    def all_output(self) -> str:
        """Every tool return concatenated --- the evidence available to the model.

        This is the corpus a claim in the answer has to be traceable to. It
        deliberately includes *all* tools, not just the documentation search:
        a k-point spacing taken from ``build_process_inputs`` is grounded
        just as legitimately as one quoted from the docs.
        """
        return "\n".join(c.output for c in self.calls)


def _stringify(content: Any) -> str:
    """Flatten a tool return to text for substring searching.

    Tool returns here are dicts, lists, and dataclass-ish objects as often as
    strings, and a number inside a nested dict is still a number the model was
    given. ``repr`` keeps nested structure visible where ``str`` on a dict
    would too, but ``repr`` is stable for floats, which is what matters: the
    grounding check compares numeric literals.
    """
    return content if isinstance(content, str) else repr(content)


def trace_run(agent: Agent, prompt: str, **kwargs: Any) -> RunTrace:
    """Run ``agent`` on ``prompt`` and record what it did.

    Args:
        agent: The agent to run.
        prompt: The user turn to send.
        **kwargs: Forwarded to ``agent.run_sync`` (e.g. ``deps``).

    Returns:
        A :class:`RunTrace` of the tool calls, their outputs, and the answer.
    """
    with capture_run_messages() as messages:
        result = agent.run_sync(prompt, **kwargs)

    trace = RunTrace(messages=list(messages))

    # Tool calls and their returns arrive in separate messages, linked by
    # tool_call_id. Walk once, index by id, so a return is attached to its own
    # call even when the model issued several in parallel.
    by_id: dict[str, ToolCall] = {}
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    call = ToolCall(name=part.tool_name, args=part.args)
                    trace.calls.append(call)
                    if part.tool_call_id:
                        by_id[part.tool_call_id] = call
        elif isinstance(message, ModelRequest):
            for request_part in message.parts:
                if isinstance(request_part, ToolReturnPart):
                    returned_to = by_id.get(request_part.tool_call_id or "")
                    if returned_to is not None:
                        returned_to.output = _stringify(request_part.content)

    trace.answer = str(getattr(result, "output", "") or "")
    if not trace.answer:
        # A run ending in a deferred tool request has no text output; fall back
        # to the last text the model produced so citation checks still apply.
        for message in reversed(messages):
            if isinstance(message, ModelResponse):
                texts = [p.content for p in message.parts if isinstance(p, TextPart)]
                if texts:
                    trace.answer = "\n".join(texts)
                    break

    return trace


def ungrounded_quantities(trace: RunTrace, prompt: str = "") -> set[str]:
    """Quantities in the answer that no tool output and no user turn contains.

    This is the fabrication check. A number the model states as physics has to
    come from somewhere the user can verify: a tool return, or their own
    question echoed back. Anything else it recalled, and a recalled cutoff is
    indistinguishable from a correct one until a calculation has burned CPU
    hours on it.

    Args:
        trace: A recorded run.
        prompt: The user turn, so a value the user supplied isn't reported as
            invented when the agent repeats it.

    Returns:
        Normalised numeric literals with no source. Empty means every physics
        number in the answer is traceable.
    """
    return _ungrounded(trace.answer, trace.all_output, prompt)


def assert_grounded_quantities(trace: RunTrace, prompt: str = "") -> None:
    """Fail if the answer states a physical quantity no tool provided."""
    invented = ungrounded_quantities(trace, prompt)
    assert not invented, (
        f"answer asserts quantities absent from every tool output: "
        f"{sorted(invented)}\ntools called: {trace.tool_names}\n"
        f"answer:\n{trace.answer}"
    )


def assert_consulted_docs(trace: RunTrace, tool: str = "search_aiida_docs") -> None:
    """Fail if the agent answered a knowledge question without looking it up."""
    assert trace.called(tool), (
        f"agent answered without calling {tool!r}; "
        f"tools called: {trace.tool_names}\nanswer:\n{trace.answer}"
    )


#: Matches the bracketed citation both prompts mandate, e.g.
#: ``[howto/run_workflows  §  Work chains]`` or ``[quantumespresso: ...]``.
_CITATION = re.compile(r"\[[^\]]+\]")


def assert_cited(trace: RunTrace) -> None:
    """Fail if an answer built on retrieved docs carries no citation.

    Both system prompts make inline attribution mandatory. An uncited answer
    next to a successful ``search_aiida_docs`` call is the signature of a model
    that retrieved and then answered from memory anyway.
    """
    assert _CITATION.search(trace.answer), (
        f"answer cites no source despite calling {trace.tool_names}\n"
        f"answer:\n{trace.answer}"
    )
