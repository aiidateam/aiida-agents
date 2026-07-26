"""Sanity checks for the Execution agent system prompt.

Like the Analysis agent's equivalent, these are intentionally lightweight —
they guard against the prompt file being erased, truncated, or drifting out of
step with the tools actually registered on the agent. Whether the prompt
*routes* well is evaluated manually.
"""

from __future__ import annotations

from aiida_agents.agents.execution import _READ_TOOLS
from aiida_agents.agents.execution import _SYSTEM_PROMPT as SYSTEM_PROMPT

#: HITL-gated writes, registered via ``agent.tool_plain`` rather than in
#: ``_READ_TOOLS``, so they are not discoverable from that list.
_WRITE_TOOL_NAMES = ("execute_workflow_spec", "import_structure")


class TestSystemPrompt:
    def test_is_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)

    def test_non_empty(self) -> None:
        assert SYSTEM_PROMPT.strip()

    def test_mentions_every_registered_tool(self) -> None:
        """A tool the agent can call but the prompt never names is a tool it won't use.

        Derived from ``_READ_TOOLS`` rather than a hardcoded list so that adding
        a tool without giving the model any guidance on when to reach for it
        fails here instead of silently shipping. ``search_aiida_docs`` is the
        case that motivated this: it was registered and available, but the
        prompt's only imperative tool ordering did not include it, so
        workflow-shaped questions bypassed the docs entirely.
        """
        names = [t.__name__ for t in _READ_TOOLS] + list(_WRITE_TOOL_NAMES)
        missing = [n for n in names if n not in SYSTEM_PROMPT]
        assert not missing, f"prompt does not mention registered tools: {missing}"

    def test_separates_knowledge_questions_from_setup_requests(self) -> None:
        """The progression must not be the prompt's only entry point.

        Regression guard: the ordered ``discover -> ... -> execute`` list was
        introduced by "requests a calculation or asks to set up a workflow",
        which a question that merely *names* a WorkChain also matches. The
        agent then walked the progression -- which contains no documentation
        step -- and answered parameter questions from memory. The prompt has to
        say somewhere that naming a workflow is not by itself a setup request.
        """
        assert "which kind of request is this?" in SYSTEM_PROMPT.lower()
        assert "does not make it a setup request" in SYSTEM_PROMPT
