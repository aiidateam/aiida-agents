"""The contract a plugin implements to extend the agent.

A plugin package (aiida-quantumespresso, ...) points one ``aiida_agents.plugins``
entry point at a provider object implementing :class:`AgentPlugin`. Every hook is
optional: implement only the ones you have.

The types here are deliberately plain (dataclasses + a structural ``Protocol``)
and import nothing from pydantic-ai or chromadb, so a plugin can build a provider
without those in its dependency tree (ADR-08's write-gating still applies: a tool
declaring ``writes=True`` is registered behind the human-in-the-loop gate).
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "AgentPlugin",
    "AgentTool",
    "RagCorpus",
]

# The entry-point group a plugin declares its provider under.
PLUGIN_ENTRY_POINT_GROUP = "aiida_agents.plugins"


@dataclass(frozen=True)
class RagCorpus:
    """A documentation corpus a plugin wants indexed and cited as its own.

    ``name`` and ``version`` form the index identity: a version bump resolves to a
    different collection and triggers a rebuild rather than serving a stale index.
    Set ``version`` from the plugin's own distribution version so the corpus can
    never silently disagree with the installed code.

    Give exactly one source. ``text_dir`` (pre-rendered text the plugin ships or
    builds) is used as-is; otherwise ``docs_repo`` is cloned at ``docs_ref`` and
    rendered with the plugin's own ``docs`` extra in an isolated build
    environment.

    ``docs_url`` is where the same documentation is published, as a template
    taking ``{version}`` and ``{page}`` --- for a Read the Docs project that is
    ``"https://<project>.readthedocs.io/en/{version}/{page}.html"``. Give it and
    every passage retrieved from this corpus is cited with a link the user can
    open; leave it unset and the passage is still returned, just unlinked. It is
    optional because contributing documentation and hosting it are separate
    things, and a guessed URL is worse than none.
    """

    name: str
    version: str
    docs_repo: str | None = None
    docs_ref: str | None = None
    docs_subdir: str | None = None
    text_dir: Path | None = None
    docs_url: str | None = None

    @property
    def has_source(self) -> bool:
        """True if the corpus names somewhere to read documentation from."""
        return self.text_dir is not None or self.docs_repo is not None


@dataclass(frozen=True)
class AgentTool:
    """A tool a plugin contributes to the agent.

    ``fn`` is a plain function; its name, signature, and docstring become the
    tool the model sees, exactly as for a built-in tool, so document it for a
    reader who has never used your plugin.

    ``writes=True`` marks a tool that changes state (submitting, storing,
    deleting). Such a tool is registered behind the human-in-the-loop gate: the
    user is shown the call and must approve it before it runs (ADR-08). Read
    tools are wrapped so a failure comes back to the model as a recoverable
    retry instead of aborting the run.
    """

    fn: t.Callable[..., t.Any]
    writes: bool = False

    @property
    def name(self) -> str:
        """The tool name the model sees (the function's own name)."""
        return getattr(self.fn, "__name__", repr(self.fn))


@t.runtime_checkable
class AgentPlugin(t.Protocol):
    """What a provider may offer. Every hook is optional.

    Hooks are read defensively: a provider that omits a hook, or whose hook
    raises, is skipped for that hook alone and never breaks agent construction.
    """

    name: str

    def tools(self) -> t.Sequence[AgentTool]:
        """Tools to register on the agent."""
        ...

    def rag_corpora(self) -> t.Sequence[RagCorpus]:
        """Documentation corpora to index and cite."""
        ...

    def prompt_fragment(self) -> str | None:
        """Domain guidance to add to the system prompt.

        Say what only this plugin knows (its conventions, its physics). Do not
        restate the agent's workflow: the core prompt wins on any conflict, and
        the fragment is budgeted (see ``discovery.MAX_PROMPT_FRAGMENT_CHARS``).
        """
        ...
