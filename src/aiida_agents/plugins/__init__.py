"""Plugin extensibility: let an installed AiiDA plugin extend the agent.

A plugin declares one ``aiida_agents.plugins`` entry point pointing at a provider
object; the agent discovers it at build time and registers whatever the provider
offers. Nothing here is required for a plugin's *processes* to be usable: those
are already found through AiiDA's own ``aiida.workflows`` / ``aiida.calculations``
registry. This channel carries what that registry cannot: domain tools,
documentation corpora, and prompt guidance.

All three hooks are wired: ``get_agent()`` registers a contributed tool and
appends a contributed fragment to the system prompt, and ``aiida-agents rag
build`` indexes each contributed ``rag_corpora`` entry into its own collection
(``rag.indexing.index_plugin_corpora``), which ``search_aiida_docs`` then
searches alongside the core AiiDA docs, attributing each hit to the corpus it
came from.

Public API
----------
AgentPlugin, AgentTool, RagCorpus, PLUGIN_ENTRY_POINT_GROUP
    The contract a plugin implements. Every hook is optional.
discover_plugins()
    Load every registered provider, error-isolated: a broken plugin is logged
    and skipped, never fatal to ``get_agent()``.
LoadedPlugin
    What one provider successfully contributed.
"""

from __future__ import annotations

from aiida_agents.plugins.discovery import LoadedPlugin, discover_plugins
from aiida_agents.plugins.spec import (
    PLUGIN_ENTRY_POINT_GROUP,
    AgentPlugin,
    AgentTool,
    RagCorpus,
)

__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "AgentPlugin",
    "AgentTool",
    "LoadedPlugin",
    "RagCorpus",
    "discover_plugins",
]
