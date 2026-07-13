"""Command-line interface for aiida-agents.

Public API
----------
cli
    The root Click command group: the ``aiida-agents`` console entry point.

The command surface is split by concern:

* ``agent``: build, run, and probe the agent for a command
* ``commands``: the root group and the core commands (chat/ask/check/warm)
* ``config``: the ``config`` group and the effective-settings data behind it
* ``doctor``: the ``doctor`` command and its health checks (profile/model/RAG/docs)
* ``_guards``: the fail-fast guard for a mistyped settings key
* ``hitl``: the human-in-the-loop write-approval flow
* ``mcp``: the ``mcp`` group (run the MCP server from the CLI)
* ``ollama``: local Ollama model provisioning (presence checks, pull prompts)
* ``output``: the shared console and reply rendering
* ``rag``: the ``rag`` group and the docs-toolchain provisioning helpers
* ``repl``: the interactive REPL loop and prompt_toolkit session
"""

from __future__ import annotations

from aiida_agents.cli.commands import cli

__all__ = ["cli"]
