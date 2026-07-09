"""Command-line interface for aiida-agents.

The Click entry point (``cli``) plus its command surface, split by concern:

* ``commands``: the root group and the core commands (chat/ask/check/warm/doctor)
* ``config``: the ``config`` group and the effective-settings data behind it
* ``rag``: the ``rag`` group and the docs-toolchain provisioning helpers
* ``mcp``: the ``mcp`` group (run the MCP server from the CLI)
* ``_guards``: the fail-fast guard for a mistyped settings key
* ``repl``: the interactive REPL loop and prompt_toolkit session
* ``hitl``: the human-in-the-loop write-approval flow (ADR-08)
* ``session``: build, run, and probe the agent for a command
* ``ollama``: local Ollama model provisioning (presence checks, pull prompts)
* ``output``: the shared console and reply rendering
"""

from __future__ import annotations

from aiida_agents.cli.commands import cli

__all__ = ["cli"]
