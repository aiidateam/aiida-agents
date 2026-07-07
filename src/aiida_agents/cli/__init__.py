"""Command-line interface for aiida-agents.

The Click entry point (``cli``) plus its command surface, split by concern:

* ``commands``:the Click group and subcommands
* ``repl``:the interactive REPL loop and prompt_toolkit session
* ``hitl``:the human-in-the-loop write-approval flow (ADR-08)
* ``session``:build, run, and probe the agent for a command
* ``ollama``:local Ollama model provisioning (presence checks, pull prompts)
* ``config``:the ``config show`` data
* ``output``:the shared console and reply rendering
"""

from __future__ import annotations

from aiida_agents.cli.commands import cli

__all__ = ["cli"]
