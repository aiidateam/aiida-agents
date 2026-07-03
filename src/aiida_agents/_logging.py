from __future__ import annotations

import logging
from pydantic_ai.messages import (
    ModelMessage,
    ToolReturnPart,
    ToolCallPart,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

logger = logging.getLogger(__name__)
file_logger = logging.getLogger("aiida_agents.file_debug")
file_logger.propagate = True
file_logger.setLevel(logging.DEBUG)

responses_logger = logging.getLogger("aiida_agents.responses_debug")
responses_logger.propagate = True
responses_logger.setLevel(logging.DEBUG)

console = Console()


def _log_tool_calls_debug(messages: list[ModelMessage], console: Console) -> None:
    """Log tool calls and returns at DEBUG level with rich formatting."""
    # Check effective log level of the root logger first.
    # If not DEBUG, do not print anything to the console.
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        return

    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                console.print()
                console.print(
                    f"[bold cyan]→ TOOL CALLED:[/bold cyan] [yellow]{part.tool_name}[/yellow]"
                )
                console.print(f"  [dim]ID:[/dim] {part.tool_call_id}")
                console.print(f"  [dim]Args:[/dim] {part.args}")
                console.print()
                # Log to the file logger so it goes to log files/handlers
                file_logger.debug(
                    "→ TOOL CALLED: %s with args %s (id: %s)",
                    part.tool_name,
                    part.args,
                    part.tool_call_id,
                )
            elif isinstance(part, ToolReturnPart):
                console.print()
                console.print(
                    f"[bold green]← TOOL RETURNED:[/bold green] [yellow]{part.tool_name}[/yellow]"
                )
                console.print(f"  [dim]ID:[/dim] {part.tool_call_id}")
                console.print(
                    Panel(
                        str(part.content),
                        title=f"Tool Return: {part.tool_name}",
                        border_style="green",
                    )
                )
                console.print()
                # Log to the file logger so it goes to log files/handlers
                file_logger.debug(
                    "← TOOL RETURNED: %s -> %s", part.tool_name, str(part.content)
                )


def _print_agent(text: str) -> None:  # pragma: no cover
    """Print an agent reply, blank-line padded so it stands clear of the ``You:``
    turns on either side: a highlighted label, then the body as markdown so
    tables and formatting render.
    """
    console.print()
    console.print("Agent:", style="bold green")
    console.print(Markdown(text))
    console.print()
    responses_logger.debug("Agent Response: %s", text)


def _suppress_noisy_loggers() -> None:
    """Reduce verbosity of third-party debug logs."""
    for logger_name in ["asyncio", "openai", "httpcore", "chromadb", "markdown_it"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
