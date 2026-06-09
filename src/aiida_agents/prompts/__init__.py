"""Prompt templates for the AiiDA agents."""

from __future__ import annotations

from importlib.resources import files

SYSTEM_PROMPT = (
    files(__package__).joinpath("system_prompt.md").read_text(encoding="utf-8").strip()
)

__all__ = ["SYSTEM_PROMPT"]
