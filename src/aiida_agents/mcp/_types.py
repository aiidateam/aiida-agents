"""Shared type aliases for the MCP layer (tools, resources, ...)."""

from __future__ import annotations

import typing as t

from pydantic import Field

# A node identifier: a pk (int) or a uuid (str).
Identifier = t.Annotated[int | str, Field(description="Node pk (int) or uuid (str)")]
