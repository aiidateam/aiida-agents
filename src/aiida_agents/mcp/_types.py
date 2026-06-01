"""Shared type aliases for the MCP layer (tools, resources, ...)."""

from __future__ import annotations

import typing as t

from pydantic import Field

__all__ = ["Identifier"]

# A node identifier: a pk or a uuid, both as a plain string. Using ``str``
# (rather than ``int | str``) means the MCP Inspector sends the value as-is
# without requiring JSON quotes; the loader coerces a purely numeric
# identifier back to an integer pk.
Identifier = t.Annotated[
    str, Field(description="Node pk or uuid (e.g. '42' or '0cef…')")
]
