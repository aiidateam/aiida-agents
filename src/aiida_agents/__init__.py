"""aiida-agents: natural-language exploration of an AiiDA database.

The typed settings groups are re-exported here so callers can build and
inject configuration (e.g. ``get_model(model_settings=ModelSettings(...))``)
without reaching into the private ``_settings`` module. Only the settings
types live at the package root: they depend on nothing beyond pydantic, so
``import aiida_agents`` stays cheap. The factories (``get_model``,
``get_embedding_function``) are deliberately not re-exported here, because
importing them pulls the full agent / RAG stack (pydantic-ai, chromadb, the
optional sentence-transformers/torch); import them from their own modules when
you need them.
"""

from __future__ import annotations

from aiida_agents._settings import (
    AgentSettings,
    LoggingSettings,
    ModelSettings,
    OllamaSettings,
    RagSettings,
    ReplSettings,
    ServerSettings,
)

__all__ = [
    "AgentSettings",
    "LoggingSettings",
    "ModelSettings",
    "OllamaSettings",
    "RagSettings",
    "ReplSettings",
    "ServerSettings",
]
