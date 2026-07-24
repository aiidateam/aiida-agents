"""Discover and validate plugin providers, without ever breaking the agent.

A provider is third-party code that runs while the agent is being built, so every
step here is isolated: a provider that fails to import, a hook that raises, or a
hook that returns the wrong shape costs that plugin (or that one hook) and
nothing else. A broken plugin must never make the agent unusable.
"""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass
from importlib.metadata import entry_points

from aiida_agents.plugins.spec import (
    PLUGIN_ENTRY_POINT_GROUP,
    AgentTool,
    RagCorpus,
)

logger = logging.getLogger(__name__)

__all__ = ["LoadedPlugin", "discover_plugins"]

# A fragment is concatenated into the system prompt, so it costs tokens on every
# call for every user. Truncate loudly rather than let one plugin crowd out the
# agent's own instructions.
MAX_PROMPT_FRAGMENT_CHARS = 2000


@dataclass(frozen=True)
class LoadedPlugin:
    """What one provider successfully contributed."""

    name: str
    tools: tuple[AgentTool, ...] = ()
    corpora: tuple[RagCorpus, ...] = ()
    prompt_fragment: str | None = None


def _call_hook(plugin_name: str, provider: object, hook: str) -> t.Any:
    """Call an optional hook, returning None if absent or if it raises."""
    fn = getattr(provider, hook, None)
    if fn is None:
        return None
    if not callable(fn):
        logger.warning(
            "plugin %r: %s is not callable; ignoring that hook", plugin_name, hook
        )
        return None
    try:
        return fn()
    except Exception:
        # Deliberately broad: a third-party hook may raise anything, and the cost
        # must be that one hook, not the agent.
        logger.exception(
            "plugin %r: %s() raised; ignoring that hook", plugin_name, hook
        )
        return None


def _read_tools(plugin_name: str, provider: object) -> tuple[AgentTool, ...]:
    """Validated tools from a provider; malformed entries are dropped, not fatal."""
    raw = _call_hook(plugin_name, provider, "tools")
    if raw is None:
        return ()
    if not isinstance(raw, t.Sequence) or isinstance(raw, (str, bytes)):
        logger.warning(
            "plugin %r: tools() must return a sequence of AgentTool, got %s; ignoring",
            plugin_name,
            type(raw).__name__,
        )
        return ()
    tools: list[AgentTool] = []
    for item in raw:
        if not isinstance(item, AgentTool):
            logger.warning(
                "plugin %r: tools() yielded %s, expected AgentTool; ignoring it",
                plugin_name,
                type(item).__name__,
            )
            continue
        if not callable(item.fn):
            logger.warning(
                "plugin %r: tool %r has a non-callable fn; ignoring it",
                plugin_name,
                item.name,
            )
            continue
        tools.append(item)
    return tuple(tools)


def _read_corpora(plugin_name: str, provider: object) -> tuple[RagCorpus, ...]:
    """Validated corpora from a provider; a corpus with no source is dropped."""
    raw = _call_hook(plugin_name, provider, "rag_corpora")
    if raw is None:
        return ()
    if not isinstance(raw, t.Sequence) or isinstance(raw, (str, bytes)):
        logger.warning(
            "plugin %r: rag_corpora() must return a sequence of RagCorpus, got %s; ignoring",
            plugin_name,
            type(raw).__name__,
        )
        return ()
    corpora: list[RagCorpus] = []
    for item in raw:
        if not isinstance(item, RagCorpus):
            logger.warning(
                "plugin %r: rag_corpora() yielded %s, expected RagCorpus; ignoring it",
                plugin_name,
                type(item).__name__,
            )
            continue
        if not item.has_source:
            logger.warning(
                "plugin %r: corpus %r names no text_dir and no docs_repo; ignoring it",
                plugin_name,
                item.name,
            )
            continue
        corpora.append(item)
    return tuple(corpora)


def _read_fragment(plugin_name: str, provider: object) -> str | None:
    """Validated, budgeted prompt fragment from a provider."""
    raw = _call_hook(plugin_name, provider, "prompt_fragment")
    if raw is None:
        return None
    if not isinstance(raw, str):
        logger.warning(
            "plugin %r: prompt_fragment() must return a string, got %s; ignoring",
            plugin_name,
            type(raw).__name__,
        )
        return None
    fragment = raw.strip()
    if not fragment:
        return None
    if len(fragment) > MAX_PROMPT_FRAGMENT_CHARS:
        logger.warning(
            "plugin %r: prompt fragment is %d chars, truncating to %d",
            plugin_name,
            len(fragment),
            MAX_PROMPT_FRAGMENT_CHARS,
        )
        fragment = fragment[:MAX_PROMPT_FRAGMENT_CHARS]
    return fragment


def _drop_name_collisions(plugins: list[LoadedPlugin]) -> list[LoadedPlugin]:
    """Drop tools whose name is already taken, keeping the first plugin's.

    Tool names are a flat namespace from the model's point of view, so two
    plugins claiming ``suggest_pseudos`` is a bug. Resolving it silently (last
    one wins) would make behaviour depend on entry-point order, so the collision
    is logged and the later tool dropped.
    """
    seen: dict[str, str] = {}
    resolved: list[LoadedPlugin] = []
    for plugin in plugins:
        kept: list[AgentTool] = []
        for tool in plugin.tools:
            owner = seen.get(tool.name)
            if owner is not None:
                logger.warning(
                    "plugin %r: tool name %r is already registered by plugin %r; "
                    "ignoring the duplicate",
                    plugin.name,
                    tool.name,
                    owner,
                )
                continue
            seen[tool.name] = plugin.name
            kept.append(tool)
        resolved.append(
            LoadedPlugin(
                name=plugin.name,
                tools=tuple(kept),
                corpora=plugin.corpora,
                prompt_fragment=plugin.prompt_fragment,
            )
        )
    return resolved


def discover_plugins() -> tuple[LoadedPlugin, ...]:
    """Load every provider registered under ``aiida_agents.plugins``.

    Providers are loaded in entry-point-name order, so what the agent ends up
    with never depends on installation order. Each provider's import and each of
    its hooks are isolated: a failure is logged and skipped.
    """
    try:
        registered = sorted(
            entry_points(group=PLUGIN_ENTRY_POINT_GROUP), key=lambda ep: ep.name
        )
    except Exception:
        logger.exception("could not read the %r entry points", PLUGIN_ENTRY_POINT_GROUP)
        return ()

    plugins: list[LoadedPlugin] = []
    for ep in registered:
        try:
            provider = ep.load()
        except Exception:
            # A plugin whose provider fails to import (bad dependency, syntax
            # error, ...) must not take the agent down with it.
            logger.exception("plugin %r: provider failed to load; skipping", ep.name)
            continue
        plugins.append(
            LoadedPlugin(
                name=str(getattr(provider, "name", ep.name)),
                tools=_read_tools(ep.name, provider),
                corpora=_read_corpora(ep.name, provider),
                prompt_fragment=_read_fragment(ep.name, provider),
            )
        )

    resolved = _drop_name_collisions(plugins)
    if resolved:
        logger.info(
            "discovered %d agent plugin(s): %s",
            len(resolved),
            ", ".join(p.name for p in resolved),
        )
    return tuple(resolved)
