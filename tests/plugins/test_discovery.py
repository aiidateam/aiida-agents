"""Tests for plugin discovery: what a provider contributes, and what it can't break.

A provider is third-party code that runs while the agent is built, so the
guarantee under test is containment: a plugin that fails to import, raises from a
hook, or returns the wrong shape costs that plugin (or that one hook) and nothing
else. ``get_agent()`` must survive all of it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from aiida_agents.plugins import AgentTool, LoadedPlugin, RagCorpus, discover_plugins
from aiida_agents.plugins.discovery import MAX_PROMPT_FRAGMENT_CHARS


def _tool() -> str:
    """A plugin read tool."""
    return "ok"


def _writer() -> str:
    """A plugin write tool."""
    return "wrote"


class _EntryPoint:
    """Stands in for an importlib entry point, whose ``load`` we control."""

    def __init__(self, name: str, provider: Any, boom: bool = False) -> None:
        self.name = name
        self._provider = provider
        self._boom = boom

    def load(self) -> Any:
        if self._boom:
            raise ImportError(f"{self.name} is broken")
        return self._provider


# Registers the given fake entry points as the installed plugin providers.
_Register = Callable[..., None]


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> _Register:
    """Register fake providers under the plugin entry-point group."""

    def _register(*eps: _EntryPoint) -> None:
        monkeypatch.setattr(
            "aiida_agents.plugins.discovery.entry_points",
            lambda group: list(eps),
        )

    return _register


class TestContributions:
    def test_provider_contributes_tools_corpora_and_fragment(
        self, registered: _Register
    ) -> None:
        """Each hook's output reaches the loaded plugin."""

        class Provider:
            name = "qe"

            def tools(self) -> Sequence[AgentTool]:
                return [AgentTool(_tool), AgentTool(_writer, writes=True)]

            def rag_corpora(self) -> Sequence[RagCorpus]:
                return [RagCorpus(name="qe", version="4.0", text_dir=Path("/docs"))]

            def prompt_fragment(self) -> str:
                return "Use ecutrho = 8 x ecutwfc."

        registered(_EntryPoint("qe", Provider()))
        (plugin,) = discover_plugins()

        assert plugin.name == "qe"
        assert [tool.name for tool in plugin.tools] == ["_tool", "_writer"]
        assert [tool.writes for tool in plugin.tools] == [False, True]
        assert plugin.corpora[0].name == "qe"
        assert plugin.prompt_fragment == "Use ecutrho = 8 x ecutwfc."

    def test_hooks_are_optional(self, registered: _Register) -> None:
        """A provider implementing nothing is still a valid plugin."""

        class Bare:
            name = "bare"

        registered(_EntryPoint("bare", Bare()))
        (plugin,) = discover_plugins()

        assert plugin == LoadedPlugin(name="bare")  # all contributions empty

    def test_providers_load_in_entry_point_name_order(
        self, registered: _Register
    ) -> None:
        """Order is deterministic, so behaviour can't depend on install order."""

        class Provider:
            def __init__(self, name: str) -> None:
                self.name = name

        registered(
            _EntryPoint("zeta", Provider("zeta")),
            _EntryPoint("alpha", Provider("alpha")),
        )
        assert [p.name for p in discover_plugins()] == ["alpha", "zeta"]


class TestContainment:
    """A broken plugin is logged and skipped; it never breaks discovery."""

    def test_provider_that_fails_to_import_is_skipped(
        self, registered: _Register
    ) -> None:
        class Good:
            name = "good"

            def tools(self) -> Sequence[AgentTool]:
                return [AgentTool(_tool)]

        registered(
            _EntryPoint("broken", None, boom=True),
            _EntryPoint("good", Good()),
        )
        plugins = discover_plugins()

        assert [p.name for p in plugins] == ["good"]  # the good one still loads

    def test_raising_hook_costs_only_that_hook(self, registered: _Register) -> None:
        """A hook that raises is dropped; the provider's other hooks survive."""

        class Partly:
            name = "partly"

            def tools(self) -> Sequence[AgentTool]:
                raise RuntimeError("boom")

            def prompt_fragment(self) -> str:
                return "still here"

        registered(_EntryPoint("partly", Partly()))
        (plugin,) = discover_plugins()

        assert plugin.tools == ()
        assert plugin.prompt_fragment == "still here"

    @pytest.mark.parametrize(
        "hook, returned",
        [
            pytest.param("tools", "not-a-sequence", id="tools-wrong-type"),
            pytest.param("tools", [object()], id="tools-not-AgentTool"),
            pytest.param("rag_corpora", [object()], id="corpora-not-RagCorpus"),
            pytest.param("prompt_fragment", 42, id="fragment-not-str"),
        ],
    )
    def test_malformed_hook_output_is_dropped(
        self, registered: _Register, hook: str, returned: Any
    ) -> None:
        """Wrong shapes are rejected at the boundary, not passed to the agent."""
        provider = type("Provider", (), {"name": "p", hook: lambda self: returned})()
        registered(_EntryPoint("p", provider))
        (plugin,) = discover_plugins()

        assert plugin.tools == ()
        assert plugin.corpora == ()
        assert plugin.prompt_fragment is None

    def test_corpus_without_a_source_is_dropped(self, registered: _Register) -> None:
        """A corpus naming neither text_dir nor docs_repo can't be built."""

        class Provider:
            name = "p"

            def rag_corpora(self) -> Sequence[RagCorpus]:
                return [RagCorpus(name="p", version="1.0")]

        registered(_EntryPoint("p", Provider()))
        (plugin,) = discover_plugins()

        assert plugin.corpora == ()


class TestPolicies:
    def test_duplicate_tool_name_keeps_the_first_and_warns(
        self, registered: _Register, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Tool names are a flat namespace, so a collision is a bug, not a
        last-one-wins race. The first plugin in name order keeps the name.
        """

        class Provider:
            def __init__(self, name: str) -> None:
                self.name = name

            def tools(self) -> Sequence[AgentTool]:
                return [AgentTool(_tool)]

        registered(
            _EntryPoint("alpha", Provider("alpha")),
            _EntryPoint("beta", Provider("beta")),
        )
        with caplog.at_level(logging.WARNING):
            alpha, beta = discover_plugins()

        assert [tool.name for tool in alpha.tools] == ["_tool"]
        assert beta.tools == ()  # the duplicate is dropped, not silently preferred
        assert "already registered" in caplog.text

    def test_oversized_prompt_fragment_is_truncated_and_warns(
        self, registered: _Register, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A fragment costs tokens on every call, so it is budgeted loudly."""

        class Provider:
            name = "p"

            def prompt_fragment(self) -> str:
                return "x" * (MAX_PROMPT_FRAGMENT_CHARS + 500)

        registered(_EntryPoint("p", Provider()))
        with caplog.at_level(logging.WARNING):
            (plugin,) = discover_plugins()

        assert plugin.prompt_fragment is not None
        assert len(plugin.prompt_fragment) == MAX_PROMPT_FRAGMENT_CHARS
        assert "truncating" in caplog.text

    def test_no_plugins_installed_is_the_normal_case(
        self, registered: _Register
    ) -> None:
        registered()
        assert discover_plugins() == ()
