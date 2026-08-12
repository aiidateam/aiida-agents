"""Tests for cli/agent.py: settings resolution, agent build, reachability probe."""

from __future__ import annotations

import asyncio

import pytest
from click.testing import CliRunner

from aiida_agents._settings import _Provider
from aiida_agents.cli import cli
from aiida_agents.cli.agent import _resolve_model_settings


@pytest.mark.parametrize(
    "flag_model, env_model, expected",
    [
        pytest.param("flag-model", "env-model", "flag-model", id="flag-beats-env"),
        pytest.param(None, "env-model", "env-model", id="env-when-no-flag"),
        pytest.param(None, None, "qwen3.5:2b", id="default-when-neither"),
    ],
)
def test_resolve_model_settings_precedence(
    monkeypatch: pytest.MonkeyPatch,
    flag_model: str | None,
    env_model: str | None,
    expected: str,
) -> None:
    """A ``--model`` flag beats ``AIIDA_AGENTS_MODEL``, which beats the default."""
    if env_model is None:
        monkeypatch.delenv("AIIDA_AGENTS_MODEL", raising=False)
    else:
        monkeypatch.setenv("AIIDA_AGENTS_MODEL", env_model)
    assert _resolve_model_settings(None, flag_model).model == expected


def test_build_agent_reports_missing_api_key_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing cloud API key surfaces as a clean CLI error, not a traceback.

    Uses the real ``load_profile`` (the autouse test profile), mocking only the
    genuinely external step: ``get_agent``, made to raise as pydantic-ai would
    for a missing key.
    """
    from pydantic_ai.exceptions import UserError

    def _no_key(**kwargs: object) -> object:
        raise UserError("Set the `OPENROUTER_API_KEY` environment variable")

    monkeypatch.setattr("aiida_agents.agents.get_agent", _no_key)
    result = CliRunner().invoke(
        cli, ["--provider", "openrouter", "--model", "openrouter/free", "ask", "hi"]
    )
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output
    assert not isinstance(result.exception, UserError)


class _FakeModelsPage:
    def __init__(self, ids: list[str]) -> None:
        self.data = [type("M", (), {"id": i})() for i in ids]


class _FakeAsyncClient:
    def __init__(
        self, ids: list[str], *, hang: bool = False, base_url: str = "http://fake"
    ) -> None:
        self._ids, self._hang = ids, hang
        self.models = self
        self.base_url = base_url

    async def list(self) -> _FakeModelsPage:
        if self._hang:
            await asyncio.sleep(1)
        return _FakeModelsPage(self._ids)


def test_list_model_ids_returns_advertised_ids() -> None:
    """The listing helper collects the endpoint's model ids."""
    from aiida_agents.cli.agent import _list_model_ids

    assert asyncio.run(_list_model_ids(_FakeAsyncClient(["a", "b"]))) == {"a", "b"}


def test_list_model_ids_times_out_as_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow endpoint becomes a connection error (never an indefinite hang)."""
    from aiida_agents.cli import agent

    monkeypatch.setattr(agent, "_REACHABILITY_TIMEOUT", 0.01)
    with pytest.raises(ConnectionError, match="could not connect"):
        asyncio.run(agent._list_model_ids(_FakeAsyncClient([], hang=True)))


@pytest.mark.parametrize(
    "provider, model, ids, expected_ok",
    [
        pytest.param(
            "ollama",
            "qwen3",
            ["qwen3:latest", "other:1b"],
            True,
            id="ollama-untagged-matches-latest",
        ),
        pytest.param(
            "ollama", "qwen3:9b", ["qwen3:9b"], True, id="ollama-tagged-exact"
        ),
        pytest.param(
            "openrouter", "vendor/model", ["a/b"], False, id="cloud-not-listed"
        ),
    ],
)
def test_probe_reachable_reports_endpoint_and_model_availability(
    monkeypatch: pytest.MonkeyPatch,
    provider: _Provider,
    model: str,
    ids: list[str],
    expected_ok: bool,
) -> None:
    """The no-generation probe reports the endpoint and model count, and resolves
    availability: an untagged Ollama name matches the ':latest' the listing shows,
    a tagged one matches exactly, and a cloud model absent from the listing is not
    marked available.
    """
    from aiida_agents._settings import ModelSettings
    from aiida_agents.cli import agent

    client = _FakeAsyncClient(ids, base_url="http://endpoint")

    class _FakeModel:
        def __init__(self) -> None:
            self.client = client

    monkeypatch.setattr(
        "aiida_agents.agents._models.get_model", lambda model_settings: _FakeModel()
    )

    reach = agent._probe_reachable(ModelSettings(provider=provider, model=model))

    assert reach.endpoint == "http://endpoint"
    assert reach.n_models == len(ids)
    assert reach.model_ok is expected_ok
