"""Tests for cli/agent.py: settings resolution, agent build, reachability probe."""

from __future__ import annotations

import asyncio

import pytest
from click.testing import CliRunner

from aiida_agents._settings import _Provider
from aiida_agents.cli import cli
from aiida_agents.agents.planner import _SPECIALISTS, Specialist, Step
from aiida_agents.cli.agent import _AGENT_CHOICES, _resolve_model_settings


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


@pytest.mark.parametrize(
    "provider, message, expected",
    [
        pytest.param(
            "openrouter",
            "API key not set for the provider",
            "API key not set; set the provider's API key",
            id="key-not-set",
        ),
        pytest.param(
            "openrouter",
            "401 Unauthorized",
            "authentication failed; check the provider's API key",
            id="auth-failed",
        ),
        pytest.param(
            "ollama",
            "connection refused",
            "could not reach the endpoint; is the server running?",
            id="unreachable",
        ),
        pytest.param(
            "ollama",
            "model 'qwen3' not found (status 404)",
            "model not pulled (ollama pull qwen3)",
            id="ollama-not-pulled",
        ),
        pytest.param("openrouter", "something odd", None, id="unrecognised-is-none"),
    ],
)
def test_probe_failure_hint_routes_message(
    provider: _Provider, message: str, expected: str | None
) -> None:
    """A probe failure the user can fix is phrased as the fix, routed by its text.

    ``None`` for anything unrecognised, so the caller falls back to the
    exception's own words instead of inventing advice for a failure nobody
    classified.
    """
    from aiida_agents._settings import ModelSettings
    from aiida_agents.cli.agent import _probe_failure_hint

    hint = _probe_failure_hint(
        ModelSettings(provider=provider, model="qwen3"), RuntimeError(message)
    )

    assert hint == expected


@pytest.mark.parametrize(
    "provider, model_ok, expected",
    [
        pytest.param("ollama", True, "available", id="available"),
        pytest.param("ollama", False, "not_pulled", id="ollama-missing-is-fatal"),
        pytest.param("openrouter", False, "unlisted", id="cloud-missing-is-a-note"),
    ],
)
def test_model_availability_policy(
    provider: _Provider, model_ok: bool, expected: str
) -> None:
    """A model the endpoint doesn't advertise is "not pulled" for Ollama (its
    listing is authoritative) but only "unlisted" for a cloud endpoint (whose
    listing may be partial); an advertised model is available either way.
    """
    from aiida_agents.cli.agent import _model_availability, _Reachability

    availability = _model_availability(
        _Reachability("http://endpoint", 2, model_ok), provider
    )

    assert availability == expected


@pytest.mark.parametrize("name", _SPECIALISTS)
def test_naming_a_specialist_runs_that_specialist(name: Specialist) -> None:
    """``--agent <name>`` bypasses the planner and reaches the agent it names.

    Worth pinning per specialist rather than once: the narrowing this goes
    through used to collapse every name except ``execution`` to ``analysis``,
    so ``--agent codegen`` ran the Analysis agent and answered as if nothing
    were wrong. Parametrising over ``_SPECIALISTS`` means the next specialist
    added is covered without anyone remembering to add a case.
    """
    from aiida_agents._settings import ModelSettings
    from aiida_agents.cli.agent import _resolve_plan

    # The settings are never used on this path -- naming a specialist skips the
    # planner, so no model call happens -- but the signature wants them.
    settings = ModelSettings(provider="ollama", model="m")

    assert _resolve_plan(name, "a question", settings) == [Step(name, "")]


def test_every_agent_choice_is_either_a_specialist_or_auto() -> None:
    """``--agent`` cannot offer a name no specialist answers to.

    ``_build_agent`` rejects anything outside ``_SPECIALISTS``, so a choice
    that is neither a specialist nor ``auto`` would be an option the CLI
    advertises and then refuses.
    """
    assert set(_AGENT_CHOICES) - {"auto"} == set(_SPECIALISTS)
