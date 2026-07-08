"""Unit tests for cli helpers that need no live model or database."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from click.testing import CliRunner
from prompt_toolkit.keys import Keys
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from rich.console import Console

from aiida_agents._logging import TRACE_LOGGER_NAMES
from aiida_agents.cli import cli
from aiida_agents.cli.session import _resolve_model_settings
from aiida_agents.cli.ollama import _ollama_lists_model, _ollama_model_names
from aiida_agents.cli.config import _config_rows
from aiida_agents.cli.output import _format_duration, _log_tool_calls_debug
from aiida_agents.cli.repl import (
    _cap_history,
    _history_file,
    _key_bindings,
    _prompt_continuation,
)


def _turn(i: int) -> list[ModelMessage]:
    """One user turn with a tool round: a call/return pair lives inside it."""
    return [
        ModelRequest(parts=[UserPromptPart(content=f"q{i}")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="t", args={}, tool_call_id=f"c{i}")]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="t", content="r", tool_call_id=f"c{i}")]
        ),
        ModelResponse(parts=[TextPart(content=f"a{i}")]),
    ]


def _user_turns(messages: list[ModelMessage]) -> int:
    return sum(
        isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in m.parts)
        for m in messages
    )


@pytest.mark.parametrize("max_turns", [1, 2, 3])
def test_cap_history_keeps_last_n_whole_turns(max_turns: int) -> None:
    """The window holds the last ``max_turns`` turns and starts on a boundary."""
    messages = [m for i in range(3) for m in _turn(i)]
    capped = _cap_history(messages, max_turns)

    assert _user_turns(capped) == min(max_turns, 3)
    # The window starts on a user-turn boundary, never mid tool round.
    assert isinstance(capped[0], ModelRequest)
    assert any(isinstance(p, UserPromptPart) for p in capped[0].parts)


def test_cap_history_never_orphans_a_tool_return() -> None:
    """Every tool return kept in the window still has its matching tool call.

    Regression for the count-based ``history[-N:]`` slice, which could start
    mid-pair and leave an unpaired ``tool_use``/``tool_result`` the provider
    rejects.
    """
    messages = [m for i in range(4) for m in _turn(i)]
    capped = _cap_history(messages, max_turns=2)

    call_ids = {
        p.tool_call_id for m in capped for p in m.parts if isinstance(p, ToolCallPart)
    }
    return_ids = {
        p.tool_call_id for m in capped for p in m.parts if isinstance(p, ToolReturnPart)
    }
    assert return_ids and return_ids <= call_ids


def test_cap_history_returns_input_when_within_budget() -> None:
    """Under the turn budget, the history is returned untouched."""
    messages = [m for i in range(2) for m in _turn(i)]
    assert _cap_history(messages, max_turns=5) is messages


@pytest.mark.parametrize(
    "xdg, expected",
    [
        pytest.param(
            "/custom/data",
            Path("/custom/data/aiida-agents/repl-history"),
            id="xdg-set",
        ),
        pytest.param(
            None,
            Path.home() / ".local" / "share" / "aiida-agents" / "repl-history",
            id="xdg-unset",
        ),
    ],
)
def test_history_file_location(
    monkeypatch: pytest.MonkeyPatch, xdg: str | None, expected: Path
) -> None:
    """History lives under ``$XDG_DATA_HOME``, falling back to ``~/.local/share``."""
    if xdg is None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
    assert _history_file() == expected


def test_key_bindings_flip_enter_and_newline() -> None:
    """Exactly Enter and Esc+Enter are bound (the deliberate flip of
    prompt_toolkit's multiline default; see ``_key_bindings``). ``Keys.Enter``
    is the ``ControlM`` alias prompt_toolkit stores for a bare ``"enter"``.
    """
    bound = {binding.keys for binding in _key_bindings().bindings}
    assert bound == {(Keys.Enter,), (Keys.Escape, Keys.Enter)}


def test_prompt_continuation_aligns_under_prompt() -> None:
    """The continuation fills the prompt width so wrapped lines line up under it."""
    assert _prompt_continuation(len("You: "), 0, 0) == ".... "


def _tool_round() -> list[ModelMessage]:
    """One tool call/return pair."""
    return [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="test_tool", args={"x": 42}, tool_call_id="call-1"
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="test_tool",
                    content={"status": "ok"},
                    tool_call_id="call-1",
                )
            ]
        ),
    ]


def test_log_tool_calls_debug(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Console rendering is gated on DEBUG; the trace log always records.

    Regression: the file trace must not depend on the console log level,
    otherwise a log file written at INFO holds agent responses but no tool
    calls.
    """
    console = Console(color_system=None)
    root_logger = logging.getLogger()
    initial_level = root_logger.level

    try:
        with caplog.at_level(logging.DEBUG, logger=TRACE_LOGGER_NAMES[0]):
            # At INFO, nothing goes to the console, but the trace logger still
            # records both the call and the return.
            root_logger.setLevel(logging.INFO)
            _log_tool_calls_debug(_tool_round(), console)
            assert capsys.readouterr().out == ""
            assert "→ TOOL CALLED: test_tool" in caplog.text
            assert "← TOOL RETURNED: test_tool" in caplog.text

            # At DEBUG, the formatted tool calls and returns are printed too.
            root_logger.setLevel(logging.DEBUG)
            _log_tool_calls_debug(_tool_round(), console)
            captured = capsys.readouterr()
            assert "→ TOOL CALLED: test_tool" in captured.out
            assert "ID: call-1" in captured.out
            assert "Args: {'x': 42}" in captured.out
            assert "← TOOL RETURNED: test_tool" in captured.out
            assert "{'status': 'ok'}" in captured.out
    finally:
        root_logger.setLevel(initial_level)


@pytest.mark.parametrize(
    "seconds, expected",
    [
        pytest.param(0.0, "0.0s", id="zero"),
        pytest.param(12.34, "12.3s", id="sub-minute"),
        pytest.param(59.9, "59.9s", id="just-under-a-minute"),
        pytest.param(60.0, "1m 0s", id="exactly-a-minute"),
        pytest.param(132.0, "2m 12s", id="minutes"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    """Sub-minute times read as seconds; a minute or more as ``Xm Ys``."""
    assert _format_duration(seconds) == expected


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


def test_cli_exposes_expected_commands() -> None:
    """The top-level group lists every subcommand we ship."""
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("ask", "chat", "check", "config", "doctor", "mcp", "rag", "warm"):
        assert command in result.output


def test_dash_h_is_a_help_alias() -> None:
    """`-h` works as a help alias at the group and on subcommands."""
    assert CliRunner().invoke(cli, ["-h"]).exit_code == 0
    assert CliRunner().invoke(cli, ["config", "-h"]).exit_code == 0
    assert CliRunner().invoke(cli, ["ask", "-h"]).exit_code == 0
    assert CliRunner().invoke(cli, ["rag", "build", "-h"]).exit_code == 0


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError("sphinx build failed"), id="sphinx"),
        pytest.param(OSError("embedder unreachable"), id="embedder"),
    ],
)
def test_rag_build_reports_index_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """A build failure (sphinx, or the embedder over the network) surfaces as a
    clean CLI error, not a traceback."""
    monkeypatch.setattr("aiida_agents.cli.commands._module_missing", lambda name: False)
    # No real Ollama pull during the test if the embed model is absent.
    monkeypatch.setattr(
        "aiida_agents.cli.commands._prompt_pull_ollama_model", lambda model: None
    )

    def _boom(force: bool, progress: object = None) -> None:
        raise exc

    monkeypatch.setattr("aiida_agents.rag.index_docs", _boom)
    result = CliRunner().invoke(cli, ["rag", "build"])
    assert result.exit_code == 1
    assert "RAG build failed" in result.output
    # Converted, not leaked as an uncaught traceback.
    assert not isinstance(result.exception, (RuntimeError, OSError))


def test_rag_build_declining_toolchain_install_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sphinx plus a declined install gives a clean, actionable error."""
    monkeypatch.setattr(
        "aiida_agents.cli.commands._module_missing", lambda name: name == "sphinx"
    )
    result = CliRunner().invoke(cli, ["rag", "build"], input="n\n")
    assert result.exit_code == 1
    assert "not installed" in result.output
    assert "aiida-core[docs]" in result.output


@pytest.mark.parametrize(
    "outcome_name, expected, unexpected",
    [
        pytest.param(
            "ALREADY_PRESENT", "already built", "index ready", id="already-present"
        ),
        pytest.param(
            "EMPTY_CORPUS", "left unchanged", "index ready", id="empty-corpus"
        ),
        pytest.param("BUILT", "RAG index ready", "already built", id="built"),
    ],
)
def test_rag_build_reports_outcome(
    monkeypatch: pytest.MonkeyPatch, outcome_name: str, expected: str, unexpected: str
) -> None:
    """The final line reflects what ``index_docs`` actually did, so re-running on
    an up-to-date index reads as a no-op, not a fresh build."""
    from aiida_agents.rag import IndexOutcome

    monkeypatch.setattr("aiida_agents.cli.commands._module_missing", lambda name: False)
    monkeypatch.setattr(
        "aiida_agents.cli.commands._prompt_pull_ollama_model", lambda model: None
    )
    outcome = getattr(IndexOutcome, outcome_name)
    monkeypatch.setattr(
        "aiida_agents.rag.index_docs", lambda force, progress=None: outcome
    )
    result = CliRunner().invoke(cli, ["rag", "build"])
    assert result.exit_code == 0
    assert expected in result.output
    assert unexpected not in result.output


def test_config_show_command_runs() -> None:
    """``config show`` renders without touching AiiDA or a model."""
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0


def test_build_agent_reports_missing_api_key_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing cloud API key surfaces as a clean CLI error, not a traceback."""
    from pydantic_ai.exceptions import UserError

    monkeypatch.setattr("aiida.load_profile", lambda profile=None: None)

    def _no_key(**kwargs: object) -> object:
        raise UserError("Set the `OPENROUTER_API_KEY` environment variable")

    monkeypatch.setattr("aiida_agents.agents.get_agent", _no_key)
    result = CliRunner().invoke(
        cli, ["--provider", "openrouter", "--model", "openrouter/free", "ask", "hi"]
    )
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output
    assert not isinstance(result.exception, UserError)


def test_config_rows_report_value_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each row carries its group, resolved value, env var, and where it came from."""
    monkeypatch.setenv("AIIDA_AGENTS_MODEL", "some-model")
    monkeypatch.delenv("AIIDA_AGENTS_PROVIDER", raising=False)

    from_env = {row[1]: row for row in _config_rows(provider=None, model=None)}
    assert from_env["model"] == (
        "model",
        "model",
        "some-model",
        "AIIDA_AGENTS_MODEL",
        "env",
    )
    assert from_env["provider"][4] == "default"

    from_flag = {row[1]: row for row in _config_rows(provider="openai", model=None)}
    assert from_flag["provider"][2] == "openai"
    assert from_flag["provider"][4] == "flag"


def test_config_rows_disambiguate_base_url_by_group() -> None:
    """The two ``base_url`` fields are told apart by group and env var, so the
    rendered table never shows one label for two different settings.
    """
    base_urls = {
        row[0]: row[3] for row in _config_rows(None, None) if row[1] == "base_url"
    }
    assert base_urls == {
        "model": "AIIDA_AGENTS_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
    }


def test_config_rows_mask_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys are reported as set/unset, never echoed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret")
    rows = {row[1]: row for row in _config_rows(None, None)}
    assert rows["openrouter_api_key"][2] == "set"
    assert "sk-super-secret" not in str(rows)


def test_ollama_model_names_parses_list_output() -> None:
    """Names come from the first column, with the header row skipped."""
    out = (
        "NAME                                            ID    SIZE\n"
        "qwen3.5:9b                                      abc   6.6 GB\n"
        "MichelRosselli/apertus:8b-instruct-2509-q4_k_m  def   5.1 GB\n"
    )
    assert _ollama_model_names(out) == {
        "qwen3.5:9b",
        "MichelRosselli/apertus:8b-instruct-2509-q4_k_m",
    }


def test_ollama_model_names_empty_when_no_models() -> None:
    """A header-only listing yields no names (nothing pulled)."""
    assert _ollama_model_names("NAME  ID  SIZE  MODIFIED\n") == set()


@pytest.mark.parametrize(
    "model, present",
    [
        pytest.param("mxbai-embed-large", True, id="untagged-matches-latest"),
        pytest.param("qwen3.5:9b", True, id="tagged-exact"),
        pytest.param("not-there", False, id="absent"),
    ],
)
def test_ollama_lists_model_normalizes_latest_tag(model: str, present: bool) -> None:
    """An untagged configured name matches the ':latest' `ollama list` shows."""
    out = (
        "NAME                        ID   SIZE\n"
        "mxbai-embed-large:latest    abc  669 MB\n"
        "qwen3.5:9b                  def  6.6 GB\n"
    )
    assert _ollama_lists_model(model, out) is present


def test_check_verifies_reachability_without_generating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`check` must probe reachability, never warm/generate the model."""
    calls: list[str] = []
    monkeypatch.setattr(
        "aiida_agents.cli.commands._check_reachable",
        lambda settings: calls.append("reachable"),
    )
    monkeypatch.setattr(
        "aiida_agents.cli.commands._probe_model",
        lambda settings: calls.append("generate"),
    )
    result = CliRunner().invoke(cli, ["check"])
    assert result.exit_code == 0
    assert calls == ["reachable"]


def test_warm_fires_the_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """`warm` runs the generation probe that loads the model."""
    calls: list[str] = []
    monkeypatch.setattr(
        "aiida_agents.cli.commands._probe_model",
        lambda settings: calls.append("generate"),
    )
    result = CliRunner().invoke(cli, ["warm"])
    assert result.exit_code == 0
    assert calls == ["generate"]


class _FakeModelsPage:
    def __init__(self, ids: list[str]) -> None:
        self.data = [type("M", (), {"id": i})() for i in ids]


class _FakeAsyncClient:
    def __init__(self, ids: list[str], *, hang: bool = False) -> None:
        self._ids, self._hang = ids, hang
        self.models = self

    async def list(self) -> _FakeModelsPage:
        if self._hang:
            await asyncio.sleep(1)
        return _FakeModelsPage(self._ids)


def test_list_model_ids_returns_advertised_ids() -> None:
    """The listing helper collects the endpoint's model ids."""
    from aiida_agents.cli.session import _list_model_ids

    assert asyncio.run(_list_model_ids(_FakeAsyncClient(["a", "b"]))) == {"a", "b"}


def test_list_model_ids_times_out_as_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow endpoint becomes a connection error (never an indefinite hang)."""
    from aiida_agents.cli import session

    monkeypatch.setattr(session, "_REACHABILITY_TIMEOUT", 0.01)
    with pytest.raises(ConnectionError, match="could not connect"):
        asyncio.run(session._list_model_ids(_FakeAsyncClient([], hang=True)))


def test_version_option_prints_version() -> None:
    """`--version` reports the installed package version and exits cleanly."""
    from importlib.metadata import version

    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert version("aiida-agents") in result.output


# --- fail-fast on a mistyped setting key -----------------------------------


def _one_typo() -> list[tuple[str, str | None]]:
    return [("AIIDA_AGENS_PROVIDER", "AIIDA_AGENTS_PROVIDER")]


def test_action_command_fails_fast_on_mistyped_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that acts on config exits 2, before doing work, on a typo'd key."""
    monkeypatch.setattr(
        "aiida_agents.cli.commands.find_unrecognized_settings", _one_typo
    )
    result = CliRunner().invoke(cli, ["rag", "status"])
    assert result.exit_code == 2
    assert "AIIDA_AGENS_PROVIDER" in result.output
    assert "did you mean AIIDA_AGENTS_PROVIDER" in result.output


def test_config_show_flags_but_still_runs_on_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`config show` reports the typo yet renders the table (exit 0), to aid debugging."""
    monkeypatch.setattr(
        "aiida_agents.cli.commands.find_unrecognized_settings", _one_typo
    )
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "AIIDA_AGENS_PROVIDER" in result.output


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["check", "--help"], id="command-help"),
        pytest.param(["rag", "--help"], id="group-help"),
        pytest.param(["rag", "status", "-h"], id="subcommand-h"),
    ],
)
def test_help_bypasses_the_settings_check(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    """--help / -h resolve before the guard runs, so they work despite a typo."""
    monkeypatch.setattr(
        "aiida_agents.cli.commands.find_unrecognized_settings", _one_typo
    )
    assert CliRunner().invoke(cli, args).exit_code == 0


# --- config init / path -----------------------------------------------------


def test_env_template_covers_exactly_the_recognised_settings() -> None:
    """The template scaffolds every recognised setting and nothing else: it can't
    teach a typo, and no setting is silently missing.
    """
    from aiida_agents._settings import _known_env_var_names
    from aiida_agents.cli.config import _env_template

    keys = {
        key
        for line in _env_template().splitlines()
        if line.startswith("# ") and "=" in line
        # env-var lines are `# NAME=...`; description lines (`#   ... x=y`) and
        # section headers are excluded because a key token has no spaces.
        for key in [line[2:].split("=", 1)[0]]
        if " " not in key
    }
    assert keys == _known_env_var_names()


def test_config_rows_cover_all_recognised_settings() -> None:
    """`config show` covers exactly the recognised settings, so it stays in step
    with `config init` and the typo-detector (one source, no drift).
    """
    from aiida_agents._settings import _known_env_var_names

    env_vars = {row[3] for row in _config_rows(None, None)}
    assert env_vars == _known_env_var_names()


def test_config_init_writes_template_and_refuses_overwrite(tmp_path: Path) -> None:
    """`config init` writes a template, then declines to clobber it without --force."""
    first = CliRunner().invoke(cli, ["config", "init"])
    assert first.exit_code == 0
    written = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "AIIDA_AGENTS_PROVIDER=ollama" in written

    second = CliRunner().invoke(cli, ["config", "init"])
    assert second.exit_code == 1
    assert "already exists" in second.output

    forced = CliRunner().invoke(cli, ["config", "init", "--force"])
    assert forced.exit_code == 0


def test_config_path_reports_the_env_file(tmp_path: Path) -> None:
    """`config path` names the .env it reads and whether it is present."""
    result = CliRunner().invoke(cli, ["config", "path"])
    assert result.exit_code == 0
    assert ".env" in result.output
    assert "not present" in result.output


# --- rag search / dotenv / parse-args --------------------------------------


def test_rag_search_errors_cleanly_without_an_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rag search` with no index is a clean CLI error, not a traceback."""
    monkeypatch.setattr("aiida_agents.cli.commands.find_unrecognized_settings", list)
    monkeypatch.setattr(
        "aiida_agents.rag.retriever.docs_index_available", lambda: False
    )
    result = CliRunner().invoke(cli, ["rag", "search", "what is a calcjob"])
    assert result.exit_code == 1
    assert "No RAG index" in result.output


def test_dotenv_keys_parses_assignments(tmp_path: Path) -> None:
    """Keys are upper-cased; comments, blanks, non-assignments, and `export ` handled."""
    from aiida_agents.cli.config import _dotenv_keys

    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nAIIDA_AGENTS_MODEL=foo\n"
        "export OLLAMA_BASE_URL=http://x\nnot_an_assignment\nlower=bar\n",
        encoding="utf-8",
    )
    assert _dotenv_keys(env) == {"AIIDA_AGENTS_MODEL", "OLLAMA_BASE_URL", "LOWER"}


def test_dotenv_keys_missing_file_is_empty(tmp_path: Path) -> None:
    """A missing .env yields no keys rather than raising."""
    from aiida_agents.cli.config import _dotenv_keys

    assert _dotenv_keys(tmp_path / "absent.env") == set()


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param({"a": 1}, {"a": 1}, id="dict-passthrough"),
        pytest.param('{"a": 1}', {"a": 1}, id="json-string"),
        pytest.param("not json", {}, id="malformed-json"),
        pytest.param(None, {}, id="none"),
        pytest.param("[1, 2]", {}, id="json-but-not-a-dict"),
    ],
)
def test_parse_args_coerces_to_dict(
    raw: str | dict[str, object] | None, expected: dict[str, object]
) -> None:
    """Tool-call args become a dict whether they arrive as a dict, JSON, or junk."""
    from aiida_agents.cli.hitl import _parse_args

    assert _parse_args(raw) == expected
