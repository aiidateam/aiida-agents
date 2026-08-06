"""Tests for cli/output.py: reply rendering and duration formatting."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from rich.console import Console

from aiida_agents._logging import TRACE_LOGGER_NAMES
from aiida_agents.cli.output import _format_duration, _log_tool_calls_debug


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


def test_print_agent_renders_labelled_reply(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reply prints under a bold 'Agent:' label with its body rendered as
    markdown, so emphasis and tables show instead of raw ``**`` syntax.
    """
    from aiida_agents.cli.output import _print_agent

    _print_agent("hello **world**")

    out = capsys.readouterr().out
    assert "Agent:" in out
    assert "hello" in out
    assert "world" in out
    assert "**" not in out  # markdown was rendered, not echoed verbatim


def test_print_reply_raw_when_piped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Piped/redirected output is the raw Markdown source, so `ask ... > out.md`
    and pipelines stay clean and re-renderable, not box-drawing rich output.
    """
    from aiida_agents.cli import output

    # Force non-terminal so the raw branch is exercised deterministically,
    # independent of how pytest captures stdout (e.g. under `pytest -s`).
    monkeypatch.setattr(output, "console", Console(force_terminal=False))
    output._print_reply("# Title\n\n**bold**")

    out = capsys.readouterr().out
    assert "# Title" in out  # raw markdown preserved
    assert "**bold**" in out


def test_print_reply_renders_markdown_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On an interactive terminal the reply is rendered, so the `#`/`**` markup is
    consumed rather than printed literally.
    """
    from aiida_agents.cli import output

    monkeypatch.setattr(output, "console", Console(force_terminal=True, width=80))
    output._print_reply("# Title")

    out = capsys.readouterr().out
    assert "Title" in out
    assert "# Title" not in out


def test_print_reply_raw_flag_forces_source_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--raw` prints the Markdown source even on a terminal (copy-paste / piping
    override), bypassing the interactive rendering.
    """
    from aiida_agents.cli import output

    monkeypatch.setattr(output, "console", Console(force_terminal=True, width=80))
    output._print_reply("# Title", raw=True)

    assert "# Title" in capsys.readouterr().out


def _codegen_call(code: object) -> list[ModelMessage]:
    """A message list with one ``run_aiida_code`` tool call carrying ``code``."""
    return [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="run_aiida_code", args={"code": code}, tool_call_id="c1"
                )
            ]
        )
    ]


def test_generated_code_extracts_run_aiida_code_snippets() -> None:
    """Only ``run_aiida_code`` args are collected, in order; other tools ignored."""
    from aiida_agents.cli.output import _generated_code

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="search_aiida_code",
                    args={"task": "count nodes"},
                    tool_call_id="c0",
                )
            ]
        ),
        *_codegen_call("print(len(qb.all()))"),
    ]
    assert _generated_code(messages) == ["print(len(qb.all()))"]


def test_generated_code_handles_json_string_args() -> None:
    """Some providers deliver tool args as a JSON string rather than a dict."""
    from aiida_agents.cli.output import _generated_code

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="run_aiida_code",
                    args='{"code": "print(1)"}',
                    tool_call_id="c1",
                )
            ]
        )
    ]
    assert _generated_code(messages) == ["print(1)"]


def test_generated_code_empty_without_codegen() -> None:
    """A turn with no ``run_aiida_code`` call yields no snippets."""
    from aiida_agents.cli.output import _generated_code

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart(tool_name="query_nodes", args={}, tool_call_id="c0")]
        )
    ]
    assert _generated_code(messages) == []


def test_save_generated_code_writes_a_runnable_py_file(tmp_path: Path) -> None:
    """Each snippet becomes its own ``.py`` file, with a header and the code."""
    from aiida_agents.cli.output import _save_generated_code

    paths = _save_generated_code(["print('hi')"], tmp_path)

    assert len(paths) == 1
    assert paths[0].suffix == ".py"
    text = paths[0].read_text(encoding="utf-8")
    assert text.startswith("# Generated by aiida-agents")
    assert "print('hi')" in text


def test_show_generated_code_prints_the_snippet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The code the agent ran is shown under a label."""
    from aiida_agents.cli.output import show_generated_code

    show_generated_code(_codegen_call("print('hi')"), Console(color_system=None))

    out = capsys.readouterr().out
    assert "Generated code" in out
    assert "print" in out


def test_show_generated_code_is_a_noop_without_codegen(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-codegen turn prints nothing."""
    from aiida_agents.cli.output import show_generated_code

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart(tool_name="query_nodes", args={}, tool_call_id="c0")]
        )
    ]
    show_generated_code(messages, Console(color_system=None))

    assert capsys.readouterr().out == ""
