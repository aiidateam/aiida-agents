"""Tests for cli/output.py: reply rendering and duration formatting."""

from __future__ import annotations

import logging

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


class TestSourcesFooter:
    """The documentation links a run consulted, printed by the CLI.

    Live testing found the gap this closes: `search_aiida_docs` returned a
    correctly pinned readthedocs URL, the prompt asked the model to cite it,
    and the reply contained no link at all. Retrieval knows which pages it
    returned; that is not a judgement call, so the CLI prints them rather than
    depending on the model to repeat them.
    """

    @staticmethod
    def _messages(*tool_outputs: str) -> list[object]:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        return [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="search_aiida_docs", content=out)
                    for out in tool_outputs
                ]
            )
        ]

    def test_a_retrieved_url_is_printed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiida_agents.cli.output import _print_sources

        _print_sources(
            self._messages(
                "[howto/query § S]\nhttps://aiida-core.readthedocs.io/en/v2.8.0/"
                "howto/query.html#s\nsome text"
            )  # type: ignore[arg-type]
        )

        assert "howto/query.html#s" in capsys.readouterr().out

    def test_nothing_is_printed_when_no_tool_returned_a_url(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A database question consults no docs; an empty heading would mislead."""
        from aiida_agents.cli.output import _print_sources

        _print_sources(self._messages("{'pk': 334407}"))  # type: ignore[arg-type]

        assert capsys.readouterr().out == ""

    def test_the_same_page_twice_is_listed_once(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two excerpts from one page are one source."""
        from aiida_agents.cli.output import _print_sources

        url = "https://aiida-core.readthedocs.io/en/v2.8.0/howto/query.html"
        _print_sources(self._messages(f"a {url}", f"b {url}"))  # type: ignore[arg-type]

        assert capsys.readouterr().out.count(url) == 1

    def test_the_list_is_capped(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A long list reads as a bibliography rather than a pointer."""
        from aiida_agents.cli.output import _MAX_SOURCES, _print_sources

        urls = " ".join(f"https://example.invalid/p{n}.html" for n in range(10))
        _print_sources(self._messages(urls))  # type: ignore[arg-type]

        assert capsys.readouterr().out.count("https://") == _MAX_SOURCES

    def test_trailing_punctuation_is_not_part_of_the_link(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiida_agents.cli.output import _print_sources

        _print_sources(self._messages("see https://example.invalid/a.html, and then"))  # type: ignore[arg-type]

        out = capsys.readouterr().out
        assert "a.html" in out
        assert "a.html," not in out
