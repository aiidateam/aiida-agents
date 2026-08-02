"""Tests for turning solved forum threads into eval cases.

The fetching half needs the network and is not tested; this is the half where
the bugs live. Two things matter and both are about *rejection*: a thread that
should not become a case must not become one, because a case scored against a
worthless expected answer inflates the whole set silently -- and code inside a
post must survive, because for this corpus the code usually is the answer.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import pytest

from tests.evals.discourse import (
    MIN_ANSWER_CHARS,
    case_from_topic,
    cases_from_topics,
    html_to_text,
)

BASE = "https://aiida.discourse.group"

_LONG = "You need to use the QueryBuilder with a filter on the process state. " * 3


def _topic(
    *,
    posts: list[dict[str, t.Any]] | None = None,
    accepted: dict[str, t.Any] | None = None,
    topic_id: int = 42,
    title: str = "How do I query for failed calculations in my database?",
    tags: list[str] | None = None,
) -> dict[str, t.Any]:
    """A thread in the shape ``/t/<id>.json`` returns it."""
    if posts is None:
        posts = [
            {
                "post_number": 1,
                "username": "asker",
                "cooked": "<p>I have thousands of calculations and cannot find "
                "the failed ones. What is the right way to do this?</p>",
            },
            {
                "post_number": 2,
                "username": "maintainer",
                "cooked": f"<p>{_LONG}</p>",
                "accepted_answer": True,
            },
        ]
    topic: dict[str, t.Any] = {
        "id": topic_id,
        "title": title,
        "slug": "how-do-i-query",
        "post_stream": {"posts": posts},
    }
    if accepted is not None:
        topic["accepted_answer"] = accepted
    if tags is not None:
        topic["tags"] = tags
    return topic


class TestHtmlToText:
    """Discourse serves rendered HTML; the code in it is usually the answer."""

    def test_a_code_block_survives_as_a_fence(self) -> None:
        cooked = (
            '<p>Try this:</p><pre><code class="lang-python">qb = QueryBuilder()\n'
            "qb.append(CalcJobNode)</code></pre>"
        )

        text = html_to_text(cooked)

        assert "```python" in text
        assert "qb.append(CalcJobNode)" in text

    def test_code_keeps_its_line_breaks(self) -> None:
        """Flattened to one line, a snippet is useless as an expected answer."""
        cooked = "<pre><code>line_one()\nline_two()</code></pre>"

        assert "line_one()\nline_two()" in html_to_text(cooked)

    def test_a_fence_with_no_language_is_still_fenced(self) -> None:
        text = html_to_text("<pre><code>verdi process list</code></pre>")

        assert "```" in text
        assert "verdi process list" in text

    def test_html_entities_are_unescaped(self) -> None:
        """&gt; in a traceback or a shell prompt must read as > again."""
        cooked = "<pre><code>a &lt; b &amp;&amp; c &gt; d</code></pre>"

        assert "a < b && c > d" in html_to_text(cooked)

    def test_tags_are_stripped_from_prose(self) -> None:
        text = html_to_text(
            '<p>Use the <a href="#x"><code>QueryBuilder</code></a>.</p>'
        )

        assert text == "Use the QueryBuilder."

    def test_paragraphs_become_line_breaks_not_run_ons(self) -> None:
        text = html_to_text("<p>First point.</p><p>Second point.</p>")

        assert "First point.\nSecond point." in text


class TestAcceptedAnswers:
    """Which post counts as the solution."""

    def test_a_post_flagged_accepted_is_found(self) -> None:
        case = case_from_topic(_topic(), BASE)

        assert case is not None
        assert case.answer.startswith("You need to use the QueryBuilder")

    def test_a_top_level_accepted_answer_pointer_is_followed(self) -> None:
        """The solved plugin has also reported it this way."""
        posts = [
            {"post_number": 1, "username": "asker", "cooked": "<p>" + _LONG + "</p>"},
            {"post_number": 3, "username": "helper", "cooked": f"<p>{_LONG}</p>"},
        ]

        case = case_from_topic(
            _topic(posts=posts, accepted={"post_number": 3}),
            BASE,
        )

        assert case is not None
        assert case.answer.startswith("You need to use")

    def test_an_inlined_accepted_answer_is_used(self) -> None:
        """Sometimes the answer is inlined rather than pointed at."""
        posts = [
            {"post_number": 1, "username": "asker", "cooked": "<p>" + _LONG + "</p>"}
        ]
        accepted = {"post_number": 9, "username": "helper", "cooked": f"<p>{_LONG}</p>"}

        case = case_from_topic(_topic(posts=posts, accepted=accepted), BASE)

        assert case is not None


class TestRejection:
    """Threads that must not become cases.

    Each of these would otherwise produce a case scored against something
    worthless, which makes the whole set read as easier than it is.
    """

    def test_a_thread_with_no_accepted_answer_is_rejected(self) -> None:
        posts = [
            {"post_number": 1, "username": "asker", "cooked": "<p>Help?</p>"},
            {"post_number": 2, "username": "other", "cooked": f"<p>{_LONG}</p>"},
        ]

        assert case_from_topic(_topic(posts=posts), BASE) is None

    def test_a_thanks_that_worked_answer_is_rejected(self) -> None:
        """Marked as the solution constantly, and says nothing."""
        posts = [
            {"post_number": 1, "username": "asker", "cooked": "<p>" + _LONG + "</p>"},
            {
                "post_number": 2,
                "username": "helper",
                "cooked": "<p>Thanks, that worked!</p>",
                "accepted_answer": True,
            },
        ]

        assert case_from_topic(_topic(posts=posts), BASE) is None

    def test_the_answer_length_bar_is_where_it_claims_to_be(self) -> None:
        """Guards the constant itself, not just some value below it."""
        posts = [
            {"post_number": 1, "username": "asker", "cooked": "<p>" + _LONG + "</p>"},
            {
                "post_number": 2,
                "username": "helper",
                "cooked": "<p>" + "y" * (MIN_ANSWER_CHARS - 1) + "</p>",
                "accepted_answer": True,
            },
        ]
        assert case_from_topic(_topic(posts=posts), BASE) is None

        posts[1]["cooked"] = "<p>" + "y" * MIN_ANSWER_CHARS + "</p>"
        assert case_from_topic(_topic(posts=posts), BASE) is not None

    def test_a_self_answered_thread_is_rejected(self) -> None:
        """ "Never mind, I had a typo" is marked solved as often as a real answer."""
        posts = [
            {"post_number": 1, "username": "asker", "cooked": "<p>" + _LONG + "</p>"},
            {
                "post_number": 2,
                "username": "asker",
                "cooked": f"<p>{_LONG}</p>",
                "accepted_answer": True,
            },
        ]

        assert case_from_topic(_topic(posts=posts), BASE) is None

    def test_a_thread_with_no_posts_is_rejected(self) -> None:
        assert case_from_topic(_topic(posts=[]), BASE) is None

    def test_a_question_too_short_to_be_one_is_rejected(self) -> None:
        posts = [
            {"post_number": 1, "username": "asker", "cooked": "<p>?</p>"},
            {
                "post_number": 2,
                "username": "helper",
                "cooked": f"<p>{_LONG}</p>",
                "accepted_answer": True,
            },
        ]

        assert case_from_topic(_topic(posts=posts, title="Help"), BASE) is None


class TestTheCase:
    """What a usable thread turns into."""

    def test_the_title_is_part_of_the_question(self) -> None:
        """It often carries the whole question; as metadata it would be lost."""
        case = case_from_topic(_topic(), BASE)

        assert case is not None
        assert case.question.startswith("How do I query for failed calculations")

    def test_the_url_points_back_at_the_thread(self) -> None:
        """So a failing case can be read in context rather than guessed at."""
        case = case_from_topic(_topic(), BASE)

        assert case is not None
        assert case.url == f"{BASE}/t/how-do-i-query/42"

    def test_the_row_is_shaped_for_a_dataset(self) -> None:
        case = case_from_topic(_topic(tags=["querybuilder"]), BASE)

        assert case is not None
        row = case.as_row()
        assert set(row) == {"name", "inputs", "expected_output", "metadata"}
        assert row["metadata"]["tags"] == ["querybuilder"]

    def test_a_topic_with_no_integer_id_is_rejected(self) -> None:
        topic = _topic()
        topic["id"] = None

        assert case_from_topic(topic, BASE) is None


class TestBatches:
    def test_the_same_thread_twice_yields_one_case(self) -> None:
        """Discourse search paginates with overlap."""
        cases = cases_from_topics([_topic(), _topic()], BASE)

        assert len(cases) == 1

    def test_unusable_threads_are_dropped_not_fatal(self) -> None:
        """One malformed thread must not cost the whole scrape."""
        cases = cases_from_topics([_topic(), {"id": 1}, _topic(topic_id=99)], BASE)

        assert sorted(c.topic_id for c in cases) == [42, 99]


@pytest.mark.parametrize("missing", ["post_stream", "id"])
def test_a_topic_missing_a_required_key_is_rejected(missing: str) -> None:
    topic = _topic()
    del topic[missing]

    assert case_from_topic(topic, BASE) is None


class TestDatasetRoundTrip:
    """The scrape's output must be what the eval tier can load.

    This is the seam that would otherwise fail only on a real run, after the
    scrape has already spent a minute against a community server and the eval
    has started spending tokens.
    """

    def test_written_cases_load_back_as_a_dataset(self, tmp_path: Path) -> None:
        from pydantic_evals import Dataset

        from tests.evals.discourse import write_dataset

        cases = cases_from_topics([_topic(), _topic(topic_id=99)], BASE)
        out = tmp_path / "discourse.yaml"

        write_dataset(cases, out)
        loaded: Dataset[str, str, dict[str, t.Any]] = Dataset.from_file(out)

        assert len(loaded.cases) == 2
        assert loaded.cases[0].inputs.startswith("How do I query")
        assert (loaded.cases[0].expected_output or "").startswith("You need to use")

    def test_the_thread_url_survives_into_the_dataset(self, tmp_path: Path) -> None:
        """A failing case has to be readable in context to be actionable."""
        from pydantic_evals import Dataset

        from tests.evals.discourse import write_dataset

        out = tmp_path / "discourse.yaml"
        write_dataset(cases_from_topics([_topic()], BASE), out)

        loaded: Dataset[str, str, dict[str, t.Any]] = Dataset.from_file(out)
        assert (loaded.cases[0].metadata or {})["url"] == (
            f"{BASE}/t/how-do-i-query/42"
        )

    def test_the_directory_is_created_if_absent(self, tmp_path: Path) -> None:
        from tests.evals.discourse import write_dataset

        out = tmp_path / "datasets" / "discourse.yaml"
        write_dataset(cases_from_topics([_topic()], BASE), out)

        assert out.exists()


class TestScopeInTheDataset:
    """The scope label is written into the case, not computed at scoring time.

    Stored so it can be audited, corrected by hand, or disagreed with. A
    heuristic label that only existed in memory would be impossible to check
    against the thread it came from.
    """

    def test_a_case_carries_its_scope(self) -> None:
        row = case_from_topic(_topic(), BASE).as_row()  # type: ignore[union-attr]

        assert row["metadata"]["scope"] in {"in_scope", "out_of_scope", "unclear"}

    def test_an_infrastructure_thread_is_labelled_out_of_scope(self) -> None:
        row = case_from_topic(
            _topic(title="How to use scp/rsync instead of sftp for transport"),
            BASE,
        ).as_row()  # type: ignore[union-attr]

        assert row["metadata"]["scope"] == "out_of_scope"

    def test_the_signals_that_decided_it_are_recorded(self) -> None:
        """So a label someone disputes can be checked without re-reading the thread."""
        topic = _topic(title="sftp transport trouble on my cluster")
        row = case_from_topic(topic, BASE).as_row()  # type: ignore[union-attr]

        assert "sftp" in row["metadata"]["scope_signals"]


class TestTagShapes:
    """Discourse does not agree with itself about what a tag is.

    Most installs return a list of strings; the live AiiDA forum returned
    objects, which reached a ``str.join`` and killed a scrape after it had
    already fetched sixty threads. A tag is a hint for the scope classifier and
    is never worth failing a run over, so every unexpected shape degrades to
    "no tags" rather than an exception.
    """

    def test_string_tags_are_kept(self) -> None:
        case = case_from_topic(_topic(tags=["querybuilder", "provenance"]), BASE)

        assert case is not None
        assert case.tags == ("querybuilder", "provenance")

    def test_object_tags_are_reduced_to_their_names(self) -> None:
        """The shape the live forum actually returned."""
        topic = _topic()
        topic["tags"] = [{"name": "scheduler", "id": 12}, {"name": "slurm"}]

        case = case_from_topic(topic, BASE)

        assert case is not None
        assert case.tags == ("scheduler", "slurm")

    def test_a_mixed_list_keeps_what_it_can(self) -> None:
        topic = _topic()
        topic["tags"] = ["query", {"name": "provenance"}, 42, None]

        case = case_from_topic(topic, BASE)

        assert case is not None
        assert case.tags == ("query", "provenance")

    def test_a_nonsense_tags_field_is_not_fatal(self) -> None:
        topic = _topic()
        topic["tags"] = "not-a-list"

        case = case_from_topic(topic, BASE)

        assert case is not None
        assert case.tags == ()

    def test_object_tags_reach_the_scope_classifier(self) -> None:
        """The end the crash was at: tags must be usable, not merely survivable."""
        topic = _topic(title="Something is odd")
        topic["tags"] = [{"name": "scheduler"}]

        row = case_from_topic(topic, BASE).as_row()  # type: ignore[union-attr]

        assert row["metadata"]["scope"] == "out_of_scope"
