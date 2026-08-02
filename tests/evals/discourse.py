"""Turn solved AiiDA Discourse threads into evaluation cases.

Every eval case this project has had so far was written by us, which makes them
a test of what we thought to ask. The forum is the opposite: real questions real
users actually had, and --- for a solved thread --- an answer someone with
standing marked correct. That is ground truth nobody on this project invented,
and it is the only source of it we have.

The split here is deliberate. Fetching needs the network and cannot be tested in
CI; parsing is where the bugs live and is pure, so it is separated and covered.
``dev/fetch_discourse.py`` is a thin wrapper that does the HTTP and calls into
this module.

A thread becomes a case only if it clears every bar in :func:`case_from_topic`.
Being strict is the point: a question with no accepted answer, or one answered
with "fixed it, thanks", scores our agent against nothing and quietly makes the
whole set look easier than it is.
"""

from __future__ import annotations

import html
import re
import typing as t
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DiscourseCase",
    "case_from_topic",
    "cases_from_topics",
    "html_to_text",
    "write_dataset",
]

#: Shortest accepted answer worth scoring against. "Thanks, that worked!" is
#: marked as the solution on plenty of threads; as an expected output it says
#: nothing, and a judge comparing to it would pass almost any reply.
MIN_ANSWER_CHARS = 120

#: Shortest question worth asking. A one-line title with no body is usually a
#: thread whose real content is in a screenshot or a linked traceback.
MIN_QUESTION_CHARS = 40

_CODE_BLOCK = re.compile(
    r"<pre[^>]*>\s*<code(?P<attrs>[^>]*)>(?P<code>.*?)</code>\s*</pre>",
    re.DOTALL,
)
_LANG = re.compile(r"lang-(\w+)")
_TAG = re.compile(r"<[^>]+>")
_BLANK_RUN = re.compile(r"\n{3,}")


def html_to_text(cooked: str) -> str:
    """Flatten Discourse's rendered HTML to text, keeping code blocks intact.

    Discourse serves posts as ``cooked`` HTML. Stripping tags naively would
    turn a traceback or a QueryBuilder snippet into an unindented run-on line
    --- and for this corpus the code *is* the answer, so it is fenced back up
    with its language rather than flattened with the prose.

    Entities are left alone until the very end on purpose: unescaping a code
    block first would turn ``a &lt; b`` into ``a < b``, and the tag-stripping
    pass that follows would then delete ``< b ...>`` as if it were markup.

    Args:
        cooked: A post's rendered HTML.

    Returns:
        Plain text with ```` ``` ```` fenced code blocks.
    """

    def _fence(match: re.Match[str]) -> str:
        language = _LANG.search(match.group("attrs"))
        code = match.group("code").strip("\n")
        return f"\n```{language.group(1) if language else ''}\n{code}\n```\n"

    text = _CODE_BLOCK.sub(_fence, cooked)
    text = re.sub(r"</p>|<br\s*/?>", "\n", text)
    text = _TAG.sub("", text)
    return _BLANK_RUN.sub("\n\n", html.unescape(text)).strip()


@dataclass(frozen=True)
class DiscourseCase:
    """One solved thread, as a question and the answer that resolved it."""

    topic_id: int
    title: str
    question: str
    answer: str
    url: str
    tags: tuple[str, ...] = ()

    def as_row(self) -> dict[str, t.Any]:
        """The case in the shape ``pydantic_evals.Dataset`` loads from YAML.

        The scope label is computed here rather than at scoring time so it is
        written into the dataset and can be audited, corrected by hand, or
        disagreed with --- a heuristic label that only existed in memory would
        be impossible to check.
        """
        from tests.evals.scope import classify_scope

        verdict = classify_scope(self.title, self.question, self.tags)
        return {
            "name": f"discourse-{self.topic_id}",
            "inputs": self.question,
            "expected_output": self.answer,
            "metadata": {
                "url": self.url,
                "title": self.title,
                "tags": list(self.tags),
                "scope": verdict.scope,
                "scope_signals": list(verdict.signals),
            },
        }


def _posts(topic: dict[str, t.Any]) -> list[dict[str, t.Any]]:
    stream = topic.get("post_stream") or {}
    posts = stream.get("posts")
    return posts if isinstance(posts, list) else []


def _accepted_post(topic: dict[str, t.Any]) -> dict[str, t.Any] | None:
    """The post marked as the solution, if the thread has one.

    Two shapes are accepted because the discourse-solved plugin has used both:
    a top-level ``accepted_answer`` naming a post number, and a per-post
    ``accepted_answer: true`` flag. Neither is assumed present.
    """
    posts = _posts(topic)
    for post in posts:
        if post.get("accepted_answer") is True:
            return post

    accepted = topic.get("accepted_answer")
    if isinstance(accepted, dict):
        number = accepted.get("post_number")
        for post in posts:
            if post.get("post_number") == number:
                return post
        # The plugin sometimes inlines the answer instead of pointing at it.
        if accepted.get("cooked"):
            return accepted
    return None


def case_from_topic(topic: dict[str, t.Any], base_url: str) -> DiscourseCase | None:
    """One thread as an eval case, or ``None`` if it is not usable as one.

    Rejected: a thread with no accepted answer (nothing to score against), one
    whose accepted answer is too short to be an answer, one whose question is
    too short to be a question, and one where the solution is the asker's own
    post --- "never mind, I had a typo" is marked solved as often as a real
    answer is, and it teaches an agent nothing.

    Args:
        topic: A parsed ``/t/<id>.json`` response.
        base_url: Forum root, e.g. ``https://aiida.discourse.group``.

    Returns:
        The case, or ``None`` if the thread does not clear the bars above.
    """
    accepted = _accepted_post(topic)
    if accepted is None:
        return None

    posts = _posts(topic)
    if not posts:
        return None
    asked = posts[0]

    if accepted.get("post_number") == asked.get("post_number"):
        return None
    if accepted.get("username") and accepted.get("username") == asked.get("username"):
        return None

    question = html_to_text(str(asked.get("cooked", "")))
    answer = html_to_text(str(accepted.get("cooked", "")))
    title = str(topic.get("title", "")).strip()

    # The title carries real signal -- often the whole question -- so it is
    # part of the prompt rather than metadata, and it counts toward the length.
    if title and not question.lower().startswith(title.lower()[:30]):
        question = f"{title}\n\n{question}" if question else title

    if len(question) < MIN_QUESTION_CHARS or len(answer) < MIN_ANSWER_CHARS:
        return None

    topic_id = topic.get("id")
    if not isinstance(topic_id, int):
        return None

    tags = topic.get("tags")
    return DiscourseCase(
        topic_id=topic_id,
        title=title,
        question=question,
        answer=answer,
        url=f"{base_url.rstrip('/')}/t/{topic.get('slug', 't')}/{topic_id}",
        tags=tuple(tags) if isinstance(tags, list) else (),
    )


def cases_from_topics(
    topics: t.Iterable[dict[str, t.Any]], base_url: str
) -> list[DiscourseCase]:
    """Every usable case in a batch of threads, deduplicated by topic id."""
    seen: dict[int, DiscourseCase] = {}
    for topic in topics:
        case = case_from_topic(topic, base_url)
        if case is not None:
            seen.setdefault(case.topic_id, case)
    return list(seen.values())


def write_dataset(cases: t.Sequence[DiscourseCase], out_path: Path) -> None:
    """Write cases in the format :meth:`pydantic_evals.Dataset.from_file` reads.

    Goes through ``Dataset``/``Case`` rather than dumping YAML directly, so the
    scrape cannot emit a file the eval tier then fails to load --- the two ends
    are held together by the same types.
    """
    from pydantic_evals import Case, Dataset

    dataset: Dataset[str, str, dict[str, t.Any]] = Dataset(
        name="aiida-discourse",
        cases=[Case(**case.as_row()) for case in cases],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_file(out_path)
