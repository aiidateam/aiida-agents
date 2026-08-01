#!/usr/bin/env python
"""Build an evaluation dataset from solved AiiDA Discourse threads.

Run this once to produce ``tests/evals/datasets/discourse.yaml``, the set of
real user questions the answer-quality evals score against:

    python dev/fetch_discourse.py --limit 60

Only the HTTP lives here. Every decision about what makes a usable case ---
and every line worth testing --- is in ``tests/evals/discourse``, which this
imports. That split is why the parsing has coverage while this does not: it
needs the live forum, so it cannot run in CI.

Be considerate: this hits a community server, so requests are paced and the
default limit is small. Re-run it when you want fresher cases, not on a loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import typing as t
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from tests.evals.discourse import cases_from_topics, write_dataset  # noqa: E402

BASE_URL = "https://aiida.discourse.group"
OUT_PATH = _ROOT / "tests" / "evals" / "datasets" / "discourse.yaml"

#: Seconds between requests. A scrape is not worth degrading a community forum
#: for; at this pace a 60-case run takes about a minute.
PAUSE_SECONDS = 1.0

_USER_AGENT = "aiida-agents-eval-builder (https://github.com/aiidateam/aiida-agents)"


def _get_json(url: str, timeout: float = 30.0) -> dict[str, t.Any]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def solved_topic_ids(base_url: str, limit: int) -> list[int]:
    """Ids of solved threads, newest first, across as many pages as needed."""
    ids: list[int] = []
    page = 1
    while len(ids) < limit:
        url = f"{base_url}/search.json?q=status%3Asolved&page={page}"
        try:
            payload = _get_json(url)
        except urllib.error.URLError as exc:
            print(f"  search page {page} failed: {exc}", file=sys.stderr)
            break

        topics = payload.get("topics") or []
        if not topics:
            break
        for topic in topics:
            topic_id = topic.get("id")
            if isinstance(topic_id, int) and topic_id not in ids:
                ids.append(topic_id)
        page += 1
        time.sleep(PAUSE_SECONDS)
    return ids[:limit]


def fetch_topics(base_url: str, ids: t.Sequence[int]) -> list[dict[str, t.Any]]:
    """Full thread JSON for each id, skipping any that cannot be fetched."""
    topics: list[dict[str, t.Any]] = []
    for index, topic_id in enumerate(ids, start=1):
        try:
            topics.append(_get_json(f"{base_url}/t/{topic_id}.json"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"  [{index}/{len(ids)}] topic {topic_id} skipped: {exc}")
            continue
        print(f"  [{index}/{len(ids)}] topic {topic_id}")
        time.sleep(PAUSE_SECONDS)
    return topics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=60, help="Threads to consider (default: 60)."
    )
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    print(f"searching {args.base_url} for solved threads...")
    ids = solved_topic_ids(args.base_url, args.limit)
    if not ids:
        print("no solved threads found -- is the forum reachable?", file=sys.stderr)
        return 1
    print(f"found {len(ids)} solved threads; fetching each...")

    topics = fetch_topics(args.base_url, ids)
    cases = cases_from_topics(topics, args.base_url)

    # The rejection count is the interesting number: a scrape that keeps almost
    # nothing usually means the forum reports accepted answers in a shape
    # ``discourse.case_from_topic`` does not recognise, not that the threads
    # are bad.
    print(
        f"\n{len(cases)} usable cases from {len(topics)} threads "
        f"({len(topics) - len(cases)} rejected: no accepted answer, "
        f"self-answered, or too short)"
    )
    if not cases:
        return 1

    write_dataset(cases, args.out)
    print(f"wrote {args.out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
