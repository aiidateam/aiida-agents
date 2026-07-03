#!/usr/bin/env python3
"""Manual verification script for the RAG pipeline.

Run from the repo root:
    uv run python dev/test_rag.py --reindex    # first time — force a full rebuild
    uv run python dev/test_rag.py              # subsequent runs — use cached index
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the AiiDA RAG pipeline.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force a full rebuild of the vector DB.",
    )
    args = parser.parse_args()

    from aiida_agents.rag import index_docs, query_docs

    print("=== Indexing AiiDA docs ===")
    index_docs(force=args.reindex)

    test_queries = [
        "What is a CalcJobNode?",
        "How do I set up a WorkChain?",
        "What is the difference between CalcJob and WorkChain?",
        "How does the provenance graph work?",
        "What are KpointsData nodes?",
    ]

    print("\n=== Querying ===\n")
    all_good = True

    for query in test_queries:
        print(f"Q: {query}")
        results = query_docs(query, limit=2)

        for i, r in enumerate(results, 1):
            section = r.get("section", "?")
            source = r["source"]
            snippet = r["text"][r["text"].find("\n") + 1 :][:200].strip()

            print(f"  [{i}] {source}  §  {section}")
            print(f"       {snippet}...")

            # Simple self-check: warn if the source looks unrelated to the query
            query_lower = query.lower()
            source_lower = source.lower()
            section_lower = section.lower()

            unrelated = False
            if "workchain" in query_lower and "ssh" in source_lower:
                unrelated = True
            if "calcjob" in query_lower and "sqlite" in source_lower:
                unrelated = True
            if (
                "kpoints" in query_lower
                and "database" in source_lower
                and "kpoint" not in section_lower
            ):
                unrelated = True

            if unrelated:
                print("  ⚠️  WARNING: result looks unrelated — chunking may need tuning")
                all_good = False

        print()

    if all_good:
        print(
            "✅  All results look reasonable. Reindex with --reindex if you change the chunker."
        )
    else:
        print(
            "⚠️  Some results look off. Run with --reindex to rebuild after any code changes."
        )


if __name__ == "__main__":
    main()
