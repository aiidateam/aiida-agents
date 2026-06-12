"""Unit tests for the RAG retriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from aiida_agents.rag.retriever import (
    _chunk_text,
    _extract_text_sections,
    _split_large_text,
    query_docs,
)


class TestExtractTextSections:
    def test_single_heading(self) -> None:
        text = "KpointsData\n===========\nBody text here.\n"
        sections = _extract_text_sections(text)
        assert any(title == "KpointsData" for title, _ in sections)

    def test_multiple_headings(self) -> None:
        text = "First\n=====\nBody one.\n\nSecond\n======\nBody two.\n"
        sections = _extract_text_sections(text)
        titles = [t for t, _ in sections]
        assert "First" in titles
        assert "Second" in titles

    def test_no_headings_returns_full_text(self) -> None:
        text = "Just a plain paragraph with no headings at all."
        sections = _extract_text_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == ""
        assert "plain paragraph" in sections[0][1]

    def test_preamble_before_first_heading(self) -> None:
        text = "Intro text.\n\nSection\n=======\nBody.\n"
        sections = _extract_text_sections(text)
        assert sections[0][0] == ""
        assert "Intro" in sections[0][1]

    def test_body_does_not_include_next_heading(self) -> None:
        # Underline must be at least as long as the title
        text = "Alpha\n=====\nBody A.\n\nBeta\n====\nBody B.\n"
        sections = _extract_text_sections(text)
        body_a = next(body for title, body in sections if title == "Alpha")
        assert "Body B" not in body_a


class TestSplitLargeText:
    def test_short_text_is_not_split(self) -> None:
        text = "Short text."
        assert _split_large_text(text, max_chars=1000) == [text]

    def test_splits_at_paragraph_boundary(self) -> None:
        text = ("A" * 300) + "\n\n" + ("B" * 300)
        chunks = _split_large_text(text, max_chars=400)
        assert len(chunks) == 2

    def test_all_chunks_within_max_chars(self) -> None:
        text = " ".join(["word"] * 500)
        chunks = _split_large_text(text, max_chars=100)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_no_empty_chunks(self) -> None:
        text = "A" * 50 + "\n\n" + "B" * 50
        chunks = _split_large_text(text, max_chars=60)
        assert all(c.strip() for c in chunks)


class TestChunkText:
    def test_returns_list_of_dicts(self) -> None:
        text = "WorkChain\n=========\nA work chain is a process.\n"
        chunks = _chunk_text(text, source="topics/workflows.txt")
        assert all(isinstance(c, dict) for c in chunks)
        assert all({"text", "source", "section"} <= c.keys() for c in chunks)

    def test_source_preserved(self) -> None:
        text = "Title\n=====\nSome content here that is long enough.\n"
        chunks = _chunk_text(text, source="topics/data_types.txt")
        assert all(c["source"] == "topics/data_types.txt" for c in chunks)

    def test_section_title_in_chunk_text(self) -> None:
        text = "KpointsData\n===========\n" + "x" * 200 + "\n"
        chunks = _chunk_text(text, source="topics/data_types.txt")
        assert any("KpointsData" in c["text"] for c in chunks)

    def test_short_sections_discarded(self) -> None:
        text = "Title\n=====\nToo short.\n"
        chunks = _chunk_text(text, source="any.txt")
        assert len(chunks) == 0

    def test_fallback_for_headingless_file(self) -> None:
        text = "x" * 500  # long enough, no headings
        chunks = _chunk_text(text, source="misc.txt")
        assert len(chunks) == 1
        # No headings → parsed as preamble, falls back to single chunk
        assert chunks[0]["section"] in ("(full file)", "(preamble)")


class TestQueryDocs:
    def test_returns_empty_list_when_collection_missing(self) -> None:
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch("aiida_agents.rag.embeddings.get_embedding_function"),
        ):
            results = query_docs("anything")

        assert results == []

    def test_returns_results_with_expected_keys(self) -> None:
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "aiida_docs"
        mock_client.list_collections.return_value = [mock_col]
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc text"]],
            "metadatas": [[{"source": "topics/foo.txt", "section": "Foo"}]],
        }
        mock_client.get_collection.return_value = mock_collection

        mock_embed_fn = MagicMock()
        mock_embed_fn.embed_query.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch("aiida_agents.rag.retriever._get_client", return_value=mock_client),
            patch(
                "aiida_agents.rag.embeddings.get_embedding_function",
                return_value=mock_embed_fn,
            ),
        ):
            results = query_docs("What is Foo?", limit=1)

        assert len(results) == 1
        assert results[0]["text"] == "doc text"
        assert results[0]["source"] == "topics/foo.txt"
        assert results[0]["section"] == "Foo"
