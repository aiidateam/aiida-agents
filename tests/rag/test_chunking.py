"""Unit tests for the RAG chunking logic (pure text -> chunks)."""

from __future__ import annotations

from pathlib import Path

from aiida_agents.rag.chunking import (
    _chunk_text,
    _extract_text_sections,
    _fence_segments,
    _load_docs,
    _split_fence_aware,
    _split_large_text,
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


class TestSplitFenceAware:
    """Fenced code blocks (the rag._textbuild corpus format) stay atomic."""

    _CODE_BLOCK = "```python\nfrom aiida.engine import submit\nsubmit(builder)\n```"

    def test_fence_segments_alternate(self) -> None:
        text = f"Intro prose.\n{self._CODE_BLOCK}\nTrailing prose.\n"
        segments = _fence_segments(text)
        assert [fenced for _, fenced in segments] == [False, True, False]
        assert "".join(segment for segment, _ in segments) == text

    def test_unterminated_fence_extends_to_end(self) -> None:
        text = "Prose.\n```python\ncode without closing fence\n"
        segments = _fence_segments(text)
        assert segments[-1][1] is True
        assert "code without closing" in segments[-1][0]

    def test_fence_never_split(self) -> None:
        """A fence near the size limit moves to its own chunk instead of
        being cut in half; every produced chunk has balanced fences."""
        text = ("prose " * 60).strip() + "\n" + self._CODE_BLOCK + "\n" + "tail prose."
        chunks = _split_fence_aware(text, max_chars=350)
        assert any(self._CODE_BLOCK in chunk for chunk in chunks)
        for chunk in chunks:
            assert chunk.count("```") % 2 == 0

    def test_oversized_fence_kept_whole(self) -> None:
        big_code = "```python\n" + "x = 1\n" * 100 + "```"
        chunks = _split_fence_aware(f"Intro.\n{big_code}\nOutro.", max_chars=200)
        assert any(chunk.strip() == big_code for chunk in chunks)

    def test_plain_text_delegates_to_large_split(self) -> None:
        text = ("A" * 300) + "\n\n" + ("B" * 300)
        assert _split_fence_aware(text, max_chars=400) == _split_large_text(
            text, max_chars=400
        )


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


class TestLoadDocs:
    def test_excludes_changelog_keeps_concept_pages(self, tmp_path: Path) -> None:
        """The changelog is dropped from the index; concept pages are kept."""
        (tmp_path / "reference").mkdir()
        (tmp_path / "reference" / "_changelog.txt").write_text(
            "Changelog\n=========\n" + "old release note. " * 30
        )
        (tmp_path / "topics").mkdir()
        (tmp_path / "topics" / "concepts.txt").write_text(
            "Concepts\n========\n" + "conceptual prose. " * 30
        )

        sources = {c["source"] for c in _load_docs(str(tmp_path))}

        assert "topics/concepts.txt" in sources
        assert "reference/_changelog.txt" not in sources
