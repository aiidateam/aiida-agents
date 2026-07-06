"""Parse Sphinx text output into embeddable chunks.

Pure text processing: read the ``sphinx-build -b text`` ``.txt`` files and turn
them into section-based chunks. No embeddings, no vector store, no network.

Sphinx text output keeps RST's underline heading convention but is already
clean prose (directives resolved, includes expanded, no ``:ref:`` noise), so
chunking keys off the headings.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TARGET_CHUNK_CHARS = 2000  # ~512 tokens; 2026 benchmark sweet spot
_MIN_CHUNK_CHARS = 150  # discard stubs shorter than this

# Corpus files excluded from the searchable index (posix-relative to the
# corpus root). The changelog is release notes, not conceptual prose, yet its
# chunks repeatedly out-ranked concept pages for questions like "what is a
# CalcJobNode" — noise a researcher never wants back. The corpus directory
# still mirrors the full docs; this only curates what gets embedded.
_EXCLUDED_SOURCES: frozenset[str] = frozenset({"reference/_changelog.txt"})

# A code-fence line: ``` optionally indented. Matches both the opening
# ```lang line and the bare closing ``` (the corpus format from rag._textbuild).
_FENCE_RE = re.compile(r"^\s*```")


def _extract_text_sections(text: str) -> list[tuple[str, str]]:
    """Parse a Sphinx .txt file into (heading, body) pairs.

    Sphinx text output uses the same underline convention as RST but
    the file is already clean prose — no directives, no :ref: noise.

    A heading looks like:
        KpointsData
        ===========
        Body text here...

    Returns list of (title, body) tuples.
    The first tuple has title='' for content before the first heading.
    """
    heading_chars = set("=-~^+#*")
    lines = text.splitlines(keepends=True)

    # Find all heading positions: line i is a heading title if line i+1
    # is an underline of the same length (or longer). Lines inside a fenced
    # code block never count: console output frequently contains
    # heading-lookalikes, and a section cut there would split the fence.
    heading_positions: list[int] = []  # index of the TITLE line
    in_fence = False
    for i in range(len(lines) - 1):
        if _FENCE_RE.match(lines[i]):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        title_line = lines[i].rstrip("\n")
        next_line = lines[i + 1].rstrip("\n")
        if (
            title_line.strip()  # title is non-empty
            and len(next_line) >= len(title_line.strip())  # underline is long enough
            and len(next_line) >= 3
            and all(c in heading_chars for c in next_line.strip())
            and len(set(next_line.strip())) == 1  # all same char
        ):
            heading_positions.append(i)

    if not heading_positions:
        return [("", text.strip())]

    sections: list[tuple[str, str]] = []

    # Content before first heading
    if heading_positions[0] > 0:
        pre = "".join(lines[: heading_positions[0]]).strip()
        if pre:
            sections.append(("", pre))

    for pos, title_idx in enumerate(heading_positions):
        title = lines[title_idx].strip()
        # Body starts 2 lines after title (title + underline)
        body_start = title_idx + 2
        # Body ends at the next heading title line (or EOF)
        body_end = (
            heading_positions[pos + 1]
            if pos + 1 < len(heading_positions)
            else len(lines)
        )
        body = "".join(lines[body_start:body_end]).strip()
        sections.append((title, body))

    return sections


def _fence_segments(text: str) -> list[tuple[str, bool]]:
    """Split ``text`` into ``(segment, is_fenced)`` pieces, in order.

    A fenced segment runs from a line starting with ``` to the next such
    line, inclusive (the corpus format emitted by ``rag._textbuild``). An
    unterminated fence extends to the end of the text.
    """
    segments: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            if in_fence:
                buf.append(line)
                segments.append(("".join(buf), True))
                buf = []
            else:
                if buf:
                    segments.append(("".join(buf), False))
                buf = [line]
            in_fence = not in_fence
        else:
            buf.append(line)
    if buf:
        segments.append(("".join(buf), in_fence))
    return segments


def _split_fence_aware(text: str, max_chars: int) -> list[str]:
    """Split like ``_split_large_text``, but never inside a fenced code block.

    Fenced blocks are atomic: they merge with neighbouring prose when the
    combination fits ``max_chars``, and otherwise stand as their own chunk,
    kept whole even when they alone exceed the limit. Splitting a fence in
    two would leave both halves unparseable as code.
    """
    if "```" not in text:
        return _split_large_text(text, max_chars)

    atoms: list[str] = []
    for segment, fenced in _fence_segments(text):
        if fenced:
            atoms.append(segment.strip("\n"))
        else:
            atoms.extend(_split_large_text(segment, max_chars))

    chunks: list[str] = []
    current = ""
    for atom in atoms:
        if not atom.strip():
            continue
        candidate = f"{current}\n{atom}" if current else atom
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            chunks.append(current)
            current = atom
    if current:
        chunks.append(current)
    return chunks


def _split_large_text(text: str, max_chars: int) -> list[str]:
    """Recursively split text that exceeds max_chars.

    Tries to split at natural boundaries:
      1. Double newline (paragraph)
      2. Single newline (line)
      3. Sentence end
      4. Hard character split as last resort
    """
    if len(text) <= max_chars:
        return [text]

    for sep in ["\n\n", "\n", ". ", "? ", "! "]:
        parts = text.split(sep)
        if len(parts) < 2:
            continue

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > max_chars:
                    chunks.extend(_split_large_text(part, max_chars))
                    current = ""
                else:
                    current = part.strip()
        if current:
            chunks.append(current)

        if len(chunks) > 1:
            return [c for c in chunks if c.strip()]

    # Last resort: hard split
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _chunk_text(text: str, source: str) -> list[dict[str, str]]:
    """Split a sphinx .txt file into embeddable chunks.

    Each chunk gets a breadcrumb prefix:
        "KpointsData — AiiDA Topics > Data Types"

    The section title comes FIRST so the embedding is dominated by the topic,
    not the file path. No index-time prefix is added here; any embedding-side
    prefixing is handled by the embedding function.
    """
    sections = _extract_text_sections(text)
    chunks: list[dict[str, str]] = []

    # Build a readable topic label from the file path
    parts = Path(source).with_suffix("").parts  # drop .txt
    topic_label = (
        " > ".join(
            p.replace("_", " ").replace("-", " ").title()
            for p in parts[:-1]  # skip the filename itself
        )
        or "AiiDA"
    )

    for title, body in sections:
        if not body or len(body) < _MIN_CHUNK_CHARS:
            continue

        breadcrumb = (
            f"{title} — AiiDA {topic_label}" if title else f"AiiDA {topic_label}"
        )

        for sub in _split_fence_aware(body, _TARGET_CHUNK_CHARS):
            if len(sub.strip()) < _MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    "text": breadcrumb + "\n" + sub.strip(),
                    "source": source,
                    "section": title or "(preamble)",
                }
            )

    # Fallback for files with no headings
    if not chunks and len(text.strip()) >= _MIN_CHUNK_CHARS:
        chunks.append(
            {
                "text": f"AiiDA {topic_label}\n{text.strip()}",
                "source": source,
                "section": "(full file)",
            }
        )

    return chunks


def _load_docs(text_dir: str) -> list[dict[str, str]]:
    """Walk text_dir and return all chunks from .txt files.

    Files in ``_EXCLUDED_SOURCES`` are skipped so their content never reaches
    the index.
    """
    chunks: list[dict[str, str]] = []
    for path in Path(text_dir).rglob("*.txt"):
        rel = path.relative_to(text_dir).as_posix()
        if rel in _EXCLUDED_SOURCES:
            logger.debug("excluding %s from the index", rel)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(_chunk_text(text, source=rel))
        except Exception as exc:
            logger.warning("skipping %s: %s", path, exc)

    avg = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
    logger.info(
        "loaded %d chunks from %s  (avg %.0f chars/chunk)",
        len(chunks),
        text_dir,
        avg,
    )
    return chunks
