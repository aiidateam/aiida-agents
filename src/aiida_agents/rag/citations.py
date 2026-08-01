"""Turn a retrieved passage into a link the user can open.

Retrieval already knew where every passage came from: ``source`` is the sphinx
doc id it was rendered from (``howto/query``) and ``section`` is the heading it
sat under. What it did not do was say so in a form anyone could follow. An
answer citing ``[howto/query  §  Constructing a query]`` is asking the reader to
go and find that themselves --- and finding the right page is the thing users
say is hardest about the documentation in the first place.

Both halves of a URL are already on hand, so this builds one:

    howto/query  §  Constructing a query
    -> https://aiida-core.readthedocs.io/en/v2.8.0/howto/query.html#constructing-a-query

The version segment is the *same tag the corpus was rendered from*, not
``latest``. A citation that silently pointed at a newer page would be the worst
kind of wrong: the quoted text and the linked page disagreeing, with nothing on
screen to say why.

The anchor is reconstructed rather than recorded --- the sphinx text builder the
corpus is rendered with emits headings, not the HTML ids the web build assigns
them --- so :func:`anchor_for` reimplements docutils' ``make_id``. When it gets
one wrong the link still opens the correct *page*, which is the part that
matters; that degradation is deliberate, and it is why a wrong anchor is not
treated as a failure worth suppressing the whole link over.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["anchor_for", "citation_url", "page_url"]

#: Where the pinned aiida-core documentation is published. ``version`` is filled
#: from the tag the corpus was built at (``store._DOCS_TAG``), so the link and
#: the indexed text always describe the same release.
CORE_DOCS_TEMPLATE = "https://aiida-core.readthedocs.io/en/{version}/{page}.html"

#: Section labels that name no heading. ``chunking._chunk_text`` uses these for
#: a passage taken from a file's preamble, or from a file with no headings at
#: all. Neither has an anchor to point at, so the bare page is the honest link.
_UNANCHORABLE = frozenset({"(preamble)", "(full file)"})

# docutils' make_id, which sphinx uses to turn a heading into an HTML id:
# lowercase, strip accents, collapse every run of non-alphanumerics into a
# single hyphen, then drop leading digits/hyphens and trailing hyphens.
_NON_ID_CHARS = re.compile(r"[^a-z0-9]+")
_NON_ID_AT_ENDS = re.compile(r"^[-0-9]+|-+$")


def anchor_for(section: str) -> str:
    """The HTML id sphinx will have given a heading, or ``""`` if there is none.

    Args:
        section: The heading text as the text builder recorded it, or one of
            the ``(preamble)`` / ``(full file)`` sentinels.

    Returns:
        The anchor *without* a leading ``#``, or an empty string for a passage
        that sits under no heading.
    """
    if not section or section in _UNANCHORABLE:
        return ""

    folded = unicodedata.normalize("NFKD", section.lower())
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    slug = _NON_ID_CHARS.sub("-", " ".join(ascii_only.split()))
    return _NON_ID_AT_ENDS.sub("", slug)


def page_url(template: str, source: str, version: str) -> str:
    """The documentation page a doc id was rendered to.

    Args:
        template: A URL with ``{version}`` and ``{page}`` placeholders.
        source: The extensionless sphinx doc id, e.g. ``howto/query``.
        version: The docs ref the corpus was built at, e.g. ``v2.8.0``.
    """
    return template.format(version=version, page=source)


def citation_url(
    template: str | None, source: str, section: str, version: str
) -> str | None:
    """The deep link for one retrieved passage, or ``None`` if it has no home.

    ``template`` is ``None`` for a corpus that never said where it is published
    --- a plugin may contribute documentation without hosting it anywhere. That
    is a normal case, not an error: the passage is still returned, just without
    a link.

    Args:
        template: The corpus's URL template, or ``None`` if it declares none.
        source: The extensionless sphinx doc id.
        section: The heading the passage sat under.
        version: The docs ref the corpus was built at.

    Returns:
        An absolute URL, anchored at the section where one could be derived.
    """
    if not template or not source:
        return None

    url = page_url(template, source, version)
    anchor = anchor_for(section)
    return f"{url}#{anchor}" if anchor else url
