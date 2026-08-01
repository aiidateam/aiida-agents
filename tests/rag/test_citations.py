"""Tests for turning a retrieved passage into a link.

The anchor is reconstructed rather than recorded, so these pin it against the
rule sphinx actually uses (docutils' ``make_id``) --- including the cases where
a heading is not a tidy sentence, which is where a hand-rolled slugifier
diverges. The other half is what happens when a link cannot be built: a passage
must still come back, unlinked, rather than being dropped or given a guess.
"""

from __future__ import annotations

import pytest

from aiida_agents.rag.citations import (
    CORE_DOCS_TEMPLATE,
    anchor_for,
    citation_url,
    page_url,
)


class TestAnchors:
    """What sphinx will have called the heading."""

    @pytest.mark.parametrize(
        "section, expected",
        [
            pytest.param("Setting up a code", "setting-up-a-code", id="plain"),
            pytest.param("QueryBuilder", "querybuilder", id="camel-case"),
            pytest.param("What is a CalcJob?", "what-is-a-calcjob", id="punctuation"),
            pytest.param("Data types & nodes", "data-types-nodes", id="ampersand"),
            pytest.param("Using `verdi`", "using-verdi", id="backticks"),
            pytest.param("Step 1: install", "step-1-install", id="colon-and-digit"),
            pytest.param("Réglages", "reglages", id="accents-folded"),
            pytest.param("  spaced   out  ", "spaced-out", id="runs-collapse"),
        ],
    )
    def test_a_heading_becomes_its_html_id(self, section: str, expected: str) -> None:
        assert anchor_for(section) == expected

    def test_a_leading_digit_is_dropped_like_docutils_does(self) -> None:
        """An HTML id may not start with a digit, so docutils strips them."""
        assert anchor_for("2 ways to query") == "ways-to-query"

    @pytest.mark.parametrize("section", ["(preamble)", "(full file)"])
    def test_a_sentinel_section_yields_no_anchor(self, section: str) -> None:
        """These name a passage under no heading -- there is nothing to point at."""
        assert anchor_for(section) == ""

    def test_an_empty_section_yields_no_anchor(self) -> None:
        assert anchor_for("") == ""

    def test_a_heading_of_only_punctuation_yields_no_anchor(self) -> None:
        """Rather than a stray hyphen, which would be a broken fragment."""
        assert anchor_for("---") == ""


class TestCitationUrls:
    """The link a passage is cited with."""

    def test_a_core_passage_links_to_the_pinned_page_and_section(self) -> None:
        url = citation_url(
            CORE_DOCS_TEMPLATE, "howto/query", "Constructing a query", "v2.8.0"
        )

        assert url == (
            "https://aiida-core.readthedocs.io/en/v2.8.0/"
            "howto/query.html#constructing-a-query"
        )

    def test_the_version_is_the_one_the_corpus_was_built_at(self) -> None:
        """A citation must never point at a page the indexed text is not from."""
        url = citation_url(CORE_DOCS_TEMPLATE, "howto/query", "Q", "v2.9.0")

        assert "/en/v2.9.0/" in url  # type: ignore[operator]
        assert "latest" not in url  # type: ignore[operator]

    def test_a_passage_under_no_heading_links_to_the_bare_page(self) -> None:
        """Better an unanchored page than a fragment that goes nowhere."""
        url = citation_url(CORE_DOCS_TEMPLATE, "intro/about", "(preamble)", "v2.8.0")

        assert url == "https://aiida-core.readthedocs.io/en/v2.8.0/intro/about.html"

    def test_a_corpus_that_declares_no_url_is_not_given_one(self) -> None:
        """A plugin may contribute docs without publishing them anywhere."""
        assert citation_url(None, "howto/run", "Running", "1.0") is None

    def test_a_passage_with_no_source_is_not_given_one(self) -> None:
        assert citation_url(CORE_DOCS_TEMPLATE, "", "Running", "v2.8.0") is None

    def test_a_plugin_template_is_used_as_given(self) -> None:
        """The plugin says where its docs live; nothing here guesses."""
        template = (
            "https://aiida-quantumespresso.readthedocs.io/en/{version}/{page}.html"
        )

        url = citation_url(template, "user_guide/pw", "Inputs", "v5.0.0")

        assert url == (
            "https://aiida-quantumespresso.readthedocs.io/en/v5.0.0/"
            "user_guide/pw.html#inputs"
        )


class TestPageUrls:
    def test_a_nested_doc_id_keeps_its_path(self) -> None:
        url = page_url(CORE_DOCS_TEMPLATE, "topics/data_types", "v2.8.0")

        assert url.endswith("/topics/data_types.html")
