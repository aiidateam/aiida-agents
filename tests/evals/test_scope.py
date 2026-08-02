"""Tests for labelling a forum thread by whether the agent could answer it.

The cases marked "real" below are taken verbatim from the first live scrape.
They are here because they are what prompted the split: two of the first three
threads pulled from the forum were about SSH transport and batch schedulers,
which no amount of prompt work would let this agent answer.

What matters most is the direction of a mistake. Labelling an infrastructure
thread in-scope reports a failure the agent could never have avoided; labelling
a data question out-of-scope lets a real miss pass as honest refusal. The
second is worse, so the out-of-scope signals are narrow and specific.
"""

from __future__ import annotations

import pytest

from tests.evals.scope import (
    IN_SCOPE,
    OUT_OF_SCOPE,
    UNCLEAR,
    classify_scope,
)


def _scope(title: str, question: str = "", tags: list[str] | None = None) -> str:
    return classify_scope(title, question, tags or []).scope


class TestRealThreads:
    """Threads from the first live scrape, labelled as they must be."""

    def test_an_sftp_transport_thread_is_out_of_scope(self) -> None:
        assert (
            _scope(
                "How to use scp/rsync instead of sftp for transport",
                "I am setting up the computer within AiiDA for a remote HPC",
            )
            == OUT_OF_SCOPE
        )

    def test_a_scheduler_plugin_thread_is_out_of_scope(self) -> None:
        assert (
            _scope(
                "Kill job allocation when workflow ends",
                "I have been working on a persistent scheduler plugin",
            )
            == OUT_OF_SCOPE
        )


class TestOutOfScope:
    @pytest.mark.parametrize(
        "title",
        [
            "Cannot connect over ssh to my cluster",
            "verdi computer setup fails",
            "Slurm jobs stay queued forever",
            "pip install aiida-core breaks",
            "RabbitMQ connection refused",
            "The daemon will not start",
            "How do I write a plugin entry point?",
            "postgres password authentication failed",
            "Docker image cannot find the profile",
        ],
    )
    def test_infrastructure_questions_are_out_of_scope(self, title: str) -> None:
        assert _scope(title) == OUT_OF_SCOPE

    def test_a_tag_alone_is_enough(self) -> None:
        """Discourse tags carry the topic even when the prose does not."""
        assert _scope("Something odd happening", "", ["scheduler"]) == OUT_OF_SCOPE


class TestInScope:
    @pytest.mark.parametrize(
        "title",
        [
            "How do I query for failed calculations?",
            "Filtering nodes by extras",
            "What is a CalcJobNode?",
            "Finding the descendant nodes of a structure",
            "Which cutoff does the protocol use?",
            "Difference between a WorkChain and a CalcJob",
            "Getting the exit code of a finished process",
        ],
    )
    def test_data_and_concept_questions_are_in_scope(self, title: str) -> None:
        assert _scope(title) == IN_SCOPE


class TestPrecedence:
    """A thread naming both is overwhelmingly the infrastructure one."""

    def test_infrastructure_wins_over_an_incidental_mention(self) -> None:
        """A transport thread mentions "calculation" almost every time."""
        assert (
            _scope(
                "sftp transport fails",
                "my calculation cannot copy its files back from the cluster",
            )
            == OUT_OF_SCOPE
        )

    def test_a_pure_data_question_is_not_dragged_out_of_scope(self) -> None:
        """The mistake that would matter: a real miss excused as honest refusal."""
        assert (
            _scope(
                "How do I find all structures in a group?",
                "I want to query my provenance graph for StructureData in a group",
            )
            == IN_SCOPE
        )


class TestUnclear:
    """Neither signal fired, and guessing would pollute both numbers."""

    def test_an_unrecognised_thread_is_unclear(self) -> None:
        assert _scope("A strange thing happened yesterday", "Any ideas?") == UNCLEAR

    def test_unclear_carries_no_signals(self) -> None:
        assert classify_scope("Hello", "Hi everyone").signals == ()


class TestAuditability:
    """A blunt classifier gets some wrong; the wrong ones must be findable."""

    def test_the_matched_signals_are_reported(self) -> None:
        verdict = classify_scope("sftp and slurm trouble", "")

        assert set(verdict.signals) >= {"sftp", "slurm"}

    def test_word_boundaries_are_respected(self) -> None:
        """Substring matching turns half the corpus out-of-scope by accident."""
        assert _scope("Regrouping my results by formula") != OUT_OF_SCOPE


class TestTheAnswerIsNeverRead:
    """Classifying from the accepted answer would leak what is being measured.

    A thread would be called in-scope precisely when somebody had already
    solved it in a way the agent could reproduce, which makes the in-scope
    score self-fulfilling.
    """

    def test_the_signature_takes_no_answer(self) -> None:
        import inspect

        assert "answer" not in inspect.signature(classify_scope).parameters
