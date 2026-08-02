"""Whether a forum question is one this agent could ever answer.

Written after looking at real scraped threads, which turned out not to be what
anyone assumed. AiiDA's forum is where people go when something is *broken*:
SSH transport, schedulers, installation, plugin development. The agent is built
for a different job --- explore my data, diagnose a failed run, set up a
calculation, explain a concept --- and the two overlap far less than the phrase
"real user questions" suggests.

That matters for measurement. Scoring every thread against one bar produces a
number that mostly says "the agent cannot configure SSH", which is true,
already known, and not what anyone wants to track. Worse, the number would then
move for reasons unrelated to any change we make.

So each case is labelled, and the two kinds are scored against different bars:

``IN_SCOPE``
    Answerable from the database, the documentation, or a workflow's own
    definition. Bar: did it actually answer.

``OUT_OF_SCOPE``
    Infrastructure and environment. Bar: did it *decline honestly* --- because
    the real failure here is confidently inventing an answer, not admitting a
    limit.

``UNCLEAR``
    Neither set of signals fired. Reported separately rather than guessed at:
    a heuristic that quietly assigns its uncertain cases pollutes both numbers,
    and how many land here is itself worth knowing.
"""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass

__all__ = ["IN_SCOPE", "OUT_OF_SCOPE", "UNCLEAR", "ScopeVerdict", "classify_scope"]

IN_SCOPE = "in_scope"
OUT_OF_SCOPE = "out_of_scope"
UNCLEAR = "unclear"

#: Signals that a thread is about running or configuring AiiDA rather than
#: about the science recorded in it. Checked first and decisive: a transport
#: question routinely mentions "calculation", so a thread naming both is
#: overwhelmingly the infrastructure one.
_OUT_OF_SCOPE_SIGNALS = (
    # Getting data on and off a machine
    "sftp",
    "scp",
    "rsync",
    "ssh",
    "transport",
    "verdi computer",
    # Batch systems
    "scheduler",
    "slurm",
    "torque",
    "pbs",
    "lsf",
    "hyperqueue",
    "allocation",
    "queue",
    # Installing and running the stack itself
    "pip install",
    "conda",
    "rabbitmq",
    "postgres",
    "postgresql",
    "daemon",
    "verdi setup",
    "verdi quicksetup",
    "installation",
    "virtualenv",
    "docker",
    # Developing against AiiDA rather than using it
    "subclass",
    "entry point",
    "plugin development",
    "writing a workchain",
    "migration",
    "upgrade aiida",
)

#: Signals that a thread is about the user's own data, or about how AiiDA works
#: conceptually --- the questions the agent holds tools for.
_IN_SCOPE_SIGNALS = (
    "querybuilder",
    "query",
    "provenance",
    "exit code",
    "exit_status",
    "failed calculation",
    "restart",
    "group",
    "extras",
    "attributes",
    "structuredata",
    "pseudopotential",
    "cutoff",
    "kpoints",
    "protocol",
    "output node",
    "input node",
    "descendant",
    "ancestor",
    "calcjob",
    "workchain node",
    "what is a",
    "difference between",
)


@dataclass(frozen=True)
class ScopeVerdict:
    """Which bar a case should be scored against, and what decided it.

    ``signals`` exists so a disputed label can be audited without re-reading
    the thread: a classifier this blunt will get some wrong, and hiding *why*
    would make the wrong ones impossible to find.
    """

    scope: str
    signals: tuple[str, ...]


def _matches(haystack: str, needles: t.Sequence[str]) -> tuple[str, ...]:
    """Which needles appear in ``haystack``, as whole words where meaningful.

    Word boundaries matter more than they look: "queue" inside "queued" is a
    real signal, but "group" inside "grouping" is not the AiiDA concept, and
    substring matching turns half the corpus out-of-scope.
    """
    return tuple(
        needle
        for needle in needles
        if re.search(rf"(?<![a-z]){re.escape(needle)}", haystack)
    )


def classify_scope(
    title: str, question: str, tags: t.Sequence[str] = ()
) -> ScopeVerdict:
    """Label a thread by whether the agent could answer it at all.

    Only the title, the asking post and the tags are read --- never the
    accepted answer. Classifying from the answer would leak the thing being
    measured: a thread would be called in-scope precisely when someone had
    already solved it in a way the agent could reproduce.

    Args:
        title: The thread title.
        question: The asking post, as text.
        tags: The thread's Discourse tags, if any.

    Returns:
        A :class:`ScopeVerdict`. ``UNCLEAR`` when neither set of signals fired.
    """
    haystack = " ".join([title, question, " ".join(tags)]).lower()

    out_of_scope = _matches(haystack, _OUT_OF_SCOPE_SIGNALS)
    if out_of_scope:
        return ScopeVerdict(OUT_OF_SCOPE, out_of_scope)

    in_scope = _matches(haystack, _IN_SCOPE_SIGNALS)
    if in_scope:
        return ScopeVerdict(IN_SCOPE, in_scope)

    return ScopeVerdict(UNCLEAR, ())
