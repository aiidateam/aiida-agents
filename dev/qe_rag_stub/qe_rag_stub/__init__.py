"""Dev-only stand-in for the ``aiida_agents.plugins`` entry point
aiida-quantumespresso would eventually declare itself.

Contributes two things, one of each kind the plugin protocol supports:

*A documentation corpus.* aiida-quantumespresso's own docs, so
``search_aiida_docs`` has something QE-specific (e.g. what ``kpoints_distance``
means for ``PwBandsWorkChain``) to retrieve and cross-corpus attribution
(``[quantumespresso: ...]``) can be exercised locally.

*A tool.* ``read_scf_convergence``, which reads pw.x's own electronic
convergence trace. It lives here rather than in ``aiida-agents`` because it
parses Quantum ESPRESSO's output format, and nothing in the core tool layer
knows about any particular code (ADR-10). This is also the first exercise of
the ``tools()`` hook: the mechanism was specified and documented before
anything used it.

Delete this package (and uninstall it) once aiida-quantumespresso ships the
real integration.
"""

from __future__ import annotations

from aiida_agents.plugins import AgentTool, RagCorpus

from qe_rag_stub.convergence import read_scf_convergence

_QE_DOCS_VERSION = "5.0.0"  # pinned aiida-quantumespresso release tag (v5.0.0)

_PROMPT_FRAGMENT = """\
Quantum ESPRESSO calculations that fail to reach self-consistency are the most
common failure you will be asked about. `diagnose_process_failure` tells you the
exit code and the remedies the work chain registers; it cannot tell you which of
those remedies fits, because that depends on how the SCF cycle behaved. Call
`read_scf_convergence` on the calculation (the `retrieved_files_pk` the
diagnosis reports) before recommending one.

A cycle that was still decreasing and simply ran out of iterations wants a
larger `electron_maxstep`. One that oscillated or stalled does not -- more
iterations of a cycle that is not settling changes nothing, and the answer lies
with mixing or smearing instead. Say which of the two you observed, and quote
the accuracy and threshold you were given rather than characterising them.

Check `ionic_steps` before you generalise. A relax runs one SCF cycle per ionic
step, and `trend` describes only the last one. "The SCF oscillated" is wrong
about a relaxation whose earlier cycles each converged cleanly -- look at
`cycles` and say which step behaved how.\
"""


class _QuantumEspressoStub:
    name = "quantumespresso"

    def rag_corpora(self) -> list[RagCorpus]:
        return [
            RagCorpus(
                name="quantumespresso",
                version=_QE_DOCS_VERSION,
                docs_repo="https://github.com/aiidateam/aiida-quantumespresso.git",
                docs_ref=f"v{_QE_DOCS_VERSION}",
                docs_subdir="docs",
                # Where the same docs are published, so a hit from this corpus
                # is cited with a link. {version} is filled from docs_ref, so
                # the page linked is the one the corpus was rendered from.
                docs_url="https://aiida-quantumespresso.readthedocs.io/en/{version}/{page}.html",
            )
        ]

    def tools(self) -> list[AgentTool]:
        # Read-only: it opens a retrieved file and returns numbers, so it needs
        # no approval gate and is wrapped for recoverable retries like any other
        # read tool.
        return [AgentTool(fn=read_scf_convergence, writes=False)]

    def prompt_fragment(self) -> str:
        return _PROMPT_FRAGMENT


PROVIDER = _QuantumEspressoStub()
