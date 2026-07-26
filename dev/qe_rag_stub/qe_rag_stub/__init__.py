"""Dev-only stand-in for the ``aiida_agents.plugins`` entry point
aiida-quantumespresso would eventually declare itself.

Registers aiida-quantumespresso's own docs as a RAG corpus, so
``search_aiida_docs`` has something QE-specific (e.g. what ``kpoints_distance``
means for ``PwBandsWorkChain``) to retrieve, and cross-corpus attribution
(``[quantumespresso: ...]``) can be exercised locally. Delete this package (and
uninstall it) once aiida-quantumespresso ships the real integration.
"""

from __future__ import annotations

from aiida_agents.plugins import RagCorpus

_QE_DOCS_VERSION = "5.0.0"  # pinned aiida-quantumespresso release tag (v5.0.0)


class _QuantumEspressoDocsStub:
    name = "quantumespresso"

    def rag_corpora(self) -> list[RagCorpus]:
        return [
            RagCorpus(
                name="quantumespresso",
                version=_QE_DOCS_VERSION,
                docs_repo="https://github.com/aiidateam/aiida-quantumespresso.git",
                docs_ref=f"v{_QE_DOCS_VERSION}",
                docs_subdir="docs",
            )
        ]


PROVIDER = _QuantumEspressoDocsStub()
