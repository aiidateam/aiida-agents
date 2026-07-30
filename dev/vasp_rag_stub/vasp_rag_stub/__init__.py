"""Dev-only stand-in for the ``aiida_agents.plugins`` entry point
aiida-vasp would eventually declare itself.

Sibling of ``dev/qe_rag_stub``, and the same bargain: aiida-vasp's *workflows*
need nothing from us --- installing the package registers its entry points, and
``list_workflows`` / ``describe_workflow`` / ``build_workflow_inputs`` read them
generically --- but its *documentation* is what tells an agent what an
``INCAR`` tag means or which protocol a ``VaspRelaxWorkChain`` expects, and
nothing indexes that unless someone declares the corpus.

Registering it here is also the second corpus in the profile, which is the point
of having two: it exercises cross-corpus attribution (``[vasp: ...]`` alongside
``[quantumespresso: ...]``) against a real second body of docs rather than a
fixture. Delete this package (and uninstall it) once aiida-vasp ships the real
integration.
"""

from __future__ import annotations

from aiida_agents.plugins import RagCorpus

_VASP_DOCS_VERSION = "5.1.0"  # pinned aiida-vasp release tag (v5.1.0)


class _VaspDocsStub:
    name = "vasp"

    def rag_corpora(self) -> list[RagCorpus]:
        return [
            RagCorpus(
                name="vasp",
                version=_VASP_DOCS_VERSION,
                docs_repo="https://github.com/aiida-vasp/aiida-vasp.git",
                docs_ref=f"v{_VASP_DOCS_VERSION}",
                # Holds the ``source/`` sphinx root the indexer expects.
                docs_subdir="docs",
            )
        ]


PROVIDER = _VaspDocsStub()
