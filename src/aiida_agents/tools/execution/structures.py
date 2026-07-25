"""Import a structure file into the profile as a ``StructureData`` node.

Every other execution tool takes a structure as a reference to a node that
already exists (``{"pk": N}``). That leaves the most common real request --
"relax *this* CIF" -- with no way in, so this is the one write tool that turns
a file on disk into a node the rest of the pipeline can reference.

Parsing is delegated to ASE, which reads CIF/POSCAR/xyz/... and is what
``StructureData(ase=...)`` consumes directly. ASE is an optional dependency
(``aiida-agents[structures]``): missing it is reported as an actionable error
rather than an ImportError traceback, the same way the docs toolchain is
handled in ``rag.indexing``.
"""

from __future__ import annotations

import importlib.util
import logging
import typing as t
from pathlib import Path

from aiida import orm
from pydantic import Field

from .._types import StructureImportResult

logger = logging.getLogger(__name__)

__all__ = ["import_structure"]


class StructureImportError(Exception):
    """A structure file could not be read or converted.

    Not an ``AiidaException``: the failure is about the file or the parsing
    toolchain, not a node lookup, so ``describe_aiida_error``'s identifier
    hints would misdescribe it.
    """


def import_structure(
    filepath: t.Annotated[
        str,
        Field(
            description=(
                "Path to the structure file to import, e.g. '/data/si.cif' or "
                "'POSCAR'. Read from the machine the agent runs on."
            )
        ),
    ],
    file_format: t.Annotated[
        str | None,
        Field(
            description=(
                "ASE format name ('cif', 'vasp', 'xyz', ...). Omit to let ASE "
                "infer it from the file extension, which is usually correct."
            )
        ),
    ] = None,
    label: t.Annotated[
        str | None,
        Field(description="Label for the stored node. Defaults to the filename."),
    ] = None,
) -> StructureImportResult:
    """Import a structure file (CIF, POSCAR, xyz, ...) as a stored StructureData.

    Use this when the user points at a structure *file* that is not in the
    database yet. It writes one node and returns its `pk`, which every other
    tool then takes as a `{"pk": N}` reference. If the structure is already in
    the profile, search for it with `query_nodes`/`search_structures` instead of
    importing a second copy.

    Args:
        filepath: Path to the structure file.
        file_format: ASE format name; inferred from the extension when omitted.
        label: Label for the stored node; defaults to the filename.

    Returns:
        The stored node's pk, uuid, formula, site count, and label.

    Raises:
        StructureImportError: The file is missing, ASE is not installed, or the
            file could not be parsed into a structure.
    """
    logger.debug(
        "import_structure(filepath=%r, file_format=%r, label=%r)",
        filepath,
        file_format,
        label,
    )

    path = Path(filepath).expanduser()
    # Check the file before the toolchain: a typo'd path is the far more common
    # mistake, and reporting it first is more useful than an install hint.
    if not path.is_file():
        msg = (
            f"No structure file at {str(path)!r}. Give a path to an existing "
            "file, or ask the user where the structure lives."
        )
        raise StructureImportError(msg)

    if importlib.util.find_spec("ase") is None:
        msg = (
            "Reading structure files needs ASE, which is not installed in this "
            "environment. Install it with:\n"
            "    uv pip install 'aiida-agents[structures]'\n"
            "then retry. (A structure already in the database needs no import -- "
            "find it with query_nodes or search_structures.)"
        )
        raise StructureImportError(msg)

    from ase.io import read as ase_read  # noqa: PLC0415  (optional dependency)

    try:
        atoms = ase_read(str(path), format=file_format)
    except Exception as exc:
        fmt = f"format {file_format!r}" if file_format else "the inferred format"
        msg = (
            f"Could not read {str(path)!r} using {fmt}: {exc}. "
            "Check the file is a supported structure format, or pass "
            "'file_format' explicitly."
        )
        raise StructureImportError(msg) from exc

    # ase.io.read returns a list when the file holds a trajectory; a structure
    # input is a single frame, so make the ambiguity explicit rather than
    # silently importing the first (or last) image.
    if isinstance(atoms, list):
        msg = (
            f"{str(path)!r} holds {len(atoms)} frames (a trajectory). Extract "
            "the single frame you want into its own file and import that."
        )
        raise StructureImportError(msg)

    try:
        structure = orm.StructureData(ase=atoms)
    except Exception as exc:
        msg = f"Could not convert {str(path)!r} into a StructureData: {exc}"
        raise StructureImportError(msg) from exc

    structure.label = label or path.name
    structure.store()

    logger.info("imported %s as StructureData<%s>", path, structure.pk)
    return {
        "pk": t.cast(int, structure.pk),
        "uuid": str(structure.uuid),
        # AiiDA's ORM is only partially typed; get_formula has no annotations.
        "formula": structure.get_formula(),  # type: ignore[no-untyped-call]
        "num_sites": len(structure.sites),
        "label": structure.label,
    }
