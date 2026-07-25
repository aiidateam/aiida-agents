"""Tests for ``import_structure``.

ASE is an optional dependency (``aiida-agents[structures]``) and is not
installed in the default test environment, so the paths that do not need it --
a missing file, and the missing-dependency error itself -- are asserted
unconditionally, while the real parse/store round trip is skipped unless ASE is
actually present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from aiida import orm

from aiida_agents.tools.execution.structures import (
    StructureImportError,
    import_structure,
)

_HAS_ASE = importlib.util.find_spec("ase") is not None

# A minimal 2-atom silicon CIF, enough for ASE to parse into a structure.
_SILICON_CIF = """data_silicon
_cell_length_a 3.8
_cell_length_b 3.8
_cell_length_c 3.8
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.0 0.0 0.0
Si2 0.5 0.5 0.5
"""


def test_missing_file_is_a_clean_error(tmp_path: Path) -> None:
    """A typo'd path reports the path, not an OSError traceback.

    Checked before the ASE probe on purpose: a wrong path is the far more
    common mistake, so it must not be masked by an install hint.
    """
    with pytest.raises(StructureImportError, match="No structure file at"):
        import_structure(str(tmp_path / "does_not_exist.cif"))


@pytest.mark.skipif(_HAS_ASE, reason="ASE installed; the missing-dep path cannot run")
def test_missing_ase_reports_the_extra_to_install(tmp_path: Path) -> None:
    """Without ASE the failure names the exact install command, not ImportError."""
    path = tmp_path / "si.cif"
    path.write_text(_SILICON_CIF)

    with pytest.raises(StructureImportError, match=r"aiida-agents\[structures\]"):
        import_structure(str(path))


@pytest.mark.skipif(not _HAS_ASE, reason="needs ASE to parse a structure file")
@pytest.mark.usefixtures("aiida_profile")
def test_imports_a_cif_as_a_stored_structure(tmp_path: Path) -> None:
    """A CIF becomes a stored StructureData whose pk is usable as a reference."""
    path = tmp_path / "si.cif"
    path.write_text(_SILICON_CIF)

    result = import_structure(str(path))

    assert result["num_sites"] == 2
    assert "Si" in result["formula"]
    assert result["label"] == "si.cif"
    # The pk is the point of the tool: it must load back as a stored structure,
    # which is what a submission's {"pk": N} reference resolves against.
    loaded = orm.load_node(result["pk"])
    assert isinstance(loaded, orm.StructureData)
    assert loaded.is_stored


@pytest.mark.skipif(not _HAS_ASE, reason="needs ASE to parse a structure file")
@pytest.mark.usefixtures("aiida_profile")
def test_explicit_label_overrides_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "si.cif"
    path.write_text(_SILICON_CIF)

    result = import_structure(str(path), label="my-silicon")

    assert result["label"] == "my-silicon"


@pytest.mark.skipif(not _HAS_ASE, reason="needs ASE to attempt a parse")
@pytest.mark.usefixtures("aiida_profile")
def test_unparseable_file_is_a_clean_error(tmp_path: Path) -> None:
    """A file ASE cannot read reports the path and suggests file_format."""
    path = tmp_path / "junk.cif"
    path.write_text("this is not a structure file")

    with pytest.raises(StructureImportError, match="Could not read"):
        import_structure(str(path))
