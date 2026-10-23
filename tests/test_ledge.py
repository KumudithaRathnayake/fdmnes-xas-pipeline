"""Tests for the L-edge / molecule-from-XYZ extension.

Covers: xyz_parser, AbsorberSettings.edge / ProtocolSettings extras,
build_fdmnes_input's "Molecule" branch, and batch_config's "xyz_file" field.
Also includes a round-trip regression test against a small synthetic
demo molecule (an idealized octahedral Cu(H2O)6, tests/fixtures/
demo_Cu_hexaaqua.xyz -- invented coordinates, not a real/published
structure) to pin down the exact Molecule-block text format.
"""

from pathlib import Path

import pytest

from fdmnes_pipeline.config import AbsorberSettings, ProtocolSettings
from fdmnes_pipeline.input_generator import build_fdmnes_input, generate_input_file
from fdmnes_pipeline.models import JobSpec, MoleculeStructure
from fdmnes_pipeline.xyz_parser import parse_xyz

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# xyz_parser
# ---------------------------------------------------------------------------


def _write_xyz(tmp_path, text):
    path = tmp_path / "mol.xyz"
    path.write_text(text)
    return path


def test_parse_xyz_basic(tmp_path):
    path = _write_xyz(
        tmp_path,
        "3\ncomment line\nCu 0.0 0.0 0.0\nC 1.0 0.0 0.0\nH 1.5 0.5 0.0\n",
    )
    molecule = parse_xyz(path)

    assert molecule.atoms == [
        (29, 0.0, 0.0, 0.0),
        (6, 1.0, 0.0, 0.0),
        (1, 1.5, 0.5, 0.0),
    ]
    assert molecule.cell == (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    assert molecule.source_xyz == path


def test_parse_xyz_custom_cell(tmp_path):
    path = _write_xyz(tmp_path, "1\ncomment\nZn 0.0 0.0 0.0\n")
    molecule = parse_xyz(path, cell=(2.0, 2.0, 2.0, 90.0, 90.0, 90.0))
    assert molecule.cell == (2.0, 2.0, 2.0, 90.0, 90.0, 90.0)


def test_parse_xyz_accepts_numeric_atomic_number(tmp_path):
    path = _write_xyz(tmp_path, "1\ncomment\n29 0.0 0.0 0.0\n")
    molecule = parse_xyz(path)
    assert molecule.atoms == [(29, 0.0, 0.0, 0.0)]


def test_parse_xyz_unknown_symbol_raises(tmp_path):
    path = _write_xyz(tmp_path, "1\ncomment\nXx 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="unrecognized element symbol"):
        parse_xyz(path)


def test_parse_xyz_atom_count_mismatch_raises(tmp_path):
    path = _write_xyz(tmp_path, "5\ncomment\nC 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="header says 5 atoms"):
        parse_xyz(path)


def test_parse_xyz_too_short_raises(tmp_path):
    path = _write_xyz(tmp_path, "1\n")
    with pytest.raises(ValueError, match="too short"):
        parse_xyz(path)


# ---------------------------------------------------------------------------
# config: edge + new absorbers + protocol extras
# ---------------------------------------------------------------------------


def test_absorber_settings_accepts_cu_and_zn():
    assert AbsorberSettings(element="Cu").atomic_number == 29
    assert AbsorberSettings(element="Zn").atomic_number == 30


def test_absorber_settings_edge_defaults_to_none():
    assert AbsorberSettings(element="Ni").edge is None


def test_absorber_settings_rejects_blank_edge():
    with pytest.raises(ValueError):
        AbsorberSettings(element="Cu", edge="   ")


def test_protocol_settings_l_edge_extras_default_off():
    protocol = ProtocolSettings()
    assert protocol.use_tddft is False
    assert protocol.use_spinorbit is False
    assert protocol.vmax is None
    assert protocol.estart is None
    assert protocol.overad is None


# ---------------------------------------------------------------------------
# build_fdmnes_input: Molecule branch + new keywords
# ---------------------------------------------------------------------------


def test_build_fdmnes_input_molecule_uses_molecule_keyword_and_ledge_header():
    molecule = MoleculeStructure(atoms=[(29, 0.0, 0.0, 0.0), (6, 1.0, 0.0, 0.0)])
    absorber = AbsorberSettings(element="Cu", mode="all", edge="L23")
    protocol = ProtocolSettings(
        radius=7.5,
        use_green=True,
        vmax=-6.0,
        use_tddft=True,
        use_spinorbit=True,
    )

    text = build_fdmnes_input(molecule, absorber, Path("/out/Cu_out"), protocol)

    assert text.startswith("! FDMNES input generated from XYZ using Python\n")
    assert "Molecule\n  1.0 1.0 1.0 90.0 90.0 90.0\n" in text
    assert "    29   0.00000000   0.00000000   0.00000000\n" in text
    assert "Edge\n  L23\n" in text
    assert "Vmax\n  -6.0\n" in text
    assert "TDDFT\n" in text
    assert "Spinorbit\n" in text
    assert "Cif_file" not in text


def test_build_fdmnes_input_estart_and_overad():
    molecule = MoleculeStructure(atoms=[(30, 0.0, 0.0, 0.0)])
    absorber = AbsorberSettings(element="Zn", edge="L23")
    protocol = ProtocolSettings(use_tddft=True, estart=-15.0, overad=1.2, use_convolution=True)

    text = build_fdmnes_input(molecule, absorber, Path("/out/Zn_out"), protocol)

    assert "Overad\n  1.2\n" in text
    assert "Estart\n  -15.0\n" in text
    # Estart appears after the structure block and before Convolution, matching
    # your ZnTAPP reference input's convolution-stage placement.
    assert text.index("Molecule") < text.index("Estart") < text.index("Convolution")


def test_build_fdmnes_input_cif_mode_unaffected_by_new_fields():
    """The whole point: a plain CIF/K-edge job's output is byte-identical to
    before, even though the function now supports molecules/edges too."""
    absorber = AbsorberSettings(element="Ni", mode="all")  # edge=None
    protocol = ProtocolSettings(radius=7.5)  # all L-edge extras default off

    text = build_fdmnes_input(Path("/x/a.cif"), absorber, Path("/x/out"), protocol)

    assert text.startswith("! Ni-HHTP FDMNES input generated by Python\n")
    assert "Edge" not in text
    assert "Vmax" not in text
    assert "TDDFT" not in text
    assert "Overad" not in text
    assert "Spinorbit" not in text
    assert "Estart" not in text
    assert "Cif_file\n  /x/a.cif\n" in text


# ---------------------------------------------------------------------------
# JobSpec validation
# ---------------------------------------------------------------------------


def test_jobspec_requires_exactly_one_structure_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        JobSpec(
            name="bad",
            element="Cu",
            filout=tmp_path / "out",
            input_file=tmp_path / "in.txt",
            # neither structure_file nor molecule set
        )
    with pytest.raises(ValueError, match="exactly one"):
        JobSpec(
            name="bad",
            element="Cu",
            filout=tmp_path / "out",
            input_file=tmp_path / "in.txt",
            structure_file=tmp_path / "a.cif",
            molecule=MoleculeStructure(atoms=[(29, 0.0, 0.0, 0.0)]),
        )


def test_jobspec_molecule_mode_generates_without_missing_structure_warning(tmp_path, capsys):
    job = JobSpec(
        name="mol_job",
        element="Cu",
        edge="L23",
        filout=tmp_path / "out" / "Cu_out",
        input_file=tmp_path / "out" / "Cu_input.txt",
        molecule=MoleculeStructure(atoms=[(29, 0.0, 0.0, 0.0)]),
    )
    generate_input_file(job, ProtocolSettings())

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
    assert job.input_file.exists()


# ---------------------------------------------------------------------------
# Regression: round-trip against a synthetic demo molecule
# ---------------------------------------------------------------------------


def test_demo_hexaaqua_roundtrip_matches_reference_atoms():
    """tests/fixtures/demo_Cu_hexaaqua.xyz is an idealized, invented
    octahedral Cu(H2O)6 (Cu-O = 2.10 A, ideal geometry) -- not a real or
    published structure, just a fixed, known set of coordinates to pin
    down the exact Molecule-block text format byte-for-byte. Parsing it
    back and regenerating should reproduce every atom exactly."""
    xyz_path = FIXTURES / "demo_Cu_hexaaqua.xyz"
    molecule = parse_xyz(xyz_path)

    assert len(molecule.atoms) == 19
    assert molecule.atoms[0] == (29, 0.0, 0.0, 0.0)
    assert molecule.atoms[-1] == (1, -0.75906199, 0.0, -2.68772859)
    # Cu, O, H -- exactly the elements present in this demo molecule
    assert {z for z, *_ in molecule.atoms} == {29, 8, 1}

    absorber = AbsorberSettings(element="Cu", edge="L23")
    protocol = ProtocolSettings(
        radius=7.5,
        energy_range="-10. 0.1 15. 0.2 35. 0.5 60.",
        use_green=True,
        vmax=-6.0,
        use_tddft=True,
        use_spinorbit=True,
    )
    text = build_fdmnes_input(molecule, absorber, Path("Sim/Demo_Cu_hexaaqua"), protocol)

    # Every atom row present, formatted consistently -- spot-check first and last.
    assert "    29   0.00000000   0.00000000   0.00000000\n" in text
    assert "     1   -0.75906199   0.00000000   -2.68772859\n" in text

    molecule_block = text.split("Molecule\n", 1)[1].split("\n\n", 1)[0]
    atom_rows = [line for line in molecule_block.splitlines()[1:] if line.strip()]
    assert len(atom_rows) == 19  # cell line excluded, every atom present exactly once

    assert "Z_absorber\n  29\n" in text
    assert "Edge\n  L23\n" in text
