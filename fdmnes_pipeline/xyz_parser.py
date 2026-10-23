"""Parse a standard .xyz file into a MoleculeStructure for FDMNES's "Molecule" block.

Your CuTAPP-pyridine / ZnTAPP-pyridine reference input files were built
from .xyz structures -- their own header comments say "generated from XYZ
using ASE". This module does the equivalent step in plain Python, with no
`ase` dependency, so the codebase stays install-free on the FDMNES machine.

Standard XYZ format assumed (the de facto convention ASE and everything
else writes):

    <N>
    <comment line, ignored>
    <Symbol> <x> <y> <z>
    ... N such lines ...

Coordinates are assumed to already be in Angstrom (FDMNES's expected unit),
matching what ASE's .xyz writer produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .models import DEFAULT_MOLECULE_CELL, MoleculeStructure

# Full periodic table, 1-118, so any ligand/framework atom in a real .xyz
# file resolves -- not just the elements you've used as absorbers so far
# (that shorter, absorber-only list is config.ATOMIC_NUMBERS).
_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
]
ELEMENT_Z = {symbol.upper(): z for z, symbol in enumerate(_SYMBOLS, start=1)}


def _resolve_z(token: str, line_no: int, path: Path) -> int:
    if token.isdigit():
        return int(token)
    symbol = "".join(ch for ch in token if ch.isalpha())
    z = ELEMENT_Z.get(symbol.upper())
    if z is None:
        raise ValueError(
            f"{path}: unrecognized element symbol {token!r} on line {line_no}"
        )
    return z


def parse_xyz(
    path: Path,
    cell: Tuple[float, float, float, float, float, float] = DEFAULT_MOLECULE_CELL,
) -> MoleculeStructure:
    """Read a standard .xyz file and return a MoleculeStructure.

    ``cell`` is the dummy unit cell FDMNES's "Molecule" block still wants
    even for an isolated (non-periodic) molecule -- your CuTAPP and ZnTAPP
    reference files both use the same placeholder,
    "1.0 1.0 1.0 90.0 90.0 90.0", which is the default here too.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is too short to be a valid .xyz file")

    try:
        n_atoms = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: first line must be the atom count") from exc

    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"{path}: header says {n_atoms} atoms but only "
            f"{len(atom_lines)} non-blank atom lines were found"
        )

    atoms = []
    for i, line in enumerate(atom_lines[:n_atoms], start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}: line {i} doesn't look like 'Symbol x y z': {line!r}")
        z = _resolve_z(parts[0], i, path)
        x, y, zc = (float(v) for v in parts[1:4])
        atoms.append((z, x, y, zc))

    return MoleculeStructure(atoms=atoms, cell=cell, source_xyz=path)
