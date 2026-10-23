"""Shared data structures used across the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class RunStatus(str, Enum):
    """Outcome of a single FDMNES calculation, per plan stage 5."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    WARNING = "WARNING"  # ran to completion, but something looked off
    NOT_RUN = "NOT_RUN"


# Dummy unit cell FDMNES's "Molecule" block still wants even for an isolated
# (non-periodic) molecule. Both the CuTAPP and ZnTAPP reference inputs use
# this exact placeholder cell.
DEFAULT_MOLECULE_CELL: Tuple[float, float, float, float, float, float] = (
    1.0, 1.0, 1.0, 90.0, 90.0, 90.0,
)


@dataclass
class MoleculeStructure:
    """An isolated-molecule structure for FDMNES's "Molecule" block, as an
    alternative to a CIF file (see xyz_parser.parse_xyz to build one from a
    .xyz file, e.g. the kind ASE produces).
    """

    atoms: List[Tuple[int, float, float, float]]  # (atomic_number, x, y, z) in Angstrom
    cell: Tuple[float, float, float, float, float, float] = DEFAULT_MOLECULE_CELL
    source_xyz: Optional[Path] = None  # for provenance only; not required

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("MoleculeStructure needs at least one atom")
        if self.source_xyz is not None:
            self.source_xyz = Path(self.source_xyz)


@dataclass
class JobSpec:
    """One FDMNES calculation to generate (and optionally run).

    Mirrors the "USER SETTINGS" cell of the original notebook plus the
    commented-out batch example, but as a reusable, validated object
    instead of module-level globals.

    Structure input is either a CIF file (``structure_file``, the original
    Ni-HHTP/Co-HHTP workflow) or an isolated-molecule structure
    (``molecule``, for the CuTAPP/ZnTAPP-pyridine-style jobs built from
    .xyz files) -- set exactly one of the two, never both.

    ``edge`` selects the FDMNES threshold (e.g. "L23", "L1", "M45"). Leave
    it as None for the default K edge -- that's what keeps existing K-edge
    jobs byte-identical to before.
    """

    name: str
    element: str  # e.g. 'Ni', 'Co', 'Cu', 'Zn'
    filout: Path
    input_file: Path
    structure_file: Optional[Path] = None
    molecule: Optional[MoleculeStructure] = None
    absorber_mode: str = "all"  # 'all' or 'selected'
    absorber_index: int = 1
    edge: Optional[str] = None  # None -> FDMNES default (K); else e.g. "L23"

    def __post_init__(self) -> None:
        if (self.structure_file is None) == (self.molecule is None):
            raise ValueError(
                f"JobSpec {self.name!r} must set exactly one of structure_file "
                f"(a CIF path) or molecule (a MoleculeStructure), not both or neither."
            )
        if self.structure_file is not None:
            self.structure_file = Path(self.structure_file)
        self.filout = Path(self.filout)
        self.input_file = Path(self.input_file)


@dataclass
class RunResult:
    """Result of attempting to execute one job through FDMNES."""

    job: JobSpec
    returncode: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    fdmnes_error_file: Optional[Path] = None
    timed_out: bool = False
    launched: bool = False


@dataclass
class ValidationResult:
    """Result of inspecting a job's output files after (or instead of) a run."""

    job_name: str
    status: RunStatus
    reasons: list = field(default_factory=list)
    bav_file: Optional[Path] = None
    conv_file: Optional[Path] = None
    raw_spectrum_file: Optional[Path] = None

    def add_reason(self, text: str) -> None:
        self.reasons.append(text)

    def summary_line(self) -> str:
        return f"{self.job_name} -> {self.status.value}"
