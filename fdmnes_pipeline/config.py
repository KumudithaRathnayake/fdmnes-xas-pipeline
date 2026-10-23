"""Standardized FDMNES calculation protocol (plan stage 2).

The idea, from the project plan: rather than re-deciding Radius / Green vs
FDM / SCF / PBE96 / Quadrupole / energy range for every structure, settle on
one set of "common simulation settings" once they've been validated against
a known-good calculation (the earlier working Co-HHTP run: Z_absorber 27,
Radius 5.0, this energy grid, Convolution on), and then reuse it for every
structural model being compared.

Edit DEFAULT_PROTOCOL below once you've locked in your standard settings.
Per-job overrides are still possible (see input_generator.generate_batch).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


# Absorber element -> atomic number. This is deliberately just the elements
# you've actually used as an *absorber* (not a full periodic table -- for
# parsing every atom in an .xyz file's ligand/framework, see
# xyz_parser.ELEMENT_Z). Extend as needed for new absorbers.
ATOMIC_NUMBERS = {
    "Ni": 28,
    "Co": 27,
    "Cu": 29,
    "Zn": 30,
}


@dataclass(frozen=True)
class ProtocolSettings:
    """The physics/calculation settings that should be common across structures.

    The first block (radius through use_convolution) is the original
    K-edge/CIF protocol. The second block are extras that showed up in your
    L-edge (Cu/Zn-TAPP-pyridine) reference inputs -- each one is a no-op
    (the corresponding FDMNES keyword is simply omitted) unless you set it,
    so turning them on/off never changes output for jobs that don't use them.
    """

    radius: float = 5.0
    energy_range: str = "-5.0 0.2 10.0 0.5 25.0 1.0 60.0 2.0 100.0"
    use_green: bool = False
    use_pbe96: bool = False
    use_scf: bool = False
    use_quadrupole: bool = False
    use_convolution: bool = True

    # L-edge-oriented extras, all optional/off by default.
    use_tddft: bool = False  # "TDDFT" -- often needed for L23 edges of 3d elements
    use_spinorbit: bool = False  # "Spinorbit" -- heavy elements / L edges
    vmax: Optional[float] = None  # "Vmax" <value> -- cap the outer potential (molecules not in gas phase)
    estart: Optional[float] = None  # "Estart" <value> -- lower starting energy for the convoluted output
    overad: Optional[float] = None  # "Overad" <value> -- as used in your ZnTAPP reference input

    def with_overrides(self, **kwargs) -> "ProtocolSettings":
        """Return a copy with specific fields overridden, e.g. for a one-off
        radius-convergence test without touching the shared default."""
        return replace(self, **kwargs)


@dataclass(frozen=True)
class AbsorberSettings:
    """Which atom(s) FDMNES should treat as the absorber for a given job,
    and which edge/threshold to probe."""

    element: str
    mode: str = "all"  # 'all' -> Z_absorber, 'selected' -> Absorber <index>
    index: int = 1
    edge: Optional[str] = None  # None -> FDMNES default (K edge); else "L23", "L1", "M45", ...

    def __post_init__(self) -> None:
        if self.element not in ATOMIC_NUMBERS:
            raise ValueError(
                f"Unknown element {self.element!r}; add it to ATOMIC_NUMBERS "
                f"in config.py if it's a valid FDMNES absorber."
            )
        if self.mode not in {"all", "selected"}:
            raise ValueError("mode must be 'all' or 'selected'")
        if self.mode == "selected" and (not isinstance(self.index, int) or self.index < 1):
            raise ValueError("index must be a positive integer when mode='selected'")
        if self.edge is not None and not self.edge.strip():
            raise ValueError("edge must be a non-empty string (e.g. 'L23') if provided")

    @property
    def atomic_number(self) -> int:
        return ATOMIC_NUMBERS[self.element]


# The protocol validated on the earlier working Co-HHTP calculation
# (Z_absorber 27, Radius 5.0, Convolution). Treat this as frozen once you've
# confirmed the Ni-HHTP generator output reproduces a normal manual FDMNES
# run -- see plan stage 3/4.
DEFAULT_PROTOCOL = ProtocolSettings()
