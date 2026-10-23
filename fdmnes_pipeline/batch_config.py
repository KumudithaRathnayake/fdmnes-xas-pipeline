"""Load a batch of JobSpecs + a shared ProtocolSettings from a JSON file.

Deliberately JSON + stdlib only (no PyYAML dependency), since this codebase
needs to run unmodified on whichever machine actually has FDMNES installed.
See ``examples/jobs.json`` for a CIF/K-edge example and
``examples/jobs_ledge.json`` for an .xyz/L-edge one. Shape:

{
  "protocol": { "radius": 5.0, "energy_range": "...", "use_scf": false, ... },
  "jobs": [
    {
      "name": "Ni_HHTP_pristine",
      "structure_file": "/path/to/Ni_HHTP_pristine.cif",
      "element": "Ni",
      "absorber_mode": "all",
      "absorber_index": 1,
      "filout": "/path/to/out/Ni_HHTP_pristine/Ni_HHTP_pristine_FDM",
      "input_file": "/path/to/out/Ni_HHTP_pristine/Ni_HHTP_pristine_input.txt",
      "protocol_overrides": { "radius": 7.5 }
    },
    {
      "name": "CuTAPP_1_Pyridine",
      "xyz_file": "/path/to/CuTAPP_1_Pyridine.xyz",
      "element": "Cu",
      "edge": "L23",
      "filout": "/path/to/out/CuTAPP_Pyridine/CuTAPP_Py",
      "input_file": "/path/to/out/CuTAPP_Pyridine/CuTAPP_Pyridine_input.txt",
      "protocol_overrides": {
        "radius": 7.5, "use_green": true, "vmax": -6.0,
        "use_tddft": true, "use_spinorbit": true
      }
    }
  ]
}

Every job sets exactly one of "structure_file" (a CIF path -- the original
K-edge/Ni-HHTP-Co-HHTP workflow) or "xyz_file" (an .xyz path, parsed with
xyz_parser.parse_xyz into a "Molecule" block -- the L-edge/TAPP-pyridine
workflow). "edge" is optional on any job (e.g. "L23"); omit it for the
default K edge. Any field under "protocol" that's omitted falls back to
``ProtocolSettings()`` defaults. Per-job "protocol_overrides" only need to
list the fields that differ from the shared protocol.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Tuple

from .config import ProtocolSettings
from .models import DEFAULT_MOLECULE_CELL, JobSpec
from .xyz_parser import parse_xyz

_PROTOCOL_FIELD_NAMES = {f.name for f in fields(ProtocolSettings)}


def _protocol_from_dict(data: dict) -> ProtocolSettings:
    unknown = set(data) - _PROTOCOL_FIELD_NAMES
    if unknown:
        raise ValueError(f"Unknown protocol setting(s): {sorted(unknown)}")
    return ProtocolSettings(**data)


def load_config(path: Path) -> Tuple[ProtocolSettings, List[JobSpec], Dict[str, ProtocolSettings]]:
    """Return (shared_protocol, jobs, per_job_protocol_overrides)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    protocol = _protocol_from_dict(data.get("protocol", {}))

    jobs: List[JobSpec] = []
    per_job_protocol: Dict[str, ProtocolSettings] = {}

    for entry in data.get("jobs", []):
        name = entry["name"]
        structure_file = entry.get("structure_file")
        xyz_file = entry.get("xyz_file")
        if bool(structure_file) == bool(xyz_file):
            raise ValueError(
                f"job {name!r} must set exactly one of 'structure_file' (CIF) "
                f"or 'xyz_file' (.xyz), not both or neither"
            )

        molecule = None
        if xyz_file:
            xyz_path = Path(xyz_file)
            if not xyz_path.is_absolute():
                # Resolve relative to the config file itself, not the caller's
                # cwd, so an example config works regardless of where you run
                # the CLI from.
                xyz_path = (path.parent / xyz_path).resolve()
            cell = entry.get("cell")
            molecule = parse_xyz(xyz_path, cell=tuple(cell) if cell else DEFAULT_MOLECULE_CELL)

        job = JobSpec(
            name=name,
            element=entry["element"],
            filout=Path(entry["filout"]),
            input_file=Path(entry["input_file"]),
            structure_file=Path(structure_file) if structure_file else None,
            molecule=molecule,
            absorber_mode=entry.get("absorber_mode", "all"),
            absorber_index=entry.get("absorber_index", 1),
            edge=entry.get("edge"),
        )
        jobs.append(job)

        overrides = entry.get("protocol_overrides")
        if overrides:
            merged = {**protocol.__dict__, **overrides}
            per_job_protocol[job.name] = _protocol_from_dict(merged)

    return protocol, jobs, per_job_protocol
