"""Tests for loading a batch config JSON file."""

import json

import pytest

from fdmnes_pipeline.batch_config import load_config


def _write_config(tmp_path, data):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(data))
    return path


def test_load_config_basic(tmp_path):
    data = {
        "protocol": {"radius": 6.0, "use_scf": True},
        "jobs": [
            {
                "name": "job_a",
                "structure_file": "/x/a.cif",
                "element": "Ni",
                "filout": "/x/out/a",
                "input_file": "/x/out/a_input.txt",
            }
        ],
    }
    path = _write_config(tmp_path, data)

    protocol, jobs, per_job = load_config(path)

    assert protocol.radius == 6.0
    assert protocol.use_scf is True
    assert len(jobs) == 1
    assert jobs[0].name == "job_a"
    assert jobs[0].element == "Ni"
    assert per_job == {}


def test_load_config_per_job_override(tmp_path):
    data = {
        "protocol": {"radius": 5.0},
        "jobs": [
            {
                "name": "job_a",
                "structure_file": "/x/a.cif",
                "element": "Co",
                "filout": "/x/out/a",
                "input_file": "/x/out/a_input.txt",
                "protocol_overrides": {"radius": 9.0},
            }
        ],
    }
    path = _write_config(tmp_path, data)

    protocol, jobs, per_job = load_config(path)

    assert protocol.radius == 5.0
    assert per_job["job_a"].radius == 9.0
    # override only touched radius -- everything else still comes from the shared protocol
    assert per_job["job_a"].use_convolution == protocol.use_convolution


def test_load_config_unknown_protocol_field_raises(tmp_path):
    data = {"protocol": {"not_a_real_setting": 1}, "jobs": []}
    path = _write_config(tmp_path, data)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_defaults_when_protocol_omitted(tmp_path):
    data = {"jobs": []}
    path = _write_config(tmp_path, data)

    protocol, jobs, per_job = load_config(path)

    assert protocol.radius == 5.0  # ProtocolSettings default
    assert jobs == []


def test_load_config_xyz_job_with_edge(tmp_path):
    xyz_path = tmp_path / "CuTAPP.xyz"
    xyz_path.write_text("2\ncomment\nCu 0.0 0.0 0.0\nN 1.0 0.0 0.0\n")

    data = {
        "protocol": {"radius": 7.5, "use_green": True},
        "jobs": [
            {
                "name": "CuTAPP_1_Pyridine",
                "xyz_file": str(xyz_path),
                "element": "Cu",
                "edge": "L23",
                "filout": "/x/out/CuTAPP_Py",
                "input_file": "/x/out/CuTAPP_input.txt",
                "protocol_overrides": {"vmax": -6.0, "use_tddft": True, "use_spinorbit": True},
            }
        ],
    }
    path = _write_config(tmp_path, data)

    protocol, jobs, per_job = load_config(path)

    job = jobs[0]
    assert job.structure_file is None
    assert job.molecule is not None
    assert job.molecule.atoms == [(29, 0.0, 0.0, 0.0), (7, 1.0, 0.0, 0.0)]
    assert job.edge == "L23"
    assert per_job["CuTAPP_1_Pyridine"].vmax == -6.0
    assert per_job["CuTAPP_1_Pyridine"].radius == 7.5  # inherited from shared protocol


def test_load_config_rejects_both_structure_file_and_xyz_file(tmp_path):
    data = {
        "jobs": [
            {
                "name": "bad",
                "structure_file": "/x/a.cif",
                "xyz_file": "/x/a.xyz",
                "element": "Ni",
                "filout": "/x/out",
                "input_file": "/x/in.txt",
            }
        ]
    }
    path = _write_config(tmp_path, data)

    with pytest.raises(ValueError, match="exactly one"):
        load_config(path)


def test_load_config_rejects_neither_structure_file_nor_xyz_file(tmp_path):
    data = {
        "jobs": [
            {
                "name": "bad",
                "element": "Ni",
                "filout": "/x/out",
                "input_file": "/x/in.txt",
            }
        ]
    }
    path = _write_config(tmp_path, data)

    with pytest.raises(ValueError, match="exactly one"):
        load_config(path)
