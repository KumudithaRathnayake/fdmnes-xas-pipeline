"""Tests for FdmnesRunner against the fake executable in conftest.py."""

from pathlib import Path

import pytest

from fdmnes_pipeline.config import ProtocolSettings
from fdmnes_pipeline.input_generator import generate_input_file
from fdmnes_pipeline.models import JobSpec
from fdmnes_pipeline.runner import FdmnesRunner


def _make_job(tmp_path: Path, name: str = "job1") -> JobSpec:
    job = JobSpec(
        name=name,
        structure_file=tmp_path / "structure.cif",
        element="Ni",
        filout=tmp_path / "out" / f"{name}_FDM",
        input_file=tmp_path / "out" / f"{name}_input.txt",
    )
    generate_input_file(job, ProtocolSettings(radius=5.0, use_convolution=True))
    return job


def test_missing_executable_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FdmnesRunner(executable=tmp_path / "does_not_exist")


def test_run_single_success(tmp_path, fake_fdmnes, monkeypatch):
    monkeypatch.setenv("FAKE_FDMNES_MODE", "success")
    job = _make_job(tmp_path)
    runner = FdmnesRunner(executable=fake_fdmnes, timeout_seconds=10)

    result = runner.run_single(job)

    assert result.launched
    assert result.returncode == 0
    assert not result.timed_out
    assert result.fdmnes_error_file is None
    assert Path(f"{job.filout}_bav.txt").exists()
    assert Path(f"{job.filout}_conv.txt").exists()
    # fdmfile.txt must have been written next to the executable, per the manual
    assert (fake_fdmnes.parent / "fdmfile.txt").exists()


def test_run_single_crash_sets_error_file(tmp_path, fake_fdmnes, monkeypatch):
    monkeypatch.setenv("FAKE_FDMNES_MODE", "crash")
    job = _make_job(tmp_path)
    runner = FdmnesRunner(executable=fake_fdmnes, timeout_seconds=10)

    result = runner.run_single(job)

    assert result.launched
    assert result.returncode == 1
    assert result.fdmnes_error_file is not None
    assert result.fdmnes_error_file.exists()


def test_run_single_timeout(tmp_path, fake_fdmnes, monkeypatch):
    monkeypatch.setenv("FAKE_FDMNES_MODE", "hang")
    job = _make_job(tmp_path)
    runner = FdmnesRunner(executable=fake_fdmnes, timeout_seconds=1)

    result = runner.run_single(job)

    assert result.launched
    assert result.timed_out
    assert result.returncode is None


def test_run_single_missing_input_file_short_circuits(tmp_path, fake_fdmnes):
    job = JobSpec(
        name="ghost",
        structure_file=tmp_path / "a.cif",
        element="Ni",
        filout=tmp_path / "out" / "ghost_FDM",
        input_file=tmp_path / "does_not_exist_input.txt",
    )
    runner = FdmnesRunner(executable=fake_fdmnes)

    result = runner.run_single(job)

    assert not result.launched
    assert "does not exist" in result.stderr


def test_run_batch_isolates_failures(tmp_path, fake_fdmnes, monkeypatch):
    monkeypatch.setenv("FAKE_FDMNES_MODE", "success")
    good_job = _make_job(tmp_path, "good")
    runner = FdmnesRunner(executable=fake_fdmnes, timeout_seconds=10)

    # Run one success, then switch the fake to crash mode for the second job,
    # and make sure run_batch keeps going (stop_on_first_failure=False).
    results = [runner.run_single(good_job)]
    monkeypatch.setenv("FAKE_FDMNES_MODE", "crash")
    bad_job = _make_job(tmp_path, "bad")
    results.append(runner.run_single(bad_job))

    assert results[0].returncode == 0
    assert results[1].returncode == 1

    batch_results = runner.run_batch([good_job], stop_on_first_failure=True)
    assert len(batch_results) == 1
