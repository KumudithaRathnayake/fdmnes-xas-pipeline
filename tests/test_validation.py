"""Tests for output validation against synthetic good/bad output sets."""

from pathlib import Path

from fdmnes_pipeline.models import JobSpec, RunResult, RunStatus
from fdmnes_pipeline.validation import validate_batch, validate_run


def _job(tmp_path: Path, name: str = "structure_01") -> JobSpec:
    return JobSpec(
        name=name,
        structure_file=tmp_path / "a.cif",
        element="Ni",
        filout=tmp_path / "out" / name,
        input_file=tmp_path / "out" / f"{name}_input.txt",
    )


def _write_good_outputs(job: JobSpec) -> None:
    job.filout.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{job.filout}_bav.txt").write_text("FDMNES bav file\nRadius = 5.000\n")
    Path(f"{job.filout}.txt").write_text("Energy Spectrum\n0.0 0.01\n1.0 0.02\n2.0 0.05\n")
    Path(f"{job.filout}_conv.txt").write_text(
        "Energy Spectrum\n0.0 0.01\n1.0 0.03\n2.0 0.08\n3.0 0.10\n"
    )


def test_validate_run_success(tmp_path):
    job = _job(tmp_path)
    _write_good_outputs(job)

    result = validate_run(job)

    assert result.status == RunStatus.SUCCESS
    assert result.reasons == []
    assert result.summary_line() == "structure_01 -> SUCCESS"


def test_validate_run_missing_bav_is_failed(tmp_path):
    job = _job(tmp_path)
    job.filout.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{job.filout}.txt").write_text("Energy Spectrum\n0.0 0.01\n")
    Path(f"{job.filout}_conv.txt").write_text("Energy Spectrum\n0.0 0.01\n1.0 0.02\n2.0 0.03\n")

    result = validate_run(job)

    assert result.status == RunStatus.FAILED
    assert any("_bav.txt" in r for r in result.reasons)


def test_validate_run_missing_conv_is_failed_when_expected(tmp_path):
    job = _job(tmp_path)
    job.filout.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{job.filout}_bav.txt").write_text("FDMNES bav file\n")
    Path(f"{job.filout}.txt").write_text("Energy Spectrum\n0.0 0.01\n")

    result = validate_run(job, expect_convolution=True)
    assert result.status == RunStatus.FAILED

    result_no_conv = validate_run(job, expect_convolution=False)
    assert result_no_conv.status == RunStatus.SUCCESS


def test_validate_run_nan_in_conv_is_warning(tmp_path):
    job = _job(tmp_path)
    job.filout.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{job.filout}_bav.txt").write_text("FDMNES bav file\n")
    Path(f"{job.filout}.txt").write_text("Energy Spectrum\n0.0 0.01\n")
    Path(f"{job.filout}_conv.txt").write_text("Energy Spectrum\n0.0 0.01\n1.0 nan\n2.0 0.05\n")

    result = validate_run(job)

    assert result.status == RunStatus.WARNING
    assert any("NaN" in r for r in result.reasons)


def test_validate_run_bav_error_language_downgrades_to_warning(tmp_path):
    job = _job(tmp_path)
    _write_good_outputs(job)
    Path(f"{job.filout}_bav.txt").write_text("FDMNES bav file\nError: something looked odd\n")

    result = validate_run(job)

    assert result.status == RunStatus.WARNING
    assert any("error language" in r for r in result.reasons)


def test_validate_run_fdmnes_error_file_forces_failed(tmp_path):
    job = _job(tmp_path)
    _write_good_outputs(job)  # outputs look fine...
    error_file = tmp_path / "fdmnes_error.txt"
    error_file.write_text("boom")
    run_result = RunResult(job=job, launched=True, returncode=1, fdmnes_error_file=error_file)

    result = validate_run(job, run_result=run_result)

    # ...but the run itself signaled a crash, so FAILED wins regardless of file contents.
    assert result.status == RunStatus.FAILED
    assert any("fdmnes_error.txt" in r for r in result.reasons)


def test_validate_run_not_launched(tmp_path):
    job = _job(tmp_path)
    run_result = RunResult(job=job, launched=False, stderr="input file missing")

    result = validate_run(job, run_result=run_result)

    assert result.status == RunStatus.NOT_RUN


def test_validate_batch_produces_plan_style_table(tmp_path):
    good = _job(tmp_path, "structure_01")
    _write_good_outputs(good)

    bad = _job(tmp_path, "structure_02")
    bad.filout.parent.mkdir(parents=True, exist_ok=True)
    # no output files at all -> FAILED

    results = validate_batch([good, bad])

    assert results[0].status == RunStatus.SUCCESS
    assert results[1].status == RunStatus.FAILED
    assert [r.summary_line() for r in results] == [
        "structure_01 -> SUCCESS",
        "structure_02 -> FAILED",
    ]


def test_validate_batch_requires_matching_length(tmp_path):
    job = _job(tmp_path)
    import pytest

    with pytest.raises(ValueError):
        validate_batch([job], run_results=[])
