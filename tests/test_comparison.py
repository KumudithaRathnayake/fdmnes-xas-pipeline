"""Tests for fdmnes_pipeline.comparison -- the experimental-benchmarking
stage that mirrors Chen et al., Chem. Mater. 2024, 36, 2304-2313's
non-ML benchmarking protocol (normalize -> energy-align -> Pearson r),
plus the R-factor as a standard complementary metric.
"""

import math
from pathlib import Path

import pytest

from fdmnes_pipeline.comparison import (
    ComparisonResult,
    _linear_interp,
    _pearson_r,
    _r_factor,
    align_and_normalize,
    batch_report_table,
    compare_batch,
    compare_conv_file_to_experiment,
    compare_job_to_experiment,
    find_experimental_file,
    load_computed_spectrum,
    load_computed_spectrum_from_file,
    load_experimental_spectrum,
    report_table,
    write_batch_summary_csv,
)
from fdmnes_pipeline.models import JobSpec, MoleculeStructure


def _make_job(tmp_path, name, filout_name):
    """A job whose filout lives in its own subfolder, e.g.
    tmp_path/name/filout_name -- mirrors a real per-structure output
    folder layout."""
    out_dir = tmp_path / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return JobSpec(
        name=name,
        element="Ni",
        filout=out_dir / filout_name,
        input_file=out_dir / f"{filout_name}_input.txt",
        molecule=MoleculeStructure(atoms=[(28, 0.0, 0.0, 0.0)]),
    )


def _gaussian_spectrum(peak_energy, sigma, scale=1.0, lo=8330.0, hi=8370.0, step=0.5):
    n = int(round((hi - lo) / step)) + 1
    energy = [lo + i * step for i in range(n)]
    intensity = [scale * math.exp(-((e - peak_energy) ** 2) / (2 * sigma**2)) for e in energy]
    return energy, intensity


def _write_two_col(path, energy, intensity, header=None):
    lines = []
    if header:
        lines.append(header)
    for e, i in zip(energy, intensity):
        lines.append(f"{e:.4f} {i:.6f}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def test_linear_interp_basic():
    xp, fp = [0.0, 1.0, 2.0], [0.0, 10.0, 0.0]
    assert _linear_interp(0.5, xp, fp) == pytest.approx(5.0)
    assert _linear_interp(1.5, xp, fp) == pytest.approx(5.0)


def test_linear_interp_clamps_outside_range():
    xp, fp = [1.0, 2.0], [10.0, 20.0]
    assert _linear_interp(-5.0, xp, fp) == 10.0
    assert _linear_interp(50.0, xp, fp) == 20.0


def test_pearson_r_perfect_correlation():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 6.0, 8.0]  # perfectly proportional
    assert _pearson_r(a, b) == pytest.approx(1.0)


def test_pearson_r_perfect_anticorrelation():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    assert _pearson_r(a, b) == pytest.approx(-1.0)


def test_r_factor_zero_for_identical_spectra():
    a = [0.1, 0.5, 1.0, 0.3]
    assert _r_factor(a, a) == pytest.approx(0.0)


def test_r_factor_positive_for_mismatched_spectra():
    computed = [0.0, 0.0, 0.0]
    experimental = [1.0, 1.0, 1.0]
    # sum((e-c)^2) / sum(e^2) = 3*(1-0)^2 / 3*(1^2) = 1.0
    assert _r_factor(computed, experimental) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Loading spectra
# ---------------------------------------------------------------------------


def test_load_experimental_spectrum_skips_header_and_comments(tmp_path):
    path = tmp_path / "exp.txt"
    path.write_text(
        "# Ni K-edge, beamline 20-BM\n"
        "Energy(eV) Intensity(a.u.)\n"
        "8330.0 0.01\n"
        "8340.0 0.5\n"
        "8350.0 1.0\n"
        "! trailing comment\n"
    )
    energy, intensity = load_experimental_spectrum(path)
    assert energy == [8330.0, 8340.0, 8350.0]
    assert intensity == [0.01, 0.5, 1.0]


def test_load_experimental_spectrum_accepts_csv(tmp_path):
    path = tmp_path / "exp.csv"
    path.write_text("energy,intensity\n8330.0,0.1\n8331.0,0.2\n")
    energy, intensity = load_experimental_spectrum(path)
    assert energy == [8330.0, 8331.0]
    assert intensity == [0.1, 0.2]


def test_load_experimental_spectrum_too_few_rows_raises(tmp_path):
    path = tmp_path / "exp.txt"
    path.write_text("8330.0 0.1\n")
    with pytest.raises(ValueError, match="fewer than 2"):
        load_experimental_spectrum(path)


def test_load_computed_spectrum_reads_conv_file(tmp_path):
    job = JobSpec(
        name="synthetic",
        element="Ni",
        filout=tmp_path / "out" / "synthetic",
        input_file=tmp_path / "out" / "synthetic_input.txt",
        molecule=MoleculeStructure(atoms=[(28, 0.0, 0.0, 0.0)]),
    )
    conv_file = tmp_path / "out" / "synthetic_conv.txt"
    conv_file.parent.mkdir(parents=True, exist_ok=True)
    energy, intensity = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, energy, intensity, header="! FDMNES convoluted output")

    loaded_energy, loaded_intensity = load_computed_spectrum(job)
    assert loaded_energy == pytest.approx(energy)
    # written with fixed 6-decimal formatting, so allow for that rounding
    assert loaded_intensity == pytest.approx(intensity, abs=1e-5)


def test_load_computed_spectrum_from_file_direct(tmp_path):
    conv_file = tmp_path / "already_have_this_conv.txt"
    energy, intensity = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, energy, intensity, header="    Energy    <xanes>")

    loaded_energy, loaded_intensity = load_computed_spectrum_from_file(conv_file)
    assert loaded_energy == pytest.approx(energy)
    assert loaded_intensity == pytest.approx(intensity, abs=1e-5)


def test_load_computed_spectrum_from_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_computed_spectrum_from_file(tmp_path / "nope_conv.txt")


def test_load_computed_spectrum_missing_file_raises(tmp_path):
    job = JobSpec(
        name="missing",
        element="Ni",
        filout=tmp_path / "out" / "missing",
        input_file=tmp_path / "out" / "missing_input.txt",
        molecule=MoleculeStructure(atoms=[(28, 0.0, 0.0, 0.0)]),
    )
    with pytest.raises(FileNotFoundError):
        load_computed_spectrum(job)


# ---------------------------------------------------------------------------
# align_and_normalize
# ---------------------------------------------------------------------------


def test_align_and_normalize_auto_shift_matches_peak_offset():
    computed = _gaussian_spectrum(8350.0, 5.0, scale=1.0)
    experimental = _gaussian_spectrum(8352.0, 5.0, scale=0.7)  # +2 eV, different scale

    grid, comp_i, exp_i, shift = align_and_normalize(computed, experimental, shift_eV="auto")

    assert shift == pytest.approx(2.0)
    assert max(comp_i) == pytest.approx(1.0)  # normalized
    assert max(exp_i) == pytest.approx(1.0)
    # after correcting for the known +2 eV offset, the two Gaussians should
    # coincide almost exactly
    r = _pearson_r(comp_i, exp_i)
    assert r > 0.999


def test_align_and_normalize_manual_shift_overrides_auto():
    computed = _gaussian_spectrum(8350.0, 5.0)
    experimental = _gaussian_spectrum(8352.0, 5.0)

    grid, comp_i, exp_i, shift = align_and_normalize(computed, experimental, shift_eV=0.0)
    assert shift == 0.0
    # unaligned (2 eV off): correlation should be noticeably worse than the
    # auto-aligned case, though the two Gaussians still overlap somewhat
    r_unaligned = _pearson_r(comp_i, exp_i)

    _, comp_i2, exp_i2, _ = align_and_normalize(computed, experimental, shift_eV="auto")
    r_aligned = _pearson_r(comp_i2, exp_i2)

    assert r_aligned > r_unaligned


def test_align_and_normalize_raises_when_no_overlap():
    computed = _gaussian_spectrum(8350.0, 5.0, lo=8330.0, hi=8370.0)
    # experimental spectrum's own peak sits inside its own range -- these
    # two energy ranges simply don't overlap at all, unlike an
    # underflowed-to-zero tail
    experimental = _gaussian_spectrum(9020.0, 5.0, lo=9000.0, hi=9040.0)
    with pytest.raises(ValueError, match="do not overlap"):
        align_and_normalize(computed, experimental, shift_eV=0.0)


# ---------------------------------------------------------------------------
# End-to-end: compare_job_to_experiment + report_table
# ---------------------------------------------------------------------------


def test_compare_job_to_experiment_end_to_end(tmp_path):
    job = JobSpec(
        name="Ni_HHTP_pristine",
        element="Ni",
        filout=tmp_path / "out" / "Ni_HHTP_pristine",
        input_file=tmp_path / "out" / "Ni_HHTP_pristine_input.txt",
        molecule=MoleculeStructure(atoms=[(28, 0.0, 0.0, 0.0)]),
    )
    conv_file = tmp_path / "out" / "Ni_HHTP_pristine_conv.txt"
    conv_file.parent.mkdir(parents=True, exist_ok=True)
    comp_energy, comp_intensity = _gaussian_spectrum(8350.0, 5.0, scale=1.0)
    _write_two_col(conv_file, comp_energy, comp_intensity)

    exp_file = tmp_path / "experimental_Ni_HHTP.txt"
    exp_energy, exp_intensity = _gaussian_spectrum(8351.5, 5.0, scale=0.8)
    _write_two_col(exp_file, exp_energy, exp_intensity, header="# beamline export")

    result = compare_job_to_experiment(job, exp_file)

    assert isinstance(result, ComparisonResult)
    assert result.job_name == "Ni_HHTP_pristine"
    assert result.shift_applied_eV == pytest.approx(1.5)
    assert result.pearson_r > 0.999
    assert result.r_factor < 0.01

    table = report_table([result])
    assert "Ni_HHTP_pristine" in table
    assert "Pearson r" in table
    assert "R-factor" in table


def test_compare_conv_file_to_experiment_no_config_needed(tmp_path):
    """The entry point for 'I already have a convoluted file' -- no
    JobSpec/JSON batch config required, just two file paths."""
    conv_file = tmp_path / "Ni_HHTP_pristine_Output_conv.txt"
    comp_energy, comp_intensity = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, comp_energy, comp_intensity, header="    Energy    <xanes>")

    exp_file = tmp_path / "experimental_Ni_HHTP.txt"
    exp_energy, exp_intensity = _gaussian_spectrum(8351.5, 5.0, scale=0.8)
    _write_two_col(exp_file, exp_energy, exp_intensity, header="# beamline export")

    result = compare_conv_file_to_experiment(conv_file, exp_file)

    assert result.job_name == "Ni_HHTP_pristine_Output_conv"  # defaults to the file's stem
    assert result.shift_applied_eV == pytest.approx(1.5)
    assert result.pearson_r > 0.999

    # an explicit label overrides the filename-derived default
    named = compare_conv_file_to_experiment(conv_file, exp_file, job_name="Ni_HHTP_pristine")
    assert named.job_name == "Ni_HHTP_pristine"


def test_comparison_result_to_csv(tmp_path):
    computed = _gaussian_spectrum(8350.0, 5.0)
    experimental = _gaussian_spectrum(8351.5, 5.0, scale=0.8)
    grid, comp_i, exp_i, shift = align_and_normalize(computed, experimental)
    result = ComparisonResult(
        job_name="Ni_HHTP_pristine",
        pearson_r=_pearson_r(comp_i, exp_i),
        r_factor=_r_factor(comp_i, exp_i),
        shift_applied_eV=shift,
        energy_grid=grid,
        computed_intensity=comp_i,
        experimental_intensity=exp_i,
    )

    out_path = tmp_path / "comparison.csv"
    result.to_csv(out_path)

    text = out_path.read_text()
    lines = text.splitlines()
    assert lines[0] == "# job_name: Ni_HHTP_pristine"
    assert lines[1].startswith("# pearson_r:")
    assert lines[2].startswith("# r_factor:")
    assert lines[3].startswith("# shift_applied_eV:")
    assert lines[4] == "Energy_eV,Computed_normalized,Experimental_normalized"
    # one data row per point on the aligned grid
    assert len(lines) - 5 == len(grid)
    # spot check the first data row parses back out correctly
    first_energy, first_comp, first_exp = lines[5].split(",")
    assert float(first_energy) == pytest.approx(grid[0], abs=1e-3)
    assert float(first_comp) == pytest.approx(comp_i[0], abs=1e-5)
    assert float(first_exp) == pytest.approx(exp_i[0], abs=1e-5)


def test_compare_job_to_experiment_manual_shift(tmp_path):
    job = JobSpec(
        name="manual_shift_job",
        element="Cu",
        filout=tmp_path / "out" / "job",
        input_file=tmp_path / "out" / "job_input.txt",
        molecule=MoleculeStructure(atoms=[(29, 0.0, 0.0, 0.0)]),
        edge="L23",
    )
    conv_file = tmp_path / "out" / "job_conv.txt"
    conv_file.parent.mkdir(parents=True, exist_ok=True)
    comp_energy, comp_intensity = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, comp_energy, comp_intensity)

    exp_file = tmp_path / "exp.txt"
    # e.g. a big, known offset -- peak stays inside this spectrum's own range
    exp_energy, exp_intensity = _gaussian_spectrum(8470.0, 5.0, lo=8450.0, hi=8490.0)
    _write_two_col(exp_file, exp_energy, exp_intensity)

    result = compare_job_to_experiment(job, exp_file, shift_eV=120.0)
    assert result.shift_applied_eV == 120.0
    assert result.pearson_r > 0.999


# ---------------------------------------------------------------------------
# Plotting (skipped when matplotlib isn't installed)
# ---------------------------------------------------------------------------


def test_plot_comparison(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from fdmnes_pipeline.comparison import plot_comparison

    computed = _gaussian_spectrum(8350.0, 5.0)
    experimental = _gaussian_spectrum(8351.0, 5.0)
    grid, comp_i, exp_i, shift = align_and_normalize(computed, experimental)
    result = ComparisonResult(
        job_name="plot_test",
        pearson_r=_pearson_r(comp_i, exp_i),
        r_factor=_r_factor(comp_i, exp_i),
        shift_applied_eV=shift,
        energy_grid=grid,
        computed_intensity=comp_i,
        experimental_intensity=exp_i,
    )
    out_path = tmp_path / "comparison.png"
    fig = plot_comparison(result, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    fig.savefig  # sanity: it's a real matplotlib figure object


# ---------------------------------------------------------------------------
# Batch mode: find_experimental_file, compare_batch, summaries
# ---------------------------------------------------------------------------


def test_find_experimental_file_matches_naming_convention(tmp_path):
    job = _make_job(tmp_path, "Ni_HHTP_pristine", "NiHHTP_Pristine")
    exp_file = job.filout.parent / "NiHHTP_Pristine_EXP.txt"
    exp_file.write_text("dummy")

    found = find_experimental_file(job)
    assert found == exp_file


def test_find_experimental_file_any_extension(tmp_path):
    job = _make_job(tmp_path, "job", "SomeOutput")
    exp_file = job.filout.parent / "SomeOutput_EXP.dat"
    exp_file.write_text("dummy")

    assert find_experimental_file(job) == exp_file


def test_find_experimental_file_no_extension(tmp_path):
    job = _make_job(tmp_path, "job", "SomeOutput")
    exp_file = job.filout.parent / "SomeOutput_EXP"
    exp_file.write_text("dummy")

    assert find_experimental_file(job) == exp_file


def test_find_experimental_file_missing_returns_none(tmp_path):
    job = _make_job(tmp_path, "job", "SomeOutput")
    assert find_experimental_file(job) is None


def test_find_experimental_file_shared_directory_override(tmp_path):
    job = _make_job(tmp_path, "job", "SomeOutput")
    shared_dir = tmp_path / "all_experimental_data"
    shared_dir.mkdir()
    exp_file = shared_dir / "SomeOutput_EXP.csv"
    exp_file.write_text("dummy")

    # not found in the job's own output folder...
    assert find_experimental_file(job) is None
    # ...but found when pointed at the shared folder
    assert find_experimental_file(job, experimental_dir=shared_dir) == exp_file


def test_find_experimental_file_fallback_on_name_mismatch(tmp_path):
    # filout base name is "NiCo_HHTP_SQL_FDM" but the experimental file is
    # just "NiCo_HHTP_EXP" -- doesn't match the exact convention, but it's
    # the only "_EXP" file in the folder, so the loose fallback picks it up.
    job = _make_job(tmp_path, "NiCo_HHTP_sql", "NiCo_HHTP_SQL_FDM")
    exp_file = job.filout.parent / "NiCo_HHTP_EXP.dat"
    exp_file.write_text("dummy")

    assert find_experimental_file(job) == exp_file


def test_find_experimental_file_fallback_is_case_insensitive(tmp_path):
    job = _make_job(tmp_path, "job", "SomeOutput")
    exp_file = job.filout.parent / "somoutput_exp.dat"
    exp_file.write_text("dummy")

    assert find_experimental_file(job) == exp_file


def test_find_experimental_file_exact_match_preferred_over_fallback(tmp_path):
    # When the exact convention *does* match, it wins even if a
    # differently-named "_EXP" file also happens to sit in the folder.
    job = _make_job(tmp_path, "job", "SomeOutput")
    exact = job.filout.parent / "SomeOutput_EXP.dat"
    exact.write_text("exact")
    decoy = job.filout.parent / "old_backup_EXP.dat"
    decoy.write_text("decoy")

    assert find_experimental_file(job) == exact


def test_find_experimental_file_ambiguous_fallback_returns_none(tmp_path):
    # Two differently-named "_EXP" files and no exact match -- can't guess,
    # so this must NOT silently pick one.
    job = _make_job(tmp_path, "job", "SomeOutput")
    (job.filout.parent / "run1_EXP.dat").write_text("a")
    (job.filout.parent / "run2_EXP.dat").write_text("b")

    assert find_experimental_file(job) is None


def test_compare_batch_reports_ambiguous_exp_files(tmp_path):
    job = _make_job(tmp_path, "job", "SomeOutput")
    conv_file = Path(f"{job.filout}_conv.txt")
    e, i = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, e, i)
    (job.filout.parent / "run1_EXP.dat").write_text("a")
    (job.filout.parent / "run2_EXP.dat").write_text("b")

    entries = compare_batch([job], save_plot=False)

    assert entries[0].result is None
    assert "multiple files containing '_EXP'" in entries[0].skipped_reason
    assert "run1_EXP.dat" in entries[0].skipped_reason
    assert "run2_EXP.dat" in entries[0].skipped_reason


def test_compare_batch_uses_fallback_match(tmp_path):
    job = _make_job(tmp_path, "NiCo_HHTP_sql", "NiCo_HHTP_SQL_FDM")
    conv_file = Path(f"{job.filout}_conv.txt")
    e, i = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, e, i)
    exp_e, exp_i = _gaussian_spectrum(8351.0, 5.0)
    _write_two_col(job.filout.parent / "NiCo_HHTP_EXP.dat", exp_e, exp_i)

    entries = compare_batch([job], save_plot=False)

    assert entries[0].result is not None
    assert entries[0].skipped_reason is None


def test_compare_batch_multiple_jobs_matched_by_convention(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    jobs = []
    for name, filout_name, comp_peak, exp_peak in [
        ("Ni_HHTP_pristine", "NiHHTP_Pristine", 8350.0, 8351.5),
        ("Co_HHTP_pristine", "CoHHTP_Pristine", 7720.0, 7721.0),
    ]:
        job = _make_job(tmp_path, name, filout_name)
        jobs.append(job)
        conv_file = Path(f"{job.filout}_conv.txt")
        e, i = _gaussian_spectrum(comp_peak, 5.0, lo=comp_peak - 20, hi=comp_peak + 20)
        _write_two_col(conv_file, e, i, header="    Energy    <xanes>")

        exp_file = job.filout.parent / f"{filout_name}_EXP.txt"
        e2, i2 = _gaussian_spectrum(exp_peak, 5.0, scale=0.8, lo=exp_peak - 20, hi=exp_peak + 20)
        _write_two_col(exp_file, e2, i2, header="# beamline export")

    entries = compare_batch(jobs)

    assert len(entries) == 2
    for entry, expected_shift in zip(entries, [1.5, 1.0]):
        assert entry.result is not None
        assert entry.skipped_reason is None
        assert entry.result.shift_applied_eV == pytest.approx(expected_shift)
        assert entry.result.pearson_r > 0.999

    # saved next to each job's own FDMNES output, not somewhere shared
    ni_entry, co_entry = entries
    assert ni_entry.csv_path == jobs[0].filout.parent / "NiHHTP_Pristine_comparison.csv"
    assert ni_entry.csv_path.exists()
    assert ni_entry.plot_path == jobs[0].filout.parent / "NiHHTP_Pristine_comparison.png"
    assert ni_entry.plot_path.exists()
    assert co_entry.csv_path == jobs[1].filout.parent / "CoHHTP_Pristine_comparison.csv"
    assert co_entry.csv_path.exists()

    table = batch_report_table(entries)
    assert "Ni_HHTP_pristine" in table
    assert "Co_HHTP_pristine" in table
    assert "csv:" in table
    assert "plot:" in table


def test_compare_batch_skips_missing_computed_spectrum(tmp_path):
    job = _make_job(tmp_path, "no_run_yet", "NotRunYet")
    # no _conv.txt written at all -- job hasn't been run through FDMNES yet
    exp_file = job.filout.parent / "NotRunYet_EXP.txt"
    e, i = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(exp_file, e, i)

    entries = compare_batch([job], save_plot=False)

    assert len(entries) == 1
    assert entries[0].result is None
    assert "no computed spectrum found" in entries[0].skipped_reason
    assert "NotRunYet_conv.txt" in entries[0].skipped_reason


def test_compare_batch_skips_missing_experimental_file(tmp_path):
    job = _make_job(tmp_path, "no_exp_yet", "NoExpYet")
    conv_file = Path(f"{job.filout}_conv.txt")
    e, i = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, e, i)
    # no matching "NoExpYet_EXP.*" file added

    entries = compare_batch([job], save_plot=False)

    assert len(entries) == 1
    assert entries[0].result is None
    assert "no experimental file matching" in entries[0].skipped_reason
    assert "NoExpYet_EXP.*" in entries[0].skipped_reason


def test_compare_batch_no_csv_no_plot_flags(tmp_path):
    job = _make_job(tmp_path, "job", "Output")
    conv_file = Path(f"{job.filout}_conv.txt")
    e, i = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, e, i)
    exp_file = job.filout.parent / "Output_EXP.txt"
    _write_two_col(exp_file, e, i)

    entries = compare_batch([job], save_csv=False, save_plot=False)

    assert entries[0].result is not None
    assert entries[0].csv_path is None
    assert entries[0].plot_path is None
    # nothing new written to the output folder besides the conv/exp files
    written = {p.name for p in job.filout.parent.iterdir()}
    assert written == {"Output_conv.txt", "Output_EXP.txt"}


def test_write_batch_summary_csv(tmp_path):
    job_ok = _make_job(tmp_path, "ok_job", "OkOutput")
    conv_file = Path(f"{job_ok.filout}_conv.txt")
    e, i = _gaussian_spectrum(8350.0, 5.0)
    _write_two_col(conv_file, e, i)
    exp_file = job_ok.filout.parent / "OkOutput_EXP.txt"
    _write_two_col(exp_file, e, i)

    job_missing = _make_job(tmp_path, "missing_job", "MissingOutput")
    # no conv.txt for this one -- will be skipped

    entries = compare_batch([job_ok, job_missing], save_plot=False)

    summary_path = tmp_path / "batch_summary.csv"
    write_batch_summary_csv(entries, summary_path)

    import csv as csv_module

    with summary_path.open() as f:
        rows = list(csv_module.reader(f))

    assert rows[0] == ["job_name", "status", "pearson_r", "r_factor", "shift_applied_eV", "note"]
    assert rows[1][0] == "ok_job"
    assert rows[1][1] == "OK"
    assert float(rows[1][2]) == pytest.approx(1.0, abs=1e-6)  # identical spectra -> perfect correlation
    assert rows[2][0] == "missing_job"
    assert rows[2][1] == "SKIPPED"
    assert "no computed spectrum found" in rows[2][5]
