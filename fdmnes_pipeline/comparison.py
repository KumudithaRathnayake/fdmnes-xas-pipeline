"""Benchmark computed FDMNES XANES spectra against experimental spectra.

This is the "connect existing comparison code" stage of the plan, built
around the same non-ML benchmarking protocol used in:

    Chen et al., "Robust Machine Learning Inference from X-ray Absorption
    Near Edge Spectra through Featurization", Chem. Mater. 2024, 36,
    2304-2313 (see their Figure 4 / "Benchmarking of Computed XANES", and
    Figure 10 / "Model Interpretability" on energy-shift sensitivity).

That paper's actual benchmarking step (as opposed to its ML models, which
this module has nothing to do with) is three plain steps, done per
spectrum pair:

  1. normalize each spectrum so its maximum intensity is 1.0
  2. energy-align the computed spectrum onto the experimental one -- either
     automatically, by matching peak positions (what their Figure 4 does
     across FDMNES/FEFF/VASP/OCEAN), or with a fixed manual shift you
     supply yourself (what their own repo's code does for raw VASP output,
     e.g. a constant +120 eV offset determined once by hand)
  3. score the match with the Pearson correlation coefficient between the
     two spectra on a shared, interpolated energy grid -- their headline
     number, e.g. "FDMNES (0.99)" in Figure 4

FDMNES's own ``Convolution`` step already applies lifetime/instrumental
broadening to ``_conv.txt``, so -- unlike the raw VASP dielectric-function
spectra that paper's own pipeline smooths by hand before comparing --
nothing extra needs broadening here.

As a complement to Pearson r (not used in that paper, but standard
practice when reporting a XANES/EXAFS fit), this module also reports the
R-factor: sum((exp - calc)^2) / sum(exp^2), on the same aligned, normalized
grid. Lower is better; 0 is a perfect match.

Deliberately stdlib-only for the scoring itself (no numpy needed for
~100-point spectra) -- keeps it runnable anywhere this package already
runs, including the FDMNES machine itself. ``plot_comparison`` is the one
exception: it needs matplotlib, imported lazily, only when you call it.
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .models import JobSpec

Spectrum = Tuple[List[float], List[float]]  # (energy, intensity)


@dataclass
class ComparisonResult:
    job_name: str
    pearson_r: float
    r_factor: float
    shift_applied_eV: float
    energy_grid: List[float]
    computed_intensity: List[float]  # aligned + normalized, on energy_grid
    experimental_intensity: List[float]  # normalized, on energy_grid

    def summary_line(self) -> str:
        return (
            f"{self.job_name} -> Pearson r = {self.pearson_r:.4f}, "
            f"R-factor = {self.r_factor:.4f} (shift {self.shift_applied_eV:+.2f} eV)"
        )

    def to_csv(self, path: Union[str, Path]) -> None:
        """Write the fit summary (as leading `#`-comment lines) plus the
        aligned, normalized spectra -- the same numbers ``plot_comparison``
        draws -- to a plain CSV file, for re-plotting or further analysis
        outside Python (Origin, Excel, a notebook, ...). `#` lines are
        comments; most plotting/spreadsheet tools either skip them
        automatically or you can delete them by hand.
        """
        path = Path(path)
        lines = [
            f"# job_name: {self.job_name}",
            f"# pearson_r: {self.pearson_r:.6f}",
            f"# r_factor: {self.r_factor:.6f}",
            f"# shift_applied_eV: {self.shift_applied_eV:.6f}",
            "Energy_eV,Computed_normalized,Experimental_normalized",
        ]
        for e, c, x in zip(self.energy_grid, self.computed_intensity, self.experimental_intensity):
            lines.append(f"{e:.4f},{c:.6f},{x:.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Loading spectra
# ---------------------------------------------------------------------------


def _sort_by_energy(energy: Sequence[float], intensity: Sequence[float]) -> Spectrum:
    pairs = sorted(zip(energy, intensity))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _read_numeric_columns(
    path: Path, energy_col: int, intensity_col: int
) -> Spectrum:
    """Parse a whitespace/comma/tab-delimited text file into (energy,
    intensity) lists, tolerating headers and comment lines the same way
    ``validation.py`` already does for FDMNES's own output: any line that
    isn't fully numeric (a header row, a ``#`` / ``!`` comment, a blank
    line) is skipped rather than raising.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    needed = max(energy_col, intensity_col) + 1

    energies: List[float] = []
    intensities: List[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < needed:
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue  # header row using non-numeric column labels
        energies.append(values[energy_col])
        intensities.append(values[intensity_col])

    if len(energies) < 2:
        raise ValueError(f"{path}: found fewer than 2 parseable (energy, intensity) rows")
    return _sort_by_energy(energies, intensities)


def load_computed_spectrum_from_file(conv_file: Union[str, Path], intensity_col: int = 1) -> Spectrum:
    """Load a convoluted spectrum directly from its file path -- for when
    you already have a ``_conv.txt`` on hand (e.g. copied off the FDMNES
    machine by hand) and don't need a ``JobSpec``/JSON batch config to
    find it.

    ``intensity_col`` picks which intensity column to use when the file
    has more than one (e.g. several absorber sites, or separate E1E1/E1E2
    terms) -- default is the first column after energy.
    """
    conv_file = Path(conv_file)
    if not conv_file.exists():
        raise FileNotFoundError(f"No such file: {conv_file}")
    return _read_numeric_columns(conv_file, energy_col=0, intensity_col=intensity_col)


def load_computed_spectrum(job: JobSpec, intensity_col: int = 1) -> Spectrum:
    """Load a job's convoluted spectrum from ``{filout}_conv.txt`` -- the
    FDMNES output that already has instrumental/lifetime broadening
    applied, so it's the right file to compare against experiment (not the
    raw, pre-convolution ``{filout}.txt``).
    """
    conv_file = Path(f"{job.filout}_conv.txt")
    if not conv_file.exists():
        raise FileNotFoundError(
            f"No convoluted spectrum for job {job.name!r}: {conv_file} "
            "(run the job, or check job.filout)"
        )
    return load_computed_spectrum_from_file(conv_file, intensity_col=intensity_col)


def load_experimental_spectrum(
    path: Union[str, Path], energy_col: int = 0, intensity_col: int = 1
) -> Spectrum:
    """Load an experimental spectrum from a plain text/CSV file: energy in
    one column, intensity in another (default: the first two columns).
    Any header row and comment lines (``#`` / ``!``) are skipped
    automatically -- point this at whatever two-column export your
    beamline processing software (Athena, Larch, ...) produced.
    """
    return _read_numeric_columns(Path(path), energy_col=energy_col, intensity_col=intensity_col)


# ---------------------------------------------------------------------------
# Normalize, align, interpolate
# ---------------------------------------------------------------------------


def _normalize_max(intensity: Sequence[float]) -> List[float]:
    peak = max(intensity)
    if peak <= 0:
        raise ValueError("cannot normalize a spectrum whose maximum intensity is <= 0")
    return [v / peak for v in intensity]


def _peak_energy(energy: Sequence[float], intensity: Sequence[float]) -> float:
    idx = max(range(len(intensity)), key=lambda i: intensity[i])
    return energy[idx]


def _median_step(energy: Sequence[float]) -> float:
    diffs = sorted(b - a for a, b in zip(energy, energy[1:]) if b > a)
    if not diffs:
        raise ValueError("spectrum has fewer than 2 distinct energy points")
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return diffs[mid]
    return (diffs[mid - 1] + diffs[mid]) / 2


def _arange(lo: float, hi: float, step: float) -> List[float]:
    n = int((hi - lo) / step) + 1
    return [pt for pt in (lo + i * step for i in range(max(n, 0))) if pt <= hi + 1e-9]


def _linear_interp(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    """1D linear interpolation with edge clamping (same semantics as
    ``numpy.interp``'s default). ``xp`` must be sorted ascending."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    i = bisect.bisect_right(xp, x) - 1
    x0, x1 = xp[i], xp[i + 1]
    y0, y1 = fp[i], fp[i + 1]
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def align_and_normalize(
    computed: Spectrum,
    experimental: Spectrum,
    shift_eV: Union[str, float] = "auto",
    grid_step_eV: Optional[float] = None,
) -> Tuple[List[float], List[float], List[float], float]:
    """Normalize both spectra to max intensity 1, energy-shift the computed
    spectrum onto the experimental one, and resample both onto a shared
    energy grid covering their post-shift overlap.

    ``shift_eV``:
      ``"auto"`` (default) -- shift the computed spectrum so its peak
        energy matches the experimental spectrum's peak energy (mirrors
        Figure 4 of the reference benchmarking paper).
      a number -- apply this fixed shift instead (mirrors the constant
        offset the reference paper's own pipeline applies for VASP, e.g.
        ``120.0``).
      ``0`` -- no shift.

    Returns ``(energy_grid, computed_intensity, experimental_intensity, shift_used_eV)``.
    """
    comp_e, comp_i = computed
    exp_e, exp_i = experimental
    comp_i = _normalize_max(comp_i)
    exp_i = _normalize_max(exp_i)

    if shift_eV == "auto":
        shift = _peak_energy(exp_e, exp_i) - _peak_energy(comp_e, comp_i)
    else:
        shift = float(shift_eV)

    shifted_comp_e = [e + shift for e in comp_e]

    lo = max(min(shifted_comp_e), min(exp_e))
    hi = min(max(shifted_comp_e), max(exp_e))
    if lo >= hi:
        raise ValueError(
            "computed and experimental spectra do not overlap in energy after "
            f"a {shift:+.2f} eV shift -- check units/edge, or pass an explicit shift_eV"
        )

    if grid_step_eV is None:
        grid_step_eV = min(_median_step(exp_e), _median_step(shifted_comp_e))
    grid = _arange(lo, hi, grid_step_eV)

    computed_on_grid = [_linear_interp(e, shifted_comp_e, comp_i) for e in grid]
    experimental_on_grid = [_linear_interp(e, exp_e, exp_i) for e in grid]

    return grid, computed_on_grid, experimental_on_grid, shift


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _pearson_r(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = (var_a * var_b) ** 0.5
    if denom == 0:
        raise ValueError(
            "cannot compute Pearson correlation: one spectrum is constant on the comparison grid"
        )
    return cov / denom


def _r_factor(computed: Sequence[float], experimental: Sequence[float]) -> float:
    numerator = sum((e - c) ** 2 for c, e in zip(computed, experimental))
    denominator = sum(e**2 for e in experimental)
    if denominator == 0:
        raise ValueError("cannot compute R-factor: experimental spectrum is all zero on the comparison grid")
    return numerator / denominator


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compare_job_to_experiment(
    job: JobSpec,
    experimental_path: Union[str, Path],
    shift_eV: Union[str, float] = "auto",
    computed_intensity_col: int = 1,
    experimental_energy_col: int = 0,
    experimental_intensity_col: int = 1,
    grid_step_eV: Optional[float] = None,
) -> ComparisonResult:
    """Load a job's computed spectrum and an experimental spectrum, align +
    normalize them per the protocol above, and score the match."""
    computed = load_computed_spectrum(job, intensity_col=computed_intensity_col)
    experimental = load_experimental_spectrum(
        experimental_path,
        energy_col=experimental_energy_col,
        intensity_col=experimental_intensity_col,
    )
    grid, comp_i, exp_i, shift = align_and_normalize(
        computed, experimental, shift_eV=shift_eV, grid_step_eV=grid_step_eV
    )
    return ComparisonResult(
        job_name=job.name,
        pearson_r=_pearson_r(comp_i, exp_i),
        r_factor=_r_factor(comp_i, exp_i),
        shift_applied_eV=shift,
        energy_grid=grid,
        computed_intensity=comp_i,
        experimental_intensity=exp_i,
    )


def compare_conv_file_to_experiment(
    conv_file: Union[str, Path],
    experimental_path: Union[str, Path],
    job_name: Optional[str] = None,
    shift_eV: Union[str, float] = "auto",
    computed_intensity_col: int = 1,
    experimental_energy_col: int = 0,
    experimental_intensity_col: int = 1,
    grid_step_eV: Optional[float] = None,
) -> ComparisonResult:
    """Same as ``compare_job_to_experiment``, but for a ``_conv.txt`` you
    already have on hand -- no ``JobSpec``/JSON batch config required.
    This is the entry point to reach for when you just want to benchmark
    one already-completed FDMNES run without wiring it into a config file
    first.

    ``job_name`` is only a label, used for ``ComparisonResult.job_name``
    and any plot title -- defaults to the conv file's name.
    """
    conv_file = Path(conv_file)
    computed = load_computed_spectrum_from_file(conv_file, intensity_col=computed_intensity_col)
    experimental = load_experimental_spectrum(
        experimental_path,
        energy_col=experimental_energy_col,
        intensity_col=experimental_intensity_col,
    )
    grid, comp_i, exp_i, shift = align_and_normalize(
        computed, experimental, shift_eV=shift_eV, grid_step_eV=grid_step_eV
    )
    return ComparisonResult(
        job_name=job_name or conv_file.stem,
        pearson_r=_pearson_r(comp_i, exp_i),
        r_factor=_r_factor(comp_i, exp_i),
        shift_applied_eV=shift,
        energy_grid=grid,
        computed_intensity=comp_i,
        experimental_intensity=exp_i,
    )


def report_table(results: Iterable[ComparisonResult]) -> str:
    """Plain-text summary table, same style as ``validation.report_table``:

        Ni_HHTP_pristine -> Pearson r = 0.9873, R-factor = 0.0041 (shift +1.20 eV)
    """
    return "\n".join(r.summary_line() for r in results)


def plot_comparison(result: ComparisonResult, out_path: Optional[Union[str, Path]] = None):
    """Overlay plot in the style of Figure 4 of the reference benchmarking
    paper: experimental vs. computed, both normalized, with the Pearson r
    annotated in the legend.

    Needs matplotlib -- not a hard dependency of this package (see
    ``pyproject.toml``'s ``plot`` extra: ``pip install -e ".[plot]"``).
    Everything else in this module works without it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_comparison needs matplotlib installed (`pip install matplotlib`, "
            "or `pip install -e \".[plot]\"` from the package root); the rest of "
            "fdmnes_pipeline.comparison works without it."
        ) from exc

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(result.energy_grid, result.experimental_intensity, label="Exp (1.00)", color="black", lw=2)
    ax.plot(
        result.energy_grid,
        result.computed_intensity,
        label=f"FDMNES ({result.pearson_r:.2f})",
        color="tab:green",
        lw=2,
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(result.job_name)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Batch mode: every job in a config, matched to its experimental file by
# naming convention, outputs saved next to each job's own FDMNES output.
# ---------------------------------------------------------------------------


@dataclass
class BatchComparisonEntry:
    """One job's outcome from ``compare_batch``: either a ``result`` (and
    the paths it was saved to, if any), or a ``skipped_reason`` explaining
    why it couldn't be compared (no computed spectrum yet, no matching
    experimental file, or spectra that don't overlap in energy)."""

    job_name: str
    result: Optional[ComparisonResult] = None
    skipped_reason: Optional[str] = None
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None

    def line(self) -> str:
        if self.result is None:
            return f"{self.job_name} -> SKIPPED ({self.skipped_reason})"
        line = self.result.summary_line()
        saved = []
        if self.csv_path is not None:
            saved.append(f"csv: {self.csv_path}")
        if self.plot_path is not None:
            saved.append(f"plot: {self.plot_path}")
        if saved:
            line += "  [" + ", ".join(saved) + "]"
        return line


def _files_containing_exp(search_dir: Path) -> List[Path]:
    """Every file directly inside ``search_dir`` whose name contains
    ``_EXP`` (case-insensitive) anywhere in it -- the loose fallback match
    used when a job's ``filout`` base name doesn't exactly match the
    experimental file's name (e.g. a typo, an abbreviation, or a name that
    changed after the experimental file was already dropped in)."""
    if not search_dir.is_dir():
        return []
    return sorted(p for p in search_dir.iterdir() if p.is_file() and "_exp" in p.name.lower())


def find_experimental_file(
    job: JobSpec, experimental_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """Locate a job's experimental spectrum.

    Two ways a file can match, tried in order:

    1. Exact naming convention: a job whose FDMNES output base name is
       e.g. ``NiHHTP_Pristine`` (i.e. ``Path(job.filout).name``) matches
       an experimental file named ``NiHHTP_Pristine_EXP`` plus any
       extension (``.txt``, ``.dat``, ``.csv``, ...). This is checked
       first because it's unambiguous even when several jobs share a
       folder.
    2. Loose fallback: if no exact match is found and the search
       directory contains exactly one file whose name contains ``_EXP``
       anywhere (case-insensitive), that file is used. This covers the
       common case of a job's own output folder holding exactly one
       FDMNES run and one experimental file, even if the experimental
       file's name doesn't exactly echo ``filout`` (a typo, a shortened
       name, ...). If the folder contains more than one ``_EXP``-named
       file, the match is ambiguous and this step is skipped -- the
       caller reports that ambiguity rather than guessing.

    Searched directory: ``experimental_dir`` if given (one shared folder
    for every job), otherwise the job's own output folder
    (``Path(job.filout).parent``) -- i.e. experimental files can either
    live alongside each FDMNES output, or all together in one place you
    point at explicitly.

    Returns ``None`` if nothing matches, or if the loose fallback is
    ambiguous (the caller decides how to report that -- ``compare_batch``
    records it as a skip, not an error, since a batch run over many jobs
    shouldn't abort just because one experimental file hasn't been added
    yet, or is named ambiguously).
    """
    search_dir = Path(experimental_dir) if experimental_dir else Path(job.filout).parent
    base = f"{Path(job.filout).name}_EXP"

    exact = search_dir / base
    if exact.is_file():
        return exact

    matches = sorted(p for p in search_dir.glob(f"{base}.*") if p.is_file())
    if matches:
        return matches[0]

    candidates = _files_containing_exp(search_dir)
    if len(candidates) == 1:
        return candidates[0]
    return None


def compare_batch(
    jobs: Iterable[JobSpec],
    experimental_dir: Optional[Union[str, Path]] = None,
    shift_eV: Union[str, float] = "auto",
    computed_intensity_col: int = 1,
    experimental_energy_col: int = 0,
    experimental_intensity_col: int = 1,
    grid_step_eV: Optional[float] = None,
    save_csv: bool = True,
    save_plot: bool = True,
) -> List[BatchComparisonEntry]:
    """Benchmark every job in ``jobs`` against its matching experimental
    spectrum (see ``find_experimental_file`` for the naming convention),
    saving each job's comparison CSV and overlay plot into that job's own
    output folder (``Path(job.filout).parent``), next to its `_conv.txt`.

    A job is skipped (not an error -- the rest of the batch still runs)
    when its computed spectrum doesn't exist yet, no matching experimental
    file is found, or the two spectra don't overlap in energy after
    alignment. ``save_plot`` silently has no effect (CSV is still saved)
    if matplotlib isn't installed.
    """
    entries: List[BatchComparisonEntry] = []
    for job in jobs:
        conv_file = Path(f"{job.filout}_conv.txt")
        if not conv_file.exists():
            entries.append(
                BatchComparisonEntry(job.name, skipped_reason=f"no computed spectrum found: {conv_file}")
            )
            continue

        exp_file = find_experimental_file(job, experimental_dir=experimental_dir)
        if exp_file is None:
            search_dir = Path(experimental_dir) if experimental_dir else Path(job.filout).parent
            candidates = _files_containing_exp(search_dir)
            if len(candidates) > 1:
                names = ", ".join(p.name for p in candidates)
                reason = (
                    f"multiple files containing '_EXP' found in {search_dir} ({names}) -- "
                    f"ambiguous which one belongs to this job; rename it to "
                    f"'{Path(job.filout).name}_EXP.<ext>' to match unambiguously"
                )
            else:
                reason = (
                    f"no experimental file matching '{Path(job.filout).name}_EXP.*' "
                    f"(or containing '_EXP') found in {search_dir}"
                )
            entries.append(BatchComparisonEntry(job.name, skipped_reason=reason))
            continue

        try:
            result = compare_job_to_experiment(
                job,
                exp_file,
                shift_eV=shift_eV,
                computed_intensity_col=computed_intensity_col,
                experimental_energy_col=experimental_energy_col,
                experimental_intensity_col=experimental_intensity_col,
                grid_step_eV=grid_step_eV,
            )
        except ValueError as exc:
            entries.append(BatchComparisonEntry(job.name, skipped_reason=str(exc)))
            continue

        out_dir = Path(job.filout).parent
        base = Path(job.filout).name
        csv_path = None
        plot_path = None

        if save_csv:
            csv_path = out_dir / f"{base}_comparison.csv"
            result.to_csv(csv_path)

        if save_plot:
            candidate_plot_path = out_dir / f"{base}_comparison.png"
            try:
                plot_comparison(result, out_path=candidate_plot_path)
                plot_path = candidate_plot_path
            except ImportError:
                pass  # matplotlib not installed -- CSV (if requested) is still saved

        entries.append(BatchComparisonEntry(job.name, result=result, csv_path=csv_path, plot_path=plot_path))

    return entries


def batch_report_table(entries: Iterable[BatchComparisonEntry]) -> str:
    """Plain-text summary, one line per job, in config order:

        Ni_HHTP_pristine -> Pearson r = 0.9881, R-factor = 0.0123 (shift +8335.31 eV)  [csv: ..., plot: ...]
        Co_HHTP_pristine -> SKIPPED (no experimental file matching 'Co_HHTP_pristine_EXP.*' found in ...)
    """
    return "\n".join(e.line() for e in entries)


def write_batch_summary_csv(entries: Iterable[BatchComparisonEntry], path: Union[str, Path]) -> None:
    """Write one combined row per job (name, status, metrics, and -- for a
    skipped job -- why) to a single CSV, for a quick side-by-side ranking
    across every structure in the batch."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["job_name", "status", "pearson_r", "r_factor", "shift_applied_eV", "note"])
        for e in entries:
            if e.result is not None:
                writer.writerow(
                    [
                        e.job_name,
                        "OK",
                        f"{e.result.pearson_r:.6f}",
                        f"{e.result.r_factor:.6f}",
                        f"{e.result.shift_applied_eV:.6f}",
                        "",
                    ]
                )
            else:
                writer.writerow([e.job_name, "SKIPPED", "", "", "", e.skipped_reason or ""])
