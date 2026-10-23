"""Output validation (plan stage 5).

    "A failed FDMNES job shouldn't silently enter the fitting stage.
     The automation should check for expected outputs, particularly
     things such as *_bav.txt / *_conv.txt, and potentially inspect the
     .bav file for errors."

This module checks, per job:
  1. If a RunResult is available: did FDMNES actually launch, exit 0,
     avoid timing out, and avoid creating ``fdmnes_error.txt``? (The
     manual's own troubleshooting section names that file as the crash
     signal.)
  2. Do the expected output files exist at all (``_bav.txt``, and
     ``_conv.txt`` if convolution was requested)?
  3. Is the convoluted spectrum numerically sane (parses, more than a
     couple of rows, no NaN/inf, energy column increasing)?
  4. A soft, best-effort scan of the ``_bav.txt`` text for common
     error/warning language. FDMNES's exact error phrasing isn't
     documented in the excerpts available here, so this only ever
     downgrades a result to WARNING (for human review), never to FAILED
     on its own -- don't treat it as a substitute for reading the bav
     file yourself the first few times.

The result is exactly the kind of table the plan describes:

    structure_01 -> SUCCESS
    structure_02 -> SUCCESS
    structure_03 -> FAILED
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .models import JobSpec, RunResult, RunStatus, ValidationResult

# Soft heuristics only -- see module docstring. Kept short and
# conservative on purpose to minimize false positives.
_BAV_WARNING_PHRASES = (
    "error",
    "fatal",
    "not found",
    "does not exist",
    "stop",
    "abnormal",
)


def _bav_path(filout: Path) -> Path:
    return Path(f"{filout}_bav.txt")


def _conv_path(filout: Path) -> Path:
    return Path(f"{filout}_conv.txt")


def _raw_spectrum_path(filout: Path) -> Path:
    return Path(f"{filout}.txt")


def _check_conv_file(path: Path, result: ValidationResult) -> None:
    """Best-effort numeric sanity check of a convoluted-spectrum file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result.add_reason(f"could not read {path.name}: {exc}")
        return

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue  # header / comment line
        rows.append(values)

    if not rows:
        result.add_reason(f"{path.name} contains no parseable numeric rows")
        return

    if len(rows) < 3:
        result.add_reason(f"{path.name} has suspiciously few rows ({len(rows)})")

    flat_nonfinite = any(
        (math.isnan(v) or math.isinf(v)) for row in rows for v in row
    )
    if flat_nonfinite:
        result.add_reason(f"{path.name} contains NaN/Inf values")

    energies = [row[0] for row in rows]
    if any(b < a for a, b in zip(energies, energies[1:])):
        result.add_reason(f"{path.name} energy column is not monotonically increasing")


def validate_run(
    job: JobSpec,
    run_result: Optional[RunResult] = None,
    expect_convolution: bool = True,
    inspect_bav_text: bool = True,
) -> ValidationResult:
    """Validate one job's outputs. Works with or without a RunResult --
    you can validate files from a run that was launched outside this
    package too, as long as ``job.filout`` matches.
    """
    result = ValidationResult(job_name=job.name, status=RunStatus.SUCCESS)

    if run_result is not None:
        if not run_result.launched:
            result.status = RunStatus.NOT_RUN
            result.add_reason(run_result.stderr or "job was not launched")
            return result
        if run_result.timed_out:
            result.status = RunStatus.FAILED
            result.add_reason(
                f"FDMNES timed out after {run_result.elapsed_seconds:.1f}s"
                if run_result.elapsed_seconds is not None
                else "FDMNES timed out"
            )
        if run_result.fdmnes_error_file is not None:
            result.status = RunStatus.FAILED
            result.add_reason(
                f"fdmnes_error.txt was created ({run_result.fdmnes_error_file}); "
                "see manual section VII- Troubleshooting"
            )
        if run_result.returncode not in (0, None):
            result.status = RunStatus.FAILED
            result.add_reason(f"FDMNES exited with return code {run_result.returncode}")

    bav_file = _bav_path(job.filout)
    conv_file = _conv_path(job.filout)
    raw_file = _raw_spectrum_path(job.filout)
    result.bav_file = bav_file
    result.conv_file = conv_file
    result.raw_spectrum_file = raw_file

    if not bav_file.exists() or bav_file.stat().st_size == 0:
        result.status = RunStatus.FAILED
        result.add_reason(f"expected output not found or empty: {bav_file}")
    elif inspect_bav_text:
        try:
            bav_text = bav_file.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            bav_text = ""
        hits = [p for p in _BAV_WARNING_PHRASES if p in bav_text]
        if hits:
            if result.status == RunStatus.SUCCESS:
                result.status = RunStatus.WARNING
            result.add_reason(
                f"{bav_file.name} contains possible error language ({', '.join(hits)}); "
                "review manually"
            )

    if not raw_file.exists() or raw_file.stat().st_size == 0:
        if result.status == RunStatus.SUCCESS:
            result.status = RunStatus.WARNING
        result.add_reason(f"expected output not found or empty: {raw_file}")

    if expect_convolution:
        if not conv_file.exists() or conv_file.stat().st_size == 0:
            result.status = RunStatus.FAILED
            result.add_reason(f"expected output not found or empty: {conv_file}")
        else:
            _check_conv_file(conv_file, result)
            if result.reasons and result.status == RunStatus.SUCCESS:
                result.status = RunStatus.WARNING

    return result


def validate_batch(
    jobs: Iterable[JobSpec],
    run_results: Optional[Sequence[RunResult]] = None,
    expect_convolution: bool = True,
) -> List[ValidationResult]:
    """Validate a batch of jobs, optionally paired 1:1 with run_batch results."""
    jobs = list(jobs)
    if run_results is not None and len(run_results) != len(jobs):
        raise ValueError("run_results must be the same length as jobs, in the same order")

    results = []
    for i, job in enumerate(jobs):
        run_result = run_results[i] if run_results is not None else None
        results.append(
            validate_run(job, run_result=run_result, expect_convolution=expect_convolution)
        )
    return results


def report_table(results: Iterable[ValidationResult]) -> str:
    """Render a plain-text summary table like the one in the plan:

        structure_01 -> SUCCESS
        structure_02 -> SUCCESS
        structure_03 -> FAILED
    """
    lines = []
    for r in results:
        lines.append(r.summary_line())
        for reason in r.reasons:
            lines.append(f"    - {reason}")
    return "\n".join(lines)
