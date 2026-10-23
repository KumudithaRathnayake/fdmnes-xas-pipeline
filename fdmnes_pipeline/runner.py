"""Automated FDMNES execution (plan stage 4).

Conceptually, per the plan:

    Python
      |
      +-- generate input        (input_generator.py)
      +-- generate/update fdmfile.txt
      +-- launch FDMNES
      +-- wait/check completion

This module owns the last three steps. It deliberately runs FDMNES once
*per job* rather than handing it one big multi-job ``fdmfile.txt`` -- per
the plan's own validation goal ("structure_01 -> SUCCESS, structure_02 ->
FAILED"), one structure's crash should not take the rest of the batch down
with it, and per-job timeouts are only possible this way.

IMPORTANT: per the FDMNES manual ("VI- Running"), ``fdmfile.txt`` must live
in the *same directory as the executable*. This runner writes/overwrites
that file next to the executable you point it at before every launch --
if you run other FDMNES jobs by hand from that same install, expect
``fdmfile.txt`` there to be overwritten.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .models import JobSpec, RunResult


class FdmnesRunner:
    """Launches FDMNES for one or more already-generated input files.

    Parameters
    ----------
    executable:
        Path to the FDMNES executable (e.g. ``fdmnes_linux64`` or
        ``fdmnes_win64.exe``). Must exist; ``fdmfile.txt`` is written next
        to it before each launch.
    mpirun_path:
        If set, launch via MPI: ``mpirun -np <num_processes> <executable>``
        (see manual "V- Parallelization"). Leave as None to run
        sequentially.
    num_processes:
        Number of MPI processes when ``mpirun_path`` is set.
    timeout_seconds:
        Kill the FDMNES process if it runs longer than this. None = no
        timeout. XANES calculations -- especially SCF ones with a large
        radius -- can legitimately run for a long time, so set this
        generously rather than tightly; a too-tight timeout will look
        identical to a real hang in the validation report.
    extra_mpirun_args:
        Extra arguments inserted after ``-np <n>``, e.g.
        ``["-x", "HOST_NUM_FOR_MUMPS=2"]`` for the two-level parallelism
        described in the manual.
    """

    def __init__(
        self,
        executable: Path,
        mpirun_path: Optional[Path] = None,
        num_processes: int = 1,
        timeout_seconds: Optional[float] = None,
        extra_mpirun_args: Optional[Sequence[str]] = None,
    ) -> None:
        self.executable = Path(executable)
        if not self.executable.exists():
            raise FileNotFoundError(f"FDMNES executable not found: {self.executable}")
        self.mpirun_path = Path(mpirun_path) if mpirun_path else None
        self.num_processes = num_processes
        self.timeout_seconds = timeout_seconds
        self.extra_mpirun_args = list(extra_mpirun_args or [])
        self._fdmfile_path = self.executable.parent / "fdmfile.txt"
        self._error_file_path = self.executable.parent / "fdmnes_error.txt"

    def _write_fdmfile(self, input_file: Path) -> None:
        body = f"1\n{Path(input_file).resolve()}\n"
        self._fdmfile_path.write_text(body, encoding="utf-8", newline="\n")

    def _build_command(self) -> List[str]:
        if self.mpirun_path:
            cmd = [str(self.mpirun_path), "-np", str(self.num_processes)]
            cmd += self.extra_mpirun_args
            cmd += [str(self.executable)]
            return cmd
        return [str(self.executable)]

    def _clear_stale_error_file(self) -> None:
        if self._error_file_path.exists():
            try:
                self._error_file_path.unlink()
            except OSError:
                pass

    def run_single(self, job: JobSpec) -> RunResult:
        """Run FDMNES on exactly one job's already-generated input file."""
        if not job.input_file.exists():
            result = RunResult(job=job, launched=False)
            result.stderr = f"Input file does not exist: {job.input_file}"
            return result

        self._clear_stale_error_file()
        self._write_fdmfile(job.input_file)

        command = self._build_command()
        start = time.monotonic()
        result = RunResult(job=job, launched=True)
        try:
            proc = subprocess.run(
                command,
                cwd=self.executable.parent,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            result.returncode = proc.returncode
            result.stdout = proc.stdout
            result.stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            result.stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        finally:
            result.elapsed_seconds = time.monotonic() - start

        if self._error_file_path.exists():
            result.fdmnes_error_file = self._error_file_path

        return result

    def run_batch(
        self, jobs: Iterable[JobSpec], stop_on_first_failure: bool = False
    ) -> List[RunResult]:
        """Run FDMNES once per job (isolated launches, per-job timeout).

        Returns one RunResult per job, in the order given, so it lines up
        directly with a plan-stage-5 style report.
        """
        results: List[RunResult] = []
        for job in jobs:
            result = self.run_single(job)
            results.append(result)
            failed = result.timed_out or not result.launched or (result.returncode not in (0,))
            if stop_on_first_failure and failed:
                break
        return results
