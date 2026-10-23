"""Command-line entry point tying the pipeline stages together.

Examples
--------
Generate input files for every job in a config, without running FDMNES:

    python -m fdmnes_pipeline.cli generate --config examples/jobs.json

Run FDMNES (sequential) for every job, then validate outputs:

    python -m fdmnes_pipeline.cli run --config examples/jobs.json \\
        --executable /path/to/fdmnes_linux64 --timeout 7200

    python -m fdmnes_pipeline.cli validate --config examples/jobs.json

Do all three in one go:

    python -m fdmnes_pipeline.cli pipeline --config examples/jobs.json \\
        --executable /path/to/fdmnes_linux64 --timeout 7200

Run in parallel over MPI instead:

    python -m fdmnes_pipeline.cli run --config examples/jobs.json \\
        --executable /path/to/fdmnes_linux64_mpi \\
        --mpirun /usr/bin/mpirun --np 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .batch_config import load_config
from .comparison import compare_batch, compare_conv_file_to_experiment, compare_job_to_experiment
from .comparison import batch_report_table as comparison_batch_report_table
from .comparison import report_table as comparison_report_table
from .comparison import write_batch_summary_csv
from .input_generator import generate_batch
from .runner import FdmnesRunner
from .validation import report_table, validate_batch


def _cmd_generate(args: argparse.Namespace) -> int:
    protocol, jobs, per_job = load_config(args.config)
    generate_batch(jobs, protocol, per_job_protocol=per_job)
    print(f"Generated {len(jobs)} input file(s).")
    return 0


def _cmd_run(args: argparse.Namespace) -> List:
    _, jobs, _ = load_config(args.config)
    runner = FdmnesRunner(
        executable=Path(args.executable),
        mpirun_path=Path(args.mpirun) if args.mpirun else None,
        num_processes=args.np,
        timeout_seconds=args.timeout,
    )
    results = runner.run_batch(jobs, stop_on_first_failure=args.stop_on_first_failure)
    for job, result in zip(jobs, results):
        status = "timed out" if result.timed_out else f"exit code {result.returncode}"
        print(f"{job.name}: launched={result.launched} {status} ({result.elapsed_seconds:.1f}s)"
              if result.elapsed_seconds is not None
              else f"{job.name}: launched={result.launched} {status}")
        if result.fdmnes_error_file:
            print(f"    fdmnes_error.txt created: {result.fdmnes_error_file}")
    return results


def _cmd_validate(args: argparse.Namespace, run_results=None) -> int:
    _, jobs, _ = load_config(args.config)
    results = validate_batch(
        jobs, run_results=run_results, expect_convolution=not args.no_convolution_check
    )
    print(report_table(results))
    any_failed = any(r.status.value == "FAILED" for r in results)
    return 1 if any_failed else 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    _cmd_generate(args)
    results = _cmd_run(args)
    return _cmd_validate(args, run_results=results)


def _parse_shift(raw: str) -> "float | str":
    if raw == "auto":
        return raw
    try:
        return float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shift must be 'auto' or a number, got {raw!r}")


def _cmd_compare(args: argparse.Namespace) -> int:
    if args.conv_file is not None:
        # Direct mode: an already-completed _conv.txt, no config/job needed.
        result = compare_conv_file_to_experiment(
            args.conv_file,
            experimental_path=args.experimental,
            job_name=args.name,
            shift_eV=args.shift,
            computed_intensity_col=args.computed_intensity_col,
            experimental_energy_col=args.exp_energy_col,
            experimental_intensity_col=args.exp_intensity_col,
        )
    else:
        if not args.config or not args.job:
            print("compare needs either --conv-file, or both --config and --job.")
            return 1
        _, jobs, _ = load_config(args.config)
        by_name = {job.name: job for job in jobs}
        if args.job not in by_name:
            available = ", ".join(sorted(by_name)) or "(no jobs in config)"
            print(f"No job named {args.job!r} in {args.config}. Available: {available}")
            return 1
        result = compare_job_to_experiment(
            by_name[args.job],
            experimental_path=args.experimental,
            shift_eV=args.shift,
            computed_intensity_col=args.computed_intensity_col,
            experimental_energy_col=args.exp_energy_col,
            experimental_intensity_col=args.exp_intensity_col,
        )
    print(comparison_report_table([result]))

    if args.output_csv:
        result.to_csv(args.output_csv)
        print(f"Saved comparison data (metrics + aligned spectra): {args.output_csv}")

    if args.plot:
        from .comparison import plot_comparison

        plot_comparison(result, out_path=args.plot)
        print(f"Saved comparison plot: {args.plot}")

    return 0


def _cmd_compare_batch(args: argparse.Namespace) -> int:
    _, jobs, _ = load_config(args.config)
    entries = compare_batch(
        jobs,
        experimental_dir=args.experimental_dir,
        shift_eV=args.shift,
        computed_intensity_col=args.computed_intensity_col,
        experimental_energy_col=args.exp_energy_col,
        experimental_intensity_col=args.exp_intensity_col,
        save_csv=not args.no_csv,
        save_plot=not args.no_plot,
    )
    print(comparison_batch_report_table(entries))

    if args.summary_csv:
        write_batch_summary_csv(entries, args.summary_csv)
        print(f"\nSaved batch summary: {args.summary_csv}")

    n_ok = sum(1 for e in entries if e.result is not None)
    n_skipped = len(entries) - n_ok
    print(f"\n{n_ok} compared, {n_skipped} skipped, {len(entries)} total.")
    return 1 if n_skipped else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdmnes_pipeline",
        description="FDMNES input generation / execution / validation for Ni-HHTP & Co-HHTP",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_arg(p):
        p.add_argument("--config", required=True, type=Path, help="Path to jobs JSON config")

    def add_run_args(p):
        p.add_argument("--executable", required=True, type=Path, help="Path to the FDMNES executable")
        p.add_argument("--mpirun", default=None, help="Path to mpirun, to run in parallel")
        p.add_argument("--np", type=int, default=1, help="Number of MPI processes (with --mpirun)")
        p.add_argument("--timeout", type=float, default=None, help="Per-job timeout in seconds")
        p.add_argument(
            "--stop-on-first-failure",
            action="store_true",
            help="Stop the batch as soon as one job fails/times out",
        )

    p_generate = sub.add_parser("generate", help="Generate FDMNES input files only")
    add_config_arg(p_generate)
    p_generate.set_defaults(func=_cmd_generate)

    p_run = sub.add_parser("run", help="Run FDMNES for every job in the config")
    add_config_arg(p_run)
    add_run_args(p_run)
    p_run.set_defaults(func=lambda a: 0 if _cmd_run(a) is not None else 1)

    p_validate = sub.add_parser("validate", help="Validate existing FDMNES outputs")
    add_config_arg(p_validate)
    p_validate.add_argument(
        "--no-convolution-check",
        action="store_true",
        help="Don't require a _conv.txt file (e.g. Convolution wasn't used)",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_pipeline = sub.add_parser("pipeline", help="generate + run + validate in one go")
    add_config_arg(p_pipeline)
    add_run_args(p_pipeline)
    p_pipeline.add_argument(
        "--no-convolution-check",
        action="store_true",
        help="Don't require a _conv.txt file (e.g. Convolution wasn't used)",
    )
    p_pipeline.set_defaults(func=_cmd_pipeline)

    p_compare = sub.add_parser(
        "compare",
        help="Benchmark a computed spectrum against an experimental spectrum",
        description=(
            "Two ways to point at the computed spectrum: "
            "(1) --conv-file <path>, for a _conv.txt you already have on hand -- no config needed; or "
            "(2) --config <path> --job <name>, to look it up from a JSON batch config's job.filout."
        ),
    )
    p_compare.add_argument(
        "--conv-file", type=Path, default=None,
        help="Path to an already-completed FDMNES _conv.txt file (use this instead of --config/--job)",
    )
    p_compare.add_argument("--config", type=Path, default=None, help="Path to jobs JSON config (use with --job)")
    p_compare.add_argument("--job", default=None, help="Job name from the config to compare (use with --config)")
    p_compare.add_argument(
        "--name", default=None,
        help="Label for the result/plot when using --conv-file (default: the file's name)",
    )
    p_compare.add_argument(
        "--experimental", required=True, type=Path, help="Path to a 2-column (energy, intensity) experimental spectrum"
    )
    p_compare.add_argument(
        "--shift",
        default="auto",
        type=_parse_shift,
        help="'auto' to align computed/experimental peak positions (default), or a fixed eV shift, e.g. 2.5",
    )
    p_compare.add_argument("--computed-intensity-col", type=int, default=1, help="Intensity column in the _conv.txt file (0-indexed)")
    p_compare.add_argument("--exp-energy-col", type=int, default=0, help="Energy column in the experimental file (0-indexed)")
    p_compare.add_argument("--exp-intensity-col", type=int, default=1, help="Intensity column in the experimental file (0-indexed)")
    p_compare.add_argument(
        "--output-csv", type=Path, default=None,
        help="Save the fit metrics (Pearson r, R-factor, shift) and the aligned/normalized spectra to this CSV path",
    )
    p_compare.add_argument("--plot", type=Path, default=None, help="Save an overlay plot to this path (requires matplotlib)")
    p_compare.set_defaults(func=_cmd_compare)

    p_compare_batch = sub.add_parser(
        "compare-batch",
        help="Benchmark every job in a config against its matching experimental spectrum",
        description=(
            "Compares every job in --config against an experimental file matched by naming "
            "convention: a job whose FDMNES output is e.g. 'NiHHTP_Pristine' matches an "
            "experimental file named 'NiHHTP_Pristine_EXP' (any extension). Saves each job's "
            "comparison CSV and overlay plot into that job's own output folder, next to its "
            "_conv.txt. Jobs with no computed spectrum yet, or no matching experimental file, "
            "are skipped (reported, not an error) -- the rest of the batch still runs."
        ),
    )
    add_config_arg(p_compare_batch)
    p_compare_batch.add_argument(
        "--experimental-dir", type=Path, default=None,
        help="Search this one shared folder for '<job>_EXP.*' files, instead of each job's own output folder",
    )
    p_compare_batch.add_argument(
        "--shift",
        default="auto",
        type=_parse_shift,
        help="'auto' to align computed/experimental peak positions (default), or a fixed eV shift, e.g. 2.5 -- applied to every job",
    )
    p_compare_batch.add_argument("--computed-intensity-col", type=int, default=1, help="Intensity column in each _conv.txt file (0-indexed)")
    p_compare_batch.add_argument("--exp-energy-col", type=int, default=0, help="Energy column in the experimental files (0-indexed)")
    p_compare_batch.add_argument("--exp-intensity-col", type=int, default=1, help="Intensity column in the experimental files (0-indexed)")
    p_compare_batch.add_argument(
        "--no-csv", action="store_true", help="Don't save a per-job comparison CSV (saved by default)"
    )
    p_compare_batch.add_argument(
        "--no-plot", action="store_true",
        help="Don't save per-job overlay plots (saved by default; needs matplotlib -- silently skipped if it's not installed)",
    )
    p_compare_batch.add_argument(
        "--summary-csv", type=Path, default=None,
        help="Also write one combined summary row per job (for ranking across the whole batch) to this CSV path",
    )
    p_compare_batch.set_defaults(func=_cmd_compare_batch)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
