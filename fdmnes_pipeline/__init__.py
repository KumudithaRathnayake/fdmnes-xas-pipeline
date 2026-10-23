"""
fdmnes_pipeline
================

Automation pipeline for FDMNES XANES simulations, built for the Ni-HHTP /
Co-HHTP 2D MOF project.

This package implements the following stages of the project plan:

    Structures ready
         |
    [ FDMNES INPUT GENERATOR ]   <- input_generator.py  (frozen format)
         |
    Test generated input manually
         |
    [ AUTOMATED FDMNES RUNNER ]  <- runner.py
         |
    [ OUTPUT VALIDATION ]        <- validation.py
         |
    [ BENCHMARK VS EXPERIMENT ]  <- comparison.py (Pearson r + R-factor,
                                     mirrors Chen et al. Chem. Mater. 2024)
         |
    [ BATCH BENCHMARKING ]       <- comparison.py's compare_batch: every
                                     job in a config vs. its matching
                                     "<job>_EXP.<ext>" experimental file
         |
    Region-weighted scoring (pre-edge/white-line/...)   <- not built yet

``comparison.py`` covers "connect existing comparison code": normalize,
energy-align (auto peak-matching or a manual shift), and score with
Pearson correlation and R-factor, for one job or -- via ``compare_batch``
-- every job in a config at once, each matched to its own experimental
file by filename convention and saved into its own output folder.
Region-weighted (pre-edge/white-line) scoring isn't built yet -- add it
on top of ``compare_job_to_experiment``/``compare_batch`` if needed.
"""

from .config import ProtocolSettings, AbsorberSettings
from .models import JobSpec, MoleculeStructure, RunResult, ValidationResult, RunStatus
from .input_generator import build_fdmnes_input, generate_input_file, generate_batch
from .runner import FdmnesRunner
from .validation import validate_run, validate_batch
from .xyz_parser import parse_xyz
from .comparison import (
    BatchComparisonEntry,
    ComparisonResult,
    align_and_normalize,
    compare_batch,
    compare_conv_file_to_experiment,
    compare_job_to_experiment,
    find_experimental_file,
    load_computed_spectrum,
    load_computed_spectrum_from_file,
    load_experimental_spectrum,
    plot_comparison,
    write_batch_summary_csv,
)
from .comparison import batch_report_table as comparison_batch_report_table
from .comparison import report_table as comparison_report_table

__all__ = [
    "ProtocolSettings",
    "AbsorberSettings",
    "JobSpec",
    "MoleculeStructure",
    "RunResult",
    "ValidationResult",
    "RunStatus",
    "build_fdmnes_input",
    "generate_input_file",
    "generate_batch",
    "FdmnesRunner",
    "validate_run",
    "validate_batch",
    "parse_xyz",
    "ComparisonResult",
    "BatchComparisonEntry",
    "align_and_normalize",
    "compare_batch",
    "compare_conv_file_to_experiment",
    "compare_job_to_experiment",
    "find_experimental_file",
    "load_computed_spectrum",
    "load_computed_spectrum_from_file",
    "load_experimental_spectrum",
    "plot_comparison",
    "write_batch_summary_csv",
    "comparison_batch_report_table",
    "comparison_report_table",
]

__version__ = "0.1.0"
