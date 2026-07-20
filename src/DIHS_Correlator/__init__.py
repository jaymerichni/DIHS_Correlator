"""DIHS-based tephra correlation package."""

__version__ = "0.1.0"

from DIHS_Correlator.api import (
    calibrate_perturbative_resolvedness_from_outputs,
    perturbative_simple_run,
    perturbative_triple_run,
    perturbative_triple_run_with_resolvedness,
    plot_pseudo_unknown_margin_histogram_from_outputs,
    plot_pseudo_unknown_margin_from_outputs,
    pseudo_unknown_run,
    simple_run,
    triple_run,
)

__all__ = [
    "__version__",
    "simple_run",
    "triple_run",
    "perturbative_simple_run",
    "perturbative_triple_run",
    "perturbative_triple_run_with_resolvedness",
    "calibrate_perturbative_resolvedness_from_outputs",
    "plot_pseudo_unknown_margin_histogram_from_outputs",
    "plot_pseudo_unknown_margin_from_outputs",
    "pseudo_unknown_run",
]
