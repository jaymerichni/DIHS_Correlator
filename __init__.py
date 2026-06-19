"""DIHS-based tephra correlation package.
"""

from DIHS_Correlator.api import (
    calibrate_perturbative_resolvedness_from_outputs,
    perturbative_simple_run,
    perturbative_triple_run,
    plot_pseudo_unknown_margin_histogram_from_outputs,
    plot_pseudo_unknown_margin_from_outputs,
    pseudo_unknown_run,
    simple_run,
    triple_run,
)

__all__ = [
    "simple_run",
    "triple_run",
    "perturbative_simple_run",
    "perturbative_triple_run",
    "calibrate_perturbative_resolvedness_from_outputs",
    "plot_pseudo_unknown_margin_histogram_from_outputs",
    "plot_pseudo_unknown_margin_from_outputs",
    "pseudo_unknown_run",
    "method_comparison_run",
]
