"""Refactored tephra correlation package.

This package keeps the same computational behavior as the original scripts,
while reorganizing the code into modular layers.
"""

from Tephra_Correlator_Refactored.api import (
    calibrate_perturbative_resolvedness_from_outputs,
    method_comparison_run,
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
