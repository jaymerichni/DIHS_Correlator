"""Compatibility package entrypoint for importing the repository as DIHS_Correlator."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__path__ = [str(_ROOT)] + list(__path__)

from .api import (  # noqa: E402,F401
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
