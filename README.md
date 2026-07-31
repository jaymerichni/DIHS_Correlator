# DIHS Tephra Correlator

DIHS Tephra Correlator is a Python package and Flask web application for interpretable tephra-source correlation using the Depth-Integrated Harmonic Score (DIHS) framework.

## Installation

Install the package from the repository root:

```bash
python -m pip install -e .
```

For the smoke tests used in this branch:

```bash
python -m pip install -e ".[test]"
```

## Repository Structure

- `src/DIHS_Correlator/api.py`: public entry points for the DIHS workflows.
- `src/DIHS_Correlator/core/`: clustering, transforms, and DIHS calculations.
- `src/DIHS_Correlator/workflows/`: orchestration for single, triple, perturbative, and resolvedness runs.
- `src/DIHS_Correlator/viz/`: plot generation for HS curves, pairwise matrices, and pseudo-unknown diagnostics.
- `src/DIHS_Correlator/io/`: output-path and file-writing helpers.
- `src/DIHS_Correlator/web/`: Flask application, Jinja templates, static assets, and module entrypoint.
- `tests/`: smoke tests covering importability, deterministic toy execution, packaged assets, and deployment path guards.

## Running the Web Interface

Local launcher:

```bash
dihs-tephra-correlator
```

Equivalent module entrypoint:

```bash
python -m DIHS_Correlator.web
```

The built-in launcher reads:

- `DIHS_CORRELATOR_HOST` or `HOST`
- `DIHS_CORRELATOR_PORT` or `PORT`

If neither is set, it starts on `127.0.0.1:5000`.

## Available Workflows

- `simple_run`: one clustering model on one dataset.
- `triple_run`: all three clustering models under a shared configuration.
- `perturbative_simple_run`: uncertainty propagation for one model.
- `perturbative_triple_run`: uncertainty propagation for all three models.
- `perturbative_triple_run_with_resolvedness`: perturbative triple run plus pseudo-unknown resolvedness calibration.
- `pseudo_unknown_run`: direct pseudo-unknown threshold calibration from saved or in-memory inputs.

## Input Expectations

- Inputs are `pandas.DataFrame` objects or uploaded CSV files.
- One column must identify the class or source label, typically `controlcode`.
- Numeric columns are used as candidate features unless excluded.
- Supported transforms are `none`, `ilr`, `clr`, and `scaled`.

For perturbative workflows, major and trace perturbation columns can be supplied explicitly. If omitted, the package tries to infer the expected geochemical columns.

## Public API

```python
from DIHS_Correlator import (
    simple_run,
    triple_run,
    perturbative_simple_run,
    perturbative_triple_run,
    perturbative_triple_run_with_resolvedness,
    pseudo_unknown_run,
    plot_pseudo_unknown_margin_from_outputs,
    plot_pseudo_unknown_margin_histogram_from_outputs,
    calibrate_perturbative_resolvedness_from_outputs,
)
```

## License

This software is released under the BSD 3-Clause License.
