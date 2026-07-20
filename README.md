# DIHS Tephra Correlator

DIHS Tephra Correlator is a Python research package for interpretable tephra-source correlation using the Depth-Integrated Harmonic Score (DIHS) framework (Aymerich et al., 2026). The installable Python package is named `DIHS_Correlator`.

The repository implements:
- recursive clustering-based correlation of unknown samples against candidate source classes,
- DIHS computation across depth levels,
- perturbative uncertainty propagation,
- pseudo-unknown resolvedness calibration,
- post-hoc reporting and visualization from saved outputs,
- a packaged local Flask interface for browser-based analysis.

## Scientific Scope

The package is designed for compositional geochemical datasets where each row represents a sample and one column encodes class identity, for example a volcanic source or unit. The framework estimates class affinity by integrating harmonic-score behavior across recursive partition depth.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

After installation, `import DIHS_Correlator` works from any working directory.

## Repository Structure

- `src/DIHS_Correlator/api.py`: public, user-facing entry points.
- `src/DIHS_Correlator/core/`: core transformations, recursive clustering, and DIHS metrics.
- `src/DIHS_Correlator/workflows/`: orchestration logic for complete analyses.
- `src/DIHS_Correlator/viz/`: plotting utilities for HS curves, pairwise matrices, and pseudo-unknown diagnostics.
- `src/DIHS_Correlator/io/`: output-path and file-writing helpers.
- `src/DIHS_Correlator/web/`: packaged Flask application, Jinja template, static front-end assets, and module entrypoint.
- `pyproject.toml`: packaging metadata for editable and build installs.

## Analysis Modes

1. **Single run (`simple_run`)**
- One model (`agglomerative`, `kmeans`, or `gaussian`) on one dataset.
- Returns depth-resolved HS outputs and integrated DIHS outputs.

2. **Triple run (`triple_run`)**
- Executes all three models under a shared configuration.
- Returns combined outputs plus per-model artifacts when requested.

3. **Perturbative single/triple (`perturbative_simple_run`, `perturbative_triple_run`)**
- Propagates measurement uncertainty via repeated perturbation.
- Produces ensemble summaries, Top-1 frequencies, and margin statistics.
- Accepts `integration_depth` to force both DIHS summaries and ensemble pairwise plots to use the same cumulative depth.

4. **Pseudo-unknown calibration (`pseudo_unknown_run`)**
- Performs controlled positive/negative pseudo-unknown experiments.
- Estimates threshold-dependent resolvedness behavior.

5. **Integrated resolvedness workflow (`perturbative_triple_run_with_resolvedness`)**
- Combines perturbative triple analysis with Top-1 pseudo-unknown validation.
- Reports empirical resolvedness by model at a shared integration depth.

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

## Minimal Usage Example

```python
import pandas as pd
from DIHS_Correlator import simple_run

df = pd.read_csv("geochemistry.csv")

hs_per_depth = simple_run(
    df=df,
    model_type="kmeans",
    transform_type="clr",
    class_column="controlcode",
    unknown_sample="Unknown_A",
    random_state=42,
    compute_pairwise=True,
    write_files=False,
    plot_everything=False,
)
```

## Graphical Interface

This repository includes a packaged Flask app for running the main workflows from a browser.

```bash
python -m pip install -e .
dihs-tephra-correlator
```

An equivalent module-based launch command is also available:

```bash
python -m DIHS_Correlator.web
```

The app starts a local server on `http://127.0.0.1:5000` by default. It currently exposes:
- `simple_run`
- `triple_run`
- `perturbative_simple_run`
- `perturbative_triple_run`
- `perturbative_triple_run_with_resolvedness`

The browser UI is designed for local, single-user runs. It provides progress tracking for perturbative and resolvedness workflows, collapsible result sections, and a zoom/pan plot viewer. Relative output directories entered in the form are resolved from the directory where you launch the app.

If you want the banner to show the archived DOI after a Zenodo release, start the app with `DIHS_CORRELATOR_SOFTWARE_DOI` set in the environment.

## Input Expectations

- Input object: `pandas.DataFrame`.
- Class/label column: configurable, default `controlcode`.
- Feature columns: numeric columns not excluded by `exclude_columns`.
- Transformation options: `none`, `ilr`, `clr`, `scaled`.

For perturbative workflows, major and trace perturbation columns can be passed explicitly. If omitted, default normalized geochemical column names are resolved when present.

## Output Conventions

Depending on the function and flags, workflows can return:
- in-memory `DataFrame` outputs,
- detailed dictionaries when `return_details=True`,
- optional CSV and plot artifacts when `write_files=True` and `plot_everything=True`.

Typical output directories include `Results*` folders for metrics and trees, `Plots` folders for SVG figures, and model-specific subfolders for triple workflows.

## Methodological Notes

- Non-deterministic models (`kmeans`, `gaussian`) accept `random_state` for reproducibility.
- Perturbative summaries are computed at the maximum common depth across iterations by default. Passing `integration_depth` to perturbative workflows forces both reported DIHS summaries and ensemble pairwise DIHS matrices to use that same cumulative depth.
- Resolvedness calibration supports target precision levels and threshold reporting.
