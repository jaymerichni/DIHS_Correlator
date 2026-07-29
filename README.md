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

From the repository root, install the base package with:

```bash
python -m pip install -e .
```

After installation, `import DIHS_Correlator` works from any working directory.

For the bundled notebooks, manuscript workflows, and release checks, install the reproducibility extras:

```bash
python -m pip install -e ".[reproducibility,test]"
```

## Repository Structure

- `src/DIHS_Correlator/api.py`: public, user-facing entry points.
- `src/DIHS_Correlator/core/`: core transformations, recursive clustering, and DIHS metrics.
- `src/DIHS_Correlator/workflows/`: orchestration logic for complete analyses.
- `src/DIHS_Correlator/viz/`: plotting utilities for HS curves, pairwise matrices, and pseudo-unknown diagnostics.
- `src/DIHS_Correlator/io/`: output-path and file-writing helpers.
- `src/DIHS_Correlator/web/`: packaged Flask application, Jinja template, static front-end assets, and module entrypoint.
- `data/raw/`: acquisition notes for third-party benchmark inputs that are not redistributed in this repository.
- `data/processed/`: bundled synthetic scenario outputs plus a placeholder directory for the locally generated Italian benchmark dataset.
- `scripts/`: notebooks and scripts used to reproduce the main manuscript examples and benchmark workflows.
- `pyproject.toml`: packaging metadata for editable and build installs.

## Reproducibility Scripts

The repository includes the synthetic scenario data and the analysis scripts used to reproduce the manuscript workflows. The Petrelli/GEOROC-derived Italian benchmark inputs and the derived `full_italian_data.csv` file are not redistributed here; acquisition and reconstruction notes are provided under `data/raw/` and `data/processed/caio_italy_benchmark/`.

1. `scripts/1a_synthetic_scenario_gen.ipynb`
- Generates the six two-dimensional synthetic scenarios discussed in the synthetic validation section and Supporting Text S2.
- Writes the combined benchmark table and overview figure to `data/processed/synthetic_scenarios/`.

2. `scripts/1b_synthetic_scenario_comparison.py`
- Benchmarks DIHS against the centroid-distance and Mahalanobis-distance baselines described in Supporting Text S3.
- Repeats the comparison across the synthetic scenarios, the three clustering models (`agglomerative`, `kmeans`, `gaussian`), and a sweep of unknown sample sizes, matching the manuscript's sample-size sensitivity framing.
- By default, reads `data/processed/synthetic_scenarios/all_scenarios_combined.csv` and writes its benchmark outputs under `results/1_benchmarking_comparison/`.

3. `scripts/2_caio_source_attribution.ipynb`
- Reproduces the Italian benchmark case study centered on the Caio outcrop and the Roman Magmatic Province association.
- Uses a locally generated `data/processed/caio_italy_benchmark/full_italian_data.csv`, removes accidental `Unnamed:` index columns defensively, and runs perturbative triple workflows plus pseudo-unknown resolvedness calibration for three feature spaces.
- Writes analysis outputs to `results/2_caio_source_attribution/coupled/`, `results/2_caio_source_attribution/major_only/`, and `results/2_caio_source_attribution/trace_only/`.

4. `scripts/3_sensitivity_data_size.py`
- Runs pseudo-unknown sensitivity experiments on the coupled Italian dataset across four stratified dataset sizes (`25%`, `50%`, `75%`, `100%`) and a predefined grid of pseudo-unknown sample sizes.
- Uses a locally generated `data/processed/caio_italy_benchmark/full_italian_data.csv`, the three clustering models (`agglomerative`, `kmeans`, `gaussian`), and writes one result folder per dataset size under `results/3_sensitivity_data_size/`, plus combined summary tables inside each dataset-size folder.
- This script is intended to support the manuscript's practical discussion of how sample size influences margin stability and resolvedness behavior.

These scripts are intended to be run from this repository checkout. The notebooks and the benchmarking script resolve paths against the repository layout directly so the bundled `data/` and `scripts/` folders stay portable inside an archival snapshot. For the Italian benchmark workflows, first follow `data/raw/README.md` and `data/processed/caio_italy_benchmark/README.md` to obtain the raw inputs and generate the local benchmark CSV.

Typical usage from the repository root:

```bash
python -m pip install -e .
python scripts/1b_synthetic_scenario_comparison.py
```

For the notebooks, open `scripts/1a_synthetic_scenario_gen.ipynb` and `scripts/2_caio_source_attribution.ipynb` in Jupyter or VS Code after the editable install.

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

## Reproducing

For command-by-command execution of the bundled notebooks and scripts, see [REPRODUCING.md](REPRODUCING.md). Additional workflow-specific notes are provided in [scripts/README.md](scripts/README.md) and [data/README.md](data/README.md).

## License

This software is released under the BSD 3-Clause License. See [LICENSE](LICENSE).

External benchmark inputs referenced in `data/raw/README.md` and the derived
`data/processed/caio_italy_benchmark/full_italian_data.csv` are not redistributed in this repository.
Those external data remain governed by their original source terms and citation requirements.

## Citation

Software citation metadata for the repository snapshot are stored in [CITATION.cff](CITATION.cff). Final release-only fields such as a Zenodo DOI and archival release date should be added only when the release is published.
