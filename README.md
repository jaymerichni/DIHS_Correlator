# DIHS Correlator

DIHS Correlator is a Python research package for interpretable tephra-source correlation using the Depth-Integrated Harmonic Score (DIHS) framework.

The repository implements:
- recursive clustering-based correlation of unknown samples against candidate source classes,
- DIHS computation across depth levels,
- perturbative uncertainty propagation,
- pseudo-unknown resolvedness calibration,
- post-hoc reporting and visualization from saved outputs.

## Scientific Scope

The package is designed for compositional geochemical datasets where each row represents a sample and one column encodes class identity (for example, volcanic source or stratigraphic unit). The framework estimates class affinity by integrating harmonic-score behavior across recursive partition depth.

## Repository Structure

- `api.py`: public, user-facing entry points.
- `core/`: fundamental algorithms (transformations, recursive clustering, DIHS metrics).
- `workflows/`: orchestration logic for complete analyses.
- `viz/`: plotting utilities (HS curves, pairwise matrices, resolvedness plots).
- `io/`: path and writer/loading helpers.
- `tests/`: regression tests for API delegation and workflow analysis behavior.

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

## Input Expectations

- Input object: `pandas.DataFrame`.
- Class/label column: configurable (default `controlcode`).
- Feature columns: numeric columns not excluded by `exclude_columns`.
- Transformation options: `none`, `ilr`, `clr`, `scaled`.

For perturbative workflows, major and trace perturbation columns can be passed explicitly. If omitted, default normalized geochemical column names are resolved when present.

## Output Conventions

Depending on function and flags:
- in-memory `DataFrame` outputs (default),
- optional detailed dictionaries (`return_details=True`),
- optional CSV/plot artifacts (`write_files=True`, `plot_everything=True`).

Typical output directories include:
- `Results*` folders for metrics and trees,
- `Plots` folders for SVG figures,
- model-specific subfolders for triple workflows.

## Methodological Notes

- Non-deterministic models (`kmeans`, `gaussian`) accept `random_state` for reproducibility.
- Perturbative summaries are computed at the maximum common depth across iterations.
- Resolvedness calibration supports target precision levels and threshold reporting.

## Reproducibility and Validation

The repository includes unit/regression tests under `tests/`. A standard test invocation is:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Citation

When using this software in scientific work, cite the corresponding DIHS methodology publication and include the software version/commit used for analysis.
