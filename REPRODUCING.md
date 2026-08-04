# Reproducing The Bundled Workflows

This repository contains both the installable `DIHS_Correlator` package and the notebooks and scripts used to reproduce the manuscript workflows. Run the commands below from the repository root unless noted otherwise.

## Environment setup

If you want a recorded environment snapshot, create a Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate dihs-correlator-reproducibility
python -m pip install -e .
```

If you prefer a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[reproducibility,test]"
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[reproducibility,test]"
```

## Execution order

The synthetic validation inputs are bundled under `data/processed/synthetic_scenarios/`. The Petrelli/GEOROC-derived Italian benchmark inputs and the derived `full_italian_data.csv` file are not redistributed in this repository.

Before running the Italian benchmark workflows:

1. Follow the acquisition notes in `data/raw/README.md`.
2. Place local copies of the benchmark source files at `data/raw/georock-data.csv` and `data/raw/Results_Caio.xlsx`.
3. Run `scripts/0_raw_data_preprocessing.ipynb` to generate `data/processed/caio_italy_benchmark/full_italian_data.csv`.

The generated benchmark CSV is expected to contain 3907 rows, the 17 normalized
geochemical variables plus `lettercode`.

```bash
jupyter nbconvert --execute --to notebook --inplace scripts/0_raw_data_preprocessing.ipynb
jupyter nbconvert --execute --to notebook --inplace scripts/1a_synthetic_scenario_gen.ipynb
python scripts/1b_synthetic_scenario_comparison.py
jupyter nbconvert --execute --to notebook --inplace scripts/2_caio_source_attribution.ipynb
python scripts/3_sensitivity_data_size.py
```

## Workflow summary

| Workflow | Purpose | Inputs | Main outputs | Seed / determinism notes |
|---|---|---|---|---|
| `scripts/0_raw_data_preprocessing.ipynb` | Rebuild the processed Italian benchmark dataset from user-supplied raw source files. | Local copies of `data/raw/georock-data.csv` and `data/raw/Results_Caio.xlsx` obtained as described in `data/raw/README.md` | `data/processed/caio_italy_benchmark/full_italian_data.csv` | Deterministic data-preparation workflow. |
| `scripts/1a_synthetic_scenario_gen.ipynb` | Generate the synthetic validation scenarios used in the manuscript. | Notebook-defined synthetic parameters | `data/processed/synthetic_scenarios/all_scenarios_combined.csv`, `data/processed/synthetic_scenarios/synthetic_scenarios.svg` | Deterministic scenario construction with notebook-controlled parameters. |
| `scripts/1b_synthetic_scenario_comparison.py` | Compare DIHS against centroid-distance and Mahalanobis-distance baselines across scenarios and unknown sample sizes. | `data/processed/synthetic_scenarios/all_scenarios_combined.csv` | `results/1_benchmarking_comparison/` | Uses `RANDOM_STATE = 42`. |
| `scripts/2_caio_source_attribution.ipynb` | Reproduce the Caio benchmark case study and resolvedness analysis across coupled, major-only, and trace-only feature spaces. | Locally generated `data/processed/caio_italy_benchmark/full_italian_data.csv` | `results/2_caio_source_attribution/{coupled,major_only,trace_only}/` | Deterministic unless notebook parameters are changed interactively. |
| `scripts/3_sensitivity_data_size.py` | Evaluate pseudo-unknown behavior across dataset-size reductions and sample-size sweeps. | Locally generated `data/processed/caio_italy_benchmark/full_italian_data.csv` | `results/3_sensitivity_data_size/` | Uses `RANDOM_STATE = 12345`. |

## Practical notes

- The benchmark comparison and dataset-size sensitivity scripts are the heaviest workflows in the archive and may take substantially longer than the notebooks depending on CPU count and BLAS configuration.
- The notebooks in `scripts/` are stored without trusted execution state so that stale errors and path-specific outputs are not preserved in the release snapshot.
- The Caio source-attribution notebook and the dataset-size sensitivity script require the locally generated Italian benchmark CSV and will not run until the preprocessing step has been completed.
- If you change output locations, keep the working-directory assumptions aligned with the paths documented in the notebooks and scripts.
