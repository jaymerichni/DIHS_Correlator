# Data Overview

This directory contains the bundled synthetic validation data together with the placeholder directories and documentation needed to reconstruct the Italian benchmark dataset locally.

## Layout

- `raw/`: acquisition instructions for the upstream source files used to reconstruct the Italian benchmark dataset. The raw files themselves are not redistributed here.
- `processed/caio_italy_benchmark/`: placeholder directory where the locally generated Italian benchmark dataset should be written after running `scripts/0_raw_data_preprocessing.ipynb`.
- `processed/synthetic_scenarios/`: generated synthetic validation dataset and overview figure used by the synthetic comparison workflow.

The raw-source acquisition notes are documented in [raw/README.md](raw/README.md).

## Bundled and local files

- `processed/synthetic_scenarios/all_scenarios_combined.csv`
- `processed/synthetic_scenarios/synthetic_scenarios.svg`

The synthetic scenario files above are bundled in the repository. The Italian benchmark file `processed/caio_italy_benchmark/full_italian_data.csv` is intentionally not redistributed; create it locally after following the instructions in `raw/README.md`.
