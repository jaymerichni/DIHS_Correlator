# Data Overview

This directory contains the raw and processed datasets bundled with the `DIHS_Correlator` reproducibility archive.

## Layout

- `raw/`: upstream source files used to reconstruct the Italian benchmark dataset.
- `processed/caio_italy_benchmark/`: processed Italian benchmark dataset used by the Caio case-study notebook and the dataset-size sensitivity script.
- `processed/synthetic_scenarios/`: generated synthetic validation dataset and overview figure used by the synthetic comparison workflow.

The raw-source provenance narrative is documented in [raw/README.md](raw/README.md).

## Processed and generated files

- `processed/caio_italy_benchmark/full_italian_data.csv`
- `processed/synthetic_scenarios/all_scenarios_combined.csv`
- `processed/synthetic_scenarios/synthetic_scenarios.svg`

These files are the direct inputs consumed by the bundled manuscript workflows in `scripts/`.
