# Data Overview

This directory contains the raw and processed datasets bundled with the `DIHS_Correlator` reproducibility archive.

## Layout

- `raw/`: upstream source files used to reconstruct the Italian benchmark dataset.
- `processed/caio_italy_benchmark/`: processed Italian benchmark dataset used by the Caio case-study notebook and the dataset-size sensitivity script.
- `processed/synthetic_scenarios/`: generated synthetic validation dataset and overview figure used by the synthetic comparison workflow.

The raw-source provenance narrative is documented in [raw/README.md](raw/README.md).

## Raw input checksums

The following SHA-256 checksums were recorded from the repository snapshot on 2026-07-22.

| File | SHA-256 |
|---|---|
| `raw/georock-data.csv` | `CE0EE5DB593C74B72A5B020D5BB304FF39BEF11BA5E3432DF27DDB4E68459EB3` |
| `raw/Results_Caio.xlsx` | `2DC2F9483ABF76BE309B9AB4B467054EB2CFD57B39138B3B497C205B8E746499` |

## Processed and generated files

- `processed/caio_italy_benchmark/full_italian_data.csv`
- `processed/synthetic_scenarios/all_scenarios_combined.csv`
- `processed/synthetic_scenarios/synthetic_scenarios.svg`

These files are the direct inputs consumed by the bundled manuscript workflows in `scripts/`.
