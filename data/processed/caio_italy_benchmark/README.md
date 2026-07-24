# Local Benchmark Dataset

This directory is intentionally shipped without `full_italian_data.csv`.

That file is derived from external benchmark inputs associated with Petrelli et al. (2017) and must be generated locally rather than redistributed as part of this repository snapshot.

## To create `full_italian_data.csv`

1. Follow the acquisition instructions in [../../raw/README.md](../../raw/README.md).
2. Place local copies of the required source files at:
   - `data/raw/georock-data.csv`
   - `data/raw/Results_Caio.xlsx`
3. Run `scripts/0_raw_data_preprocessing.ipynb`.

The preprocessing notebook writes the resulting benchmark file here as:

- `data/processed/caio_italy_benchmark/full_italian_data.csv`

This locally generated CSV is required by:

- `scripts/2_caio_source_attribution.ipynb`
- `scripts/3_sensitivity_data_size.py`
