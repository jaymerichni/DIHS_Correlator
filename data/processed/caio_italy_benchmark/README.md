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

Expected structure of the generated CSV:

- 3907 rows total.
- 17 geochemical variables plus `lettercode`.
- 17 `Caio` rows and 3890 candidate-source rows.
- No accidental index column such as `Unnamed: 0`.

Expected geochemical variables:

- `SIO2N`, `TIO2N`, `AL2O3N`, `FE2O3TN`, `CAON`, `MGON`, `MNON`, `NA2ON`, `K2ON`, `P2O5N`, `NbN`, `ZrN`, `LaN`, `CeN`, `SrN`, `BaN`, `RbN`

Quick validation snippet:

```python
import pandas as pd

df = pd.read_csv("data/processed/caio_italy_benchmark/full_italian_data.csv")
df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

expected_cols = {
   "SIO2N", "TIO2N", "AL2O3N", "FE2O3TN", "CAON", "MGON", "MNON",
   "NA2ON", "K2ON", "P2O5N", "NbN", "ZrN", "LaN", "CeN", "SrN", "BaN", "RbN", "lettercode"
}
assert len(df) == 3907
assert set(df.columns) == expected_cols
assert (df["lettercode"] == "Caio").sum() == 17
assert not any(str(c).startswith("Unnamed:") for c in df.columns)
print(df["lettercode"].value_counts().sort_index())
```

This locally generated CSV is required by:

- `scripts/2_caio_source_attribution.ipynb`
- `scripts/3_sensitivity_data_size.py`
