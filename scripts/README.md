# Reproducibility Scripts

This directory contains the notebooks and scripts used to reproduce the analyses presented in the manuscript **"A New Machine Learning Approach for Interpretable Tephra-Source Correlation: Introducing the Depth-Integrated Harmonic Score (DIHS)"**.

With one stated exception, the materials in this folder were prepared as original reproducibility workflows for the manuscript. The exception is `0_raw_data_preprocessing.ipynb`, which is an adapted preprocessing workflow based on Petrelli et al., 2017 and is used here to reconstruct the benchmark input dataset from the raw source files bundled in `data/raw/`.

## Contents and manuscript linkage

1. `0_raw_data_preprocessing.ipynb`

Prepares the Italian benchmark dataset from the raw source tables in `data/raw/`, performs the main cleaning and harmonization steps, and writes the processed coupled dataset used by the downstream case-study analyses. This notebook supports the data-preparation stage behind the Italian benchmark case study in the manuscript.

2. `1a_synthetic_scenario_gen.ipynb`

Generates the six synthetic scenarios used to illustrate DIHS behavior under controlled geometric configurations. This notebook supports the manuscript's synthetic validation section and the synthetic scenario material discussed in Supporting Text S3.

3. `1b_synthetic_scenario_comparison.py`

Runs the synthetic benchmarking comparison between DIHS and the alternative distance-based baselines used in the manuscript. It reproduces the comparative analysis associated with the synthetic validation section and the benchmark framing described in Supporting Text S2.

4. `2_caio_source_attribution.ipynb`

Reproduces the Italian case study centered on the Caio outcrop and its relationship to the Roman Magmatic Province and related volcanic provinces. This notebook supports the main manuscript section devoted to empirical tephra-source attribution using the Italian benchmark dataset.

5. `3_sensitivity_data_size.py`

Evaluates how pseudo-unknown calibration results change as the available benchmark dataset is reduced in size while preserving class proportions. This script supports the manuscript's discussion of robustness, sample-size sensitivity, and resolvedness stability.

## Notes for reproducibility

- These workflows are intended to be run from this repository layout so that the bundled `data/` and `scripts/` directories resolve correctly through relative paths.
- The notebooks and scripts in this folder are included to make the analytical path from raw or processed inputs to manuscript results transparent and inspectable.
- When using `0_raw_data_preprocessing.ipynb` or the raw files in `data/raw/`, please cite both this DIHS manuscript and the original Petrelli et al., 2017 source described in `data/raw/README.md`.
