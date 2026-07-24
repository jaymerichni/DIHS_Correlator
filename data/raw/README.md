# Raw Data Acquisition

This directory is intentionally distributed without the Petrelli/GEOROC-derived benchmark source files. The repository keeps the expected directory structure for reproducibility, but users must obtain the benchmark inputs independently and place local copies here before running the Italian benchmark workflows.

## Expected local filenames

Place the acquired files at these exact paths:

- `data/raw/georock-data.csv`
- `data/raw/Results_Caio.xlsx`

The preprocessing notebook `scripts/0_raw_data_preprocessing.ipynb` expects those filenames and writes the derived benchmark table to `data/processed/caio_italy_benchmark/full_italian_data.csv`.

## Source references

Petrelli, M., Bizzarri, R., Morgavi, D., Baldanza, A., and Perugini, D. (2017). Combining machine learning techniques, microanalyses and large geochemical datasets for tephrochronological studies in complex volcanic areas: New age constraints for the Pleistocene magmatism of central Italy. *Quaternary Geochronology*, 40, 33-44. ISSN 1871-1014. https://doi.org/10.1016/j.quageo.2016.12.003

Project materials associated with the Petrelli et al. (2017) study were referenced in the original project repository:

- `https://bitbucket.org/maurizio_petrelli/petrelli_et_al_2016_quaternary_geochronology/src/master/`

## How to use this directory

1. Retrieve the benchmark source files from the original study resources or other authorized upstream locations.
2. Save or rename your local copies as `georock-data.csv` and `Results_Caio.xlsx`.
3. Place them in `data/raw/`.
4. Run `scripts/0_raw_data_preprocessing.ipynb` to generate the local processed benchmark CSV needed by the Caio and sensitivity workflows.

## Citation note

If you use the preprocessing workflow or the benchmark materials reconstructed from these external sources, please cite both this DIHS manuscript and the original Petrelli et al. (2017) study.
