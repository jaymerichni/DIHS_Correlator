# Raw Data Provenance

This folder contains the raw source files used to assemble the Italian benchmark dataset for the manuscript **"A New Machine Learning Approach for Interpretable Tephra-Source Correlation: Introducing the Depth-Integrated Harmonic Score (DIHS)"** by Aymerich et al. (2026).

Both files in this directory originate from the same source study, cited here as Petrelli et al. (2017):

- `georock-data.csv`
- `Results_Caio.xlsx`

These files are retained in their raw-input role for reproducibility. They are used by `scripts/0_raw_data_preprocessing.ipynb`, which adapts a preprocessing workflow from Petrelli et al., 2017 to reconstruct the processed benchmark dataset used in the repository's Italian case-study analyses.

## Source reference

Petrelli, M., Bizzarri, R., Morgavi, D., Baldanza, A., and Perugini, D. (2017). Combining machine learning techniques, microanalyses and large geochemical datasets for tephrochronological studies in complex volcanic areas: New age constraints for the Pleistocene magmatism of central Italy. *Quaternary Geochronology*, 40, 33-44. ISSN 1871-1014. https://doi.org/10.1016/j.quageo.2016.12.003

## Citation note

If these raw files or the preprocessing workflow are reused, the original source study Petrelli et al. (2017) should be cited alongside the DIHS manuscript.
