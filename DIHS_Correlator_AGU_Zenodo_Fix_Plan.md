# DIHS_Correlator Reproducibility Branch — Fix Plan for AGU/Zenodo Release

## Purpose

Apply the changes below to the `reproducibility-branch` of:

`https://github.com/jaymerichni/DIHS_Correlator`

The goal is to prepare a technically coherent, reproducible, and appropriately licensed `v1.0.0` release for submission to AGU's *Geochemistry, Geophysics, Geosystems* and archival in Zenodo.

Keep the implementation simple. Do not introduce unnecessary infrastructure, containers, extensive CI, or a large test suite unless required to complete the tasks below.

---

# Priority 1 — Release blockers

## 1. Add the missing `scikit-bio` dependency

### Problem

`src/DIHS_Correlator/core/transforms.py` imports:

```python
from skbio.stats.composition import clr, ilr
```

However, `scikit-bio` is missing from both `pyproject.toml` and `environment.yml`.

This causes:

```text
ModuleNotFoundError: No module named 'skbio'
```

when importing the installed package.

### Required changes

Add `scikit-bio` to the base package dependencies in `pyproject.toml`.

Example:

```toml
dependencies = [
    "flask",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-bio",
    "scikit-learn",
]
```

Add the exact tested `scikit-bio` version to `environment.yml`.

### Acceptance criteria

From a clean environment:

```bash
python -m pip install -e .
python -c "import DIHS_Correlator; print(DIHS_Correlator.__version__)"
```

must run successfully without any manual dependency installation.

---

## 2. Prevent the preprocessing notebook from writing an index column

### Problem

The final save operation in:

```text
scripts/0_raw_data_preprocessing.ipynb
```

currently writes the pandas index to:

```text
data/processed/caio_italy_benchmark/full_italian_data.csv
```

This can create an unintended numeric column such as:

```text
Unnamed: 0
```

The analysis pipeline may then interpret this artificial index as a numerical feature and include it in clustering.

### Required changes

Save the CSV using:

```python
data.to_csv(
    "../data/processed/caio_italy_benchmark/full_italian_data.csv",
    index=False,
)
```

Add defensive removal of accidental index columns when loading the benchmark dataset in downstream workflows:

```python
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
```

Add validation at the end of preprocessing:

```python
assert len(data) == 3907
assert data["lettercode"].isna().sum() == 0
assert not any(str(c).startswith("Unnamed:") for c in data.columns)
```

Also validate that the expected columns are present.

The expected structure is:

- 17 geochemical variables.
- One class-label column: `lettercode`.
- 3,907 rows in total:
  - 3,890 candidate-source observations.
  - 17 Caio observations.

### Acceptance criteria

The generated CSV must:

- Contain no numeric index column.
- Have exactly 3,907 rows.
- Have the expected geochemical variables plus `lettercode`.
- Have no missing `lettercode` values.

---

## 3. Expand the Caio notebook to reproduce all manuscript analyses

### Problem

The current:

```text
scripts/2_caio_source_attribution.ipynb
```

appears to reproduce only the coupled major-plus-trace feature-space analysis.

The manuscript reports nine combinations:

1. Coupled major plus trace elements:
   - Agglomerative-Ward.
   - K-Means.
   - Gaussian Mixture Model.

2. Major elements only:
   - Agglomerative-Ward.
   - K-Means.
   - Gaussian Mixture Model.

3. Trace elements only:
   - Agglomerative-Ward.
   - K-Means.
   - Gaussian Mixture Model.

The manuscript also reports:

- 100 perturbative runs.
- ±2% perturbation for major elements.
- ±10% perturbation for trace elements.
- 100 pseudo-unknown iterations.
- Pseudo-unknown subsets of 17 observations.

### Required changes

Implement three clearly labelled feature-space configurations.

A simple structure is:

```python
feature_spaces = {
    "coupled": {
        "columns": major_cols + trace_cols,
        "transform_type": "clr",
        "major_cols": major_cols,
        "trace_cols": trace_cols,
    },
    "major_only": {
        "columns": major_cols,
        "transform_type": "clr",
        "major_cols": major_cols,
        "trace_cols": [],
    },
    "trace_only": {
        "columns": trace_cols,
        "transform_type": "scaled",
        "major_cols": [],
        "trace_cols": trace_cols,
    },
}
```

Use the exact transformation settings that were used to generate the manuscript results. If the current code or manuscript indicates a different trace-only transformation, preserve the manuscript-generating configuration.

Save outputs separately:

```text
results/2_caio_source_attribution/coupled/
results/2_caio_source_attribution/major_only/
results/2_caio_source_attribution/trace_only/
```

Each feature space must run all three clustering models.

### Acceptance criteria

The notebook or replacement script must generate all nine model-feature-space combinations and save the values required to reconstruct manuscript Table 1 and the Caio figures.

---

## 5. Correct the six-province sensitivity analysis

### Problem

The manuscript reports 172,800 sensitivity runs based on:

- Six most populated provinces.
- Twelve unknown-sample sizes.
- Four candidate-source dataset sizes.
- Three clustering models.
- 100 iterations.
- Positive and negative conditions.

This gives:

```text
6 × 12 × 4 × 3 × 100 × 2 = 172,800
```

The current:

```text
scripts/3_sensitivity_data_size.py
```

removes `Caio` and `IAVP`, but appears to retain nine candidate provinces. The pseudo-unknown routine then iterates over every eligible class, which would not match the 172,800-run design.

### Required changes

Explicitly define and retain the six manuscript provinces.

Based on the current dataset description, these appear to be:

```python
SENSITIVITY_CLASSES = [
    "AI",
    "PF",
    "EV",
    "RMP",
    "VV",
    "PI",
]
```

Verify these against the actual manuscript-generating analysis before committing.

Filter explicitly:

```python
work = work[work[CLASS_COLUMN].isin(SENSITIVITY_CLASSES)]
```

Add:

```python
assert work[CLASS_COLUMN].nunique() == 6
```

Calculate and print the expected number of runs before execution.

Verify the full six-province subset size reported in the manuscript, including the stated `N=3479`, against the reconstructed dataset.

### Acceptance criteria

The script must:

- Use exactly six source classes.
- Produce exactly 172,800 planned runs.
- Save enough metadata to verify the run design.
- Reproduce the manuscript's positive- and negative-condition definitions.

---

# Priority 2 — Data acquisition and provenance

## 6. Strengthen the external-data acquisition READMEs

### Context

The decision to exclude the previously published Italian and Caio input data from the repository is appropriate where redistribution rights are unclear.

The repository must nevertheless make reconstruction of the benchmark dataset precise and unambiguous.

```

Avoid directing users only to a repository homepage or mutable `master` branch.

### Expected reconstructed output

Document that preprocessing should generate:

```text
data/processed/caio_italy_benchmark/full_italian_data.csv
```

with:

- 3,907 rows.
- 17 geochemical variables.
- `lettercode`.
- No accidental index column.
- Expected counts by source class.

### Acceptance criteria

A reader who has not previously seen the data must be able to obtain the exact inputs and generate the expected processed dataset without guessing filenames, sheets, paths, or versions.

---

## 7. Clarify provenance of the preprocessing notebook

### Problem

The scripts documentation describes:

```text
0_raw_data_preprocessing.ipynb
```

as adapted from Petrelli et al. (2017).

Removing the input data does not by itself resolve the copyright or attribution status of adapted code.

### Required changes

State clearly whether the preprocessing notebook was:

- Independently reimplemented from the published method; or
- Copied or adapted from an earlier code repository.

If code was copied or adapted:

- Identify the exact upstream source.
- Preserve the upstream copyright notice.
- Preserve or comply with the upstream licence.
- Obtain permission if no reuse licence exists.

### Acceptance criteria

The repository must not imply that adapted third-party code is wholly original unless that is accurate.

---

# Priority 3 — Licensing, versioning, and citation metadata

## 8. Add a root software licence

### Problem

The repository currently lacks a root-level `LICENSE` file.

Without an explicit licence, default copyright applies and users do not have general permission to reuse, modify, or redistribute the software.

### Required decision

Choose one licence and apply it consistently.

### Option A — BSD 3-Clause

Use this if unrestricted academic and commercial adoption, including proprietary derivatives, is acceptable.

SPDX identifier:

```text
BSD-3-Clause
```

### Option B — GPL v3

Use this if commercial use is acceptable but distributed derivative software should remain open and provide source code.

SPDX identifier:

```text
GPL-3.0-only
```

### Required changes

Add:

```text
LICENSE
```

Update `pyproject.toml`:

```toml
license = {file = "LICENSE"}
```

Update `CITATION.cff`:

```yaml
license: BSD-3-Clause
```

or:

```yaml
license: GPL-3.0-only
```

Add a short licensing section to the root README.

Clearly state that external data are not redistributed and remain governed by their original terms.

### Acceptance criteria

The selected licence must be identical in:

- `LICENSE`.
- `pyproject.toml`.
- `CITATION.cff`.
- Root README.

---

## 9. Standardise the release version to `1.0.0`

### Problem

The repository currently uses `0.1.0` in multiple files, while the manuscript describes the archived release as Version 1.0.0.

### Required changes

Set the release version to:

```text
1.0.0
```

in:

- `pyproject.toml`.
- `CITATION.cff`.
- `src/DIHS_Correlator/__init__.py`.
- `CHANGELOG.md`.
- Any README examples.
- Flask interface version display, if applicable.

Do not insert a release date until the GitHub release is actually published.

Do not insert the Zenodo DOI until Zenodo has minted it.

### Acceptance criteria

A repository-wide search must show no stale `0.1.0` references intended to describe the archival release.

---

## 10. Review `CITATION.cff`

### Required changes

Ensure that:

- `version` is `1.0.0`.
- The software licence is specified.
- The release date is the actual GitHub release date.
- Software creators reflect actual software contributions rather than automatically copying the manuscript author list.
- ORCIDs and affiliations are added where known.
- No placeholder DOI remains.
- The unpublished article is not used as an incomplete preferred citation unless deliberately justified.
- The file validates successfully.

### Acceptance criteria

Validate `CITATION.cff` with a standard CFF validator before release.

---

# Priority 4 — Manuscript/repository consistency

## 11. Correct the Supporting Text S2/S3 references

### Problem

The repository READMEs currently reverse the Supporting Information references.

Correct mapping:

- **Text S2:** synthetic dataset generation.
- **Text S3:** centroid-distance and Mahalanobis-distance baseline definitions.

### Required changes

Correct this in:

- Root `README.md`.
- `scripts/README.md`.
- Notebook headings or comments.
- Any other repository documentation.

### Acceptance criteria

A repository-wide search must show the correct S2/S3 mapping everywhere.

---

# Priority 6 — Minimal testing and release verification

## 16. Add minimal automated tests

Do not build an extensive test suite. Add only a few high-value tests.

### Required tests

#### Import and version test

Confirm that:

```python
import DIHS_Correlator
```

works and that the package version equals the `pyproject.toml` release version.

#### Deterministic toy-analysis test

Use a very small synthetic dataset and fixed seed to confirm that:

- The workflow runs.
- DIHS values remain in `[0, 1]`.
- The expected leading class is returned.
- Repeated execution with fixed seeds is stable.

#### Optional web asset test

Confirm that the Flask templates and static assets are included in the built wheel.

### Acceptance criteria

The tests must run successfully in a clean environment.

---

## 17. Perform a complete clean-environment reproduction check

### Required process

Create a fresh environment using only the committed environment instructions.

Then:

1. Install the package.
2. Import the package.
3. Run the synthetic dataset generator.
4. Confirm the generated synthetic CSV agrees with the committed synthetic dataset, allowing only negligible floating-point serialization differences.
5. Acquire the external files using only the repository instructions.
6. Run raw-data preprocessing.
7. Validate the 3,907-row processed dataset.
8. Run all three Caio feature spaces and all three clustering models.
9. Run the initialisation-stability analysis.
10. Run the six-province sensitivity analysis.
11. Confirm the planned run count is exactly 172,800.
12. Generate compact manuscript summary CSVs.
13. Generate manuscript figures.
14. Compare Table 1 and figure data numerically with the submitted manuscript.

Record:

- Python version.
- Operating system.
- Key dependency versions.
- Approximate runtime.
- Approximate RAM requirements.
- Random seeds.
- Whether values should match exactly or within a stated tolerance.

### Acceptance criteria

No hidden local files, manual path edits, or undeclared dependencies may be required.

---

# Non-goals

The following are not required for this release unless they become necessary to fix a blocker:

- Docker.
- PyPI publication.
- Extensive continuous integration.
- Large unit-test coverage.
- Automatic cloud deployment.
- Workflow orchestration platforms.
- Archiving every intermediate clustering object.
- Reengineering the package architecture.

---

# Final completion checklist

## Data handling

- [ ] External input data remain excluded from the repository.
- [ ] Acquisition instructions identify exact files and immutable versions.
- [ ] Excel worksheet and expected columns documented.
- [ ] Processed dataset contains 3,907 rows.
- [ ] No accidental index column is written.
- [ ] Expected source-class counts documented.
- [ ] Preprocessing-code provenance clarified.

## Software

- [ ] `scikit-bio` declared.
- [ ] Clean package import succeeds.
- [ ] Wheel builds successfully.
- [ ] Flask assets are included.
- [ ] Minimal tests pass.
- [ ] Version is `1.0.0` everywhere.

## Documentation and licensing

- [ ] Root `LICENSE` added.
- [ ] Licence metadata consistent.
- [ ] S2/S3 references corrected.
- [ ] `CITATION.cff` validated.
- [ ] Open Research wording updated.
- [ ] Local Flask interface described accurately.
- [ ] No placeholder release date or DOI remains at publication.


## Release

- [ ] Full clean-environment run completed.
- [ ] Zenodo integration enabled before GitHub release publication.
- [ ] `v1.0.0` tag points to the intended final commit.
- [ ] Zenodo archive inspected.
- [ ] Version-specific DOI inserted into manuscript.
