# DIHS_Correlator: Zenodo, Licensing, Reproducibility, and AGU Submission Guide

## Scope

This document summarizes the audit and recommendations for publishing the GitHub branch:

- Repository: `jaymerichni/DIHS_Correlator`
- Target branch: `reproducibility-branch`
- Intended archive: GitHub release linked to Zenodo
- Intended journal: AGU *Geochemistry, Geophysics, Geosystems* (G-Cubed)

The objective is to create a versioned, citable, reproducible software release that contains the software, analysis workflows, notebooks, and redistributable research data used in the manuscript.

---

# 1. Recommended publication model

Use a single GitHub release from `reproducibility-branch`, archived automatically by Zenodo.

The release should be treated primarily as a **software record**, with associated notebooks, scripts, and reproducibility data included in the same archive.

Recommended sequence:

1. Correct the repository issues described below.
2. Confirm the licensing and redistribution status of every included data file.
3. Connect the GitHub repository to Zenodo.
4. Create a GitHub release from the exact final commit on `reproducibility-branch`.
5. Allow Zenodo to archive that release and mint a DOI.
6. Cite the version-specific Zenodo DOI in the manuscript.
7. Link the final paper DOI back to the Zenodo record after publication.

Do not cite only the live GitHub repository. GitHub is the active-development platform; Zenodo is the permanent, versioned archive.

---

# 2. Current repository assessment

The branch already contains substantial reproducibility material, including:

- The installable `DIHS_Correlator` Python package.
- A local Flask-based browser interface.
- Synthetic datasets.
- The Italian/Caio benchmark data.
- Scripts and notebooks for:
  - raw-data preprocessing;
  - synthetic scenario generation;
  - synthetic benchmark comparison;
  - Caio source attribution;
  - sample-size sensitivity analyses.
- `pyproject.toml`.
- `CITATION.cff`.
- Repository-level documentation.

The scientific organization is generally good, but several issues should be resolved before creating the archival release.

---

# 3. Release blockers

## 3.1 Inconsistent capitalization of data paths

The repository contains:

```text
data/Processed/
```

but some notebooks and scripts refer to:

```text
data/processed/
```

This may work on case-insensitive systems such as Windows but fail on Linux and other case-sensitive systems.

### Recommended correction

Use lowercase consistently:

```text
data/raw/
data/processed/
```

On Windows, force Git to detect the case-only rename:

```bash
git mv data/Processed data/processed_tmp
git mv data/processed_tmp data/processed
```

Then search the entire repository for all variants:

```text
data/Processed
data/processed
Data/Raw
data/raw
```

Update the paths consistently in:

- `README.md`;
- scripts documentation;
- notebooks;
- analysis scripts;
- Flask defaults;
- expected-output instructions.

---

## 3.2 No explicit software licence

The repository should contain a root-level licence file.

Recommended software licence:

```text
BSD-3-Clause
```

Add:

```text
LICENSE
```

The copyright holder must be correct. Confirm whether ownership belongs to:

- the individual software creator;
- all software contributors;
- an employer;
- a university or research institution.

Example copyright line:

```text
Copyright (c) 2026 Jan Aymerich and DIHS_Correlator contributors
```

Do not use this wording until ownership has been confirmed.

Add the licence to `pyproject.toml`:

```toml
[project]
license = "BSD-3-Clause"
license-files = ["LICENSE"]
```

Add it to `CITATION.cff`:

```yaml
license: BSD-3-Clause
```

The BSD 3-Clause licence should apply to the software code only, not automatically to third-party datasets.

---

## 3.3 Data licensing and redistribution rights

The repository contains data with different origins and potentially different legal conditions. These files should not all be placed under the software licence.

Recommended high-level allocation:

| Component | Recommended treatment |
|---|---|
| DIHS_Correlator source code | BSD 3-Clause |
| Original documentation | BSD 3-Clause or CC BY 4.0 |
| Original synthetic datasets | CC BY 4.0 or CC0 |
| GEOROC-derived data | Upstream GEOROC terms, likely CC BY-SA 4.0, subject to confirming applicability to the historical extract |
| Original Caio analytical data | Existing upstream licence, or written permission from the rightsholder |
| Derived datasets | A licence compatible with all upstream components |

Detailed data recommendations are provided in Section 8.

---

## 3.4 Clean reproducibility run not yet demonstrated

The complete workflow should be tested in a new environment before release.

Issues identified include:

- `openpyxl` is needed to read `Results_Caio.xlsx`, but it is not currently declared as a dependency.
- Notebook dependencies such as Jupyter are not declared.
- At least one notebook contains a saved error from an earlier execution.
- Runtime dependencies are unbounded.
- The analysis environment is not locked.
- Some notebook paths may depend on the current working directory.

Before release:

1. Create a new virtual environment.
2. Install the package using only the documented instructions.
3. Execute every notebook from the first cell to the last.
4. Run every analysis script.
5. Confirm all expected outputs.
6. Confirm that the workflow also runs on Linux.
7. Remove stale notebook errors.
8. Either save successful outputs or strip outputs consistently.
9. Record the exact environment used for the manuscript.

---

# 4. Python packaging recommendations

## 4.1 Improve `pyproject.toml`

The file should contain richer package metadata.

Suggested structure:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "DIHS_Correlator"
version = "0.1.0"
description = "DIHS-based software for interpretable tephra-source correlation."
readme = "README.md"
requires-python = ">=3.10"
license = "BSD-3-Clause"
license-files = ["LICENSE"]

authors = [
    {name = "Jan Aymerich"}
]

keywords = [
    "tephrochronology",
    "geochemistry",
    "machine learning",
    "DIHS"
]

dependencies = [
    "flask",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn"
]

[project.optional-dependencies]
reproducibility = [
    "jupyterlab",
    "ipykernel",
    "nbconvert",
    "openpyxl"
]

test = [
    "build",
    "pytest",
    "twine"
]

[project.urls]
Homepage = "https://github.com/jaymerichni/DIHS_Correlator"
Repository = "https://github.com/jaymerichni/DIHS_Correlator"
Issues = "https://github.com/jaymerichni/DIHS_Correlator/issues"
```

Dependency versions may remain reasonably broad for the library itself, but the exact manuscript environment should also be recorded separately.

---

## 4.2 Add a reproducible environment

Create one of the following:

```text
environment.yml
```

or:

```text
pylock.toml
```

or another lock file generated by the chosen environment manager.

The environment record should preserve:

- Python version;
- NumPy version;
- pandas version;
- Matplotlib version;
- scikit-learn version;
- Flask version;
- OpenPyXL version;
- Jupyter and notebook execution versions;
- any other runtime dependencies.

Also record:

- operating system;
- approximate RAM requirements;
- expected runtime for each workflow;
- random seeds;
- whether numerical results should match exactly or within tolerance.

---

## 4.3 Decide what belongs in the package distribution

The repository contains two related objects:

1. The installable Python package.
2. The complete manuscript reproducibility archive.

Zenodo will archive the entire GitHub release snapshot. Therefore, the Python wheel and source distribution do not necessarily need to include all notebooks and datasets.

A lean `MANIFEST.in` could contain:

```text
include README.md
include LICENSE
include CITATION.cff
recursive-include src/DIHS_Correlator/web/templates *.html
recursive-include src/DIHS_Correlator/web/static *.css *.js
```

The full GitHub/Zenodo archive can still contain:

- `data/`;
- `scripts/`;
- notebooks;
- environment files;
- reproducibility documentation;
- expected outputs.

When testing the release, inspect both:

```bash
python -m build
twine check dist/*
```

Also install the built wheel into a clean environment and confirm that the Flask interface and package entry points work.

---

# 5. Recommended repository structure

A clear structure would be:

```text
DIHS_Correlator/
├── README.md
├── REPRODUCING.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── environment.yml
├── MANIFEST.in
├── src/
│   └── DIHS_Correlator/
├── tests/
├── scripts/
│   ├── README.md
│   ├── 0_raw_data_preprocessing.ipynb
│   ├── 1a_synthetic_scenario_gen.ipynb
│   ├── 1b_synthetic_scenario_comparison.py
│   ├── 2_caio_source_attribution.ipynb
│   └── 3_sensitivity_data_size.py
├── data/
│   ├── README.md
│   ├── LICENSES.md
│   ├── raw/
│   ├── processed/
│   └── synthetic/
└── .github/
    └── workflows/
        └── tests.yml
```

---

# 6. Reproducibility documentation

Create a dedicated:

```text
REPRODUCING.md
```

It should provide exact commands rather than only descriptive instructions.

Example:

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

Then document the analysis order:

```bash
jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace \
  scripts/0_raw_data_preprocessing.ipynb

jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace \
  scripts/1a_synthetic_scenario_gen.ipynb

python scripts/1b_synthetic_scenario_comparison.py

jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace \
  scripts/2_caio_source_attribution.ipynb

python scripts/3_sensitivity_data_size.py
```

The commands above must be adjusted to the final working-directory logic and tested exactly as documented.

For every workflow, report:

- purpose;
- command;
- input files;
- output files;
- expected output directory;
- random seed;
- expected runtime;
- approximate memory use;
- main expected numerical result;
- corresponding manuscript figure, table, or section.

---

# 7. Automated testing

Recommended minimum tests:

```text
tests/test_import.py
tests/test_minimal_run.py
tests/test_version.py
tests/test_web_assets.py
```

The automated checks should confirm that:

1. The package imports successfully.
2. The package version matches `pyproject.toml`.
3. A minimal deterministic analysis completes.
4. Fixed-seed results have the expected shape and classes.
5. Flask templates and static assets are included.
6. The built wheel installs successfully.
7. Documented paths work on Linux.
8. The command-line interface, if present, starts correctly.

A GitHub Actions workflow should test at least:

- Linux;
- the minimum supported Python version;
- one current Python version.

---

# 8. Data provenance and licensing

## 8.1 Upstream Petrelli repository

The data derive from the repository associated with:

> Petrelli, M., Bizzarri, R., Morgavi, D., Baldanza, A., and Perugini, D. (2017). Combining machine learning techniques, microanalyses and large geochemical datasets for tephrochronological studies in complex volcanic areas. *Quaternary Geochronology*, 40, 33–44.

Original project repository:

```text
https://bitbucket.org/maurizio_petrelli/petrelli_et_al_2016_quaternary_geochronology/src/master/
```

Final publication:

```text
https://doi.org/10.1016/j.quageo.2016.12.003
```

The fact that files are publicly downloadable does not itself establish an open licence or permission to redistribute them in another archive.

---

## 8.2 `georock-data.csv`

### Origin

`georock-data.csv` appears to be a filtered extract derived from GEOROC data and prepared for the Petrelli et al. analysis.

It is therefore third-party or upstream-derived data, not software created within DIHS_Correlator.

### Recommended treatment

Treat it as GEOROC-derived data.

Current GEOROC compilations are distributed under:

```text
CC BY-SA 4.0
```

However, the file appears to originate from a historical extract. The present GEOROC terms do not automatically prove that the same licence applied to that exact 2016/2017 export.

Therefore, the safest wording is:

> GEOROC-derived data. Current GEOROC compilations are distributed under CC BY-SA 4.0; confirmation should be obtained that these terms also apply to the historical extract included here.

### Best options

In order of legal clarity:

1. Obtain confirmation from GEOROC that the historical extract may be redistributed under CC BY-SA 4.0.
2. Obtain confirmation from Maurizio Petrelli regarding provenance and redistribution terms.
3. Recreate the filtered dataset from a current, versioned GEOROC compilation with an explicit licence.
4. If the exact historical extract cannot be licensed, exclude it and provide acquisition/reconstruction instructions.

### Attribution

Documentation should credit:

- GEOROC;
- Petrelli et al. (2017);
- the original analytical publications represented in the GEOROC dataset, where identifiable.

### Filename

The current filename is:

```text
georock-data.csv
```

The recognized database name is GEOROC. Consider renaming it:

```text
georoc-data.csv
```

Only rename it if doing so does not break scripts or reproducibility. Otherwise retain the original filename and explain it.

---

## 8.3 `Results_Caio.xlsx`

### Origin

`Results_Caio.xlsx` appears to contain original analytical measurements produced for the Petrelli et al. study.

### Current licensing position

No explicit open-data licence has been verified for this file.

The journal article being copyrighted by Elsevier does not necessarily mean that Elsevier owns the raw dataset. However, neither the article nor the public Bitbucket availability provides a sufficiently clear licence for redistribution.

### Recommended treatment

Do not assign a licence to the file yourself.

Preferred solution:

- Obtain written permission from the relevant rightsholder.
- Ask for authorization to redistribute the unchanged file in the Zenodo archive.
- Ideally request permission under:

```text
CC BY 4.0
```

Suggested wording for the permission request:

> We are preparing a reproducibility archive in Zenodo for a study that reuses the Caio analytical data made publicly available with Petrelli et al. (2017). We would like to include the original `Results_Caio.xlsx` file, unchanged, in the archived release. Could you please confirm that you and, where applicable, the other rightsholders authorize us to redistribute this file in the Zenodo archive under the Creative Commons Attribution 4.0 International licence, with full attribution to Petrelli et al. (2017)?

Keep the written response as documentation.

### If permission is not obtained

Remove the file from the Zenodo release and replace it with a metadata file, for example:

```text
data/external/Results_Caio_README.md
```

That file should provide:

- original repository URL;
- publication DOI;
- original filename;
- access date;
- SHA-256 checksum;
- expected local path;
- instructions for downloading and placing the file;
- script or notebook that processes it;
- a statement explaining why it is not redistributed.

A stronger long-term solution would be for the original authors to deposit the dataset in Zenodo or another repository under an explicit open-data licence.

---

## 8.4 `full_italian_data.csv`

This processed file is derived from upstream data.

Its licence must be compatible with all upstream components.

If it includes GEOROC-derived material governed by CC BY-SA 4.0, the derived dataset will generally also need to be distributed under:

```text
CC BY-SA 4.0
```

If `Results_Caio.xlsx` is also incorporated, the permission for the Caio data must permit redistribution within a CC BY-SA derivative.

This point should be included explicitly when requesting permission. Merely obtaining permission to include the original Excel file under CC BY 4.0 may not by itself resolve the licence of a combined derivative if share-alike conditions apply.

A useful permission request could therefore ask for both:

1. redistribution of the unchanged `Results_Caio.xlsx`; and
2. incorporation of those data into a CC BY-SA 4.0 processed dataset.

---

## 8.5 Synthetic datasets

Datasets generated entirely by the DIHS_Correlator authors can be licensed independently.

Recommended options:

```text
CC BY 4.0
```

or:

```text
CC0 1.0
```

Use CC BY 4.0 when attribution is desired.

Use CC0 when maximum reuse with no attribution requirement is preferred.

For an academic reproducibility archive, CC BY 4.0 is a reasonable default.

---

# 9. Mixed licences in one Zenodo record

A single Zenodo record may contain elements governed by different licences.

For this project, the final allocation could be:

| Component | Licence |
|---|---|
| DIHS_Correlator code | BSD 3-Clause |
| Original documentation | BSD 3-Clause or CC BY 4.0 |
| Original synthetic data | CC BY 4.0 |
| GEOROC-derived data | CC BY-SA 4.0, once applicability is confirmed |
| `Results_Caio.xlsx` | CC BY 4.0 only with explicit permission |
| Derived GEOROC dataset | CC BY-SA 4.0 |

Zenodo may allow multiple licences to be declared at record level, but record-level metadata alone is not sufficiently precise. The repository must state which licence applies to which files.

Add:

```text
data/LICENSES.md
```

The root `README.md` should state:

> Unless otherwise indicated, the source code in this repository is licensed under the BSD 3-Clause licence. Research data and third-party materials are excluded from that licence and are governed by the terms described in `data/LICENSES.md`.

The root `LICENSE` should apply only to the software code.

---

# 10. Suggested `data/LICENSES.md`

```markdown
# Data licences and provenance

The BSD 3-Clause licence in the repository root applies to the
DIHS_Correlator software code. It does not automatically apply to
research data or third-party materials.

## georock-data.csv

**Description:** Filtered compilation of Italian volcanic geochemical
analyses derived from GEOROC and used in Petrelli et al. (2017).

**Upstream source:** GEOROC — Geochemistry of Rocks of the Oceans and
Continents.

**Intermediate source:** Petrelli et al. (2017).

**Licence:** Current GEOROC compilations are distributed under
CC BY-SA 4.0. Applicability to this historical extract should be
confirmed before redistribution.

**Modifications:** Filtered and classified for the Petrelli et al.
analysis.

**Attribution:** Cite GEOROC, Petrelli et al. (2017), and the original
analytical data sources where available.

## Results_Caio.xlsx

**Description:** Major- and trace-element measurements obtained from
the Caio samples and used in Petrelli et al. (2017).

**Source:** Petrelli et al. (2017).

**Licence:** To be confirmed.

**Redistribution:** This file may be included in the Zenodo archive
only if an existing open licence is identified or written permission
is obtained from the relevant rightsholder.

## full_italian_data.csv

**Description:** Processed dataset generated from GEOROC-derived
reference data and Caio analytical data.

**Licence:** To be finalized after the licences and permissions for all
upstream components have been confirmed. If the GEOROC share-alike terms
apply, this derived dataset should be distributed under CC BY-SA 4.0.

## Synthetic datasets

**Description:** Synthetic datasets generated by the DIHS_Correlator
authors.

**Licence:** CC BY 4.0.
```

Update this document once the upstream permissions are resolved.

---

# 11. `CITATION.cff`

The citation file should describe the software release itself.

Recommended first-release structure:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite the archived version below."
type: software
title: "DIHS Tephra Correlator"
version: "0.1.0"
date-released: 2026-XX-XX
license: BSD-3-Clause
abstract: >
  DIHS Tephra Correlator is a Python package for interpretable
  tephra-source correlation using the Depth-Integrated Harmonic
  Score framework.
repository-code: "https://github.com/jaymerichni/DIHS_Correlator"
url: "https://github.com/jaymerichni/DIHS_Correlator"
authors:
  - family-names: "Aymerich"
    given-names: "Jan"
    orcid: "https://orcid.org/0000-0000-0000-0000"
    affiliation: "Institution"
keywords:
  - tephra correlation
  - tephrochronology
  - machine learning
  - geochemistry
  - DIHS
```

Replace placeholders with accurate information.

## Software authorship

Do not automatically copy the full manuscript author list.

Software authors should reflect substantial contributions to:

- software design;
- implementation;
- testing;
- documentation;
- maintenance;
- reproducibility workflows.

Other paper contributors can remain manuscript authors without necessarily being software authors.

## `preferred-citation`

For the first release, it is preferable to cite the software record itself.

Avoid making an unpublished manuscript the sole preferred citation.

After the article is published:

- add the article DOI as a related identifier in Zenodo;
- optionally add it to `CITATION.cff`;
- continue to cite the software DOI separately in the paper and in future reuse.

## Release date

Set:

```yaml
date-released:
```

to the actual GitHub release date, not the date on which the citation file was drafted.

---

# 12. Version number

Two reasonable options are:

```text
v0.1.0
```

or:

```text
v1.0.0
```

Use `v0.1.0` if:

- this is the first public research release;
- the interface may still change;
- the API is not yet considered stable.

Use `v1.0.0` if:

- the public interface is considered stable;
- the package is ready for general use;
- breaking changes will be managed under semantic versioning.

AGU and Zenodo do not require the first archived version to be `1.0.0`.

The following must agree:

- Git tag;
- GitHub release title;
- `pyproject.toml`;
- `CITATION.cff`;
- package `__version__`, if present;
- Zenodo metadata;
- manuscript availability statement.

---

# 13. Zenodo and GitHub workflow

## 13.1 Before the release

Confirm that:

- the repository is public;
- the exact release commit is final;
- all licences are present;
- data permissions are resolved;
- the package builds;
- all workflows run;
- notebooks are clean;
- metadata are complete;
- version numbers agree;
- no secrets or private data are present.

## 13.2 Connect GitHub to Zenodo

Do this before publishing the GitHub release:

1. Sign in to Zenodo.
2. Link the GitHub account.
3. Link ORCID, if available.
4. Open the GitHub integration page.
5. Synchronize repositories.
6. Enable `DIHS_Correlator`.

## 13.3 Create the release from the specific branch

A GitHub release can target a tag created from `reproducibility-branch`.

Recommended tag:

```text
v0.1.0
```

or:

```text
v1.0.0
```

Recommended release title:

```text
DIHS_Correlator v0.1.0 — reproducibility release for Aymerich et al.
```

Before publishing, verify that the tag points to the intended final commit.

Recommended release notes should summarize:

- principal software functionality;
- manuscript analyses included;
- installation instructions;
- reproducibility materials;
- data licensing;
- known limitations;
- exact manuscript version associated with the release.

After publishing the GitHub release, confirm that Zenodo has:

- ingested the release;
- generated the archive;
- assigned the version-specific DOI;
- imported the correct metadata;
- recognized the intended licences;
- listed the correct creators;
- preserved the expected files.

---

# 14. Which DOI to cite

Zenodo usually provides:

1. A version-specific DOI for the individual release.
2. A concept DOI representing all versions of the software.

For the paper, cite the:

```text
version-specific DOI
```

This identifies the exact software used for the reported analysis.

Use the concept DOI for:

- a general project badge;
- an overview page;
- a link intended to resolve to the software project across versions.

---

# 15. Citation in the AGU manuscript

The software should appear in three places:

1. In the Methods section.
2. In the Open Research or Data and Software Availability statement.
3. In the reference list.

## 15.1 Methods citation

Example:

> Data correlation, benchmark analyses, sensitivity analyses, and result processing were performed using DIHS_Correlator version 0.1.0 (Aymerich et al., 2026).

The software should be cited where its analytical role is described, not only in the availability statement.

## 15.2 Open Research statement

Example:

> Version 0.1.0 of DIHS_Correlator, the software used for the analyses reported in this study, is preserved in Zenodo (Aymerich et al., 2026) at https://doi.org/10.5281/zenodo.XXXXXXX. The source code is licensed under the BSD 3-Clause licence. Research data and third-party materials are governed by the file-specific terms described in `data/LICENSES.md`. Active development is hosted on GitHub at https://github.com/jaymerichni/DIHS_Correlator. The archived release contains the DIHS implementation and documentation; the notebooks and scripts used to generate the synthetic datasets, perform the benchmark and sensitivity analyses, process the results, and produce the reported outputs; and the analytical input tables that may legally be redistributed.

Adjust the wording to the final file contents and licences.

## 15.3 Reference-list entry

Example:

> Aymerich, J., Author, A. A., and Author, B. B. (2026). *DIHS_Correlator* (Version 0.1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Use the exact creator names and citation exported from the final Zenodo record.

---

# 16. README additions

Add a licensing section:

```markdown
## Licensing

Unless otherwise indicated, the DIHS_Correlator source code is
licensed under the BSD 3-Clause licence.

Research data and third-party materials are excluded from the software
licence. Their provenance, redistribution conditions, and file-specific
licences are documented in `data/LICENSES.md`.
```

Add a citation section:

```markdown
## Citation

If you use DIHS_Correlator, please cite the archived software release:

Aymerich, J., et al. (2026). DIHS_Correlator (Version 0.1.0)
[Software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

Add a reproducibility section linking to:

```text
REPRODUCING.md
```

---

# 17. Additional recommended files

## `CHANGELOG.md`

Document:

- release version;
- release date;
- major functions;
- manuscript analyses;
- breaking changes;
- known limitations.

## `CONTRIBUTING.md`

Explain:

- how to report issues;
- how to propose changes;
- testing expectations;
- style conventions;
- how contributors are credited.

## `.gitignore`

Consider including:

```text
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.venv/
venv/
.ipynb_checkpoints/
.pytest_cache/
.coverage
htmlcov/
.DS_Store
Thumbs.db
Results*/
dist_check/
```

Do not ignore outputs that are intentionally part of the archived reproducibility record.

---

# 18. Checksums and provenance

For external or third-party source files, record a SHA-256 checksum.

Example:

```bash
sha256sum data/raw/Results_Caio.xlsx
sha256sum data/raw/georock-data.csv
```

On Windows PowerShell:

```powershell
Get-FileHash data\raw\Results_Caio.xlsx -Algorithm SHA256
Get-FileHash data\raw\georock-data.csv -Algorithm SHA256
```

Record:

- filename;
- checksum;
- source URL;
- access date;
- upstream version;
- original creator;
- licence;
- modifications performed.

This is especially important if a file cannot be redistributed and users must download it separately.

---

# 19. Recommended release checklist

## Repository and paths

- [ ] Rename `data/Processed` to `data/processed`.
- [ ] Correct every path reference.
- [ ] Test paths on Linux.
- [ ] Remove temporary and generated files that are not part of the archive.
- [ ] Confirm that no secrets, credentials, or private information are present.

## Software licence

- [ ] Add `LICENSE`.
- [ ] Confirm the legal copyright holder.
- [ ] Apply BSD 3-Clause to source code.
- [ ] Add the licence to `pyproject.toml`.
- [ ] Add the licence to `CITATION.cff`.
- [ ] Exclude data and third-party materials from the root software licence.

## Data rights

- [ ] Add `data/LICENSES.md`.
- [ ] Confirm the licence applicable to the historical GEOROC extract.
- [ ] Obtain written permission for `Results_Caio.xlsx`, or exclude it.
- [ ] Confirm permission to incorporate Caio data into a CC BY-SA derivative.
- [ ] Assign a licence to synthetic datasets.
- [ ] Document the licence of `full_italian_data.csv`.
- [ ] Add provenance and checksums.
- [ ] Cite GEOROC, Petrelli et al., and original analytical sources.

## Reproducibility

- [ ] Add `openpyxl`.
- [ ] Add notebook dependencies.
- [ ] Add a reproducibility optional dependency group.
- [ ] Add an environment or lock file.
- [ ] Execute every notebook from top to bottom.
- [ ] Run all Python scripts.
- [ ] Remove stale notebook errors.
- [ ] Record random seeds.
- [ ] Record expected outputs.
- [ ] Record runtimes and hardware requirements.
- [ ] Add `REPRODUCING.md`.

## Packaging

- [ ] Improve `pyproject.toml` metadata.
- [ ] Add authors.
- [ ] Add project URLs.
- [ ] Add keywords.
- [ ] Decide whether the source distribution is lean or contains the full archive.
- [ ] Build the package.
- [ ] Run `twine check`.
- [ ] Install the wheel in a clean environment.
- [ ] Verify that web templates and static assets are included.

## Citation metadata

- [ ] Confirm the software author list.
- [ ] Add ORCIDs.
- [ ] Add affiliations.
- [ ] Correct the release date.
- [ ] Ensure all version numbers match.
- [ ] Remove or revise the unpublished `preferred-citation`.
- [ ] Validate `CITATION.cff`.

## Testing

- [ ] Add package import test.
- [ ] Add version consistency test.
- [ ] Add minimal deterministic analysis test.
- [ ] Add Flask asset test.
- [ ] Add Linux continuous integration.
- [ ] Test at least the minimum supported Python version.

## Zenodo release

- [ ] Connect GitHub to Zenodo before publishing the release.
- [ ] Enable the repository in Zenodo.
- [ ] Create the release tag from `reproducibility-branch`.
- [ ] Verify the tag commit.
- [ ] Publish the GitHub release.
- [ ] Confirm successful Zenodo ingestion.
- [ ] Verify Zenodo creators, title, version, licences, and files.
- [ ] Record the version-specific DOI.
- [ ] Add the DOI to the active README and citation metadata.
- [ ] Cite the version DOI in the manuscript.
- [ ] Link the paper DOI to Zenodo after publication.

---

# 20. Recommended final licensing model

Subject to confirmation of upstream rights:

| Repository component | Licence |
|---|---|
| DIHS_Correlator code | BSD 3-Clause |
| Original documentation | BSD 3-Clause or CC BY 4.0 |
| Synthetic datasets | CC BY 4.0 |
| `georock-data.csv` | CC BY-SA 4.0, after confirming applicability to the historical extract |
| `Results_Caio.xlsx` | CC BY 4.0 only with explicit rightsholder permission |
| `full_italian_data.csv` | CC BY-SA 4.0, provided all upstream permissions are compatible |
| Other third-party materials | Original upstream terms |

Do not place the entire archive under BSD 3-Clause.

Do not place `Results_Caio.xlsx` under CC BY 4.0 without authorization.

Do not assume that public availability is equivalent to an open licence.

---

# 21. Final recommendation

The branch is a strong basis for an AGU reproducibility release, but it should not yet be tagged for Zenodo.

The principal unresolved items are:

1. path capitalization;
2. absence of a software licence;
3. licensing of the historical GEOROC extract;
4. redistribution permission for `Results_Caio.xlsx`;
5. licensing of the combined processed dataset;
6. missing reproducibility dependencies;
7. absence of a verified clean execution;
8. incomplete package and citation metadata.

Once these points have been resolved, create a versioned GitHub release from the final commit of `reproducibility-branch`, allow Zenodo to archive it, and cite the resulting version-specific DOI in the AGU manuscript.
