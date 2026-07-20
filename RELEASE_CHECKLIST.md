# Release Checklist

Use this checklist before creating the Zenodo-linked archival release or submitting the repository with a manuscript.

## App And Packaging

- Run `python -m pip install -e .` from a clean environment.
- Start the local interface with `dihs-tephra-correlator` and confirm the home page loads.
- Confirm `python -m DIHS_Correlator.web` also starts the same local interface.
- Build a wheel with `python -m pip wheel . --no-deps --no-build-isolation -w dist_check`.
- Verify the wheel contains `DIHS_Correlator/web/templates/index.html`, `DIHS_Correlator/web/static/app.css`, and `DIHS_Correlator/web/static/app.js`.

## Documentation

- Update the package version in `pyproject.toml` and `src/DIHS_Correlator/__init__.py` together.
- Replace the DOI banner placeholder by setting `DIHS_CORRELATOR_SOFTWARE_DOI` after the Zenodo DOI is minted.
- Re-read the README after any UI or workflow changes so the app description stays synchronized.

## Publication Metadata

- Add a `LICENSE` file once you decide the release license.
- Add a `CITATION.cff` file with the final author list, title, version, release date, and DOI.
- Tag the archival release in Git before connecting it to Zenodo.

## Clean Repository

- Remove generated local outputs such as `Results*` folders before tagging a release.
- Remove temporary build products such as `build/`, `dist/`, and `*.egg-info/` before publishing the release snapshot.
- Run the smoke tests below from a clean checkout:

```bash
python -m unittest discover -s tests
```
