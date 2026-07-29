from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import math
import re

import pandas as pd

import DIHS_Correlator as dc


def test_import_and_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None
    expected = match.group(1)
    assert dc.__version__ == expected


def _toy_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.00, 1.10, 0.90, 1.20, 0.80, 8.80, 9.10, 9.30, 8.70, 9.20, 1.05],
            "y": [1.00, 0.90, 1.10, 1.20, 0.80, 8.90, 9.20, 8.80, 9.30, 9.00, 1.10],
            "lettercode": [
                "AI",
                "AI",
                "AI",
                "AI",
                "AI",
                "PF",
                "PF",
                "PF",
                "PF",
                "PF",
                "Caio",
            ],
        }
    )


def _run_toy_analysis() -> pd.DataFrame:
    result = dc.simple_run(
        df=_toy_dataset(),
        model_type="kmeans",
        transform_type="scaled",
        class_column="lettercode",
        unknown_sample="Caio",
        random_state=123,
        compute_pairwise=True,
        plot_everything=False,
        write_files=False,
        max_depth=10,
        verbose=False,
        return_details=True,
    )
    unknown_class = result["unknown_class"]
    ranked = result["dihs_total"].copy()
    ranked = ranked[ranked["neighbor_unit"].astype(str) != str(unknown_class)]
    return ranked.sort_values("total_product", ascending=False).reset_index(drop=True)


def test_toy_analysis_is_deterministic_and_bounded() -> None:
    first = _run_toy_analysis()
    second = _run_toy_analysis()

    assert not first.empty
    assert not second.empty

    best_1 = first.iloc[0]
    best_2 = second.iloc[0]

    assert str(best_1["neighbor_unit"]) == "AI"
    assert str(best_2["neighbor_unit"]) == "AI"

    assert first["total_product"].between(0.0, 1.0).all()
    assert second["total_product"].between(0.0, 1.0).all()

    assert math.isclose(
        float(best_1["total_product"]),
        float(best_2["total_product"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_packaged_web_assets_present() -> None:
    web_root = files("DIHS_Correlator.web")

    assert (web_root / "templates" / "index.html").is_file()
    assert (web_root / "static" / "app.css").is_file()
    assert (web_root / "static" / "app.js").is_file()
    assert (web_root / "static" / "branding" / "logo_web.png").is_file()
