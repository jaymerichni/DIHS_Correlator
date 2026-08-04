from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from DIHS_Correlator.viz import hs_curves
from DIHS_Correlator.workflows import perturbative


def _base_plot_df() -> pd.DataFrame:
    rows = [
        # Neighbor A
        {
            "depth_level": 0,
            "neighbor_unit": "A",
            "harmonic_score": 0.8,
            "harmonic_score_std": 0.10,
            "model": "kmeans",
            "transform": "clr",
            "tag": "shallow_a",
        },
        {
            "depth_level": 1,
            "neighbor_unit": "A",
            "harmonic_score": 0.5,
            "harmonic_score_std": 0.05,
            "model": "kmeans",
            "transform": "clr",
            "tag": "mid_a",
        },
        {
            "depth_level": 2,
            "neighbor_unit": "A",
            "harmonic_score": 0.2,
            "harmonic_score_std": 0.02,
            "model": "kmeans",
            "transform": "clr",
            "tag": "deep_a",
        },
        # Neighbor B
        {
            "depth_level": 0,
            "neighbor_unit": "B",
            "harmonic_score": 0.7,
            "harmonic_score_std": 0.11,
            "model": "kmeans",
            "transform": "clr",
            "tag": "shallow_b",
        },
        {
            "depth_level": 1,
            "neighbor_unit": "B",
            "harmonic_score": 0.4,
            "harmonic_score_std": 0.07,
            "model": "kmeans",
            "transform": "clr",
            "tag": "mid_b",
        },
        {
            "depth_level": 2,
            "neighbor_unit": "B",
            "harmonic_score": 0.3,
            "harmonic_score_std": 0.04,
            "model": "kmeans",
            "transform": "clr",
            "tag": "deep_b",
        },
        # Unknown class rows that must be excluded from plotting
        {
            "depth_level": 0,
            "neighbor_unit": 0,
            "harmonic_score": 0.95,
            "harmonic_score_std": 0.01,
            "model": "kmeans",
            "transform": "clr",
            "tag": "unknown",
        },
        {
            "depth_level": 1,
            "neighbor_unit": 0,
            "harmonic_score": 0.80,
            "harmonic_score_std": 0.02,
            "model": "kmeans",
            "transform": "clr",
            "tag": "unknown",
        },
    ]
    return pd.DataFrame(rows)


def test_prepare_plot_df_force_root_one_false_preserves_values_and_depths() -> None:
    original = _base_plot_df()
    original_before = original.copy(deep=True)

    out = hs_curves._prepare_plot_df(
        df=original,
        value_col="harmonic_score",
        std_col="harmonic_score_std",
        force_root_one=False,
        max_depth=None,
        unknown_class=0,
    )

    expected = original[original["neighbor_unit"].astype(str) != "0"].copy()
    expected = expected.sort_values(["neighbor_unit", "depth_level"]).reset_index(drop=True)
    got = out.sort_values(["neighbor_unit", "depth_level"]).reset_index(drop=True)

    pdt.assert_frame_equal(got, expected)
    pdt.assert_frame_equal(original, original_before)


def test_prepare_plot_df_force_root_one_adds_root_and_preserves_first_split() -> None:
    out = hs_curves._prepare_plot_df(
        df=_base_plot_df(),
        value_col="harmonic_score",
        std_col="harmonic_score_std",
        force_root_one=True,
        max_depth=None,
        unknown_class=0,
    )

    assert not (out["neighbor_unit"].astype(str) == "0").any()

    a = out[out["neighbor_unit"].astype(str) == "A"].sort_values("depth_level")
    assert a["depth_level"].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(a["harmonic_score"].to_numpy(), [1.0, 0.8, 0.5, 0.2])
    np.testing.assert_allclose(a["harmonic_score_std"].to_numpy(), [0.0, 0.10, 0.05, 0.02])
    assert a.iloc[0]["tag"] == "shallow_a"


def test_prepare_plot_df_force_root_one_respects_max_depth_before_shift() -> None:
    out = hs_curves._prepare_plot_df(
        df=_base_plot_df(),
        value_col="harmonic_score",
        std_col="harmonic_score_std",
        force_root_one=True,
        max_depth=1,
        unknown_class=0,
    )

    a = out[out["neighbor_unit"].astype(str) == "A"].sort_values("depth_level")
    assert a["depth_level"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(a["harmonic_score"].to_numpy(), [1.0, 0.8, 0.5])

    b = out[out["neighbor_unit"].astype(str) == "B"].sort_values("depth_level")
    assert b["depth_level"].tolist() == [0, 1, 2]


def test_plot_hs_curves_uses_transformed_depth_bounds_with_root_display(monkeypatch) -> None:
    rows = []
    for depth in range(15):
        rows.append(
            {
                "depth_level": depth,
                "neighbor_unit": "A",
                "harmonic_score": max(0.0, 1.0 - (0.03 * depth)),
            }
        )
        rows.append(
            {
                "depth_level": depth,
                "neighbor_unit": 0,
                "harmonic_score": max(0.0, 1.0 - (0.02 * depth)),
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = hs_curves.plt.subplots(figsize=(4, 3))

    monkeypatch.setattr(hs_curves.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(hs_curves.plt, "show", lambda: None)
    monkeypatch.setattr(hs_curves.plt, "close", lambda *args, **kwargs: None)

    hs_curves.plot_hs_curves(
        df=df,
        value_col="harmonic_score",
        with_shade=False,
        max_neighbors=None,
        neighbor_order="alphabetical",
        max_depth=14,
        force_root_one=True,
        output_path=None,
        unknown_class=0,
    )

    xlim = ax.get_xlim()
    assert xlim[0] == 0.0
    assert xlim[1] == 15.0

    xticks = ax.get_xticks().tolist()
    assert 0.0 in xticks
    assert 15.0 in xticks

    xdata = ax.lines[0].get_xdata()
    assert int(max(xdata)) == 15


def test_perturbative_plot_uses_chosen_integration_depth(monkeypatch, tmp_path) -> None:
    captured: dict[str, int | None] = {}

    def _fake_run_single_model_workflow(**kwargs):
        rows = []
        for depth in range(4):
            rows.append(
                {
                    "unknown_class": kwargs["unknown_sample"],
                    "transform": kwargs["transform_type"],
                    "model": kwargs["model_type"],
                    "depth_level": depth,
                    "neighbor_unit": "A",
                    "harmonic_score": 0.8 - (0.05 * depth),
                }
            )
            rows.append(
                {
                    "unknown_class": kwargs["unknown_sample"],
                    "transform": kwargs["transform_type"],
                    "model": kwargs["model_type"],
                    "depth_level": depth,
                    "neighbor_unit": "B",
                    "harmonic_score": 0.6 - (0.04 * depth),
                }
            )
            rows.append(
                {
                    "unknown_class": kwargs["unknown_sample"],
                    "transform": kwargs["transform_type"],
                    "model": kwargs["model_type"],
                    "depth_level": depth,
                    "neighbor_unit": kwargs["unknown_sample"],
                    "harmonic_score": 0.3 - (0.01 * depth),
                }
            )

        hs_per_depth = pd.DataFrame(rows)
        dihs_total = pd.DataFrame(
            {
                "unknown_class": [kwargs["unknown_sample"], kwargs["unknown_sample"]],
                "neighbor_unit": ["A", kwargs["unknown_sample"]],
                "total_product": [0.8, 0.3],
            }
        )
        return {
            "hs_per_depth": hs_per_depth,
            "dihs_total": dihs_total,
            "pairwise_per_depth_matrices": None,
        }

    def _fake_plot_hs_curves(**kwargs):
        captured["max_depth"] = kwargs.get("max_depth")

    monkeypatch.setattr(
        perturbative, "run_single_model_workflow", _fake_run_single_model_workflow
    )
    monkeypatch.setattr(perturbative, "plot_hs_curves", _fake_plot_hs_curves)
    monkeypatch.setattr(perturbative, "_plot_top1_fraction", lambda *args, **kwargs: None)

    df = pd.DataFrame(
        {
            "x": [1.0, 1.1, 0.9],
            "lettercode": ["A", "B", "U"],
        }
    )

    result = perturbative.perturbative_simple_run_workflow(
        df=df,
        model_type="kmeans",
        transform_type="clr",
        unknown_sample="U",
        class_column="lettercode",
        random_state=123,
        n_iterations=1,
        major_cols=[],
        trace_cols=[],
        compute_pairwise=False,
        plot_everything=True,
        write_files=False,
        output_dir=str(tmp_path),
        max_depth=100,
        integration_depth=1,
        verbose=False,
    )

    assert captured["max_depth"] == 1
    assert result["dihs_integration_depth"] == 1
