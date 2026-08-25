from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

import DIHS_Correlator as dc


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


def test_uncertainty_summary_reports_precedence_layers() -> None:
    df = _toy_dataset()
    uncertainty_config = {
        "global": {"value": 0.03},
        "features": {"x": {"value": 0.01}},
        "cell_table": pd.DataFrame(
            [{"index": 0, "feature": "x", "value": 0.005}]
        ),
    }

    result = dc.perturbative_simple_run(
        df=df,
        model_type="kmeans",
        transform_type="scaled",
        class_column="lettercode",
        unknown_sample="Caio",
        random_state=123,
        n_iterations=2,
        major_cols=["x"],
        trace_cols=["y"],
        perturbation_seed=99,
        compute_pairwise=False,
        plot_everything=False,
        write_files=False,
        max_depth=10,
        verbose=False,
        return_details=True,
        uncertainty_config=uncertainty_config,
    )

    assert result["uncertainty_summary"]["resolved_cells"] == 22
    assert result["uncertainty_summary"]["missing_cells"] == 0

    observed = {
        (row["source_level"], row["source_name"]): int(row["cell_count"])
        for row in result["uncertainty_sources"].to_dict("records")
    }
    assert observed == {
        ("cell", "x"): 1,
        ("feature", "x"): 10,
        ("group", "trace"): 11,
    }


def test_perturbative_parallel_matches_serial() -> None:
    df = _toy_dataset()
    common = dict(
        df=df,
        model_type="kmeans",
        transform_type="scaled",
        class_column="lettercode",
        unknown_sample="Caio",
        random_state=123,
        n_iterations=3,
        major_cols=["x"],
        trace_cols=["y"],
        perturbation_seed=99,
        compute_pairwise=False,
        plot_everything=False,
        write_files=False,
        max_depth=10,
        verbose=False,
        return_details=True,
    )
    serial = dc.perturbative_simple_run(**common, n_jobs=1)
    parallel = dc.perturbative_simple_run(**common, n_jobs=2)

    pdt.assert_frame_equal(
        serial["hs_iterations"]
        .sort_values(["iteration", "depth_level", "neighbor_unit"], kind="stable")
        .reset_index(drop=True),
        parallel["hs_iterations"]
        .sort_values(["iteration", "depth_level", "neighbor_unit"], kind="stable")
        .reset_index(drop=True),
    )
    pdt.assert_frame_equal(
        serial["dihs_summary"]
        .sort_values(["neighbor_unit"], kind="stable")
        .reset_index(drop=True),
        parallel["dihs_summary"]
        .sort_values(["neighbor_unit"], kind="stable")
        .reset_index(drop=True),
    )
    pdt.assert_frame_equal(
        serial["top1_frequency"]
        .sort_values(["neighbor_unit"], kind="stable")
        .reset_index(drop=True),
        parallel["top1_frequency"]
        .sort_values(["neighbor_unit"], kind="stable")
        .reset_index(drop=True),
    )


def test_pseudo_unknown_parallel_matches_serial() -> None:
    df = _toy_dataset()
    common = dict(
        df=df,
        model_type="kmeans",
        transform_type="scaled",
        class_column="lettercode",
        sample_size=2,
        n_iterations=2,
        random_state=7,
        max_depth=10,
        plot_everything=False,
        write_files=False,
        verbose=False,
        return_details=True,
    )
    serial = dc.pseudo_unknown_run(**common, n_jobs=1)
    parallel = dc.pseudo_unknown_run(**common, n_jobs=2)

    pdt.assert_frame_equal(
        serial["run_results"].sort_values(["run_id"], kind="stable").reset_index(drop=True),
        parallel["run_results"].sort_values(["run_id"], kind="stable").reset_index(drop=True),
    )
    pdt.assert_frame_equal(
        serial["threshold_summary"].sort_values(
            ["integration_depth"], kind="stable"
        ).reset_index(drop=True),
        parallel["threshold_summary"].sort_values(
            ["integration_depth"], kind="stable"
        ).reset_index(drop=True),
    )
