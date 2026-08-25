from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from DIHS_Correlator.core.clustering import create_model, recursive_cluster
from DIHS_Correlator.workflows import single_run


def _legacy_recursive_cluster(
    df,
    features,
    model_type,
    depth=0,
    max_depth=100,
    unknown_class=None,
    path="",
    random_state=None,
    class_column="controlcode",
    model_params=None,
):
    model_name = model_type.title()
    col = f"Depth_{model_name}_{depth}" if depth > 0 else f"Depth_{model_name}"
    cluster_data_store = {}

    if depth >= max_depth or len(df) <= 3 or df[class_column].nunique() == 1:
        df[col] = 0
        return df, cluster_data_store

    try:
        x = df[features].to_numpy(dtype=float)
        if np.all(np.ptp(x, axis=0) == 0):
            df[col] = 0
            return df, cluster_data_store

        model = create_model(
            model_type,
            random_state=random_state,
            model_params=model_params,
        )
        labels = model.fit_predict(x)

        uniq, cnts = np.unique(labels, return_counts=True)
        if uniq.size < 2 or cnts.max() == len(df):
            df[col] = 0
            return df, cluster_data_store

        df[col] = labels

        for label in np.unique(labels):
            cluster_data = df[df[col] == label].copy()
            current_path = f"{path}_{label}" if path else f"{label}"
            cluster_name = f"{unknown_class}_path{current_path}"
            cluster_data_store[cluster_name] = {
                "unknown_class": unknown_class,
                "path": current_path,
                "depth": depth,
                "label": int(label),
                "indices": cluster_data.index.tolist(),
                "data_path": None,
            }

    except Exception as exc:  # pragma: no cover - parity branch
        print(f"Clustering failed at depth {depth} with error: {exc}")
        df[col] = 0
        return df, cluster_data_store

    result = []
    for label in np.unique(df[col]):
        sub_df = df[df[col] == label]
        if depth + 1 >= max_depth or len(sub_df) <= 3 or sub_df[class_column].nunique() == 1:
            result.append(sub_df)
        else:
            current_path = f"{path}_{label}" if path else f"{label}"
            sub_result, sub_clusters = _legacy_recursive_cluster(
                sub_df.copy(),
                features,
                model_type,
                depth=depth + 1,
                max_depth=max_depth,
                unknown_class=unknown_class,
                path=current_path,
                random_state=random_state,
                class_column=class_column,
                model_params=model_params,
            )
            result.append(sub_result)
            cluster_data_store.update(sub_clusters)
    return pd.concat(result), cluster_data_store


def _mixed_feature_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "controlcode": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "C",
                "X",
                "X",
            ],
            "x1": [
                1.00,
                1.02,
                1.00,
                1.01,
                5.00,
                5.02,
                5.00,
                5.01,
                8.00,
                8.02,
                8.00,
                8.01,
                1.03,
                1.04,
            ],
            "x2": [
                1.00,
                0.99,
                1.00,
                1.01,
                5.00,
                5.01,
                5.00,
                4.99,
                8.00,
                7.99,
                8.00,
                8.01,
                1.02,
                1.03,
            ],
            "x3": [
                0.10,
                0.11,
                0.10,
                0.10,
                0.50,
                0.49,
                0.50,
                0.50,
                0.90,
                0.89,
                0.90,
                0.90,
                0.11,
                0.12,
            ],
        }
    )


def _small_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "controlcode": ["A", "A", "X"],
            "x1": [1.0, 1.1, 1.05],
            "x2": [2.0, 2.1, 2.05],
        }
    )


def _workflow_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.00, 1.10, 0.90, 1.20, 0.80, 5.00, 5.10, 4.90, 8.80, 9.10, 8.90, 1.05],
            "y": [1.00, 0.90, 1.10, 1.20, 0.80, 5.10, 4.90, 5.00, 8.90, 9.20, 8.80, 1.10],
            "lettercode": [
                "AI",
                "AI",
                "AI",
                "AI",
                "AI",
                "PF",
                "PF",
                "PF",
                "BT",
                "BT",
                "BT",
                "Caio",
            ],
        }
    )


def _assert_recursive_match(
    df: pd.DataFrame,
    *,
    model_type: str,
    class_column: str,
    random_state: int | None,
) -> None:
    features = [column for column in df.columns if column != class_column]
    expected_df, expected_clusters = _legacy_recursive_cluster(
        df.copy(),
        features,
        model_type,
        max_depth=12,
        unknown_class="X",
        random_state=random_state,
        class_column=class_column,
    )
    actual_df, actual_clusters = recursive_cluster(
        df.copy(),
        features,
        model_type,
        max_depth=12,
        unknown_class="X",
        random_state=random_state,
        class_column=class_column,
    )
    pdt.assert_frame_equal(actual_df, expected_df, check_dtype=False)
    assert actual_clusters == expected_clusters


def test_recursive_cluster_matches_legacy_kmeans_on_duplicate_rows() -> None:
    _assert_recursive_match(
        _mixed_feature_dataset(),
        model_type="kmeans",
        class_column="controlcode",
        random_state=123,
    )


def test_recursive_cluster_matches_legacy_agglomerative_on_duplicate_rows() -> None:
    _assert_recursive_match(
        _mixed_feature_dataset(),
        model_type="agglomerative",
        class_column="controlcode",
        random_state=None,
    )


def test_recursive_cluster_small_dataset_terminates_like_legacy() -> None:
    _assert_recursive_match(
        _small_dataset(),
        model_type="kmeans",
        class_column="controlcode",
        random_state=123,
    )


def test_run_single_model_workflow_matches_legacy_ranking(monkeypatch) -> None:
    df = _workflow_dataset()
    common = dict(
        df=df,
        model_type="kmeans",
        transform_type="scaled",
        class_column="lettercode",
        unknown_sample="Caio",
        random_state=123,
        compute_pairwise=False,
        plot_everything=False,
        write_files=False,
        max_depth=12,
        verbose=False,
    )

    actual = single_run.run_single_model_workflow(**common)
    monkeypatch.setattr(single_run, "recursive_cluster", _legacy_recursive_cluster)
    expected = single_run.run_single_model_workflow(**common)

    pdt.assert_frame_equal(
        actual["hs_per_depth"]
        .sort_values(["depth_level", "neighbor_unit"], kind="stable")
        .reset_index(drop=True),
        expected["hs_per_depth"]
        .sort_values(["depth_level", "neighbor_unit"], kind="stable")
        .reset_index(drop=True),
    )
    pdt.assert_frame_equal(
        actual["dihs_total"]
        .sort_values(["neighbor_unit"], kind="stable")
        .reset_index(drop=True),
        expected["dihs_total"]
        .sort_values(["neighbor_unit"], kind="stable")
        .reset_index(drop=True),
    )
