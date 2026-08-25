import os

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


def create_model(model_type, random_state=None, model_params=None, **kwargs):
    """
    This function builds the clustering model based on the specified type; it also controls the initialization parameters in the non-deterministic models
    """
    params = dict(model_params or {})

    if model_type == "agglomerative":
        return AgglomerativeClustering(n_clusters=2, linkage="ward", **kwargs)
    if model_type == "gaussian":
        return GaussianMixture(
            n_components=2,
            random_state=random_state,
            n_init=int(params.get("gmm_n_init", 10)),
            covariance_type=str(params.get("gmm_covariance_type", "diag")),
            reg_covar=float(params.get("gmm_reg_covar", 1e-4)),
            **kwargs,
        )
    if model_type == "kmeans":
        return KMeans(n_clusters=2, n_init='auto', random_state=random_state, **kwargs)
    raise ValueError(f"Model type '{model_type}' is not supported.")


def _depth_column_name(model_name: str, depth: int) -> str:
    return f"Depth_{model_name}_{depth}" if depth > 0 else f"Depth_{model_name}"


def _all_rows_identical(values: np.ndarray) -> bool:
    if values.shape[0] <= 1:
        return True
    return bool(np.all(values[1:] == values[:1]))


def _single_class(values: np.ndarray, row_positions: np.ndarray) -> bool:
    if row_positions.size <= 1:
        return True
    first_value = values[row_positions[0]]
    return bool(np.all(values[row_positions] == first_value))


def recursive_cluster(
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
    """
    This function is for recursively clustering the data and storing the resulting hierarchical tree
    """

    model_name = model_type.title()
    root_depth = int(depth)
    cluster_data_store = {}
    row_count = len(df)
    if row_count == 0:
        return df.copy(), cluster_data_store

    feature_matrix = df[list(features)].to_numpy(dtype=float, copy=True)
    class_values = df[class_column].to_numpy(copy=False)
    row_ids = df.index.to_numpy(copy=False)
    depth_capacity = max(int(max_depth) - root_depth, 1)
    depth_labels = np.full((depth_capacity, row_count), np.nan, dtype=float)
    visited_depths: set[int] = set()

    def _store_labels(current_depth: int, row_positions: np.ndarray, labels: np.ndarray | int) -> None:
        rel_depth = current_depth - root_depth
        depth_labels[rel_depth, row_positions] = labels
        visited_depths.add(current_depth)

    def _recurse(row_positions: np.ndarray, current_depth: int, current_path: str) -> np.ndarray:
        col = _depth_column_name(model_name, current_depth)

        # Stopping conditions: max depth reached, too few samples, or only one class present.
        if (
            current_depth >= max_depth
            or row_positions.size <= 3
            or _single_class(class_values, row_positions)
        ):
            _store_labels(current_depth, row_positions, 0)
            return row_positions

        try:
            x = feature_matrix[row_positions]

            # Check for cases where clustering would fail due to lack of variability.
            if _all_rows_identical(x):
                _store_labels(current_depth, row_positions, 0)
                return row_positions

            # Fit the model and predict cluster labels.
            model = create_model(
                model_type,
                random_state=random_state,
                model_params=model_params,
            )
            labels = np.asarray(model.fit_predict(x), dtype=float)

            # Check whether clustering produced meaningful results, i.e. at least 2 clusters
            # and not all samples in one cluster (leads to infinite recursion).
            uniq, cnts = np.unique(labels, return_counts=True)
            if uniq.size < 2 or cnts.max() == row_positions.size:
                _store_labels(current_depth, row_positions, 0)
                return row_positions

            _store_labels(current_depth, row_positions, labels)

            ordered_children = []
            for label in uniq:
                label_mask = labels == label
                child_positions = row_positions[label_mask]
                next_path = f"{current_path}_{int(label)}" if current_path else f"{int(label)}"
                cluster_name = f"{unknown_class}_path{next_path}"
                cluster_data_store[cluster_name] = {
                    "unknown_class": unknown_class,
                    "path": next_path,
                    "depth": current_depth,
                    "label": int(label),
                    "indices": row_ids[child_positions].tolist(),
                    "data_path": None,
                }

                if (
                    current_depth + 1 >= max_depth
                    or child_positions.size <= 3
                    or _single_class(class_values, child_positions)
                ):
                    ordered_children.append(child_positions)
                else:
                    ordered_children.append(
                        _recurse(child_positions, current_depth + 1, next_path)
                    )
            return np.concatenate(ordered_children)

        except Exception as e:
            print(f"Clustering failed at depth {current_depth} with error: {e}")
            _store_labels(current_depth, row_positions, 0)
            return row_positions

    ordered_positions = _recurse(np.arange(row_count, dtype=int), root_depth, path)
    output = df.iloc[ordered_positions].copy()
    for current_depth in sorted(visited_depths):
        rel_depth = current_depth - root_depth
        values = depth_labels[rel_depth, ordered_positions]
        col = _depth_column_name(model_name, current_depth)
        output[col] = values
        if not np.isnan(values).any():
            output[col] = output[col].astype(int)

    return output, cluster_data_store
