import os

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

def create_model(model_type, random_state=None, **kwargs):
    """
    This function builds the clustering model based on the specified type; it also controls the initialization parameters in the non-deterministic models
    """

    if model_type == "agglomerative":
        return AgglomerativeClustering(n_clusters=2, linkage="ward", **kwargs)
    if model_type == "gaussian":
        return GaussianMixture(
            n_components=2,
            random_state=random_state,
            n_init=10,
            covariance_type="diag",
            reg_covar=1e-4,
            **kwargs,
        )
    if model_type == "kmeans":
        return KMeans(n_clusters=2, n_init='auto', random_state=random_state, **kwargs)
    raise ValueError(f"Model type '{model_type}' is not supported.")


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
):
    """
    This function is for recursively clustering the data and storing the resulting hierarchical tree
    """

    model_name = model_type.title()
    col = f"Depth_{model_name}_{depth}" if depth > 0 else f"Depth_{model_name}"
    cluster_data_store = {}

    # Stopping conditions: max depth reached, too few samples, or only one class present
    if depth >= max_depth or len(df) <= 3 or df[class_column].nunique() == 1:
        df[col] = 0
        return df, cluster_data_store

    try:
        x = df[features].to_numpy(dtype=float)

        # Check for cases where clustering would fail due to lack of variability
        unique_rows = np.unique(x, axis=0)
        if unique_rows.shape[0] < 2:
            df[col] = 0
            return df, cluster_data_store

        # Fit the model and predict cluster labels
        model = create_model(model_type, random_state=random_state)
        labels = model.fit_predict(x)

        # Check whether clustering produced meaningful results, i.e. at least 2 clusters and not all samples in one cluster (leads to infinite recursion)
        uniq, cnts = np.unique(labels, return_counts=True)
        if uniq.size < 2 or cnts.max() == len(df):
            df[col] = 0
            return df, cluster_data_store

        # Store clustering outcome in Dataframe as the value of a column named after depth
        df[col] = labels

        # For each cluster, store metadata for downstream workflow-level persistence.
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

    except Exception as e:
        print(f"Clustering failed at depth {depth} with error: {e}")
        df[col] = 0
        return df, cluster_data_store

    """ 
    Recurse on children: 
    - iterate over each subset of the data corresponding to the current depth's clusters 
    - combine the results into a single DataFrame
    - cluster_data_store is updated with the results from each recursive call
    """
    result = []
    for label in np.unique(df[col]):
        sub_df = df[df[col] == label]

        # Stopping conditions for recursion: max depth reached, too few samples, or only one class present
        if depth + 1 >= max_depth or len(sub_df) <= 3 or sub_df[class_column].nunique() == 1:
            result.append(sub_df)
        else:
            current_path = f"{path}_{label}" if path else f"{label}"
            sub_result, sub_clusters = recursive_cluster(
                sub_df.copy(),
                features,
                model_type,
                depth=depth + 1,
                max_depth=max_depth,
                unknown_class=unknown_class,
                path=current_path,
                random_state=random_state,
                class_column=class_column,
            )
            result.append(sub_result)
            cluster_data_store.update(sub_clusters)

    # Return a dataframe similar to the input with added cluster labels + the dictionary containing metadata for each cluster
    return pd.concat(result), cluster_data_store
