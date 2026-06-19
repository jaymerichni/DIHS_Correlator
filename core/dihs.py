import numpy as np
import pandas as pd


def _sortable_unit_key(x):
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))


def compute_dihs_metrics(
    df_clustered,
    unknown_class,
    model_type,
    transform_name,
    class_column="controlcode",
    compute_pairwise=True,
    exclude_neighbors=None,
    exclude_self=False,
):
    """
    This function processes the output dataframe generated through recursive clustering and returns Harmonic Score per-depth and DIHS computations.
    If the compute_pairwise flag is activated then HS per-depth and DIHS are computed pairwise between all classes.
    """

    model_cap = model_type.title()
    depth_cols = [col for col in df_clustered.columns if col.startswith(f"Depth_{model_cap}")]

    if not depth_cols:
        tmp = df_clustered.copy()
        tmp[f"Depth_{model_cap}"] = 0
        df_work = tmp
        depth_cols = [f"Depth_{model_cap}"]
    else:
        df_work = df_clustered.copy()

    #Identify the rows corresponding to the unknown class and count them
    unk_class_mask = df_work[class_column] == unknown_class
    n_u = int(unk_class_mask.sum())
    if n_u == 0:
        return None

    #Create dictionary with total rows per class
    total_rows_per_unit = df_work[class_column].value_counts().to_dict()
    max_depth = len(depth_cols)
    full_index = pd.Index(range(max_depth), name="depth_level")

    #Extract the path at each depth level for each row, turn it into a tuple and store in new columns path_0, path_1, ..., path_{max_depth-1}
    for d in range(max_depth):
        path_col = f"path_{d}"
        if path_col not in df_work.columns:
            cols = depth_cols[: d + 1]
            df_work[path_col] = df_work[cols].apply(
                lambda row: tuple(row.fillna("").astype(str).tolist()), axis=1
            )

    active_rows = []
    pair_rows_per_depth = []

    excl = set(exclude_neighbors) if exclude_neighbors is not None else set()
    if exclude_self:
        excl.add(unknown_class)

    for d in range(max_depth):
        path_col = f"path_{d}"

        #Compute counts of each class per path at depth d
        base_counts = (
            df_work.groupby([path_col, class_column], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )

        if base_counts.empty:
            active_rows.append(
                pd.DataFrame(
                    {
                        "depth_level": [],
                        "neighbor_unit": [],
                        f"{unknown_class}_prop_sum": [],
                        "neighbor_prop_sum": [],
                    }
                )
            )
        else:
            # For each path, identify the count of the unknown class and merge it back to get the total count of the unknown class per path
            counts = base_counts.rename(columns={class_column: "neighbor_unit"})
            unknown_class_per_path = (
                counts[counts["neighbor_unit"] == unknown_class][[path_col, "count"]]
                .rename(columns={"count": "unknown_count"})
            )
            counts = counts.merge(unknown_class_per_path, on=path_col, how="left")
            counts["unknown_count"] = counts["unknown_count"].fillna(0).astype(int)
            counts = counts[counts["unknown_count"] > 0]

            if counts.empty:
                active_rows.append(
                    pd.DataFrame(
                        {
                            "depth_level": [],
                            "neighbor_unit": [],
                            f"{unknown_class}_prop_sum": [],
                            "neighbor_prop_sum": [],
                        }
                    )
                )
            else:
                # Map the neighbor unit to the total count of rows for that unit to compute the proportion of neighbors in that unit
                counts["neighbor_total"] = (
                    counts["neighbor_unit"].map(total_rows_per_unit).astype(float)
                )
                if excl:
                    counts = counts[~counts["neighbor_unit"].isin(excl)]
                counts = counts[counts["neighbor_total"].notna()]

                if counts.empty:
                    active_rows.append(
                        pd.DataFrame(
                            {
                                "depth_level": [],
                                "neighbor_unit": [],
                                f"{unknown_class}_prop_sum": [],
                                "neighbor_prop_sum": [],
                            }
                        )
                    )
                else:
                    # Compute the proportion of the unknown class and nieghbors in each path
                    counts[f"{unknown_class}_prop"] = counts["unknown_count"] / float(n_u)
                    counts["neighbor_prop"] = counts["count"] / counts["neighbor_total"]
                    # Aggregate the proportions per neighbor and depth level to compute the sum of proportions for the unknown class and neighbors in each unit at depth d
                    active_agg = (
                        counts.groupby("neighbor_unit", observed=True)[
                            [f"{unknown_class}_prop", "neighbor_prop"]
                        ]
                        .sum()
                        .reset_index()
                        .rename(
                            columns={
                                f"{unknown_class}_prop": f"{unknown_class}_prop_sum",
                                "neighbor_prop": "neighbor_prop_sum",
                            }
                        )
                    )
                    active_agg.insert(0, "depth_level", d)
                    active_rows.append(active_agg)

        # If compute_pairwise is activated, compute pairwise harmonic scores between all units at depth d
        if compute_pairwise and not base_counts.empty:
            pairs = base_counts.merge(base_counts, on=path_col, suffixes=("_a", "_b"))
            class_col_a = f"{class_column}_a"
            class_col_b = f"{class_column}_b"
            pairs["total_a"] = pairs[class_col_a].map(total_rows_per_unit).astype(float)
            pairs["total_b"] = pairs[class_col_b].map(total_rows_per_unit).astype(float)
            pairs = pairs[pairs["total_a"].notna() & pairs["total_b"].notna()]
            if not pairs.empty:
                pairs["prop_a"] = pairs["count_a"] / pairs["total_a"]
                pairs["prop_b"] = pairs["count_b"] / pairs["total_b"]
                depth_pairs_agg = (
                    pairs.groupby([class_col_a, class_col_b], observed=True)[
                        ["prop_a", "prop_b"]
                    ]
                    .sum()
                    .reset_index()
                    .rename(
                        columns={class_col_a: "unit_a", class_col_b: "unit_b"}
                    )
                )
                num = 2.0 * depth_pairs_agg["prop_a"] * depth_pairs_agg["prop_b"]
                den = depth_pairs_agg["prop_a"] + depth_pairs_agg["prop_b"]
                depth_pairs_agg["harmonic_score"] = np.where(den > 0.0, num / den, 0.0)
                depth_pairs_agg.insert(0, "depth_level", d)
                pair_rows_per_depth.append(depth_pairs_agg)
            
    # After processing all depth levels, concatenate rows to create the a format dataframe with proportions
    active_long = pd.concat(active_rows, ignore_index=True)
    if active_long.empty:
        return None

    # Turn the long format dataframe into two wide format dataframes with neighbor units as columns and depth levels as rows, filling missing values with 0
    # This allows for keeping a rectangular grid format for depth levels and neighbor units even if some combinations are missing
    pt_active = (
        active_long.pivot(
            index="depth_level",
            columns="neighbor_unit",
            values=f"{unknown_class}_prop_sum",
        )
        .reindex(full_index)
        .fillna(0.0)
    )
    pa_neighbor = (
        active_long.pivot(
            index="depth_level",
            columns="neighbor_unit",
            values="neighbor_prop_sum",
        )
        .reindex(full_index)
        .fillna(0.0)
    )

    # Go back to long format with depth levels, neighbor units and proportions to compute the harmonic score per depth and neighbor unit
    active_depth_long = pd.DataFrame(
        {
            "depth_level": np.repeat(pt_active.index.values, pt_active.shape[1]),
            "neighbor_unit": np.tile(pt_active.columns.values, pt_active.shape[0]),
            f"{unknown_class}_prop_sum": pt_active.to_numpy().ravel(),
            "neighbor_prop_sum": pa_neighbor.to_numpy().ravel(),
        }
    )
    # Insert some metadata
    active_depth_long.insert(0, "unknown_class", unknown_class)
    active_depth_long.insert(1, "transform", transform_name)
    active_depth_long.insert(2, "model", model_type)

    # Compute the harmonic score per depth and neighbor
    num = (
        2
        * active_depth_long[f"{unknown_class}_prop_sum"]
        * active_depth_long["neighbor_prop_sum"]
    )
    den = (
        active_depth_long[f"{unknown_class}_prop_sum"]
        + active_depth_long["neighbor_prop_sum"]
    )
    active_depth_long["harmonic_score"] = np.where(den > 0, num / den, 0.0)

    # Compute DIHS by integrating the harmonic score across depth levels for each neighbor unit
    depth_span = max_depth

    def dihs(arr, depth_span=depth_span):
        return arr.sum() / float(depth_span)

    active_total = active_depth_long.groupby(
        ["unknown_class", "neighbor_unit"], as_index=False
    ).agg(total_product=("harmonic_score", dihs))

    pair_depth_matrices = None
    pair_total_matrix = None

    # If compute_pairwise is activated, turn the pairwise HS at each depth into matrices with units as rows and columns
    # Compute the average across depth levels to get an overall DIHS matrix
    if compute_pairwise and pair_rows_per_depth:
        pair_depth_long = pd.concat(pair_rows_per_depth, ignore_index=True)

        units = sorted(df_work[class_column].unique(), key=_sortable_unit_key)

        pair_depth_matrices = {}
        for d in range(max_depth):
            depth_d = pair_depth_long[pair_depth_long["depth_level"] == d]

            if depth_d.empty:
                mat = pd.DataFrame(0.0, index=units, columns=units)
            else:
                mat = (
                    depth_d.pivot(
                        index="unit_a", columns="unit_b", values="harmonic_score"
                    )
                    .reindex(index=units, columns=units)
                    .fillna(0.0)
                )
                mat = 0.5 * (mat + mat.T)

            mat = mat.copy()
            arr = mat.to_numpy(copy=True)
            np.fill_diagonal(arr, 1.0)
            mat.iloc[:, :] = arr
            pair_depth_matrices[d] = mat

        stack = np.stack(
            [pair_depth_matrices[d].to_numpy(dtype=float) for d in range(max_depth)],
            axis=0,
        )
        total_from_depth = stack.mean(axis=0)
        pair_total_matrix = pd.DataFrame(total_from_depth, index=units, columns=units)
        pair_total_matrix = 0.5 * (pair_total_matrix + pair_total_matrix.T)
        arr = pair_total_matrix.to_numpy(copy=True)
        np.fill_diagonal(arr, 1.0)

        pair_total_matrix = pd.DataFrame(
            arr,
            index=pair_total_matrix.index,
            columns=pair_total_matrix.columns,
        )

    return {
    "active_depth_long": active_depth_long,
    "active_total": active_total,
    "pair_depth_matrices": pair_depth_matrices,
    "pair_total_matrix": pair_total_matrix,
    }
