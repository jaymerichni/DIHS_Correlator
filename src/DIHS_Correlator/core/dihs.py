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
    This function processes the output dataframe generated through recursive clustering and returns Harmonic Score per-depth and DIHS computations;
    if the compute_pairwise flag is activated then HS per-depth and DIHS are computed pairwise between all classes.
    """

    model_cap = model_type.title()
    depth_cols = [col for col in df_clustered.columns if col.startswith(f"Depth_{model_cap}")]

    if not depth_cols:
        depth_cols = [f"Depth_{model_cap}"]
        depth_values = np.zeros((len(df_clustered), 1), dtype=int)
        class_values = df_clustered[class_column]
    else:
        depth_values = (
            df_clustered[depth_cols].fillna("").astype(str).to_numpy(copy=True)
        )
        class_values = df_clustered[class_column]

    #Identify the rows corresponding to the unknown class and count them
    unk_class_mask = class_values == unknown_class
    n_u = int(unk_class_mask.sum())
    if n_u == 0:
        return None

    #Create dictionary with total rows per class
    total_rows_per_unit = class_values.value_counts().to_dict()
    max_depth = len(depth_cols)
    full_index = pd.Index(range(max_depth), name="depth_level")
    units = sorted(total_rows_per_unit.keys(), key=_sortable_unit_key)
    unit_index = pd.Index(units)
    unknown_unit_index = int(unit_index.get_loc(unknown_class))
    total_rows_arr = np.array([float(total_rows_per_unit[unit]) for unit in units], dtype=float)

    path_codes_per_depth = []
    cumulative_codes = None
    for d in range(max_depth):
        depth_labels = depth_values[:, d]
        if cumulative_codes is None:
            cumulative_codes = pd.factorize(depth_labels, sort=False)[0]
        else:
            cumulative_codes = pd.factorize(
                pd.MultiIndex.from_arrays([cumulative_codes, depth_labels]),
                sort=False,
            )[0]
        path_codes_per_depth.append(cumulative_codes.copy())

    active_rows = []
    pair_depth_matrices = {} if compute_pairwise else None

    excl = set(exclude_neighbors) if exclude_neighbors is not None else set()
    if exclude_self:
        excl.add(unknown_class)

    for d in range(max_depth):
        counts = (
            pd.crosstab(path_codes_per_depth[d], class_values, dropna=False)
            .reindex(columns=units, fill_value=0)
        )
        if counts.empty:
            continue

        counts_arr = counts.to_numpy(dtype=float, copy=False)
        presence_arr = counts_arr > 0.0
        unknown_counts = counts_arr[:, unknown_unit_index]
        active_mask = unknown_counts > 0.0

        if active_mask.any():
            active_presence = presence_arr[active_mask]
            active_unknown_props = unknown_counts[active_mask] / float(n_u)
            unknown_prop_sum = active_unknown_props @ active_presence.astype(float)
            neighbor_prop_sum = (counts_arr[active_mask] / total_rows_arr).sum(axis=0)
            include_mask = active_presence.any(axis=0)
            if excl:
                include_mask &= ~np.array([unit in excl for unit in units], dtype=bool)
            if include_mask.any():
                active_rows.append(
                    pd.DataFrame(
                        {
                            "depth_level": d,
                            "neighbor_unit": np.array(units, dtype=object)[include_mask],
                            f"{unknown_class}_prop_sum": unknown_prop_sum[include_mask],
                            "neighbor_prop_sum": neighbor_prop_sum[include_mask],
                        }
                    )
                )

        if compute_pairwise:
            prop_arr = counts_arr / total_rows_arr
            presence_float = presence_arr.astype(float)
            prop_sum_shared = prop_arr.T @ presence_float
            num = 2.0 * prop_sum_shared * prop_sum_shared.T
            den = prop_sum_shared + prop_sum_shared.T
            with np.errstate(divide="ignore", invalid="ignore"):
                hs = np.where(den > 0.0, num / den, 0.0)
            mat = pd.DataFrame(hs, index=units, columns=units)
            mat = 0.5 * (mat + mat.T)
            arr = mat.to_numpy(copy=True)
            np.fill_diagonal(arr, 1.0)
            mat.iloc[:, :] = arr
            pair_depth_matrices[d] = mat
             
    # After processing all depth levels, concatenate rows to create the a format dataframe with proportions
    active_long = pd.concat(active_rows, ignore_index=True) if active_rows else pd.DataFrame()
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

    pair_total_matrix = None

    # If compute_pairwise is activated, turn the pairwise HS at each depth into matrices with units as rows and columns
    # Compute the average across depth levels to get an overall DIHS matrix
    if compute_pairwise and pair_depth_matrices:
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
