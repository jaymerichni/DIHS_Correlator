import os
from typing import Any, Iterable

from pandas.api.types import is_numeric_dtype

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf

from Tephra_Correlator_Refactored.core.transforms import BASE_TRANSFORMATIONS
from Tephra_Correlator_Refactored.viz.method_comparison import (
    plot_pseudo_unknown_margin_comparison,
)
from Tephra_Correlator_Refactored.workflows.single_run import CorrelationRunner


TRANSFORM_NAME_TO_ID = {v: k for k, v in BASE_TRANSFORMATIONS.items()}
TREE_METHODS = ("dihs", "single_cut_coassociation")

DEFAULT_MAJOR_COLS = [
    "SIO2N",
    "TIO2N",
    "AL2O3N",
    "FE2O3TN",
    "CAON",
    "MGON",
    "MNON",
    "NA2ON",
    "K2ON",
    "P2O5N",
]
DEFAULT_TRACE_COLS = ["NbN", "ZrN", "LaN", "CeN", "SrN", "BaN", "RbN"]


def _log(verbose: bool, message: str):
    if verbose:
        print(message)


def _normalize_transform_type(transform_type: str) -> int:
    if not isinstance(transform_type, str):
        raise ValueError(
            "transform_type must be a string name: 'none', 'ilr', 'clr', or 'scaled'."
        )
    key = str(transform_type).strip().lower()
    if key not in TRANSFORM_NAME_TO_ID:
        raise ValueError(f"Unsupported transform_type='{transform_type}'.")
    return TRANSFORM_NAME_TO_ID[key]


def _class_key(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _class_sort_key(value: Any):
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def _resolve_unknown_class(
    df: pd.DataFrame, unknown_sample: Any, class_column: str
) -> Any:
    if unknown_sample is None:
        raise ValueError("unknown_sample must be provided for benchmark comparisons.")
    if class_column not in df.columns:
        raise ValueError(f"Input dataframe must contain class column '{class_column}'.")

    class_values = df[class_column]
    if (class_values == unknown_sample).any():
        return unknown_sample

    if isinstance(unknown_sample, str):
        for caster in (int, float):
            try:
                parsed = caster(unknown_sample)
            except Exception:
                continue
            if (class_values == parsed).any():
                return parsed

    raise ValueError(
        f"Unknown sample '{unknown_sample}' was not found in class column '{class_column}'."
    )


def _resolve_major_trace_columns(
    df: pd.DataFrame,
    major_cols: Iterable[str] | None,
    trace_cols: Iterable[str] | None,
    class_column: str,
):
    
    numeric_cols = [
        c for c in df.columns
        if c != class_column and is_numeric_dtype(df[c])
    ]
    resolved_major = list(major_cols) if major_cols is not None else [
        c for c in DEFAULT_MAJOR_COLS if c in df.columns
    ]
    resolved_trace = list(trace_cols) if trace_cols is not None else [
        c for c in DEFAULT_TRACE_COLS if c in df.columns
    ]
    resolved_major = [c for c in resolved_major if c in numeric_cols]
    resolved_trace = [c for c in resolved_trace if c in numeric_cols]
    return resolved_major, resolved_trace


def _make_pseudo_unknown_label(
    existing_values: Iterable[Any], source_class: Any, sample_iteration: int, case: str
) -> str:
    existing_keys = {_class_key(v) for v in existing_values}
    base = (
        f"__comparison_pseudo_unknown__{case}__{_class_key(source_class)}__{sample_iteration}"
    )
    candidate = base
    counter = 1
    while candidate in existing_keys:
        candidate = f"{base}__{counter}"
        counter += 1
    return candidate


def _perturb_dataframe(
    df: pd.DataFrame,
    major_cols: list[str],
    trace_cols: list[str],
    major_error: float,
    trace_error: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    perturbed = df.copy()
    if major_cols:
        x_major = perturbed[major_cols].to_numpy(dtype=float)
        eps_major = rng.uniform(-major_error, major_error, size=x_major.shape)
        perturbed[major_cols] = x_major * (1.0 + eps_major)
    if trace_cols:
        x_trace = perturbed[trace_cols].to_numpy(dtype=float)
        eps_trace = rng.uniform(-trace_error, trace_error, size=x_trace.shape)
        perturbed[trace_cols] = x_trace * (1.0 + eps_trace)
    return perturbed


def _add_metadata(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for key, value in metadata.items():
        out[key] = value
    return out


def _exclude_unknown_neighbor(df: pd.DataFrame, unknown_class: Any) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["neighbor_unit"].astype(str) != str(unknown_class)].copy()


def _exclude_self_unknown_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "unknown_class" not in df.columns:
        return df
    return df[
        df["neighbor_unit"].astype(str) != df["unknown_class"].astype(str)
    ].copy()


def _score_key_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "run_id",
        "iteration",
        "source_class",
        "source_class_key",
        "case",
        "sample_iteration",
        "sample_size",
        "true_source_present",
        "unknown_class",
        "transform",
        "model",
        "neighbor_unit",
    ]
    return [col for col in preferred if col in df.columns]


def _compute_centroid_scores(
    transformed_data: pd.DataFrame,
    features: list[str],
    unknown_class: Any,
    class_column: str,
) -> pd.DataFrame:
    work = transformed_data.copy()
    unk_mask = work[class_column] == unknown_class
    if not unk_mask.any():
        raise ValueError(f"Unknown class '{unknown_class}' is not present in transformed data.")

    x = work[features].to_numpy(dtype=float)
    mu_unknown = x[unk_mask.to_numpy()].mean(axis=0)

    rows = []
    for cls in sorted(pd.unique(work[class_column]), key=_class_sort_key):
        if _class_key(cls) == _class_key(unknown_class):
            continue
        cls_mask = work[class_column] == cls
        if not cls_mask.any():
            continue
        mu_cls = x[cls_mask.to_numpy()].mean(axis=0)
        distance = float(np.linalg.norm(mu_unknown - mu_cls))
        rows.append({"neighbor_unit": cls, "score": -distance})

    return pd.DataFrame(rows)

def _compute_mahalanobis_scores(
    transformed_data: pd.DataFrame,
    features: list[str],
    unknown_class: Any,
    class_column: str,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Mahalanobis distance between class centroids using a pooled within-class
    shrinkage covariance estimated in the transformed feature space.

    Notes
    -----
    - Works for both ILR and CLR transformed data.
    - CLR covariance is theoretically singular, but LedoitWolf shrinkage plus
      pseudoinverse makes this numerically stable.
    - Score is returned as -distance so that larger is better for ranking.
    """
    work = transformed_data.copy()
    unk_mask = work[class_column] == unknown_class
    if not unk_mask.any():
        raise ValueError(f"Unknown class '{unknown_class}' is not present in transformed data.")

    x = work[features].to_numpy(dtype=float)
    labels = work[class_column].to_numpy()

    # Unknown centroid
    mu_unknown = x[unk_mask.to_numpy()].mean(axis=0)

    # Build pooled within-class residual matrix
    residual_blocks = []
    for cls in pd.unique(labels):
        cls_mask = labels == cls
        x_cls = x[cls_mask]
        if x_cls.shape[0] < 2:
            continue
        mu_cls = x_cls.mean(axis=0, keepdims=True)
        residual_blocks.append(x_cls - mu_cls)

    if residual_blocks:
        residuals = np.vstack(residual_blocks)
    else:
        # Fallback: if every class has size 1, use the raw transformed data
        residuals = x.copy()

    # Shrinkage covariance estimate
    cov = LedoitWolf().fit(residuals).covariance_
    cov = cov + eps * np.eye(cov.shape[0])
    cov_inv = np.linalg.pinv(cov)

    rows = []
    for cls in sorted(pd.unique(labels), key=_class_sort_key):
        if _class_key(cls) == _class_key(unknown_class):
            continue
        cls_mask = labels == cls
        if not cls_mask.any():
            continue
        mu_cls = x[cls_mask].mean(axis=0)
        delta = mu_unknown - mu_cls
        distance = float(np.sqrt(delta @ cov_inv @ delta))
        rows.append({"neighbor_unit": cls, "score": -distance})

    return pd.DataFrame(rows)

def _compute_coassociation_depth_scores(
    df_clustered: pd.DataFrame,
    unknown_class: Any,
    model_type: str,
    class_column: str,
) -> pd.DataFrame:
    model_cap = model_type.title()
    depth_cols = [col for col in df_clustered.columns if col.startswith(f"Depth_{model_cap}")]
    if not depth_cols:
        tmp = df_clustered.copy()
        tmp[f"Depth_{model_cap}"] = 0
        df_work = tmp
        depth_cols = [f"Depth_{model_cap}"]
    else:
        df_work = df_clustered.copy()

    unk_mask = df_work[class_column] == unknown_class
    n_u = int(unk_mask.sum())
    if n_u == 0:
        raise ValueError(f"Unknown class '{unknown_class}' not found in clustered dataframe.")

    total_rows_per_unit = df_work[class_column].value_counts().to_dict()
    units = sorted(df_work[class_column].unique(), key=_class_sort_key)
    full_index = pd.Index(range(len(depth_cols)), name="depth_level")

    for depth_level in range(len(depth_cols)):
        path_col = f"path_{depth_level}"
        cols = depth_cols[: depth_level + 1]
        if path_col not in df_work.columns:
            df_work[path_col] = df_work[cols].apply(
                lambda row: tuple(row.fillna("").astype(str).tolist()), axis=1
            )

    rows = []
    for depth_level in range(len(depth_cols)):
        path_col = f"path_{depth_level}"
        counts = (
            df_work.groupby([path_col, class_column], observed=True)
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={class_column: "neighbor_unit"})
        )
        if counts.empty:
            continue
        unknown_counts = (
            counts[counts["neighbor_unit"] == unknown_class][[path_col, "count"]]
            .rename(columns={"count": "unknown_count"})
        )
        counts = counts.merge(unknown_counts, on=path_col, how="left")
        counts["unknown_count"] = counts["unknown_count"].fillna(0).astype(int)
        counts = counts[counts["unknown_count"] > 0]
        if counts.empty:
            continue
        counts["neighbor_total"] = counts["neighbor_unit"].map(total_rows_per_unit).astype(float)
        counts = counts[counts["neighbor_total"].notna()]
        if counts.empty:
            continue
        counts["coassociation_score"] = (
            counts["unknown_count"] * counts["count"]
        ) / (float(n_u) * counts["neighbor_total"])
        depth_scores = (
            counts.groupby("neighbor_unit", observed=True)["coassociation_score"]
            .sum()
            .reset_index()
        )
        depth_scores.insert(0, "depth_level", depth_level)
        rows.append(depth_scores)

    if rows:
        long_df = pd.concat(rows, ignore_index=True)
        pivot = (
            long_df.pivot(index="depth_level", columns="neighbor_unit", values="coassociation_score")
            .reindex(full_index)
            .reindex(columns=units)
            .fillna(0.0)
        )
    else:
        pivot = pd.DataFrame(0.0, index=full_index, columns=units)

    return pd.DataFrame(
        {
            "depth_level": np.repeat(pivot.index.values, pivot.shape[1]),
            "neighbor_unit": np.tile(pivot.columns.values, pivot.shape[0]),
            "coassociation_score": pivot.to_numpy(dtype=float).ravel(),
        }
    )


def _recompute_cumulative_scores_by_depth(
    score_iterations: pd.DataFrame,
    score_col: str,
    key_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, int | None]:
    if score_iterations.empty:
        return pd.DataFrame(), None

    max_depth_per_run = (
        score_iterations.groupby("run_id", as_index=False)["depth_level"].max()
    )
    if max_depth_per_run.empty:
        return pd.DataFrame(), None

    common_depth_level = int(max_depth_per_run["depth_level"].min())
    keys = (
        [c for c in key_columns if c in score_iterations.columns]
        if key_columns is not None
        else [c for c in score_iterations.columns if c not in {"depth_level", score_col}]
    )
    rows = []
    for integration_depth in range(common_depth_level + 1):
        cut = score_iterations[score_iterations["depth_level"] <= integration_depth].copy()
        if cut.empty:
            continue
        agg = (
            cut.groupby(keys, as_index=False)
            .agg(score_sum=(score_col, "sum"))
            .assign(score=lambda x: x["score_sum"] / float(integration_depth + 1))
            .drop(columns=["score_sum"])
        )
        agg.insert(0, "integration_depth", integration_depth)
        rows.append(agg)

    if not rows:
        return pd.DataFrame(), common_depth_level
    return pd.concat(rows, ignore_index=True), common_depth_level


def _single_cut_scores_by_depth(
    score_iterations: pd.DataFrame,
    score_col: str,
    key_columns: list[str] | None = None,
) -> pd.DataFrame:
    if score_iterations.empty:
        return pd.DataFrame()
    keep_cols = (
        [c for c in key_columns if c in score_iterations.columns]
        if key_columns is not None
        else [c for c in score_iterations.columns if c not in {"depth_level", score_col}]
    )
    out = score_iterations[keep_cols + ["depth_level", score_col]].copy()
    out.insert(0, "integration_depth", out["depth_level"].astype(int))
    out = out.drop(columns=["depth_level"]).rename(columns={score_col: "score"})
    return out


def _extract_top_rankings(score_iterations: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "method",
        "run_id",
        "iteration",
        "integration_depth",
        "unknown_class",
        "transform",
        "model",
        "source_class",
        "source_class_key",
        "case",
        "sample_iteration",
        "sample_size",
        "true_source_present",
        "true_source_class",
        "true_source_key",
        "top1_class",
        "top2_class",
        "top1_score",
        "top2_score",
        "margin",
    ]

    if score_iterations.empty:
        return pd.DataFrame(columns=expected_columns)

    required = {"method", "run_id", "neighbor_unit", "score"}
    missing = required.difference(score_iterations.columns)

    if missing:
        raise ValueError(
            "_extract_top_rankings received non-empty score_iterations but "
            f"is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(score_iterations.columns)}\n"
            f"Shape: {score_iterations.shape}\n"
            f"Head:\n{score_iterations.head(10).to_string()}"
        )

    rows = []

    for (method, run_id), sub in score_iterations.groupby(["method", "run_id"]):
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)

        if sub.empty:
            continue

        top1_class = sub["neighbor_unit"].iloc[0]
        top1_score = float(sub["score"].iloc[0])

        top2_class = sub["neighbor_unit"].iloc[1] if len(sub) > 1 else np.nan
        top2_score = float(sub["score"].iloc[1]) if len(sub) > 1 else np.nan
        margin = top1_score - top2_score if len(sub) > 1 else np.nan

        metadata_cols = [
            c
            for c in (
                "iteration",
                "integration_depth",
                "unknown_class",
                "transform",
                "model",
                "source_class",
                "source_class_key",
                "case",
                "sample_iteration",
                "sample_size",
                "true_source_present",
                "true_source_class",
                "true_source_key",
            )
            if c in sub.columns
        ]

        row = {
            "method": method,
            "run_id": int(run_id),
        }

        for col in metadata_cols:
            row[col] = sub[col].iloc[0]

        row.update(
            {
                "top1_class": top1_class,
                "top2_class": top2_class,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "margin": margin,
            }
        )

        rows.append(row)

    if not rows:
        raise ValueError(
            "_extract_top_rankings received non-empty score_iterations, "
            "but produced no ranking rows.\n"
            f"Shape: {score_iterations.shape}\n"
            f"Columns: {list(score_iterations.columns)}\n"
            f"Head:\n{score_iterations.head(10).to_string()}"
        )

    return pd.DataFrame(rows)


def _summarize_method_scores(
    score_iterations: pd.DataFrame,
    common_depth_level: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if score_iterations.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    score_iterations = score_iterations.copy()
    score_summary = (
        score_iterations.groupby(["method", "neighbor_unit"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            n_runs=("score", "count"),
        )
        .sort_values(["method", "score_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    run_rankings = _extract_top_rankings(score_iterations)
    winner_counts = (
        run_rankings.groupby(["method", "top1_class"])
        .size()
        .reset_index(name="wins")
        if not run_rankings.empty
        else pd.DataFrame(columns=["method", "top1_class", "wins"])
    )
    run_counts = (
        run_rankings.groupby("method").size().reset_index(name="n_runs")
        if not run_rankings.empty
        else pd.DataFrame(columns=["method", "n_runs"])
    )

    rows = []
    for method, sub in score_summary.groupby("method", as_index=False):
        sub = sub.sort_values("score_mean", ascending=False).reset_index(drop=True)
        top1_class = sub["neighbor_unit"].iloc[0] if not sub.empty else np.nan
        top2_class = sub["neighbor_unit"].iloc[1] if len(sub) > 1 else np.nan
        top1_score_mean = float(sub["score_mean"].iloc[0]) if not sub.empty else np.nan
        top2_score_mean = float(sub["score_mean"].iloc[1]) if len(sub) > 1 else np.nan
        margin = top1_score_mean - top2_score_mean if len(sub) > 1 else np.nan

        n_runs_match = run_counts[run_counts["method"] == method]
        n_runs = int(n_runs_match["n_runs"].iloc[0]) if not n_runs_match.empty else 0
        wins_match = winner_counts[
            (winner_counts["method"] == method) & (winner_counts["top1_class"] == top1_class)
        ]
        top1_frequency = (
            float(wins_match["wins"].iloc[0]) / float(n_runs)
            if (n_runs > 0 and not wins_match.empty)
            else np.nan
        )

        rows.append(
            {
                "method": method,
                "top1_class": top1_class,
                "top2_class": top2_class,
                "top1_score_mean": top1_score_mean,
                "top2_score_mean": top2_score_mean,
                "margin": margin,
                "top1_frequency": top1_frequency,
                "n_runs": n_runs,
                "common_depth_level": (
                    common_depth_level if method in TREE_METHODS else np.nan
                ),
            }
        )

    return score_summary, run_rankings, pd.DataFrame(rows)


def _build_main_comparison_table(
    benchmark_summary: pd.DataFrame | None,
    pseudo_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    if benchmark_summary is None or benchmark_summary.empty:
        benchmark_table = pd.DataFrame(columns=["method"])
    else:
        benchmark_table = benchmark_summary.rename(
            columns={
                "top1_class": "benchmark_top1",
                "top2_class": "benchmark_top2",
                "margin": "benchmark_margin",
                "top1_frequency": "benchmark_top1_frequency",
                "common_depth_level": "benchmark_common_depth_level",
            }
        )[
            [
                "method",
                "benchmark_top1",
                "benchmark_top2",
                "benchmark_margin",
                "benchmark_top1_frequency",
                "benchmark_common_depth_level",
            ]
        ]

    if pseudo_summary is None or pseudo_summary.empty:
        pseudo_table = pd.DataFrame(columns=["method"])
    else:
        pseudo_table = pseudo_summary.rename(
            columns={
                "positive_top1_accuracy": "pseudo_unknown_top1_accuracy",
                "margin_auroc": "pseudo_unknown_margin_auroc",
                "positive_margin_median": "pseudo_unknown_positive_margin_median",
                "negative_margin_median": "pseudo_unknown_negative_margin_median",
            }
        )[
            [
                "method",
                "pseudo_unknown_top1_accuracy",
                "pseudo_unknown_margin_auroc",
                "pseudo_unknown_positive_margin_median",
                "pseudo_unknown_negative_margin_median",
            ]
        ]

    if benchmark_table.empty:
        return pseudo_table.copy()
    if pseudo_table.empty:
        return benchmark_table.copy()
    return benchmark_table.merge(pseudo_table, on="method", how="outer")


def _run_comparison_ensemble(
    *,
    df: pd.DataFrame,
    unknown_class: Any,
    class_column: str,
    model_type: str,
    transform_id: int,
    random_state: int | None,
    n_perturbations: int,
    max_depth: int,
    exclude_columns,
    major_cols: list[str],
    trace_cols: list[str],
    major_error: float,
    trace_error: float,
    perturbation_seed: int | None,
    context_metadata: dict[str, Any] | None = None,
    apply_perturbation: bool = True,
    verbose: bool = False,
    progress_label: str | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(perturbation_seed)
    runner = CorrelationRunner(
        base_output_dir="./Results",
        save_trees=False,
        save_cluster_data=False,
        save_untransformed=False,
    )
    exclude_set = set(exclude_columns) | {class_column}
    runner.set_feature_columns(df.copy(), exclude=tuple(exclude_set), verbose=False)

    hs_rows = []
    coassociation_rows = []
    centroid_rows = []
    mahalanobis_rows = []

    for iteration in range(int(n_perturbations)):
        if verbose:
            label = progress_label or "comparison ensemble"
            print(f"{label}: perturbation {iteration + 1}/{int(n_perturbations)}")
        perturbed = (
            _perturb_dataframe(
                df=df,
                major_cols=major_cols,
                trace_cols=trace_cols,
                major_error=major_error,
                trace_error=trace_error,
                rng=rng,
            )
            if apply_perturbation
            else df.copy()
        )
        run = runner.run_combination(
            data=perturbed,
            transform_type=transform_id,
            model_type=model_type,
            random_state=random_state if model_type in ("kmeans", "gaussian") else None,
            unknown_class=unknown_class,
            class_column=class_column,
            compute_pairwise=False,
            write_outputs=False,
            max_depth=max_depth,
            return_intermediates=True,
        )

        metadata = {"run_id": iteration, "iteration": iteration}
        if context_metadata:
            metadata.update(context_metadata)

        hs_i = _add_metadata(run["metrics_per_depth"], metadata)
        hs_rows.append(hs_i)

        coassoc_i = _compute_coassociation_depth_scores(
            df_clustered=run["clustered_tree"],
            unknown_class=unknown_class,
            model_type=model_type,
            class_column=class_column,
        )
        coassoc_i["unknown_class"] = unknown_class
        coassoc_i["transform"] = run["transform_name"]
        coassoc_i["model"] = model_type
        coassoc_i = _add_metadata(coassoc_i, metadata)
        coassociation_rows.append(coassoc_i)

        centroid_i = _compute_centroid_scores(
            transformed_data=run["transformed_data"],
            features=run["features"],
            unknown_class=unknown_class,
            class_column=class_column,
        )
        centroid_i["unknown_class"] = unknown_class
        centroid_i["transform"] = run["transform_name"]
        centroid_i["model"] = model_type
        centroid_i = _add_metadata(centroid_i, metadata)
        centroid_rows.append(centroid_i)

        mahalanobis_i = _compute_mahalanobis_scores(
            transformed_data=run["transformed_data"],
            features=run["features"],
            unknown_class=unknown_class,
            class_column=class_column,
        )
        mahalanobis_i["unknown_class"] = unknown_class
        mahalanobis_i["transform"] = run["transform_name"]
        mahalanobis_i["model"] = model_type
        mahalanobis_i = _add_metadata(mahalanobis_i, metadata)
        mahalanobis_rows.append(mahalanobis_i)

    hs_iterations = pd.concat(hs_rows, ignore_index=True) if hs_rows else pd.DataFrame()
    coassociation_iterations = (
        pd.concat(coassociation_rows, ignore_index=True)
        if coassociation_rows
        else pd.DataFrame()
    )
    centroid_iterations = (
        pd.concat(centroid_rows, ignore_index=True) if centroid_rows else pd.DataFrame()
    )

    mahalanobis_iterations = (
        pd.concat(mahalanobis_rows, ignore_index=True)
        if mahalanobis_rows
        else pd.DataFrame()
    )

    score_keys = _score_key_columns(hs_iterations)
    dihs_by_depth, dihs_common_depth = _recompute_cumulative_scores_by_depth(
        hs_iterations, score_col="harmonic_score", key_columns=score_keys
    )
    single_cut_by_depth = _single_cut_scores_by_depth(
        coassociation_iterations,
        score_col="coassociation_score",
        key_columns=score_keys,
    )

    common_depth_level = dihs_common_depth

    by_depth_frames = []
    final_frames = []
    for method_name, table in (
        ("dihs", dihs_by_depth),
        ("single_cut_coassociation", single_cut_by_depth),
    ):
        if table.empty:
            continue
        method_table = table.copy()
        method_table.insert(0, "method", method_name)
        by_depth_frames.append(method_table)
        if common_depth_level is not None:
            final_frames.append(
                method_table[method_table["integration_depth"] == common_depth_level].copy()
            )

    if not centroid_iterations.empty:
        centroid_final = centroid_iterations.copy()
        centroid_final.insert(0, "method", "centroid_distance")
        centroid_final["integration_depth"] = common_depth_level
        final_frames.append(centroid_final)

    if not mahalanobis_iterations.empty:
        mahalanobis_final = mahalanobis_iterations.copy()
        mahalanobis_final.insert(0, "method", "mahalanobis_distance")
        mahalanobis_final["integration_depth"] = common_depth_level
        final_frames.append(mahalanobis_final)

    score_iterations_by_depth = (
        pd.concat(by_depth_frames, ignore_index=True) if by_depth_frames else pd.DataFrame()
    )
    score_iterations = (
        pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame()
    )

    score_iterations_by_depth = _exclude_unknown_neighbor(
        score_iterations_by_depth, unknown_class
    )
    score_iterations = _exclude_unknown_neighbor(score_iterations, unknown_class)

    score_summary, run_rankings, method_summary = _summarize_method_scores(
        score_iterations=score_iterations,
        common_depth_level=common_depth_level,
    )

    return {
        "common_depth_level": common_depth_level,
        "hs_iterations": hs_iterations,
        "coassociation_iterations": coassociation_iterations,
        "dihs_iterations_by_depth": dihs_by_depth,
        "single_cut_iterations_by_depth": single_cut_by_depth,
        "score_iterations": score_iterations,
        "score_iterations_by_depth": score_iterations_by_depth,
        "score_summary": score_summary,
        "run_rankings": run_rankings,
        "method_summary": method_summary,
    }

def _iter_unique_sample_position_sets(
    *,
    n_pool: int,
    sample_size: int,
    n_requested: int,
    rng: np.random.Generator,
):
    """
    Yield unique subsets of row positions from an unknown pool.

    Behavior
    --------
    - If n_requested >= number of possible combinations, this yields every
      possible combination exactly once.
    - Otherwise, it yields n_requested unique random combinations.
    - Combinations are represented as integer positions into unknown_pool,
      not dataframe index labels.
    """
    n_pool = int(n_pool)
    sample_size = int(sample_size)
    n_requested = int(n_requested)

    if sample_size <= 0:
        raise ValueError("sample_size must be > 0.")
    if n_pool <= 0:
        raise ValueError("n_pool must be > 0.")
    if sample_size > n_pool:
        raise ValueError(
            f"Requested sample_size={sample_size}, but the unknown pool only "
            f"contains {n_pool} rows."
        )
    if n_requested <= 0:
        raise ValueError("n_requested must be > 0.")

    n_possible = comb(n_pool, sample_size)
    n_to_run = min(n_requested, n_possible)

    # Exhaustive mode: every possible subset exactly once.
    if n_to_run == n_possible:
        all_combos = list(combinations(range(n_pool), sample_size))
        order = rng.permutation(len(all_combos))

        for idx in order:
            yield all_combos[int(idx)]

        return

    # Random unique mode.
    # For modest combination spaces, enumerate then select without replacement.
    # This avoids rejection-sampling inefficiency.
    if n_possible <= 250_000:
        all_combos = list(combinations(range(n_pool), sample_size))
        chosen = rng.choice(len(all_combos), size=n_to_run, replace=False)

        for idx in chosen:
            yield all_combos[int(idx)]

        return

    # Large combination space: draw random subsets and reject duplicates.
    seen = set()

    while len(seen) < n_to_run:
        positions = tuple(
            sorted(
                rng.choice(
                    n_pool,
                    size=sample_size,
                    replace=False,
                ).tolist()
            )
        )

        if positions in seen:
            continue

        seen.add(positions)
        yield positions

def _run_direct_unknown_sampling_comparison(
    *,
    df: pd.DataFrame,
    unknown_sample: Any,
    true_source_class: Any,
    model_type: str,
    transform_id: int,
    class_column: str,
    sample_size: int,
    n_sampling_iterations: int,
    random_state: int | None,
    max_depth: int,
    exclude_columns,
    major_cols: list[str],
    trace_cols: list[str],
    major_error: float,
    trace_error: float,
    perturbation_seed: int | None,
    verbose: bool,
) -> dict[str, Any]:
    unknown_key = _class_key(unknown_sample)
    true_source_key = _class_key(true_source_class)

    class_keys = df[class_column].apply(_class_key)
    unknown_mask = class_keys == unknown_key

    unknown_pool = df.loc[unknown_mask].copy()
    source_df = df.loc[~unknown_mask].copy()

    if unknown_pool.empty:
        raise ValueError(
            f"Unknown sample '{unknown_sample}' not found in class column '{class_column}'."
        )

    sample_size = int(sample_size)
    n_unknown_points = int(len(unknown_pool))

    if n_unknown_points < sample_size:
        raise ValueError(
            f"Requested sample_size={sample_size}, but the unknown pool only "
            f"contains {n_unknown_points} rows."
        )

    n_possible_combinations = comb(n_unknown_points, sample_size)
    effective_n_sampling_iterations = min(
        int(n_sampling_iterations),
        int(n_possible_combinations),
    )

    sampling_mode = (
        "exhaustive_unique"
        if effective_n_sampling_iterations == n_possible_combinations
        else "random_unique"
    )

    rng = np.random.default_rng(random_state)

    score_rows = []
    score_by_depth_rows = []
    run_id_counter = 0

    if verbose:
        print(
            f"Direct unknown resampling | sample_size={sample_size} | "
            f"unknown_points={n_unknown_points} | "
            f"possible_combinations={n_possible_combinations} | "
            f"requested_iterations={int(n_sampling_iterations)} | "
            f"used_iterations={effective_n_sampling_iterations} | "
            f"mode={sampling_mode} | "
            f"true_source={true_source_class}",
            flush=True,
        )

    unknown_index_values = unknown_pool.index.to_numpy()

    unique_position_sets = _iter_unique_sample_position_sets(
        n_pool=n_unknown_points,
        sample_size=sample_size,
        n_requested=effective_n_sampling_iterations,
        rng=rng,
    )

    progress_every = max(1, effective_n_sampling_iterations // 20)

    for sample_iteration, sampled_positions in enumerate(unique_position_sets):
        if verbose and (
            sample_iteration == 0
            or (sample_iteration + 1) % progress_every == 0
            or (sample_iteration + 1) == effective_n_sampling_iterations
        ):
            print(
                f"[PID {os.getpid()}] "
                f"model={model_type} | "
                f"sample_size={sample_size} | "
                f"iteration {sample_iteration + 1}/{effective_n_sampling_iterations} | "
                f"mode={sampling_mode}",
                flush=True,
            )

        sampled_positions = np.asarray(sampled_positions, dtype=int)
        sampled_indices = unknown_index_values[sampled_positions]

        sampled_unknown = unknown_pool.loc[sampled_indices].copy()
        case_df = pd.concat([source_df, sampled_unknown], ignore_index=False).copy()

        case_result = _run_comparison_ensemble(
            df=case_df,
            unknown_class=unknown_sample,
            class_column=class_column,
            model_type=model_type,
            transform_id=transform_id,
            random_state=random_state,
            n_perturbations=1,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            major_cols=major_cols,
            trace_cols=trace_cols,
            major_error=major_error,
            trace_error=trace_error,
            perturbation_seed=perturbation_seed,
            context_metadata={
                "sample_iteration": sample_iteration,
                "sample_size": sample_size,
                "true_source_class": true_source_class,
                "true_source_key": true_source_key,
                "n_unknown_points": n_unknown_points,
                "n_possible_combinations": n_possible_combinations,
                "requested_sampling_iterations": int(n_sampling_iterations),
                "used_sampling_iterations": effective_n_sampling_iterations,
                "sampling_mode": sampling_mode,
            },
            apply_perturbation=False,
            verbose=False,
        )

        score_final = case_result["score_iterations"].copy()

        # Fallback: if final scores are empty, use the deepest available by-depth table.
        if score_final.empty:
            score_by_depth_fallback = case_result.get(
                "score_iterations_by_depth",
                pd.DataFrame(),
            ).copy()

            if (
                not score_by_depth_fallback.empty
                and "integration_depth" in score_by_depth_fallback.columns
            ):
                deepest_depth = int(score_by_depth_fallback["integration_depth"].max())

                score_final = score_by_depth_fallback[
                    score_by_depth_fallback["integration_depth"] == deepest_depth
                ].copy()

                if verbose and sample_iteration == 0:
                    print(
                        f"Using by-depth fallback | model={model_type} | "
                        f"sample_size={sample_size} | deepest_depth={deepest_depth} | "
                        f"rows={len(score_final)}",
                        flush=True,
                    )

        # Hard diagnostic if both final and fallback are empty.
        if score_final.empty and sample_iteration == 0:
            diagnostic = {}

            for key, value in case_result.items():
                if isinstance(value, pd.DataFrame):
                    diagnostic[key] = {
                        "shape": value.shape,
                        "columns": list(value.columns),
                    }
                else:
                    diagnostic[key] = type(value).__name__

            raise ValueError(
                "No score rows were produced by _run_comparison_ensemble for the first "
                "sampled case.\n"
                f"model_type={model_type}\n"
                f"sample_size={sample_size}\n"
                f"unknown_sample={unknown_sample}\n"
                f"case_df_shape={case_df.shape}\n"
                f"class_counts:\n{case_df[class_column].value_counts().to_string()}\n"
                f"columns={list(case_df.columns)}\n"
                f"case_result diagnostic={diagnostic}"
            )

        if not score_final.empty:
            score_final["run_id"] = run_id_counter
            score_rows.append(score_final)

        score_by_depth = case_result["score_iterations_by_depth"].copy()

        if not score_by_depth.empty:
            score_by_depth["run_id"] = run_id_counter
            score_by_depth_rows.append(score_by_depth)

        run_id_counter += 1

    score_iterations = (
        pd.concat(score_rows, ignore_index=True)
        if score_rows
        else pd.DataFrame()
    )

    score_iterations_by_depth = (
        pd.concat(score_by_depth_rows, ignore_index=True)
        if score_by_depth_rows
        else pd.DataFrame()
    )

    score_iterations = _exclude_unknown_neighbor(score_iterations, unknown_sample)
    score_iterations_by_depth = _exclude_unknown_neighbor(
        score_iterations_by_depth,
        unknown_sample,
    )

    comparison_runs = _extract_top_rankings(score_iterations)

    if comparison_runs.empty:
        raise ValueError(
            "No comparison runs were produced after all sampled cases.\n"
            f"model={model_type}\n"
            f"sample_size={sample_size}\n"
            f"score_iterations_shape={score_iterations.shape}\n"
            f"score_iterations_columns={list(score_iterations.columns)}\n"
            f"score_iterations_by_depth_shape={score_iterations_by_depth.shape}\n"
            f"score_iterations_by_depth_columns={list(score_iterations_by_depth.columns)}"
        )

    comparison_runs["top1_is_true_source"] = (
        comparison_runs["top1_class"].apply(_class_key) == true_source_key
    )

    comparison_runs["n_unknown_points"] = n_unknown_points
    comparison_runs["n_possible_combinations"] = n_possible_combinations
    comparison_runs["requested_sampling_iterations"] = int(n_sampling_iterations)
    comparison_runs["used_sampling_iterations"] = effective_n_sampling_iterations
    comparison_runs["sampling_mode"] = sampling_mode

    method_summary_rows = []

    for method, sub in comparison_runs.groupby("method"):
        method_summary_rows.append(
            {
                "method": method,
                "top1_accuracy": (
                    float(sub["top1_is_true_source"].mean())
                    if not sub.empty
                    else np.nan
                ),
                "margin_median": (
                    float(sub["margin"].median())
                    if sub["margin"].notna().any()
                    else np.nan
                ),
                "margin_mean": (
                    float(sub["margin"].mean())
                    if sub["margin"].notna().any()
                    else np.nan
                ),
                "n_runs": int(len(sub)),
                "n_unknown_points": n_unknown_points,
                "n_possible_combinations": n_possible_combinations,
                "requested_sampling_iterations": int(n_sampling_iterations),
                "used_sampling_iterations": effective_n_sampling_iterations,
                "sampling_mode": sampling_mode,
            }
        )

    return {
        "comparison_runs": comparison_runs,
        "method_summary": pd.DataFrame(method_summary_rows),
        "score_iterations": score_iterations,
        "score_iterations_by_depth": score_iterations_by_depth,
    }

def run_direct_unknown_sample_size_curve(
    *,
    df: pd.DataFrame,
    unknown_sample: Any,
    true_source_class: Any,
    class_column: str = "controlcode",
    model_type: str = "agglomerative",
    transform_type: str = "clr",
    sample_sizes: Iterable[int] = (3, 5, 10, 15, 20, 25, 30),
    n_sampling_iterations: int = 100,
    random_state: int | None = None,
    max_depth: int = 100,
    major_cols: Iterable[str] | None = None,
    trace_cols: Iterable[str] | None = None,
    major_error: float = 0.02,
    trace_error: float = 0.10,
    perturbation_seed: int | None = None,
    exclude_columns=(),
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run direct unknown sample-size benchmarking.

    Important behavior
    ------------------
    n_sampling_iterations is treated as a cap per sample size.

    For each sample size k, the number of possible unique subsets is:

        comb(n_unknown_points, k)

    The number of runs actually used is:

        min(n_sampling_iterations, comb(n_unknown_points, k))

    If all possible combinations fit under the cap, every combination is used
    exactly once. Otherwise, random unique combinations are sampled without
    repeating any subset within that sample size.
    """
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in dataframe.")

    model = str(model_type).strip().lower()
    if model not in {"agglomerative", "kmeans", "gaussian"}:
        raise ValueError(f"Unsupported model_type='{model_type}'.")

    if int(n_sampling_iterations) <= 0:
        raise ValueError("n_sampling_iterations must be > 0.")

    transform_id = _normalize_transform_type(transform_type)

    resolved_major_cols, resolved_trace_cols = _resolve_major_trace_columns(
        df=df,
        major_cols=major_cols,
        trace_cols=trace_cols,
        class_column=class_column,
    )

    unknown_key = _class_key(unknown_sample)
    class_keys = df[class_column].apply(_class_key)
    n_unknown_points = int((class_keys == unknown_key).sum())

    if n_unknown_points <= 0:
        raise ValueError(
            f"Unknown sample '{unknown_sample}' not found in class column '{class_column}'."
        )

    summary_rows = []
    run_rows = []

    for sample_size in sample_sizes:
        sample_size = int(sample_size)

        if sample_size <= 0:
            raise ValueError("All sample sizes must be > 0.")

        if sample_size > n_unknown_points:
            raise ValueError(
                f"Requested sample_size={sample_size}, but unknown sample "
                f"{unknown_sample!r} only has {n_unknown_points} rows."
            )

        n_possible_combinations = comb(n_unknown_points, sample_size)
        effective_n_sampling_iterations = min(
            int(n_sampling_iterations),
            int(n_possible_combinations),
        )

        sampling_mode = (
            "exhaustive_unique"
            if effective_n_sampling_iterations == n_possible_combinations
            else "random_unique"
        )

        if verbose:
            print(
                "Running direct unknown sample-size comparison | "
                f"sample_size={sample_size} | "
                f"unknown_points={n_unknown_points} | "
                f"possible_combinations={n_possible_combinations} | "
                f"requested_iterations={int(n_sampling_iterations)} | "
                f"used_iterations={effective_n_sampling_iterations} | "
                f"mode={sampling_mode}"
            )

        res = _run_direct_unknown_sampling_comparison(
            df=df,
            unknown_sample=unknown_sample,
            true_source_class=true_source_class,
            model_type=model,
            transform_id=transform_id,
            class_column=class_column,
            sample_size=sample_size,
            n_sampling_iterations=effective_n_sampling_iterations,
            random_state=random_state,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            major_cols=resolved_major_cols,
            trace_cols=resolved_trace_cols,
            major_error=major_error,
            trace_error=trace_error,
            perturbation_seed=perturbation_seed,
            verbose=verbose,
        )

        method_summary = res["method_summary"].copy()
        if not method_summary.empty:
            method_summary["sample_size"] = sample_size
            method_summary["n_unknown_points"] = n_unknown_points
            method_summary["n_possible_combinations"] = n_possible_combinations
            method_summary["requested_sampling_iterations"] = int(n_sampling_iterations)
            method_summary["used_sampling_iterations"] = effective_n_sampling_iterations
            method_summary["sampling_mode"] = sampling_mode
            summary_rows.append(method_summary)

        comparison_runs = res["comparison_runs"].copy()
        if not comparison_runs.empty:
            comparison_runs["sample_size"] = sample_size
            comparison_runs["n_unknown_points"] = n_unknown_points
            comparison_runs["n_possible_combinations"] = n_possible_combinations
            comparison_runs["requested_sampling_iterations"] = int(n_sampling_iterations)
            comparison_runs["used_sampling_iterations"] = effective_n_sampling_iterations
            comparison_runs["sampling_mode"] = sampling_mode
            run_rows.append(comparison_runs)

    summary_df = (
        pd.concat(summary_rows, ignore_index=True)
        if summary_rows
        else pd.DataFrame()
    )

    runs_df = (
        pd.concat(run_rows, ignore_index=True)
        if run_rows
        else pd.DataFrame()
    )

    return {
        "method_summary_by_sample_size": summary_df,
        "comparison_runs_by_sample_size": runs_df,
    }

def _run_pseudo_unknown_method_comparison(
    *,
    df: pd.DataFrame,
    model_type: str,
    transform_id: int,
    class_column: str,
    sample_size: int,
    n_sampling_iterations: int,
    n_perturbations: int,
    excluded_classes: Iterable[Any] | None,
    random_state: int | None,
    max_depth: int,
    exclude_columns,
    major_cols: list[str],
    trace_cols: list[str],
    major_error: float,
    trace_error: float,
    perturbation_seed: int | None,
    verbose: bool,
) -> dict[str, Any]:
    excluded_keys = {_class_key(v) for v in (excluded_classes or [])}
    class_counts = df[class_column].value_counts(dropna=False)
    unique_classes = sorted(df[class_column].dropna().unique(), key=_class_sort_key)

    eligible_classes = []
    skipped_rows = []
    for source_class in unique_classes:
        source_key = _class_key(source_class)
        count = int(class_counts.loc[source_class])
        if source_key in excluded_keys:
            skipped_rows.append(
                {
                    "source_class": source_class,
                    "source_class_key": source_key,
                    "count": count,
                    "reason": "excluded",
                }
            )
            continue
        if count <= int(sample_size):
            skipped_rows.append(
                {
                    "source_class": source_class,
                    "source_class_key": source_key,
                    "count": count,
                    "reason": "insufficient_rows_for_positive_case",
                }
            )
            continue
        eligible_classes.append(source_class)

    if not eligible_classes:
        raise ValueError("No eligible classes available for pseudo-unknown comparisons.")

    sampling_rng = np.random.default_rng(random_state)
    tree_by_depth_rows = []
    distance_rows = []
    run_id_counter = 0

    if verbose:
        print(
            f"Pseudo-unknown method comparison | Classes: {len(eligible_classes)} | Sample size: {sample_size} | Sampling iterations per class: {n_sampling_iterations} | No perturbation ensemble inside pseudo-unknown runs"
        )

    for source_class in eligible_classes:
        source_key = _class_key(source_class)
        class_mask = df[class_column].apply(lambda x: _class_key(x) == source_key)
        class_indices = df.index[class_mask].to_numpy()

        if verbose:
            print(f"Pseudo-unknown source class: {source_class} ({len(class_indices)} rows)")

        for sample_iteration in range(int(n_sampling_iterations)):
            sampled_indices = sampling_rng.choice(
                class_indices, size=int(sample_size), replace=False
            )
            sampled_index_set = set(sampled_indices.tolist())

            positive_df = df.copy()
            positive_unknown = _make_pseudo_unknown_label(
                df[class_column].unique(), source_class, sample_iteration, "positive"
            )
            positive_df.loc[list(sampled_index_set), class_column] = positive_unknown

            negative_unknown = _make_pseudo_unknown_label(
                positive_df[class_column].unique(), source_class, sample_iteration, "negative"
            )
            negative_df = positive_df.copy()
            negative_df.loc[list(sampled_index_set), class_column] = negative_unknown
            keep_mask = (~class_mask) | negative_df.index.isin(sampled_index_set)
            negative_df = negative_df.loc[keep_mask].copy()

            for case_name, case_df, case_unknown, source_present in (
                ("positive", positive_df, positive_unknown, True),
                ("negative", negative_df, negative_unknown, False),
            ):
                if verbose:
                    print(
                        f"  sampling iteration {sample_iteration + 1}/{int(n_sampling_iterations)} | case={case_name}"
                    )
                case_result = _run_comparison_ensemble(
                    df=case_df,
                    unknown_class=case_unknown,
                    class_column=class_column,
                    model_type=model_type,
                    transform_id=transform_id,
                    random_state=random_state,
                    n_perturbations=1,
                    max_depth=max_depth,
                    exclude_columns=exclude_columns,
                    major_cols=major_cols,
                    trace_cols=trace_cols,
                    major_error=major_error,
                    trace_error=trace_error,
                    perturbation_seed=perturbation_seed,
                    context_metadata={
                        "source_class": source_class,
                        "source_class_key": source_key,
                        "case": case_name,
                        "sample_iteration": sample_iteration,
                        "sample_size": int(sample_size),
                        "true_source_present": source_present,
                    },
                    apply_perturbation=False,
                    verbose=False,
                )

                tree_scores = case_result["score_iterations_by_depth"].copy()
                if not tree_scores.empty:
                    tree_scores["run_id"] = run_id_counter
                    tree_by_depth_rows.append(tree_scores)

                distance_scores = case_result["score_iterations"].copy()
                if not distance_scores.empty:
                    distance_scores = distance_scores[
                        distance_scores["method"].isin(["centroid_distance", "mahalanobis_distance"])
                    ].copy()
                    if not distance_scores.empty:
                        distance_scores["run_id"] = run_id_counter
                        distance_rows.append(distance_scores)

                run_id_counter += 1

    tree_scores_by_depth = (
        pd.concat(tree_by_depth_rows, ignore_index=True)
        if tree_by_depth_rows
        else pd.DataFrame()
    )
    distance_scores = (
        pd.concat(distance_rows, ignore_index=True) if distance_rows else pd.DataFrame()
    )

    if not tree_scores_by_depth.empty:
        max_depth_per_run = (
            tree_scores_by_depth.groupby("run_id", as_index=False)["integration_depth"].max()
        )
        common_depth_level = int(max_depth_per_run["integration_depth"].min())
        tree_scores_final = tree_scores_by_depth[
            tree_scores_by_depth["integration_depth"] == common_depth_level
        ].copy()
    else:
        common_depth_level = None
        tree_scores_final = pd.DataFrame()

    if not distance_scores.empty:
        distance_scores["integration_depth"] = common_depth_level

    comparison_runs = pd.concat(
        [x for x in (tree_scores_final, distance_scores) if not x.empty],
        ignore_index=True,
    ) if (not tree_scores_final.empty or not distance_scores.empty) else pd.DataFrame()
    comparison_runs = _exclude_self_unknown_neighbors(comparison_runs)
    comparison_runs = _extract_top_rankings(comparison_runs)
    if not comparison_runs.empty:
        comparison_runs["top1_is_true_source"] = (
            comparison_runs["top1_class"].apply(_class_key)
            == comparison_runs["source_class_key"]
        )
        comparison_runs["is_true_positive"] = (
            comparison_runs["true_source_present"] & comparison_runs["top1_is_true_source"]
        )
    skipped_df = pd.DataFrame(skipped_rows)
    eligible_df = pd.DataFrame(
        {
            "source_class": eligible_classes,
            "source_class_key": [_class_key(c) for c in eligible_classes],
            "count": [int(class_counts.loc[c]) for c in eligible_classes],
        }
    )

    method_summary_rows = []
    for method, sub in comparison_runs.groupby("method", as_index=False):
        positive = sub[sub["case"] == "positive"].copy()
        negative = sub[sub["case"] == "negative"].copy()
        valid_auc = sub[sub["margin"].notna()].copy()
        if valid_auc["case"].nunique() == 2:
            y_true = (valid_auc["case"] == "positive").astype(int)
            margin_auroc = float(roc_auc_score(y_true, valid_auc["margin"]))
        else:
            margin_auroc = np.nan

        method_summary_rows.append(
            {
                "method": method,
                "positive_top1_accuracy": (
                    float(positive["top1_is_true_source"].mean()) if not positive.empty else np.nan
                ),
                "positive_margin_median": (
                    float(positive["margin"].median()) if not positive.empty else np.nan
                ),
                "negative_margin_median": (
                    float(negative["margin"].median()) if not negative.empty else np.nan
                ),
                "margin_median_gap": (
                    float(positive["margin"].median() - negative["margin"].median())
                    if (not positive.empty and not negative.empty)
                    else np.nan
                ),
                "margin_auroc": margin_auroc,
                "n_positive_runs": int(len(positive)),
                "n_negative_runs": int(len(negative)),
            }
        )

    return {
        "comparison_runs": comparison_runs,
        "method_summary": pd.DataFrame(method_summary_rows),
        "eligible_classes": eligible_df,
        "skipped_classes": skipped_df,
    }


def run_method_comparison(
    *,
    df: pd.DataFrame,
    unknown_sample: Any | None = None,
    class_column: str = "controlcode",
    model_type: str = "agglomerative",
    transform_type: str = "clr",
    random_state: int | None = None,
    n_perturbations: int = 100,
    major_cols: Iterable[str] | None = None,
    trace_cols: Iterable[str] | None = None,
    major_error: float = 0.02,
    trace_error: float = 0.10,
    perturbation_seed: int | None = None,
    run_benchmark: bool = True,
    run_pseudo_unknown: bool = True,
    pseudo_unknown_sample_size: int = 17,
    pseudo_unknown_iterations: int = 10,
    excluded_classes: Iterable[Any] | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    plot_everything: bool = False,
    write_files: bool = False,
    output_dir: str = "./Results_method_comparison",
    plot_output_dir: str | None = None,
    verbose: bool = True,
):
    """
    Compare DIHS against compact Stage-1 baselines on the same transformed data
    and tree ensemble.

    Implemented methods:
    - DIHS
    - Centroid distance in transformed space
    - Mahalanobis distance in transformed space
    - Single-cut co-association at the common depth
    - Hierarchical persistence via average co-association across depths
    """
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in dataframe.")
    if int(n_perturbations) <= 0:
        raise ValueError("n_perturbations must be > 0.")
    if int(pseudo_unknown_sample_size) <= 0:
        raise ValueError("pseudo_unknown_sample_size must be > 0.")
    if int(pseudo_unknown_iterations) <= 0:
        raise ValueError("pseudo_unknown_iterations must be > 0.")

    model = str(model_type).strip().lower()
    if model not in {"agglomerative", "kmeans", "gaussian"}:
        raise ValueError(f"Unsupported model_type='{model_type}'.")

    transform_id = _normalize_transform_type(transform_type)
    resolved_major_cols, resolved_trace_cols = _resolve_major_trace_columns(
        df=df,
        major_cols=major_cols,
        trace_cols=trace_cols,
        class_column=class_column,
    )

    benchmark_result = None
    if run_benchmark:
        unknown_class = _resolve_unknown_class(
            df=df,
            unknown_sample=unknown_sample,
            class_column=class_column,
        )
        _log(
            verbose,
            f"Running benchmark method comparison | model={model} | transform={transform_type} | perturbations={n_perturbations}",
        )
        benchmark_result = _run_comparison_ensemble(
            df=df,
            unknown_class=unknown_class,
            class_column=class_column,
            model_type=model,
            transform_id=transform_id,
            random_state=random_state,
            n_perturbations=n_perturbations,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            major_cols=resolved_major_cols,
            trace_cols=resolved_trace_cols,
            major_error=major_error,
            trace_error=trace_error,
            perturbation_seed=perturbation_seed,
            apply_perturbation=True,
            verbose=verbose,
            progress_label="Benchmark ensemble",
        )

    pseudo_unknown_result = None
    if run_pseudo_unknown:
        _log(
            verbose,
            f"Running pseudo-unknown comparison | model={model} | transform={transform_type} | sample_size={pseudo_unknown_sample_size} | sampling_iterations={pseudo_unknown_iterations} | no perturbation ensemble",
        )
        pseudo_unknown_result = _run_pseudo_unknown_method_comparison(
            df=df,
            model_type=model,
            transform_id=transform_id,
            class_column=class_column,
            sample_size=pseudo_unknown_sample_size,
            n_sampling_iterations=pseudo_unknown_iterations,
            n_perturbations=n_perturbations,
            excluded_classes=excluded_classes,
            random_state=random_state,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            major_cols=resolved_major_cols,
            trace_cols=resolved_trace_cols,
            major_error=major_error,
            trace_error=trace_error,
            perturbation_seed=perturbation_seed,
            verbose=verbose,
        )

    benchmark_summary = (
        None if benchmark_result is None else benchmark_result["method_summary"]
    )
    pseudo_summary = (
        None if pseudo_unknown_result is None else pseudo_unknown_result["method_summary"]
    )
    main_comparison_table = _build_main_comparison_table(
        benchmark_summary=benchmark_summary,
        pseudo_summary=pseudo_summary,
    )

    artifacts = {}
    if write_files:
        os.makedirs(output_dir, exist_ok=True)
        main_table_path = os.path.join(output_dir, "method_comparison_main_table.csv")
        main_comparison_table.to_csv(main_table_path, index=False)
        artifacts["main_comparison_table_csv"] = main_table_path

        if benchmark_result is not None:
            benchmark_summary_path = os.path.join(
                output_dir, "method_comparison_benchmark_summary.csv"
            )
            benchmark_scores_path = os.path.join(
                output_dir, "method_comparison_benchmark_scores.csv"
            )
            benchmark_runs_path = os.path.join(
                output_dir, "method_comparison_benchmark_run_rankings.csv"
            )
            benchmark_result["method_summary"].to_csv(benchmark_summary_path, index=False)
            benchmark_result["score_summary"].to_csv(benchmark_scores_path, index=False)
            benchmark_result["run_rankings"].to_csv(benchmark_runs_path, index=False)
            artifacts.update(
                {
                    "benchmark_summary_csv": benchmark_summary_path,
                    "benchmark_scores_csv": benchmark_scores_path,
                    "benchmark_run_rankings_csv": benchmark_runs_path,
                }
            )

        if pseudo_unknown_result is not None:
            pseudo_runs_path = os.path.join(
                output_dir, "method_comparison_pseudo_unknown_runs.csv"
            )
            pseudo_summary_path = os.path.join(
                output_dir, "method_comparison_pseudo_unknown_summary.csv"
            )
            eligible_path = os.path.join(
                output_dir, "method_comparison_pseudo_unknown_eligible_classes.csv"
            )
            skipped_path = os.path.join(
                output_dir, "method_comparison_pseudo_unknown_skipped_classes.csv"
            )
            pseudo_unknown_result["comparison_runs"].to_csv(pseudo_runs_path, index=False)
            pseudo_unknown_result["method_summary"].to_csv(pseudo_summary_path, index=False)
            pseudo_unknown_result["eligible_classes"].to_csv(eligible_path, index=False)
            pseudo_unknown_result["skipped_classes"].to_csv(skipped_path, index=False)
            artifacts.update(
                {
                    "pseudo_unknown_runs_csv": pseudo_runs_path,
                    "pseudo_unknown_summary_csv": pseudo_summary_path,
                    "pseudo_unknown_eligible_classes_csv": eligible_path,
                    "pseudo_unknown_skipped_classes_csv": skipped_path,
                }
            )

    if plot_everything and pseudo_unknown_result is not None:
        if plot_output_dir is None:
            plot_output_dir = os.path.join(output_dir, "Plots")
        margin_plot_path = (
            os.path.join(plot_output_dir, "method_comparison_pseudo_unknown_margins.svg")
            if write_files
            else None
        )
        plot_pseudo_unknown_margin_comparison(
            comparison_runs=pseudo_unknown_result["comparison_runs"],
            output_path=margin_plot_path,
        )
        artifacts["pseudo_unknown_margin_plot_path"] = margin_plot_path

    return {
        "benchmark": benchmark_result,
        "pseudo_unknown": pseudo_unknown_result,
        "main_comparison_table": main_comparison_table,
        "artifacts": artifacts,
    }
