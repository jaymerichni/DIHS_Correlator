import os
import warnings
from glob import glob
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import comb

from DIHS_Correlator.core.transforms import BASE_TRANSFORMATIONS
from DIHS_Correlator.viz.hs_curves import plot_hs_curves
from DIHS_Correlator.viz.pairwise import plot_pairwise_matrix
from DIHS_Correlator.viz.pseudo_unknown import (
    plot_margin_comparison,
    plot_margin_histogram,
    plot_perturbative_calibration_overlay,
)

from DIHS_Correlator.workflows.pseudo_unknown import (
    run_pseudo_unknown_experiments,
)
from DIHS_Correlator.workflows.single_run import CorrelationRunner

SUPPORTED_MODELS = ("agglomerative", "kmeans", "gaussian")
TRANSFORM_NAME_TO_ID = {v: k for k, v in BASE_TRANSFORMATIONS.items()}
DEFAULT_MAJOR_COLS = ["SIO2N","TIO2N","AL2O3N","FE2O3TN","CAON","MGON","MNON","NA2ON","K2ON","P2O5N"]
DEFAULT_TRACE_COLS = ["NbN", "ZrN", "LaN", "CeN", "SrN", "BaN", "RbN"]

def _log(verbose: bool, message: str):
    if verbose:
        print(message)

def _print_progress(current: int, total: int, width: int = 30):
    if total <= 0:
        return
    ratio = min(max(current / float(total), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()

def _normalize_transform_type(transform_type: str) -> int:
    if not isinstance(transform_type, str):
        raise ValueError(
            "transform_type must be a string name: 'none', 'ilr', 'clr', or 'scaled'."
        )
    key = str(transform_type).strip().lower()
    if key not in TRANSFORM_NAME_TO_ID:
        raise ValueError(f"Unsupported transform_type='{transform_type}'.")
    return TRANSFORM_NAME_TO_ID[key]

def _resolve_unknown_class(
    df: pd.DataFrame, unknown_sample: Any, class_column: str = "controlcode"
) -> Any:
    if unknown_sample is None:
        return 0
    if class_column not in df.columns:
        raise ValueError(f"Input dataframe must contain class column '{class_column}'.")

    control_values = df[class_column]
    if (control_values == unknown_sample).any():
        return unknown_sample

    if isinstance(unknown_sample, str):
        for caster in (int, float):
            try:
                parsed = caster(unknown_sample)
            except Exception:
                continue
            if (control_values == parsed).any():
                return parsed

    raise ValueError(
        f"Unknown sample '{unknown_sample}' was not found in class column '{class_column}'."
    )

def _prepare_working_df(df: pd.DataFrame, class_column: str) -> pd.DataFrame:
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in dataframe.")
    return df.copy()


def _class_key(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _class_match_mask(values: pd.Series, target: Any) -> pd.Series:
    target_key = _class_key(target)
    return values.apply(lambda value: _class_key(value) == target_key)


def _finite_float_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]

def _run_single_model(
    *,
    df: pd.DataFrame,
    model_type: str,
    transform_type: str = "clr",
    unknown_sample: Any = 0,
    class_column: str = "controlcode",
    random_state: int | None = None,
    compute_pairwise: bool = True,
    plot_everything: bool = False,
    write_files: bool = False,
    output_dir: str = "./Results",
    plot_output_dir: str | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    
    model = model_type.lower().strip()
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_type='{model_type}'.")
    transform_id = _normalize_transform_type(transform_type)
    work_df = _prepare_working_df(df, class_column=class_column)
    unknown_class = _resolve_unknown_class(work_df, unknown_sample, class_column=class_column)

    runner = CorrelationRunner(
        base_output_dir=output_dir,
        save_trees=write_files,
        save_cluster_data=save_cluster_data,
        save_untransformed=save_untransformed,
    )
    exclude_set = set(exclude_columns) | {class_column}
    runner.set_feature_columns(work_df, exclude=tuple(exclude_set), verbose=verbose)

    effective_random_state = random_state if model in ("kmeans", "gaussian") else None
    run = runner.run_combination(
        data=work_df,
        transform_type=transform_id,
        model_type=model,
        random_state=effective_random_state,
        unknown_class=unknown_class,
        class_column=class_column,
        compute_pairwise=compute_pairwise,
        write_outputs=write_files,
        max_depth=max_depth,
    )

    artifacts: dict[str, Any] = {}
    if write_files and compute_pairwise:
        pairwise_out = os.path.join(
            output_dir,
            "Trees",
            f"{run['transform_name']}_{model}",
            "PairwiseMatrices",
        )
        artifacts["pairwise_depth_paths"] = runner.save_pairwise_matrices_all_depths(
            output_dir=pairwise_out
        )
        artifacts["pairwise_total_csv"] = runner.save_pairwise_total_matrix(
            output_dir=pairwise_out
        )

    if plot_everything:
        if plot_output_dir is None:
            plot_output_dir = os.path.join(output_dir, "Plots")
        os.makedirs(plot_output_dir, exist_ok=True)

        hs_curve_path = os.path.join(
            plot_output_dir, f"hs_curve_{run['transform_name']}_{model}.svg"
        )
        plot_hs_curves(
            df=run["metrics_per_depth"],
            value_col="harmonic_score",
            with_shade=False,
            output_path=hs_curve_path,
            title=f"HS vs depth | {run['transform_name']} + {model}",
            max_depth=max_depth,
            unknown_class=unknown_class,
        )
        artifacts["hs_curve_path"] = hs_curve_path

        if compute_pairwise and runner.get_pairwise_total_matrix() is not None:
            pairwise_plot_path = os.path.join(
                plot_output_dir, f"pairwise_total_{run['transform_name']}_{model}.svg"
            )
            plot_pairwise_matrix(
                matrix=runner.get_pairwise_total_matrix(),
                title=f"Pairwise DIHS | {run['transform_name']} + {model}",
                output_path=pairwise_plot_path,
                unknown_class=unknown_class,
            )
            artifacts["pairwise_total_plot_path"] = pairwise_plot_path

    return {
        "hs_per_depth": run["metrics_per_depth"],
        "dihs_total": run["total_metrics"],
        "pairwise_total_matrix": runner.get_pairwise_total_matrix()
        if compute_pairwise
        else None,
        "pairwise_per_depth_matrices": runner.last_pair_depth_matrices
        if compute_pairwise
        else None,
        "transform_name": run["transform_name"],
        "model_type": model,
        "unknown_class": unknown_class,
        "artifacts": artifacts,
    }


def _resolve_major_trace_columns(
    df: pd.DataFrame,
    major_cols: list[str] | None,
    trace_cols: list[str] | None,
    class_column: str = "controlcode",
):
    numeric_cols = set(
        df.select_dtypes(include="number").columns.drop(class_column, errors="ignore")
    )

    def _resolve_subset(
        subset_name: str,
        requested_cols: list[str] | None,
        default_cols: list[str],
    ) -> list[str]:
        if requested_cols is None:
            present_defaults = [c for c in default_cols if c in df.columns]
            return [c for c in present_defaults if c in numeric_cols]

        requested = list(requested_cols)
        if len(requested) == 0:
            return []

        missing = [c for c in requested if c not in df.columns]
        non_numeric = [c for c in requested if c in df.columns and c not in numeric_cols]
        resolved = [c for c in requested if c in numeric_cols]

        if not resolved:
            details = []
            if missing:
                details.append(f"missing columns: {missing}")
            if non_numeric:
                details.append(f"non-numeric columns: {non_numeric}")
            detail_text = " ".join(details) if details else "No matching numeric columns were found."
            raise ValueError(
                f"No valid {subset_name} columns were resolved from the explicit list. {detail_text}"
            )

        if missing or non_numeric:
            details = []
            if missing:
                details.append(f"missing columns: {missing}")
            if non_numeric:
                details.append(f"non-numeric columns: {non_numeric}")
            warnings.warn(
                f"Ignoring unresolved {subset_name} columns. Using {resolved}. "
                + " ".join(details),
                stacklevel=3,
            )

        return resolved

    resolved_major = _resolve_subset("major", major_cols, DEFAULT_MAJOR_COLS)
    resolved_trace = _resolve_subset("trace", trace_cols, DEFAULT_TRACE_COLS)

    if (
        (major_cols is None or trace_cols is None)
        and not resolved_major
        and not resolved_trace
    ):
        warnings.warn(
            "No perturbation columns were resolved. "
            "The perturbative run will proceed without feature perturbations. "
            "Pass major_cols and/or trace_cols explicitly if your dataset uses different column names.",
            stacklevel=3,
        )

    return resolved_major, resolved_trace


def _exclude_unknown_neighbor(
    dihs_iterations: pd.DataFrame, unknown_class: Any
) -> pd.DataFrame:
    if dihs_iterations.empty:
        return dihs_iterations
    return dihs_iterations[
        dihs_iterations["neighbor_unit"].astype(str) != str(unknown_class)
    ].copy()


def _compute_top1_stats(
    dihs_iterations: pd.DataFrame, unknown_class: Any
):
    dihs_iterations = _exclude_unknown_neighbor(dihs_iterations, unknown_class)
    if dihs_iterations.empty:
        return pd.DataFrame(
            columns=["neighbor_unit", "wins", "top1_fraction", "n_iterations"]
        )
    candidates = (
        dihs_iterations["neighbor_unit"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    idx = dihs_iterations.groupby("iteration")["total_product"].idxmax()
    winners_counts = (
        dihs_iterations.loc[idx, ["iteration", "neighbor_unit"]]
        .assign(neighbor_unit=lambda x: x["neighbor_unit"].astype(str))
        .groupby("neighbor_unit")
        .size()
        .reset_index(name="wins")
    )
    winners = pd.DataFrame({"neighbor_unit": candidates}).merge(
        winners_counts, on="neighbor_unit", how="left"
    )
    winners["wins"] = winners["wins"].fillna(0).astype(int)
    n_iter = dihs_iterations["iteration"].nunique()
    winners["n_iterations"] = n_iter
    winners["top1_fraction"] = winners["wins"] / float(n_iter)
    winners = winners.sort_values("top1_fraction", ascending=False).reset_index(drop=True)
    return winners


def _compute_margin_stats(dihs_iterations: pd.DataFrame, unknown_class: Any):
    dihs_iterations = _exclude_unknown_neighbor(dihs_iterations, unknown_class)
    rows = []
    for it, sub in dihs_iterations.groupby("iteration"):
        scores = sub["total_product"].sort_values(ascending=False).to_numpy()
        if scores.size == 0:
            continue
        top1 = float(scores[0])
        top2 = float(scores[1]) if scores.size > 1 else np.nan
        margin = top1 - top2 if scores.size > 1 else np.nan
        rows.append({"iteration": it, "top1": top1, "top2": top2, "dihs_margin": margin})
    margin_df = pd.DataFrame(rows)
    if margin_df.empty:
        summary = pd.DataFrame(
            [{"margin_mean": np.nan, "margin_std": np.nan, "margin_median": np.nan}]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "margin_mean": margin_df["dihs_margin"].mean(),
                    "margin_std": margin_df["dihs_margin"].std(),
                    "margin_median": margin_df["dihs_margin"].median(),
                }
            ]
        )
    return margin_df, summary


def _recompute_dihs_iterations_on_common_depth(
    hs_iterations: pd.DataFrame,
):
    if hs_iterations.empty:
        return pd.DataFrame(), None

    max_depth_per_iter = (
        hs_iterations.groupby("iteration", as_index=False)["depth_level"].max()
    )
    if max_depth_per_iter.empty:
        return pd.DataFrame(), None

    common_depth_level = int(max_depth_per_iter["depth_level"].min())
    depth_span = float(common_depth_level + 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

    hs_common = hs_iterations[hs_iterations["depth_level"] <= common_depth_level].copy()
    if hs_common.empty:
        return pd.DataFrame(), common_depth_level

    keys = ["unknown_class", "transform", "model", "iteration", "neighbor_unit"]
    dihs_iterations = (
        hs_common.groupby(keys, as_index=False)
        .agg(hs_sum=("harmonic_score", "sum"))
        .assign(total_product=lambda x: x["hs_sum"] / depth_span)
        .drop(columns=["hs_sum"])
    )
    return dihs_iterations, common_depth_level


def _infer_pseudo_unknown_common_depth(output_dir: str) -> int | None:
    summary_path = os.path.join(output_dir, "pseudo_unknown_threshold_summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        if (
            not summary.empty
            and "common_depth_level" in summary.columns
            and summary["common_depth_level"].notna().any()
        ):
            return int(summary["common_depth_level"].dropna().iloc[0])

    by_depth_path = os.path.join(output_dir, "pseudo_unknown_run_results_by_depth.csv")
    if os.path.exists(by_depth_path):
        runs = pd.read_csv(by_depth_path)
        if not runs.empty and "integration_depth" in runs.columns:
            valid = runs["integration_depth"].dropna()
            if not valid.empty:
                return int(valid.max())
    return None


def _load_pseudo_unknown_margin_inputs(
    output_dir: str,
    integration_depth: int,
    target_precision: float,
):
    runs_by_depth_path = os.path.join(output_dir, "pseudo_unknown_run_results_by_depth.csv")
    runs_path = os.path.join(output_dir, "pseudo_unknown_runs.csv")
    thresholds_by_depth_path = os.path.join(
        output_dir, "pseudo_unknown_thresholds_by_target_precision_by_depth.csv"
    )
    thresholds_path = os.path.join(
        output_dir, "pseudo_unknown_thresholds_by_target_precision.csv"
    )
    threshold_summary_path = os.path.join(output_dir, "pseudo_unknown_threshold_summary.csv")

    if os.path.exists(runs_by_depth_path):
        results_df = pd.read_csv(runs_by_depth_path)
        if "integration_depth" not in results_df.columns:
            raise ValueError(
                f"Expected 'integration_depth' column in '{runs_by_depth_path}'."
            )
        results_df = results_df[
            results_df["integration_depth"] == int(integration_depth)
        ].copy()
    elif os.path.exists(runs_path):
        results_df = pd.read_csv(runs_path)
    else:
        raise FileNotFoundError(
            f"No pseudo-unknown run results found in '{output_dir}'."
        )

    if results_df.empty:
        raise ValueError(
            f"No pseudo-unknown runs found for integration_depth={integration_depth} in '{output_dir}'."
        )

    threshold = np.nan
    if os.path.exists(thresholds_by_depth_path):
        thresholds_df = pd.read_csv(thresholds_by_depth_path)
        if not thresholds_df.empty:
            depth_mask = thresholds_df["integration_depth"] == int(integration_depth)
            precision_mask = np.isclose(
                thresholds_df["target_precision"].astype(float),
                float(target_precision),
            )
            matched = thresholds_df[depth_mask & precision_mask]
            if not matched.empty:
                threshold = float(matched["resolvedness_threshold"].iloc[0])
    elif os.path.exists(thresholds_path):
        thresholds_df = pd.read_csv(thresholds_path)
        if not thresholds_df.empty:
            precision_mask = np.isclose(
                thresholds_df["target_precision"].astype(float),
                float(target_precision),
            )
            matched = thresholds_df[precision_mask]
            if not matched.empty:
                threshold = float(matched["resolvedness_threshold"].iloc[0])
    elif os.path.exists(threshold_summary_path):
        summary_df = pd.read_csv(threshold_summary_path)
        if not summary_df.empty and "resolvedness_threshold" in summary_df.columns:
            threshold = float(summary_df["resolvedness_threshold"].iloc[0])

    return results_df, threshold


def _read_perturbative_metric_frames(output_dir: str):
    iter_dirs = sorted(
        path
        for path in glob(os.path.join(output_dir, "iter_*"))
        if os.path.isdir(path)
    )
    if not iter_dirs:
        raise FileNotFoundError(
            f"No iteration folders like 'iter_000' were found in '{output_dir}'. "
            "Pass a single-model perturbative output folder."
        )

    frames = []
    max_depths = []
    for iter_dir in iter_dirs:
        matches = sorted(glob(os.path.join(iter_dir, "Trees", "*", "metrics_*.csv")))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No metrics CSV found inside '{iter_dir}'."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple metrics CSV files found inside '{iter_dir}'. "
                "Pass a single-model perturbative output folder."
            )
        metrics_path = matches[0]
        metrics_df = pd.read_csv(metrics_path)
        required = {"depth_level", "neighbor_unit", "harmonic_score", "unknown_class"}
        missing = required.difference(metrics_df.columns)
        if missing:
            raise ValueError(
                f"Metrics file '{metrics_path}' is missing columns: {sorted(missing)}"
            )
        if metrics_df.empty:
            continue
        max_depths.append(int(metrics_df["depth_level"].max()))
        frames.append((iter_dir, metrics_df))

    if not frames:
        raise ValueError(f"No valid perturbative metrics were found in '{output_dir}'.")

    return frames, int(min(max_depths))


def _compute_margin_from_hs_metrics(
    metrics_df: pd.DataFrame,
    integration_depth: int,
):
    max_depth = int(metrics_df["depth_level"].max())
    if max_depth < int(integration_depth):
        return None

    unknown_class = metrics_df["unknown_class"].iloc[0]
    hs_cut = metrics_df[metrics_df["depth_level"] <= int(integration_depth)].copy()
    if hs_cut.empty:
        return None

    dihs_df = (
        hs_cut.groupby("neighbor_unit", as_index=False)
        .agg(hs_sum=("harmonic_score", "sum"))
        .assign(total_product=lambda x: x["hs_sum"] / float(int(integration_depth) + 1))
        .drop(columns=["hs_sum"])
    )
    dihs_df = _exclude_unknown_neighbor(dihs_df, unknown_class)
    if dihs_df.empty:
        return None

    scores = dihs_df["total_product"].sort_values(ascending=False).to_numpy(dtype=float)
    if scores.size == 0:
        return None

    top1 = float(scores[0])
    top2 = float(scores[1]) if scores.size > 1 else np.nan
    margin = top1 - top2 if scores.size > 1 else np.nan
    return {
        "unknown_class": unknown_class,
        "top1": top1,
        "top2": top2,
        "dihs_margin": margin,
    }


def _load_perturbative_margin_inputs(
    output_dir: str,
    integration_depth: int,
):
    frames, common_depth_level = _read_perturbative_metric_frames(output_dir)
    if int(integration_depth) > common_depth_level:
        raise ValueError(
            f"Requested integration_depth={integration_depth} exceeds the perturbative "
            f"common depth {common_depth_level} in '{output_dir}'."
        )

    rows = []
    for iteration_index, (iter_dir, metrics_df) in enumerate(frames):
        row = _compute_margin_from_hs_metrics(metrics_df, integration_depth=integration_depth)
        if row is None:
            continue
        row["iteration"] = iteration_index
        row["iteration_dir"] = iter_dir
        rows.append(row)

    margins_df = pd.DataFrame(rows)
    return margins_df, common_depth_level


def _compute_precision_at_threshold(
    results_df: pd.DataFrame,
    threshold: float,
):
    valid = results_df[results_df["dihs_margin"].notna()].copy()
    if valid.empty:
        return {
            "precision": np.nan,
            "coverage": np.nan,
            "n_runs_above_threshold": 0,
        }

    above = valid[valid["dihs_margin"] >= float(threshold)].copy()
    if above.empty:
        return {
            "precision": np.nan,
            "coverage": 0.0,
            "n_runs_above_threshold": 0,
        }

    return {
        "precision": float(above["is_true_positive"].astype(bool).mean()),
        "coverage": float(len(above) / float(len(valid))),
        "n_runs_above_threshold": int(len(above)),
    }


def _prepare_margin_plot_data_from_outputs(
    *,
    pseudo_unknown_output_dir: str,
    perturbative_output_dir: str | None,
    integration_depth: int | None,
    target_precision: float,
    verbose: bool,
):
    pseudo_common_depth = _infer_pseudo_unknown_common_depth(pseudo_unknown_output_dir)
    if integration_depth is None:
        chosen_depth = pseudo_common_depth
        perturbative_common_depth = None
        if perturbative_output_dir is not None:
            _, perturbative_common_depth = _read_perturbative_metric_frames(
                perturbative_output_dir
            )
            if chosen_depth is None:
                chosen_depth = perturbative_common_depth
            else:
                chosen_depth = min(chosen_depth, perturbative_common_depth)
        if chosen_depth is None:
            raise ValueError(
                "Could not infer an integration depth from the saved outputs. "
                "Pass integration_depth explicitly."
            )
    else:
        chosen_depth = int(integration_depth)
        perturbative_common_depth = None

    _log(
        verbose,
        f"Loading saved margin data at integration_depth={chosen_depth}",
    )
    pseudo_results, threshold = _load_pseudo_unknown_margin_inputs(
        output_dir=pseudo_unknown_output_dir,
        integration_depth=chosen_depth,
        target_precision=target_precision,
    )

    perturbative_margins_df = pd.DataFrame()
    if perturbative_output_dir is not None:
        perturbative_margins_df, perturbative_common_depth = _load_perturbative_margin_inputs(
            output_dir=perturbative_output_dir,
            integration_depth=chosen_depth,
        )
        _log(
            verbose,
            f"Loaded perturbative margins from {len(perturbative_margins_df)} iterations "
            f"(common depth {perturbative_common_depth}).",
        )

    return {
        "pseudo_results": pseudo_results,
        "threshold": threshold,
        "integration_depth": chosen_depth,
        "pseudo_common_depth_level": pseudo_common_depth,
        "perturbative_margins": perturbative_margins_df,
        "perturbative_common_depth_level": perturbative_common_depth,
    }


def _load_pseudo_unknown_calibration_tables(
    output_dir: str,
    integration_depth: int,
):
    threshold_curve_by_depth_path = os.path.join(
        output_dir, "pseudo_unknown_threshold_curve_by_depth.csv"
    )
    threshold_curve_path = os.path.join(output_dir, "pseudo_unknown_threshold_curve.csv")
    thresholds_by_precision_by_depth_path = os.path.join(
        output_dir, "pseudo_unknown_thresholds_by_target_precision_by_depth.csv"
    )
    thresholds_by_precision_path = os.path.join(
        output_dir, "pseudo_unknown_thresholds_by_target_precision.csv"
    )

    if os.path.exists(threshold_curve_by_depth_path):
        threshold_curve = pd.read_csv(threshold_curve_by_depth_path)
        if "integration_depth" in threshold_curve.columns:
            threshold_curve = threshold_curve[
                threshold_curve["integration_depth"] == int(integration_depth)
            ].copy()
    elif os.path.exists(threshold_curve_path):
        threshold_curve = pd.read_csv(threshold_curve_path)
    else:
        raise FileNotFoundError(
            f"No pseudo-unknown threshold curve CSV found in '{output_dir}'."
        )

    if os.path.exists(thresholds_by_precision_by_depth_path):
        thresholds_by_precision = pd.read_csv(thresholds_by_precision_by_depth_path)
        if "integration_depth" in thresholds_by_precision.columns:
            thresholds_by_precision = thresholds_by_precision[
                thresholds_by_precision["integration_depth"] == int(integration_depth)
            ].copy()
    elif os.path.exists(thresholds_by_precision_path):
        thresholds_by_precision = pd.read_csv(thresholds_by_precision_path)
    else:
        raise FileNotFoundError(
            f"No pseudo-unknown precision-threshold CSV found in '{output_dir}'."
        )

    return threshold_curve, thresholds_by_precision


def _resolve_display_threshold(
    *,
    result: dict[str, Any],
    threshold_mode: str,
    target_precision: float,
):
    mode = str(threshold_mode).strip().lower()
    if mode == "target_precision":
        threshold_value = result["threshold"]
        threshold_pct = int(round(100.0 * float(target_precision)))
        threshold_label = (
            None
            if not np.isfinite(threshold_value)
            else f"{threshold_pct}% threshold = {threshold_value:.3f}"
        )
        return {
            "threshold_mode": mode,
            "display_threshold": threshold_value,
            "threshold_label": threshold_label,
            "threshold_precision": float(target_precision) if np.isfinite(threshold_value) else np.nan,
            "threshold_coverage": np.nan,
            "n_runs_above_threshold": np.nan,
            "perturbative_mean_margin": np.nan,
        }

    if mode == "perturbative_mean":
        perturb_df = result["perturbative_margins"]
        if perturb_df.empty:
            raise ValueError(
                "threshold_mode='perturbative_mean' requires a perturbative output folder."
            )
        threshold_value = float(perturb_df["dihs_margin"].mean())
        stats = _compute_precision_at_threshold(
            results_df=result["pseudo_results"],
            threshold=threshold_value,
        )
        if np.isfinite(stats["precision"]):
            threshold_label = (
                f"{100 * stats['precision']:.0f}% confidence threshold | ΔDIHS perturbative run mean = "
                f"{threshold_value:.2f}"
            )
        else:
            threshold_label = f"Perturbative mean = {threshold_value:.2f}"

        return {
            "threshold_mode": mode,
            "display_threshold": threshold_value,
            "threshold_label": threshold_label,
            "threshold_precision": stats["precision"],
            "threshold_coverage": stats["coverage"],
            "n_runs_above_threshold": stats["n_runs_above_threshold"],
            "perturbative_mean_margin": threshold_value,
        }

    raise ValueError(
        "threshold_mode must be either 'target_precision' or 'perturbative_mean'."
    )


def _build_perturbative_calibration_outputs(
    *,
    pseudo_results: pd.DataFrame,
    perturbative_margins: pd.DataFrame,
    thresholds_by_target_precision: pd.DataFrame,
    target_precisions: list[float],
    integration_depth: int,
):
    if perturbative_margins.empty:
        raise ValueError("No perturbative margins were found to calibrate.")

    thresholds_map = {}
    if not thresholds_by_target_precision.empty:
        for _, row in thresholds_by_target_precision.iterrows():
            target = float(row["target_precision"])
            tau = row.get("resolvedness_threshold", np.nan)
            thresholds_map[target] = float(tau) if pd.notna(tau) else np.nan

    target_precisions = sorted({float(x) for x in target_precisions}, reverse=True)

    calibrated_rows = []
    for _, row in perturbative_margins.iterrows():
        margin = float(row["dihs_margin"])
        stats = _compute_precision_at_threshold(
            results_df=pseudo_results,
            threshold=margin,
        )
        out = row.to_dict()
        out["integration_depth"] = int(integration_depth)
        out["calibrated_precision"] = stats["precision"]
        out["calibrated_coverage"] = stats["coverage"]
        out["n_pseudo_runs_above_margin"] = stats["n_runs_above_threshold"]

        achieved = []
        for target in target_precisions:
            tau = thresholds_map.get(target, np.nan)
            col = f"above_{int(round(100.0 * target))}"
            out[col] = bool(np.isfinite(tau) and margin >= tau)
            if out[col]:
                achieved.append(target)

        out["highest_precision_regime"] = max(achieved) if achieved else np.nan
        calibrated_rows.append(out)

    calibrated_runs = pd.DataFrame(calibrated_rows)
    margins = calibrated_runs["dihs_margin"].to_numpy(dtype=float)
    calibrated_precision = calibrated_runs["calibrated_precision"].to_numpy(dtype=float)
    calibrated_coverage = calibrated_runs["calibrated_coverage"].to_numpy(dtype=float)
    finite_precision = _finite_float_values(calibrated_precision)
    finite_coverage = _finite_float_values(calibrated_coverage)

    mean_margin = float(np.mean(margins))
    median_margin = float(np.median(margins))
    mean_stats = _compute_precision_at_threshold(pseudo_results, threshold=mean_margin)
    median_stats = _compute_precision_at_threshold(pseudo_results, threshold=median_margin)

    summary_row = {
        "integration_depth": int(integration_depth),
        "n_iterations": int(len(calibrated_runs)),
        "margin_mean": float(np.mean(margins)),
        "margin_median": float(np.median(margins)),
        "margin_std": float(np.std(margins, ddof=1)) if len(margins) > 1 else np.nan,
        "margin_iqr": float(np.percentile(margins, 75) - np.percentile(margins, 25)),
        "calibrated_precision_mean": (
            float(np.mean(finite_precision)) if finite_precision.size else np.nan
        ),
        "calibrated_precision_median": (
            float(np.median(finite_precision)) if finite_precision.size else np.nan
        ),
        "calibrated_precision_std": (
            float(np.std(finite_precision, ddof=1))
            if finite_precision.size > 1
            else np.nan
        ),
        "calibrated_precision_iqr": (
            float(np.percentile(finite_precision, 75) - np.percentile(finite_precision, 25))
            if finite_precision.size
            else np.nan
        ),
        "calibrated_coverage_mean": (
            float(np.mean(finite_coverage)) if finite_coverage.size else np.nan
        ),
        "calibrated_coverage_median": (
            float(np.median(finite_coverage)) if finite_coverage.size else np.nan
        ),
        "precision_at_mean_margin": mean_stats["precision"],
        "coverage_at_mean_margin": mean_stats["coverage"],
        "precision_at_median_margin": median_stats["precision"],
        "coverage_at_median_margin": median_stats["coverage"],
    }

    regime_rows = []
    for target in target_precisions:
        tau = thresholds_map.get(target, np.nan)
        if np.isfinite(tau):
            count_above = int((calibrated_runs["dihs_margin"] >= tau).sum())
            fraction_above = float(count_above / float(len(calibrated_runs)))
        else:
            count_above = 0
            fraction_above = np.nan
        summary_row[f"fraction_above_{int(round(100.0 * target))}"] = fraction_above
        regime_rows.append(
            {
                "integration_depth": int(integration_depth),
                "target_precision": target,
                "resolvedness_threshold": tau,
                "n_iterations_above_threshold": count_above,
                "fraction_iterations_above_threshold": fraction_above,
            }
        )

    return calibrated_runs, pd.DataFrame([summary_row]), pd.DataFrame(regime_rows)


def _aggregate_pairwise_iteration_totals(
    matrices: list[pd.DataFrame],
):
    if not matrices:
        return None, None
    all_units = set()
    for mat in matrices:
        all_units.update(mat.index.astype(str).tolist())
        all_units.update(mat.columns.astype(str).tolist())
    def _unit_sort_key_str(x):
        s = str(x)
        try:
            return (0, float(s))
        except Exception:
            return (1, s)
        
    all_units = sorted(all_units, key=_unit_sort_key_str)

    stack = []
    for mat in matrices:
        m = mat.copy()
        m.index = m.index.astype(str)
        m.columns = m.columns.astype(str)
        aligned = m.reindex(index=all_units, columns=all_units).fillna(0.0)
        aligned = 0.5 * (aligned + aligned.T)
        np.fill_diagonal(aligned.values, 1.0)
        stack.append(aligned.to_numpy(dtype=float))
    arr = np.stack(stack, axis=0)
    mean_df = pd.DataFrame(arr.mean(axis=0), index=all_units, columns=all_units)
    std_df = pd.DataFrame(arr.std(axis=0, ddof=0), index=all_units, columns=all_units)
    return mean_df, std_df


def _plot_top1_fraction(top1_df: pd.DataFrame, output_path: str | None = None):
    if top1_df.empty:
        return
    labels = top1_df["neighbor_unit"].astype(str).tolist()
    vals = top1_df["top1_fraction"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 9))
    if vals.sum() <= 0:
        plt.close(fig)
        return
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    for t in texts:
        t.set_fontsize(11)
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Top-1 frequency across perturbation iterations")
    ax.axis("equal")
    fig.tight_layout()
    if output_path is None:
        plt.show()
        plt.close(fig)
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def simple_run(
    *,
    df: pd.DataFrame,
    model_type: str,
    transform_type: str = "clr",
    unknown_sample: Any = 0,
    class_column: str = "controlcode",
    random_state: int | None = None,
    compute_pairwise: bool = True,
    plot_everything: bool = False,
    write_files: bool = False,
    output_dir: str = "./Results",
    plot_output_dir: str | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
):
    result = _run_single_model(
        df=df,
        model_type=model_type,
        transform_type=transform_type,
        unknown_sample=unknown_sample,
        class_column=class_column,
        random_state=random_state,
        compute_pairwise=compute_pairwise,
        plot_everything=plot_everything,
        write_files=write_files,
        output_dir=output_dir,
        plot_output_dir=plot_output_dir,
        max_depth=max_depth,
        exclude_columns=exclude_columns,
        save_cluster_data=save_cluster_data,
        save_untransformed=save_untransformed,
        verbose=verbose,
    )
    if return_details:
        return result
    return result["hs_per_depth"]


def triple_run(
    *,
    df: pd.DataFrame,
    transform_type: str = "clr",
    unknown_sample: Any = 0,
    class_column: str = "controlcode",
    random_state: int | None = None,
    compute_pairwise: bool = True,
    plot_everything: bool = False,
    write_files: bool = False,
    output_dir: str = "./Results_triple",
    plot_output_dir: str | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
):
    """Run agglomerative, kmeans and gaussian under one shared configuration."""
    model_results = {}
    _log(verbose, "Starting triple run (agglomerative, kmeans, gaussian)...")
    for model in SUPPORTED_MODELS:
        _log(verbose, f"Running model: {model}")
        model_out = os.path.join(output_dir, model) if write_files else output_dir
        model_plot_out = (
            os.path.join(plot_output_dir, model)
            if (plot_output_dir is not None and write_files)
            else plot_output_dir
        )
        model_results[model] = _run_single_model(
            df=df,
            model_type=model,
            transform_type=transform_type,
            unknown_sample=unknown_sample,
            class_column=class_column,
            random_state=random_state,
            compute_pairwise=compute_pairwise,
            plot_everything=plot_everything,
            write_files=write_files,
            output_dir=model_out,
            plot_output_dir=model_plot_out,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            save_cluster_data=save_cluster_data,
            save_untransformed=save_untransformed,
            verbose=verbose,
        )

    hs_combined = pd.concat(
        [model_results[m]["hs_per_depth"] for m in SUPPORTED_MODELS], ignore_index=True
    )
    dihs_combined = pd.concat(
        [model_results[m]["dihs_total"] for m in SUPPORTED_MODELS], ignore_index=True
    )

    combined = {
        "hs_per_depth": hs_combined,
        "dihs_total": dihs_combined,
        "models": model_results,
    }
    if return_details:
        return combined
    return combined["hs_per_depth"]


def perturbative_simple_run(
    *,
    df: pd.DataFrame,
    model_type: str,
    transform_type: str = "clr",
    unknown_sample: Any = 0,
    class_column: str = "controlcode",
    random_state: int | None = None,
    n_iterations: int = 100,
    major_cols: list[str] | None = None,
    trace_cols: list[str] | None = None,
    major_error: float = 0.02,
    trace_error: float = 0.10,
    perturbation_seed: int | None = None,
    compute_pairwise: bool = True,
    plot_everything: bool = False,
    write_files: bool = False,
    output_dir: str = "./Results_perturbative",
    plot_output_dir: str | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
):
    """Perturbative ensemble run for one model.

    Returns mean HS-by-depth dataframe by default. Set return_details=True for
    full outputs (per-iteration tables, top-1 stats, margins, pairwise aggregates).
    """
    major_cols_resolved, trace_cols_resolved = _resolve_major_trace_columns(
        df, major_cols, trace_cols, class_column=class_column
    )
    unknown_class = _resolve_unknown_class(df, unknown_sample, class_column=class_column)
    _log(verbose, f"Starting perturbative run for model='{model_type}'")
    _log(
        verbose,
        f"Unknown class resolved to: {unknown_class} | Iterations: {n_iterations} | Max depth: {max_depth}",
    )
    _log(
        verbose,
        f"Feature-space perturbation setup (shown once): majors={len(major_cols_resolved)}, traces={len(trace_cols_resolved)}",
    )
    rng = np.random.default_rng(perturbation_seed)

    hs_iters = []
    dihs_iters = []
    pairwise_totals = []
    artifacts = {"iteration_dirs": []}

    for it in range(n_iterations):
        perturbed = df.copy()
        if major_cols_resolved:
            x_major = perturbed[major_cols_resolved].to_numpy(dtype=float)
            eps_major = rng.uniform(-major_error, major_error, size=x_major.shape)
            perturbed[major_cols_resolved] = x_major * (1.0 + eps_major)
        if trace_cols_resolved:
            x_trace = perturbed[trace_cols_resolved].to_numpy(dtype=float)
            eps_trace = rng.uniform(-trace_error, trace_error, size=x_trace.shape)
            perturbed[trace_cols_resolved] = x_trace * (1.0 + eps_trace)

        iter_out = (
            os.path.join(output_dir, f"iter_{it:03d}")
            if (write_files or save_cluster_data)
            else output_dir
        )
        if write_files or save_cluster_data:
            os.makedirs(iter_out, exist_ok=True)
            artifacts["iteration_dirs"].append(iter_out)

        run = _run_single_model(
            df=perturbed,
            model_type=model_type,
            transform_type=transform_type,
            unknown_sample=unknown_class,
            class_column=class_column,
            random_state=random_state,
            compute_pairwise=compute_pairwise,
            plot_everything=False,
            write_files=write_files,
            output_dir=iter_out,
            plot_output_dir=None,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            save_cluster_data=save_cluster_data,
            save_untransformed=save_untransformed,
            verbose=False,
        )

        hs_i = run["hs_per_depth"].copy()
        hs_i["iteration"] = it
        hs_iters.append(hs_i)

        dihs_i = run["dihs_total"].copy()
        dihs_i["iteration"] = it
        dihs_iters.append(dihs_i)

        if compute_pairwise and run["pairwise_total_matrix"] is not None:
            pairwise_totals.append(run["pairwise_total_matrix"])
        if verbose:
            _print_progress(it + 1, n_iterations)

    hs_iterations = pd.concat(hs_iters, ignore_index=True) if hs_iters else pd.DataFrame()
    dihs_iterations_native = (
        pd.concat(dihs_iters, ignore_index=True) if dihs_iters else pd.DataFrame()
    )
    dihs_iterations, common_depth_level = _recompute_dihs_iterations_on_common_depth(
        hs_iterations
    )

    hs_group = ["unknown_class", "transform", "model", "depth_level", "neighbor_unit"]
    hs_summary = (
        hs_iterations.groupby(hs_group, as_index=False)
        .agg(
            harmonic_score_mean=("harmonic_score", "mean"),
            harmonic_score_std=("harmonic_score", "std"),
            n_iterations=("harmonic_score", "count"),
        )
        if not hs_iterations.empty
        else pd.DataFrame()
    )
    if not hs_summary.empty:
        with np.errstate(divide="ignore", invalid="ignore"):
            hs_summary["harmonic_score_cv"] = (
                hs_summary["harmonic_score_std"] / hs_summary["harmonic_score_mean"]
            )
        hs_summary["harmonic_score"] = hs_summary["harmonic_score_mean"]

    dihs_group = ["unknown_class", "neighbor_unit"]
    dihs_summary = (
        dihs_iterations.groupby(dihs_group, as_index=False)
        .agg(
            total_product_mean=("total_product", "mean"),
            total_product_std=("total_product", "std"),
            n_iterations=("total_product", "count"),
        )
        if not dihs_iterations.empty
        else pd.DataFrame()
    )
    if not dihs_summary.empty:
        dihs_summary["total_product"] = dihs_summary["total_product_mean"]

    top1_frequency = _compute_top1_stats(dihs_iterations, unknown_class)
    margin_per_iteration, margin_summary = _compute_margin_stats(
        dihs_iterations, unknown_class
    )
    pairwise_mean, pairwise_std = _aggregate_pairwise_iteration_totals(pairwise_totals)

    if plot_everything:
        _log(verbose, "Generating ensemble summary plots...")
        if plot_output_dir is None:
            plot_output_dir = os.path.join(output_dir, "Plots")
        os.makedirs(plot_output_dir, exist_ok=True)
        hs_curve_path = os.path.join(plot_output_dir, f"mean_hs_curve_{model_type}.svg")
        top1_plot_path = os.path.join(plot_output_dir, f"top1_fraction_{model_type}.svg")
        pairwise_plot_path = os.path.join(
            plot_output_dir, f"pairwise_total_mean_{model_type}.svg"
        )

        if not hs_summary.empty:
            plot_hs_curves(
                df=hs_summary,
                value_col="harmonic_score_mean",
                std_col="harmonic_score_std",
                with_shade=True,
                force_root_one=True,
                max_depth=common_depth_level,
                output_path=hs_curve_path,
                title=f"HS mean +/- SD vs depth | {model_type}",
                unknown_class=unknown_class,
            )
        _plot_top1_fraction(top1_frequency, output_path=top1_plot_path)
        if compute_pairwise and pairwise_mean is not None:
            plot_pairwise_matrix(
                matrix=pairwise_mean,
                title=f"Mean pairwise DIHS | {model_type}",
                output_path=pairwise_plot_path,
                unknown_class=unknown_class,
            )
        artifacts["mean_hs_curve_path"] = hs_curve_path
        artifacts["top1_fraction_plot_path"] = top1_plot_path
        artifacts["pairwise_total_mean_plot_path"] = pairwise_plot_path

    result = {
        "common_depth_level": common_depth_level,
        "hs_mean_per_depth": hs_summary,
        "hs_iterations": hs_iterations,
        "dihs_summary": dihs_summary,
        "dihs_iterations": dihs_iterations,
        "dihs_iterations_native": dihs_iterations_native,
        "top1_frequency": top1_frequency,
        "margin_per_iteration": margin_per_iteration,
        "margin_summary": margin_summary,
        "pairwise_total_mean_matrix": pairwise_mean if compute_pairwise else None,
        "pairwise_total_std_matrix": pairwise_std if compute_pairwise else None,
        "artifacts": artifacts,
    }
    _log(verbose, "Perturbative run completed.")
    if return_details:
        return result
    return result["hs_mean_per_depth"]


def perturbative_triple_run(
    *,
    df: pd.DataFrame,
    transform_type: str = "clr",
    unknown_sample: Any = 0,
    class_column: str = "controlcode",
    random_state: int | None = None,
    n_iterations: int = 100,
    major_cols: list[str] | None = None,
    trace_cols: list[str] | None = None,
    major_error: float = 0.02,
    trace_error: float = 0.10,
    perturbation_seed: int | None = None,
    compute_pairwise: bool = True,
    plot_everything: bool = False,
    write_files: bool = False,
    output_dir: str = "./Results_perturbative_triple",
    plot_output_dir: str | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
):
    """Perturbative ensemble run for all three models."""
    model_results = {}
    _log(verbose, "Starting perturbative triple run...")
    for model in SUPPORTED_MODELS:
        _log(verbose, f"Model {model}:")
        model_out = os.path.join(output_dir, model) if write_files else output_dir
        model_plot_out = (
            os.path.join(plot_output_dir, model)
            if (plot_output_dir is not None and write_files)
            else plot_output_dir
        )
        model_results[model] = perturbative_simple_run(
            df=df,
            model_type=model,
            transform_type=transform_type,
            unknown_sample=unknown_sample,
            class_column=class_column,
            random_state=random_state,
            n_iterations=n_iterations,
            major_cols=major_cols,
            trace_cols=trace_cols,
            major_error=major_error,
            trace_error=trace_error,
            perturbation_seed=perturbation_seed,
            compute_pairwise=compute_pairwise,
            plot_everything=plot_everything,
            write_files=write_files,
            output_dir=model_out,
            plot_output_dir=model_plot_out,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            save_cluster_data=save_cluster_data,
            save_untransformed=save_untransformed,
            verbose=verbose,
            return_details=True,
        )

    hs_mean_all = pd.concat(
        [model_results[m]["hs_mean_per_depth"] for m in SUPPORTED_MODELS], ignore_index=True
    )
    dihs_summary_all = pd.concat(
        [model_results[m]["dihs_summary"] for m in SUPPORTED_MODELS], ignore_index=True
    )
    top1_all = pd.concat(
        [
            model_results[m]["top1_frequency"].assign(model=m)
            for m in SUPPORTED_MODELS
            if not model_results[m]["top1_frequency"].empty
        ],
        ignore_index=True,
    ) if any(not model_results[m]["top1_frequency"].empty for m in SUPPORTED_MODELS) else pd.DataFrame(
        columns=["neighbor_unit", "wins", "top1_fraction", "n_iterations", "model"]
    )
    combined = {
        "hs_mean_per_depth": hs_mean_all,
        "dihs_summary": dihs_summary_all,
        "top1_frequency": top1_all,
        "models": model_results,
    }
    if return_details:
        return combined
    return combined["hs_mean_per_depth"]


def pseudo_unknown_run(
    *,
    df: pd.DataFrame,
    model_type: str,
    transform_type: str = "clr",
    class_column: str = "controlcode",
    sample_size: int = 5,
    n_iterations: int = 10,
    excluded_classes: list[Any] | None = None,
    random_state: int | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    target_precision: float = 0.95,
    reported_precisions: list[float] | None = None,
    min_runs_above_threshold: int = 1,
    plot_everything: bool = True,
    write_files: bool = False,
    output_dir: str = "./Results_pseudo_unknown",
    plot_output_dir: str | None = None,
    verbose: bool = True,
    return_details: bool = False,
):
    """
    Run pseudo-unknown experiments for one model.
    """
    result = run_pseudo_unknown_experiments(
        df=df,
        model_type=model_type,
        transform_type=transform_type,
        class_column=class_column,
        sample_size=sample_size,
        n_iterations=n_iterations,
        excluded_classes=excluded_classes,
        random_state=random_state,
        max_depth=max_depth,
        exclude_columns=exclude_columns,
        target_precision=target_precision,
        reported_precisions=reported_precisions,
        min_runs_above_threshold=min_runs_above_threshold,
        plot_everything=plot_everything,
        write_files=write_files,
        output_dir=output_dir,
        plot_output_dir=plot_output_dir,
        verbose=verbose,
    )
    if return_details:
        return result
    return result["run_results"]


def plot_pseudo_unknown_margin_from_outputs(
    *,
    pseudo_unknown_output_dir: str,
    perturbative_output_dir: str | None = None,
    integration_depth: int | None = None,
    target_precision: float = 0.95,
    threshold_mode: str = "target_precision",
    output_path: str | None = None,
    title: str | None = None,
    verbose: bool = True,
    return_details: bool = False,
):
    """Regenerate the pseudo-unknown margin plot from saved CSV outputs.

    When a perturbative output folder is provided, the perturbative margin
    distribution is recomputed at the same integration depth and overlaid as a
    full-width density band spanning the pseudo-unknown groups.
    """
    result = _prepare_margin_plot_data_from_outputs(
        pseudo_unknown_output_dir=pseudo_unknown_output_dir,
        perturbative_output_dir=perturbative_output_dir,
        integration_depth=integration_depth,
        target_precision=target_precision,
        verbose=verbose,
    )
    threshold_info = _resolve_display_threshold(
        result=result,
        threshold_mode=threshold_mode,
        target_precision=target_precision,
    )

    plot_margin_comparison(
        results_df=result["pseudo_results"],
        output_path=output_path,
        threshold=threshold_info["display_threshold"],
        threshold_label=threshold_info["threshold_label"],
        perturbative_margins=result["perturbative_margins"]["dihs_margin"].to_numpy(dtype=float)
        if not result["perturbative_margins"].empty
        else None,
        integration_depth=result["integration_depth"],
        target_precision=target_precision,
        title=title,
    )

    result["output_path"] = output_path
    result.update(threshold_info)
    if return_details:
        return result
    return result["pseudo_results"]


def plot_pseudo_unknown_margin_histogram_from_outputs(
    *,
    pseudo_unknown_output_dir: str,
    perturbative_output_dir: str | None = None,
    integration_depth: int | None = None,
    target_precision: float = 0.95,
    threshold_mode: str = "target_precision",
    bins: int = 32,
    output_path: str | None = None,
    title: str | None = None,
    verbose: bool = True,
    return_details: bool = False,
):
    """Regenerate the pseudo-unknown margin histogram from saved CSV outputs."""
    result = _prepare_margin_plot_data_from_outputs(
        pseudo_unknown_output_dir=pseudo_unknown_output_dir,
        perturbative_output_dir=perturbative_output_dir,
        integration_depth=integration_depth,
        target_precision=target_precision,
        verbose=verbose,
    )
    threshold_info = _resolve_display_threshold(
        result=result,
        threshold_mode=threshold_mode,
        target_precision=target_precision,
    )

    plot_margin_histogram(
        results_df=result["pseudo_results"],
        output_path=output_path,
        threshold=threshold_info["display_threshold"],
        threshold_label=threshold_info["threshold_label"],
        perturbative_margins=result["perturbative_margins"]["dihs_margin"].to_numpy(dtype=float)
        if not result["perturbative_margins"].empty
        else None,
        integration_depth=result["integration_depth"],
        target_precision=target_precision,
        title=title,
        bins=bins,
    )

    result["output_path"] = output_path
    result.update(threshold_info)
    if return_details:
        return result
    return result["pseudo_results"]


def calibrate_perturbative_resolvedness_from_outputs(
    *,
    pseudo_unknown_output_dir: str,
    perturbative_output_dir: str,
    integration_depth: int | None = None,
    target_precisions: list[float] | None = None,
    bins: int = 12,
    output_dir: str | None = None,
    plot_output_path: str | None = None,
    title: str | None = None,
    verbose: bool = True,
    return_details: bool = False,
):
    """Calibrate perturbative margins against pseudo-unknown resolvedness outputs."""
    precisions = list(target_precisions or [0.95, 0.90, 0.85, 0.80, 0.75])
    result = _prepare_margin_plot_data_from_outputs(
        pseudo_unknown_output_dir=pseudo_unknown_output_dir,
        perturbative_output_dir=perturbative_output_dir,
        integration_depth=integration_depth,
        target_precision=max(precisions),
        verbose=verbose,
    )

    threshold_curve, thresholds_by_target_precision = _load_pseudo_unknown_calibration_tables(
        output_dir=pseudo_unknown_output_dir,
        integration_depth=result["integration_depth"],
    )
    if not thresholds_by_target_precision.empty:
        thresholds_by_target_precision = thresholds_by_target_precision[
            thresholds_by_target_precision["target_precision"].astype(float).isin(
                [float(x) for x in precisions]
            )
        ].copy()

    calibrated_runs, calibration_summary, regime_summary = _build_perturbative_calibration_outputs(
        pseudo_results=result["pseudo_results"],
        perturbative_margins=result["perturbative_margins"],
        thresholds_by_target_precision=thresholds_by_target_precision,
        target_precisions=precisions,
        integration_depth=result["integration_depth"],
    )

    artifacts = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        calibrated_runs_path = os.path.join(
            output_dir, "perturbative_calibrated_runs.csv"
        )
        calibration_summary_path = os.path.join(
            output_dir, "perturbative_calibration_summary.csv"
        )
        regime_summary_path = os.path.join(
            output_dir, "perturbative_regime_summary.csv"
        )
        threshold_curve_path = os.path.join(
            output_dir, "pseudo_unknown_threshold_curve_for_depth.csv"
        )
        target_thresholds_path = os.path.join(
            output_dir, "pseudo_unknown_target_thresholds_for_depth.csv"
        )

        calibrated_runs.to_csv(calibrated_runs_path, index=False)
        calibration_summary.to_csv(calibration_summary_path, index=False)
        regime_summary.to_csv(regime_summary_path, index=False)
        threshold_curve.to_csv(threshold_curve_path, index=False)
        thresholds_by_target_precision.to_csv(target_thresholds_path, index=False)

        artifacts.update(
            {
                "calibrated_runs_csv": calibrated_runs_path,
                "calibration_summary_csv": calibration_summary_path,
                "regime_summary_csv": regime_summary_path,
                "threshold_curve_csv": threshold_curve_path,
                "target_thresholds_csv": target_thresholds_path,
            }
        )

        if plot_output_path is None:
            plot_output_path = os.path.join(
                output_dir, "perturbative_vs_pseudounknown_calibration.svg"
            )

    plot_perturbative_calibration_overlay(
        threshold_curve=threshold_curve,
        perturbative_runs=calibrated_runs,
        pseudo_results=result["pseudo_results"],
        output_path=plot_output_path,
        title=title,
        bins=bins,
    )
    artifacts["plot_path"] = plot_output_path

    out = {
        "integration_depth": result["integration_depth"],
        "pseudo_results": result["pseudo_results"],
        "threshold_curve": threshold_curve,
        "thresholds_by_target_precision": thresholds_by_target_precision,
        "perturbative_calibrated_runs": calibrated_runs,
        "perturbative_calibration_summary": calibration_summary,
        "perturbative_regime_summary": regime_summary,
        "artifacts": artifacts,
    }
    if return_details:
        return out
    return out["perturbative_calibrated_runs"]


def _normalize_target_precisions(
    target_precisions: list[float] | None,
) -> list[float]:
    precisions = sorted(
        {float(value) for value in (target_precisions or [0.95, 0.90, 0.85, 0.80, 0.75])},
        reverse=True,
    )
    if not precisions:
        raise ValueError("target_precisions must contain at least one value.")
    for precision in precisions:
        if not 0 < precision <= 1:
            raise ValueError("All target_precisions must be in the interval (0, 1].")
    return precisions


def _compute_perturbative_margins_at_depth(
    hs_iterations: pd.DataFrame,
    *,
    integration_depth: int,
) -> pd.DataFrame:
    if hs_iterations.empty:
        raise ValueError("No perturbative HS iterations were found to calibrate.")

    if int(integration_depth) < 0:
        raise ValueError("integration_depth must be >= 0.")

    max_depth_per_iter = hs_iterations.groupby("iteration", as_index=False)["depth_level"].max()
    if max_depth_per_iter.empty:
        raise ValueError("No perturbative depth information was found to calibrate.")

    too_shallow = max_depth_per_iter[max_depth_per_iter["depth_level"] < int(integration_depth)]
    if not too_shallow.empty:
        raise ValueError(
            f"Requested integration_depth={integration_depth} exceeds the depth reached by "
            f"{len(too_shallow)} perturbative iterations."
        )

    rows = []
    for iteration, sub in hs_iterations.groupby("iteration", as_index=False):
        row = _compute_margin_from_hs_metrics(
            metrics_df=sub,
            integration_depth=int(integration_depth),
        )
        if row is None:
            continue
        row["iteration"] = int(iteration)
        rows.append(row)

    margins = pd.DataFrame(rows)
    if margins.empty:
        raise ValueError(
            f"No perturbative margins could be computed at integration_depth={integration_depth}."
        )
    return margins


def _select_top1_candidate_for_calibration(
    perturbative_result: dict[str, Any],
    *,
    unknown_class: Any,
    model: str,
) -> dict[str, Any]:
    top1_frequency = perturbative_result["top1_frequency"].copy()
    dihs_summary = perturbative_result["dihs_summary"].copy()

    if not dihs_summary.empty:
        dihs_summary = dihs_summary[
            dihs_summary["neighbor_unit"].apply(_class_key) != _class_key(unknown_class)
        ].copy()
        dihs_summary["neighbor_unit_key"] = dihs_summary["neighbor_unit"].apply(_class_key)
        dihs_summary = dihs_summary.sort_values(
            ["total_product_mean", "total_product_std", "neighbor_unit_key"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    mean_rank_map = {
        row["neighbor_unit_key"]: idx
        for idx, (_, row) in enumerate(dihs_summary.iterrows(), start=1)
    }
    dihs_lookup = {
        row["neighbor_unit_key"]: row.to_dict() for _, row in dihs_summary.iterrows()
    }

    selection_method = "top1_frequency"
    if not top1_frequency.empty:
        top1_frequency = top1_frequency.copy()
        top1_frequency["neighbor_unit_key"] = top1_frequency["neighbor_unit"].apply(_class_key)
        max_fraction = float(top1_frequency["top1_fraction"].max())
        winners = top1_frequency[
            np.isclose(top1_frequency["top1_fraction"].astype(float), max_fraction)
        ].copy()
        if len(winners) > 1:
            winners["mean_rank"] = winners["neighbor_unit_key"].map(mean_rank_map).fillna(np.inf)
            winners = winners.sort_values(
                ["mean_rank", "wins", "neighbor_unit_key"],
                ascending=[True, False, True],
            ).reset_index(drop=True)
            chosen = winners.iloc[0]
            selection_method = "top1_frequency_tie_broken_by_mean_dihs"
            warnings.warn(
                f"Model '{model}' produced a Top-1 frequency tie across "
                f"{winners['neighbor_unit'].tolist()}. "
                f"Using '{chosen['neighbor_unit']}' for pseudo-unknown calibration.",
                stacklevel=3,
            )
        else:
            chosen = winners.iloc[0]
    elif not dihs_summary.empty:
        chosen_row = dihs_summary.iloc[0]
        chosen = pd.Series(
            {
                "neighbor_unit": chosen_row["neighbor_unit"],
                "neighbor_unit_key": chosen_row["neighbor_unit_key"],
                "wins": np.nan,
                "top1_fraction": np.nan,
            }
        )
        selection_method = "mean_dihs_fallback"
        warnings.warn(
            f"Model '{model}' did not produce Top-1 frequency data. "
            f"Using the highest mean DIHS class '{chosen_row['neighbor_unit']}' "
            f"for pseudo-unknown calibration.",
            stacklevel=3,
        )
    else:
        raise ValueError(
            f"Model '{model}' did not produce any candidate DIHS values for Top-1 calibration."
        )

    chosen_key = (
        chosen["neighbor_unit_key"]
        if "neighbor_unit_key" in chosen.index
        else _class_key(chosen["neighbor_unit"])
    )
    chosen_mean = dihs_lookup.get(chosen_key, {})
    secondary_rows = dihs_summary[dihs_summary["neighbor_unit_key"] != chosen_key].head(1)
    secondary = secondary_rows.iloc[0].to_dict() if not secondary_rows.empty else {}

    wins = chosen.get("wins", np.nan)
    return {
        "model": model,
        "top1_class": chosen["neighbor_unit"],
        "top1_class_key": chosen_key,
        "top1_frequency": (
            float(chosen["top1_fraction"]) if pd.notna(chosen["top1_fraction"]) else np.nan
        ),
        "top1_wins": int(wins) if pd.notna(wins) else np.nan,
        "top1_mean_dihs": chosen_mean.get("total_product_mean", np.nan),
        "top1_dihs_std": chosen_mean.get("total_product_std", np.nan),
        "top1_dihs_rank": mean_rank_map.get(chosen_key, np.nan),
        "top2_class": secondary.get("neighbor_unit", np.nan),
        "top2_class_key": secondary.get("neighbor_unit_key", np.nan),
        "top2_mean_dihs": secondary.get("total_product_mean", np.nan),
        "top2_dihs_std": secondary.get("total_product_std", np.nan),
        "selection_method": selection_method,
    }


def _select_pseudo_unknown_outputs_at_depth(
    pseudo_unknown_result: dict[str, Any],
    *,
    integration_depth: int,
    target_precisions: list[float],
) -> dict[str, pd.DataFrame]:
    run_results_by_depth = pseudo_unknown_result["run_results_by_depth"]
    if run_results_by_depth.empty:
        raise ValueError("No pseudo-unknown run results were produced for calibration.")

    pseudo_results = run_results_by_depth[
        run_results_by_depth["integration_depth"] == int(integration_depth)
    ].copy()
    if pseudo_results.empty:
        raise ValueError(
            f"No pseudo-unknown runs were found at integration_depth={integration_depth}."
        )

    threshold_curve_by_depth = pseudo_unknown_result["threshold_curve_by_depth"]
    if not threshold_curve_by_depth.empty and "integration_depth" in threshold_curve_by_depth.columns:
        threshold_curve = threshold_curve_by_depth[
            threshold_curve_by_depth["integration_depth"] == int(integration_depth)
        ].copy()
    else:
        threshold_curve = threshold_curve_by_depth.copy()

    thresholds_by_target_precision_by_depth = pseudo_unknown_result[
        "thresholds_by_target_precision_by_depth"
    ]
    if (
        not thresholds_by_target_precision_by_depth.empty
        and "integration_depth" in thresholds_by_target_precision_by_depth.columns
    ):
        thresholds_by_target_precision = thresholds_by_target_precision_by_depth[
            thresholds_by_target_precision_by_depth["integration_depth"]
            == int(integration_depth)
        ].copy()
    else:
        thresholds_by_target_precision = thresholds_by_target_precision_by_depth.copy()

    if not thresholds_by_target_precision.empty:
        thresholds_by_target_precision = thresholds_by_target_precision[
            thresholds_by_target_precision["target_precision"].astype(float).isin(
                [float(value) for value in target_precisions]
            )
        ].copy()

    return {
        "pseudo_results": pseudo_results,
        "threshold_curve": threshold_curve,
        "thresholds_by_target_precision": thresholds_by_target_precision,
    }


def _flatten_threshold_summary(
    thresholds_by_target_precision: pd.DataFrame,
) -> dict[str, Any]:
    summary = {}
    for _, row in thresholds_by_target_precision.iterrows():
        pct = int(round(100.0 * float(row["target_precision"])))
        summary[f"resolvedness_threshold_{pct}"] = (
            float(row["resolvedness_threshold"])
            if pd.notna(row["resolvedness_threshold"])
            else np.nan
        )
        summary[f"precision_above_threshold_{pct}"] = (
            float(row["precision_above_threshold"])
            if pd.notna(row["precision_above_threshold"])
            else np.nan
        )
        summary[f"coverage_above_threshold_{pct}"] = (
            float(row["coverage_above_threshold"])
            if pd.notna(row["coverage_above_threshold"])
            else np.nan
        )
        summary[f"n_runs_above_threshold_{pct}"] = (
            int(row["n_runs_above_threshold"])
            if pd.notna(row["n_runs_above_threshold"])
            else np.nan
        )
    return summary


def perturbative_triple_run_with_resolvedness(
    *,
    df: pd.DataFrame,
    transform_type: str = "clr",
    unknown_sample: Any = 0,
    class_column: str = "controlcode",
    random_state: int | None = None,
    n_iterations: int = 100,
    major_cols: list[str] | None = None,
    trace_cols: list[str] | None = None,
    major_error: float = 0.02,
    trace_error: float = 0.10,
    perturbation_seed: int | None = None,
    compute_pairwise: bool = True,
    plot_everything: bool = False,
    write_files: bool = False,
    output_dir: str = "./Results_perturbative_triple_resolvedness",
    plot_output_dir: str | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    pseudo_unknown_iterations: int = 100,
    pseudo_unknown_sample_size: int | None = None,
    pseudo_unknown_random_state: int | None = None,
    target_precisions: list[float] | None = None,
    min_runs_above_threshold: int = 1,
    integration_depth: int | None = None,
    verbose: bool = True,
    return_details: bool = False,
):
    """Run perturbative triple correlation plus Top-1 pseudo-unknown resolvedness calibration."""
    work_df = _prepare_working_df(df, class_column=class_column)
    unknown_class = _resolve_unknown_class(
        work_df, unknown_sample, class_column=class_column
    )
    precision_targets = _normalize_target_precisions(target_precisions)

    if pseudo_unknown_sample_size is None:
        inferred_size = int(_class_match_mask(work_df[class_column], unknown_class).sum())
        if inferred_size <= 0:
            raise ValueError(
                "Could not infer pseudo_unknown_sample_size from the unknown sample rows. "
                "Pass pseudo_unknown_sample_size explicitly."
            )
        pseudo_sample_size = inferred_size
    else:
        pseudo_sample_size = int(pseudo_unknown_sample_size)
        if pseudo_sample_size <= 0:
            raise ValueError("pseudo_unknown_sample_size must be > 0.")

    if int(pseudo_unknown_iterations) <= 0:
        raise ValueError("pseudo_unknown_iterations must be > 0.")
    if int(min_runs_above_threshold) <= 0:
        raise ValueError("min_runs_above_threshold must be > 0.")

    pseudo_random_state = (
        random_state if pseudo_unknown_random_state is None else pseudo_unknown_random_state
    )

    summary_rows = []
    model_results = {}

    _log(
        verbose,
        "Starting perturbative triple run with Top-1 pseudo-unknown resolvedness calibration...",
    )
    _log(
        verbose,
        f"Unknown class resolved to: {unknown_class} | Pseudo-unknown sample size: {pseudo_sample_size}",
    )

    for model in SUPPORTED_MODELS:
        _log(verbose, f"Model {model}: perturbative ensemble")
        model_root = os.path.join(output_dir, model)
        perturbative_output_dir = os.path.join(model_root, "perturbative")
        pseudo_output_dir = os.path.join(model_root, "pseudo_unknown")
        resolvedness_output_dir = os.path.join(model_root, "resolvedness")

        perturbative_plot_dir = None
        resolvedness_plot_dir = None
        if plot_everything:
            if plot_output_dir is None:
                perturbative_plot_dir = os.path.join(model_root, "Plots", "perturbative")
                resolvedness_plot_dir = os.path.join(model_root, "Plots", "resolvedness")
            else:
                perturbative_plot_dir = os.path.join(plot_output_dir, model, "perturbative")
                resolvedness_plot_dir = os.path.join(plot_output_dir, model, "resolvedness")

        perturbative_result = perturbative_simple_run(
            df=work_df,
            model_type=model,
            transform_type=transform_type,
            unknown_sample=unknown_class,
            class_column=class_column,
            random_state=random_state,
            n_iterations=n_iterations,
            major_cols=major_cols,
            trace_cols=trace_cols,
            major_error=major_error,
            trace_error=trace_error,
            perturbation_seed=perturbation_seed,
            compute_pairwise=compute_pairwise,
            plot_everything=plot_everything,
            write_files=write_files,
            output_dir=perturbative_output_dir,
            plot_output_dir=perturbative_plot_dir,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            save_cluster_data=save_cluster_data,
            save_untransformed=save_untransformed,
            verbose=verbose,
            return_details=True,
        )

        top1_candidate = _select_top1_candidate_for_calibration(
            perturbative_result,
            unknown_class=unknown_class,
            model=model,
        )
        top1_key = top1_candidate["top1_class_key"]

        pseudo_df = work_df.loc[
            ~_class_match_mask(work_df[class_column], unknown_class)
        ].copy()
        top1_mask = _class_match_mask(pseudo_df[class_column], top1_candidate["top1_class"])
        top1_source_count = int(top1_mask.sum())
        if top1_source_count <= pseudo_sample_size:
            raise ValueError(
                f"Model '{model}' selected Top-1 class '{top1_candidate['top1_class']}', "
                f"but it contains {top1_source_count} rows after removing the real unknown "
                f"class and therefore cannot support pseudo-unknown sample_size={pseudo_sample_size}."
            )

        excluded_classes = [
            value
            for value in pseudo_df[class_column].drop_duplicates().tolist()
            if _class_key(value) != top1_key
        ]

        _log(
            verbose,
            f"Model {model}: pseudo-unknown calibration on Top-1 class '{top1_candidate['top1_class']}'",
        )
        pseudo_unknown_result = pseudo_unknown_run(
            df=pseudo_df,
            model_type=model,
            transform_type=transform_type,
            class_column=class_column,
            sample_size=pseudo_sample_size,
            n_iterations=pseudo_unknown_iterations,
            excluded_classes=excluded_classes,
            random_state=pseudo_random_state,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            target_precision=max(precision_targets),
            reported_precisions=precision_targets,
            min_runs_above_threshold=min_runs_above_threshold,
            plot_everything=False,
            write_files=write_files,
            output_dir=pseudo_output_dir,
            plot_output_dir=None,
            verbose=verbose,
            return_details=True,
        )

        perturbative_common_depth = perturbative_result["common_depth_level"]
        pseudo_common_depth = pseudo_unknown_result["common_depth_level"]
        if integration_depth is None:
            available_depths = [
                int(depth)
                for depth in (perturbative_common_depth, pseudo_common_depth)
                if depth is not None
            ]
            if not available_depths:
                raise ValueError(
                    f"Model '{model}' did not produce a common integration depth."
                )
            chosen_depth = min(available_depths)
        else:
            chosen_depth = int(integration_depth)
            if chosen_depth < 0:
                raise ValueError("integration_depth must be >= 0.")
            if (
                perturbative_common_depth is not None
                and chosen_depth > int(perturbative_common_depth)
            ):
                raise ValueError(
                    f"Requested integration_depth={chosen_depth} exceeds the perturbative "
                    f"common depth {perturbative_common_depth} for model '{model}'."
                )
            if pseudo_common_depth is not None and chosen_depth > int(pseudo_common_depth):
                raise ValueError(
                    f"Requested integration_depth={chosen_depth} exceeds the pseudo-unknown "
                    f"common depth {pseudo_common_depth} for model '{model}'."
                )

        perturbative_margins = _compute_perturbative_margins_at_depth(
            perturbative_result["hs_iterations"],
            integration_depth=chosen_depth,
        )
        pseudo_depth_outputs = _select_pseudo_unknown_outputs_at_depth(
            pseudo_unknown_result,
            integration_depth=chosen_depth,
            target_precisions=precision_targets,
        )

        calibrated_runs, calibration_summary, regime_summary = (
            _build_perturbative_calibration_outputs(
                pseudo_results=pseudo_depth_outputs["pseudo_results"],
                perturbative_margins=perturbative_margins,
                thresholds_by_target_precision=pseudo_depth_outputs[
                    "thresholds_by_target_precision"
                ],
                target_precisions=precision_targets,
                integration_depth=chosen_depth,
            )
        )

        transform_name = str(transform_type).strip().lower()
        hs_summary = perturbative_result["hs_mean_per_depth"]
        if not hs_summary.empty and "transform" in hs_summary.columns:
            transform_name = str(hs_summary["transform"].iloc[0])

        summary_row = calibration_summary.iloc[0].to_dict()
        summary_row.update(
            {
                "model": model,
                "transform": transform_name,
                "unknown_class": unknown_class,
                "top1_class": top1_candidate["top1_class"],
                "top1_class_key": top1_candidate["top1_class_key"],
                "top1_frequency": top1_candidate["top1_frequency"],
                "top1_wins": top1_candidate["top1_wins"],
                "top1_mean_dihs": top1_candidate["top1_mean_dihs"],
                "top1_dihs_std": top1_candidate["top1_dihs_std"],
                "top1_dihs_rank": top1_candidate["top1_dihs_rank"],
                "top2_class": top1_candidate["top2_class"],
                "top2_class_key": top1_candidate["top2_class_key"],
                "top2_mean_dihs": top1_candidate["top2_mean_dihs"],
                "top2_dihs_std": top1_candidate["top2_dihs_std"],
                "top1_selection_method": top1_candidate["selection_method"],
                "empirical_resolvedness": summary_row["precision_at_mean_margin"],
                "n_pseudo_runs_above_mean_margin": _compute_precision_at_threshold(
                    pseudo_depth_outputs["pseudo_results"],
                    threshold=summary_row["margin_mean"],
                )["n_runs_above_threshold"],
                "perturbative_iterations": int(n_iterations),
                "pseudo_unknown_iterations": int(pseudo_unknown_iterations),
                "pseudo_unknown_sample_size": int(pseudo_sample_size),
                "top1_source_count": int(top1_source_count),
                "calibration_depth": int(chosen_depth),
                "perturbative_common_depth_level": perturbative_common_depth,
                "pseudo_unknown_common_depth_level": pseudo_common_depth,
            }
        )
        summary_row.update(
            _flatten_threshold_summary(
                pseudo_depth_outputs["thresholds_by_target_precision"]
            )
        )
        summary_rows.append(summary_row)

        resolvedness_artifacts = {}
        top1_summary_df = pd.DataFrame([top1_candidate])
        resolvedness_summary_df = pd.DataFrame([summary_row])

        if write_files:
            os.makedirs(resolvedness_output_dir, exist_ok=True)
            top1_summary_path = os.path.join(
                resolvedness_output_dir, "top1_candidate_summary.csv"
            )
            perturbative_margins_path = os.path.join(
                resolvedness_output_dir, "perturbative_margins_for_depth.csv"
            )
            pseudo_results_path = os.path.join(
                resolvedness_output_dir, "pseudo_unknown_runs_for_depth.csv"
            )
            threshold_curve_path = os.path.join(
                resolvedness_output_dir, "pseudo_unknown_threshold_curve_for_depth.csv"
            )
            target_thresholds_path = os.path.join(
                resolvedness_output_dir,
                "pseudo_unknown_target_thresholds_for_depth.csv",
            )
            calibrated_runs_path = os.path.join(
                resolvedness_output_dir, "perturbative_calibrated_runs.csv"
            )
            calibration_summary_path = os.path.join(
                resolvedness_output_dir, "perturbative_calibration_summary.csv"
            )
            regime_summary_path = os.path.join(
                resolvedness_output_dir, "perturbative_regime_summary.csv"
            )
            resolvedness_summary_path = os.path.join(
                resolvedness_output_dir, "resolvedness_summary.csv"
            )

            top1_summary_df.to_csv(top1_summary_path, index=False)
            perturbative_margins.to_csv(perturbative_margins_path, index=False)
            pseudo_depth_outputs["pseudo_results"].to_csv(pseudo_results_path, index=False)
            pseudo_depth_outputs["threshold_curve"].to_csv(threshold_curve_path, index=False)
            pseudo_depth_outputs["thresholds_by_target_precision"].to_csv(
                target_thresholds_path, index=False
            )
            calibrated_runs.to_csv(calibrated_runs_path, index=False)
            calibration_summary.to_csv(calibration_summary_path, index=False)
            regime_summary.to_csv(regime_summary_path, index=False)
            resolvedness_summary_df.to_csv(resolvedness_summary_path, index=False)

            resolvedness_artifacts.update(
                {
                    "top1_candidate_summary_csv": top1_summary_path,
                    "perturbative_margins_csv": perturbative_margins_path,
                    "pseudo_unknown_runs_csv": pseudo_results_path,
                    "threshold_curve_csv": threshold_curve_path,
                    "target_thresholds_csv": target_thresholds_path,
                    "calibrated_runs_csv": calibrated_runs_path,
                    "calibration_summary_csv": calibration_summary_path,
                    "regime_summary_csv": regime_summary_path,
                    "resolvedness_summary_csv": resolvedness_summary_path,
                }
            )

        if plot_everything:
            os.makedirs(resolvedness_plot_dir, exist_ok=True)
            mean_margin = float(summary_row["margin_mean"])
            empirical_resolvedness = summary_row["empirical_resolvedness"]
            threshold_label = (
                f"Empirical resolvedness = {100.0 * empirical_resolvedness:.0f}% | "
                f"perturbative mean = {mean_margin:.3f}"
                if np.isfinite(empirical_resolvedness)
                else f"Perturbative mean = {mean_margin:.3f}"
            )
            title_suffix = (
                f"{transform_name} + {model} | Top-1 = {top1_candidate['top1_class']}"
            )
            comparison_plot_path = os.path.join(
                resolvedness_plot_dir, "resolvedness_margin_comparison.svg"
            )
            histogram_plot_path = os.path.join(
                resolvedness_plot_dir, "resolvedness_margin_histogram.svg"
            )
            overlay_plot_path = os.path.join(
                resolvedness_plot_dir, "resolvedness_calibration_overlay.svg"
            )

            plot_margin_comparison(
                results_df=pseudo_depth_outputs["pseudo_results"],
                output_path=comparison_plot_path,
                threshold=mean_margin,
                threshold_label=threshold_label,
                perturbative_margins=perturbative_margins["dihs_margin"].to_numpy(
                    dtype=float
                ),
                integration_depth=chosen_depth,
                target_precision=max(precision_targets),
                title=f"Top-1 pseudo-unknown resolvedness | {title_suffix}",
            )
            plot_margin_histogram(
                results_df=pseudo_depth_outputs["pseudo_results"],
                output_path=histogram_plot_path,
                threshold=mean_margin,
                threshold_label=threshold_label,
                perturbative_margins=perturbative_margins["dihs_margin"].to_numpy(
                    dtype=float
                ),
                integration_depth=chosen_depth,
                target_precision=max(precision_targets),
                title=f"Resolvedness margin distributions | {title_suffix}",
            )
            plot_perturbative_calibration_overlay(
                threshold_curve=pseudo_depth_outputs["threshold_curve"],
                perturbative_runs=calibrated_runs,
                pseudo_results=pseudo_depth_outputs["pseudo_results"],
                output_path=overlay_plot_path,
                title=f"Resolvedness calibration overlay | {title_suffix}",
            )

            resolvedness_artifacts.update(
                {
                    "margin_comparison_plot_path": comparison_plot_path,
                    "margin_histogram_plot_path": histogram_plot_path,
                    "calibration_overlay_plot_path": overlay_plot_path,
                }
            )

        model_results[model] = {
            "perturbative": perturbative_result,
            "pseudo_unknown": pseudo_unknown_result,
            "resolvedness": {
                "integration_depth": int(chosen_depth),
                "top1_candidate_summary": top1_summary_df,
                "perturbative_margins": perturbative_margins,
                "pseudo_results": pseudo_depth_outputs["pseudo_results"],
                "threshold_curve": pseudo_depth_outputs["threshold_curve"],
                "thresholds_by_target_precision": pseudo_depth_outputs[
                    "thresholds_by_target_precision"
                ],
                "perturbative_calibrated_runs": calibrated_runs,
                "perturbative_calibration_summary": calibration_summary,
                "perturbative_regime_summary": regime_summary,
                "resolvedness_summary": resolvedness_summary_df,
                "artifacts": resolvedness_artifacts,
            },
        }

    summary_df = pd.DataFrame(summary_rows)
    root_artifacts = {}
    if write_files:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(
            output_dir, "perturbative_triple_resolvedness_summary.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        root_artifacts["summary_csv"] = summary_path

    out = {
        "summary": summary_df,
        "models": model_results,
        "unknown_class": unknown_class,
        "pseudo_unknown_sample_size": int(pseudo_sample_size),
        "target_precisions": precision_targets,
        "artifacts": root_artifacts,
    }
    if return_details:
        return out
    return out["summary"]
