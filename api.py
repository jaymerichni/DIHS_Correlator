import os
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
        save_cluster_data=write_files and save_cluster_data,
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
        if write_files:
            os.makedirs(plot_output_dir, exist_ok=True)

        hs_curve_path = None
        if write_files:
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
            pairwise_plot_path = None
            if write_files:
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
    numeric_cols = df.select_dtypes(include="number").columns.drop(class_column, errors="ignore")
    resolved_major = major_cols or [c for c in DEFAULT_MAJOR_COLS if c in df.columns]
    resolved_trace = trace_cols or [c for c in DEFAULT_TRACE_COLS if c in df.columns]
    resolved_major = [c for c in resolved_major if c in numeric_cols]
    resolved_trace = [c for c in resolved_trace if c in numeric_cols]
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
        "calibrated_precision_mean": float(np.nanmean(calibrated_precision)),
        "calibrated_precision_median": float(np.nanmedian(calibrated_precision)),
        "calibrated_precision_std": (
            float(np.nanstd(calibrated_precision, ddof=1))
            if np.sum(np.isfinite(calibrated_precision)) > 1
            else np.nan
        ),
        "calibrated_precision_iqr": (
            float(
                np.nanpercentile(calibrated_precision, 75)
                - np.nanpercentile(calibrated_precision, 25)
            )
            if np.any(np.isfinite(calibrated_precision))
            else np.nan
        ),
        "calibrated_coverage_mean": float(np.nanmean(calibrated_coverage)),
        "calibrated_coverage_median": float(np.nanmedian(calibrated_coverage)),
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

        iter_out = os.path.join(output_dir, f"iter_{it:03d}") if write_files else output_dir
        if write_files:
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
        hs_curve_path = (
            os.path.join(plot_output_dir, f"mean_hs_curve_{model_type}.svg")
            if write_files
            else None
        )
        top1_plot_path = (
            os.path.join(plot_output_dir, f"top1_fraction_{model_type}.svg")
            if write_files
            else None
        )
        pairwise_plot_path = (
            os.path.join(plot_output_dir, f"pairwise_total_mean_{model_type}.svg")
            if write_files
            else None
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