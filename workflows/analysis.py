import os
import warnings
from glob import glob
from typing import Any

import numpy as np
import pandas as pd
from DIHS_Correlator.workflows.utils import (
    _class_key,
    _log,
    _resolve_major_trace_columns,
)


def _exclude_unknown_neighbor(
    dihs_iterations: pd.DataFrame, unknown_class: Any
) -> pd.DataFrame:
    if dihs_iterations.empty:
        return dihs_iterations
    return dihs_iterations[
        dihs_iterations["neighbor_unit"].astype(str) != str(unknown_class)
    ].copy()


def _finite_float_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


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
            raise FileNotFoundError(f"No metrics CSV found inside '{iter_dir}'.")
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


