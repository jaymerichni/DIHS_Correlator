import os
from typing import Any, Iterable

import numpy as np
import pandas as pd

from DIHS_Correlator.core.transforms import BASE_TRANSFORMATIONS
from DIHS_Correlator.viz.pseudo_unknown import (
    plot_margin_comparison,
    plot_margin_histogram,
    plot_threshold_diagnostics,
)
from DIHS_Correlator.workflows.single_run import CorrelationRunner
from DIHS_Correlator.workflows.utils import _normalize_transform_type, _class_key


TRANSFORM_NAME_TO_ID = {v: k for k, v in BASE_TRANSFORMATIONS.items()}


def _class_sort_key(value: Any):
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def _make_pseudo_unknown_label(
    existing_values: Iterable[Any], source_class: Any, iteration: int, case: str
) -> str:
    existing_keys = {_class_key(v) for v in existing_values}
    base = f"__pseudo_unknown__{case}__{_class_key(source_class)}__{iteration}"
    candidate = base
    counter = 1
    while candidate in existing_keys:
        candidate = f"{base}__{counter}"
        counter += 1
    return candidate


def _extract_margin_result(
    dihs: pd.DataFrame,
    run_metadata: pd.Series,
):
    unknown_class = run_metadata["unknown_class"]
    source_class = run_metadata["source_class"]
    case = run_metadata["case"]
    iteration = int(run_metadata["iteration"])
    sample_size = int(run_metadata["sample_size"])
    run_id = int(run_metadata["run_id"])

    dihs = dihs.copy()
    dihs = dihs[dihs["neighbor_unit"].astype(str) != str(unknown_class)].copy()
    dihs = dihs.sort_values("total_product", ascending=False).reset_index(drop=True)

    top1_class = dihs["neighbor_unit"].iloc[0] if not dihs.empty else np.nan
    top1_dihs = float(dihs["total_product"].iloc[0]) if not dihs.empty else np.nan
    top2_class = dihs["neighbor_unit"].iloc[1] if len(dihs) > 1 else np.nan
    top2_dihs = float(dihs["total_product"].iloc[1]) if len(dihs) > 1 else np.nan
    margin = top1_dihs - top2_dihs if len(dihs) > 1 else np.nan

    source_present = case == "positive"
    top1_is_true_source = (
        pd.notna(top1_class) and _class_key(top1_class) == _class_key(source_class)
    )
    is_true_positive = bool(source_present and top1_is_true_source)

    return {
        "run_id": run_id,
        "source_class": source_class,
        "source_class_key": _class_key(source_class),
        "iteration": iteration,
        "case": case,
        "sample_size": sample_size,
        "unknown_class": unknown_class,
        "top1_class": top1_class,
        "top1_class_key": _class_key(top1_class) if pd.notna(top1_class) else np.nan,
        "top2_class": top2_class,
        "top2_class_key": _class_key(top2_class) if pd.notna(top2_class) else np.nan,
        "top1_dihs": top1_dihs,
        "top2_dihs": top2_dihs,
        "dihs_margin": margin,
        "true_source_present": source_present,
        "top1_is_true_source": top1_is_true_source,
        "is_true_positive": is_true_positive,
    }


def _recompute_dihs_for_all_depths(hs_iterations: pd.DataFrame):
    if hs_iterations.empty:
        return pd.DataFrame(), None

    max_depth_per_run = (
        hs_iterations.groupby("run_id", as_index=False)["depth_level"].max()
    )
    if max_depth_per_run.empty:
        return pd.DataFrame(), None

    common_depth_level = int(max_depth_per_run["depth_level"].min())
    all_rows = []
    keys = [
        "run_id",
        "source_class",
        "source_class_key",
        "case",
        "iteration",
        "sample_size",
        "true_source_present",
        "unknown_class",
        "transform",
        "model",
        "neighbor_unit",
    ]

    for integration_depth in range(common_depth_level + 1):
        hs_cut = hs_iterations[hs_iterations["depth_level"] <= integration_depth].copy()
        if hs_cut.empty:
            continue
        dihs_depth = (
            hs_cut.groupby(keys, as_index=False)
            .agg(hs_sum=("harmonic_score", "sum"))
            .assign(total_product=lambda x: x["hs_sum"] / float(integration_depth + 1))
            .drop(columns=["hs_sum"])
        )
        dihs_depth.insert(0, "integration_depth", integration_depth)
        all_rows.append(dihs_depth)

    if not all_rows:
        return pd.DataFrame(), common_depth_level

    return pd.concat(all_rows, ignore_index=True), common_depth_level


def _summarize_margin_results_from_dihs_all_depths(
    dihs_iterations_by_depth: pd.DataFrame,
):
    if dihs_iterations_by_depth.empty:
        return pd.DataFrame()

    rows = []
    for (integration_depth, run_id), sub in dihs_iterations_by_depth.groupby(
        ["integration_depth", "run_id"], as_index=False
    ):
        metadata = sub.iloc[0]
        row = _extract_margin_result(sub, metadata)
        row["integration_depth"] = int(integration_depth)
        row["run_id"] = int(run_id)
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_threshold_curve(
    results_df: pd.DataFrame,
    target_precision: float,
    min_runs_above_threshold: int,
):
    valid = results_df[results_df["dihs_margin"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(), np.nan

    valid["is_true_positive"] = valid["is_true_positive"].astype(bool)
    thresholds = np.sort(valid["dihs_margin"].unique())

    rows = []
    chosen_tau = np.nan
    for tau in thresholds:
        above = valid[valid["dihs_margin"] >= tau]
        if above.empty:
            continue
        n_above = int(len(above))
        tp_above = int(above["is_true_positive"].sum())
        precision = tp_above / float(n_above)
        coverage = n_above / float(len(valid))
        rows.append(
            {
                "threshold": float(tau),
                "n_runs_above_threshold": n_above,
                "n_true_positives_above_threshold": tp_above,
                "precision": precision,
                "coverage": coverage,
            }
        )
        if (
            np.isnan(chosen_tau)
            and n_above >= int(min_runs_above_threshold)
            and precision >= float(target_precision)
        ):
            chosen_tau = float(tau)

    return pd.DataFrame(rows), chosen_tau


def _compute_threshold_outputs_for_depths(
    run_results_by_depth: pd.DataFrame,
    target_precision: float,
    target_precisions: Iterable[float],
    min_runs_above_threshold: int,
    n_eligible_classes: int,
):
    if run_results_by_depth.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    threshold_curve_rows = []
    thresholds_rows = []
    threshold_summary_rows = []

    for integration_depth, sub in run_results_by_depth.groupby("integration_depth", as_index=False):
        n_total_runs = int(len(sub))
        n_valid_runs = int(sub["dihs_margin"].notna().sum())

        threshold_curve, _ = _compute_threshold_curve(
            results_df=sub,
            target_precision=target_precision,
            min_runs_above_threshold=min_runs_above_threshold,
        )
        if not threshold_curve.empty:
            threshold_curve.insert(0, "integration_depth", int(integration_depth))
            threshold_curve_rows.append(threshold_curve)

        thresholds_by_target = _build_precision_threshold_table(
            threshold_curve=threshold_curve,
            target_precisions=target_precisions,
            min_runs_above_threshold=min_runs_above_threshold,
            n_total_runs=n_total_runs,
            n_valid_runs=n_valid_runs,
            n_eligible_classes=n_eligible_classes,
        )
        thresholds_by_target.insert(0, "integration_depth", int(integration_depth))
        thresholds_rows.append(thresholds_by_target)

        main_row = thresholds_by_target[
            thresholds_by_target["target_precision"] == float(target_precision)
        ].copy()
        if main_row.empty:
            main_row = pd.DataFrame(
                [
                    {
                        "integration_depth": int(integration_depth),
                        "target_precision": float(target_precision),
                        "resolvedness_threshold": np.nan,
                        "precision_above_threshold": np.nan,
                        "coverage_above_threshold": np.nan,
                        "n_runs_above_threshold": 0,
                        "n_total_runs": n_total_runs,
                        "n_valid_runs": n_valid_runs,
                        "n_eligible_classes": int(n_eligible_classes),
                        "min_runs_above_threshold": int(min_runs_above_threshold),
                    }
                ]
            )
        threshold_summary_rows.append(main_row)

    threshold_curve_by_depth = (
        pd.concat(threshold_curve_rows, ignore_index=True)
        if threshold_curve_rows
        else pd.DataFrame()
    )
    thresholds_by_target_precision_by_depth = (
        pd.concat(thresholds_rows, ignore_index=True)
        if thresholds_rows
        else pd.DataFrame()
    )
    threshold_summary_by_depth = (
        pd.concat(threshold_summary_rows, ignore_index=True)
        if threshold_summary_rows
        else pd.DataFrame()
    )
    return (
        threshold_curve_by_depth,
        thresholds_by_target_precision_by_depth,
        threshold_summary_by_depth,
    )


def _build_precision_threshold_table(
    threshold_curve: pd.DataFrame,
    target_precisions: Iterable[float],
    min_runs_above_threshold: int,
    n_total_runs: int,
    n_valid_runs: int,
    n_eligible_classes: int,
):
    rows = []
    ordered_targets = sorted({float(x) for x in target_precisions}, reverse=True)

    if threshold_curve.empty:
        for target in ordered_targets:
            rows.append(
                {
                    "target_precision": target,
                    "resolvedness_threshold": np.nan,
                    "precision_above_threshold": np.nan,
                    "coverage_above_threshold": np.nan,
                    "n_runs_above_threshold": 0,
                    "n_total_runs": int(n_total_runs),
                    "n_valid_runs": int(n_valid_runs),
                    "n_eligible_classes": int(n_eligible_classes),
                    "min_runs_above_threshold": int(min_runs_above_threshold),
                }
            )
        return pd.DataFrame(rows)

    for target in ordered_targets:
        selected = threshold_curve[
            (threshold_curve["precision"] >= target)
            & (threshold_curve["n_runs_above_threshold"] >= int(min_runs_above_threshold))
        ]

        if selected.empty:
            rows.append(
                {
                    "target_precision": target,
                    "resolvedness_threshold": np.nan,
                    "precision_above_threshold": np.nan,
                    "coverage_above_threshold": np.nan,
                    "n_runs_above_threshold": 0,
                    "n_total_runs": int(n_total_runs),
                    "n_valid_runs": int(n_valid_runs),
                    "n_eligible_classes": int(n_eligible_classes),
                    "min_runs_above_threshold": int(min_runs_above_threshold),
                }
            )
            continue

        best = selected.sort_values("threshold", ascending=True).iloc[0]
        rows.append(
            {
                "target_precision": target,
                "resolvedness_threshold": float(best["threshold"]),
                "precision_above_threshold": float(best["precision"]),
                "coverage_above_threshold": float(best["coverage"]),
                "n_runs_above_threshold": int(best["n_runs_above_threshold"]),
                "n_total_runs": int(n_total_runs),
                "n_valid_runs": int(n_valid_runs),
                "n_eligible_classes": int(n_eligible_classes),
                "min_runs_above_threshold": int(min_runs_above_threshold),
            }
        )

    return pd.DataFrame(rows)


def run_pseudo_unknown_experiments(
    *,
    df: pd.DataFrame,
    model_type: str,
    transform_type: str = "clr",
    class_column: str = "controlcode",
    sample_size: int = 5,
    n_iterations: int = 10,
    excluded_classes: Iterable[Any] | None = None,
    random_state: int | None = None,
    max_depth: int = 100,
    exclude_columns=(),
    target_precision: float = 0.95,
    reported_precisions: Iterable[float] | None = None,
    min_runs_above_threshold: int = 1,
    plot_everything: bool = True,
    write_files: bool = False,
    output_dir: str = "./Results_pseudo_unknown",
    plot_output_dir: str | None = None,
    verbose: bool = True,
    progress_callback=None,
):
    """
    Run positive and negative pseudo-unknown experiments for each eligible class.

    Positive case: sampled rows are relabeled as unknown while the rest of the
    source class remains in the dataset.

    Negative case: sampled rows are relabeled as unknown and all remaining rows
    of that source class are removed from the dataset.
    """
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in dataframe.")
    if int(sample_size) <= 0:
        raise ValueError("sample_size must be > 0.")
    if int(n_iterations) <= 0:
        raise ValueError("n_iterations must be > 0.")
    if not 0 < float(target_precision) <= 1:
        raise ValueError("target_precision must be in the interval (0, 1].")
    if reported_precisions is not None:
        for precision in reported_precisions:
            if not 0 < float(precision) <= 1:
                raise ValueError("All reported_precisions must be in the interval (0, 1].")

    model = str(model_type).strip().lower()
    if model not in {"agglomerative", "kmeans", "gaussian"}:
        raise ValueError(f"Unsupported model_type='{model_type}'.")
    transform_id = _normalize_transform_type(transform_type)

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
                {"source_class": source_class, "source_class_key": source_key, "count": count, "reason": "excluded"}
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
        raise ValueError("No eligible classes available for pseudo-unknown experiments.")

    rng = np.random.default_rng(random_state)
    runner = CorrelationRunner(
        base_output_dir=output_dir,
        save_trees=False,
        save_cluster_data=False,
        save_untransformed=False,
    )
    exclude_set = set(exclude_columns) | {class_column}
    runner.set_feature_columns(df.copy(), exclude=tuple(exclude_set), verbose=False)

    hs_rows = []
    run_id_counter = 0
    total_runs = len(eligible_classes) * int(n_iterations) * 2
    completed_runs = 0
    if verbose:
        print(
            f"Running pseudo-unknown experiments | Classes: {len(eligible_classes)} | Sample size: {sample_size} | Iterations per class: {n_iterations}"
        )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "pseudo_unknown",
                "message": "Pseudo-unknown calibration",
                "current": 0,
                "total": total_runs,
                "fraction": 0.0,
            }
        )

    for source_class in eligible_classes:
        source_key = _class_key(source_class)
        class_mask = df[class_column].apply(lambda x: _class_key(x) == source_key)
        class_indices = df.index[class_mask].to_numpy()

        if verbose:
            print(f"Source class: {source_class} ({len(class_indices)} rows)")

        for iteration in range(int(n_iterations)):
            sampled_indices = rng.choice(class_indices, size=int(sample_size), replace=False)
            sampled_index_set = set(sampled_indices.tolist())

            positive_df = df.copy()
            positive_unknown = _make_pseudo_unknown_label(
                df[class_column].unique(), source_class, iteration, "positive"
            )
            positive_df.loc[list(sampled_index_set), class_column] = positive_unknown

            negative_unknown = _make_pseudo_unknown_label(
                positive_df[class_column].unique(), source_class, iteration, "negative"
            )
            negative_df = positive_df.copy()
            negative_df.loc[list(sampled_index_set), class_column] = negative_unknown
            keep_mask = (~class_mask) | negative_df.index.isin(sampled_index_set)
            negative_df = negative_df.loc[keep_mask].copy()

            positive_run = runner.run_combination(
                data=positive_df,
                transform_type=transform_id,
                model_type=model,
                random_state=random_state if model in ("kmeans", "gaussian") else None,
                unknown_class=positive_unknown,
                class_column=class_column,
                compute_pairwise=False,
                write_outputs=False,
                max_depth=max_depth,
            )
            hs_positive = positive_run["metrics_per_depth"].copy()
            hs_positive["run_id"] = run_id_counter
            hs_positive["source_class"] = source_class
            hs_positive["source_class_key"] = _class_key(source_class)
            hs_positive["case"] = "positive"
            hs_positive["iteration"] = iteration
            hs_positive["sample_size"] = int(sample_size)
            hs_positive["true_source_present"] = True
            hs_rows.append(hs_positive)
            run_id_counter += 1
            completed_runs += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "pseudo_unknown",
                        "message": (
                            f"Pseudo-unknown calibration | {source_class} | "
                            f"positive case {iteration + 1} of {int(n_iterations)}"
                        ),
                        "current": completed_runs,
                        "total": total_runs,
                        "fraction": completed_runs / float(total_runs),
                    }
                )

            negative_run = runner.run_combination(
                data=negative_df,
                transform_type=transform_id,
                model_type=model,
                random_state=random_state if model in ("kmeans", "gaussian") else None,
                unknown_class=negative_unknown,
                class_column=class_column,
                compute_pairwise=False,
                write_outputs=False,
                max_depth=max_depth,
            )
            hs_negative = negative_run["metrics_per_depth"].copy()
            hs_negative["run_id"] = run_id_counter
            hs_negative["source_class"] = source_class
            hs_negative["source_class_key"] = _class_key(source_class)
            hs_negative["case"] = "negative"
            hs_negative["iteration"] = iteration
            hs_negative["sample_size"] = int(sample_size)
            hs_negative["true_source_present"] = False
            hs_rows.append(hs_negative)
            run_id_counter += 1
            completed_runs += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "pseudo_unknown",
                        "message": (
                            f"Pseudo-unknown calibration | {source_class} | "
                            f"negative case {iteration + 1} of {int(n_iterations)}"
                        ),
                        "current": completed_runs,
                        "total": total_runs,
                        "fraction": completed_runs / float(total_runs),
                    }
                )

    hs_iterations = pd.concat(hs_rows, ignore_index=True) if hs_rows else pd.DataFrame()
    dihs_iterations_by_depth, common_depth_level = _recompute_dihs_for_all_depths(
        hs_iterations
    )
    if common_depth_level is None:
        dihs_iterations = pd.DataFrame()
        results_df = pd.DataFrame()
        run_results_by_depth = pd.DataFrame()
    else:
        dihs_iterations = dihs_iterations_by_depth[
            dihs_iterations_by_depth["integration_depth"] == common_depth_level
        ].copy()
        run_results_by_depth = _summarize_margin_results_from_dihs_all_depths(
            dihs_iterations_by_depth
        )
        results_df = run_results_by_depth[
            run_results_by_depth["integration_depth"] == common_depth_level
        ].copy()
    skipped_df = pd.DataFrame(skipped_rows)
    eligible_df = pd.DataFrame(
        {
            "source_class": eligible_classes,
            "source_class_key": [_class_key(c) for c in eligible_classes],
            "count": [int(class_counts.loc[c]) for c in eligible_classes],
        }
    )

    summary_by_class = (
        results_df.groupby(["source_class_key", "case"], as_index=False)
        .agg(
            n_runs=("dihs_margin", "size"),
            margin_mean=("dihs_margin", "mean"),
            margin_std=("dihs_margin", "std"),
            margin_median=("dihs_margin", "median"),
            top1_true_fraction=("top1_is_true_source", "mean"),
        )
        if not results_df.empty
        else pd.DataFrame()
    )
    summary_by_case = (
        results_df.groupby("case", as_index=False)
        .agg(
            n_runs=("dihs_margin", "size"),
            margin_mean=("dihs_margin", "mean"),
            margin_std=("dihs_margin", "std"),
            margin_median=("dihs_margin", "median"),
            true_positive_fraction=("is_true_positive", "mean"),
        )
        if not results_df.empty
        else pd.DataFrame()
    )

    n_valid_runs = int(results_df["dihs_margin"].notna().sum())
    precision_targets = list(reported_precisions or [0.95, 0.90, 0.85, 0.80, 0.75])
    precision_targets.append(float(target_precision))
    (
        threshold_curve_by_depth,
        thresholds_by_target_precision_by_depth,
        threshold_summary_by_depth,
    ) = _compute_threshold_outputs_for_depths(
        run_results_by_depth=run_results_by_depth,
        target_precision=target_precision,
        target_precisions=precision_targets,
        min_runs_above_threshold=min_runs_above_threshold,
        n_eligible_classes=len(eligible_classes),
    )
    threshold_curve = (
        threshold_curve_by_depth[
            threshold_curve_by_depth["integration_depth"] == common_depth_level
        ].copy()
        if (
            common_depth_level is not None
            and not threshold_curve_by_depth.empty
            and "integration_depth" in threshold_curve_by_depth.columns
        )
        else pd.DataFrame()
    )
    thresholds_by_target_precision = (
        thresholds_by_target_precision_by_depth[
            thresholds_by_target_precision_by_depth["integration_depth"]
            == common_depth_level
        ].copy()
        if (
            common_depth_level is not None
            and not thresholds_by_target_precision_by_depth.empty
            and "integration_depth" in thresholds_by_target_precision_by_depth.columns
        )
        else pd.DataFrame()
    )
    threshold_summary = (
        threshold_summary_by_depth[
            threshold_summary_by_depth["integration_depth"] == common_depth_level
        ].copy()
        if (
            common_depth_level is not None
            and not threshold_summary_by_depth.empty
            and "integration_depth" in threshold_summary_by_depth.columns
        )
        else pd.DataFrame()
    )

    main_threshold_row = thresholds_by_target_precision[
        thresholds_by_target_precision["target_precision"] == float(target_precision)
    ]
    if not main_threshold_row.empty:
        resolvedness_threshold = float(main_threshold_row["resolvedness_threshold"].iloc[0])
    else:
        resolvedness_threshold = np.nan

    if np.isfinite(resolvedness_threshold):
        results_df["resolved"] = results_df["dihs_margin"] >= resolvedness_threshold
        above = results_df[results_df["resolved"]].copy()
        precision_above = (
            float(above["is_true_positive"].mean()) if not above.empty else np.nan
        )
        coverage_above = float(len(above) / n_valid_runs) if n_valid_runs else np.nan
        n_above = int(len(above))
    else:
        results_df["resolved"] = False
        precision_above = np.nan
        coverage_above = np.nan
        n_above = 0

    if threshold_summary.empty:
        threshold_summary = pd.DataFrame(
            [
                {
                    "integration_depth": common_depth_level,
                    "resolvedness_threshold": resolvedness_threshold,
                    "target_precision": float(target_precision),
                    "min_runs_above_threshold": int(min_runs_above_threshold),
                    "precision_above_threshold": precision_above,
                    "coverage_above_threshold": coverage_above,
                    "n_runs_above_threshold": n_above,
                    "n_total_runs": int(len(results_df)),
                    "n_valid_runs": int(n_valid_runs),
                    "n_eligible_classes": int(len(eligible_classes)),
                    "common_depth_level": common_depth_level,
                }
            ]
        )
    else:
        threshold_summary = threshold_summary.copy()
        threshold_summary["resolvedness_threshold"] = resolvedness_threshold
        threshold_summary["precision_above_threshold"] = precision_above
        threshold_summary["coverage_above_threshold"] = coverage_above
        threshold_summary["n_runs_above_threshold"] = n_above
        threshold_summary["common_depth_level"] = common_depth_level

    artifacts = {}
    if write_files:
        os.makedirs(output_dir, exist_ok=True)
        hs_iterations_path = os.path.join(output_dir, "pseudo_unknown_hs_iterations.csv")
        results_path = os.path.join(output_dir, "pseudo_unknown_runs.csv")
        summary_by_class_path = os.path.join(output_dir, "pseudo_unknown_summary_by_class.csv")
        summary_by_case_path = os.path.join(output_dir, "pseudo_unknown_summary_by_case.csv")
        eligible_path = os.path.join(output_dir, "pseudo_unknown_eligible_classes.csv")
        skipped_path = os.path.join(output_dir, "pseudo_unknown_skipped_classes.csv")
        threshold_curve_path = os.path.join(output_dir, "pseudo_unknown_threshold_curve.csv")
        threshold_summary_path = os.path.join(output_dir, "pseudo_unknown_threshold_summary.csv")
        thresholds_by_precision_path = os.path.join(
            output_dir, "pseudo_unknown_thresholds_by_target_precision.csv"
        )
        dihs_iterations_path = os.path.join(output_dir, "pseudo_unknown_dihs_iterations.csv")
        dihs_by_depth_path = os.path.join(output_dir, "pseudo_unknown_dihs_by_depth.csv")
        run_results_by_depth_path = os.path.join(
            output_dir, "pseudo_unknown_run_results_by_depth.csv"
        )
        threshold_curve_by_depth_path = os.path.join(
            output_dir, "pseudo_unknown_threshold_curve_by_depth.csv"
        )
        thresholds_by_precision_by_depth_path = os.path.join(
            output_dir, "pseudo_unknown_thresholds_by_target_precision_by_depth.csv"
        )

        hs_iterations.to_csv(hs_iterations_path, index=False)
        results_df.to_csv(results_path, index=False)
        dihs_iterations.to_csv(dihs_iterations_path, index=False)
        run_results_by_depth.to_csv(run_results_by_depth_path, index=False)
        summary_by_class.to_csv(summary_by_class_path, index=False)
        summary_by_case.to_csv(summary_by_case_path, index=False)
        eligible_df.to_csv(eligible_path, index=False)
        skipped_df.to_csv(skipped_path, index=False)
        dihs_iterations_by_depth.to_csv(dihs_by_depth_path, index=False)
        threshold_curve.to_csv(threshold_curve_path, index=False)
        threshold_curve_by_depth.to_csv(threshold_curve_by_depth_path, index=False)
        threshold_summary.to_csv(threshold_summary_path, index=False)
        thresholds_by_target_precision.to_csv(thresholds_by_precision_path, index=False)
        thresholds_by_target_precision_by_depth.to_csv(
            thresholds_by_precision_by_depth_path, index=False
        )

        artifacts.update(
            {
                "hs_iterations_csv": hs_iterations_path,
                "results_csv": results_path,
                "dihs_iterations_csv": dihs_iterations_path,
                "run_results_by_depth_csv": run_results_by_depth_path,
                "summary_by_class_csv": summary_by_class_path,
                "summary_by_case_csv": summary_by_case_path,
                "eligible_classes_csv": eligible_path,
                "skipped_classes_csv": skipped_path,
                "dihs_by_depth_csv": dihs_by_depth_path,
                "threshold_curve_csv": threshold_curve_path,
                "threshold_curve_by_depth_csv": threshold_curve_by_depth_path,
                "threshold_summary_csv": threshold_summary_path,
                "thresholds_by_target_precision_csv": thresholds_by_precision_path,
                "thresholds_by_target_precision_by_depth_csv": thresholds_by_precision_by_depth_path,
            }
        )

    if plot_everything:
        if plot_output_dir is None:
            plot_output_dir = os.path.join(output_dir, "Plots")
        os.makedirs(plot_output_dir, exist_ok=True)
        margin_plot_path = os.path.join(
            plot_output_dir, "pseudo_unknown_margin_comparison.svg"
        )
        threshold_plot_path = os.path.join(
            plot_output_dir, "pseudo_unknown_threshold_diagnostics.svg"
        )
        histogram_plot_path = os.path.join(
            plot_output_dir, "pseudo_unknown_margin_histogram.svg"
        )

        plot_margin_comparison(
            results_df=results_df,
            output_path=margin_plot_path,
            threshold=resolvedness_threshold,
            integration_depth=common_depth_level,
            target_precision=target_precision,
        )
        plot_margin_histogram(
            results_df=results_df,
            output_path=histogram_plot_path,
            threshold=resolvedness_threshold,
            integration_depth=common_depth_level,
            target_precision=target_precision,
        )
        plot_threshold_diagnostics(
            threshold_curve=threshold_curve,
            output_path=threshold_plot_path,
            precision_targets=sorted({float(x) for x in precision_targets}, reverse=True),
        )

        artifacts["margin_plot_path"] = margin_plot_path
        artifacts["margin_histogram_path"] = histogram_plot_path
        artifacts["threshold_plot_path"] = threshold_plot_path

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "complete",
                "message": "Pseudo-unknown calibration complete",
                "current": total_runs,
                "total": total_runs,
                "fraction": 1.0,
            }
        )

    return {
        "run_results": results_df,
        "hs_iterations": hs_iterations,
        "dihs_iterations": dihs_iterations,
        "run_results_by_depth": run_results_by_depth,
        "dihs_iterations_by_depth": dihs_iterations_by_depth,
        "summary_by_class": summary_by_class,
        "summary_by_case": summary_by_case,
        "eligible_classes": eligible_df,
        "skipped_classes": skipped_df,
        "threshold_curve": threshold_curve,
        "threshold_curve_by_depth": threshold_curve_by_depth,
        "threshold_summary": threshold_summary,
        "thresholds_by_target_precision": thresholds_by_target_precision,
        "thresholds_by_target_precision_by_depth": thresholds_by_target_precision_by_depth,
        "resolvedness_threshold": resolvedness_threshold,
        "common_depth_level": common_depth_level,
        "artifacts": artifacts,
    }
