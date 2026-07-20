"""
Resolvedness calibration workflow.

Orchestrates perturbative triple ensemble runs combined with Top-1 pseudo-unknown
calibration to quantify resolvedness as a function of perturbative margins and
target precision thresholds.
"""

import os
import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd

from DIHS_Correlator.viz.pairwise import plot_pairwise_matrix
from DIHS_Correlator.viz.pseudo_unknown import (
    plot_margin_comparison,
    plot_margin_histogram,
    plot_perturbative_calibration_overlay,
)
from DIHS_Correlator.workflows.analysis import (
    _build_perturbative_calibration_outputs,
    _compute_margin_from_hs_metrics,
    _compute_precision_at_threshold,
    _flatten_threshold_summary,
    _normalize_target_precisions,
    _select_pseudo_unknown_outputs_at_depth,
    _select_top1_candidate_for_calibration,
    _compute_perturbative_margins_at_depth,
)
from DIHS_Correlator.workflows.perturbative import (
    _aggregate_pairwise_iteration_totals_at_depth,
    _compute_perturbative_outputs_at_depth,
    perturbative_simple_run_workflow,
)
from DIHS_Correlator.workflows.pseudo_unknown import run_pseudo_unknown_experiments
from DIHS_Correlator.workflows.utils import (
    _class_key,
    _class_match_mask,
    _log,
    _prepare_working_df,
    _resolve_unknown_class,
)

SUPPORTED_MODELS = ("agglomerative", "kmeans", "gaussian")


def _emit_progress(
    progress_callback,
    *,
    stage: str,
    message: str | None = None,
    current: int | None = None,
    total: int | None = None,
    fraction: float | None = None,
) -> None:
    if progress_callback is None:
        return

    payload: dict[str, Any] = {"stage": stage}
    if message is not None:
        payload["message"] = message
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    if fraction is not None:
        payload["fraction"] = min(max(float(fraction), 0.0), 1.0)
    progress_callback(payload)


def _apply_perturbative_outputs_at_depth(
    perturbative_result: dict[str, Any],
    *,
    unknown_class: Any,
    integration_depth: int,
):
    depth_outputs = _compute_perturbative_outputs_at_depth(
        hs_iterations=perturbative_result["hs_iterations"],
        unknown_class=unknown_class,
        integration_depth=integration_depth,
    )
    perturbative_result["dihs_iterations"] = depth_outputs["dihs_iterations"]
    perturbative_result["dihs_summary"] = depth_outputs["dihs_summary"]
    perturbative_result["top1_frequency"] = depth_outputs["top1_frequency"]
    perturbative_result["margin_per_iteration"] = depth_outputs["margin_per_iteration"]
    perturbative_result["margin_summary"] = depth_outputs["margin_summary"]
    perturbative_result["dihs_integration_depth"] = int(integration_depth)
    return depth_outputs


def _run_pseudo_unknown_for_top1(
    *,
    pseudo_unknown_run_fn,
    pseudo_df: pd.DataFrame,
    top1_candidate: dict[str, Any],
    model: str,
    class_column: str,
    transform_type: str,
    pseudo_sample_size: int,
    pseudo_unknown_iterations: int,
    pseudo_random_state: int | None,
    max_depth: int,
    exclude_columns,
    precision_targets: list[float],
    min_runs_above_threshold: int,
    write_files: bool,
    output_dir: str,
    verbose: bool,
    progress_callback=None,
):
    top1_key = top1_candidate["top1_class_key"]
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
    pseudo_unknown_kwargs = {
        "df": pseudo_df,
        "model_type": model,
        "transform_type": transform_type,
        "class_column": class_column,
        "sample_size": pseudo_sample_size,
        "n_iterations": pseudo_unknown_iterations,
        "excluded_classes": excluded_classes,
        "random_state": pseudo_random_state,
        "max_depth": max_depth,
        "exclude_columns": exclude_columns,
        "target_precision": max(precision_targets),
        "reported_precisions": precision_targets,
        "min_runs_above_threshold": min_runs_above_threshold,
        "plot_everything": False,
        "write_files": write_files,
        "output_dir": output_dir,
        "plot_output_dir": None,
        "verbose": verbose,
    }
    if (
        progress_callback is not None
        and "progress_callback" in inspect.signature(pseudo_unknown_run_fn).parameters
    ):
        pseudo_unknown_kwargs["progress_callback"] = progress_callback
    pseudo_unknown_result = pseudo_unknown_run_fn(**pseudo_unknown_kwargs)
    return pseudo_unknown_result, top1_source_count


def perturbative_triple_run_with_resolvedness_workflow(
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
    pairwise_plot_order: list[Any] | None = None,
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
    progress_callback=None,
    # Optional function overrides for dependency injection.
    perturbative_simple_run_fn=None,
    pseudo_unknown_run_fn=None,
) -> dict[str, Any] | pd.DataFrame:
    """
    Run perturbative triple correlation plus Top-1 pseudo-unknown resolvedness calibration.

    This workflow orchestrates:
    1. Perturbative ensemble runs for each of three models (agglomerative, kmeans, gaussian)
    2. Top-1 candidate selection based on DIHS frequency and mean values
    3. Pseudo-unknown calibration on the selected Top-1 class for each model
    4. Resolvedness threshold computation at target precision levels
    5. Integration and visualization of results

    Parameters
    ----------
    df : pd.DataFrame
        Input data with samples (rows) and chemical/trace elements (columns)
    transform_type : str, default "clr"
        Data transformation: "none", "ilr", "clr", or "scaled"
    unknown_sample : Any, default 0
        Value representing the unknown sample class
    class_column : str, default "controlcode"
        Column name containing sample class labels
    random_state : int | None, default None
        Random seed for reproducibility
    n_iterations : int, default 100
        Number of perturbative iterations per model
    major_cols : list[str] | None, default None
        Column names for major elements; uses defaults if None
    trace_cols : list[str] | None, default None
        Column names for trace elements; uses defaults if None
    major_error : float, default 0.02
        Perturbation error magnitude for major elements
    trace_error : float, default 0.10
        Perturbation error magnitude for trace elements
    perturbation_seed : int | None, default None
        Random seed for perturbation generation
    compute_pairwise : bool, default True
        Whether to compute pairwise DIHS matrices
    plot_everything : bool, default False
        Whether to generate visualization plots
    write_files : bool, default False
        Whether to save outputs to disk
    output_dir : str, default "./Results_perturbative_triple_resolvedness"
        Root output directory
    plot_output_dir : str | None, default None
        Plot output directory; uses output_dir if None
    max_depth : int, default 100
        Maximum clustering depth
    exclude_columns : tuple, default ()
        Columns to exclude from analysis
    pairwise_plot_order : list[Any] | None, default None
        Custom ordering for pairwise plot visualization
    save_cluster_data : bool, default False
        Whether to save cluster membership data
    save_untransformed : bool, default False
        Whether to save untransformed metrics
    pseudo_unknown_iterations : int, default 100
        Number of pseudo-unknown iterations per model
    pseudo_unknown_sample_size : int | None, default None
        Sample size for pseudo-unknown experiments; inferred from data if None
    pseudo_unknown_random_state : int | None, default None
        Random seed for pseudo-unknown experiments
    target_precisions : list[float] | None, default None
        Target precision levels for threshold computation; defaults to [0.95, 0.90, 0.85, 0.80, 0.75]
    min_runs_above_threshold : int, default 1
        Minimum runs required above threshold for valid calibration
    integration_depth : int | None, default None
        Depth level for margin/threshold integration; auto-selects if None
    verbose : bool, default True
        Whether to print progress messages
    return_details : bool, default False
        If True, return full details dict; if False, return summary dataframe only
    perturbative_simple_run_fn : callable | None
        Optional override for perturbative workflow execution.
    pseudo_unknown_run_fn : callable | None
        Optional override for pseudo-unknown workflow execution.

    Returns
    -------
    dict[str, Any] | pd.DataFrame
        If return_details=False: DataFrame with resolvedness summary (one row per model)
        If return_details=True: Dict with "summary", "models", "unknown_class",
            "pseudo_unknown_sample_size", "target_precisions", "artifacts"
    """
    if perturbative_simple_run_fn is None:
        perturbative_simple_run_fn = perturbative_simple_run_workflow

    if pseudo_unknown_run_fn is None:
        pseudo_unknown_run_fn = run_pseudo_unknown_experiments

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
    total_models = len(SUPPORTED_MODELS)
    perturbative_phase_fraction = 0.58

    _log(
        verbose,
        "Starting perturbative triple run with Top-1 pseudo-unknown resolvedness calibration...",
    )
    _log(
        verbose,
        f"Unknown class resolved to: {unknown_class} | Pseudo-unknown sample size: {pseudo_sample_size}",
    )

    _emit_progress(
        progress_callback,
        stage="resolvedness",
        message="Starting perturbative triple run with resolvedness calibration",
        current=0,
        total=total_models,
        fraction=0.0,
    )

    def _make_model_phase_progress(
        *,
        model_index: int,
        model_name: str,
        phase_start: float,
        phase_span: float,
        default_stage: str,
        default_message: str,
    ):
        if progress_callback is None:
            return None

        def _callback(payload: dict[str, Any]) -> None:
            local_fraction = payload.get("fraction")
            current = payload.get("current")
            total = payload.get("total")
            if local_fraction is None and current is not None and total:
                local_fraction = float(current) / float(total)
            if local_fraction is None:
                local_fraction = 0.0
            overall_fraction = (
                model_index + phase_start + (phase_span * min(max(float(local_fraction), 0.0), 1.0))
            ) / float(total_models)
            message = str(payload.get("message", default_message))
            if not message.lower().startswith(f"{model_name}:".lower()):
                message = f"{model_name}: {message}"
            _emit_progress(
                progress_callback,
                stage=str(payload.get("stage", default_stage)),
                message=message,
                current=current,
                total=total,
                fraction=overall_fraction,
            )

        return _callback

    for model_index, model in enumerate(SUPPORTED_MODELS):
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

        perturbative_kwargs = {
            "df": work_df,
            "model_type": model,
            "transform_type": transform_type,
            "unknown_sample": unknown_class,
            "class_column": class_column,
            "random_state": random_state,
            "n_iterations": n_iterations,
            "major_cols": major_cols,
            "trace_cols": trace_cols,
            "major_error": major_error,
            "trace_error": trace_error,
            "perturbation_seed": perturbation_seed,
            "compute_pairwise": compute_pairwise,
            "plot_everything": plot_everything,
            "write_files": write_files,
            "output_dir": perturbative_output_dir,
            "plot_output_dir": perturbative_plot_dir,
            "max_depth": max_depth,
            "exclude_columns": exclude_columns,
            "pairwise_plot_order": pairwise_plot_order,
            "save_cluster_data": save_cluster_data,
            "save_untransformed": save_untransformed,
            "verbose": verbose,
        }
        perturbative_params = inspect.signature(perturbative_simple_run_fn).parameters
        if "progress_callback" in perturbative_params:
            perturbative_kwargs["progress_callback"] = _make_model_phase_progress(
                model_index=model_index,
                model_name=model,
                phase_start=0.0,
                phase_span=perturbative_phase_fraction,
                default_stage="perturbative_iterations",
                default_message="Perturbative ensemble",
            )
        if integration_depth is not None:
            if "integration_depth" in perturbative_params:
                perturbative_kwargs["integration_depth"] = int(integration_depth)
            elif "pairwise_integration_depth" in perturbative_params:
                perturbative_kwargs["pairwise_integration_depth"] = int(integration_depth)

        perturbative_result = perturbative_simple_run_fn(**perturbative_kwargs)
        perturbative_common_depth = perturbative_result["common_depth_level"]
        pseudo_df = work_df.loc[
            ~_class_match_mask(work_df[class_column], unknown_class)
        ].copy()
        if integration_depth is not None:
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

            _apply_perturbative_outputs_at_depth(
                perturbative_result,
                unknown_class=unknown_class,
                integration_depth=chosen_depth,
            )
            top1_candidate = _select_top1_candidate_for_calibration(
                perturbative_result,
                unknown_class=unknown_class,
                model=model,
            )
            pseudo_unknown_result, top1_source_count = _run_pseudo_unknown_for_top1(
                pseudo_unknown_run_fn=pseudo_unknown_run_fn,
                pseudo_df=pseudo_df,
                top1_candidate=top1_candidate,
                model=model,
                class_column=class_column,
                transform_type=transform_type,
                pseudo_sample_size=pseudo_sample_size,
                pseudo_unknown_iterations=pseudo_unknown_iterations,
                pseudo_random_state=pseudo_random_state,
                max_depth=max_depth,
                exclude_columns=exclude_columns,
                precision_targets=precision_targets,
                min_runs_above_threshold=min_runs_above_threshold,
                write_files=write_files,
                output_dir=pseudo_output_dir,
                verbose=verbose,
                progress_callback=_make_model_phase_progress(
                    model_index=model_index,
                    model_name=model,
                    phase_start=perturbative_phase_fraction,
                    phase_span=1.0 - perturbative_phase_fraction,
                    default_stage="pseudo_unknown",
                    default_message="Pseudo-unknown calibration",
                ),
            )
            pseudo_common_depth = pseudo_unknown_result["common_depth_level"]
            if pseudo_common_depth is not None and chosen_depth > int(pseudo_common_depth):
                raise ValueError(
                    f"Requested integration_depth={chosen_depth} exceeds the pseudo-unknown "
                    f"common depth {pseudo_common_depth} for model '{model}'."
                )
        else:
            if perturbative_common_depth is None:
                raise ValueError(
                    f"Model '{model}' did not produce a perturbative common integration depth."
                )

            _apply_perturbative_outputs_at_depth(
                perturbative_result,
                unknown_class=unknown_class,
                integration_depth=int(perturbative_common_depth),
            )
            top1_candidate = _select_top1_candidate_for_calibration(
                perturbative_result,
                unknown_class=unknown_class,
                model=model,
            )

            chosen_depth = None
            pseudo_unknown_result = None
            top1_source_count = None
            stable_alignment = False
            max_alignment_attempts = 6
            for attempt_index in range(max_alignment_attempts):
                pseudo_unknown_result, top1_source_count = _run_pseudo_unknown_for_top1(
                    pseudo_unknown_run_fn=pseudo_unknown_run_fn,
                    pseudo_df=pseudo_df,
                    top1_candidate=top1_candidate,
                    model=model,
                    class_column=class_column,
                    transform_type=transform_type,
                    pseudo_sample_size=pseudo_sample_size,
                    pseudo_unknown_iterations=pseudo_unknown_iterations,
                    pseudo_random_state=pseudo_random_state,
                    max_depth=max_depth,
                    exclude_columns=exclude_columns,
                    precision_targets=precision_targets,
                    min_runs_above_threshold=min_runs_above_threshold,
                    write_files=write_files,
                    output_dir=pseudo_output_dir,
                    verbose=verbose,
                    progress_callback=_make_model_phase_progress(
                        model_index=model_index,
                        model_name=model,
                        phase_start=perturbative_phase_fraction
                        + ((1.0 - perturbative_phase_fraction) * (attempt_index / float(max_alignment_attempts))),
                        phase_span=(1.0 - perturbative_phase_fraction)
                        / float(max_alignment_attempts),
                        default_stage="pseudo_unknown",
                        default_message=(
                            f"Pseudo-unknown calibration attempt {attempt_index + 1} "
                            f"of {max_alignment_attempts}"
                        ),
                    ),
                )
                pseudo_common_depth = pseudo_unknown_result["common_depth_level"]
                available_depths = [
                    int(depth)
                    for depth in (perturbative_common_depth, pseudo_common_depth)
                    if depth is not None
                ]
                if not available_depths:
                    raise ValueError(
                        f"Model '{model}' did not produce a common integration depth."
                    )
                next_chosen_depth = min(available_depths)
                _apply_perturbative_outputs_at_depth(
                    perturbative_result,
                    unknown_class=unknown_class,
                    integration_depth=next_chosen_depth,
                )
                next_top1_candidate = _select_top1_candidate_for_calibration(
                    perturbative_result,
                    unknown_class=unknown_class,
                    model=model,
                )
                same_depth = chosen_depth == int(next_chosen_depth)
                same_top1 = (
                    next_top1_candidate["top1_class_key"]
                    == top1_candidate["top1_class_key"]
                )
                chosen_depth = int(next_chosen_depth)
                top1_candidate = next_top1_candidate
                if same_depth and same_top1:
                    stable_alignment = True
                    break

            if not stable_alignment:
                raise RuntimeError(
                    f"Model '{model}' did not converge to a stable Top-1 pseudo-unknown "
                    f"calibration depth alignment."
                )

        pairwise_plot_matches_chosen_depth = (
            perturbative_result.get("pairwise_integration_depth") == int(chosen_depth)
        )
        if compute_pairwise:
            pairwise_depth_iterations = perturbative_result.get(
                "pairwise_depth_matrices_per_iteration"
            )
            if pairwise_depth_iterations and not pairwise_plot_matches_chosen_depth:
                pairwise_mean, pairwise_std = _aggregate_pairwise_iteration_totals_at_depth(
                    pairwise_depth_iterations,
                    integration_depth=chosen_depth,
                )
                perturbative_result["pairwise_total_mean_matrix"] = pairwise_mean
                perturbative_result["pairwise_total_std_matrix"] = pairwise_std
            perturbative_result["pairwise_integration_depth"] = int(chosen_depth)

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
            if (
                compute_pairwise
                and perturbative_result.get("pairwise_total_mean_matrix") is not None
                and not pairwise_plot_matches_chosen_depth
            ):
                pairwise_plot_path = perturbative_result["artifacts"].get(
                    "pairwise_total_mean_plot_path"
                )
                if pairwise_plot_path is None:
                    pairwise_plot_path = os.path.join(
                        perturbative_plot_dir, f"pairwise_total_mean_{model}.svg"
                    )
                plot_pairwise_matrix(
                    matrix=perturbative_result["pairwise_total_mean_matrix"],
                    title=f"Mean pairwise DIHS | {model} | depth {chosen_depth}",
                    output_path=pairwise_plot_path,
                    unknown_class=unknown_class,
                    class_order=pairwise_plot_order,
                )
                perturbative_result["artifacts"][
                    "pairwise_total_mean_plot_path"
                ] = pairwise_plot_path
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
        _emit_progress(
            progress_callback,
            stage="resolvedness_model_complete",
            message=f"{model}: resolvedness calibration complete",
            current=model_index + 1,
            total=total_models,
            fraction=(model_index + 1) / float(total_models),
        )

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
    _emit_progress(
        progress_callback,
        stage="complete",
        message="Resolvedness workflow complete",
        current=total_models,
        total=total_models,
        fraction=1.0,
    )
    if return_details:
        return out
    return out["summary"]
