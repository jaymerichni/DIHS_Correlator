import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from DIHS_Correlator.viz.pseudo_unknown import (
    plot_margin_comparison,
    plot_margin_histogram,
)
from DIHS_Correlator.workflows.analysis import (
    _build_perturbative_calibration_outputs,
    _compute_margin_from_hs_metrics,
    _compute_precision_at_threshold,
    _exclude_unknown_neighbor,
    _finite_float_values,
    _flatten_threshold_summary,
    _infer_pseudo_unknown_common_depth,
    _load_pseudo_unknown_calibration_tables,
    _load_pseudo_unknown_margin_inputs,
    _load_perturbative_margin_inputs,
    _normalize_target_precisions,
    _prepare_margin_plot_data_from_outputs,
    _resolve_display_threshold,
    _select_pseudo_unknown_outputs_at_depth,
    _select_top1_candidate_for_calibration,
)


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
        calibrated_runs_path = os.path.join(output_dir, "perturbative_calibrated_runs.csv")
        calibration_summary_path = os.path.join(output_dir, "perturbative_calibration_summary.csv")
        regime_summary_path = os.path.join(output_dir, "perturbative_regime_summary.csv")
        threshold_curve_path = os.path.join(output_dir, "pseudo_unknown_threshold_curve_for_depth.csv")
        target_thresholds_path = os.path.join(output_dir, "pseudo_unknown_target_thresholds_for_depth.csv")

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

    if plot_output_path is not None:
        os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
        warnings.warn(
            "plot_output_path is currently a placeholder; the reporting module does not yet render a plot file.",
            stacklevel=3,
        )

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
