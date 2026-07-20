from typing import Any

import pandas as pd

from DIHS_Correlator.workflows.pseudo_unknown import (
    run_pseudo_unknown_experiments,
)
from DIHS_Correlator.workflows.perturbative import (
    perturbative_simple_run_workflow,
    perturbative_triple_run_workflow,
)
from DIHS_Correlator.workflows.reporting import (
    calibrate_perturbative_resolvedness_from_outputs as _calibrate_perturbative_resolvedness_from_outputs,
    plot_pseudo_unknown_margin_histogram_from_outputs as _plot_pseudo_unknown_margin_histogram_from_outputs,
    plot_pseudo_unknown_margin_from_outputs as _plot_pseudo_unknown_margin_from_outputs,
)
from DIHS_Correlator.workflows.resolvedness import (
    perturbative_triple_run_with_resolvedness_workflow,
)
from DIHS_Correlator.workflows.single_run import (
    run_single_model_workflow,
    triple_run_workflow,
)

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
    pairwise_plot_order: list[Any] | None = None,
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
):
    result = run_single_model_workflow(
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
        pairwise_plot_order=pairwise_plot_order,
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
    pairwise_plot_order: list[Any] | None = None,
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
):
    """Run agglomerative, kmeans and gaussian under one shared configuration."""
    combined = triple_run_workflow(
        df=df,
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
        pairwise_plot_order=pairwise_plot_order,
        save_cluster_data=save_cluster_data,
        save_untransformed=save_untransformed,
        verbose=verbose,
    )
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
    pairwise_plot_order: list[Any] | None = None,
    integration_depth: int | None = None,
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
    progress_callback=None,
):
    """Perturbative ensemble run for one model."""
    result = perturbative_simple_run_workflow(
        df=df,
        model_type=model_type,
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
        output_dir=output_dir,
        plot_output_dir=plot_output_dir,
        max_depth=max_depth,
        exclude_columns=exclude_columns,
        pairwise_plot_order=pairwise_plot_order,
        integration_depth=integration_depth,
        save_cluster_data=save_cluster_data,
        save_untransformed=save_untransformed,
        verbose=verbose,
        progress_callback=progress_callback,
    )
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
    pairwise_plot_order: list[Any] | None = None,
    integration_depth: int | None = None,
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
    return_details: bool = False,
    progress_callback=None,
):
    """Perturbative ensemble run for all three models."""
    result = perturbative_triple_run_workflow(
        df=df,
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
        output_dir=output_dir,
        plot_output_dir=plot_output_dir,
        max_depth=max_depth,
        exclude_columns=exclude_columns,
        pairwise_plot_order=pairwise_plot_order,
        integration_depth=integration_depth,
        save_cluster_data=save_cluster_data,
        save_untransformed=save_untransformed,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    if return_details:
        return result
    return result["hs_mean_per_depth"]


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
    progress_callback=None,
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
        progress_callback=progress_callback,
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
    return _plot_pseudo_unknown_margin_from_outputs(
        pseudo_unknown_output_dir=pseudo_unknown_output_dir,
        perturbative_output_dir=perturbative_output_dir,
        integration_depth=integration_depth,
        target_precision=target_precision,
        threshold_mode=threshold_mode,
        output_path=output_path,
        title=title,
        verbose=verbose,
        return_details=return_details,
    )


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
    return _plot_pseudo_unknown_margin_histogram_from_outputs(
        pseudo_unknown_output_dir=pseudo_unknown_output_dir,
        perturbative_output_dir=perturbative_output_dir,
        integration_depth=integration_depth,
        target_precision=target_precision,
        threshold_mode=threshold_mode,
        bins=bins,
        output_path=output_path,
        title=title,
        verbose=verbose,
        return_details=return_details,
    )


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
    return _calibrate_perturbative_resolvedness_from_outputs(
        pseudo_unknown_output_dir=pseudo_unknown_output_dir,
        perturbative_output_dir=perturbative_output_dir,
        integration_depth=integration_depth,
        target_precisions=target_precisions,
        bins=bins,
        output_dir=output_dir,
        plot_output_path=plot_output_path,
        title=title,
        verbose=verbose,
        return_details=return_details,
    )


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
):
    """Run perturbative triple correlation plus Top-1 pseudo-unknown resolvedness calibration."""
    result = perturbative_triple_run_with_resolvedness_workflow(
        df=df,
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
        output_dir=output_dir,
        plot_output_dir=plot_output_dir,
        max_depth=max_depth,
        exclude_columns=exclude_columns,
        pairwise_plot_order=pairwise_plot_order,
        save_cluster_data=save_cluster_data,
        save_untransformed=save_untransformed,
        pseudo_unknown_iterations=pseudo_unknown_iterations,
        pseudo_unknown_sample_size=pseudo_unknown_sample_size,
        pseudo_unknown_random_state=pseudo_unknown_random_state,
        target_precisions=target_precisions,
        min_runs_above_threshold=min_runs_above_threshold,
        integration_depth=integration_depth,
        verbose=verbose,
        return_details=return_details,
        progress_callback=progress_callback,
    )
    if return_details:
        return result
    return result["summary"]

