import os
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DIHS_Correlator.viz.hs_curves import plot_hs_curves
from DIHS_Correlator.viz.pairwise import plot_pairwise_matrix
from DIHS_Correlator.workflows.single_run import (
    SUPPORTED_MODELS,
    run_single_model_workflow,
)

from DIHS_Correlator.workflows.utils import (
    _log,
    _print_progress,
    _resolve_major_trace_columns,
    _resolve_unknown_class,
)


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


def _compute_top1_stats(dihs_iterations: pd.DataFrame, unknown_class: Any):
    dihs_iterations = dihs_iterations[
        dihs_iterations["neighbor_unit"].astype(str) != str(unknown_class)
    ].copy()
    if dihs_iterations.empty:
        return pd.DataFrame(
            columns=["neighbor_unit", "wins", "top1_fraction", "n_iterations"]
        )

    candidates = (
        dihs_iterations["neighbor_unit"].astype(str).drop_duplicates().sort_values().tolist()
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
    dihs_iterations = dihs_iterations[
        dihs_iterations["neighbor_unit"].astype(str) != str(unknown_class)
    ].copy()
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


def _aggregate_pairwise_iteration_totals(pairwise_totals: Iterable[pd.DataFrame]):
    matrices = list(pairwise_totals)
    if not matrices:
        return None, None
    all_units = []
    for mat in matrices:
        if mat is None:
            continue
        all_units.extend([str(x) for x in mat.index.tolist()])
    all_units = sorted(set(all_units), key=lambda s: (0, s) if s.isdigit() else (1, s))

    stack = []
    for mat in matrices:
        m = mat.copy()
        m.index = m.index.astype(str)
        m.columns = m.columns.astype(str)
        aligned = m.reindex(index=all_units, columns=all_units).fillna(0.0)
        aligned = 0.5 * (aligned + aligned.T)
        arr = aligned.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(arr, 1.0)
        stack.append(arr)
    arr = np.stack(stack, axis=0)
    mean_df = pd.DataFrame(arr.mean(axis=0), index=all_units, columns=all_units)
    std_df = pd.DataFrame(arr.std(axis=0, ddof=0), index=all_units, columns=all_units)
    return mean_df, std_df


def _compute_pairwise_total_matrix_at_depth(
    pairwise_depth_matrices: dict[int, pd.DataFrame] | None,
    integration_depth: int,
):
    if not pairwise_depth_matrices:
        return None

    depth = int(integration_depth)
    if depth < 0:
        raise ValueError("integration_depth must be >= 0.")

    depth_lookup = {int(key): value for key, value in pairwise_depth_matrices.items()}
    if depth_lookup and depth > max(depth_lookup):
        raise ValueError(
            f"Requested integration_depth={depth} exceeds the available pairwise depth "
            f"{max(depth_lookup)}."
        )

    selected_matrices = []
    all_units = []
    for current_depth in range(depth + 1):
        if current_depth not in depth_lookup:
            raise ValueError(
                f"Missing pairwise matrix for depth {current_depth} while recomputing "
                f"the pairwise DIHS total at integration_depth={depth}."
            )
        matrix = depth_lookup[current_depth].copy()
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        all_units.extend(matrix.index.tolist())
        all_units.extend(matrix.columns.tolist())
        selected_matrices.append(matrix)

    if not selected_matrices:
        return None

    all_units = sorted(set(all_units), key=lambda s: (0, s) if s.isdigit() else (1, s))
    stack = []
    for matrix in selected_matrices:
        aligned = matrix.reindex(index=all_units, columns=all_units).fillna(0.0)
        aligned = 0.5 * (aligned + aligned.T)
        arr = aligned.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(arr, 1.0)
        stack.append(arr)

    total_from_depth = np.stack(stack, axis=0).mean(axis=0)
    total_matrix = pd.DataFrame(total_from_depth, index=all_units, columns=all_units)
    total_matrix = 0.5 * (total_matrix + total_matrix.T)
    arr = total_matrix.to_numpy(copy=True)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=total_matrix.index, columns=total_matrix.columns)


def _aggregate_pairwise_iteration_totals_at_depth(
    pairwise_depth_iterations: Iterable[dict[int, pd.DataFrame] | None],
    integration_depth: int,
):
    totals = []
    for pairwise_depth_matrices in pairwise_depth_iterations:
        total_matrix = _compute_pairwise_total_matrix_at_depth(
            pairwise_depth_matrices=pairwise_depth_matrices,
            integration_depth=integration_depth,
        )
        if total_matrix is not None:
            totals.append(total_matrix)
    return _aggregate_pairwise_iteration_totals(totals)


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


def perturbative_simple_run_workflow(
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
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
):
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
    pairwise_depth_iterations = []
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

        run = run_single_model_workflow(
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
            pairwise_plot_order=pairwise_plot_order,
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

        if compute_pairwise and run["pairwise_per_depth_matrices"] is not None:
            pairwise_depth_iterations.append(run["pairwise_per_depth_matrices"])
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
    pairwise_integration_depth = common_depth_level if compute_pairwise else None
    if compute_pairwise and pairwise_depth_iterations and pairwise_integration_depth is not None:
        pairwise_mean, pairwise_std = _aggregate_pairwise_iteration_totals_at_depth(
            pairwise_depth_iterations,
            integration_depth=pairwise_integration_depth,
        )
    else:
        pairwise_mean, pairwise_std = None, None

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
                title=f"Mean pairwise DIHS | {model_type} | depth {pairwise_integration_depth}",
                output_path=pairwise_plot_path,
                unknown_class=unknown_class,
                class_order=pairwise_plot_order,
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
        "pairwise_integration_depth": pairwise_integration_depth,
        "pairwise_depth_matrices_per_iteration": (
            pairwise_depth_iterations if compute_pairwise else None
        ),
        "artifacts": artifacts,
    }
    _log(verbose, "Perturbative run completed.")
    return result


def perturbative_triple_run_workflow(
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
    save_cluster_data: bool = False,
    save_untransformed: bool = False,
    verbose: bool = True,
):
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
        model_results[model] = perturbative_simple_run_workflow(
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
            pairwise_plot_order=pairwise_plot_order,
            save_cluster_data=save_cluster_data,
            save_untransformed=save_untransformed,
            verbose=verbose,
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
    return {
        "hs_mean_per_depth": hs_mean_all,
        "dihs_summary": dihs_summary_all,
        "top1_frequency": top1_all,
        "models": model_results,
    }
