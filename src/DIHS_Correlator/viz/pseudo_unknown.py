import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CASE_COLORS = {
    "positive": "#2E8B57",
    "failure": "#C05A11",
    "perturbative": "#1F4E79",
}


def _save_or_show(fig, output_path: Optional[str] = None):
    fig.tight_layout()
    if output_path is None:
        plt.show()
        plt.close(fig)
        return
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _style_boxplot(bp, color: str, alpha: float, zorder: float):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor(color)
        patch.set_linewidth(2.0)
        patch.set_zorder(zorder)

    for line_group in ("whiskers", "caps", "medians"):
        for line in bp[line_group]:
            line.set_color(color)
            line.set_linewidth(1.8 if line_group != "medians" else 2.2)
            line.set_zorder(zorder + 0.1)


def _draw_boxplot(ax, values, position: float, width: float, color: str, alpha: float, zorder: float):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False

    bp = ax.boxplot(
        [vals],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    _style_boxplot(bp, color=color, alpha=alpha, zorder=zorder)
    return True


def _draw_horizontal_density_band(
    ax,
    values,
    x_min: float,
    x_max: float,
    color: str,
    zorder: float,
    n_bins: int = 45,
    max_alpha: float = 0.24,
    smooth_points: int = 480,
):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False

    y_min = float(vals.min())
    y_max = float(vals.max())
    if np.isclose(y_min, y_max):
        pad = max(1e-3, abs(y_min) * 0.02 + 1e-3)
        y_min -= pad
        y_max += pad

    n_bins = max(40, int(n_bins))
    counts, edges = np.histogram(vals, bins=n_bins, range=(y_min, y_max), density=False)

    # Smooth the raw histogram and then interpolate it to a denser vertical grid
    # so the rendered band looks continuous rather than visibly binned.
    kernel = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    if counts.size >= kernel.size:
        counts = np.convolve(counts.astype(float), kernel, mode="same")
    else:
        counts = counts.astype(float)

    peak = float(np.max(counts))
    if peak <= 0.0:
        return False

    density = counts / peak
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth_points = max(160, int(smooth_points))
    y_grid = np.linspace(y_min, y_max, smooth_points)
    density = np.interp(y_grid, centers, density, left=0.0, right=0.0)

    fine_kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    fine_kernel /= fine_kernel.sum()
    if density.size >= fine_kernel.size:
        density = np.convolve(density, fine_kernel, mode="same")

    rgba = np.array(to_rgba(color), dtype=float)
    img = np.zeros((density.size, 2, 4), dtype=float)
    img[:, :, :3] = rgba[:3]
    img[:, :, 3] = max_alpha * np.clip(density, 0.0, 1.0)[:, None]

    ax.imshow(
        img,
        extent=[x_min, x_max, y_grid[0], y_grid[-1]],
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=zorder,
    )
    return True


def _extract_margin_groups(
    results_df: pd.DataFrame,
    perturbative_margins: np.ndarray | list[float] | None = None,
):
    valid = results_df[results_df["dihs_margin"].notna()].copy()
    if valid.empty:
        return {}

    groups = {
        "positive": valid[
            (valid["case"] == "positive") & (valid["top1_is_true_source"].astype(bool))
        ]["dihs_margin"].to_numpy(dtype=float),
        "failure": valid[
            ((valid["case"] == "positive") & (~valid["top1_is_true_source"].astype(bool)))
            | (valid["case"] == "negative")
        ]["dihs_margin"].to_numpy(dtype=float),
    }

    if perturbative_margins is not None:
        perturb_vals = np.asarray(perturbative_margins, dtype=float)
        perturb_vals = perturb_vals[np.isfinite(perturb_vals)]
        groups["perturbative"] = perturb_vals

    return groups


def plot_margin_comparison(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    threshold: float | None = None,
    threshold_label: str | None = None,
    title: str | None = None,
    perturbative_margins: np.ndarray | list[float] | None = None,
    integration_depth: int | None = None,
    target_precision: float = 0.95,
):
    if results_df.empty:
        return

    group_values = _extract_margin_groups(
        results_df=results_df,
        perturbative_margins=perturbative_margins,
    )
    if not group_values:
        return

    positions = {
        "positive": 1.0,
        "failure": 2.0,
    }
    label_map = {
        "positive": "True Positives (box plot)",
        "failure": "False Positives/Out-Of-Distribution (box plot)",
        "perturbative": "ΔDIHS of Perturbative Runs (density band)",
    }
    perturb_vals = group_values.get("perturbative")

    if title is None:
        title = "Pseudo-Unknown DIHS Margin Comparison"
        if integration_depth is not None:
            title = f"{title} | depth {integration_depth}"

    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(10, 9))
    legend_handles = []

    if perturb_vals is not None and perturb_vals.size > 0:
        drawn = _draw_horizontal_density_band(
            ax=ax,
            values=perturb_vals,
            x_min=0.4,
            x_max=2.6,
            color=CASE_COLORS["perturbative"],
            zorder=0.5,
        )
        if drawn:
            legend_handles.append(
                Patch(
                    facecolor=CASE_COLORS["perturbative"],
                    edgecolor=CASE_COLORS["perturbative"],
                    alpha=0.24,
                    label=label_map["perturbative"],
                )
            )

    for group_name in ("positive", "failure"):
        drawn = _draw_boxplot(
            ax=ax,
            values=group_values[group_name],
            position=positions[group_name],
            width=0.55,
            color=CASE_COLORS[group_name],
            alpha=0.35,
            zorder=3.0,
        )
        if drawn:
            legend_handles.append(
                Patch(
                    facecolor=CASE_COLORS[group_name],
                    edgecolor=CASE_COLORS[group_name],
                    alpha=0.35,
                    label=label_map[group_name],
                )
            )

    if threshold is not None and np.isfinite(threshold):
        threshold_pct = int(round(100.0 * float(target_precision)))
        legend_text = (
            threshold_label
            if threshold_label is not None
            else f"{threshold_pct}% threshold = {threshold:.3f}"
        )
        ax.axhline(
            threshold,
            color="#333333",
            linestyle="--",
            linewidth=1.8,
            label=legend_text,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="--",
                linewidth=1.8,
                label=legend_text,
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, frameon=True, loc="best", fontsize=20)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(
        [label_map[key] for key in ("positive", "failure")],
        fontsize=18
    )

    ax.set_ylabel("DIHS margin", fontsize=24)
    ax.set_title(title, fontsize=26)

    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22)

    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.4, 2.6)

    _save_or_show(fig, output_path=output_path)


def plot_margin_histogram(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    threshold: float | None = None,
    threshold_label: str | None = None,
    title: str | None = None,
    perturbative_margins: np.ndarray | list[float] | None = None,
    integration_depth: int | None = None,
    target_precision: float = 0.95,
    bins: int = 10,
):
    if results_df.empty:
        return

    groups = _extract_margin_groups(
        results_df=results_df,
        perturbative_margins=perturbative_margins,
    )
    if not groups:
        return

    ordered = ("positive", "failure", "perturbative")
    label_map = {
        "positive": "Positive",
        "failure": "False Positives or Out-Of-Distribution",
        "perturbative": "Perturbative",
    }

    available = [groups[key] for key in ordered if key in groups and groups[key].size > 0]
    if not available:
        return

    all_vals = np.concatenate(available)
    x_min = max(0.0, float(all_vals.min()) - 0.02)
    x_max = float(all_vals.max()) + 0.02
    if np.isclose(x_min, x_max):
        x_max = x_min + 0.05
    edges = np.linspace(x_min, x_max, max(10, int(bins)) + 1)

    if title is None:
        title = "Pseudo-Unknown DIHS Margin Distributions"
        if integration_depth is not None:
            title = f"{title} | depth {integration_depth}"

    fig, ax = plt.subplots(figsize=(9, 6))
    for key in ordered:
        if key not in groups:
            continue
        vals = groups[key]
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=edges,
            density=True,
            alpha=0.24,
            color=CASE_COLORS[key],
            histtype="stepfilled",
            label=label_map[key],
        )
        ax.hist(
            vals,
            bins=edges,
            density=True,
            color=CASE_COLORS[key],
            histtype="step",
            linewidth=1.8,
        )

    if threshold is not None and np.isfinite(threshold):
        threshold_pct = int(round(100.0 * float(target_precision)))
        legend_text = (
            threshold_label
            if threshold_label is not None
            else f"{threshold_pct}% threshold = {threshold:.3f}"
        )
        ax.axvline(
            threshold,
            color="#333333",
            linestyle="--",
            linewidth=1.8,
            label=legend_text,
        )

    ax.set_xlabel("DIHS margin")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.set_xlim(left=0.0)
    ax.legend(frameon=True, loc="best")

    _save_or_show(fig, output_path=output_path)


def plot_perturbative_calibration_overlay(
    threshold_curve: pd.DataFrame,
    perturbative_runs: pd.DataFrame,
    pseudo_results: pd.DataFrame | None = None,
    output_path: Optional[str] = None,
    title: str | None = None,
    target_thresholds: pd.DataFrame | None = None,
    bins: int = 14,
):
    if threshold_curve.empty or perturbative_runs.empty:
        return

    curve = threshold_curve.sort_values("threshold").copy()
    groups = {}
    if pseudo_results is not None and not pseudo_results.empty:
        groups = _extract_margin_groups(results_df=pseudo_results)

    ordered_density = []
    positive_vals = groups.get("positive")
    failure_vals = groups.get("failure")
    perturbative_vals = perturbative_runs["dihs_margin"].dropna().to_numpy(dtype=float)
    if positive_vals is not None and positive_vals.size > 0:
        ordered_density.append(("positive", positive_vals, "Positive"))
    if failure_vals is not None and failure_vals.size > 0:
        ordered_density.append(
            ("failure", failure_vals, "False Positives or Out-Of-Distribution")
        )
    if perturbative_vals.size > 0:
        ordered_density.append(("perturbative", perturbative_vals, "Perturbative"))

    if not ordered_density:
        return

    if title is None:
        title = "Perturbative Margins vs Pseudo-Unknown Calibration"

    all_vals = np.concatenate([vals for _, vals, _ in ordered_density])
    x_min = max(0.0, min(float(curve["threshold"].min()), float(all_vals.min())) - 0.02)
    x_max = max(float(curve["threshold"].max()), float(all_vals.max())) + 0.02
    if np.isclose(x_min, x_max):
        x_max = x_min + 0.05
    edges = np.linspace(x_min, x_max, max(8, int(bins)) + 1)

    mean_margin = float(np.mean(perturbative_vals)) if perturbative_vals.size > 0 else np.nan

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    ax2 = ax1.twinx()

    ax1.fill_between(
        curve["threshold"].to_numpy(dtype=float),
        0.0,
        curve["precision"].to_numpy(dtype=float),
        color="#1f4e79",
        alpha=0.08,
        zorder=0.5,
    )
    ax1.plot(
        curve["threshold"],
        curve["precision"],
        color="#1f4e79",
        linewidth=2.4,
        label="Pseudo-unknown precision",
        zorder=3.0,
    )

    density_patches = []
    for key, vals, label in ordered_density:
        alpha = 0.22 if key == "perturbative" else 0.12
        line = 1.9 if key == "perturbative" else 1.4
        ax2.hist(
            vals,
            bins=edges,
            density=True,
            color=CASE_COLORS[key],
            alpha=alpha,
            histtype="stepfilled",
            zorder=1.0,
        )
        ax2.hist(
            vals,
            bins=edges,
            density=True,
            color=CASE_COLORS[key],
            linewidth=line,
            histtype="step",
            zorder=2.0,
        )
        density_patches.append(
            Patch(
                facecolor=CASE_COLORS[key],
                edgecolor=CASE_COLORS[key],
                alpha=alpha,
                label=f"{label} density",
            )
        )

    if np.isfinite(mean_margin):
        ax1.axvline(
            mean_margin,
            color=CASE_COLORS["perturbative"],
            linestyle="--",
            linewidth=1.8,
            label=f"Perturbative mean = {mean_margin:.3f}",
            zorder=4.0,
        )

    if (
        target_thresholds is not None
        and not target_thresholds.empty
        and {"target_precision", "resolvedness_threshold"}.issubset(target_thresholds.columns)
    ):
        finite_thresholds = target_thresholds[
            target_thresholds["resolvedness_threshold"].notna()
        ].sort_values("target_precision", ascending=False)
        if not finite_thresholds.empty:
            cmap = plt.get_cmap("cividis")
            denom = max(1, len(finite_thresholds) - 1)
            for idx, (_, row) in enumerate(finite_thresholds.iterrows()):
                target = float(row["target_precision"])
                threshold = float(row["resolvedness_threshold"])
                ax1.axvline(
                    threshold,
                    color=cmap(idx / denom),
                    linestyle=":",
                    linewidth=1.5,
                    label=f"{int(round(100.0 * target))}% target = {threshold:.3f}",
                    zorder=3.6,
                )

    ax1.set_xlabel("DIHS margin threshold")
    ax1.set_ylabel("Pseudo-unknown precision", color="#1f4e79")
    ax2.set_ylabel("Margin density", color=CASE_COLORS["perturbative"])
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylim(bottom=0.0)
    ax1.set_xlim(left=0.0)
    ax1.grid(axis="both", linestyle="--", alpha=0.22)
    ax1.set_title(title)

    lines = ax1.get_lines()
    labels = [line.get_label() for line in lines] + [patch.get_label() for patch in density_patches]
    ax1.legend(lines + density_patches, labels, loc="best", frameon=True)

    _save_or_show(fig, output_path=output_path)


def plot_threshold_diagnostics(
    threshold_curve: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Resolvedness Threshold Diagnostics",
    precision_targets: list[float] | None = None,
):
    if threshold_curve.empty:
        return

    df = threshold_curve.sort_values("threshold").copy()

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        df["threshold"],
        df["precision"],
        color="#1f4e79",
        linewidth=2.2,
        label="Precision above threshold",
    )
    ax2.plot(
        df["threshold"],
        df["coverage"],
        color="#c27c0e",
        linewidth=2.2,
        label="Coverage above threshold",
    )

    targets = precision_targets or [0.95]
    for target in targets:
        ax1.axhline(target, color="#777777", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.set_xlabel("DIHS margin threshold")
    ax1.set_ylabel("Precision", color="#1f4e79")
    ax2.set_ylabel("Coverage", color="#c27c0e")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylim(0.0, 1.05)
    ax1.grid(axis="both", linestyle="--", alpha=0.25)
    ax1.set_title(title)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)

    _save_or_show(fig, output_path=output_path)
