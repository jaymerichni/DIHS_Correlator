import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_plot_df(
    df: pd.DataFrame,
    value_col: str,
    std_col: Optional[str] = None,
    force_root_one: bool = False,
    max_depth: Optional[int] = None,
    unknown_class=0,
):
    d = df.copy()
    d = d[d["neighbor_unit"].astype(str) != str(unknown_class)]
    if d.empty:
        return d

    if force_root_one:
        d.loc[d["depth_level"] == 0, value_col] = 1.0
        if std_col is not None and std_col in d.columns:
            d.loc[d["depth_level"] == 0, std_col] = 0.0

    if max_depth is not None:
        d = d[d["depth_level"] <= int(max_depth)]

    return d


def plot_hs_curves(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    value_col: str = "harmonic_score",
    std_col: Optional[str] = None,
    with_shade: bool = False,
    max_neighbors: Optional[int] = 12,
    neighbor_order: str = "best_overall",
    max_depth: Optional[int] = None,
    force_root_one: bool = False,
    title: Optional[str] = None,
    unknown_class=0,
):
    """
    Plot HS vs depth as curves.

    """
    if value_col not in df.columns:
        raise ValueError(f"value_col='{value_col}' not found in dataframe.")
    if with_shade and (std_col is None or std_col not in df.columns):
        raise ValueError("with_shade=True requires a valid std_col in dataframe.")

    d = _prepare_plot_df(
        df=df,
        value_col=value_col,
        std_col=std_col,
        force_root_one=force_root_one,
        max_depth=max_depth,
        unknown_class=unknown_class,
    )
    if d.empty:
        raise RuntimeError("No data to plot after filtering.")

    if neighbor_order == "best_overall":
        rank = d.groupby("neighbor_unit")[value_col].mean().sort_values(ascending=False)
        ordered = rank.index.tolist()
    elif neighbor_order == "alphabetical":
        ordered = sorted(d["neighbor_unit"].astype(str).unique().tolist())
    else:
        raise ValueError("neighbor_order must be 'best_overall' or 'alphabetical'.")

    if max_neighbors is not None:
        ordered = ordered[: int(max_neighbors)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    for nu in ordered:
        sub = d[d["neighbor_unit"].astype(str) == str(nu)].sort_values("depth_level")
        if sub.empty:
            continue
        x = sub["depth_level"].to_numpy(dtype=int)
        y = sub[value_col].to_numpy(dtype=float)
        ax.plot(x, y, label=str(nu), linewidth=2.5)
        if with_shade and std_col is not None:
            e = sub[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - e, y + e, alpha=0.20)

    ax.set_xlabel("Depth level")
    if with_shade and std_col is not None:
        ax.set_ylabel("HS (mean +/- SD)")
    else:
        ax.set_ylabel("HS")

    observed_max_depth = int(d["depth_level"].max())
    if max_depth is not None:
        depth_min = 0
        depth_max = min(int(max_depth), observed_max_depth)
    else:
        depth_min = int(d["depth_level"].min())
        depth_max = observed_max_depth

    ax.set_xlim(depth_min, depth_max)
    ax.set_ylim(0, 1)
    ax.margins(x=0, y=0)
    ax.set_xticks(np.arange(depth_min, depth_max + 1, 1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.35)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)
    if title:
        ax.set_title(title)
    ax.legend(title="Neighbor", ncol=2, frameon=True)

    fig.tight_layout()
    if output_path is None:
        plt.show()
        plt.close(fig)
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


