import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pseudo_unknown_margin_comparison(
    comparison_runs: pd.DataFrame,
    output_path: str | None = None,
):
    if comparison_runs.empty:
        return

    df = comparison_runs.copy()
    df = df[df["margin"].notna()].copy()
    if df.empty:
        return

    methods = list(df["method"].drop_duplicates())
    cases = ["positive", "negative"]
    width = 0.35
    spacing = 1.2

    fig, ax = plt.subplots(figsize=(max(10, 2.8 * len(methods)), 6))
    positions = []
    labels = []

    for idx, method in enumerate(methods):
        center = idx * spacing
        for offset, case in zip((-width / 2.0, width / 2.0), cases):
            sub = df[(df["method"] == method) & (df["case"] == case)]["margin"].to_numpy()
            if sub.size == 0:
                continue
            pos = center + offset
            positions.append(pos)
            labels.append((method, case))
            ax.boxplot(
                sub,
                positions=[pos],
                widths=width * 0.85,
                patch_artist=True,
                boxprops={
                    "facecolor": "#d95f02" if case == "positive" else "#1b9e77",
                    "alpha": 0.55,
                },
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
            )

    ax.set_xticks([idx * spacing for idx in range(len(methods))])
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Margin")
    ax.set_title("Pseudo-Unknown Margin Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [
        plt.Line2D([0], [0], color="#d95f02", lw=10, alpha=0.55, label="positive"),
        plt.Line2D([0], [0], color="#1b9e77", lw=10, alpha=0.55, label="negative"),
    ]
    ax.legend(handles=legend_handles, frameon=True)

    fig.tight_layout()
    if output_path is None:
        plt.show()
        plt.close(fig)
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", format="svg")
    plt.close(fig)
