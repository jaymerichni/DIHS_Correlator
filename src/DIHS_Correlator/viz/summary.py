import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricSummaries:
    def plot_total_harmonic_comparison(
        self,
        agglomerative_csv: str,
        gaussian_csv: str,
        kmeans_csv: str,
        legend_path: str | None = None,
        unknown_class: int = 0,
        output_path: str | None = None,
        desired_letter_order: list[str] | None = None,
    ):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 18

        agglom_df = pd.read_csv(agglomerative_csv)
        gauss_df = pd.read_csv(gaussian_csv)
        kmeans_df = pd.read_csv(kmeans_csv)

        if "unknown_class" in agglom_df.columns:
            agglom_df = agglom_df[agglom_df["unknown_class"] == unknown_class]
        if "unknown_class" in gauss_df.columns:
            gauss_df = gauss_df[gauss_df["unknown_class"] == unknown_class]
        if "unknown_class" in kmeans_df.columns:
            kmeans_df = kmeans_df[kmeans_df["unknown_class"] == unknown_class]

        agglom_df = agglom_df[agglom_df["neighbor_unit"].astype(str) != "0"]
        gauss_df = gauss_df[gauss_df["neighbor_unit"].astype(str) != "0"]
        kmeans_df = kmeans_df[kmeans_df["neighbor_unit"].astype(str) != "0"]

        if legend_path is not None:
            legend_df = pd.read_csv(legend_path)
            code_to_label = dict(zip(legend_df["controlcode"], legend_df["unit"]))
        else:
            legend_df = None
            code_to_label = {}

        if desired_letter_order is not None:
            if legend_df is None:
                raise ValueError("desired_letter_order requires legend_path.")
            label_to_code = {v: int(k) for k, v in code_to_label.items()}
            ordered_units = []
            for lbl in desired_letter_order:
                code = label_to_code.get(lbl)
                if code is not None and code != 0 and (agglom_df["neighbor_unit"] == code).any():
                    ordered_units.append(code)
        else:
            ordered_units = agglom_df.sort_values("total_product", ascending=False)[
                "neighbor_unit"
            ].tolist()

        x_letter_labels = [code_to_label.get(code, str(code)) for code in ordered_units]

        def _align(df, mean_col, std_col=None):
            df = df[df["neighbor_unit"].isin(ordered_units)].copy()
            df = df.set_index("neighbor_unit").reindex(ordered_units)
            y_mean = df[mean_col].values
            y_std = df[std_col].values if std_col is not None else None
            return y_mean, y_std

        agglom_y, _ = _align(agglom_df, "total_product")
        gauss_mean, gauss_std = _align(gauss_df, "total_product_mean", "total_product_std")
        kmeans_mean, kmeans_std = _align(kmeans_df, "total_product_mean", "total_product_std")

        stacked = np.vstack([gauss_mean, kmeans_mean, agglom_y])
        combined_mean = stacked.mean(axis=0)
        combined_std = stacked.std(axis=0, ddof=0)

        fig, ax = plt.subplots(figsize=(14, 8))
        n = len(ordered_units)
        x = np.arange(n)
        width = 0.2

        ax.bar(
            x - width,
            gauss_mean,
            width=width,
            yerr=gauss_std,
            capsize=6,
            alpha=0.75,
            color="#798a73",
            label="Gaussian (100 iter, mean +/- SD)",
        )
        ax.bar(
            x,
            kmeans_mean,
            width=width,
            yerr=kmeans_std,
            capsize=6,
            alpha=0.75,
            color="#e0dac9",
            label="KMeans (100 iter, mean +/- SD)",
        )
        ax.bar(
            x + width,
            agglom_y,
            width=width,
            alpha=0.9,
            color="#3f403c",
            label="Ward-Agglomerative",
        )
        ax.errorbar(
            x,
            combined_mean,
            yerr=combined_std,
            fmt="--",
            color="red",
            ecolor="red",
            capsize=5,
            linewidth=2,
            marker="o",
            markersize=6,
            label="Mean +/- SD (Gaussian, KMeans, Agglomerative)",
        )

        ax.margins(x=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(axis="y", labelsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(x_letter_labels, rotation=45, ha="right", fontsize=20)
        ax.set_xlabel("Potential sources", fontsize=22)
        ax.set_ylabel("Total harmonic score", fontsize=22)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=16)
        fig.tight_layout()

        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    def plot_dataset_comparison_per_neighbor(
        self,
        base_results_dir: str,
        legend_path: str | None = None,
        unknown_class: int = 0,
        output_path: str | None = None,
        desired_letter_order: list[str] | None = None,
    ):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 18
        base_results_dir = Path(base_results_dir)
        if legend_path is not None:
            legend_df = pd.read_csv(legend_path)
            code_to_label = dict(zip(legend_df["controlcode"], legend_df["unit"]))
            all_neighbors = [int(c) for c in legend_df["controlcode"].unique() if int(c) != 0]
        else:
            legend_df = None
            code_to_label = {}
            all_neighbors = []

        if desired_letter_order is not None:
            if legend_df is None:
                raise ValueError("desired_letter_order requires legend_path.")
            label_to_code = {v: int(k) for k, v in code_to_label.items()}
            neighbor_units = [label_to_code[lbl] for lbl in desired_letter_order if label_to_code.get(lbl) in all_neighbors]
        else:
            if legend_df is not None:
                neighbor_units = sorted(all_neighbors)
            else:
                # Infer neighbor units directly from files if no legend is provided.
                neighbor_set = set()
                probe_specs = [
                    ("majors", "clr"),
                    ("traces", "clr"),
                    ("ratios", "scaled"),
                ]
                for subdir, transform_name in probe_specs:
                    dataset_root = base_results_dir / subdir
                    for model in ["agglomerative", "kmeans", "gaussian"]:
                        csv_path = dataset_root / "Trees" / f"{transform_name}_{model}" / f"total_metrics_{model}_0.csv"
                        if not csv_path.exists():
                            continue
                        d = pd.read_csv(csv_path)
                        if "unknown_class" in d.columns:
                            d = d[d["unknown_class"] == unknown_class]
                        if "neighbor_unit" in d.columns:
                            neighbor_set.update(d["neighbor_unit"].astype(str).tolist())
                neighbor_set.discard("0")
                neighbor_units = sorted(neighbor_set)

        dataset_specs = [
            ("Major elements", "majors", "clr"),
            ("Trace elements", "traces", "clr"),
            ("Ratios", "ratios", "scaled"),
        ]
        models = ["agglomerative", "kmeans", "gaussian"]

        means = {lbl: {u: np.nan for u in neighbor_units} for lbl, _, _ in dataset_specs}
        stds = {lbl: {u: 0.0 for u in neighbor_units} for lbl, _, _ in dataset_specs}

        for dataset_label, subdir, transform_name in dataset_specs:
            dataset_root = base_results_dir / subdir
            for neighbor in neighbor_units:
                vals = []
                for model in models:
                    csv_path = dataset_root / "Trees" / f"{transform_name}_{model}" / f"total_metrics_{model}_0.csv"
                    if not csv_path.exists():
                        continue
                    df = pd.read_csv(csv_path)
                    if "unknown_class" in df.columns:
                        df = df[df["unknown_class"] == unknown_class]
                    df = df[df["neighbor_unit"].astype(str) == str(neighbor)]
                    if df.empty:
                        continue
                    vals.append(float(df["total_product"].iloc[0]))
                if vals:
                    arr = np.array(vals, dtype=float)
                    means[dataset_label][neighbor] = arr.mean()
                    stds[dataset_label][neighbor] = arr.std(ddof=0) if len(arr) > 1 else 0.0
                else:
                    means[dataset_label][neighbor] = 0.0
                    stds[dataset_label][neighbor] = 0.0

        sorted_neighbors = neighbor_units
        n = len(sorted_neighbors)
        x = np.arange(n)
        majors_mean = np.array([means["Major elements"][u] for u in sorted_neighbors], dtype=float)
        majors_std = np.array([stds["Major elements"][u] for u in sorted_neighbors], dtype=float)
        traces_mean = np.array([means["Trace elements"][u] for u in sorted_neighbors], dtype=float)
        traces_std = np.array([stds["Trace elements"][u] for u in sorted_neighbors], dtype=float)
        ratios_mean = np.array([means["Ratios"][u] for u in sorted_neighbors], dtype=float)
        ratios_std = np.array([stds["Ratios"][u] for u in sorted_neighbors], dtype=float)
        stacked = np.vstack([majors_mean, traces_mean, ratios_mean])
        combined_mean = np.nanmean(stacked, axis=0)
        combined_std = stacked.std(axis=0, ddof=0)
        x_labels = [code_to_label.get(u, str(u)) for u in sorted_neighbors]

        fig, ax = plt.subplots(figsize=(14, 6))
        width = 0.22
        ax.bar(x - width, majors_mean, width=width, yerr=majors_std, capsize=5, alpha=0.8, color="#1f77b4", label="Major elements (mean +/- SD across models)")
        ax.bar(x, traces_mean, width=width, yerr=traces_std, capsize=5, alpha=0.8, color="#ff7f0e", label="Trace elements (mean +/- SD across models)")
        ax.bar(x + width, ratios_mean, width=width, yerr=ratios_std, capsize=5, alpha=0.8, color="#2ca02c", label="Ratios (mean +/- SD across models)")
        ax.errorbar(x, combined_mean, yerr=combined_std, fmt="--", color="red", ecolor="red", capsize=5, linewidth=2, marker="o", markersize=6, label="Mean(M, T, R) +/- SD")

        ax.margins(x=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(axis="y", labelsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=20)
        ax.set_xlabel("Potential sources", fontsize=22)
        ax.set_ylabel("Total harmonic score", fontsize=22)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=15)
        fig.tight_layout()

        if output_path is not None:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
