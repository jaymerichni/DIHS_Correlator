import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DIHS_Correlator.core.clustering import recursive_cluster
from DIHS_Correlator.core.dihs import compute_dihs_metrics
from DIHS_Correlator.core.transforms import (
    BASE_TRANSFORMATIONS,
    apply_transformation,
    set_feature_columns,
)
from DIHS_Correlator.io.output_paths import cluster_dir, ensure_dir, pairwise_dir, tree_dir
from DIHS_Correlator.io.writers import (
    save_dataframe,
    save_pairwise_matrices_all_depths,
    save_pairwise_total_matrix,
)
from DIHS_Correlator.viz.hs_curves import plot_hs_curves
from DIHS_Correlator.viz.pairwise import plot_pairwise_matrix

from DIHS_Correlator.workflows.utils import (
    _log,
    _normalize_transform_type,
    _prepare_working_df,
    _resolve_unknown_class,
)

SUPPORTED_MODELS = ("agglomerative", "kmeans", "gaussian")
TRANSFORM_NAME_TO_ID = {v: k for k, v in BASE_TRANSFORMATIONS.items()}


class CorrelationRunner:
    """
    This class manages the execution of the clustering and DIHS metric computations for a single run.
    """
    def __init__(
        self,
        base_output_dir="./Results",
        save_trees=True,
        save_cluster_data=False,
        save_untransformed=False,
    ):
        self.base_output_dir = base_output_dir
        self.save_trees = save_trees
        self.save_cluster_data = save_cluster_data
        self.save_untransformed = save_untransformed
        self.transformations = BASE_TRANSFORMATIONS.copy()
        self.feature_columns = []

        self.last_pair_depth_matrices = None
        self.last_pair_total_matrix = None
        self.last_transform_name = None
        self.last_model_type = None

    def set_feature_columns(self, data, exclude=(), verbose=True):
        self.feature_columns = set_feature_columns(data, exclude=exclude)
        if verbose:
            print(f"Feature columns set: {self.feature_columns}")

    def run_combination(
        self,
        data,
        transform_type,
        model_type,
        random_state=None,
        unknown_class=0,
        class_column="controlcode",
        compute_pairwise=True,
        write_outputs=True,
        max_depth=100,
        return_intermediates=False,
    ):
        """
        Given a dataset, model type and transformation, run the clustering and metric computation.
        """
        transform_name = self.transformations[transform_type]
        original_data = data.copy()

        if self.save_trees:
            ensure_dir(tree_dir(self.base_output_dir, transform_name, model_type))
        if self.save_cluster_data:
            ensure_dir(cluster_dir(self.base_output_dir, transform_name, model_type))

        transformed_data, features = apply_transformation(
            data=data,
            feature_columns=self.feature_columns,
            transform_type=transform_type,
            class_column=class_column,
        )

        cdir = cluster_dir(self.base_output_dir, transform_name, model_type)
        initial_dataset_path = os.path.join(cdir, "initial_dataset.csv")
        if self.save_cluster_data:
            to_write = original_data if self.save_untransformed else transformed_data
            save_dataframe(to_write, initial_dataset_path, index=False)

        df_clustered, sample_clusters = recursive_cluster(
            transformed_data.copy(),
            features,
            model_type,
            max_depth=max_depth,
            path="",
            random_state=random_state,
            unknown_class=unknown_class,
            class_column=class_column,
        )

        if self.save_cluster_data:
            source_df = original_data if self.save_untransformed else df_clustered
            for cluster_name, meta in sample_clusters.items():
                cluster_file = os.path.join(cdir, f"{cluster_name}.csv")
                indices = meta.get("indices", [])
                save_dataframe(source_df.loc[indices], cluster_file, index=False)
                meta["data_path"] = cluster_file

        sample_clusters["0_initial"] = {
            "sample": 0,
            "path": "initial",
            "depth": 0,
            "label": -1,
            "data_path": initial_dataset_path if self.save_cluster_data else None,
        }

        model_name_cap = model_type.title()
        depth_cols = [c for c in df_clustered.columns if c.startswith(f"Depth_{model_name_cap}")]
        output_cols = [class_column] + depth_cols
        df_clustered_output = df_clustered[output_cols]

        tree_out = tree_dir(self.base_output_dir, transform_name, model_type)
        metrics = compute_dihs_metrics(
            df_clustered=df_clustered_output,
            unknown_class=unknown_class,
            model_type=model_type,
            transform_name=transform_name,
            class_column=class_column,
            compute_pairwise=compute_pairwise,
        )
        if metrics is None:
            raise ValueError(
                f"No rows found for unknown_class={unknown_class}. "
                "Check your dataset encoding or pass the correct unknown_class."
            )

        metrics_per_depth = metrics["active_depth_long"]
        total_metrics = metrics["active_total"]
        self.last_pair_depth_matrices = metrics["pair_depth_matrices"]
        self.last_pair_total_matrix = metrics["pair_total_matrix"]
        self.last_transform_name = transform_name
        self.last_model_type = model_type

        if write_outputs:
            save_dataframe(
                metrics_per_depth,
                os.path.join(tree_out, f"metrics_{model_type}_{unknown_class}.csv"),
                index=False,
            )
            save_dataframe(
                total_metrics,
                os.path.join(tree_out, f"total_metrics_{model_type}_{unknown_class}.csv"),
                index=False,
            )
            if self.save_trees:
                save_dataframe(
                    df_clustered_output,
                    os.path.join(tree_out, f"tree_{model_type}_{unknown_class}.csv"),
                    index=False,
                )

        result = {
            "transform_name": transform_name,
            "model_type": model_type,
            "samples_processed": 1,
            "metrics_per_depth": metrics_per_depth,
            "total_metrics": total_metrics,
            "sample_clusters": {"0": sample_clusters},
        }
        if return_intermediates:
            result["transformed_data"] = transformed_data.copy()
            result["features"] = list(features)
            result["clustered_tree"] = df_clustered_output.copy()
        return result

    def get_pairwise_total_matrix(self):
        return self.last_pair_total_matrix

    def save_pairwise_matrices_all_depths(self, output_dir=None, prefix="pairwise_matrix_depth"):
        if self.last_pair_depth_matrices is None:
            return []
        if output_dir is None:
            transform_name = self.last_transform_name or "unknown_transform"
            model_type = self.last_model_type or "unknown_model"
            output_dir = pairwise_dir(self.base_output_dir, transform_name, model_type)
        return save_pairwise_matrices_all_depths(
            self.last_pair_depth_matrices, output_dir=output_dir, prefix=prefix
        )

    def save_pairwise_total_matrix(self, output_dir=None):
        if self.last_pair_total_matrix is None:
            return None
        if output_dir is None:
            transform_name = self.last_transform_name or "unknown_transform"
            model_type = self.last_model_type or "unknown_model"
            output_dir = pairwise_dir(self.base_output_dir, transform_name, model_type)
        return save_pairwise_total_matrix(self.last_pair_total_matrix, output_dir=output_dir)


def run_single_model_workflow(
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
) -> dict[str, Any]:
    model = model_type.lower().strip()
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_type='{model_type}'.")
    transform_id = _normalize_transform_type(transform_type)
    work_df = _prepare_working_df(df, class_column=class_column)
    unknown_class = _resolve_unknown_class(work_df, unknown_sample, class_column=class_column)

    runner = CorrelationRunner(
        base_output_dir=output_dir,
        save_trees=write_files,
        save_cluster_data=save_cluster_data,
        save_untransformed=save_untransformed,
    )
    exclude_set = set(exclude_columns) | {class_column}
    runner.set_feature_columns(work_df, exclude=tuple(exclude_set), verbose=verbose)

    effective_random_state = random_state if model in ("kmeans", "gaussian") else None
    run = runner.run_combination(
        data=work_df,
        transform_type=transform_id,
        model_type=model,
        random_state=effective_random_state,
        unknown_class=unknown_class,
        class_column=class_column,
        compute_pairwise=compute_pairwise,
        write_outputs=write_files,
        max_depth=max_depth,
    )

    artifacts: dict[str, Any] = {}
    if write_files and compute_pairwise:
        pairwise_out = os.path.join(
            output_dir,
            "Trees",
            f"{run['transform_name']}_{model}",
            "PairwiseMatrices",
        )
        artifacts["pairwise_depth_paths"] = runner.save_pairwise_matrices_all_depths(
            output_dir=pairwise_out
        )
        artifacts["pairwise_total_csv"] = runner.save_pairwise_total_matrix(
            output_dir=pairwise_out
        )

    if plot_everything:
        if plot_output_dir is None:
            plot_output_dir = os.path.join(output_dir, "Plots")
        os.makedirs(plot_output_dir, exist_ok=True)

        hs_curve_path = os.path.join(
            plot_output_dir, f"hs_curve_{run['transform_name']}_{model}.svg"
        )
        plot_hs_curves(
            df=run["metrics_per_depth"],
            value_col="harmonic_score",
            with_shade=False,
            output_path=hs_curve_path,
            title=f"HS vs depth | {run['transform_name']} + {model}",
            max_depth=max_depth,
            unknown_class=unknown_class,
        )
        artifacts["hs_curve_path"] = hs_curve_path

        if compute_pairwise and runner.get_pairwise_total_matrix() is not None:
            pairwise_plot_path = os.path.join(
                plot_output_dir, f"pairwise_total_{run['transform_name']}_{model}.svg"
            )
            plot_pairwise_matrix(
                matrix=runner.get_pairwise_total_matrix(),
                title=f"Pairwise DIHS | {run['transform_name']} + {model}",
                output_path=pairwise_plot_path,
                unknown_class=unknown_class,
                class_order=pairwise_plot_order,
            )
            artifacts["pairwise_total_plot_path"] = pairwise_plot_path

    return {
        "hs_per_depth": run["metrics_per_depth"],
        "dihs_total": run["total_metrics"],
        "pairwise_total_matrix": runner.get_pairwise_total_matrix()
        if compute_pairwise
        else None,
        "pairwise_per_depth_matrices": runner.last_pair_depth_matrices
        if compute_pairwise
        else None,
        "transform_name": run["transform_name"],
        "model_type": model,
        "unknown_class": unknown_class,
        "artifacts": artifacts,
    }


def triple_run_workflow(
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
) -> dict[str, Any]:
    """Run agglomerative, kmeans and gaussian under one shared configuration."""
    model_results = {}
    _log(verbose, "Starting triple run (agglomerative, kmeans, gaussian)...")
    for model in SUPPORTED_MODELS:
        _log(verbose, f"Running model: {model}")
        model_out = os.path.join(output_dir, model) if write_files else output_dir
        model_plot_out = (
            os.path.join(plot_output_dir, model)
            if (plot_output_dir is not None and write_files)
            else plot_output_dir
        )
        model_results[model] = run_single_model_workflow(
            df=df,
            model_type=model,
            transform_type=transform_type,
            unknown_sample=unknown_sample,
            class_column=class_column,
            random_state=random_state,
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

    hs_combined = pd.concat(
        [model_results[m]["hs_per_depth"] for m in SUPPORTED_MODELS], ignore_index=True
    )
    dihs_combined = pd.concat(
        [model_results[m]["dihs_total"] for m in SUPPORTED_MODELS], ignore_index=True
    )
    return {
        "hs_per_depth": hs_combined,
        "dihs_total": dihs_combined,
        "models": model_results,
    }
