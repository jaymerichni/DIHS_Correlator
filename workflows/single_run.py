import os

import pandas as pd

from Tephra_Correlator_Refactored.core.clustering import recursive_cluster
from Tephra_Correlator_Refactored.core.dihs import compute_dihs_metrics
from Tephra_Correlator_Refactored.core.transforms import (
    BASE_TRANSFORMATIONS,
    apply_transformation,
    set_feature_columns,
)
from Tephra_Correlator_Refactored.io.output_paths import cluster_dir, ensure_dir, pairwise_dir, tree_dir
from Tephra_Correlator_Refactored.io.writers import (
    save_dataframe,
    save_pairwise_matrices_all_depths,
    save_pairwise_total_matrix,
)


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

        # Create output directories if needed
        if self.save_trees:
            ensure_dir(tree_dir(self.base_output_dir, transform_name, model_type))
        if self.save_cluster_data:
            ensure_dir(cluster_dir(self.base_output_dir, transform_name, model_type))

        # Apply transformation to the input data and get the feature columns
        transformed_data, features = apply_transformation(
            data=data,
            feature_columns=self.feature_columns,
            transform_type=transform_type,
            class_column=class_column,
        )

        # Save the initial transformed dataset if cluster data saving is enabled
        cdir = cluster_dir(self.base_output_dir, transform_name, model_type)
        initial_dataset_path = os.path.join(cdir, "initial_dataset.csv")
        if self.save_cluster_data:
            to_write = original_data if self.save_untransformed else transformed_data
            save_dataframe(to_write, initial_dataset_path, index=False)

        """
        The next block interacts with the core part of the algorithm: clustering + metric computation
        """

        # Run the recursive clustering and get the clustered dataframe and cluster metadata
        df_clustered, sample_clusters = recursive_cluster(
            transformed_data.copy(),
            features,
            model_type,
            base_output_dir=self.base_output_dir,
            save_cluster_data=self.save_cluster_data,
            save_untransformed=self.save_untransformed,
            max_depth=max_depth,
            path="",
            transform_name=transform_name,
            original_data=original_data if self.save_untransformed else None,
            random_state=random_state,
            unknown_class=unknown_class,
            class_column=class_column,
        )
 
        # Add the initial dataset as a cluster in the metadata store
        sample_clusters["0_initial"] = {
            "sample": 0,
            "path": "initial",
            "depth": 0,
            "label": -1,
            "data_path": initial_dataset_path,
        }

        # Select class and depth columns for metrics computation
        model_name_cap = model_type.title()
        depth_cols = [c for c in df_clustered.columns if c.startswith(f"Depth_{model_name_cap}")]
        output_cols = [class_column] + depth_cols
        df_clustered_output = df_clustered[output_cols]

        # Compute DIHS metrics and save outputs
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

        # Store the last computed pairwise matrices and metrics for later retrieval
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
    
    """
    The following methods allow retrieval and saving of the pairwise distance matrices computed during the metrics computation (usually used by the API)
    """

    def get_pairwise_total_matrix(self):
        return self.last_pair_total_matrix

    def get_pairwise_matrix_for_depth(self, depth_level):
        if self.last_pair_depth_matrices is None:
            return None
        return self.last_pair_depth_matrices.get(int(depth_level), None)

    def save_pairwise_matrix_for_depth(self, depth_level, output_dir=None):
        matrix = self.get_pairwise_matrix_for_depth(depth_level)
        if matrix is None:
            return None
        if output_dir is None:
            transform_name = self.last_transform_name or "unknown_transform"
            model_type = self.last_model_type or "unknown_model"
            output_dir = pairwise_dir(self.base_output_dir, transform_name, model_type)
        ensure_dir(output_dir)
        fp = os.path.join(output_dir, f"pairwise_matrix_depth_{int(depth_level):03d}.csv")
        matrix.to_csv(fp)
        return fp

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
