import os


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def tree_dir(base_output_dir: str, transform_name: str, model_type: str) -> str:
    return os.path.join(base_output_dir, "Trees", f"{transform_name}_{model_type}")


def cluster_dir(base_output_dir: str, transform_name: str, model_type: str) -> str:
    return os.path.join(base_output_dir, "ClusterData", f"{transform_name}_{model_type}")


def pairwise_dir(base_output_dir: str, transform_name: str, model_type: str) -> str:
    return os.path.join(tree_dir(base_output_dir, transform_name, model_type), "PairwiseMatrices")

