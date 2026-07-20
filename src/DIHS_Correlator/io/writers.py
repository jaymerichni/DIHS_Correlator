import os


def _ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def save_dataframe(df, path: str, index: bool = False):
    _ensure_parent_dir(path)
    df.to_csv(path, index=index)
    return path


def save_pairwise_matrices_all_depths(depth_matrices, output_dir: str, prefix="pairwise_matrix_depth"):
    if depth_matrices is None:
        return []
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for d in sorted(depth_matrices.keys()):
        fp = os.path.join(output_dir, f"{prefix}_{int(d):03d}.csv")
        depth_matrices[d].to_csv(fp)
        saved.append(fp)
    return saved


def save_pairwise_total_matrix(total_matrix, output_dir: str):
    if total_matrix is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    fp = os.path.join(output_dir, "pairwise_matrix_total.csv")
    total_matrix.to_csv(fp)
    return fp

