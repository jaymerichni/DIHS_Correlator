import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize


def _white_to_cmap(base_cmap_name: str = "YlOrBr", white_frac: float = 0.18, n: int = 256):
    base = cm.get_cmap(base_cmap_name, n)
    colors = base(np.linspace(0, 1, n))
    k = int(np.clip(white_frac, 0.0, 0.95) * n)
    colors[:k, :] = np.array([1, 1, 1, 1])
    return LinearSegmentedColormap.from_list(f"white_{base_cmap_name}", colors)


def _resolve_plot_order(
    matrix,
    unknown_class,
    sort_by_unknown: bool,
    class_order=None,
):
    base_order = list(matrix.index)
    if class_order is not None:
        requested = [str(label) for label in class_order]
        duplicates = []
        seen = set()
        for label in requested:
            if label in seen and label not in duplicates:
                duplicates.append(label)
            seen.add(label)
        if duplicates:
            raise ValueError(
                f"class_order contains duplicate labels: {duplicates}"
            )

        missing = [label for label in requested if label not in base_order]
        if missing:
            raise ValueError(
                f"class_order contains labels that are not present in the matrix: {missing}"
            )

        remaining = [label for label in base_order if label not in seen]
        return requested + remaining

    unk = str(unknown_class)
    if sort_by_unknown and unk in base_order:
        return list(matrix.loc[unk].sort_values(ascending=False).index)
    return base_order


def plot_pairwise_matrix(
    matrix,
    title=None,
    output_path=None,
    vmin=0.0,
    vmax=1.0,
    unknown_class: int = 0,
    sort_by_unknown: bool = True,
    class_order=None,
    annotate: bool = True,
    base_cmap: str = "YlOrBr",
    white_frac: float = 0.0,
    fig_scale: float = 0.75,
    min_figsize=(12, 10),
    max_figsize=None,
    title_fs: int = 26,
    axis_label_fs: int = 30,
    tick_fs: int = 28,
    annot_fs: int = 24,
    cbar_label_fs: int = 28,
    cbar_tick_fs: int = 28,
    plot_upper_triangle_only: bool = False,
    cbar_label: str = "Pairwise DIHS",
    annot_fmt: str = "{val:.2f}",
    annot_white_threshold: float = 0.75,
):
    mat = matrix.copy()
    mat.index = mat.index.map(str)
    mat.columns = mat.columns.map(str)

    mat = mat.reindex(index=mat.index, columns=mat.index).fillna(0.0)
    mat = 0.5 * (mat + mat.T)

    arr = mat.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 1.0)

    mat = pd.DataFrame(
        arr,
        index=mat.index,
        columns=mat.columns,
    )

    order = _resolve_plot_order(
        matrix=mat,
        unknown_class=unknown_class,
        sort_by_unknown=sort_by_unknown,
        class_order=class_order,
    )
    mat = mat.reindex(index=order, columns=order)

    labels = list(mat.index)
    data = mat.to_numpy(dtype=float)
    n = len(labels)

    if plot_upper_triangle_only:
        mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        data_masked = np.ma.array(data, mask=mask_upper)
    else:
        data_masked = data

    longest_label = max((len(label) for label in labels), default=1)
    cell_size = max(
        fig_scale,
        0.62 if annotate else 0.48,
        tick_fs / 52.0,
        annot_fs / 42.0 if annotate else 0.0,
    )
    width_from_cells = cell_size * n
    height_from_cells = cell_size * n
    width_from_labels = 7.0 + max(0.0, longest_label - 4) * 0.20
    height_from_labels = 6.0 + max(0.0, longest_label - 4) * 0.12

    w = max(min_figsize[0], width_from_cells + width_from_labels)
    h = max(min_figsize[1], height_from_cells + height_from_labels)

    if max_figsize is not None:
        w = min(max_figsize[0], w)
        h = min(max_figsize[1], h)

    cmap = _white_to_cmap(base_cmap_name=base_cmap, white_frac=white_frac).copy()
    cmap.set_bad(color=(1, 1, 1, 0))
    norm = Normalize(vmin=vmin, vmax=vmax)

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(data_masked, cmap=cmap, norm=norm)

    if title is not None:
        ax.set_title(title, fontsize=title_fs, pad=18)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(labels, fontsize=tick_fs)
    ax.set_xlabel("Class", fontsize=axis_label_fs, labelpad=20)
    ax.set_ylabel("Class", fontsize=axis_label_fs, labelpad=20)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.6, alpha=0.12)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    if annotate:
        for i in range(n):
            for j in range(n):
                if plot_upper_triangle_only and j > i:
                    continue
                val = data[i, j]
                txt_color = "white" if val > annot_white_threshold else "black"
                ax.text(
                    j,
                    i,
                    annot_fmt.format(val=val),
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    color=txt_color,
                )

    plt.tight_layout()
    if output_path is not None:
        out_dir = os.path.dirname(str(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
