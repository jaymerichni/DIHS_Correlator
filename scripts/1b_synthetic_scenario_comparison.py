from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import combinations
from math import comb
from multiprocessing import Pool, freeze_support
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores*",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from DIHS_Correlator.core.transforms import BASE_TRANSFORMATIONS
from DIHS_Correlator.workflows.single_run import CorrelationRunner

CSV_PATH = REPO_ROOT / "data" / "processed" / "synthetic_scenarios" / "all_scenarios_combined.csv"
OUTPUT_ROOT = REPO_ROOT / "results" / "1_benchmarking_comparison"
UNKNOWN_SAMPLE = "X"
TRUE_SOURCE_CLASS = "A"
CLASS_COLUMN = "class_label"
MODEL_TYPES = ("agglomerative", "gaussian", "kmeans")
SAMPLE_SIZES = tuple(range(1, 21))
MAX_SAMPLING_ITERATIONS = 5000
RANDOM_STATE = 42
MAX_DEPTH = 100
N_WORKERS = 8

TRANSFORM_NAME_TO_ID = {name: key for key, name in BASE_TRANSFORMATIONS.items()}
_FULL_DF: pd.DataFrame | None = None


def _class_key(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _class_sort_key(value: Any):
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def _normalize_transform_type(transform_type: str) -> int:
    if not isinstance(transform_type, str):
        raise ValueError(
            "transform_type must be one of: "
            + ", ".join(sorted(TRANSFORM_NAME_TO_ID))
        )
    key = str(transform_type).strip().lower()
    if key not in TRANSFORM_NAME_TO_ID:
        raise ValueError(f"Unsupported transform_type='{transform_type}'.")
    return TRANSFORM_NAME_TO_ID[key]


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _add_metadata(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for key, value in metadata.items():
        out[key] = value
    return out


def _exclude_unknown_neighbor(df: pd.DataFrame, unknown_class: Any) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["neighbor_unit"].apply(_class_key) != _class_key(unknown_class)].copy()


def _score_key_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "run_id",
        "iteration",
        "source_class",
        "source_class_key",
        "case",
        "sample_iteration",
        "sample_size",
        "true_source_present",
        "unknown_class",
        "transform",
        "model",
        "neighbor_unit",
    ]
    return [column for column in preferred if column in df.columns]


def _compute_centroid_scores(
    transformed_data: pd.DataFrame,
    features: list[str],
    unknown_class: Any,
    class_column: str,
) -> pd.DataFrame:
    work = transformed_data.copy()
    unknown_mask = work[class_column].apply(_class_key) == _class_key(unknown_class)
    if not unknown_mask.any():
        raise ValueError(f"Unknown class '{unknown_class}' is not present in transformed data.")

    x = work[features].to_numpy(dtype=float)
    unknown_centroid = x[unknown_mask.to_numpy()].mean(axis=0)

    rows = []
    for cls in sorted(pd.unique(work[class_column]), key=_class_sort_key):
        if _class_key(cls) == _class_key(unknown_class):
            continue
        class_mask = work[class_column].apply(_class_key) == _class_key(cls)
        if not class_mask.any():
            continue
        class_centroid = x[class_mask.to_numpy()].mean(axis=0)
        distance = float(np.linalg.norm(unknown_centroid - class_centroid))
        rows.append({"neighbor_unit": cls, "score": -distance})

    return pd.DataFrame(rows)


def _compute_mahalanobis_scores(
    transformed_data: pd.DataFrame,
    features: list[str],
    unknown_class: Any,
    class_column: str,
    eps: float = 1e-8,
) -> pd.DataFrame:
    work = transformed_data.copy()
    labels = work[class_column].to_numpy()
    label_keys = np.array([_class_key(value) for value in labels], dtype=object)
    unknown_key = _class_key(unknown_class)
    unknown_mask = label_keys == unknown_key
    if not unknown_mask.any():
        raise ValueError(f"Unknown class '{unknown_class}' is not present in transformed data.")

    x = work[features].to_numpy(dtype=float)
    unknown_centroid = x[unknown_mask].mean(axis=0)

    residual_blocks = []
    for cls in pd.unique(labels):
        cls_key = _class_key(cls)
        cls_mask = label_keys == cls_key
        x_cls = x[cls_mask]
        if x_cls.shape[0] < 2:
            continue
        residual_blocks.append(x_cls - x_cls.mean(axis=0, keepdims=True))

    residuals = np.vstack(residual_blocks) if residual_blocks else x.copy()
    covariance = LedoitWolf().fit(residuals).covariance_
    covariance = covariance + eps * np.eye(covariance.shape[0])
    covariance_inverse = np.linalg.pinv(covariance)

    rows = []
    for cls in sorted(pd.unique(labels), key=_class_sort_key):
        cls_key = _class_key(cls)
        if cls_key == unknown_key:
            continue
        class_mask = label_keys == cls_key
        if not class_mask.any():
            continue
        class_centroid = x[class_mask].mean(axis=0)
        delta = unknown_centroid - class_centroid
        distance = float(np.sqrt(delta @ covariance_inverse @ delta))
        rows.append({"neighbor_unit": cls, "score": -distance})

    return pd.DataFrame(rows)


def _recompute_cumulative_scores_by_depth(
    score_iterations: pd.DataFrame,
    score_col: str,
    key_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, int | None]:
    if score_iterations.empty:
        return pd.DataFrame(), None

    max_depth_per_run = score_iterations.groupby("run_id", as_index=False)["depth_level"].max()
    if max_depth_per_run.empty:
        return pd.DataFrame(), None

    common_depth_level = int(max_depth_per_run["depth_level"].min())
    keys = (
        [column for column in key_columns if column in score_iterations.columns]
        if key_columns is not None
        else [
            column
            for column in score_iterations.columns
            if column not in {"depth_level", score_col}
        ]
    )

    rows = []
    for integration_depth in range(common_depth_level + 1):
        cut = score_iterations[score_iterations["depth_level"] <= integration_depth].copy()
        if cut.empty:
            continue
        aggregated = (
            cut.groupby(keys, as_index=False)
            .agg(score_sum=(score_col, "sum"))
            .assign(score=lambda frame: frame["score_sum"] / float(integration_depth + 1))
            .drop(columns=["score_sum"])
        )
        aggregated.insert(0, "integration_depth", integration_depth)
        rows.append(aggregated)

    if not rows:
        return pd.DataFrame(), common_depth_level
    return pd.concat(rows, ignore_index=True), common_depth_level


def _extract_top_rankings(score_iterations: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "method",
        "run_id",
        "iteration",
        "integration_depth",
        "unknown_class",
        "transform",
        "model",
        "source_class",
        "source_class_key",
        "case",
        "sample_iteration",
        "sample_size",
        "true_source_present",
        "true_source_class",
        "true_source_key",
        "top1_class",
        "top2_class",
        "top1_score",
        "top2_score",
        "margin",
    ]
    if score_iterations.empty:
        return pd.DataFrame(columns=expected_columns)

    required = {"method", "run_id", "neighbor_unit", "score"}
    missing = sorted(required.difference(score_iterations.columns))
    if missing:
        raise ValueError(
            "_extract_top_rankings requires columns "
            f"{missing}, but only found {list(score_iterations.columns)}."
        )

    rows = []
    for (method, run_id), sub in score_iterations.groupby(["method", "run_id"]):
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        if sub.empty:
            continue

        top1_class = sub["neighbor_unit"].iloc[0]
        top1_score = float(sub["score"].iloc[0])
        top2_class = sub["neighbor_unit"].iloc[1] if len(sub) > 1 else np.nan
        top2_score = float(sub["score"].iloc[1]) if len(sub) > 1 else np.nan
        margin = top1_score - top2_score if len(sub) > 1 else np.nan

        metadata_columns = [
            column
            for column in (
                "iteration",
                "integration_depth",
                "unknown_class",
                "transform",
                "model",
                "source_class",
                "source_class_key",
                "case",
                "sample_iteration",
                "sample_size",
                "true_source_present",
                "true_source_class",
                "true_source_key",
            )
            if column in sub.columns
        ]

        row = {"method": method, "run_id": int(run_id)}
        for column in metadata_columns:
            row[column] = sub[column].iloc[0]
        row.update(
            {
                "top1_class": top1_class,
                "top2_class": top2_class,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "margin": margin,
            }
        )
        rows.append(row)

    if not rows:
        raise ValueError(
            "_extract_top_rankings received score rows but produced no ranking rows."
        )
    return pd.DataFrame(rows)


def _iter_unique_sample_position_sets(
    *,
    n_pool: int,
    sample_size: int,
    n_requested: int,
    rng: np.random.Generator,
):
    n_pool = int(n_pool)
    sample_size = int(sample_size)
    n_requested = int(n_requested)

    if sample_size <= 0:
        raise ValueError("sample_size must be > 0.")
    if n_pool <= 0:
        raise ValueError("n_pool must be > 0.")
    if sample_size > n_pool:
        raise ValueError(
            f"Requested sample_size={sample_size}, but the unknown pool only contains "
            f"{n_pool} rows."
        )
    if n_requested <= 0:
        raise ValueError("n_requested must be > 0.")

    n_possible = comb(n_pool, sample_size)
    n_to_run = min(n_requested, n_possible)

    if n_to_run == n_possible:
        all_combinations = list(combinations(range(n_pool), sample_size))
        for index in rng.permutation(len(all_combinations)):
            yield all_combinations[int(index)]
        return

    if n_possible <= 250_000:
        all_combinations = list(combinations(range(n_pool), sample_size))
        chosen = rng.choice(len(all_combinations), size=n_to_run, replace=False)
        for index in chosen:
            yield all_combinations[int(index)]
        return

    seen = set()
    while len(seen) < n_to_run:
        positions = tuple(sorted(rng.choice(n_pool, size=sample_size, replace=False).tolist()))
        if positions in seen:
            continue
        seen.add(positions)
        yield positions


def _run_comparison_ensemble(
    *,
    df: pd.DataFrame,
    unknown_class: Any,
    class_column: str,
    model_type: str,
    transform_id: int,
    random_state: int | None,
    n_perturbations: int,
    max_depth: int,
    exclude_columns: Iterable[Any],
    context_metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    progress_label: str | None = None,
) -> dict[str, Any]:
    runner = CorrelationRunner(
        base_output_dir="./results",
        save_trees=False,
        save_cluster_data=False,
        save_untransformed=False,
    )
    exclude_set = set(exclude_columns) | {class_column}
    runner.set_feature_columns(df.copy(), exclude=tuple(exclude_set), verbose=False)

    hs_rows = []
    centroid_rows = []
    mahalanobis_rows = []

    for iteration in range(int(n_perturbations)):
        if verbose:
            label = progress_label or "comparison ensemble"
            print(f"{label}: perturbation {iteration + 1}/{int(n_perturbations)}", flush=True)

        run = runner.run_combination(
            data=df.copy(),
            transform_type=transform_id,
            model_type=model_type,
            random_state=random_state if model_type in ("kmeans", "gaussian") else None,
            unknown_class=unknown_class,
            class_column=class_column,
            compute_pairwise=False,
            write_outputs=False,
            max_depth=max_depth,
            return_intermediates=True,
        )

        metadata = {"run_id": iteration, "iteration": iteration}
        if context_metadata:
            metadata.update(context_metadata)

        hs_iteration = _add_metadata(run["metrics_per_depth"], metadata)
        hs_rows.append(hs_iteration)

        centroid_iteration = _compute_centroid_scores(
            transformed_data=run["transformed_data"],
            features=run["features"],
            unknown_class=unknown_class,
            class_column=class_column,
        )
        centroid_iteration["unknown_class"] = unknown_class
        centroid_iteration["transform"] = run["transform_name"]
        centroid_iteration["model"] = model_type
        centroid_rows.append(_add_metadata(centroid_iteration, metadata))

        mahalanobis_iteration = _compute_mahalanobis_scores(
            transformed_data=run["transformed_data"],
            features=run["features"],
            unknown_class=unknown_class,
            class_column=class_column,
        )
        mahalanobis_iteration["unknown_class"] = unknown_class
        mahalanobis_iteration["transform"] = run["transform_name"]
        mahalanobis_iteration["model"] = model_type
        mahalanobis_rows.append(_add_metadata(mahalanobis_iteration, metadata))

    hs_iterations = pd.concat(hs_rows, ignore_index=True) if hs_rows else pd.DataFrame()
    centroid_iterations = (
        pd.concat(centroid_rows, ignore_index=True) if centroid_rows else pd.DataFrame()
    )
    mahalanobis_iterations = (
        pd.concat(mahalanobis_rows, ignore_index=True)
        if mahalanobis_rows
        else pd.DataFrame()
    )

    score_keys = _score_key_columns(hs_iterations)
    dihs_by_depth, dihs_common_depth = _recompute_cumulative_scores_by_depth(
        hs_iterations,
        score_col="harmonic_score",
        key_columns=score_keys,
    )
    common_depth_level = dihs_common_depth
    by_depth_frames = []
    final_frames = []

    for method_name, table in (("dihs", dihs_by_depth),):
        if table.empty:
            continue
        method_table = table.copy()
        method_table.insert(0, "method", method_name)
        by_depth_frames.append(method_table)
        if common_depth_level is not None:
            final_frames.append(
                method_table[method_table["integration_depth"] == common_depth_level].copy()
            )

    if not centroid_iterations.empty:
        centroid_final = centroid_iterations.copy()
        centroid_final.insert(0, "method", "centroid_distance")
        centroid_final["integration_depth"] = common_depth_level
        final_frames.append(centroid_final)

    if not mahalanobis_iterations.empty:
        mahalanobis_final = mahalanobis_iterations.copy()
        mahalanobis_final.insert(0, "method", "mahalanobis_distance")
        mahalanobis_final["integration_depth"] = common_depth_level
        final_frames.append(mahalanobis_final)

    score_iterations_by_depth = (
        pd.concat(by_depth_frames, ignore_index=True) if by_depth_frames else pd.DataFrame()
    )
    score_iterations = (
        pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame()
    )

    score_iterations_by_depth = _exclude_unknown_neighbor(
        score_iterations_by_depth,
        unknown_class,
    )
    score_iterations = _exclude_unknown_neighbor(score_iterations, unknown_class)

    return {
        "common_depth_level": common_depth_level,
        "hs_iterations": hs_iterations,
        "score_iterations": score_iterations,
        "score_iterations_by_depth": score_iterations_by_depth,
    }


def _run_direct_unknown_sampling_comparison(
    *,
    df: pd.DataFrame,
    unknown_sample: Any,
    true_source_class: Any,
    model_type: str,
    transform_id: int,
    class_column: str,
    sample_size: int,
    n_sampling_iterations: int,
    random_state: int | None,
    max_depth: int,
    exclude_columns: Iterable[Any],
    verbose: bool,
) -> dict[str, Any]:
    unknown_key = _class_key(unknown_sample)
    true_source_key = _class_key(true_source_class)

    class_keys = df[class_column].apply(_class_key)
    unknown_mask = class_keys == unknown_key
    unknown_pool = df.loc[unknown_mask].copy()
    source_df = df.loc[~unknown_mask].copy()

    if unknown_pool.empty:
        raise ValueError(
            f"Unknown sample '{unknown_sample}' not found in class column '{class_column}'."
        )

    sample_size = int(sample_size)
    n_unknown_points = int(len(unknown_pool))
    if n_unknown_points < sample_size:
        raise ValueError(
            f"Requested sample_size={sample_size}, but the unknown pool only contains "
            f"{n_unknown_points} rows."
        )

    n_possible_combinations = comb(n_unknown_points, sample_size)
    effective_n_sampling_iterations = min(
        int(n_sampling_iterations),
        int(n_possible_combinations),
    )
    sampling_mode = (
        "exhaustive_unique"
        if effective_n_sampling_iterations == n_possible_combinations
        else "random_unique"
    )

    rng = np.random.default_rng(random_state)
    score_rows = []
    score_by_depth_rows = []
    run_id_counter = 0

    if verbose:
        print(
            "Direct unknown resampling | "
            f"sample_size={sample_size} | "
            f"unknown_points={n_unknown_points} | "
            f"possible_combinations={n_possible_combinations} | "
            f"requested_iterations={int(n_sampling_iterations)} | "
            f"used_iterations={effective_n_sampling_iterations} | "
            f"mode={sampling_mode} | "
            f"true_source={true_source_class}",
            flush=True,
        )

    unknown_index_values = unknown_pool.index.to_numpy()
    unique_position_sets = _iter_unique_sample_position_sets(
        n_pool=n_unknown_points,
        sample_size=sample_size,
        n_requested=effective_n_sampling_iterations,
        rng=rng,
    )
    progress_every = max(1, effective_n_sampling_iterations // 20)

    for sample_iteration, sampled_positions in enumerate(unique_position_sets):
        if verbose and (
            sample_iteration == 0
            or (sample_iteration + 1) % progress_every == 0
            or (sample_iteration + 1) == effective_n_sampling_iterations
        ):
            print(
                f"[PID {os.getpid()}] "
                f"model={model_type} | "
                f"sample_size={sample_size} | "
                f"iteration {sample_iteration + 1}/{effective_n_sampling_iterations} | "
                f"mode={sampling_mode}",
                flush=True,
            )

        sampled_positions = np.asarray(sampled_positions, dtype=int)
        sampled_indices = unknown_index_values[sampled_positions]
        sampled_unknown = unknown_pool.loc[sampled_indices].copy()
        case_df = pd.concat([source_df, sampled_unknown], ignore_index=False).copy()

        case_result = _run_comparison_ensemble(
            df=case_df,
            unknown_class=unknown_sample,
            class_column=class_column,
            model_type=model_type,
            transform_id=transform_id,
            random_state=random_state,
            n_perturbations=1,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            context_metadata={
                "sample_iteration": sample_iteration,
                "sample_size": sample_size,
                "true_source_class": true_source_class,
                "true_source_key": true_source_key,
                "n_unknown_points": n_unknown_points,
                "n_possible_combinations": n_possible_combinations,
                "requested_sampling_iterations": int(n_sampling_iterations),
                "used_sampling_iterations": effective_n_sampling_iterations,
                "sampling_mode": sampling_mode,
            },
            verbose=False,
        )

        score_final = case_result["score_iterations"].copy()
        if score_final.empty:
            score_by_depth_fallback = case_result["score_iterations_by_depth"].copy()
            if (
                not score_by_depth_fallback.empty
                and "integration_depth" in score_by_depth_fallback.columns
            ):
                deepest_depth = int(score_by_depth_fallback["integration_depth"].max())
                score_final = score_by_depth_fallback[
                    score_by_depth_fallback["integration_depth"] == deepest_depth
                ].copy()

        if score_final.empty and sample_iteration == 0:
            raise ValueError(
                "No score rows were produced for the first sampled case.\n"
                f"model_type={model_type}\n"
                f"sample_size={sample_size}\n"
                f"unknown_sample={unknown_sample}\n"
                f"case_df_shape={case_df.shape}\n"
                f"class_counts:\n{case_df[class_column].value_counts().to_string()}\n"
                f"columns={list(case_df.columns)}"
            )

        if not score_final.empty:
            score_final["run_id"] = run_id_counter
            score_rows.append(score_final)

        score_by_depth = case_result["score_iterations_by_depth"].copy()
        if not score_by_depth.empty:
            score_by_depth["run_id"] = run_id_counter
            score_by_depth_rows.append(score_by_depth)

        run_id_counter += 1

    score_iterations = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    score_iterations_by_depth = (
        pd.concat(score_by_depth_rows, ignore_index=True)
        if score_by_depth_rows
        else pd.DataFrame()
    )

    score_iterations = _exclude_unknown_neighbor(score_iterations, unknown_sample)
    score_iterations_by_depth = _exclude_unknown_neighbor(
        score_iterations_by_depth,
        unknown_sample,
    )

    comparison_runs = _extract_top_rankings(score_iterations)
    if comparison_runs.empty:
        raise ValueError(
            "No comparison runs were produced after all sampled cases.\n"
            f"model={model_type}\n"
            f"sample_size={sample_size}\n"
            f"score_iterations_shape={score_iterations.shape}\n"
            f"score_iterations_columns={list(score_iterations.columns)}"
        )

    comparison_runs["top1_is_true_source"] = (
        comparison_runs["top1_class"].apply(_class_key) == true_source_key
    )
    comparison_runs["n_unknown_points"] = n_unknown_points
    comparison_runs["n_possible_combinations"] = n_possible_combinations
    comparison_runs["requested_sampling_iterations"] = int(n_sampling_iterations)
    comparison_runs["used_sampling_iterations"] = effective_n_sampling_iterations
    comparison_runs["sampling_mode"] = sampling_mode

    summary_rows = []
    for method, sub in comparison_runs.groupby("method"):
        summary_rows.append(
            {
                "method": method,
                "top1_accuracy": (
                    float(sub["top1_is_true_source"].mean())
                    if not sub.empty
                    else np.nan
                ),
                "margin_median": (
                    float(sub["margin"].median())
                    if sub["margin"].notna().any()
                    else np.nan
                ),
                "margin_mean": (
                    float(sub["margin"].mean())
                    if sub["margin"].notna().any()
                    else np.nan
                ),
                "n_runs": int(len(sub)),
                "n_unknown_points": n_unknown_points,
                "n_possible_combinations": n_possible_combinations,
                "requested_sampling_iterations": int(n_sampling_iterations),
                "used_sampling_iterations": effective_n_sampling_iterations,
                "sampling_mode": sampling_mode,
            }
        )

    return {
        "comparison_runs": comparison_runs,
        "method_summary": pd.DataFrame(summary_rows),
        "score_iterations": score_iterations,
        "score_iterations_by_depth": score_iterations_by_depth,
    }


def run_direct_unknown_sample_size_curve(
    *,
    df: pd.DataFrame,
    unknown_sample: Any,
    true_source_class: Any,
    class_column: str = CLASS_COLUMN,
    model_type: str = "agglomerative",
    transform_type: str = "scaled",
    sample_sizes: Iterable[int] = SAMPLE_SIZES,
    n_sampling_iterations: int = MAX_SAMPLING_ITERATIONS,
    random_state: int | None = RANDOM_STATE,
    max_depth: int = MAX_DEPTH,
    exclude_columns: Iterable[Any] = (),
    verbose: bool = True,
) -> dict[str, Any]:
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in dataframe.")

    model = str(model_type).strip().lower()
    if model not in {"agglomerative", "kmeans", "gaussian"}:
        raise ValueError(f"Unsupported model_type='{model_type}'.")
    if int(n_sampling_iterations) <= 0:
        raise ValueError("n_sampling_iterations must be > 0.")

    transform_id = _normalize_transform_type(transform_type)

    unknown_key = _class_key(unknown_sample)
    class_keys = df[class_column].apply(_class_key)
    n_unknown_points = int((class_keys == unknown_key).sum())
    if n_unknown_points <= 0:
        raise ValueError(
            f"Unknown sample '{unknown_sample}' not found in class column '{class_column}'."
        )

    summary_rows = []
    run_rows = []

    for sample_size in sample_sizes:
        sample_size = int(sample_size)
        if sample_size <= 0:
            raise ValueError("All sample sizes must be > 0.")
        if sample_size > n_unknown_points:
            raise ValueError(
                f"Requested sample_size={sample_size}, but unknown sample {unknown_sample!r} "
                f"only has {n_unknown_points} rows."
            )

        n_possible_combinations = comb(n_unknown_points, sample_size)
        effective_n_sampling_iterations = min(
            int(n_sampling_iterations),
            int(n_possible_combinations),
        )
        sampling_mode = (
            "exhaustive_unique"
            if effective_n_sampling_iterations == n_possible_combinations
            else "random_unique"
        )

        if verbose:
            print(
                "Running direct unknown sample-size comparison | "
                f"sample_size={sample_size} | "
                f"unknown_points={n_unknown_points} | "
                f"possible_combinations={n_possible_combinations} | "
                f"requested_iterations={int(n_sampling_iterations)} | "
                f"used_iterations={effective_n_sampling_iterations} | "
                f"mode={sampling_mode}",
                flush=True,
            )

        result = _run_direct_unknown_sampling_comparison(
            df=df,
            unknown_sample=unknown_sample,
            true_source_class=true_source_class,
            model_type=model,
            transform_id=transform_id,
            class_column=class_column,
            sample_size=sample_size,
            n_sampling_iterations=effective_n_sampling_iterations,
            random_state=random_state,
            max_depth=max_depth,
            exclude_columns=exclude_columns,
            verbose=verbose,
        )

        method_summary = result["method_summary"].copy()
        if not method_summary.empty:
            method_summary["sample_size"] = sample_size
            method_summary["n_unknown_points"] = n_unknown_points
            method_summary["n_possible_combinations"] = n_possible_combinations
            method_summary["requested_sampling_iterations"] = int(n_sampling_iterations)
            method_summary["used_sampling_iterations"] = effective_n_sampling_iterations
            method_summary["sampling_mode"] = sampling_mode
            summary_rows.append(method_summary)

        comparison_runs = result["comparison_runs"].copy()
        if not comparison_runs.empty:
            comparison_runs["sample_size"] = sample_size
            comparison_runs["n_unknown_points"] = n_unknown_points
            comparison_runs["n_possible_combinations"] = n_possible_combinations
            comparison_runs["requested_sampling_iterations"] = int(n_sampling_iterations)
            comparison_runs["used_sampling_iterations"] = effective_n_sampling_iterations
            comparison_runs["sampling_mode"] = sampling_mode
            run_rows.append(comparison_runs)

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    runs_df = pd.concat(run_rows, ignore_index=True) if run_rows else pd.DataFrame()
    return {
        "method_summary_by_sample_size": summary_df,
        "comparison_runs_by_sample_size": runs_df,
    }


def direct_unknown_sample_size_run(
    *,
    df: pd.DataFrame,
    unknown_sample: Any,
    true_source_class: Any,
    class_column: str = CLASS_COLUMN,
    model_type: str = "agglomerative",
    transform_type: str = "scaled",
    sample_sizes: Iterable[int] = SAMPLE_SIZES,
    n_sampling_iterations: int = MAX_SAMPLING_ITERATIONS,
    random_state: int | None = RANDOM_STATE,
    max_depth: int = MAX_DEPTH,
    exclude_columns: Iterable[Any] = (),
    write_files: bool = False,
    output_dir: str | Path = REPO_ROOT / "results" / "direct_unknown_sample_size",
    verbose: bool = True,
    return_details: bool = False,
):
    result = run_direct_unknown_sample_size_curve(
        df=df,
        unknown_sample=unknown_sample,
        true_source_class=true_source_class,
        class_column=class_column,
        model_type=model_type,
        transform_type=transform_type,
        sample_sizes=sample_sizes,
        n_sampling_iterations=n_sampling_iterations,
        random_state=random_state,
        max_depth=max_depth,
        exclude_columns=exclude_columns,
        verbose=verbose,
    )

    if write_files:
        output_dir = Path(output_dir)
        summary_path = _write_csv(
            result["method_summary_by_sample_size"],
            output_dir / "direct_unknown_sample_size_summary.csv",
        )
        runs_path = _write_csv(
            result["comparison_runs_by_sample_size"],
            output_dir / "direct_unknown_sample_size_runs.csv",
        )
        result["artifacts"] = {
            "summary_csv": str(summary_path),
            "runs_csv": str(runs_path),
        }
    else:
        result["artifacts"] = {}

    if return_details:
        return result
    return result["method_summary_by_sample_size"]


def _init_worker(full_df: pd.DataFrame):
    global _FULL_DF
    _FULL_DF = full_df


def _worker_dataframe() -> pd.DataFrame:
    if _FULL_DF is None:
        raise RuntimeError("Worker dataframe has not been initialised.")
    return _FULL_DF


def _infer_true_source_class(
    scenario_df: pd.DataFrame,
    class_column: str,
    unknown_sample: Any,
    fallback: Any,
) -> Any:
    if "is_true_source" not in scenario_df.columns:
        return fallback
    mask = scenario_df["is_true_source"].fillna(False).astype(bool)
    if "role" in scenario_df.columns:
        mask = mask & scenario_df["role"].astype(str).str.lower().eq("source")
    classes = scenario_df.loc[mask, class_column].dropna().unique().tolist()
    classes = [value for value in classes if _class_key(value) != _class_key(unknown_sample)]
    if not classes:
        return fallback
    if len(classes) > 1:
        raise ValueError(
            "Scenario has more than one true source class: "
            + ", ".join(map(str, classes))
        )
    return classes[0]


def _process_task(task: tuple[Any, ...]) -> pd.DataFrame:
    (
        scenario,
        model_type,
        output_root_str,
        unknown_sample,
        fallback_true_source_class,
        class_column,
        transform_type,
        sample_sizes,
        n_sampling_iterations,
        random_state,
        max_depth,
        verbose,
    ) = task

    full_df = _worker_dataframe()
    scenario_full_df = full_df[full_df["scenario"] == scenario].copy()
    if scenario_full_df.empty:
        raise ValueError(f"Scenario '{scenario}' was not found in the synthetic dataset.")

    true_source_class = _infer_true_source_class(
        scenario_full_df,
        class_column=class_column,
        unknown_sample=unknown_sample,
        fallback=fallback_true_source_class,
    )
    scenario_df = scenario_full_df.drop(
        columns=[column for column in ("scenario", "role", "is_true_source") if column in scenario_full_df.columns]
    ).copy()

    output_dir = Path(output_root_str) / str(scenario) / str(model_type)
    if verbose:
        print(
            f"[PID {os.getpid()}] Starting scenario='{scenario}', model='{model_type}'",
            flush=True,
        )

    result = direct_unknown_sample_size_run(
        df=scenario_df,
        unknown_sample=unknown_sample,
        true_source_class=true_source_class,
        class_column=class_column,
        model_type=model_type,
        transform_type=transform_type,
        sample_sizes=sample_sizes,
        n_sampling_iterations=n_sampling_iterations,
        random_state=random_state,
        max_depth=max_depth,
        exclude_columns=(),
        write_files=True,
        output_dir=output_dir,
        verbose=verbose,
        return_details=True,
    )

    summary = result["method_summary_by_sample_size"].copy()
    summary["scenario"] = scenario
    summary["model_type"] = model_type
    summary["output_dir"] = str(output_dir)

    if verbose:
        print(
            f"[PID {os.getpid()}] Finished scenario='{scenario}', model='{model_type}'",
            flush=True,
        )

    return summary


def _default_scenarios(df: pd.DataFrame) -> list[str]:
    if "scenario" not in df.columns:
        raise ValueError("The combined synthetic CSV does not contain a 'scenario' column.")
    return sorted(df["scenario"].dropna().astype(str).unique().tolist())


def _build_tasks(
    *,
    scenarios: list[str],
    models: Iterable[str],
    output_root: Path,
    unknown_sample: Any,
    true_source_class: Any,
    class_column: str,
    transform_type: str,
    sample_sizes: Iterable[int],
    n_sampling_iterations: int,
    random_state: int | None,
    max_depth: int,
    verbose: bool,
) -> list[tuple[Any, ...]]:
    return [
        (
            scenario,
            model_type,
            str(output_root),
            unknown_sample,
            true_source_class,
            class_column,
            transform_type,
            tuple(int(size) for size in sample_sizes),
            int(n_sampling_iterations),
            random_state,
            int(max_depth),
            bool(verbose),
        )
        for scenario in scenarios
        for model_type in models
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic DIHS scenarios with the direct-unknown sample-size comparison."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=CSV_PATH,
        help="Path to the combined synthetic scenarios CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Folder where the benchmark outputs should be written.",
    )
    parser.add_argument(
        "--unknown-sample",
        default=UNKNOWN_SAMPLE,
        help="Label used for the unknown class.",
    )
    parser.add_argument(
        "--true-source-class",
        default=TRUE_SOURCE_CLASS,
        help="Fallback label for the true source class.",
    )
    parser.add_argument(
        "--class-column",
        default=CLASS_COLUMN,
        help="Column containing the class labels.",
    )
    parser.add_argument(
        "--transform-type",
        default="scaled",
        help="Transformation to apply before clustering.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_TYPES),
        help="Models to benchmark.",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=list(SAMPLE_SIZES),
        help="Unknown sample sizes to benchmark.",
    )
    parser.add_argument(
        "--max-sampling-iterations",
        type=int,
        default=MAX_SAMPLING_ITERATIONS,
        help="Maximum number of unique unknown subsets to evaluate per sample size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random state for stochastic models and subset ordering.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help="Maximum recursive clustering depth.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help="Number of multiprocessing workers.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional subset of scenario names to run.",
    )
    parser.add_argument(
        "--write-combined-runs",
        action="store_true",
        help="Also write a single CSV concatenating the per-run outputs from every scenario and model.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging.",
    )
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(csv_path)
    available_scenarios = _default_scenarios(full_df)
    scenarios = available_scenarios if args.scenarios is None else list(args.scenarios)

    missing_scenarios = sorted(set(scenarios).difference(available_scenarios))
    if missing_scenarios:
        raise ValueError(
            "Requested scenarios were not found in the combined CSV: "
            + ", ".join(missing_scenarios)
        )

    models = [str(model).strip().lower() for model in args.models]
    unsupported_models = sorted(set(models).difference({"agglomerative", "gaussian", "kmeans"}))
    if unsupported_models:
        raise ValueError(
            "Unsupported models requested: " + ", ".join(unsupported_models)
        )

    sample_sizes = [int(size) for size in args.sample_sizes]
    if not sample_sizes:
        raise ValueError("At least one sample size must be provided.")

    tasks = _build_tasks(
        scenarios=scenarios,
        models=models,
        output_root=output_root,
        unknown_sample=args.unknown_sample,
        true_source_class=args.true_source_class,
        class_column=args.class_column,
        transform_type=args.transform_type,
        sample_sizes=sample_sizes,
        n_sampling_iterations=args.max_sampling_iterations,
        random_state=args.random_state,
        max_depth=args.max_depth,
        verbose=not args.quiet,
    )

    if args.workers <= 0:
        raise ValueError("workers must be > 0.")

    summaries = []
    if int(args.workers) == 1:
        _init_worker(full_df)
        for task in tasks:
            summaries.append(_process_task(task))
    else:
        with Pool(
            processes=int(args.workers),
            initializer=_init_worker,
            initargs=(full_df,),
            maxtasksperchild=1,
        ) as pool:
            for summary in pool.imap_unordered(_process_task, tasks, chunksize=1):
                summaries.append(summary)

    if not summaries:
        raise ValueError("No benchmark summaries were produced.")

    combined_summary = pd.concat(summaries, ignore_index=True)
    sort_columns = [
        column
        for column in ("scenario", "model_type", "sample_size", "method")
        if column in combined_summary.columns
    ]
    if sort_columns:
        combined_summary = combined_summary.sort_values(sort_columns).reset_index(drop=True)

    combined_summary_path = _write_csv(
        combined_summary,
        output_root / "all_scenarios_all_models_direct_unknown_sample_size_summary.csv",
    )

    if args.write_combined_runs:
        run_frames = []
        for scenario in scenarios:
            for model_type in models:
                runs_path = (
                    output_root
                    / scenario
                    / model_type
                    / "direct_unknown_sample_size_runs.csv"
                )
                if not runs_path.exists():
                    continue
                runs_df = pd.read_csv(runs_path)
                runs_df["scenario"] = scenario
                runs_df["model_type"] = model_type
                run_frames.append(runs_df)
        if run_frames:
            combined_runs = pd.concat(run_frames, ignore_index=True)
            _write_csv(
                combined_runs,
                output_root / "all_scenarios_all_models_direct_unknown_sample_size_runs.csv",
            )

    print(f"Saved combined summary to {combined_summary_path}")
    return combined_summary_path


if __name__ == "__main__":
    freeze_support()
    main()
