from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import DIHS_Correlator as dc

DATA_PATH = REPO_ROOT / "data" / "processed" / "caio_italy_benchmark" / "full_italian_data.csv"
OUTPUT_ROOT = REPO_ROOT / "results" / "3_sensitivity_data_size"
FEATURE_SPACE = "Coupled"
TRANSFORM_TYPE = "clr"
CLASS_COLUMN = "lettercode"
SAMPLE_SIZES = [1, 2, 4, 5, 7, 10, 12, 15, 18, 20, 22, 25]
DATASET_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
MODELS = ["kmeans", "gaussian", "agglomerative"]
N_ITERATIONS = 100
RANDOM_STATE = 12345
MAX_DEPTH = 100


def clean_lettercode(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "controlcode" in work.columns:
        work = work.drop(columns=["controlcode"])

    work[CLASS_COLUMN] = (
        work[CLASS_COLUMN]
        .astype(str)
        .str.strip()
        .replace({"0": "Caio", "0.0": "Caio"})
    )

    work = work[work[CLASS_COLUMN] != "Caio"]
    work = work[work[CLASS_COLUMN] != "IAVP"]
    return work.reset_index(drop=True)


def load_coupled_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing local benchmark dataset at "
            f"'{DATA_PATH}'. This repository does not redistribute the "
            "Petrelli/GEOROC-derived Italian benchmark data. Follow "
            "'data/raw/README.md' to obtain the upstream inputs and run "
            "'scripts/0_raw_data_preprocessing.ipynb' to generate "
            "'data/processed/caio_italy_benchmark/full_italian_data.csv' locally."
        )
    return clean_lettercode(pd.read_csv(DATA_PATH))


def format_dataset_size_label(fraction: float) -> str:
    return f"{int(round(100 * float(fraction)))}pct"


def stratified_rescale_dataframe(
    df: pd.DataFrame,
    *,
    fraction: float,
    class_column: str = CLASS_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    fraction = float(fraction)
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in the interval (0, 1].")
    if np.isclose(fraction, 1.0):
        return df.copy().reset_index(drop=True)

    class_counts = df[class_column].value_counts(sort=False)
    target_total = int(round(len(df) * fraction))
    exact_counts = class_counts.astype(float) * fraction
    allocated_counts = np.floor(exact_counts).astype(int)

    remainder = target_total - int(allocated_counts.sum())
    if remainder > 0:
        fractional_parts = (exact_counts - allocated_counts).sort_values(ascending=False)
        for class_value in fractional_parts.index[:remainder]:
            allocated_counts.loc[class_value] += 1

    rng = np.random.default_rng(random_state)
    sampled_frames: list[pd.DataFrame] = []

    for class_value in class_counts.index:
        n_keep = int(allocated_counts.loc[class_value])
        if n_keep <= 0:
            continue

        class_df = df[df[class_column] == class_value]
        chosen_positions = rng.choice(len(class_df), size=n_keep, replace=False)
        sampled_frames.append(class_df.iloc[np.sort(chosen_positions)])

    scaled_df = pd.concat(sampled_frames, ignore_index=False)
    scaled_df = scaled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return scaled_df


def write_dataset_metadata(
    *,
    dataset_df: pd.DataFrame,
    dataset_fraction: float,
    dataset_size_label: str,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    class_counts = (
        dataset_df[CLASS_COLUMN]
        .value_counts()
        .rename_axis(CLASS_COLUMN)
        .reset_index(name="n_rows")
    )
    class_counts.insert(0, "dataset_n_rows", int(len(dataset_df)))
    class_counts.insert(0, "dataset_fraction", float(dataset_fraction))
    class_counts.insert(0, "dataset_size", dataset_size_label)
    class_counts.to_csv(output_root / "dataset_class_counts.csv", index=False)


def run_model(args: tuple[str, pd.DataFrame, int, str]) -> tuple[str, int, str, str, dict]:
    model, df, sample_size, dataset_size_label = args
    output_dir = OUTPUT_ROOT / dataset_size_label / str(sample_size) / FEATURE_SPACE / model

    result = dc.pseudo_unknown_run(
        df=df,
        model_type=model,
        transform_type=TRANSFORM_TYPE,
        class_column=CLASS_COLUMN,
        sample_size=sample_size,
        n_iterations=N_ITERATIONS,
        output_dir=str(output_dir),
        plot_output_dir=str(output_dir),
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        plot_everything=True,
        write_files=True,
        return_details=True,
    )

    return dataset_size_label, sample_size, FEATURE_SPACE, model, result


def _tag_frame(
    frame: pd.DataFrame,
    *,
    dataset_size_label: str,
    dataset_fraction: float,
    dataset_n_rows: int,
    sample_size: int,
    feature_space: str,
    model: str,
    output_dir: Path,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    tagged = frame.copy()
    tagged.insert(0, "dataset_n_rows", int(dataset_n_rows))
    tagged.insert(0, "dataset_fraction", float(dataset_fraction))
    tagged.insert(0, "dataset_size", dataset_size_label)
    tagged.insert(0, "model", model)
    tagged.insert(0, "feature_space", feature_space)
    tagged.insert(0, "sample_size", int(sample_size))
    tagged["output_dir"] = str(output_dir)
    return tagged


def write_combined_summaries(
    *,
    dataset_df: pd.DataFrame,
    dataset_fraction: float,
    dataset_size_label: str,
    results_dict: dict[tuple[int, str, str], dict | None],
) -> None:
    threshold_frames: list[pd.DataFrame] = []
    case_frames: list[pd.DataFrame] = []
    class_frames: list[pd.DataFrame] = []
    dataset_output_root = OUTPUT_ROOT / dataset_size_label

    for (sample_size, feature_space, model), result in results_dict.items():
        if result is None:
            continue

        output_dir = dataset_output_root / str(sample_size) / feature_space / model

        threshold_frames.append(
            _tag_frame(
                result.get("threshold_summary", pd.DataFrame()),
                dataset_size_label=dataset_size_label,
                dataset_fraction=dataset_fraction,
                dataset_n_rows=len(dataset_df),
                sample_size=sample_size,
                feature_space=feature_space,
                model=model,
                output_dir=output_dir,
            )
        )
        case_frames.append(
            _tag_frame(
                result.get("summary_by_case", pd.DataFrame()),
                dataset_size_label=dataset_size_label,
                dataset_fraction=dataset_fraction,
                dataset_n_rows=len(dataset_df),
                sample_size=sample_size,
                feature_space=feature_space,
                model=model,
                output_dir=output_dir,
            )
        )
        class_frames.append(
            _tag_frame(
                result.get("summary_by_class", pd.DataFrame()),
                dataset_size_label=dataset_size_label,
                dataset_fraction=dataset_fraction,
                dataset_n_rows=len(dataset_df),
                sample_size=sample_size,
                feature_space=feature_space,
                model=model,
                output_dir=output_dir,
            )
        )

    write_dataset_metadata(
        dataset_df=dataset_df,
        dataset_fraction=dataset_fraction,
        dataset_size_label=dataset_size_label,
        output_root=dataset_output_root,
    )

    if threshold_frames:
        pd.concat(threshold_frames, ignore_index=True).to_csv(
            dataset_output_root / "sensitivity_threshold_summary.csv",
            index=False,
        )

    if case_frames:
        pd.concat(case_frames, ignore_index=True).to_csv(
            dataset_output_root / "sensitivity_summary_by_case.csv",
            index=False,
        )

    if class_frames:
        pd.concat(class_frames, ignore_index=True).to_csv(
            dataset_output_root / "sensitivity_summary_by_class.csv",
            index=False,
        )


def main() -> dict[tuple[str, int, str, str], dict | None]:
    full_df = load_coupled_dataset()
    all_results: dict[tuple[str, int, str, str], dict | None] = {}

    for dataset_fraction in DATASET_FRACTIONS:
        dataset_size_label = format_dataset_size_label(dataset_fraction)
        dataset_df = stratified_rescale_dataframe(
            full_df,
            fraction=dataset_fraction,
            class_column=CLASS_COLUMN,
            random_state=RANDOM_STATE,
        )
        dataset_results: dict[tuple[int, str, str], dict | None] = {}

        print(
            f"\nStarting dataset size: {dataset_size_label} "
            f"({len(dataset_df)} rows, target fraction={dataset_fraction:.2f})"
        )

        for sample_size in SAMPLE_SIZES:
            print(f"Starting sample size: {sample_size}")

            tasks = [
                (model, dataset_df, sample_size, dataset_size_label)
                for model in MODELS
            ]

            with ProcessPoolExecutor(max_workers=len(MODELS)) as executor:
                futures = {executor.submit(run_model, task): task for task in tasks}

                for future in as_completed(futures):
                    model, _, task_sample_size, _ = futures[future]

                    try:
                        (
                            finished_dataset_size,
                            finished_sample_size,
                            finished_feature_space,
                            finished_model,
                            result,
                        ) = future.result()

                        dataset_results[
                            (finished_sample_size, finished_feature_space, finished_model)
                        ] = result
                        all_results[
                            (
                                finished_dataset_size,
                                finished_sample_size,
                                finished_feature_space,
                                finished_model,
                            )
                        ] = result

                        print(
                            f"Finished: dataset_size={finished_dataset_size}, "
                            f"sample_size={finished_sample_size}, "
                            f"feature_space={finished_feature_space}, "
                            f"model={finished_model}"
                        )

                    except Exception as exc:
                        print(
                            f"Error: dataset_size={dataset_size_label}, "
                            f"sample_size={task_sample_size}, "
                            f"feature_space={FEATURE_SPACE}, "
                            f"model={model}: {exc}"
                        )
                        dataset_results[(task_sample_size, FEATURE_SPACE, model)] = None
                        all_results[
                            (dataset_size_label, task_sample_size, FEATURE_SPACE, model)
                        ] = None

        write_combined_summaries(
            dataset_df=dataset_df,
            dataset_fraction=dataset_fraction,
            dataset_size_label=dataset_size_label,
            results_dict=dataset_results,
        )

    print(f"\nSaved dataset-size sensitivity outputs under {OUTPUT_ROOT}")
    return all_results


if __name__ == "__main__":
    main()
