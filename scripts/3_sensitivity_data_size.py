from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
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

DATA_PATH = (
    REPO_ROOT / "data" / "processed" / "caio_italy_benchmark" / "full_italian_data.csv"
)
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
SENSITIVITY_CLASSES = ["AI", "PF", "EV", "RMP", "VV", "PI"]
EXPECTED_SIX_PROVINCE_CLASS_COUNTS = {
    "AI": 954,
    "PF": 867,
    "EV": 598,
    "RMP": 418,
    "VV": 415,
    "PI": 226,
}
EXPECTED_SIX_PROVINCE_ROWS = sum(EXPECTED_SIX_PROVINCE_CLASS_COUNTS.values())
EXPECTED_DATASET_ROWS_BY_FRACTION = {
    0.25: 869,
    0.50: 1740,
    0.75: 2609,
    1.00: 3478,
}
CONDITIONS = ("positive", "negative")


def clean_lettercode(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work = work.loc[:, ~work.columns.astype(str).str.startswith("Unnamed:")]

    if "controlcode" in work.columns:
        work = work.drop(columns=["controlcode"])

    work[CLASS_COLUMN] = (
        work[CLASS_COLUMN].astype(str).str.strip().replace({"0": "Caio", "0.0": "Caio"})
    )

    work = work[work[CLASS_COLUMN].isin(SENSITIVITY_CLASSES)]

    if work[CLASS_COLUMN].nunique() != 6:
        raise ValueError(
            "Sensitivity dataset must contain exactly six classes: "
            f"{SENSITIVITY_CLASSES}. Found {work[CLASS_COLUMN].nunique()}."
        )

    observed_class_counts = work[CLASS_COLUMN].value_counts().to_dict()
    if observed_class_counts != EXPECTED_SIX_PROVINCE_CLASS_COUNTS:
        raise ValueError(
            "Six-province class counts do not match the benchmark reconstructed "
            "from the pinned Petrelli source files. "
            f"Expected {EXPECTED_SIX_PROVINCE_CLASS_COUNTS}; "
            f"found {observed_class_counts}."
        )

    if len(work) != EXPECTED_SIX_PROVINCE_ROWS:
        raise ValueError(
            "Six-province sensitivity subset row count does not match the "
            "reconstructed benchmark dataset: "
            f"N={EXPECTED_SIX_PROVINCE_ROWS}. Found N={len(work)}. "
            "Verify benchmark reconstruction inputs and preprocessing outputs."
        )

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
    exact_counts = class_counts.astype(float) * fraction
    allocated_counts = pd.Series(
        np.rint(exact_counts.to_numpy()).astype(int),
        index=exact_counts.index,
    )

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
    scaled_df = scaled_df.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )

    observed_counts = scaled_df[class_column].value_counts(sort=False)
    observed_counts = observed_counts.reindex(
        allocated_counts.index, fill_value=0
    ).astype(int)
    if not observed_counts.equals(allocated_counts):
        raise RuntimeError(
            "Stratified rescaling produced unexpected class counts: "
            f"expected {allocated_counts.to_dict()}, found {observed_counts.to_dict()}."
        )

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


def compute_expected_total_runs() -> int:
    return (
        len(SENSITIVITY_CLASSES)
        * len(SAMPLE_SIZES)
        * len(DATASET_FRACTIONS)
        * len(MODELS)
        * int(N_ITERATIONS)
        * len(CONDITIONS)
    )


def write_run_design_metadata(*, output_root: Path, expected_total_runs: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    design = {
        "class_column": CLASS_COLUMN,
        "sensitivity_classes": SENSITIVITY_CLASSES,
        "sample_sizes": SAMPLE_SIZES,
        "dataset_fractions": DATASET_FRACTIONS,
        "models": MODELS,
        "iterations_per_class_condition": int(N_ITERATIONS),
        "conditions": list(CONDITIONS),
        "condition_definitions": {
            "positive": "true source class is present in the candidate set",
            "negative": "true source class is excluded from the candidate set",
        },
        "run_formula": (
            f"{len(SENSITIVITY_CLASSES)} x {len(SAMPLE_SIZES)} x {len(DATASET_FRACTIONS)} x "
            f"{len(MODELS)} x {int(N_ITERATIONS)} x {len(CONDITIONS)}"
        ),
        "expected_total_runs": int(expected_total_runs),
        "expected_six_province_rows": int(EXPECTED_SIX_PROVINCE_ROWS),
        "expected_six_province_class_counts": EXPECTED_SIX_PROVINCE_CLASS_COUNTS,
        "expected_dataset_rows_by_fraction": {
            format_dataset_size_label(fraction): int(n_rows)
            for fraction, n_rows in EXPECTED_DATASET_ROWS_BY_FRACTION.items()
        },
    }
    with (output_root / "run_design_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(design, f, indent=2)


def run_model(
    args: tuple[str, pd.DataFrame, int, str],
) -> tuple[str, int, str, str, dict]:
    model, df, sample_size, dataset_size_label = args
    output_dir = (
        OUTPUT_ROOT / dataset_size_label / str(sample_size) / FEATURE_SPACE / model
    )

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
    expected_total_runs = compute_expected_total_runs()
    print(
        "Planned run count: "
        f"{expected_total_runs} "
        f"({len(SENSITIVITY_CLASSES)} classes x {len(SAMPLE_SIZES)} sample sizes x "
        f"{len(DATASET_FRACTIONS)} dataset fractions x {len(MODELS)} models x "
        f"{N_ITERATIONS} iterations x {len(CONDITIONS)} conditions)"
    )
    if expected_total_runs != 172800:
        raise ValueError(
            "Sensitivity design must plan exactly 172,800 runs; "
            f"computed {expected_total_runs}."
        )

    full_df = load_coupled_dataset()
    write_run_design_metadata(
        output_root=OUTPUT_ROOT, expected_total_runs=expected_total_runs
    )

    all_results: dict[tuple[str, int, str, str], dict | None] = {}

    for dataset_fraction in DATASET_FRACTIONS:
        dataset_size_label = format_dataset_size_label(dataset_fraction)
        dataset_df = stratified_rescale_dataframe(
            full_df,
            fraction=dataset_fraction,
            class_column=CLASS_COLUMN,
            random_state=RANDOM_STATE,
        )
        expected_dataset_rows = EXPECTED_DATASET_ROWS_BY_FRACTION[dataset_fraction]
        if len(dataset_df) != expected_dataset_rows:
            raise RuntimeError(
                f"Unexpected row count for {dataset_size_label}: "
                f"expected {expected_dataset_rows}, found {len(dataset_df)}."
            )
        dataset_results: dict[tuple[int, str, str], dict | None] = {}

        print(
            f"\nStarting dataset size: {dataset_size_label} "
            f"({len(dataset_df)} rows, target fraction={dataset_fraction:.2f})"
        )

        for sample_size in SAMPLE_SIZES:
            print(f"Starting sample size: {sample_size}")

            tasks = [
                (model, dataset_df, sample_size, dataset_size_label) for model in MODELS
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
                            (
                                finished_sample_size,
                                finished_feature_space,
                                finished_model,
                            )
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
