from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import pandas as pd

from DIHS_Correlator.workflows.perturbative import perturbative_simple_run_workflow
from DIHS_Correlator.workflows.pseudo_unknown import run_pseudo_unknown_experiments
from DIHS_Correlator.workflows.resolvedness import (
    perturbative_triple_run_with_resolvedness_workflow,
)
from DIHS_Correlator.workflows.single_run import CorrelationRunner
from DIHS_Correlator.workflows.utils import (
    _normalize_transform_type,
    _prepare_working_df,
    _resolve_unknown_class,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "synthetic_scenarios" / "all_scenarios_combined.csv"
DEFAULT_RESULTS_DIR = ROOT / "results" / "benchmarks" / "stage2_recursive_clustering"

CLASS_COLUMN = "class_label"
UNKNOWN_SAMPLE = "X"
TRANSFORM_TYPE = "scaled"
RANDOM_STATE = 123
PERTURBATION_SEED = 321
MAX_DEPTH = 20
EXCLUDE_COLUMNS = ("scenario", "role", "is_true_source")
MEASURED_RUNS = 5


@dataclass(frozen=True)
class SnapshotArtifact:
    frame: pd.DataFrame
    sort_columns: tuple[str, ...] | None = None
    preserve_row_order: bool = False


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    scenario: str
    dataset_profile: str
    method: str
    notes: str
    runner: Callable[[], dict[str, Any]]
    snapshot_extractor: Callable[[dict[str, Any]], dict[str, SnapshotArtifact]]


def _load_profiles() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    profiles = {
        "medium": df.loc[df["scenario"] == "persistent_overlap"].reset_index(drop=True),
        "stress": df.loc[
            df["scenario"].isin(["persistent_overlap", "shallow_only_overlap"])
        ].reset_index(drop=True),
        "small": df.loc[df["scenario"] == "compact_separated"].reset_index(drop=True),
    }
    return profiles


def _common_single_kwargs(df: pd.DataFrame, *, model_type: str) -> dict[str, Any]:
    return {
        "df": df,
        "model_type": model_type,
        "transform_type": TRANSFORM_TYPE,
        "unknown_sample": UNKNOWN_SAMPLE,
        "class_column": CLASS_COLUMN,
        "random_state": RANDOM_STATE if model_type in {"kmeans", "gaussian"} else None,
        "compute_pairwise": False,
        "plot_everything": False,
        "write_files": False,
        "max_depth": MAX_DEPTH,
        "exclude_columns": EXCLUDE_COLUMNS,
        "verbose": False,
    }


def _run_single_model_with_intermediates(
    df: pd.DataFrame,
    *,
    model_type: str,
) -> dict[str, Any]:
    kwargs = _common_single_kwargs(df, model_type=model_type)
    model = str(kwargs["model_type"]).strip().lower()
    transform_id = _normalize_transform_type(kwargs["transform_type"])
    work_df = _prepare_working_df(kwargs["df"], class_column=CLASS_COLUMN)
    unknown_class = _resolve_unknown_class(
        work_df, UNKNOWN_SAMPLE, class_column=CLASS_COLUMN
    )

    runner = CorrelationRunner(
        base_output_dir=".",
        save_trees=False,
        save_cluster_data=False,
        save_untransformed=False,
    )
    exclude_set = set(EXCLUDE_COLUMNS) | {CLASS_COLUMN}
    runner.set_feature_columns(work_df, exclude=tuple(exclude_set), verbose=False)

    run = runner.run_combination(
        data=work_df,
        transform_type=transform_id,
        model_type=model,
        random_state=kwargs["random_state"],
        unknown_class=unknown_class,
        class_column=CLASS_COLUMN,
        compute_pairwise=False,
        write_outputs=False,
        max_depth=MAX_DEPTH,
        return_intermediates=True,
        model_params=None,
    )
    return {
        "hs_per_depth": run["metrics_per_depth"].copy(),
        "dihs_total": run["total_metrics"].copy(),
        "clustered_tree": run["clustered_tree"].copy(),
        "unknown_class": unknown_class,
        "model_type": model,
    }


def _run_perturbative_case(df: pd.DataFrame, *, model_type: str) -> dict[str, Any]:
    return perturbative_simple_run_workflow(
        df=df,
        model_type=model_type,
        transform_type=TRANSFORM_TYPE,
        unknown_sample=UNKNOWN_SAMPLE,
        class_column=CLASS_COLUMN,
        random_state=RANDOM_STATE if model_type == "kmeans" else None,
        n_iterations=2,
        major_cols=["x1"],
        trace_cols=["x2"],
        major_error=0.02,
        trace_error=0.05,
        perturbation_seed=PERTURBATION_SEED,
        compute_pairwise=False,
        plot_everything=False,
        write_files=False,
        output_dir="./Results_perturbative",
        max_depth=MAX_DEPTH,
        exclude_columns=EXCLUDE_COLUMNS,
        integration_depth=5,
        verbose=False,
        n_jobs=1,
    )


def _run_pseudo_unknown_case(df: pd.DataFrame) -> dict[str, Any]:
    return run_pseudo_unknown_experiments(
        df=df,
        model_type="kmeans",
        transform_type=TRANSFORM_TYPE,
        class_column=CLASS_COLUMN,
        sample_size=5,
        n_iterations=1,
        excluded_classes=["C", UNKNOWN_SAMPLE],
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        exclude_columns=EXCLUDE_COLUMNS,
        target_precision=0.95,
        reported_precisions=[0.95, 0.90],
        min_runs_above_threshold=1,
        plot_everything=False,
        write_files=False,
        output_dir="./Results_pseudo_unknown",
        verbose=False,
        n_jobs=1,
    )


def _run_resolvedness_case(df: pd.DataFrame) -> dict[str, Any]:
    return perturbative_triple_run_with_resolvedness_workflow(
        df=df,
        transform_type=TRANSFORM_TYPE,
        unknown_sample=UNKNOWN_SAMPLE,
        class_column=CLASS_COLUMN,
        random_state=RANDOM_STATE,
        n_iterations=1,
        major_cols=["x1"],
        trace_cols=["x2"],
        major_error=0.02,
        trace_error=0.05,
        perturbation_seed=PERTURBATION_SEED,
        compute_pairwise=False,
        plot_everything=False,
        write_files=False,
        output_dir="./Results_resolvedness",
        max_depth=MAX_DEPTH,
        exclude_columns=EXCLUDE_COLUMNS,
        pseudo_unknown_iterations=1,
        pseudo_unknown_sample_size=3,
        pseudo_unknown_random_state=RANDOM_STATE,
        target_precisions=[0.95, 0.90],
        min_runs_above_threshold=1,
        integration_depth=None,
        verbose=False,
        return_details=True,
        n_jobs=1,
    )


def _single_snapshot(result: dict[str, Any]) -> dict[str, SnapshotArtifact]:
    return {
        "clustered_tree": SnapshotArtifact(
            frame=result["clustered_tree"],
            preserve_row_order=True,
        ),
        "hs_per_depth": SnapshotArtifact(
            frame=result["hs_per_depth"],
            sort_columns=("depth_level", "neighbor_unit"),
        ),
        "dihs_total": SnapshotArtifact(
            frame=result["dihs_total"],
            sort_columns=("neighbor_unit",),
        ),
    }


def _perturbative_snapshot(result: dict[str, Any]) -> dict[str, SnapshotArtifact]:
    return {
        "hs_iterations": SnapshotArtifact(
            frame=result["hs_iterations"],
            sort_columns=("iteration", "depth_level", "neighbor_unit"),
        ),
        "dihs_summary": SnapshotArtifact(
            frame=result["dihs_summary"],
            sort_columns=("neighbor_unit",),
        ),
        "top1_frequency": SnapshotArtifact(
            frame=result["top1_frequency"],
            sort_columns=("neighbor_unit",),
        ),
    }


def _pseudo_unknown_snapshot(result: dict[str, Any]) -> dict[str, SnapshotArtifact]:
    return {
        "run_results": SnapshotArtifact(
            frame=result["run_results"],
            sort_columns=("run_id",),
        ),
        "threshold_summary": SnapshotArtifact(
            frame=result["threshold_summary"],
            sort_columns=("integration_depth",),
        ),
    }


def _resolvedness_snapshot(result: dict[str, Any]) -> dict[str, SnapshotArtifact]:
    return {
        "summary": SnapshotArtifact(
            frame=result["summary"],
            sort_columns=("model",),
        )
    }


def _build_cases(profiles: dict[str, pd.DataFrame]) -> list[BenchmarkCase]:
    medium = profiles["medium"]
    stress = profiles["stress"]
    small = profiles["small"]
    return [
        BenchmarkCase(
            case_id="single_medium_kmeans",
            scenario="run_single_model_workflow",
            dataset_profile="medium",
            method="kmeans",
            notes="persistent_overlap profile; compute_pairwise=False; write_files=False",
            runner=lambda df=medium: _run_single_model_with_intermediates(df, model_type="kmeans"),
            snapshot_extractor=_single_snapshot,
        ),
        BenchmarkCase(
            case_id="single_medium_agglomerative",
            scenario="run_single_model_workflow",
            dataset_profile="medium",
            method="agglomerative",
            notes="persistent_overlap profile; compute_pairwise=False; write_files=False",
            runner=lambda df=medium: _run_single_model_with_intermediates(
                df, model_type="agglomerative"
            ),
            snapshot_extractor=_single_snapshot,
        ),
        BenchmarkCase(
            case_id="single_stress_kmeans",
            scenario="run_single_model_workflow",
            dataset_profile="stress",
            method="kmeans",
            notes=(
                "two-scenario overlap bundle (persistent_overlap + shallow_only_overlap); "
                "compute_pairwise=False; write_files=False"
            ),
            runner=lambda df=stress: _run_single_model_with_intermediates(df, model_type="kmeans"),
            snapshot_extractor=_single_snapshot,
        ),
        BenchmarkCase(
            case_id="single_stress_agglomerative",
            scenario="run_single_model_workflow",
            dataset_profile="stress",
            method="agglomerative",
            notes=(
                "two-scenario overlap bundle (persistent_overlap + shallow_only_overlap); "
                "compute_pairwise=False; write_files=False"
            ),
            runner=lambda df=stress: _run_single_model_with_intermediates(
                df, model_type="agglomerative"
            ),
            snapshot_extractor=_single_snapshot,
        ),
        BenchmarkCase(
            case_id="perturbative_medium_kmeans",
            scenario="perturbative_simple_run_workflow",
            dataset_profile="medium",
            method="kmeans",
            notes="persistent_overlap profile; n_iterations=2; fixed perturbation_seed=321",
            runner=lambda df=medium: _run_perturbative_case(df, model_type="kmeans"),
            snapshot_extractor=_perturbative_snapshot,
        ),
        BenchmarkCase(
            case_id="perturbative_medium_agglomerative",
            scenario="perturbative_simple_run_workflow",
            dataset_profile="medium",
            method="agglomerative",
            notes="persistent_overlap profile; n_iterations=2; fixed perturbation_seed=321",
            runner=lambda df=medium: _run_perturbative_case(df, model_type="agglomerative"),
            snapshot_extractor=_perturbative_snapshot,
        ),
        BenchmarkCase(
            case_id="pseudo_unknown_medium_kmeans",
            scenario="run_pseudo_unknown_experiments",
            dataset_profile="medium",
            method="kmeans",
            notes=(
                "persistent_overlap profile; sample_size=5; n_iterations=1; "
                "excluded_classes=['C','X']"
            ),
            runner=lambda df=medium: _run_pseudo_unknown_case(df),
            snapshot_extractor=_pseudo_unknown_snapshot,
        ),
        BenchmarkCase(
            case_id="resolvedness_small_kmeans_focus",
            scenario="perturbative_triple_run_with_resolvedness_workflow",
            dataset_profile="small",
            method="kmeans (triple workflow timed)",
            notes=(
                "compact_separated profile; n_iterations=1; pseudo_unknown_iterations=1; "
                "timing covers the full triple resolvedness workflow"
            ),
            runner=lambda df=small: _run_resolvedness_case(df),
            snapshot_extractor=_resolvedness_snapshot,
        ),
    ]


def _phase_dir(results_dir: Path, phase: str) -> Path:
    return results_dir / phase


def _snapshot_dir(results_dir: Path, phase: str) -> Path:
    return _phase_dir(results_dir, phase) / "snapshots"


def _canonicalize_artifact(artifact: SnapshotArtifact) -> pd.DataFrame:
    frame = artifact.frame.copy()
    if artifact.preserve_row_order:
        frame = frame.reset_index(drop=False).rename(columns={"index": "row_position"})
    elif artifact.sort_columns:
        existing = [col for col in artifact.sort_columns if col in frame.columns]
        if existing:
            frame = frame.sort_values(existing, kind="stable").reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    return frame


def _write_snapshot_artifacts(
    *,
    case: BenchmarkCase,
    result: dict[str, Any],
    results_dir: Path,
    phase: str,
) -> dict[str, Path]:
    snap_dir = _snapshot_dir(results_dir, phase)
    snap_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for artifact_name, artifact in case.snapshot_extractor(result).items():
        path = snap_dir / f"{case.case_id}__{artifact_name}.csv"
        frame = _canonicalize_artifact(artifact)
        frame.to_csv(path, index=False, float_format="%.16g")
        paths[artifact_name] = path
    return paths


def _compare_snapshots(
    *,
    case: BenchmarkCase,
    results_dir: Path,
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    candidate_dir = _snapshot_dir(results_dir, "candidate")
    baseline_dir = _snapshot_dir(results_dir, "baseline")
    for candidate_path in sorted(candidate_dir.glob(f"{case.case_id}__*.csv")):
        baseline_path = baseline_dir / candidate_path.name
        if not baseline_path.exists():
            mismatches.append(f"Missing baseline snapshot: {baseline_path.name}")
            continue
        baseline_text = baseline_path.read_text(encoding="utf-8")
        candidate_text = candidate_path.read_text(encoding="utf-8")
        if baseline_text != candidate_text:
            mismatches.append(candidate_path.name)
    return (len(mismatches) == 0), mismatches


def _summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("No timing samples were collected.")
    return {
        "seconds_mean": statistics.fmean(samples),
        "seconds_median": statistics.median(samples),
        "seconds_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _run_once(case: BenchmarkCase) -> tuple[dict[str, Any], float, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        result = case.runner()
        elapsed = time.perf_counter() - start
    return result, elapsed, [str(item.message) for item in caught]


def _write_phase_files(
    *,
    phase: str,
    results_dir: Path,
    sample_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    phase_root = _phase_dir(results_dir, phase)
    phase_root.mkdir(parents=True, exist_ok=True)
    raw_path = phase_root / "raw_timing_samples.csv"
    summary_path = phase_root / "timing_summary.csv"
    pd.DataFrame(sample_rows).to_csv(raw_path, index=False, float_format="%.16g")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, float_format="%.16g")
    return raw_path, summary_path


def _write_config(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_path": str(DATA_PATH),
        "profiles": {
            "medium": "scenario == 'persistent_overlap'",
            "stress": "scenario in ['persistent_overlap', 'shallow_only_overlap']",
            "small": "scenario == 'compact_separated'",
        },
        "transform_type": TRANSFORM_TYPE,
        "class_column": CLASS_COLUMN,
        "unknown_sample": UNKNOWN_SAMPLE,
        "random_state": RANDOM_STATE,
        "perturbation_seed": PERTURBATION_SEED,
        "max_depth": MAX_DEPTH,
        "exclude_columns": list(EXCLUDE_COLUMNS),
        "measured_runs": MEASURED_RUNS,
    }
    (results_dir / "benchmark_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def _phase_benchmark(
    *,
    phase: str,
    cases: list[BenchmarkCase],
    results_dir: Path,
    measured_runs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for case in cases:
        warmup_result, warmup_elapsed, warmup_warnings = _run_once(case)
        snapshot_paths = _write_snapshot_artifacts(
            case=case,
            result=warmup_result,
            results_dir=results_dir,
            phase=phase,
        )
        correctness_pass = True
        mismatch_text = ""
        if phase == "candidate":
            correctness_pass, mismatches = _compare_snapshots(case=case, results_dir=results_dir)
            mismatch_text = "; ".join(mismatches)

        elapsed_samples: list[float] = []
        warning_count = len(warmup_warnings)
        for sample_index in range(1, measured_runs + 1):
            _, elapsed, run_warnings = _run_once(case)
            elapsed_samples.append(elapsed)
            warning_count += len(run_warnings)
            sample_rows.append(
                {
                    "phase": phase,
                    "case_id": case.case_id,
                    "scenario": case.scenario,
                    "dataset_profile": case.dataset_profile,
                    "method": case.method,
                    "sample_index": sample_index,
                    "wall_seconds": elapsed,
                    "warning_count": len(run_warnings),
                }
            )

        summary = _summarize_samples(elapsed_samples)
        summary_rows.append(
            {
                "phase": phase,
                "case_id": case.case_id,
                "scenario": case.scenario,
                "dataset_profile": case.dataset_profile,
                "method": case.method,
                "warmup_seconds": warmup_elapsed,
                "measured_runs": measured_runs,
                "warning_count_total": warning_count,
                "correctness_pass": correctness_pass,
                "snapshot_mismatches": mismatch_text,
                "snapshot_files": ";".join(path.name for path in snapshot_paths.values()),
                "notes": case.notes,
                **summary,
            }
        )
    raw_path, summary_path = _write_phase_files(
        phase=phase,
        results_dir=results_dir,
        sample_rows=sample_rows,
        summary_rows=summary_rows,
    )
    print(f"[{phase}] wrote raw timing samples to {raw_path}")
    print(f"[{phase}] wrote timing summary to {summary_path}")
    return pd.DataFrame(sample_rows), pd.DataFrame(summary_rows)


def _build_candidate_report(
    *,
    results_dir: Path,
    candidate_samples: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline_summary_path = _phase_dir(results_dir, "baseline") / "timing_summary.csv"
    if not baseline_summary_path.exists():
        raise FileNotFoundError(
            f"Baseline summary not found at {baseline_summary_path}. Run the baseline phase first."
        )
    baseline_summary = pd.read_csv(baseline_summary_path)

    report_rows: list[dict[str, Any]] = []
    for _, candidate_row in candidate_summary.iterrows():
        baseline_row = baseline_summary.loc[
            baseline_summary["case_id"] == candidate_row["case_id"]
        ]
        if baseline_row.empty:
            raise ValueError(f"Missing baseline row for case_id={candidate_row['case_id']}.")
        baseline_row = baseline_row.iloc[0]

        case_samples = candidate_samples.loc[
            candidate_samples["case_id"] == candidate_row["case_id"], "wall_seconds"
        ].tolist()
        repeat_count = sum(sample < float(baseline_row["seconds_median"]) for sample in case_samples)
        repeat_threshold = max(1, math.ceil(0.8 * len(case_samples)))
        correctness_pass = bool(candidate_row["correctness_pass"])
        no_new_warnings = int(candidate_row["warning_count_total"]) <= int(
            baseline_row["warning_count_total"]
        )
        candidate_median = float(candidate_row["seconds_median"])
        baseline_median = float(baseline_row["seconds_median"])
        delta_seconds = baseline_median - candidate_median
        percent_change = 100.0 * delta_seconds / baseline_median
        speedup_ratio = baseline_median / candidate_median
        repeat_pass = repeat_count >= repeat_threshold
        pass_row = (
            candidate_median < baseline_median
            and correctness_pass
            and no_new_warnings
            and repeat_pass
        )

        notes_parts = [str(candidate_row["notes"])]
        if not correctness_pass:
            notes_parts.append(f"snapshot mismatches: {candidate_row['snapshot_mismatches']}")
        if not no_new_warnings:
            notes_parts.append(
                "candidate emitted more warnings than baseline "
                f"({int(candidate_row['warning_count_total'])} vs "
                f"{int(baseline_row['warning_count_total'])})"
            )
        if not repeat_pass:
            notes_parts.append(
                f"candidate beat the baseline median in {repeat_count} of "
                f"{len(case_samples)} measured runs"
            )

        report_rows.append(
            {
                "scenario": candidate_row["scenario"],
                "dataset_profile": candidate_row["dataset_profile"],
                "method": candidate_row["method"],
                "baseline_seconds_median": baseline_median,
                "candidate_seconds_median": candidate_median,
                "delta_seconds": delta_seconds,
                "percent_change": percent_change,
                "speedup_ratio": speedup_ratio,
                "sample_count": len(case_samples),
                "pass_or_fail": "pass" if pass_row else "fail",
                "notes": " | ".join(part for part in notes_parts if part),
                "baseline_seconds_mean": float(baseline_row["seconds_mean"]),
                "candidate_seconds_mean": float(candidate_row["seconds_mean"]),
                "baseline_seconds_std": float(baseline_row["seconds_std"]),
                "candidate_seconds_std": float(candidate_row["seconds_std"]),
                "correctness_pass": correctness_pass,
                "repeat_count_below_baseline_median": repeat_count,
            }
        )
    return pd.DataFrame(report_rows)


def _write_candidate_report(results_dir: Path, report: pd.DataFrame) -> None:
    report_root = _phase_dir(results_dir, "candidate")
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "benchmark_delta_report.csv"
    md_path = report_root / "benchmark_delta_report.md"
    report.to_csv(csv_path, index=False, float_format="%.16g")

    lines = [
        "# Stage 2 Recursive Clustering Benchmark Delta Report",
        "",
        "| scenario | dataset_profile | method | baseline_seconds_median | candidate_seconds_median | delta_seconds | percent_change | speedup_ratio | sample_count | pass_or_fail | notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report.to_dict("records"):
        lines.append(
            "| {scenario} | {dataset_profile} | {method} | {baseline_seconds_median:.6f} | "
            "{candidate_seconds_median:.6f} | {delta_seconds:.6f} | {percent_change:.2f} | "
            "{speedup_ratio:.3f} | {sample_count} | {pass_or_fail} | {notes} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[candidate] wrote benchmark delta report to {csv_path}")
    print(f"[candidate] wrote markdown benchmark delta report to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 recursive clustering benchmark harness. "
            "Run the baseline phase before refactoring the clustering core, "
            "then run the candidate phase afterwards to compute delta metrics."
        )
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=("baseline", "candidate"),
        help="Whether to capture baseline timings or post-refactor candidate timings.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where raw samples, correctness snapshots, and reports are written.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=MEASURED_RUNS,
        help="Number of measured runs collected after one warm-up pass for each benchmark case.",
    )
    args = parser.parse_args()

    if int(args.measured_runs) < 5:
        raise ValueError("The Stage 2 timing protocol requires at least 5 measured runs.")

    results_dir = Path(args.results_dir).resolve()
    _write_config(results_dir)
    profiles = _load_profiles()
    cases = _build_cases(profiles)

    sample_df, summary_df = _phase_benchmark(
        phase=args.phase,
        cases=cases,
        results_dir=results_dir,
        measured_runs=int(args.measured_runs),
    )
    if args.phase == "candidate":
        report = _build_candidate_report(
            results_dir=results_dir,
            candidate_samples=sample_df,
            candidate_summary=summary_df,
        )
        _write_candidate_report(results_dir, report)


if __name__ == "__main__":
    main()
