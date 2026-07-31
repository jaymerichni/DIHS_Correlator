"""Packaged Flask interface for the DIHS Tephra Correlator."""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

os.environ.setdefault("MPLBACKEND", "Agg")

from DIHS_Correlator import (  # noqa: E402
    __version__,
    perturbative_simple_run,
    perturbative_triple_run,
    perturbative_triple_run_with_resolvedness,
    simple_run,
    triple_run,
)


MODEL_OPTIONS = ["agglomerative", "kmeans", "gaussian"]
TRANSFORM_OPTIONS = ["clr", "ilr", "scaled", "none"]
MODE_CONFIGS = {
    "simple_run": {
        "label": "Single-model DIHS run",
        "default_output_dir": "./Results",
    },
    "triple_run": {
        "label": "Three-model DIHS run",
        "default_output_dir": "./Results_triple",
    },
    "perturbative_simple_run": {
        "label": "Perturbative single-model ensemble",
        "default_output_dir": "./Results_perturbative",
    },
    "perturbative_triple_run": {
        "label": "Perturbative three-model ensemble",
        "default_output_dir": "./Results_perturbative_triple",
    },
    "perturbative_triple_run_with_resolvedness": {
        "label": "Perturbative three-model ensemble with resolvedness calibration",
        "default_output_dir": "./Results_perturbative_triple_resolvedness",
    },
}
MODE_OPTIONS = list(MODE_CONFIGS)
DEFAULT_OUTPUT_DIRS = {
    mode: config["default_output_dir"] for mode, config in MODE_CONFIGS.items()
}
MODEL_REQUIRED_MODES = {"simple_run", "perturbative_simple_run"}
PERTURBATIVE_MODES = {
    "perturbative_simple_run",
    "perturbative_triple_run",
    "perturbative_triple_run_with_resolvedness",
}
RESOLVEDNESS_MODES = {"perturbative_triple_run_with_resolvedness"}
DISPLAY_TABLE_KEYS = {
    "hs_per_depth",
    "dihs_total",
    "summary",
    "top1_candidate_summary",
    "resolvedness_summary",
    "pairwise_total_matrix",
    "hs_mean_per_depth",
    "dihs_summary",
    "top1_frequency",
    "margin_summary",
    "thresholds_by_target_precision",
    "perturbative_calibration_summary",
    "perturbative_regime_summary",
    "pairwise_total_mean_matrix",
}
RESOLVEDNESS_RESULT_SENTINELS = {"models", "target_precisions", "unknown_class"}
PLOT_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg"}
CACHE_TTL_SECONDS = 6 * 60 * 60
APP_TITLE = "DIHS Tephra Correlator"
SOFTWARE_DOI_ENV = "DIHS_CORRELATOR_SOFTWARE_DOI"
OUTPUT_ROOT_ENV = "DIHS_CORRELATOR_OUTPUT_ROOT"
HOST_ENV = "DIHS_CORRELATOR_HOST"
PORT_ENV = "DIHS_CORRELATOR_PORT"
DEFAULT_OUTPUT_ROOT_NAME = "dihs_outputs"
TABLE_LABEL_OVERRIDES = {
    "hs_per_depth": "HS Per-Depth Table",
    "dihs_total": "DIHS Summary",
    "hs_mean_per_depth": "Mean HS Per-Depth Table",
    "dihs_summary": "Perturbative DIHS Summary",
    "top1_frequency": "Top-1 Frequency Summary",
    "margin_summary": "Margin Summary",
    "pairwise_total_matrix": "Pairwise Total Matrix",
    "pairwise_total_mean_matrix": "Mean Pairwise Matrix",
    "thresholds_by_target_precision": "Thresholds By Target Precision",
    "perturbative_calibration_summary": "Perturbative Calibration Summary",
    "perturbative_regime_summary": "Perturbative Regime Summary",
}
PLOT_LABEL_OVERRIDES = {
    "hs_curve_path": "HS Curve",
    "pairwise_total_plot_path": "Pairwise DIHS Matrix",
    "mean_hs_curve_path": "Mean HS Curve",
    "top1_fraction_plot_path": "Top-1 Frequency Plot",
    "pairwise_total_mean_plot_path": "Mean Pairwise DIHS Matrix",
    "margin_plot_path": "Pseudo-Unknown Margin Comparison",
    "margin_histogram_path": "Pseudo-Unknown Margin Histogram",
    "threshold_plot_path": "Threshold Diagnostics",
    "margin_comparison_plot_path": "Resolvedness Margin Comparison",
    "calibration_overlay_plot_path": "Resolvedness Calibration Overlay",
    "output_path": "Plot",
    "plot_output_path": "Plot",
}
PATH_PART_LABEL_OVERRIDES = {
    "agglomerative": "Agglomerative",
    "kmeans": "KMeans",
    "gaussian": "Gaussian",
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("DIHS_CORRELATOR_FLASK_SECRET", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

CACHE_ROOT = Path(tempfile.gettempdir()) / "dihs_correlator_flask"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

DATASET_CACHE: dict[str, dict[str, Any]] = {}
RESULT_CACHE: dict[str, dict[str, Any]] = {}
JOB_CACHE: dict[str, dict[str, Any]] = {}
CACHE_LOCK = threading.RLock()
JOB_LOCK = threading.Lock()


def _sorted_values(values: pd.Series) -> list[Any]:
    return sorted(
        (_to_python_scalar(value) for value in values.dropna().unique().tolist()),
        key=lambda value: str(value),
    )


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _parse_optional_int(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    return int(text)


def _parse_float_list(value: str) -> list[float] | None:
    text = value.strip()
    if not text:
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def _configured_output_root() -> Path:
    configured_root = os.environ.get(OUTPUT_ROOT_ENV, "").strip()
    output_root = Path(configured_root) if configured_root else (Path.cwd() / DEFAULT_OUTPUT_ROOT_NAME)
    return output_root.resolve()


def _resolve_output_dir(value: str, default_dir: str) -> str:
    requested_text = value.strip() or default_dir
    output_root = _configured_output_root()
    requested_path = Path(requested_text)
    candidate = (
        requested_path.resolve()
        if requested_path.is_absolute()
        else (output_root / requested_path).resolve()
    )
    try:
        candidate.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Output directories must stay within '{output_root}'. "
            f"Set {OUTPUT_ROOT_ENV} to change the base directory."
        ) from exc
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _cleanup_cache() -> None:
    cutoff = time.time() - CACHE_TTL_SECONDS
    expired_temp_roots: list[Path] = []
    with CACHE_LOCK:
        expired_dataset_ids = [
            dataset_id
            for dataset_id, entry in DATASET_CACHE.items()
            if entry["updated_at"] < cutoff
        ]
        for dataset_id in expired_dataset_ids:
            DATASET_CACHE.pop(dataset_id, None)

        expired_result_ids = [
            result_id
            for result_id, entry in RESULT_CACHE.items()
            if entry["updated_at"] < cutoff
        ]
        for result_id in expired_result_ids:
            entry = RESULT_CACHE.pop(result_id, None)
            temp_root = entry.get("temp_root") if entry else None
            if temp_root:
                expired_temp_roots.append(temp_root)

    for temp_root in expired_temp_roots:
        shutil.rmtree(temp_root, ignore_errors=True)

    with JOB_LOCK:
        expired_job_ids = [
            job_id
            for job_id, entry in JOB_CACHE.items()
            if entry["updated_at"] < cutoff
        ]
        for job_id in expired_job_ids:
            entry = JOB_CACHE.pop(job_id, None)
            temp_root = entry.get("temp_root") if entry else None
            if temp_root:
                shutil.rmtree(temp_root, ignore_errors=True)


@app.before_request
def _before_request() -> None:
    _cleanup_cache()


def _build_unknown_value_maps(
    df: pd.DataFrame,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    option_map: dict[str, list[dict[str, str]]] = {}
    value_map: dict[str, dict[str, Any]] = {}
    for column in df.columns:
        options = []
        values = {}
        for index, value in enumerate(_sorted_values(df[column])):
            token = f"{column}::{index}"
            options.append({"token": token, "label": str(value)})
            values[token] = value
        option_map[column] = options
        value_map[column] = values
    return option_map, value_map


def _default_form_state(dataset_entry: dict[str, Any]) -> dict[str, Any]:
    columns = dataset_entry["columns"]
    class_column = "controlcode" if "controlcode" in columns else columns[0]
    unknown_options = dataset_entry["unknown_options"].get(class_column, [])
    unknown_token = unknown_options[0]["token"] if unknown_options else ""
    return {
        "mode": "simple_run",
        "class_column": class_column,
        "unknown_sample_token": unknown_token,
        "transform_type": "clr",
        "max_depth": 100,
        "seed_enabled": False,
        "random_state": 42,
        "compute_pairwise": True,
        "plot_everything": False,
        "write_files": False,
        "exclude_columns": [],
        "output_dir": DEFAULT_OUTPUT_DIRS["simple_run"],
        "plot_output_dir": "",
        "save_cluster_data": False,
        "save_untransformed": False,
        "model_type": MODEL_OPTIONS[0],
        "n_iterations": 100,
        "major_error": 0.02,
        "trace_error": 0.10,
        "perturbation_seed_text": "",
        "integration_depth_text": "",
        "pseudo_unknown_iterations": 100,
        "pseudo_unknown_sample_size_text": "",
        "pseudo_unknown_random_state_text": "",
        "target_precisions_text": "",
        "min_runs_above_threshold": 1,
        "major_cols": [],
        "trace_cols": [],
    }


def _build_dataset_entry(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    unknown_options, unknown_lookup = _build_unknown_value_maps(df)
    dataset_id = uuid4().hex
    entry = {
        "id": dataset_id,
        "filename": filename,
        "df": df,
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "preview_html": df.head(50).to_html(
            index=False,
            border=0,
            classes=["dataframe", "preview-table"],
            na_rep="",
        ),
        "unknown_options": unknown_options,
        "unknown_lookup": unknown_lookup,
        "updated_at": time.time(),
    }
    with CACHE_LOCK:
        DATASET_CACHE[dataset_id] = entry
    return entry


def _get_dataset_entry(dataset_id: str | None) -> dict[str, Any] | None:
    if not dataset_id:
        return None
    with CACHE_LOCK:
        entry = DATASET_CACHE.get(dataset_id)
        if entry is not None:
            entry["updated_at"] = time.time()
        return entry


def _get_result_entry(result_id: str | None) -> dict[str, Any] | None:
    if not result_id:
        return None
    with CACHE_LOCK:
        entry = RESULT_CACHE.get(result_id)
        if entry is not None:
            entry["updated_at"] = time.time()
        return entry


def _get_job_entry(job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    with JOB_LOCK:
        entry = JOB_CACHE.get(job_id)
        if entry is not None:
            entry["updated_at"] = time.time()
            return dict(entry)
    return None


def _set_job_entry(job_id: str, entry: dict[str, Any]) -> None:
    with JOB_LOCK:
        JOB_CACHE[job_id] = entry


def _update_job(job_id: str, **fields: Any) -> None:
    with JOB_LOCK:
        entry = JOB_CACHE.get(job_id)
        if entry is None:
            return
        entry.update(fields)
        entry["updated_at"] = time.time()


def _make_job_progress_callback(job_id: str):
    def _progress(payload: dict[str, Any]) -> None:
        fraction = payload.get("fraction")
        current = payload.get("current")
        total = payload.get("total")
        if fraction is None and current is not None and total:
            fraction = current / float(total)

        fields: dict[str, Any] = {}
        if fraction is not None:
            fraction = min(max(float(fraction), 0.0), 1.0)
            fields["progress_fraction"] = fraction
            fields["progress_percent"] = int(round(fraction * 100.0))
        if "stage" in payload:
            fields["stage"] = payload.get("stage")
        if "message" in payload:
            fields["message"] = payload.get("message")
        if current is not None:
            fields["current"] = int(current)
        if total is not None:
            fields["total"] = int(total)
        _update_job(job_id, **fields)

    return _progress


def _bool_from_form(name: str) -> bool:
    return request.form.get(name) == "on"


def _validate_columns(selected: list[str], allowed: list[str], field_name: str) -> list[str]:
    allowed_set = set(allowed)
    invalid = [value for value in selected if value not in allowed_set]
    if invalid:
        raise ValueError(f"Invalid {field_name}: {', '.join(invalid)}")
    return selected


def _form_state_from_request(dataset_entry: dict[str, Any]) -> dict[str, Any]:
    state = _default_form_state(dataset_entry)
    state.update(
        {
            "mode": request.form.get("mode", state["mode"]).strip(),
            "class_column": request.form.get("class_column", state["class_column"]).strip(),
            "unknown_sample_token": request.form.get(
                "unknown_sample",
                state["unknown_sample_token"],
            ).strip(),
            "transform_type": request.form.get(
                "transform_type",
                state["transform_type"],
            ).strip(),
            "max_depth": request.form.get("max_depth", str(state["max_depth"])).strip(),
            "seed_enabled": _bool_from_form("seed_enabled"),
            "random_state": request.form.get(
                "random_state",
                str(state["random_state"]),
            ).strip(),
            "compute_pairwise": _bool_from_form("compute_pairwise"),
            "plot_everything": _bool_from_form("plot_everything"),
            "write_files": _bool_from_form("write_files"),
            "exclude_columns": request.form.getlist("exclude_columns"),
            "output_dir": request.form.get("output_dir", state["output_dir"]).strip(),
            "plot_output_dir": request.form.get(
                "plot_output_dir",
                state["plot_output_dir"],
            ).strip(),
            "save_cluster_data": _bool_from_form("save_cluster_data"),
            "save_untransformed": _bool_from_form("save_untransformed"),
            "model_type": request.form.get("model_type", state["model_type"]).strip(),
            "n_iterations": request.form.get(
                "n_iterations",
                str(state["n_iterations"]),
            ).strip(),
            "major_error": request.form.get(
                "major_error",
                str(state["major_error"]),
            ).strip(),
            "trace_error": request.form.get(
                "trace_error",
                str(state["trace_error"]),
            ).strip(),
            "perturbation_seed_text": request.form.get(
                "perturbation_seed",
                state["perturbation_seed_text"],
            ).strip(),
            "integration_depth_text": request.form.get(
                "integration_depth",
                state["integration_depth_text"],
            ).strip(),
            "pseudo_unknown_iterations": request.form.get(
                "pseudo_unknown_iterations",
                str(state["pseudo_unknown_iterations"]),
            ).strip(),
            "pseudo_unknown_sample_size_text": request.form.get(
                "pseudo_unknown_sample_size",
                state["pseudo_unknown_sample_size_text"],
            ).strip(),
            "pseudo_unknown_random_state_text": request.form.get(
                "pseudo_unknown_random_state",
                state["pseudo_unknown_random_state_text"],
            ).strip(),
            "target_precisions_text": request.form.get(
                "target_precisions",
                state["target_precisions_text"],
            ).strip(),
            "min_runs_above_threshold": request.form.get(
                "min_runs_above_threshold",
                str(state["min_runs_above_threshold"]),
            ).strip(),
            "major_cols": request.form.getlist("major_cols"),
            "trace_cols": request.form.getlist("trace_cols"),
        }
    )
    return state


def _run_selected_mode(
    mode: str,
    df: pd.DataFrame,
    common: dict[str, Any],
    advanced: dict[str, Any],
    progress_callback=None,
):
    if mode == "simple_run":
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "running",
                    "message": "Running single-model DIHS analysis",
                    "fraction": 0.1,
                }
            )
        result = simple_run(
            df=df,
            model_type=advanced["model_type"],
            **common,
            return_details=True,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "complete",
                    "message": "Single-model DIHS analysis complete",
                    "fraction": 1.0,
                }
            )
        return result

    if mode == "triple_run":
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "running",
                    "message": "Running three-model DIHS analysis",
                    "fraction": 0.1,
                }
            )
        result = triple_run(
            df=df,
            **common,
            return_details=True,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "complete",
                    "message": "Three-model DIHS analysis complete",
                    "fraction": 1.0,
                }
            )
        return result

    if mode == "perturbative_simple_run":
        return perturbative_simple_run(
            df=df,
            model_type=advanced["model_type"],
            **common,
            n_iterations=advanced["n_iterations"],
            major_cols=advanced["major_cols"],
            trace_cols=advanced["trace_cols"],
            major_error=advanced["major_error"],
            trace_error=advanced["trace_error"],
            perturbation_seed=advanced["perturbation_seed"],
            integration_depth=advanced["integration_depth"],
            return_details=True,
            progress_callback=progress_callback,
        )

    if mode == "perturbative_triple_run":
        return perturbative_triple_run(
            df=df,
            **common,
            n_iterations=advanced["n_iterations"],
            major_cols=advanced["major_cols"],
            trace_cols=advanced["trace_cols"],
            major_error=advanced["major_error"],
            trace_error=advanced["trace_error"],
            perturbation_seed=advanced["perturbation_seed"],
            integration_depth=advanced["integration_depth"],
            return_details=True,
            progress_callback=progress_callback,
        )

    return perturbative_triple_run_with_resolvedness(
        df=df,
        **common,
        n_iterations=advanced["n_iterations"],
        major_cols=advanced["major_cols"],
        trace_cols=advanced["trace_cols"],
        major_error=advanced["major_error"],
        trace_error=advanced["trace_error"],
        perturbation_seed=advanced["perturbation_seed"],
        pseudo_unknown_iterations=advanced["pseudo_unknown_iterations"],
        pseudo_unknown_sample_size=advanced["pseudo_unknown_sample_size"],
        pseudo_unknown_random_state=advanced["pseudo_unknown_random_state"],
        target_precisions=advanced["target_precisions"],
        min_runs_above_threshold=advanced["min_runs_above_threshold"],
        integration_depth=advanced["integration_depth"],
        return_details=True,
        progress_callback=progress_callback,
    )


def _parse_form_submission(dataset_entry: dict[str, Any]) -> dict[str, Any]:
    columns = dataset_entry["columns"]
    numeric_columns = dataset_entry["numeric_columns"]
    mode = request.form.get("mode", "simple_run").strip()
    if mode not in MODE_OPTIONS:
        raise ValueError(f"Unsupported mode: {mode}")

    class_column = request.form.get("class_column", "").strip()
    if class_column not in columns:
        raise ValueError("Pick a valid class column.")

    unknown_token = request.form.get("unknown_sample", "").strip()
    unknown_lookup = dataset_entry["unknown_lookup"].get(class_column, {})
    if unknown_token not in unknown_lookup:
        raise ValueError("Pick an unknown sample/class from the selected class column.")
    unknown_sample = unknown_lookup[unknown_token]

    transform_type = request.form.get("transform_type", "clr").strip()
    if transform_type not in TRANSFORM_OPTIONS:
        raise ValueError(f"Unsupported transform: {transform_type}")

    max_depth = int(request.form.get("max_depth", "100"))
    random_state = int(request.form.get("random_state", "42"))
    seed_enabled = _bool_from_form("seed_enabled")
    compute_pairwise = _bool_from_form("compute_pairwise")
    plot_everything = _bool_from_form("plot_everything")
    write_files = _bool_from_form("write_files")
    save_cluster_data = _bool_from_form("save_cluster_data")
    save_untransformed = _bool_from_form("save_untransformed")
    exclude_columns = _validate_columns(
        request.form.getlist("exclude_columns"),
        columns,
        "exclude columns",
    )
    output_dir_text = request.form.get("output_dir", DEFAULT_OUTPUT_DIRS[mode]).strip()
    plot_output_dir_text = request.form.get("plot_output_dir", "").strip()

    form_state = {
        "mode": mode,
        "class_column": class_column,
        "unknown_sample_token": unknown_token,
        "transform_type": transform_type,
        "max_depth": max_depth,
        "seed_enabled": seed_enabled,
        "random_state": random_state,
        "compute_pairwise": compute_pairwise,
        "plot_everything": plot_everything,
        "write_files": write_files,
        "exclude_columns": exclude_columns,
        "output_dir": output_dir_text,
        "plot_output_dir": plot_output_dir_text,
        "save_cluster_data": save_cluster_data,
        "save_untransformed": save_untransformed,
        "model_type": request.form.get("model_type", MODEL_OPTIONS[0]).strip(),
        "n_iterations": int(request.form.get("n_iterations", "100")),
        "major_error": float(request.form.get("major_error", "0.02")),
        "trace_error": float(request.form.get("trace_error", "0.10")),
        "perturbation_seed_text": request.form.get("perturbation_seed", "").strip(),
        "integration_depth_text": request.form.get("integration_depth", "").strip(),
        "pseudo_unknown_iterations": int(
            request.form.get("pseudo_unknown_iterations", "100")
        ),
        "pseudo_unknown_sample_size_text": request.form.get(
            "pseudo_unknown_sample_size", ""
        ).strip(),
        "pseudo_unknown_random_state_text": request.form.get(
            "pseudo_unknown_random_state", ""
        ).strip(),
        "target_precisions_text": request.form.get("target_precisions", "").strip(),
        "min_runs_above_threshold": int(
            request.form.get("min_runs_above_threshold", "1")
        ),
        "major_cols": _validate_columns(
            request.form.getlist("major_cols"),
            numeric_columns,
            "major columns",
        ),
        "trace_cols": _validate_columns(
            request.form.getlist("trace_cols"),
            numeric_columns,
            "trace columns",
        ),
    }

    model_type = form_state["model_type"]
    if mode in MODEL_REQUIRED_MODES and model_type not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported model: {model_type}")

    common = {
        "transform_type": transform_type,
        "unknown_sample": unknown_sample,
        "class_column": class_column,
        "random_state": random_state if seed_enabled else None,
        "compute_pairwise": compute_pairwise,
        "plot_everything": plot_everything,
        "write_files": write_files,
        "output_dir": "",
        "plot_output_dir": None,
        "max_depth": max_depth,
        "exclude_columns": tuple(exclude_columns),
        "save_cluster_data": save_cluster_data,
        "save_untransformed": save_untransformed,
        "verbose": True,
    }

    advanced = {
        "model_type": model_type,
        "n_iterations": form_state["n_iterations"],
        "major_cols": form_state["major_cols"] or None,
        "trace_cols": form_state["trace_cols"] or None,
        "major_error": form_state["major_error"],
        "trace_error": form_state["trace_error"],
        "perturbation_seed": _parse_optional_int(form_state["perturbation_seed_text"]),
        "pseudo_unknown_iterations": form_state["pseudo_unknown_iterations"],
        "pseudo_unknown_sample_size": _parse_optional_int(
            form_state["pseudo_unknown_sample_size_text"]
        ),
        "pseudo_unknown_random_state": _parse_optional_int(
            form_state["pseudo_unknown_random_state_text"]
        ),
        "target_precisions": _parse_float_list(form_state["target_precisions_text"]),
        "min_runs_above_threshold": form_state["min_runs_above_threshold"],
        "integration_depth": _parse_optional_int(form_state["integration_depth_text"]),
    }

    temp_root = CACHE_ROOT / "runs" / uuid4().hex
    temp_root.mkdir(parents=True, exist_ok=True)
    internal_output_dir = temp_root / "outputs"
    internal_output_dir.mkdir(parents=True, exist_ok=True)

    if write_files:
        common["output_dir"] = _resolve_output_dir(
            output_dir_text,
            DEFAULT_OUTPUT_DIRS[mode],
        )
        common["plot_output_dir"] = (
            _resolve_output_dir(
                plot_output_dir_text,
                f"{DEFAULT_OUTPUT_DIRS[mode]}_plots",
            )
            if plot_output_dir_text
            else None
        )
    else:
        common["output_dir"] = str(internal_output_dir)
        if plot_everything:
            common["plot_output_dir"] = (
                _resolve_output_dir(
                    plot_output_dir_text,
                    f"{DEFAULT_OUTPUT_DIRS[mode]}_plots",
                )
                if plot_output_dir_text
                else str((temp_root / "plots").resolve())
            )
        else:
            common["plot_output_dir"] = None

    return {
        "mode": mode,
        "dataset_df": dataset_entry["df"].copy(),
        "common": common,
        "advanced": advanced,
        "form_state": form_state,
        "temp_root": temp_root,
    }


def _display_part_label(part: str) -> str:
    return PATH_PART_LABEL_OVERRIDES.get(part, TABLE_LABEL_OVERRIDES.get(part, part.replace("_", " ").title()))


def _format_label(parts: tuple[str, ...]) -> str:
    cleaned = []
    for part in parts:
        if part.isdigit():
            cleaned.append(f"Item {int(part) + 1}")
        else:
            cleaned.append(_display_part_label(part))
    return " / ".join(cleaned)


def _is_threshold_detail_column(column: str) -> bool:
    name = str(column).strip().lower()
    return "threshold" in name and re.search(r"_\d+$", name) is not None


def _resolvedness_result_root(root_value: Any) -> bool:
    return isinstance(root_value, dict) and RESOLVEDNESS_RESULT_SENTINELS.issubset(root_value)


def _is_resolvedness_summary_path(root_value: Any, path: tuple[str, ...]) -> bool:
    if not path:
        return False
    if path[-1] == "resolvedness_summary":
        return True
    return path == ("summary",) and _resolvedness_result_root(root_value)


def _table_label_for_path(root_value: Any, path: tuple[str, ...]) -> str:
    if path == ("summary",) and _resolvedness_result_root(root_value):
        return "Resolvedness Summary"
    if path[-1] == "resolvedness_summary" and len(path) >= 2 and path[0] == "models":
        return f"{path[1].title()} Resolvedness Summary"
    if path[-1] in {"dihs_total", "dihs_summary"} and len(path) >= 3 and path[0] == "models":
        return f"{path[1].title()} {TABLE_LABEL_OVERRIDES[path[-1]]}"
    if (
        path[-1] in {"dihs_total", "dihs_summary"}
        and len(path) == 1
        and isinstance(root_value, dict)
        and "models" in root_value
    ):
        return f"Combined {TABLE_LABEL_OVERRIDES[path[-1]]}"
    return _format_label(path)


def _normalize_table_for_display(key: str, df: pd.DataFrame) -> pd.DataFrame:
    if key in {"pairwise_total_matrix", "pairwise_total_mean_matrix"}:
        return df.reset_index()
    return df


def _plot_label_for_path(path: tuple[str, ...], file_path: Path) -> str:
    if not path:
        return file_path.stem.replace("_", " ").title()

    meaningful_path = tuple(part for part in path if part != "artifacts")
    if not meaningful_path:
        return file_path.stem.replace("_", " ").title()

    plot_key = meaningful_path[-1]
    label = PLOT_LABEL_OVERRIDES.get(plot_key)
    if label is None:
        return _format_label(meaningful_path)

    if len(meaningful_path) >= 3 and meaningful_path[0] == "models":
        return f"{_display_part_label(meaningful_path[1])} {label}"
    return label


def _should_collect_table(root_value: Any, path: tuple[str, ...]) -> bool:
    if not path or path[-1] not in DISPLAY_TABLE_KEYS:
        return False
    if path[-1] in {"summary", "resolvedness_summary"}:
        return _is_resolvedness_summary_path(root_value, path)
    return True


def _collect_tables(
    value: Any,
    path: tuple[str, ...] = (),
    collected: list[dict[str, Any]] | None = None,
    root_value: Any | None = None,
) -> list[dict[str, Any]]:
    if collected is None:
        collected = []
    if root_value is None:
        root_value = value
    if isinstance(value, pd.DataFrame):
        if _should_collect_table(root_value, path):
            table_df = _normalize_table_for_display(path[-1], value)
            if not table_df.empty and _is_resolvedness_summary_path(root_value, path):
                keep_columns = [
                    column
                    for column in table_df.columns
                    if not _is_threshold_detail_column(str(column))
                ]
                table_df = table_df.loc[:, keep_columns]
            collected.append(
                {
                    "path": path,
                    "label": _table_label_for_path(root_value, path),
                    "df": table_df,
                }
            )
        return collected
    if isinstance(value, dict):
        for key, nested in value.items():
            _collect_tables(nested, path + (str(key),), collected, root_value)
        return collected
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _collect_tables(nested, path + (str(index),), collected, root_value)
    return collected


def _collect_plot_paths(
    value: Any,
    path: tuple[str, ...] = (),
    collected: list[dict[str, Any]] | None = None,
    seen: set[Path] | None = None,
) -> list[dict[str, Any]]:
    if collected is None:
        collected = []
    if seen is None:
        seen = set()

    if isinstance(value, pd.DataFrame):
        return collected
    if isinstance(value, dict):
        for key, nested in value.items():
            _collect_plot_paths(nested, path + (str(key),), collected, seen)
        return collected
    if isinstance(value, (list, tuple, set)):
        for index, nested in enumerate(value):
            _collect_plot_paths(nested, path + (str(index),), collected, seen)
        return collected
    if isinstance(value, (str, os.PathLike)):
        file_path = Path(value)
        if file_path.suffix.lower() in PLOT_SUFFIXES and file_path.exists():
            resolved = file_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                collected.append(
                    {
                        "path": resolved,
                        "label": _plot_label_for_path(path, file_path),
                        "name": file_path.name,
                    }
                )
    return collected


def _collect_artifact_files(
    value: Any,
    path: tuple[str, ...] = (),
    collected: list[dict[str, Any]] | None = None,
    seen: set[Path] | None = None,
) -> list[dict[str, Any]]:
    if collected is None:
        collected = []
    if seen is None:
        seen = set()
    if isinstance(value, pd.DataFrame):
        return collected
    if isinstance(value, dict):
        for key, nested in value.items():
            _collect_artifact_files(nested, path + (str(key),), collected, seen)
        return collected
    if isinstance(value, (list, tuple, set)):
        for index, nested in enumerate(value):
            _collect_artifact_files(nested, path + (str(index),), collected, seen)
        return collected
    if isinstance(value, (str, os.PathLike)):
        file_path = Path(value)
        if file_path.exists() and file_path.is_file() and file_path.suffix.lower() not in PLOT_SUFFIXES:
            resolved = file_path.resolve()
            if resolved in seen:
                return collected
            seen.add(resolved)
            collected.append(
                {
                    "path": resolved,
                    "label": _format_label(path) if path else file_path.name,
                    "display_path": str(resolved),
                }
            )
    return collected


def _summarize_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "rows": int(len(value)),
            "columns": value.columns.tolist(),
        }
    if isinstance(value, dict):
        return {str(key): _summarize_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(nested) for nested in value]
    if isinstance(value, set):
        return [_summarize_value(nested) for nested in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return _to_python_scalar(value)


def _store_result(
    *,
    result: dict[str, Any],
    dataset_id: str,
    form_state: dict[str, Any],
    logs: str,
    temp_root: Path,
    output_dir: str,
    write_files: bool,
) -> str:
    result_id = uuid4().hex
    table_entries = []
    table_lookup = {}
    table_filenames = {}
    for index, table in enumerate(_collect_tables(result), start=1):
        table_id = f"table_{index}"
        table_df = table["df"]
        filename = f"{table['path'][-1]}.csv"
        table_lookup[table_id] = table_df
        table_filenames[table_id] = filename
        table_entries.append(
            {
                "id": table_id,
                "label": table["label"],
                "filename": filename,
                "html": table_df.to_html(
                    index=False,
                    border=0,
                    classes=["dataframe", "result-table"],
                    na_rep="",
                ),
            }
        )

    plot_entries = []
    plot_lookup = {}
    for index, plot in enumerate(_collect_plot_paths(result), start=1):
        plot_id = f"plot_{index}"
        plot_lookup[plot_id] = plot["path"]
        plot_entries.append(
            {
                "id": plot_id,
                "label": plot["label"],
                "name": plot["name"],
                "is_svg": plot["path"].suffix.lower() == ".svg",
            }
        )

    artifact_entries = []
    artifact_lookup = {}
    for index, artifact in enumerate(_collect_artifact_files(result), start=1):
        artifact_id = f"artifact_{index}"
        artifact_lookup[artifact_id] = artifact["path"]
        artifact_entries.append(
            {
                "id": artifact_id,
                "label": artifact["label"],
                "display_path": artifact["display_path"],
                "name": artifact["path"].name,
            }
        )

    with CACHE_LOCK:
        RESULT_CACHE[result_id] = {
            "id": result_id,
            "dataset_id": dataset_id,
            "form_state": form_state,
            "logs": logs,
            "tables": table_entries,
            "table_lookup": table_lookup,
            "table_filenames": table_filenames,
            "plots": plot_entries,
            "plot_lookup": plot_lookup,
            "artifacts": artifact_entries,
            "artifact_lookup": artifact_lookup,
            "details_json": json.dumps(_summarize_value(result), indent=2, default=str),
            "output_dir": output_dir,
            "write_files": write_files,
            "temp_root": temp_root,
            "updated_at": time.time(),
        }
    return result_id


def _job_payload(job_entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if job_entry is None:
        return None
    return {
        "id": job_entry["id"],
        "dataset_id": job_entry["dataset_id"],
        "mode": job_entry["mode"],
        "mode_label": job_entry["mode_label"],
        "status": job_entry["status"],
        "stage": job_entry.get("stage"),
        "message": job_entry.get("message"),
        "progress_percent": int(job_entry.get("progress_percent", 0)),
        "current": job_entry.get("current"),
        "total": job_entry.get("total"),
        "result_id": job_entry.get("result_id"),
        "error": job_entry.get("error"),
        "logs": job_entry.get("logs", ""),
    }


def _run_analysis_job(
    *,
    job_id: str,
    dataset_id: str,
    parsed: dict[str, Any],
) -> None:
    stdout_buffer = io.StringIO()
    progress_callback = _make_job_progress_callback(job_id)
    try:
        _update_job(
            job_id,
            status="running",
            stage="starting",
            message=f"Starting {MODE_CONFIGS[parsed['mode']]['label']}",
            progress_fraction=0.01,
            progress_percent=1,
        )
        with redirect_stdout(stdout_buffer):
            result = _run_selected_mode(
                parsed["mode"],
                parsed["dataset_df"].copy(),
                parsed["common"],
                parsed["advanced"],
                progress_callback=progress_callback,
            )

        logs = stdout_buffer.getvalue().strip()
        result_id = _store_result(
            result=result,
            dataset_id=dataset_id,
            form_state=parsed["form_state"],
            logs=logs,
            temp_root=parsed["temp_root"],
            output_dir=parsed["common"]["output_dir"],
            write_files=parsed["common"]["write_files"],
        )
        _update_job(
            job_id,
            status="completed",
            stage="complete",
            message="Analysis complete",
            progress_fraction=1.0,
            progress_percent=100,
            result_id=result_id,
            logs=logs,
            temp_root=None,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            stage="error",
            message=f"Analysis failed: {exc}",
            error=str(exc),
            logs=stdout_buffer.getvalue().strip(),
        )


def _render_page(
    *,
    dataset_entry: dict[str, Any] | None = None,
    result_entry: dict[str, Any] | None = None,
    job_entry: dict[str, Any] | None = None,
    form_state: dict[str, Any] | None = None,
    run_error: str | None = None,
    run_logs: str = "",
):
    if dataset_entry is None:
        dataset_payload = None
        form_payload = None
        selected_unknown_options = []
    else:
        form_payload = form_state or (
            result_entry["form_state"] if result_entry is not None else _default_form_state(dataset_entry)
        )
        selected_unknown_options = dataset_entry["unknown_options"].get(
            form_payload["class_column"],
            [],
        )
        dataset_payload = {
            "id": dataset_entry["id"],
            "filename": dataset_entry["filename"],
            "rows": dataset_entry["rows"],
            "columns_count": dataset_entry["columns_count"],
            "columns": dataset_entry["columns"],
            "numeric_columns": dataset_entry["numeric_columns"],
            "preview_html": dataset_entry["preview_html"],
            "unknown_options": dataset_entry["unknown_options"],
        }

    return render_template(
        "index.html",
        app_title=APP_TITLE,
        dataset=dataset_payload,
        package_version=__version__,
        result=result_entry,
        job=_job_payload(job_entry),
        form_state=form_payload,
        selected_unknown_options=selected_unknown_options,
        software_doi=os.environ.get(SOFTWARE_DOI_ENV, "").strip(),
        mode_options=[
            {"value": mode, "label": MODE_CONFIGS[mode]["label"]} for mode in MODE_OPTIONS
        ],
        model_options=MODEL_OPTIONS,
        transform_options=TRANSFORM_OPTIONS,
        default_output_dirs=DEFAULT_OUTPUT_DIRS,
        model_required_modes=sorted(MODEL_REQUIRED_MODES),
        perturbative_modes=sorted(PERTURBATIVE_MODES),
        resolvedness_modes=sorted(RESOLVEDNESS_MODES),
        run_error=run_error,
        run_logs=run_logs,
    )


@app.route("/", methods=["GET"])
def index():
    dataset_entry = _get_dataset_entry(request.args.get("dataset"))
    result_entry = _get_result_entry(request.args.get("result"))
    job_entry = _get_job_entry(request.args.get("job"))
    if job_entry is not None and dataset_entry is None:
        dataset_entry = _get_dataset_entry(job_entry["dataset_id"])
    if result_entry is None and job_entry is not None and job_entry.get("result_id"):
        result_entry = _get_result_entry(job_entry["result_id"])
    if result_entry is not None and dataset_entry is None:
        dataset_entry = _get_dataset_entry(result_entry["dataset_id"])
    if result_entry is not None and dataset_entry is not None:
        return _render_page(
            dataset_entry=dataset_entry,
            result_entry=result_entry,
            job_entry=job_entry,
        )
    if dataset_entry is not None:
        run_error = None
        run_logs = ""
        if job_entry is not None and job_entry["status"] == "error":
            run_error = job_entry.get("message")
            run_logs = job_entry.get("logs", "")
        return _render_page(
            dataset_entry=dataset_entry,
            job_entry=job_entry,
            run_error=run_error,
            run_logs=run_logs,
        )
    return _render_page()


@app.route("/load-data", methods=["POST"])
def load_data():
    uploaded_file = request.files.get("dataset")
    if uploaded_file is None or not uploaded_file.filename:
        flash("Choose a CSV file to continue.", "error")
        return _render_page()

    try:
        df = _read_uploaded_csv(uploaded_file)
    except Exception as exc:
        flash(f"Could not read CSV: {exc}", "error")
        return _render_page()

    if df.empty:
        flash("The uploaded CSV is empty.", "error")
        return _render_page()

    dataset_entry = _build_dataset_entry(df, uploaded_file.filename)
    return redirect(url_for("index", dataset=dataset_entry["id"]))


@app.route("/run", methods=["POST"])
def run_analysis():
    dataset_entry = _get_dataset_entry(request.form.get("dataset_id"))
    if dataset_entry is None:
        flash("Upload a CSV file again so the app can rebuild the analysis form.", "error")
        return redirect(url_for("index"))

    try:
        parsed = _parse_form_submission(dataset_entry)
    except ValueError as exc:
        return _render_page(
            dataset_entry=dataset_entry,
            form_state=_form_state_from_request(dataset_entry),
            run_error=str(exc),
        )

    job_id = uuid4().hex
    _set_job_entry(
        job_id,
        {
            "id": job_id,
            "dataset_id": dataset_entry["id"],
            "mode": parsed["mode"],
            "mode_label": MODE_CONFIGS[parsed["mode"]]["label"],
            "form_state": parsed["form_state"],
            "status": "queued",
            "stage": "queued",
            "message": f"Queued {MODE_CONFIGS[parsed['mode']]['label']}",
            "progress_fraction": 0.0,
            "progress_percent": 0,
            "current": None,
            "total": None,
            "result_id": None,
            "error": None,
            "logs": "",
            "temp_root": parsed["temp_root"],
            "updated_at": time.time(),
        },
    )
    worker = threading.Thread(
        target=_run_analysis_job,
        kwargs={
            "job_id": job_id,
            "dataset_id": dataset_entry["id"],
            "parsed": parsed,
        },
        daemon=True,
    )
    worker.start()
    return redirect(url_for("index", dataset=dataset_entry["id"], job=job_id))


@app.route("/jobs/<job_id>/status", methods=["GET"])
def job_status(job_id: str):
    job_entry = _get_job_entry(job_id)
    if job_entry is None:
        abort(404)
    return jsonify(_job_payload(job_entry))


@app.route("/downloads/tables/<result_id>/<table_id>.csv", methods=["GET"])
def download_table(result_id: str, table_id: str):
    result_entry = _get_result_entry(result_id)
    if result_entry is None:
        abort(404)
    table_df = result_entry["table_lookup"].get(table_id)
    if table_df is None:
        abort(404)
    csv_data = table_df.to_csv(index=False).encode("utf-8")
    filename = result_entry["table_filenames"].get(table_id, f"{table_id}.csv")
    return send_file(
        io.BytesIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/plots/<result_id>/<plot_id>", methods=["GET"])
def plot_file(result_id: str, plot_id: str):
    result_entry = _get_result_entry(result_id)
    if result_entry is None:
        abort(404)
    file_path = result_entry["plot_lookup"].get(plot_id)
    if file_path is None or not file_path.exists():
        abort(404)
    return send_file(file_path)


@app.route("/downloads/files/<result_id>/<artifact_id>", methods=["GET"])
def download_artifact(result_id: str, artifact_id: str):
    result_entry = _get_result_entry(result_id)
    if result_entry is None:
        abort(404)
    file_path = result_entry["artifact_lookup"].get(artifact_id)
    if file_path is None or not file_path.exists():
        abort(404)
    return send_file(file_path, as_attachment=True, download_name=file_path.name)


def create_app() -> Flask:
    """Return the configured Flask app instance for local use and testing."""
    return app


def main() -> None:
    """Run the single-process local development server."""
    host = os.environ.get(HOST_ENV, os.environ.get("HOST", "127.0.0.1"))
    port = int(os.environ.get(PORT_ENV, os.environ.get("PORT", "5000")))
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
