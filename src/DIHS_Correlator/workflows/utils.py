"""Shared workflow utilities and helpers."""

from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd

from DIHS_Correlator.core.transforms import BASE_TRANSFORMATIONS

TRANSFORM_NAME_TO_ID = {v: k for k, v in BASE_TRANSFORMATIONS.items()}
DEFAULT_MAJOR_COLS = [
    "SIO2N",
    "TIO2N",
    "AL2O3N",
    "FE2O3TN",
    "CAON",
    "MGON",
    "MNON",
    "NA2ON",
    "K2ON",
    "P2O5N",
]
DEFAULT_TRACE_COLS = ["NbN", "ZrN", "LaN", "CeN", "SrN", "BaN", "RbN"]


def _log(verbose: bool, message: str):
    """Log a message if verbose is True."""
    if verbose:
        print(message)


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str | None = None,
    message: str | None = None,
    fraction: float | None = None,
    current: int | None = None,
    total: int | None = None,
):
    """Emit a progress update without letting UI concerns break the workflow."""
    if progress_callback is None:
        return

    payload: dict[str, Any] = {}
    if stage is not None:
        payload["stage"] = stage
    if message is not None:
        payload["message"] = message
    if fraction is not None:
        payload["fraction"] = min(max(float(fraction), 0.0), 1.0)
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)

    try:
        progress_callback(payload)
    except Exception:
        return


def _scale_progress_callback(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    start: float,
    end: float,
    prefix: str | None = None,
    stage_prefix: str | None = None,
):
    """Map a child workflow's 0..1 progress range into a parent range."""
    if progress_callback is None:
        return None

    span = float(end) - float(start)

    def _wrapped(payload: dict[str, Any]):
        child = dict(payload)
        if "fraction" in child and child["fraction"] is not None:
            inner = min(max(float(child["fraction"]), 0.0), 1.0)
            child["fraction"] = float(start) + span * inner
        if prefix:
            message = child.get("message")
            child["message"] = f"{prefix}: {message}" if message else prefix
        if stage_prefix and child.get("stage"):
            child["stage"] = f"{stage_prefix}.{child['stage']}"
        _emit_progress(progress_callback, **child)

    return _wrapped


def _print_progress(current: int, total: int, width: int = 30):
    """Print a progress bar to the console."""
    if total <= 0:
        return
    ratio = min(max(current / float(total), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def _normalize_transform_type(transform_type: str) -> int:
    """Normalize and validate a transform type string to its integer ID."""
    if not isinstance(transform_type, str):
        raise ValueError(
            "transform_type must be a string name: 'none', 'ilr', 'clr', or 'scaled'."
        )
    key = str(transform_type).strip().lower()
    if key not in TRANSFORM_NAME_TO_ID:
        raise ValueError(f"Unsupported transform_type='{transform_type}'.")
    return TRANSFORM_NAME_TO_ID[key]


def _resolve_unknown_class(
    df: pd.DataFrame, unknown_sample: Any, class_column: str = "controlcode"
) -> Any:
    """Resolve the unknown class value from the dataframe."""
    if unknown_sample is None:
        return 0
    if class_column not in df.columns:
        raise ValueError(f"Input dataframe must contain class column '{class_column}'.")

    control_values = df[class_column]
    if (control_values == unknown_sample).any():
        return unknown_sample

    if isinstance(unknown_sample, str):
        for caster in (int, float):
            try:
                parsed = caster(unknown_sample)
            except Exception:
                continue
            if (control_values == parsed).any():
                return parsed

    raise ValueError(
        f"Unknown sample '{unknown_sample}' was not found in class column '{class_column}'."
    )


def _prepare_working_df(df: pd.DataFrame, class_column: str) -> pd.DataFrame:
    """Prepare a working copy of the dataframe with validation."""
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in dataframe.")
    return df.copy()


def _class_key(value: Any) -> str:
    """Convert a class value to a consistent string key."""
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _class_match_mask(values: pd.Series, target: Any) -> pd.Series:
    """Create a boolean mask for class values matching a target."""
    target_key = _class_key(target)
    return values.apply(lambda value: _class_key(value) == target_key)


def _resolve_major_trace_columns(
    df: pd.DataFrame,
    major_cols: list[str] | None,
    trace_cols: list[str] | None,
    class_column: str = "controlcode",
):
    """Resolve major/trace perturbation columns against available numeric columns."""
    numeric_cols = set(
        df.select_dtypes(include="number").columns.drop(class_column, errors="ignore")
    )

    def _resolve_subset(
        subset_name: str,
        requested_cols: list[str] | None,
        default_cols: list[str],
    ) -> list[str]:
        if requested_cols is None:
            present_defaults = [c for c in default_cols if c in df.columns]
            return [c for c in present_defaults if c in numeric_cols]

        requested = list(requested_cols)
        if len(requested) == 0:
            return []

        missing = [c for c in requested if c not in df.columns]
        non_numeric = [c for c in requested if c in df.columns and c not in numeric_cols]
        resolved = [c for c in requested if c in numeric_cols]

        if not resolved:
            details = []
            if missing:
                details.append(f"missing columns: {missing}")
            if non_numeric:
                details.append(f"non-numeric columns: {non_numeric}")
            detail_text = " ".join(details) if details else "No matching numeric columns were found."
            raise ValueError(
                f"No valid {subset_name} columns were resolved from the explicit list. {detail_text}"
            )

        if missing or non_numeric:
            details = []
            if missing:
                details.append(f"missing columns: {missing}")
            if non_numeric:
                details.append(f"non-numeric columns: {non_numeric}")
            warnings.warn(
                f"Ignoring unresolved {subset_name} columns. Using {resolved}. "
                + " ".join(details),
                stacklevel=3,
            )

        return resolved

    resolved_major = _resolve_subset("major", major_cols, DEFAULT_MAJOR_COLS)
    resolved_trace = _resolve_subset("trace", trace_cols, DEFAULT_TRACE_COLS)

    if (
        (major_cols is None or trace_cols is None)
        and not resolved_major
        and not resolved_trace
    ):
        warnings.warn(
            "No perturbation columns were resolved. "
            "The perturbative run will proceed without feature perturbations. "
            "Pass major_cols and/or trace_cols explicitly if your dataset uses different column names.",
            stacklevel=3,
        )

    return resolved_major, resolved_trace
