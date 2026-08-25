from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import copy
import warnings

import numpy as np
import pandas as pd


MODE_CODES = {"relative_envelope": 0, "standard_error": 1}
KIND_CODES = {"relative": 0, "absolute": 1}
DISTRIBUTION_CODES = {"uniform": 0, "normal": 1, "lognormal": 2}
SOURCE_CODES = {"missing": 0, "global": 1, "group": 2, "feature": 3, "cell": 4}

MODE_NAMES = {value: key for key, value in MODE_CODES.items()}
KIND_NAMES = {value: key for key, value in KIND_CODES.items()}
DISTRIBUTION_NAMES = {value: key for key, value in DISTRIBUTION_CODES.items()}
SOURCE_NAMES = {value: key for key, value in SOURCE_CODES.items()}

DEFAULT_MODE = "relative_envelope"
DEFAULT_KIND = "relative"
DEFAULT_DISTRIBUTION = "uniform"
DEFAULT_POSITIVE_ONLY = True
MIN_POSITIVE_VALUE = np.nextafter(0.0, 1.0)


@dataclass(frozen=True)
class CompiledUncertaintyModel:
    feature_columns: tuple[str, ...]
    magnitudes: np.ndarray
    mode_codes: np.ndarray
    kind_codes: np.ndarray
    distribution_codes: np.ndarray
    positive_only_mask: np.ndarray
    source_codes: np.ndarray
    source_name_codes: np.ndarray
    source_names: tuple[str, ...]
    row_id_column: str | None
    summary: dict[str, Any]


def build_effective_uncertainty_config(
    *,
    uncertainty_config: dict[str, Any] | None,
    major_cols: list[str] | None,
    trace_cols: list[str] | None,
    major_error: float,
    trace_error: float,
) -> dict[str, Any]:
    legacy_config = {
        "defaults": {
            "mode": DEFAULT_MODE,
            "kind": DEFAULT_KIND,
            "distribution": DEFAULT_DISTRIBUTION,
            "positive_only": True,
        },
        "groups": {},
    }
    if major_cols:
        legacy_config["groups"]["major"] = {
            "columns": list(major_cols),
            "value": float(major_error),
        }
    if trace_cols:
        legacy_config["groups"]["trace"] = {
            "columns": list(trace_cols),
            "value": float(trace_error),
        }

    if uncertainty_config is None:
        return legacy_config

    merged = copy.deepcopy(legacy_config)
    user_config = copy.deepcopy(dict(uncertainty_config))
    if "defaults" in user_config:
        merged.setdefault("defaults", {}).update(user_config.pop("defaults") or {})
    if "global" in user_config:
        merged["global"] = user_config.pop("global")
    if "groups" in user_config:
        merged.setdefault("groups", {}).update(user_config.pop("groups") or {})
    if "features" in user_config:
        merged.setdefault("features", {}).update(user_config.pop("features") or {})
    for key in ("cell_values", "cell_table", "row_id_column"):
        if key in user_config:
            merged[key] = user_config.pop(key)
    if user_config:
        merged.update(user_config)
    return merged


def compile_uncertainty_model(
    df: pd.DataFrame,
    *,
    class_column: str,
    uncertainty_config: dict[str, Any] | None,
    major_cols: list[str] | None,
    trace_cols: list[str] | None,
    major_error: float,
    trace_error: float,
) -> CompiledUncertaintyModel:
    feature_columns = tuple(
        df.select_dtypes(include="number").columns.drop(class_column, errors="ignore").tolist()
    )
    n_rows = len(df)
    n_features = len(feature_columns)

    magnitudes = np.full((n_rows, n_features), np.nan, dtype=float)
    mode_codes = np.full((n_rows, n_features), MODE_CODES[DEFAULT_MODE], dtype=np.int8)
    kind_codes = np.full((n_rows, n_features), KIND_CODES[DEFAULT_KIND], dtype=np.int8)
    distribution_codes = np.full(
        (n_rows, n_features), DISTRIBUTION_CODES[DEFAULT_DISTRIBUTION], dtype=np.int8
    )
    positive_only_mask = np.full((n_rows, n_features), DEFAULT_POSITIVE_ONLY, dtype=bool)
    source_codes = np.full((n_rows, n_features), SOURCE_CODES["missing"], dtype=np.int8)
    source_name_codes = np.zeros((n_rows, n_features), dtype=np.int16)
    source_name_to_code = {"missing": 0}
    source_names = ["missing"]
    warnings_list: list[str] = []

    config = build_effective_uncertainty_config(
        uncertainty_config=uncertainty_config,
        major_cols=major_cols,
        trace_cols=trace_cols,
        major_error=major_error,
        trace_error=trace_error,
    )
    defaults = _normalize_default_spec(config.get("defaults"))
    feature_index = {name: idx for idx, name in enumerate(feature_columns)}
    row_id_column = config.get("row_id_column")

    def _register_source_name(source_name: str) -> int:
        source_name = str(source_name)
        if source_name not in source_name_to_code:
            source_name_to_code[source_name] = len(source_names)
            source_names.append(source_name)
        return source_name_to_code[source_name]

    def _apply_spec_to_cells(
        *,
        row_positions: np.ndarray,
        col_positions: np.ndarray,
        spec: dict[str, Any],
        source_level: str,
        source_name: str,
    ) -> None:
        if row_positions.size == 0 or col_positions.size == 0:
            return
        magnitude = float(spec["value"])
        if magnitude < 0:
            raise ValueError("Uncertainty values must be >= 0.")
        mode_code = MODE_CODES[spec["mode"]]
        kind_code = KIND_CODES[spec["kind"]]
        distribution_code = DISTRIBUTION_CODES[spec["distribution"]]
        if distribution_code == DISTRIBUTION_CODES["lognormal"]:
            if kind_code != KIND_CODES["relative"]:
                raise ValueError("Lognormal uncertainty only supports kind='relative'.")
            if mode_code != MODE_CODES["standard_error"]:
                raise ValueError(
                    "Lognormal uncertainty currently supports mode='standard_error' only."
                )
        source_code = SOURCE_CODES[source_level]
        source_name_code = _register_source_name(source_name)
        rr = np.asarray(row_positions, dtype=int)
        cc = np.asarray(col_positions, dtype=int)
        magnitudes[np.ix_(rr, cc)] = magnitude
        mode_codes[np.ix_(rr, cc)] = mode_code
        kind_codes[np.ix_(rr, cc)] = kind_code
        distribution_codes[np.ix_(rr, cc)] = distribution_code
        positive_only_mask[np.ix_(rr, cc)] = bool(spec["positive_only"])
        source_codes[np.ix_(rr, cc)] = source_code
        source_name_codes[np.ix_(rr, cc)] = source_name_code

    global_spec_raw = config.get("global")
    if global_spec_raw is not None and n_features:
        global_spec = _normalize_source_spec(global_spec_raw, defaults)
        _apply_spec_to_cells(
            row_positions=np.arange(n_rows, dtype=int),
            col_positions=np.arange(n_features, dtype=int),
            spec=global_spec,
            source_level="global",
            source_name="global",
        )

    for group_name, group_spec_raw in (config.get("groups") or {}).items():
        if group_spec_raw is None:
            continue
        group_spec = _normalize_source_spec(group_spec_raw, defaults)
        columns = list(group_spec_raw.get("columns") or [])
        valid_columns = [col for col in columns if col in feature_index]
        ignored_columns = [col for col in columns if col not in feature_index]
        if ignored_columns:
            message = (
                f"Ignoring unresolved uncertainty columns for group '{group_name}': "
                f"{ignored_columns}"
            )
            warnings.warn(message, stacklevel=2)
            warnings_list.append(message)
        if not valid_columns:
            continue
        _apply_spec_to_cells(
            row_positions=np.arange(n_rows, dtype=int),
            col_positions=np.array([feature_index[col] for col in valid_columns], dtype=int),
            spec=group_spec,
            source_level="group",
            source_name=f"group:{group_name}",
        )

    for feature_name, feature_spec_raw in _iter_feature_specs(config.get("features")):
        if feature_name not in feature_index:
            message = f"Ignoring unresolved feature uncertainty for '{feature_name}'."
            warnings.warn(message, stacklevel=2)
            warnings_list.append(message)
            continue
        feature_spec = _normalize_source_spec(feature_spec_raw, defaults)
        _apply_spec_to_cells(
            row_positions=np.arange(n_rows, dtype=int),
            col_positions=np.array([feature_index[feature_name]], dtype=int),
            spec=feature_spec,
            source_level="feature",
            source_name=f"feature:{feature_name}",
        )

    for row_positions, feature_name, cell_spec_raw in _iter_cell_specs(
        df=df,
        feature_columns=feature_columns,
        cell_values=config.get("cell_values"),
        cell_table=config.get("cell_table"),
        row_id_column=row_id_column,
    ):
        if feature_name not in feature_index:
            message = f"Ignoring unresolved cell-wise uncertainty for feature '{feature_name}'."
            warnings.warn(message, stacklevel=2)
            warnings_list.append(message)
            continue
        cell_spec = _normalize_source_spec(cell_spec_raw, defaults)
        _apply_spec_to_cells(
            row_positions=row_positions,
            col_positions=np.array([feature_index[feature_name]], dtype=int),
            spec=cell_spec,
            source_level="cell",
            source_name=f"cell:{feature_name}",
        )

    coverage_mask = ~np.isnan(magnitudes)
    coverage_by_level = {
        SOURCE_NAMES[source_code]: int((source_codes == source_code).sum())
        for source_code in sorted(SOURCE_NAMES)
        if int((source_codes == source_code).sum()) > 0
    }
    source_name_counts = {
        source_names[code]: int((source_name_codes == code).sum())
        for code in range(len(source_names))
        if int((source_name_codes == code).sum()) > 0
    }
    missing_cells = int((~coverage_mask).sum())
    if missing_cells:
        message = (
            f"Uncertainty coverage is incomplete: {missing_cells} cell(s) will remain "
            "unperturbed because no cell, feature, group, or global uncertainty was provided."
        )
        warnings_list.append(message)

    summary = {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "feature_columns": list(feature_columns),
        "resolved_cells": int(coverage_mask.sum()),
        "missing_cells": missing_cells,
        "coverage_fraction": (
            float(coverage_mask.sum()) / float(coverage_mask.size) if coverage_mask.size else 0.0
        ),
        "coverage_by_level": coverage_by_level,
        "source_name_counts": source_name_counts,
        "warnings": warnings_list,
    }
    return CompiledUncertaintyModel(
        feature_columns=feature_columns,
        magnitudes=magnitudes,
        mode_codes=mode_codes,
        kind_codes=kind_codes,
        distribution_codes=distribution_codes,
        positive_only_mask=positive_only_mask,
        source_codes=source_codes,
        source_name_codes=source_name_codes,
        source_names=tuple(source_names),
        row_id_column=row_id_column if isinstance(row_id_column, str) else None,
        summary=summary,
    )


def uncertainty_summary_frame(model: CompiledUncertaintyModel) -> pd.DataFrame:
    rows = []
    for source_name, count in model.summary.get("source_name_counts", {}).items():
        if source_name == "missing":
            continue
        if ":" in source_name:
            level, detail = source_name.split(":", 1)
        else:
            level, detail = "global", source_name
        rows.append(
            {
                "source_level": level,
                "source_name": detail,
                "cell_count": int(count),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["source_level", "source_name", "cell_count"])
    return pd.DataFrame(rows).sort_values(
        ["source_level", "source_name"], kind="stable"
    ).reset_index(drop=True)


def perturb_dataframe(
    df: pd.DataFrame,
    *,
    model: CompiledUncertaintyModel,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if not model.feature_columns:
        return df.copy()
    perturbed = df.copy()
    base = perturbed.loc[:, list(model.feature_columns)].to_numpy(dtype=float, copy=True)
    sampled = sample_numeric_matrix(base, model=model, rng=rng)
    perturbed.loc[:, list(model.feature_columns)] = sampled
    return perturbed


def sample_numeric_matrix(
    base_values: np.ndarray,
    *,
    model: CompiledUncertaintyModel,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(base_values, dtype=float).copy()
    coverage_mask = ~np.isnan(model.magnitudes)
    if not coverage_mask.any():
        return values

    amplitudes = model.magnitudes.copy()
    relative_mask = model.kind_codes == KIND_CODES["relative"]
    amplitudes[relative_mask] = amplitudes[relative_mask] * np.abs(values[relative_mask])

    additive_masks = [
        (
            (model.mode_codes == MODE_CODES["relative_envelope"])
            & (model.distribution_codes == DISTRIBUTION_CODES["uniform"])
            & coverage_mask,
            "relative_envelope_uniform",
        ),
        (
            (model.mode_codes == MODE_CODES["relative_envelope"])
            & (model.distribution_codes == DISTRIBUTION_CODES["normal"])
            & coverage_mask,
            "relative_envelope_normal",
        ),
        (
            (model.mode_codes == MODE_CODES["standard_error"])
            & (model.distribution_codes == DISTRIBUTION_CODES["normal"])
            & coverage_mask,
            "standard_error_normal",
        ),
        (
            (model.mode_codes == MODE_CODES["standard_error"])
            & (model.distribution_codes == DISTRIBUTION_CODES["uniform"])
            & coverage_mask,
            "standard_error_uniform",
        ),
    ]
    for mask, mode_name in additive_masks:
        if not mask.any():
            continue
        amp = amplitudes[mask]
        if mode_name == "relative_envelope_uniform":
            noise = rng.uniform(-amp, amp)
        elif mode_name == "relative_envelope_normal":
            noise = rng.normal(loc=0.0, scale=amp / 2.0)
            noise = np.clip(noise, -amp, amp)
        elif mode_name == "standard_error_normal":
            noise = rng.normal(loc=0.0, scale=amp)
        else:
            radius = np.sqrt(3.0) * amp
            noise = rng.uniform(-radius, radius)
        values[mask] = values[mask] + noise

    lognormal_mask = (
        (model.mode_codes == MODE_CODES["standard_error"])
        & (model.distribution_codes == DISTRIBUTION_CODES["lognormal"])
        & coverage_mask
    )
    if lognormal_mask.any():
        sigma = np.maximum(model.magnitudes[lognormal_mask], 0.0)
        mu = -0.5 * np.square(sigma)
        factors = rng.lognormal(mean=mu, sigma=sigma)
        values[lognormal_mask] = values[lognormal_mask] * factors

    clip_mask = model.positive_only_mask & coverage_mask
    if clip_mask.any():
        values[clip_mask] = np.maximum(values[clip_mask], MIN_POSITIVE_VALUE)
    return values


def _normalize_default_spec(spec: Any) -> dict[str, Any]:
    if spec is None:
        return {
            "mode": DEFAULT_MODE,
            "kind": DEFAULT_KIND,
            "distribution": DEFAULT_DISTRIBUTION,
            "positive_only": DEFAULT_POSITIVE_ONLY,
        }
    if not isinstance(spec, dict):
        raise ValueError("uncertainty defaults must be a mapping.")
    normalized = {
        "mode": str(spec.get("mode", DEFAULT_MODE)).strip().lower(),
        "kind": str(spec.get("kind", DEFAULT_KIND)).strip().lower(),
        "distribution": str(spec.get("distribution", DEFAULT_DISTRIBUTION)).strip().lower(),
        "positive_only": bool(spec.get("positive_only", DEFAULT_POSITIVE_ONLY)),
    }
    _validate_semantics(normalized)
    return normalized


def _normalize_source_spec(spec: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    if isinstance(spec, (int, float, np.integer, np.floating)):
        normalized = dict(defaults)
        normalized["value"] = float(spec)
        return normalized
    if not isinstance(spec, dict):
        raise ValueError("Uncertainty specifications must be numbers or mappings.")
    if "value" not in spec:
        raise ValueError("Each uncertainty specification must include a 'value'.")
    normalized = dict(defaults)
    normalized["value"] = float(spec["value"])
    if "mode" in spec:
        normalized["mode"] = str(spec["mode"]).strip().lower()
    if "kind" in spec:
        normalized["kind"] = str(spec["kind"]).strip().lower()
    if "distribution" in spec:
        normalized["distribution"] = str(spec["distribution"]).strip().lower()
    if "positive_only" in spec:
        normalized["positive_only"] = bool(spec["positive_only"])
    _validate_semantics(normalized)
    return normalized


def _validate_semantics(spec: dict[str, Any]) -> None:
    if spec["mode"] not in MODE_CODES:
        raise ValueError(f"Unsupported uncertainty mode '{spec['mode']}'.")
    if spec["kind"] not in KIND_CODES:
        raise ValueError(f"Unsupported uncertainty kind '{spec['kind']}'.")
    if spec["distribution"] not in DISTRIBUTION_CODES:
        raise ValueError(
            f"Unsupported uncertainty distribution '{spec['distribution']}'."
        )


def _iter_feature_specs(features: Any):
    if features is None:
        return []
    if isinstance(features, dict):
        return list(features.items())
    if isinstance(features, pd.DataFrame):
        required = {"feature", "value"}
        if not required.issubset(features.columns):
            raise ValueError(
                "Feature uncertainty tables must contain at least 'feature' and 'value' columns."
            )
        rows = []
        for _, row in features.iterrows():
            rows.append((row["feature"], row.dropna().to_dict()))
        return rows
    raise ValueError("features uncertainty must be a mapping or dataframe.")


def _iter_cell_specs(
    *,
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    cell_values: Any,
    cell_table: Any,
    row_id_column: str | None,
):
    specs = []
    if cell_values is not None:
        specs.extend(
            _iter_cell_values_specs(
                df=df,
                feature_columns=feature_columns,
                cell_values=cell_values,
                row_id_column=row_id_column,
            )
        )
    if cell_table is not None:
        specs.extend(
            _iter_cell_table_specs(
                df=df,
                feature_columns=feature_columns,
                cell_table=cell_table,
                row_id_column=row_id_column,
            )
        )
    return specs


def _iter_cell_values_specs(
    *,
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    cell_values: Any,
    row_id_column: str | None,
):
    if isinstance(cell_values, pd.DataFrame):
        if {"feature", "value"}.issubset(cell_values.columns):
            return _iter_cell_table_specs(
                df=df,
                feature_columns=feature_columns,
                cell_table=cell_values,
                row_id_column=row_id_column,
            )
        aligned = cell_values.reindex(df.index)
        rows = []
        for feature_name in feature_columns:
            if feature_name not in aligned.columns:
                continue
            series = aligned[feature_name]
            valid_mask = series.notna().to_numpy()
            if not valid_mask.any():
                continue
            value_by_row = series[valid_mask]
            for row_position, value in zip(np.flatnonzero(valid_mask), value_by_row.to_numpy()):
                rows.append(
                    (
                        np.array([row_position], dtype=int),
                        feature_name,
                        {"value": float(value)},
                    )
                )
        return rows
    if isinstance(cell_values, dict):
        wide_df = pd.DataFrame(cell_values)
        if len(wide_df) == len(df):
            wide_df.index = df.index
        return _iter_cell_values_specs(
            df=df,
            feature_columns=feature_columns,
            cell_values=wide_df,
            row_id_column=row_id_column,
        )
    raise ValueError("cell_values must be a dataframe or mapping.")


def _iter_cell_table_specs(
    *,
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    cell_table: Any,
    row_id_column: str | None,
):
    if not isinstance(cell_table, pd.DataFrame):
        cell_table = pd.DataFrame(cell_table)
    required = {"feature", "value"}
    if not required.issubset(cell_table.columns):
        raise ValueError(
            "Cell uncertainty tables must contain at least 'feature' and 'value' columns."
        )

    row_token_column = None
    if row_id_column and row_id_column in cell_table.columns:
        row_token_column = row_id_column
    else:
        for candidate in ("row_id", "sample_id", "index"):
            if candidate in cell_table.columns:
                row_token_column = candidate
                break
    if row_token_column is None:
        raise ValueError(
            "Cell uncertainty tables require a row identifier column such as 'row_id', "
            "'sample_id', 'index', or the configured row_id_column."
        )

    row_lookup = _build_row_lookup(df, row_id_column=row_id_column, row_token_column=row_token_column)
    rows = []
    for _, row in cell_table.iterrows():
        feature_name = row["feature"]
        if feature_name not in feature_columns:
            continue
        row_token = _row_token(row[row_token_column])
        if row_token not in row_lookup:
            continue
        spec = row.drop(labels=[row_token_column]).dropna().to_dict()
        rows.append((np.array([row_lookup[row_token]], dtype=int), feature_name, spec))
    return rows


def _build_row_lookup(
    df: pd.DataFrame,
    *,
    row_id_column: str | None,
    row_token_column: str,
) -> dict[str, int]:
    if row_token_column == "index" or row_id_column is None:
        values = df.index.tolist()
    else:
        if row_id_column not in df.columns:
            raise ValueError(
                f"Configured row_id_column '{row_id_column}' was not found in the dataframe."
            )
        values = df[row_id_column].tolist()
    return {_row_token(value): int(position) for position, value in enumerate(values)}


def _row_token(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)
