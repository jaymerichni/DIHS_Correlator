"""Streamlit interface for the DIHS Correlator public API."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from DIHS_Correlator import (  # noqa: E402
    perturbative_triple_run_with_resolvedness,
    simple_run,
    triple_run,
)


MODEL_OPTIONS = ["agglomerative", "kmeans", "gaussian"]
TRANSFORM_OPTIONS = ["clr", "ilr", "scaled", "none"]
MODE_OPTIONS = [
    "simple_run",
    "triple_run",
    "perturbative_triple_run_with_resolvedness",
]


def _sorted_values(values: pd.Series) -> list[Any]:
    return sorted(values.dropna().unique().tolist(), key=lambda value: str(value))


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


def _download_dataframe(label: str, df: pd.DataFrame, file_name: str):
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def _show_dataframe(name: str, value: pd.DataFrame):
    st.subheader(name.replace("_", " ").title())
    st.dataframe(value, use_container_width=True)
    _download_dataframe(f"Download {name}.csv", value, f"{name}.csv")


def _show_artifacts(artifacts: dict[str, Any]):
    if not artifacts:
        return
    st.subheader("Saved Artifacts")
    for key, value in artifacts.items():
        st.write(f"**{key}**")
        st.write(value)


def _collect_plot_paths(value: Any) -> list[Path]:
    paths: list[Path] = []

    def visit(item: Any):
        if isinstance(item, pd.DataFrame):
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)
            return
        if isinstance(item, (str, os.PathLike)):
            path = Path(item)
            if path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg"} and path.exists():
                paths.append(path)

    visit(value)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def _show_plot_file(path: Path):
    st.write(f"**{path.name}**")
    if path.suffix.lower() == ".svg":
        components.html(path.read_text(encoding="utf-8"), height=560, scrolling=True)
    else:
        st.image(str(path), use_container_width=True)


def _show_plots(result: Any):
    plot_paths = _collect_plot_paths(result)
    if not plot_paths:
        return

    st.subheader("Plots")
    tabs = st.tabs([path.stem[:40] for path in plot_paths])
    for tab, path in zip(tabs, plot_paths):
        with tab:
            _show_plot_file(path)


def _show_result(result: Any):
    if isinstance(result, pd.DataFrame):
        _show_dataframe("result", result)
        return

    if not isinstance(result, dict):
        st.write(result)
        return

    preferred_tables = [
        "hs_per_depth",
        "dihs_total",
        "summary",
        "top1_candidate_summary",
        "resolvedness_summary",
    ]
    shown = set()
    for key in preferred_tables:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            _show_dataframe(key, value)
            shown.add(key)

    pairwise = result.get("pairwise_total_matrix")
    if isinstance(pairwise, pd.DataFrame):
        _show_dataframe("pairwise_total_matrix", pairwise.reset_index())
        shown.add("pairwise_total_matrix")

    _show_artifacts(result.get("artifacts", {}))
    shown.add("artifacts")

    _show_plots(result)

    remaining = {
        key: value
        for key, value in result.items()
        if key not in shown and not isinstance(value, pd.DataFrame)
    }
    if remaining:
        with st.expander("Additional result details"):
            st.write(remaining)


def _run_selected_mode(mode: str, df: pd.DataFrame, common: dict[str, Any], advanced: dict[str, Any]):
    if mode == "simple_run":
        return simple_run(
            df=df,
            model_type=advanced["model_type"],
            **common,
            return_details=True,
        )

    if mode == "triple_run":
        return triple_run(
            df=df,
            **common,
            return_details=True,
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
    )


def main():
    st.set_page_config(page_title="DIHS Correlator", layout="wide")
    st.title("DIHS Correlator")

    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to configure and run an analysis.")
        return

    try:
        df = _read_uploaded_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"{len(df):,} rows x {len(df.columns):,} columns")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    with st.form("run_form"):
        st.subheader("Analysis")
        mode = st.selectbox("Run mode", MODE_OPTIONS)

        left, right = st.columns(2)
        with left:
            class_default = (
                df.columns.get_loc("controlcode")
                if "controlcode" in df.columns
                else 0
            )
            class_column = st.selectbox(
                "Class column",
                df.columns.tolist(),
                index=int(class_default),
            )
            unknown_options = _sorted_values(df[class_column])
            unknown_sample = st.selectbox("Unknown sample/class", unknown_options)
            transform_type = st.selectbox("Transform", TRANSFORM_OPTIONS)
            max_depth = st.number_input("Max depth", min_value=1, value=100, step=1)

        with right:
            seed_enabled = st.checkbox("Set random state", value=False)
            random_state = st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1,
                help="Editable at all times. It is only passed to the analysis when Set random state is checked.",
            )
            compute_pairwise = st.checkbox("Compute pairwise matrices", value=True)
            plot_everything = st.checkbox("Create plots", value=False)
            write_files = st.checkbox("Write output files", value=False)

        exclude_columns = st.multiselect(
            "Exclude columns from numeric features",
            df.columns.tolist(),
            default=[],
        )

        output_dir = st.text_input(
            "Output directory",
            value={
                "simple_run": "./Results",
                "triple_run": "./Results_triple",
                "perturbative_triple_run_with_resolvedness": "./Results_perturbative_triple_resolvedness",
            }[mode],
            help="Used when Write output files is checked. Plot output defaults are based on this path.",
        )

        plot_output_dir_text = st.text_input(
            "Plot output directory (blank uses default)",
            value="",
            help="Used when Create plots is checked.",
        )

        save_cluster_data = st.checkbox(
            "Save cluster data",
            value=False,
            help="Only has an effect when Write output files is checked.",
        )
        save_untransformed = st.checkbox(
            "Save untransformed cluster data",
            value=False,
            help="Only has an effect when Save cluster data is checked.",
        )

        model_type = None
        perturbative_settings: dict[str, Any] = {}
        if mode == "simple_run":
            model_type = st.selectbox("Model", MODEL_OPTIONS)

        if mode == "perturbative_triple_run_with_resolvedness":
            with st.expander("Perturbative and resolvedness settings", expanded=True):
                p_left, p_right = st.columns(2)
                with p_left:
                    n_iterations = st.number_input(
                        "Perturbative iterations",
                        min_value=1,
                        value=100,
                        step=1,
                    )
                    major_error = st.number_input(
                        "Major element error",
                        min_value=0.0,
                        value=0.02,
                        step=0.01,
                        format="%.4f",
                    )
                    trace_error = st.number_input(
                        "Trace element error",
                        min_value=0.0,
                        value=0.10,
                        step=0.01,
                        format="%.4f",
                    )
                    perturbation_seed_text = st.text_input(
                        "Perturbation seed (blank for none)",
                        value="",
                    )
                    integration_depth_text = st.text_input(
                        "Integration depth (blank for automatic)",
                        value="",
                    )

                with p_right:
                    pseudo_unknown_iterations = st.number_input(
                        "Pseudo-unknown iterations",
                        min_value=1,
                        value=100,
                        step=1,
                    )
                    pseudo_unknown_sample_size_text = st.text_input(
                        "Pseudo-unknown sample size (blank to infer)",
                        value="",
                    )
                    pseudo_unknown_random_state_text = st.text_input(
                        "Pseudo-unknown random state (blank for none)",
                        value="",
                    )
                    target_precisions_text = st.text_input(
                        "Target precisions, comma separated (blank for defaults)",
                        value="",
                    )
                    min_runs_above_threshold = st.number_input(
                        "Minimum runs above threshold",
                        min_value=1,
                        value=1,
                        step=1,
                    )

                major_cols = st.multiselect(
                    "Major columns (blank uses defaults if present)",
                    numeric_columns,
                    default=[],
                )
                trace_cols = st.multiselect(
                    "Trace columns (blank uses defaults if present)",
                    numeric_columns,
                    default=[],
                )

                perturbative_settings = {
                    "n_iterations": int(n_iterations),
                    "major_cols": major_cols or None,
                    "trace_cols": trace_cols or None,
                    "major_error": float(major_error),
                    "trace_error": float(trace_error),
                    "perturbation_seed_text": perturbation_seed_text,
                    "pseudo_unknown_iterations": int(pseudo_unknown_iterations),
                    "pseudo_unknown_sample_size_text": pseudo_unknown_sample_size_text,
                    "pseudo_unknown_random_state_text": pseudo_unknown_random_state_text,
                    "target_precisions_text": target_precisions_text,
                    "min_runs_above_threshold": int(min_runs_above_threshold),
                    "integration_depth_text": integration_depth_text,
                }

        submitted = st.form_submit_button("Run Analysis", use_container_width=True)

    if not submitted:
        return

    common = {
        "transform_type": transform_type,
        "unknown_sample": unknown_sample,
        "class_column": class_column,
        "random_state": int(random_state) if seed_enabled else None,
        "compute_pairwise": compute_pairwise,
        "plot_everything": plot_everything,
        "write_files": write_files,
        "output_dir": output_dir,
        "plot_output_dir": plot_output_dir_text.strip() or None,
        "max_depth": int(max_depth),
        "exclude_columns": tuple(exclude_columns),
        "save_cluster_data": save_cluster_data,
        "save_untransformed": save_untransformed,
        "verbose": True,
    }
    try:
        if mode == "perturbative_triple_run_with_resolvedness":
            perturbative_settings = {
                **perturbative_settings,
                "perturbation_seed": _parse_optional_int(
                    perturbative_settings.pop("perturbation_seed_text")
                ),
                "pseudo_unknown_sample_size": _parse_optional_int(
                    perturbative_settings.pop("pseudo_unknown_sample_size_text")
                ),
                "pseudo_unknown_random_state": _parse_optional_int(
                    perturbative_settings.pop("pseudo_unknown_random_state_text")
                ),
                "target_precisions": _parse_float_list(
                    perturbative_settings.pop("target_precisions_text")
                ),
                "integration_depth": _parse_optional_int(
                    perturbative_settings.pop("integration_depth_text")
                ),
            }
    except ValueError as exc:
        st.error(f"Check the perturbative settings: {exc}")
        return

    advanced = {"model_type": model_type, **perturbative_settings}

    with st.spinner("Running DIHS analysis..."):
        stdout_buffer = io.StringIO()
        try:
            with redirect_stdout(stdout_buffer):
                result = _run_selected_mode(mode, df, common, advanced)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            logs = stdout_buffer.getvalue().strip()
            if logs:
                with st.expander("Run log"):
                    st.text(logs)
            return

    st.success("Analysis complete")
    logs = stdout_buffer.getvalue().strip()
    if logs:
        with st.expander("Run log"):
            st.text(logs)

    _show_result(result)

    if write_files:
        st.caption(f"Outputs written under: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
