"""Streamlit GUI for policy + schema compliance assessment."""

from __future__ import annotations

import html
import importlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import streamlit as st

from prert.phase4.compliance_assessor import (
    assess_policy_compliance,
    assess_policy_schema_compliance,
    resolve_default_model_path,
)
from prert.phase4.pipeline import run_phase4_validation
from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset


MODEL_PATH_ENV = "PRERT_PHASE4_MODEL_PATH"
BASELINE_DIR_ENV = "PRERT_PHASE4_BASELINE_DIR"
COMPARISON_DIRS_ENV = "PRERT_PHASE4_COMPARISON_DIRS"
OUTPUT_DIR_ENV = "PRERT_PHASE4_OUTPUT_DIR"


def main() -> None:
    st.set_page_config(page_title="PrERT Compliance Studio", page_icon="shield", layout="wide")
    _inject_styles()

    root = Path.cwd()
    default_model = resolve_default_model_path(project_root=root)
    default_baseline_dir = _resolve_phase4_path(
        env_var=BASELINE_DIR_ENV,
        fallback_relative="deployment/demo-assets/phase-3-freeze",
        secondary_relative="artifacts/phase-3-freeze",
        root=root,
    )
    default_output_dir = _resolve_phase4_path(
        env_var=OUTPUT_DIR_ENV,
        fallback_relative="artifacts/phase-4",
        root=root,
    )
    default_comparison_dirs = _resolve_default_comparison_dirs(root=root)
    _initialize_sidebar_state(
        default_model=default_model,
        default_baseline_dir=default_baseline_dir,
        default_output_dir=default_output_dir,
        default_comparison_dirs=default_comparison_dirs,
    )

    model_path_text = ""
    generation_settings: Dict[str, Any] = {}
    benchmark_settings: Dict[str, Any] = {}

    with st.sidebar:
        st.header("Workspace Tabs")
        active_tab = st.radio(
            "Open",
            options=(
                "Compliance Assessment",
                "Synthetic Data Generator",
                "Benchmark Validation",
                "Data Explorer",
            ),
            index=0,
            key="prert_active_tab",
        )
        st.divider()

        if active_tab == "Compliance Assessment":
            model_path_text = _render_compliance_settings_form()
        elif active_tab == "Synthetic Data Generator":
            generation_settings = _render_generation_settings_form()
        elif active_tab == "Benchmark Validation":
            benchmark_settings = _render_benchmark_settings_form()
        else:
            _render_explorer_settings_form(default_root=root / "artifacts")

    if active_tab == "Compliance Assessment":
        _render_hero(
            title="Compliance Studio",
            body=(
                "Upload a company privacy policy to generate a compliance score against GDPR, NIST, "
                "and ISO 27701 with evidence-backed findings and source citations. "
                "Optionally add a database schema for schema-alignment checks."
            ),
        )
        _render_compliance_screen(model_path_text=model_path_text, auto_model_path=default_model)
        return

    if active_tab == "Synthetic Data Generator":
        _render_hero(
            title="Synthetic Data Studio",
            body=(
                "Generate synthetic privacy policy and schema pairs across low, medium, and high compliance bands "
                "for testing, demos, and evaluator regression checks."
            ),
        )
        _render_synthetic_generation_screen(settings=generation_settings)
        return

    if active_tab == "Data Explorer":
        _render_hero(
            title="Artifact Data Explorer",
            body=(
                "Browse the artifacts the model and pipelines produce — controls, chunks, metrics, events, "
                "predictions, manifests, and reports — in a clean, filterable view."
            ),
        )
        _render_explorer_screen(default_root=root / "artifacts")
        return

    _render_hero(
        title="Benchmark Validation Studio",
        body=(
            "Run Phase 4 artifact validation and benchmark comparison across multiple model outputs, "
            "then review checks, metric deltas, and leaderboard rank in one place."
        ),
    )
    _render_benchmark_screen(settings=benchmark_settings)


def _initialize_sidebar_state(
    default_model: Optional[Path],
    default_baseline_dir: Path,
    default_output_dir: Path,
    default_comparison_dirs: List[Path],
) -> None:
    if "prert_compliance_settings" not in st.session_state:
        st.session_state["prert_compliance_settings"] = {
            "model_path": str(default_model) if default_model is not None else "",
        }

    if "prert_generation_settings" not in st.session_state:
        st.session_state["prert_generation_settings"] = {
            "output_dir": str(Path.cwd() / "artifacts/phase-4/synthetic-compliance"),
            "low_count": 6,
            "medium_count": 6,
            "high_count": 6,
            "seed": 42,
            "include_model_signal": False,
            "model_path": "",
            "export_upload_fixtures": True,
        }

    if "prert_benchmark_settings" not in st.session_state:
        st.session_state["prert_benchmark_settings"] = {
            "baseline_dir": str(default_baseline_dir),
            "comparison_dirs": "\n".join(str(path) for path in default_comparison_dirs),
            "output_dir": str(default_output_dir),
            "require_bayesian": False,
            "require_polisis": False,
            "polisis_advisory": True,
            "ece_threshold": 0.20,
        }


def _render_compliance_settings_form() -> str:
    state = dict(st.session_state["prert_compliance_settings"])
    st.subheader("Assessment Settings")

    with st.form("compliance_settings_form"):
        model_path = st.text_input(
            "Naive Bayes model checkpoint",
            value=str(state.get("model_path", "")),
            help="Optional checkpoint path for model-signal scoring. Leave blank to auto-detect.",
        )
        apply_clicked = st.form_submit_button("Apply Settings")

    if apply_clicked:
        st.session_state["prert_compliance_settings"] = {
            "model_path": model_path,
        }
        st.success("Assessment settings applied")

    st.caption("Supported policy files: .txt, .md, .pdf")
    st.caption("Supported schema files (optional): .sql, .txt, .json, .yaml, .yml")
    return str(st.session_state["prert_compliance_settings"]["model_path"])


def _render_generation_settings_form() -> Dict[str, Any]:
    state = dict(st.session_state["prert_generation_settings"])
    st.subheader("Generation Settings")

    with st.form("generation_settings_form"):
        output_dir = st.text_input(
            "Output directory",
            value=str(state.get("output_dir", "")),
            help="Artifact directory for generated JSONL/manifest/dictionary outputs.",
        )
        low_count = int(st.number_input("Low compliance samples", min_value=0, value=int(state.get("low_count", 6)), step=1))
        medium_count = int(
            st.number_input("Medium compliance samples", min_value=0, value=int(state.get("medium_count", 6)), step=1)
        )
        high_count = int(st.number_input("High compliance samples", min_value=0, value=int(state.get("high_count", 6)), step=1))
        seed = int(st.number_input("Random seed", value=int(state.get("seed", 42)), step=1))
        include_model_signal = st.checkbox(
            "Include model-signal scoring",
            value=bool(state.get("include_model_signal", False)),
            help="Uses Naive Bayes checkpoint scoring if a model path is available.",
        )
        model_path = st.text_input(
            "Model checkpoint path (optional)",
            value=str(state.get("model_path", "")),
        )
        export_upload_fixtures = st.checkbox(
            "Export upload fixture files",
            value=bool(state.get("export_upload_fixtures", True)),
            help="Writes per-sample policy/schema files for direct upload testing.",
        )
        apply_clicked = st.form_submit_button("Apply Settings")

    if apply_clicked:
        st.session_state["prert_generation_settings"] = {
            "output_dir": output_dir,
            "low_count": low_count,
            "medium_count": medium_count,
            "high_count": high_count,
            "seed": seed,
            "include_model_signal": include_model_signal,
            "model_path": model_path,
            "export_upload_fixtures": export_upload_fixtures,
        }
        st.success("Generation settings applied")

    return dict(st.session_state["prert_generation_settings"])


def _render_benchmark_settings_form() -> Dict[str, Any]:
    state = dict(st.session_state["prert_benchmark_settings"])
    st.subheader("Benchmark Settings")

    with st.form("benchmark_settings_form"):
        baseline_dir = st.text_input(
            "Baseline artifact directory",
            value=str(state.get("baseline_dir", "")),
            help="Directory containing phase3_manifest.json and associated Phase 3 outputs.",
        )
        comparison_dirs = st.text_area(
            "Comparison artifact directories (comma or newline separated)",
            value=str(state.get("comparison_dirs", "")),
            help="Optional folders to compare against the baseline.",
            height=120,
        )
        output_dir = st.text_input(
            "Benchmark output directory",
            value=str(state.get("output_dir", "")),
            help="Destination for Phase 4 validation JSON/markdown reports and leaderboard JSONL.",
        )
        require_bayesian = st.checkbox(
            "Require Bayesian evidence checks",
            value=bool(state.get("require_bayesian", False)),
        )
        require_polisis = st.checkbox(
            "Require Polisis source checks",
            value=bool(state.get("require_polisis", False)),
            help="When enabled, baseline validation fails if artifact source is not Polisis-based.",
        )
        polisis_advisory = st.checkbox(
            "Enable Polisis source advisory check",
            value=bool(state.get("polisis_advisory", True)),
        )
        ece_threshold = float(
            st.number_input(
                "ECE advisory threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(state.get("ece_threshold", 0.20)),
                step=0.01,
                format="%.2f",
            )
        )
        apply_clicked = st.form_submit_button("Apply Settings")

    if apply_clicked:
        st.session_state["prert_benchmark_settings"] = {
            "baseline_dir": baseline_dir,
            "comparison_dirs": comparison_dirs,
            "output_dir": output_dir,
            "require_bayesian": require_bayesian,
            "require_polisis": require_polisis,
            "polisis_advisory": polisis_advisory,
            "ece_threshold": ece_threshold,
        }
        st.success("Benchmark settings applied")

    return dict(st.session_state["prert_benchmark_settings"])


def _render_explorer_settings_form(default_root: Path) -> Dict[str, Any]:
    if "prert_explorer_settings" not in st.session_state:
        st.session_state["prert_explorer_settings"] = {
            "root_dir": str(default_root),
            "max_rows": 500,
            "search_query": "",
        }

    state = dict(st.session_state["prert_explorer_settings"])
    st.subheader("Explorer Settings")

    with st.form("explorer_settings_form"):
        root_dir = st.text_input(
            "Artifacts root directory",
            value=str(state.get("root_dir", str(default_root))),
            help="Root folder containing phase-1, phase-2, phase-3, phase-4 artifacts.",
        )
        max_rows = int(
            st.number_input(
                "Max rows to display (JSONL)",
                min_value=10,
                max_value=10000,
                value=int(state.get("max_rows", 500)),
                step=50,
            )
        )
        search_query = st.text_input(
            "Filter rows (substring match)",
            value=str(state.get("search_query", "")),
            help="Case-insensitive substring filter applied to JSONL rows.",
        )
        apply_clicked = st.form_submit_button("Apply Settings")

    if apply_clicked:
        st.session_state["prert_explorer_settings"] = {
            "root_dir": root_dir,
            "max_rows": max_rows,
            "search_query": search_query,
        }
        st.success("Explorer settings applied")

    return dict(st.session_state["prert_explorer_settings"])


def _render_hero(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">PrERT Phase 4</div>
            <h1>{title}</h1>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


EXPLORER_TEXT_SUFFIXES = {".md", ".txt", ".sql", ".yaml", ".yml", ".csv", ".log"}
EXPLORER_CODE_SUFFIXES = {".sql", ".yaml", ".yml", ".py", ".toml"}


def _render_explorer_screen(default_root: Path) -> None:
    settings = dict(st.session_state.get("prert_explorer_settings", {}))
    root_text = str(settings.get("root_dir", str(default_root))).strip() or str(default_root)
    root = Path(root_text).expanduser()

    if not root.exists() or not root.is_dir():
        st.error(f"Artifacts directory not found: {root}")
        st.caption("Update the root directory in the sidebar and apply settings.")
        return

    subdirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")],
        key=lambda item: item.name,
    )
    files_at_root = sorted(
        [path for path in root.iterdir() if path.is_file() and not path.name.startswith(".")],
        key=lambda item: item.name,
    )

    if not subdirs and not files_at_root:
        st.warning("No artifact files or subfolders found at the configured root.")
        return

    folder_options: List[str] = []
    if files_at_root:
        folder_options.append(".")
    folder_options.extend(path.name for path in subdirs)

    folder_choice = st.selectbox(
        "Artifact Folder",
        options=folder_options,
        index=0,
        key="prert_explorer_folder",
    )
    target_dir = root if folder_choice == "." else (root / folder_choice)

    files = _list_explorer_files(target_dir, include_subdirs=folder_choice != ".")
    if not files:
        st.info(f"No supported files found in {target_dir}.")
        return

    suffix_options = sorted({path.suffix.lower() or "(no ext)" for path in files})
    selected_suffixes = st.multiselect(
        "Filter by file type",
        options=suffix_options,
        default=suffix_options,
        key=f"prert_explorer_suffixes_{folder_choice}",
    )
    filtered_files = [
        path
        for path in files
        if (path.suffix.lower() or "(no ext)") in selected_suffixes
    ]
    if not filtered_files:
        st.info("No files match the selected file types.")
        return

    file_labels = [_format_explorer_file_label(path, target_dir) for path in filtered_files]
    file_choice_label = st.selectbox(
        "File",
        options=file_labels,
        index=0,
        key=f"prert_explorer_file_{folder_choice}",
    )
    selected_index = file_labels.index(file_choice_label)
    selected_path = filtered_files[selected_index]

    st.markdown("---")
    _render_explorer_file_header(selected_path, target_dir)
    _render_explorer_file_content(
        selected_path,
        max_rows=int(settings.get("max_rows", 500)),
        search_query=str(settings.get("search_query", "")),
    )


def _list_explorer_files(target_dir: Path, include_subdirs: bool = True) -> List[Path]:
    files: List[Path] = []
    iterator = target_dir.rglob("*") if include_subdirs else target_dir.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(target_dir).parts
        except ValueError:
            relative_parts = (path.name,)
        if any(part.startswith(".") for part in relative_parts):
            continue
        files.append(path)
    files.sort(key=lambda item: str(item).lower())
    return files


def _format_explorer_file_label(path: Path, base: Path) -> str:
    try:
        relative = path.relative_to(base)
    except ValueError:
        relative = path
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    return f"{relative.as_posix()}  -  {_format_byte_size(size_bytes)}"


def _format_byte_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _render_explorer_file_header(path: Path, base: Path) -> None:
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0

    try:
        relative = path.relative_to(base).as_posix()
    except ValueError:
        relative = path.name

    c1, c2, c3 = st.columns(3)
    c1.metric("File", path.name)
    c2.metric("Size", _format_byte_size(size_bytes))
    c3.metric("Type", path.suffix.lower() or "n/a")
    st.caption(f"Path: {path}")
    st.caption(f"Relative: {relative}")


def _render_explorer_file_content(path: Path, max_rows: int, search_query: str) -> None:
    suffix = path.suffix.lower()

    try:
        if suffix == ".jsonl":
            _render_jsonl_file(path, max_rows=max_rows, search_query=search_query)
        elif suffix == ".json":
            _render_json_file(path)
        elif suffix == ".md":
            _render_markdown_file(path)
        elif suffix in EXPLORER_TEXT_SUFFIXES:
            _render_text_file(path)
        else:
            _render_binary_file(path)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Failed to render file: {exc}")
        _render_binary_file(path)


def _render_jsonl_file(path: Path, max_rows: int, search_query: str) -> None:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    raw_lines = [line for line in raw_text.splitlines() if line.strip()]
    total_lines = len(raw_lines)

    query = search_query.strip().lower()
    if query:
        filtered_lines = [line for line in raw_lines if query in line.lower()]
    else:
        filtered_lines = raw_lines

    matched = len(filtered_lines)
    truncated_lines = filtered_lines[:max_rows]

    rows: List[Dict[str, Any]] = []
    parse_errors = 0
    for line in truncated_lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            rows.append({"value": value})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", str(total_lines))
    c2.metric("Matched", str(matched))
    c3.metric("Shown", str(len(rows)))
    c4.metric("Parse Errors", str(parse_errors))

    if not rows:
        st.info("No rows to display. Adjust the filter or row limit in the sidebar.")
        return

    tab_table, tab_record, tab_raw = st.tabs(["Table", "Record View", "Raw JSONL"])

    with tab_table:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab_record:
        record_index = st.number_input(
            "Row index",
            min_value=0,
            max_value=max(0, len(rows) - 1),
            value=0,
            step=1,
            key=f"prert_explorer_record_{path.name}",
        )
        st.json(rows[int(record_index)])

    with tab_raw:
        preview = "\n".join(truncated_lines)
        st.code(preview, language="json")

    st.download_button(
        "Download File",
        data=raw_text,
        file_name=path.name,
        mime="application/jsonl",
    )


def _render_json_file(path: Path) -> None:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.code(raw_text, language="json")
        return

    tab_pretty, tab_raw = st.tabs(["Structured", "Raw"])
    with tab_pretty:
        st.json(value)
    with tab_raw:
        st.code(json.dumps(value, indent=2, ensure_ascii=False), language="json")

    st.download_button(
        "Download File",
        data=raw_text,
        file_name=path.name,
        mime="application/json",
    )


def _render_markdown_file(path: Path) -> None:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    tab_render, tab_raw = st.tabs(["Rendered", "Source"])
    with tab_render:
        st.markdown(raw_text)
    with tab_raw:
        st.code(raw_text, language="markdown")

    st.download_button(
        "Download File",
        data=raw_text,
        file_name=path.name,
        mime="text/markdown",
    )


def _render_text_file(path: Path) -> None:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix in EXPLORER_CODE_SUFFIXES:
        language = suffix.lstrip(".")
        if language == "yml":
            language = "yaml"
        st.code(raw_text, language=language)
    elif suffix == ".csv":
        st.code(raw_text, language="csv")
    else:
        st.code(raw_text, language="text")

    st.download_button(
        "Download File",
        data=raw_text,
        file_name=path.name,
        mime="text/plain",
    )


def _render_binary_file(path: Path) -> None:
    try:
        file_bytes = path.read_bytes()
    except OSError as exc:
        st.error(f"Unable to read file: {exc}")
        return

    st.info("Preview is not available for this file type. Use the download button below.")
    st.download_button(
        "Download File",
        data=file_bytes,
        file_name=path.name,
        mime="application/octet-stream",
    )


def _render_compliance_screen(model_path_text: str, auto_model_path: Optional[Path]) -> None:
    _render_model_readiness(selected_model_path=model_path_text, auto_model_path=auto_model_path)

    policy_col, schema_col = st.columns(2, gap="large")
    with policy_col:
        policy_file = st.file_uploader("Privacy Policy", type=["txt", "md", "pdf"], key="policy")
    with schema_col:
        schema_file = st.file_uploader(
            "Database Schema (Optional)",
            type=["sql", "txt", "json", "yaml", "yml"],
            key="schema",
            help="Add a database schema for additional schema-alignment checks. Leave empty for policy-only analysis.",
        )

    can_run = policy_file is not None
    run_clicked = st.button("Analyze Compliance", type="primary", disabled=not can_run)

    if not run_clicked:
        if policy_file is None:
            st.info("Upload a privacy policy to start compliance assessment. Schema is optional.")
        return

    policy_result = _read_policy_upload(policy_file)

    if policy_result["error"]:
        st.error(f"Policy file error: {policy_result['error']}")
        st.caption(f"File: {policy_result['file_name']} ({policy_result['size_bytes']} bytes)")
        return

    policy_text = str(policy_result["text"])
    if not policy_text.strip():
        st.error("Policy file was parsed as empty. Verify file encoding/content and retry.")
        return

    model_path = Path(model_path_text).expanduser() if model_path_text.strip() else None

    if schema_file is not None:
        schema_result = _read_text_upload(schema_file)
        if schema_result["error"]:
            st.error(f"Schema file error: {schema_result['error']}")
            st.caption(f"File: {schema_result['file_name']} ({schema_result['size_bytes']} bytes)")
            return

        schema_text = str(schema_result["text"])
        if not schema_text.strip():
            st.error("Schema file was parsed as empty. Verify file encoding/content and retry.")
            return

        result = assess_policy_schema_compliance(
            policy_text=policy_text,
            schema_text=schema_text,
            model_path=model_path,
        )
        _render_results(result)
    else:
        result = assess_policy_compliance(
            policy_text=policy_text,
            model_path=model_path,
        )
        _render_policy_compliance_results(result)


def _render_synthetic_generation_screen(settings: Dict[str, Any]) -> None:
    st.subheader("Generate New Synthetic Dataset")
    st.write(
        "Create synthetic policy/schema records with score-banded compliance labels. "
        "Outputs include JSONL dataset, manifest, dictionary, and optional upload fixtures."
    )

    run_clicked = st.button("Generate Synthetic Data", type="primary")
    if not run_clicked:
        st.info("Adjust generation settings in the sidebar and click Generate Synthetic Data.")
        return

    output_dir = Path(str(settings.get("output_dir", "")).strip() or "artifacts/phase-4/synthetic-compliance").expanduser()
    include_model_signal = bool(settings.get("include_model_signal", False))
    model_path_text = str(settings.get("model_path", "")).strip()
    model_path = Path(model_path_text).expanduser() if model_path_text else None
    total_samples = sum(int(settings.get(f"{band}_count", 0)) for band in ("low", "medium", "high"))

    progress_bar = st.progress(0.0 if total_samples else 1.0)
    status_text = st.empty()
    status_text.caption("Preparing synthetic generation...")

    def _progress_callback(event: Dict[str, Any]) -> None:
        event_name = str(event.get("event", ""))
        if event_name == "band_start":
            band = str(event.get("band", "unknown"))
            status_text.caption(f"Generating {band} compliance samples...")
            return

        if event_name == "sample_complete":
            index = int(event.get("index", 0))
            total = max(1, int(event.get("total", total_samples or 1)))
            score = float(event.get("overall_score", 0.0))
            progress_bar.progress(min(1.0, index / total))
            status_text.caption(f"Generated {index}/{total} samples (latest score: {score:.1f})")
            return

        if event_name == "complete":
            progress_bar.progress(1.0)
            status_text.caption("Synthetic generation complete.")

    with st.spinner("Generating synthetic policy/schema dataset..."):
        try:
            manifest = generate_synthetic_policy_schema_dataset(
                output_dir=output_dir,
                counts_by_band={
                    "low": int(settings.get("low_count", 6)),
                    "medium": int(settings.get("medium_count", 6)),
                    "high": int(settings.get("high_count", 6)),
                },
                seed=int(settings.get("seed", 42)),
                include_model_signal=include_model_signal,
                model_path=model_path,
                export_upload_fixtures=bool(settings.get("export_upload_fixtures", True)),
                progress_callback=_progress_callback,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard for UI usage
            st.error(f"Synthetic generation failed: {exc}")
            return

    st.success("Synthetic dataset generated successfully.")

    output_files = manifest.get("output_files", {})
    st.markdown("**Generated Files**")
    st.write(f"Dataset: {output_files.get('dataset', '')}")
    st.write(f"Manifest: {output_files.get('manifest', '')}")
    st.write(f"Dictionary: {output_files.get('dictionary', '')}")

    summary_rows = []
    score_summary = manifest.get("score_summary", {})
    for band in ("low", "medium", "high"):
        band_summary = score_summary.get(band, {})
        summary_rows.append(
            {
                "Band": band,
                "Count": band_summary.get("count"),
                "In Target": band_summary.get("in_target_band"),
                "Min": band_summary.get("minimum"),
                "Mean": band_summary.get("mean"),
                "Max": band_summary.get("maximum"),
            }
        )
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
    st.download_button(
        "Download Synthetic Manifest (JSON)",
        data=manifest_json,
        file_name="synthetic_policy_schema_manifest.json",
        mime="application/json",
    )


def _render_benchmark_screen(settings: Dict[str, Any]) -> None:
    st.subheader("Run Phase 4 Artifact Benchmark")
    st.write(
        "Validate a baseline artifact folder, compare one or more candidate runs, "
        "and export benchmark-ready validation reports."
    )

    run_clicked = st.button("Run Benchmark Validation", type="primary")
    if not run_clicked:
        st.info("Adjust benchmark settings in the sidebar and click Run Benchmark Validation.")
        return

    baseline_dir = Path(str(settings.get("baseline_dir", "")).strip() or "artifacts/phase-3-freeze").expanduser()
    output_dir = Path(str(settings.get("output_dir", "")).strip() or "artifacts/phase-4").expanduser()
    comparison_dirs = _parse_path_entries(str(settings.get("comparison_dirs", "")))

    _render_benchmark_readiness(
        baseline_dir=baseline_dir,
        comparison_dirs=comparison_dirs,
        output_dir=output_dir,
    )

    if not baseline_dir.exists():
        st.error(f"Baseline directory does not exist: {baseline_dir}")
        return

    total_steps = len(comparison_dirs) + 2
    progress_state = {"completed": 0}
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.caption("Preparing benchmark validation...")

    def _status_callback(event: Dict[str, Any]) -> None:
        name = str(event.get("event", ""))

        if name == "baseline_start":
            status_text.caption("Validating baseline artifacts...")
            return

        if name == "baseline_complete":
            progress_state["completed"] = 1
            progress_bar.progress(min(1.0, progress_state["completed"] / max(1, total_steps)))
            status_text.caption("Baseline validation complete.")
            return

        if name == "comparison_start":
            index = int(event.get("index", 1))
            total = int(event.get("total", len(comparison_dirs)))
            status_text.caption(f"Evaluating comparison run {index}/{max(1, total)}...")
            return

        if name == "comparison_complete":
            progress_state["completed"] += 1
            progress_bar.progress(min(1.0, progress_state["completed"] / max(1, total_steps)))
            return

        if name == "summary_start":
            status_text.caption("Building benchmark summary and writing reports...")
            return

        if name == "complete":
            progress_state["completed"] = total_steps
            progress_bar.progress(1.0)
            status_text.caption("Benchmark validation complete.")

    with st.spinner("Running Phase 4 validation and benchmark comparison..."):
        try:
            report = run_phase4_validation(
                output_dir=output_dir,
                baseline_dir=baseline_dir,
                comparison_dirs=comparison_dirs,
                require_bayesian=bool(settings.get("require_bayesian", False)),
                require_polisis=bool(settings.get("require_polisis", False)),
                polisis_advisory=bool(settings.get("polisis_advisory", True)),
                ece_threshold=float(settings.get("ece_threshold", 0.20)),
                status_callback=_status_callback,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard for UI usage
            st.error(f"Benchmark validation failed: {exc}")
            return

    baseline = report.get("baseline", {})
    validation = baseline.get("validation", {})
    summary = baseline.get("summary", {})
    metrics = summary.get("metrics", {})
    output_files = report.get("output_files", {})

    passed = bool(validation.get("passed", False))
    if passed:
        st.success("Baseline validation passed.")
    else:
        st.warning("Baseline validation failed required checks. Review details below.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Passed", "Yes" if passed else "No")
    c2.metric("Test Macro F1", _format_optional_float(metrics.get("test_macro_f1")))
    c3.metric("Test Accuracy", _format_optional_float(metrics.get("test_accuracy")))
    c4.metric("Calibration ECE", _format_optional_float(metrics.get("calibration_test_ece")))

    checks = validation.get("checks", [])
    if isinstance(checks, list) and checks:
        st.subheader("Validation Checks")
        check_rows = [
            {
                "Check": str(item.get("name", "")),
                "Required": "Required" if bool(item.get("required", True)) else "Advisory",
                "Passed": "Yes" if bool(item.get("passed", False)) else "No",
                "Details": json.dumps(item.get("details", {}), ensure_ascii=False),
            }
            for item in checks
            if isinstance(item, dict)
        ]

        required_failed = sum(1 for row in check_rows if row["Required"] == "Required" and row["Passed"] == "No")
        advisory_failed = sum(1 for row in check_rows if row["Required"] == "Advisory" and row["Passed"] == "No")
        checks_total = len(check_rows)

        s1, s2, s3 = st.columns(3)
        s1.metric("Required Failures", str(required_failed))
        s2.metric("Advisory Failures", str(advisory_failed))
        s3.metric("Total Checks", str(checks_total))

        sorted_rows = sorted(
            check_rows,
            key=lambda row: (
                row["Passed"] == "Yes",
                row["Required"] != "Required",
                row["Check"],
            ),
        )

        for row in sorted_rows:
            is_passed = row["Passed"] == "Yes"
            card_bg = "#163526" if is_passed else "#4A1D1D"
            card_border = "#2A8A57" if is_passed else "#B83B3B"
            status_badge = _render_check_badge(
                text="PASS" if is_passed else "FAIL",
                background="#2A8A57" if is_passed else "#B83B3B",
                text_color="#FFFFFF",
            )
            level_badge = _render_check_badge(
                text=row["Required"].upper(),
                background="#725A1C" if row["Required"] == "Required" else "#3A4F63",
                text_color="#FFFFFF",
            )
            check_title = html.escape(str(row["Check"]))

            st.markdown(
                f"""
                <div style="padding: 0.6rem 0.75rem; border-radius: 10px; margin-bottom: 0.5rem; border: 1px solid {card_border}; background: {card_bg};">
                    <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">{status_badge}{level_badge}<span style="font-weight: 600;">{check_title}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(row["Details"])

    comparison_summary = report.get("comparison_summary", {})
    leaderboard = comparison_summary.get("leaderboard", [])
    if isinstance(leaderboard, list) and leaderboard:
        st.subheader("Leaderboard")
        leaderboard_rows = [
            {
                "Rank": row.get("rank"),
                "Run": row.get("name"),
                "Baseline": "Yes" if row.get("is_baseline") else "No",
                "Validation Passed": "Yes" if row.get("validation_passed") else "No",
                "Test Macro F1": row.get("test_macro_f1"),
                "Test Accuracy": row.get("test_accuracy"),
                "Bayesian Score": row.get("bayesian_primary_score"),
                "Calibration ECE": row.get("calibration_test_ece"),
            }
            for row in leaderboard
            if isinstance(row, dict)
        ]
        st.dataframe(leaderboard_rows, use_container_width=True, hide_index=True)

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    st.download_button(
        "Download Benchmark Report (JSON)",
        data=report_json,
        file_name="phase4_validation_report.json",
        mime="application/json",
    )

    markdown_path = Path(str(output_files.get("markdown", "")).strip())
    if markdown_path.exists():
        st.download_button(
            "Download Benchmark Report (Markdown)",
            data=markdown_path.read_text(encoding="utf-8"),
            file_name=markdown_path.name,
            mime="text/markdown",
        )
    else:
        st.caption("Markdown report was not generated for this run. JSON report is always available above.")


def _parse_path_entries(raw_value: str) -> List[Path]:
    entries = [token.strip() for token in re.split(r"[\n,]", raw_value) if token.strip()]
    paths: List[Path] = []
    seen: set[str] = set()
    for entry in entries:
        resolved = str(Path(entry).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(Path(resolved))
    return paths


def _resolve_phase4_path(
    env_var: str,
    fallback_relative: str,
    root: Path,
    secondary_relative: Optional[str] = None,
) -> Path:
    env_value = os.getenv(env_var, "").strip()
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate

    fallback = root / fallback_relative
    if fallback.exists():
        return fallback

    if secondary_relative is not None:
        secondary = root / secondary_relative
        if secondary.exists():
            return secondary
        return secondary

    return fallback


def _resolve_default_comparison_dirs(root: Path) -> List[Path]:
    env_value = os.getenv(COMPARISON_DIRS_ENV, "").strip()
    if env_value:
        return _parse_path_entries(env_value)

    candidates = [
        root / "deployment/demo-assets/phase-3-nb",
        root / "deployment/demo-assets/phase-3-logreg",
        root / "deployment/demo-assets/phase-3-privacybert",
        root / "artifacts/phase-3-nb",
        root / "artifacts/phase-3-logreg",
        root / "artifacts/phase-3-privacybert",
    ]
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def _render_model_readiness(selected_model_path: str, auto_model_path: Optional[Path]) -> None:
    candidate_text = selected_model_path.strip()
    if candidate_text:
        candidate = Path(candidate_text).expanduser()
    else:
        candidate = auto_model_path

    if candidate is not None and candidate.exists():
        st.success(f"Model checkpoint ready: {candidate}")
    else:
        st.warning(
            "Model checkpoint not found. Compliance analysis still works, but model-signal scoring will be advisory-only. "
            f"Set {MODEL_PATH_ENV} to point to a valid model.json path for deployment."
        )


def _render_benchmark_readiness(baseline_dir: Path, comparison_dirs: List[Path], output_dir: Path) -> None:
    baseline_manifest = baseline_dir / "phase3_manifest.json"
    baseline_ready = baseline_dir.exists() and baseline_manifest.exists()
    existing_comparisons = [path for path in comparison_dirs if path.exists()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Ready", "Yes" if baseline_ready else "No")
    c2.metric("Comparison Dirs Found", str(len(existing_comparisons)))
    c3.metric("Output Dir", str(output_dir))

    if baseline_ready:
        st.success(f"Baseline manifest detected: {baseline_manifest}")
    else:
        st.warning(
            "Baseline artifact is not ready. Ensure phase3_manifest.json is present in the selected baseline directory."
        )


def _format_optional_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _render_check_badge(text: str, background: str, text_color: str) -> str:
    safe_text = html.escape(text)
    return (
        "<span style=\"display:inline-block;padding:0.2rem 0.5rem;border-radius:999px;"
        f"font-size:0.72rem;font-weight:700;letter-spacing:0.02em;background:{background};color:{text_color};\">"
        f"{safe_text}</span>"
    )


def _render_results(result: Dict[str, Any]) -> None:
    st.divider()
    score = float(result["overall_score"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compliance Score", f"{score:.1f} / 100")
    c2.metric("Grade", str(result["grade"]))
    c3.metric("Status", str(result["status"]))
    c4.metric("Clauses Analyzed", int(result["summary"]["clauses_analyzed"]))

    st.progress(min(1.0, score / 100.0))

    failed_policy_checks = [item for item in result["policy_checks"] if not item.get("passed", False)]
    schema_details = list(result["schema_analysis"].get("details", []))

    if failed_policy_checks or schema_details:
        st.subheader("Priority Findings")
        for item in failed_policy_checks[:5]:
            st.error(f"{item['title']}: score {item['score']:.2f}/{item['max_score']:.2f}")
        for detail in schema_details[:5]:
            st.warning(detail)
    else:
        st.subheader("Priority Findings")
        st.success("No critical gaps detected in policy coverage or schema alignment.")

    st.subheader("Control Coverage")
    rows = [
        {
            "Check": item["title"],
            "Score": f"{item['score']:.2f}/{item['max_score']:.2f}",
            "Passed": "Yes" if item["passed"] else "No",
            "Keywords": ", ".join(item["matched_keywords"][:5]),
        }
        for item in result["policy_checks"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Schema Alignment")
    schema = result["schema_analysis"]
    st.write(f"Score: {schema['score']:.2f}/{schema['max_score']:.2f}")
    if schema["details"]:
        for detail in schema["details"]:
            st.warning(detail)
    else:
        st.success("Schema fields and policy disclosures are reasonably aligned.")

    st.subheader("Detected Data Fields")
    d1, d2 = st.columns(2)
    d1.write("PII Fields")
    d1.code("\n".join(result["detected_fields"]["pii_fields"]) or "None")
    d2.write("Sensitive Fields")
    d2.code("\n".join(result["detected_fields"]["sensitive_fields"]) or "None")

    st.subheader("Model Signal")
    model_signal = result["model_signal"]
    st.write(f"Score: {model_signal['score']:.2f}/{model_signal['max_score']:.2f}")
    for detail in model_signal["details"]:
        st.caption(detail)

    for item in result["policy_checks"]:
        if not item["evidence"]:
            continue
        with st.expander(f"Evidence: {item['title']}"):
            for evidence in item["evidence"]:
                st.markdown(f"- {evidence}")

    report_json = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        "Download Compliance Report (JSON)",
        data=report_json,
        file_name="compliance_report.json",
        mime="application/json",
    )


def _render_policy_compliance_results(result: Dict[str, Any]) -> None:
    """Render policy-only compliance results with per-regulation verdicts and citations."""
    st.divider()
    score = float(result["overall_score"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compliance Score", f"{score:.1f} / 100")
    c2.metric("Grade", str(result["grade"]))
    c3.metric("Status", str(result["status"]))
    c4.metric("Clauses Analyzed", int(result["summary"]["clauses_analyzed"]))

    st.progress(min(1.0, score / 100.0))

    bayesian_risk = result.get("bayesian_risk")
    if bayesian_risk:
        st.subheader("Bayesian Risk")
        primary_score = bayesian_risk.get("overall", {}).get("primary_score", 0.0)
        st.metric("Global Bayesian Score", f"{primary_score:.4f}")
        st.caption("Lower is better. Risk computed across all extracted clauses.")

    regulation_summary = result.get("regulation_summary", {})
    if regulation_summary:
        st.subheader("Regulation Summary")
        reg_columns = st.columns(len(regulation_summary))
        for col, (reg_name, reg_data) in zip(reg_columns, sorted(regulation_summary.items())):
            display_name = reg_name.replace("_", " ")
            pct = float(reg_data.get("compliance_pct", 0))
            col.metric(display_name, f"{pct:.0f}%")
            col.caption(
                f"Pass: {reg_data.get('pass_count', 0)} / "
                f"Fail: {reg_data.get('fail_count', 0)} / "
                f"Controls: {reg_data.get('total_controls', 0)}"
            )

    claims = result.get("claims", [])
    if claims:
        st.subheader("Per-Claim Regulation Verdicts")
        st.caption(
            f"{len(claims)} claim(s) extracted from the policy. "
            "Expand each claim to see how it was evaluated against every regulation."
        )

        for claim in claims:
            confidence = claim.get("confidence", 0.0)
            predicted_label = claim.get("predicted_label", "unknown")
            claim_label = f"{claim['check_title']} — Clause #{claim['claim_index'] + 1} (Label: {predicted_label}, Conf: {confidence:.2f})"
            with st.expander(claim_label):
                st.markdown("**Source Citation (Policy Text):**")
                st.info(claim["claim_text"])

                verdicts = claim.get("regulation_verdicts", [])
                if verdicts:
                    verdict_rows = []
                    for verdict in verdicts:
                        status_icon = "✅" if verdict["compliant"] else "❌"
                        verdict_rows.append(
                            {
                                "Status": status_icon,
                                "Regulation": verdict["regulation"].replace("_", " "),
                                "Control": verdict["control_id"],
                                "Title": verdict["control_title"],
                                "Reason": verdict["reason"],
                                "Remediation": verdict.get("remediation_advice", ""),
                            }
                        )
                    st.dataframe(verdict_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No regulation verdicts for this claim.")

    model_signal = result.get("model_signal", {})
    if model_signal:
        st.subheader("Model Signal")
        st.write(f"Score: {model_signal.get('score', 0):.2f}/{model_signal.get('max_score', 5):.2f}")
        for detail in model_signal.get("details", []):
            st.caption(detail)

    report_json = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        "Download Compliance Report (JSON)",
        data=report_json,
        file_name="policy_compliance_report.json",
        mime="application/json",
    )


def _read_policy_upload(uploaded_file: Any) -> Dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    size_bytes = len(file_bytes)

    if suffix == ".pdf":
        try:
            pypdf_module = importlib.import_module("pypdf")
            reader = pypdf_module.PdfReader(BytesIO(file_bytes))
            text_chunks = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(text_chunks).strip()
        except Exception as exc:  # pragma: no cover - runtime upload guard
            return {
                "text": "",
                "error": f"PDF parsing failed: {exc}",
                "file_name": uploaded_file.name,
                "size_bytes": size_bytes,
            }

        if not text:
            return {
                "text": "",
                "error": "PDF has no extractable text. If this is scanned content, convert to text/markdown and retry.",
                "file_name": uploaded_file.name,
                "size_bytes": size_bytes,
            }

        return {
            "text": text,
            "error": None,
            "file_name": uploaded_file.name,
            "size_bytes": size_bytes,
        }

    decoded = _decode_text_bytes(file_bytes)
    if decoded is None:
        return {
            "text": "",
            "error": "Unable to decode policy file using utf-8, utf-16, or latin-1.",
            "file_name": uploaded_file.name,
            "size_bytes": size_bytes,
        }

    return {
        "text": decoded,
        "error": None,
        "file_name": uploaded_file.name,
        "size_bytes": size_bytes,
    }


def _read_text_upload(uploaded_file: Any) -> Dict[str, Any]:
    file_bytes = uploaded_file.getvalue()
    decoded = _decode_text_bytes(file_bytes)
    if decoded is None:
        return {
            "text": "",
            "error": "Unable to decode schema file using utf-8, utf-16, or latin-1.",
            "file_name": uploaded_file.name,
            "size_bytes": len(file_bytes),
        }

    return {
        "text": decoded,
        "error": None,
        "file_name": uploaded_file.name,
        "size_bytes": len(file_bytes),
    }


def _decode_text_bytes(data: bytes) -> Optional[str]:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            decoded = data.decode(encoding)
        except UnicodeDecodeError:
            continue

        if decoded.strip():
            return decoded

    return None


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@400;600&display=swap');
            :root {
                --bg-a: #38291E;
                --bg-b: #433820;
                --bg-c: #644F2C;
                --ink: #FBFAF8;
                --accent: #B9793D;
                --accent-soft: #85560BCC;
            }
            .stApp {
                background: var(--bg-a);
                color: var(--ink);
            }
            h1, h2, h3 {
                font-family: 'Space Grotesk', sans-serif;
                color: var(--ink);
                letter-spacing: 0.2px;
            }
            p, li, .stMarkdown, .stCaption {
                font-family: 'Source Serif 4', serif;
            }
            .hero {
                padding: 1.25rem 1.5rem;
                border-radius: 14px;
                background: var(--bg-b);
                border: 1px solid var(--accent-soft);
                box-shadow: 0 8px 28px rgba(16, 35, 49, 0.06);
                margin-bottom: 1rem;
                animation: fadeIn 0.45s ease-out;
            }
            .hero-kicker {
                display: inline-block;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: var(--accent);
                margin-bottom: 0.35rem;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
