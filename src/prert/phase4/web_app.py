"""Streamlit GUI for policy + schema compliance assessment."""

from __future__ import annotations

import html
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import streamlit as st
from pypdf import PdfReader

from prert.phase4.compliance_assessor import assess_policy_schema_compliance, resolve_default_model_path
from prert.phase4.pipeline import run_phase4_validation
from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset


MODEL_PATH_ENV = "PRERT_PHASE4_MODEL_PATH"
BASELINE_DIR_ENV = "PRERT_PHASE4_BASELINE_DIR"
COMPARISON_DIRS_ENV = "PRERT_PHASE4_COMPARISON_DIRS"
OUTPUT_DIR_ENV = "PRERT_PHASE4_OUTPUT_DIR"


def main() -> None:
    st.set_page_config(page_title="PrERT Compliance Studio", page_icon="shield", layout="wide")
    _inject_styles()

    model_path_text = ""
    generation_settings: Dict[str, Any] = {}
    benchmark_settings: Dict[str, Any] = {}
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

    with st.sidebar:
        st.header("Workspace Tabs")
        active_tab = st.radio(
            "Open",
            options=("Compliance Assessment", "Synthetic Data Generator", "Benchmark Validation"),
            index=0,
        )
        st.divider()

        if active_tab == "Compliance Assessment":
            st.subheader("Assessment Settings")
            model_path_text = st.text_input(
                "Naive Bayes model checkpoint",
                value=str(default_model) if default_model is not None else "",
                help="Optional checkpoint path for model-signal scoring. Leave blank to auto-detect.",
            )
            st.caption("Supported policy files: .txt, .md, .pdf")
            st.caption("Supported schema files: .sql, .txt, .json, .yaml, .yml")
        elif active_tab == "Synthetic Data Generator":
            st.subheader("Generation Settings")
            default_output_dir = Path.cwd() / "artifacts/phase-4/synthetic-compliance"
            generation_settings["output_dir"] = st.text_input(
                "Output directory",
                value=str(default_output_dir),
                help="Artifact directory for generated JSONL/manifest/dictionary outputs.",
            )
            generation_settings["low_count"] = int(
                st.number_input("Low compliance samples", min_value=0, value=6, step=1)
            )
            generation_settings["medium_count"] = int(
                st.number_input("Medium compliance samples", min_value=0, value=6, step=1)
            )
            generation_settings["high_count"] = int(
                st.number_input("High compliance samples", min_value=0, value=6, step=1)
            )
            generation_settings["seed"] = int(st.number_input("Random seed", value=42, step=1))
            generation_settings["include_model_signal"] = st.checkbox(
                "Include model-signal scoring",
                value=False,
                help="Uses Naive Bayes checkpoint scoring if a model path is available.",
            )
            generation_settings["model_path"] = st.text_input(
                "Model checkpoint path (optional)",
                value="",
            )
            generation_settings["export_upload_fixtures"] = st.checkbox(
                "Export upload fixture files",
                value=True,
                help="Writes per-sample policy/schema files for direct upload testing.",
            )
        else:
            st.subheader("Benchmark Settings")

            benchmark_settings["baseline_dir"] = st.text_input(
                "Baseline artifact directory",
                value=str(default_baseline_dir),
                help="Directory containing phase3_manifest.json and associated Phase 3 outputs.",
            )
            benchmark_settings["comparison_dirs"] = st.text_area(
                "Comparison artifact directories (comma or newline separated)",
                value="\n".join(str(path) for path in default_comparison_dirs),
                help="Optional folders to compare against the baseline.",
                height=120,
            )
            benchmark_settings["output_dir"] = st.text_input(
                "Benchmark output directory",
                value=str(default_output_dir),
                help="Destination for Phase 4 validation JSON/markdown reports and leaderboard JSONL.",
            )
            benchmark_settings["require_bayesian"] = st.checkbox(
                "Require Bayesian evidence checks",
                value=False,
            )
            benchmark_settings["require_polisis"] = st.checkbox(
                "Require Polisis source checks",
                value=False,
                help="When enabled, baseline validation fails if artifact source is not Polisis-based.",
            )
            benchmark_settings["polisis_advisory"] = st.checkbox(
                "Enable Polisis source advisory check",
                value=True,
            )
            benchmark_settings["ece_threshold"] = float(
                st.number_input(
                    "ECE advisory threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    step=0.01,
                    format="%.2f",
                )
            )

    if active_tab == "Compliance Assessment":
        _render_hero(
            title="Compliance Studio",
            body=(
                "Upload a company privacy policy and database schema to generate a compliance score, "
                "control coverage checks, and evidence-backed findings."
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

    _render_hero(
        title="Benchmark Validation Studio",
        body=(
            "Run Phase 4 artifact validation and benchmark comparison across multiple model outputs, "
            "then review checks, metric deltas, and leaderboard rank in one place."
        ),
    )
    _render_benchmark_screen(settings=benchmark_settings)


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


def _render_compliance_screen(model_path_text: str, auto_model_path: Optional[Path]) -> None:
    _render_model_readiness(selected_model_path=model_path_text, auto_model_path=auto_model_path)

    policy_col, schema_col = st.columns(2, gap="large")
    with policy_col:
        policy_file = st.file_uploader("Privacy Policy", type=["txt", "md", "pdf"], key="policy")
    with schema_col:
        schema_file = st.file_uploader("Database Schema", type=["sql", "txt", "json", "yaml", "yml"], key="schema")

    can_run = policy_file is not None and schema_file is not None
    run_clicked = st.button("Analyze Compliance", type="primary", disabled=not can_run)

    if not run_clicked:
        st.info("Upload both files to start compliance assessment.")
        return

    policy_text = _read_policy_upload(policy_file)
    schema_text = _read_text_upload(schema_file)

    if not policy_text.strip() or not schema_text.strip():
        st.error("Unable to parse one or both uploaded files. Please verify file format and content.")
        return

    model_path = Path(model_path_text).expanduser() if model_path_text.strip() else None
    result = assess_policy_schema_compliance(
        policy_text=policy_text,
        schema_text=schema_text,
        model_path=model_path,
    )

    _render_results(result)


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


def _read_policy_upload(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        text_chunks = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text_chunks)

    return file_bytes.decode("utf-8", errors="ignore")


def _read_text_upload(uploaded_file: Any) -> str:
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


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
