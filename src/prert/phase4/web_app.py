"""Streamlit GUI for policy + schema compliance assessment."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from pypdf import PdfReader

from prert.phase4.compliance_assessor import assess_policy_schema_compliance, resolve_default_model_path
from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset


def main() -> None:
    st.set_page_config(page_title="PrERT Compliance Studio", page_icon="shield", layout="wide")
    _inject_styles()

    model_path_text = ""
    generation_settings: Dict[str, Any] = {}
    with st.sidebar:
        st.header("Workspace Tabs")
        active_tab = st.radio(
            "Open",
            options=("Compliance Assessment", "Synthetic Data Generator"),
            index=0,
        )
        st.divider()

        if active_tab == "Compliance Assessment":
            st.subheader("Assessment Settings")
            default_model = resolve_default_model_path()
            model_path_text = st.text_input(
                "Naive Bayes model checkpoint",
                value=str(default_model) if default_model is not None else "",
                help="Optional checkpoint path for model-signal scoring. Leave blank to auto-detect.",
            )
            st.caption("Supported policy files: .txt, .md, .pdf")
            st.caption("Supported schema files: .sql, .txt, .json, .yaml, .yml")
        else:
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

    if active_tab == "Compliance Assessment":
        _render_hero(
            title="Compliance Studio",
            body=(
                "Upload a company privacy policy and database schema to generate a compliance score, "
                "control coverage checks, and evidence-backed findings."
            ),
        )
        _render_compliance_screen(model_path_text=model_path_text)
        return

    _render_hero(
        title="Synthetic Data Studio",
        body=(
            "Generate synthetic privacy policy and schema pairs across low, medium, and high compliance bands "
            "for testing, demos, and evaluator regression checks."
        ),
    )
    _render_synthetic_generation_screen(settings=generation_settings)


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


def _render_compliance_screen(model_path_text: str) -> None:
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
