from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


SPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "__MODEL_ID__"
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
GENERATED_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "prert-cnm-space"
BENCHMARKS_PATH = SPACE_ROOT / "benchmarks.json"


def _discover_project_root() -> Path:
    for root in (SPACE_ROOT, *SPACE_ROOT.parents):
        if (root / "deployment").exists() or (root / "artifacts").exists():
            return root
    return SPACE_ROOT


PROJECT_ROOT = _discover_project_root()


def _add_local_source_path() -> None:
    for root in (SPACE_ROOT, *SPACE_ROOT.parents):
        source_dir = root / "src"
        if (source_dir / "prert").exists() and str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
            return


_add_local_source_path()

try:
    from prert.phase4.compliance_assessor import (
        assess_policy_compliance,
        assess_policy_compliance_stream,
        list_available_regulations,
        resolve_default_model_path,
    )
    from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset

    PHASE4_IMPORT_ERROR: Optional[Exception] = None
except Exception as phase4_exc:  # pragma: no cover - runtime deployment guard
    assess_policy_compliance = None  # type: ignore[assignment]
    assess_policy_compliance_stream = None  # type: ignore[assignment]
    list_available_regulations = None  # type: ignore[assignment]
    resolve_default_model_path = None  # type: ignore[assignment]
    generate_synthetic_policy_schema_dataset = None  # type: ignore[assignment]
    PHASE4_IMPORT_ERROR = phase4_exc


def _load_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"revision": MODEL_REVISION}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


@lru_cache(maxsize=1)
def get_classifier():
    load_kwargs = _load_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **load_kwargs)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def classify_text(text: str) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if not text or not text.strip():
        return {}, []

    raw_scores = get_classifier()(text.strip(), top_k=None)
    if raw_scores and isinstance(raw_scores[0], list):
        raw_scores = raw_scores[0]

    sorted_scores = sorted(raw_scores, key=lambda score: score["score"], reverse=True)
    label_scores = {score["label"]: float(score["score"]) for score in sorted_scores}
    return label_scores, sorted_scores


def run_compliance_assessment(
    policy_file: Any,
    policy_text: str,
    selected_regulations: Optional[List[str]],
) -> Iterator[Tuple[str, List[List[Any]], List[List[Any]], List[List[Any]], str, Dict[str, Any], Optional[str], str, str, List[List[Any]]]]:
    if PHASE4_IMPORT_ERROR is not None:
        yield _compliance_error(f"Phase 4 modules could not be imported: {PHASE4_IMPORT_ERROR}")
        return

    policy_result = _resolve_upload_or_text(policy_file, policy_text, allow_pdf=True, label="policy")
    if policy_result["error"]:
        yield _compliance_error(str(policy_result["error"]))
        return

    resolved_policy_text = str(policy_result["text"]).strip()
    if not resolved_policy_text:
        yield _compliance_error("Provide a privacy policy by upload or paste-in text.")
        return

    if assess_policy_compliance_stream is None:
        yield _compliance_error("Streaming compliance assessor is unavailable in this runtime.")
        return

    graph_state: Dict[str, Any] = {"nodes": {}, "edges": []}
    event_rows: List[List[Any]] = []
    latest_stage = "Waiting"
    final_result: Dict[str, Any] = {}
    framework_stats: Dict[str, Dict[str, float]] = {}
    started_at = time.perf_counter()

    try:
        for event in assess_policy_compliance_stream(  # type: ignore[misc]
            policy_text=resolved_policy_text,
            model_path=_resolve_model_path(""),
            selected_regulations=selected_regulations,
        ):
            event_name = str(event.get("event", ""))
            latest_stage = str(event.get("stage", latest_stage)).replace("_", " ").title()
            _append_event_row(event_rows, event_name, latest_stage, event)
            _update_graph_state(graph_state, event)
            _update_framework_stats(framework_stats, event)

            if event_name == "complete":
                final_result = _as_dict(event.get("result"))
                break

            yield (
                "### Compliance Assessment\n- Processing policy and evidence...",
                [],
                [],
                [],
                "### Findings And Evidence\n- Analysis in progress...",
                {},
                None,
                _format_live_stage(latest_stage, event_rows, framework_stats, started_at),
                _render_live_graph_html(graph_state),
                event_rows[-60:],
            )
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        yield _compliance_error(f"Compliance assessment failed: {exc}")
        return

    if not final_result:
        yield _compliance_error("Compliance assessment did not produce a final result.")
        return

    report_path = _write_temp_json("compliance_report", final_result)
    yield (
        _format_compliance_summary(final_result),
        _regulation_rows(final_result),
        _policy_check_rows(final_result),
        _claim_rows(final_result),
        _detail_markdown(final_result),
        final_result,
        report_path,
        _format_live_stage("Finalize", event_rows, framework_stats, started_at),
        _render_live_graph_html(graph_state),
        event_rows[-60:],
    )


def run_synthetic_generation(
    low_count: float,
    medium_count: float,
    high_count: float,
    seed: float,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, List[List[Any]], Any, str, str, Dict[str, Any], Dict[str, Any], Optional[str], Optional[str], Optional[str], List[Dict[str, Any]]]:
    if PHASE4_IMPORT_ERROR is not None:
        return f"### Unable to Generate\n\nPhase 4 modules could not be imported: {PHASE4_IMPORT_ERROR}", [], gr.update(choices=[], value=None), "", "", {}, {}, None, None, None, []

    output_dir = _new_synthetic_output_dir()
    total_requested = int(low_count) + int(medium_count) + int(high_count)

    def progress_callback(event: Dict[str, Any]) -> None:
        event_name = str(event.get("event", ""))
        if event_name == "sample_complete":
            completed = int(event.get("index", 0))
            total = max(1, int(event.get("total", total_requested or 1)))
            progress(completed / total, desc=f"Generated {completed}/{total} samples")
        elif event_name == "complete":
            progress(1.0, desc="Synthetic generation complete")

    try:
        manifest = generate_synthetic_policy_schema_dataset(  # type: ignore[misc]
            output_dir=output_dir,
            counts_by_band={
                "low": int(low_count),
                "medium": int(medium_count),
                "high": int(high_count),
            },
            seed=int(seed),
            include_model_signal=False,
            model_path=None,
            export_upload_fixtures=True,
            progress_callback=progress_callback,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return f"### Synthetic Generation Failed\n\n{exc}", [], gr.update(choices=[], value=None), "", "", {}, {}, None, None, None, []

    output_files = _as_dict(manifest.get("output_files"))
    dataset_path = Path(str(output_files.get("dataset", "")))
    manifest_path = Path(str(output_files.get("manifest", "")))
    rows = _read_jsonl_file(dataset_path)
    choices = _synthetic_sample_choices(rows)
    selected_choice = choices[0] if choices else None
    policy_text, schema_text, selected_row = select_synthetic_sample(selected_choice, rows)
    fixture_dir = Path(str(_as_dict(manifest.get("upload_fixtures")).get("output_dir", "")))
    fixture_zip_path = _zip_directory(fixture_dir, output_dir / "synthetic_upload_fixtures.zip")
    summary = _format_synthetic_summary(manifest)
    return (
        summary,
        _synthetic_rows(manifest),
        gr.update(choices=choices, value=selected_choice),
        policy_text,
        schema_text,
        selected_row,
        manifest,
        str(dataset_path) if dataset_path.exists() else None,
        str(manifest_path) if manifest_path.exists() else None,
        fixture_zip_path,
        rows,
    )


def show_model_benchmarks() -> Tuple[str, List[List[Any]], List[List[Any]], Dict[str, Any], Optional[str]]:
    registry = _load_benchmark_registry()
    rows = _benchmark_leaderboard_rows(registry)
    per_class_rows = _benchmark_per_class_rows(registry)
    report_path = _write_temp_json("benchmark_registry", registry)
    return _format_benchmark_summary(registry), rows, per_class_rows, registry, report_path

def select_synthetic_sample(choice: Any, rows: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
    if not rows:
        return "", "", {}

    selected_id = str(choice or "").split(" | ", 1)[0].strip()
    selected = rows[0]
    if selected_id:
        for row in rows:
            if str(row.get("sample_id", "")) == selected_id:
                selected = row
                break

    return str(selected.get("policy_text", "")), str(selected.get("schema_text", "")), dict(selected)


def scan_artifacts(root_dir_text: str, search_query: str, max_files: float) -> Tuple[str, List[List[Any]]]:
    root_dir = Path(str(root_dir_text or "artifacts")).expanduser()
    if not root_dir.exists() or not root_dir.is_dir():
        return f"### Artifacts Not Found\n\n`{root_dir}` is not a readable directory.", []

    query = str(search_query or "").strip().lower()
    rows: List[List[Any]] = []
    for path in sorted(root_dir.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root_dir).parts):
            continue
        relative = path.relative_to(root_dir).as_posix()
        if query and query not in relative.lower():
            continue
        rows.append([relative, path.suffix.lower() or "n/a", _format_byte_size(path.stat().st_size)])
        if len(rows) >= int(max_files):
            break

    return f"### Artifact Files\n\nFound {len(rows)} file(s) under `{root_dir}`.", rows


def preview_artifact(root_dir_text: str, file_path_text: str, max_chars: float) -> Tuple[str, str, Optional[str]]:
    root_dir = Path(str(root_dir_text or "artifacts")).expanduser()
    requested_path = Path(str(file_path_text or "").strip()).expanduser()
    if not str(requested_path):
        return "### No File Selected", "", None

    path = requested_path if requested_path.is_absolute() else root_dir / requested_path
    if not path.exists() or not path.is_file():
        return f"### File Not Found\n\n`{path}` is not a readable file.", "", None

    suffix = path.suffix.lower()
    try:
        if suffix in {".json", ".jsonl", ".md", ".txt", ".sql", ".yaml", ".yml", ".csv", ".log"}:
            content = path.read_text(encoding="utf-8", errors="replace")
        else:
            return _format_file_metadata(path, root_dir), "Binary preview is not available.", str(path)
    except OSError as exc:
        return f"### Preview Failed\n\n{exc}", "", None

    limit = max(1, int(max_chars))
    preview = content[:limit]
    if len(content) > limit:
        preview += "\n\n[preview truncated]"
    return _format_file_metadata(path, root_dir), preview, str(path)


def _compliance_error(message: str) -> Tuple[str, List[List[Any]], List[List[Any]], List[List[Any]], str, Dict[str, Any], Optional[str], str, str, List[List[Any]]]:
    stage = "### Stage Progress\n- Status: Failed"
    graph = _render_live_graph_html({"nodes": {}, "edges": []})
    return f"### Unable to Assess\n\n{message}", [], [], [], "", {}, None, stage, graph, []


def _append_event_row(event_rows: List[List[Any]], event_name: str, stage: str, event: Mapping[str, Any]) -> None:
    detail_bits: List[str] = []
    for key in ("check_id", "regulation", "control_id", "predicted_label", "compliant", "evidence_source"):
        if key in event:
            detail_bits.append(f"{key}={event.get(key)}")
    event_rows.append([stage, event_name, "; ".join(detail_bits)])


def _update_graph_state(graph_state: Dict[str, Any], event: Mapping[str, Any]) -> None:
    nodes = _as_dict(graph_state.get("nodes"))
    edges = _as_list(graph_state.get("edges"))
    event_name = str(event.get("event", ""))

    if event_name == "clause_start":
        claim_index = int(event.get("claim_index", 0))
        claim_node = f"claim_{claim_index}"
        nodes.setdefault(claim_node, {"id": claim_node, "label": f"Claim {claim_index + 1}", "type": "claim"})

    if event_name == "verdict_complete":
        claim_index = int(event.get("claim_index", 0))
        claim_node = f"claim_{claim_index}"
        regulation = str(event.get("regulation", "Unknown"))
        control_id = str(event.get("control_id", "Control"))
        control_node = f"{regulation}:{control_id}"
        regulation_node = f"reg_{regulation}"

        nodes.setdefault(claim_node, {"id": claim_node, "label": f"Claim {claim_index + 1}", "type": "claim"})
        nodes.setdefault(control_node, {"id": control_node, "label": control_id, "type": "control"})
        nodes.setdefault(regulation_node, {"id": regulation_node, "label": regulation, "type": "regulation"})

        compliant = bool(event.get("compliant", False))
        edges.append({"src": claim_node, "dst": control_node, "ok": compliant})
        edges.append({"src": control_node, "dst": regulation_node, "ok": compliant})

    graph_state["nodes"] = nodes
    graph_state["edges"] = edges[-220:]


def _update_framework_stats(framework_stats: Dict[str, Dict[str, float]], event: Mapping[str, Any]) -> None:
    if str(event.get("event", "")) != "verdict_complete":
        return
    regulation = str(event.get("regulation", "")).strip()
    if not regulation:
        return
    now = time.perf_counter()
    stats = framework_stats.setdefault(regulation, {"pass": 0.0, "fail": 0.0, "first": now, "last": now})
    if bool(event.get("compliant", False)):
        stats["pass"] += 1.0
    else:
        stats["fail"] += 1.0
    stats["last"] = now


def _format_live_stage(
    stage: str,
    event_rows: Sequence[Sequence[Any]],
    framework_stats: Mapping[str, Mapping[str, float]],
    started_at: float,
) -> str:
    total_events = len(event_rows)
    verdict_events = sum(1 for row in event_rows if len(row) > 1 and str(row[1]) == "verdict_complete")
    elapsed = max(0.0, time.perf_counter() - started_at)
    lines = [
        "### Stage Progress",
        f"- Current stage: **{stage}**",
        f"- Events processed: {total_events}",
        f"- Verdict links formed: {verdict_events}",
        f"- Elapsed: {elapsed:.2f}s",
    ]

    if framework_stats:
        lines.append("- Framework metrics:")
        for regulation, stats in sorted(framework_stats.items()):
            pass_count = int(float(stats.get("pass", 0.0)))
            fail_count = int(float(stats.get("fail", 0.0)))
            span = max(0.0, float(stats.get("last", 0.0)) - float(stats.get("first", 0.0)))
            lines.append(f"  - {regulation}: pass={pass_count}, fail={fail_count}, active_window={span:.2f}s")

    lines.append("- Visualization note: this graph shows evidence-link formation, not hidden neural activations.")
    return "\n".join(
        lines
    )


def _render_live_graph_html(graph_state: Mapping[str, Any]) -> str:
    nodes = _as_dict(graph_state.get("nodes"))
    edges = _as_list(graph_state.get("edges"))
    node_list = list(nodes.values())

    claim_nodes = [node for node in node_list if _as_dict(node).get("type") == "claim"]
    control_nodes = [node for node in node_list if _as_dict(node).get("type") == "control"]
    reg_nodes = [node for node in node_list if _as_dict(node).get("type") == "regulation"]

    if not node_list:
        return "<div style='padding:12px;border:1px solid #d6d6d6;border-radius:10px;background:#fafaf8;'>Awaiting events to form the evidence graph.</div>"

    positions: Dict[str, Tuple[float, float]] = {}
    width = 980
    height = 420

    def _place(group: Sequence[Mapping[str, Any]], x: float) -> None:
        if not group:
            return
        step = height / (len(group) + 1)
        for idx, node in enumerate(group, start=1):
            node_id = str(node.get("id", f"node_{idx}"))
            positions[node_id] = (x, step * idx)

    _place(claim_nodes, 120.0)
    _place(control_nodes, 500.0)
    _place(reg_nodes, 860.0)

    svg_lines: List[str] = [
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        "<defs><style>.lbl{font:12px sans-serif;fill:#213547}.muted{font:11px sans-serif;fill:#5b6b72}</style></defs>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#f6f9f7' rx='12' />",
        "<text class='muted' x='80' y='24'>Claims</text><text class='muted' x='470' y='24'>Controls</text><text class='muted' x='830' y='24'>Frameworks</text>",
    ]

    for edge in edges:
        edge_map = _as_dict(edge)
        src = str(edge_map.get("src", ""))
        dst = str(edge_map.get("dst", ""))
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        color = "#2f9e44" if edge_map.get("ok") else "#c92a2a"
        svg_lines.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-opacity='0.45' stroke-width='1.6' />")

    for node in node_list:
        node_map = _as_dict(node)
        node_id = str(node_map.get("id", ""))
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_type = str(node_map.get("type", ""))
        color = "#0ca678" if node_type == "claim" else "#1971c2" if node_type == "control" else "#f08c00"
        label = str(node_map.get("label", node_id))[:30]
        svg_lines.append(f"<circle cx='{x}' cy='{y}' r='8' fill='{color}' />")
        svg_lines.append(f"<text class='lbl' x='{x + 12}' y='{y + 4}'>{label}</text>")

    svg_lines.append("</svg>")
    return "".join(svg_lines)


def _resolve_upload_or_text(uploaded_file: Any, text_value: str, allow_pdf: bool, label: str) -> Dict[str, Any]:
    if text_value and text_value.strip():
        return {"text": text_value, "error": None, "file_name": "pasted text", "size_bytes": len(text_value.encode("utf-8"))}
    if not _has_uploaded_file(uploaded_file):
        return {"text": "", "error": None, "file_name": "", "size_bytes": 0}

    path = _uploaded_file_path(uploaded_file)
    if path is None or not path.exists():
        return {"text": "", "error": f"The uploaded {label} file could not be read.", "file_name": "", "size_bytes": 0}

    file_name = _uploaded_file_name(uploaded_file) or path.name
    suffix = Path(file_name).suffix.lower() or path.suffix.lower()
    file_bytes = path.read_bytes()

    if suffix == ".pdf":
        if not allow_pdf:
            return {"text": "", "error": f"PDF uploads are not supported for {label} files.", "file_name": file_name, "size_bytes": len(file_bytes)}
        try:
            pypdf_module = importlib.import_module("pypdf")
            reader = pypdf_module.PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:  # pragma: no cover - runtime upload guard
            return {"text": "", "error": f"PDF parsing failed: {exc}", "file_name": file_name, "size_bytes": len(file_bytes)}
        if not text:
            return {"text": "", "error": "PDF has no extractable text.", "file_name": file_name, "size_bytes": len(file_bytes)}
        return {"text": text, "error": None, "file_name": file_name, "size_bytes": len(file_bytes)}

    decoded = _decode_text_bytes(file_bytes)
    if decoded is None:
        return {"text": "", "error": f"Unable to decode {label} file using utf-8, utf-16, or latin-1.", "file_name": file_name, "size_bytes": len(file_bytes)}
    return {"text": decoded, "error": None, "file_name": file_name, "size_bytes": len(file_bytes)}


def _has_uploaded_file(uploaded_file: Any) -> bool:
    return uploaded_file is not None and str(uploaded_file).strip() != ""


def _uploaded_file_path(uploaded_file: Any) -> Optional[Path]:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    if isinstance(uploaded_file, dict):
        path_text = uploaded_file.get("path") or uploaded_file.get("name")
        return Path(path_text) if path_text else None
    path_text = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None)
    return Path(path_text) if path_text else None


def _uploaded_file_name(uploaded_file: Any) -> str:
    if isinstance(uploaded_file, dict):
        return str(uploaded_file.get("orig_name") or uploaded_file.get("name") or "")
    return str(getattr(uploaded_file, "orig_name", None) or getattr(uploaded_file, "name", ""))


def _decode_text_bytes(data: bytes) -> Optional[str]:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            decoded = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if decoded.strip():
            return decoded
    return None


def _resolve_model_path(model_path_text: str) -> Optional[Path]:
    candidate_text = str(model_path_text or "").strip()
    if candidate_text:
        return Path(candidate_text).expanduser()
    if resolve_default_model_path is None:
        return None
    return resolve_default_model_path(project_root=PROJECT_ROOT)  # type: ignore[misc]


def _new_synthetic_output_dir() -> Path:
    GENERATED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = GENERATED_OUTPUT_ROOT / f"synthetic-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows

def _synthetic_sample_choices(rows: List[Mapping[str, Any]]) -> List[str]:
    choices: List[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "sample"))
        band = str(row.get("compliance_band", "unknown"))
        score = _format_optional_float(_as_dict(row.get("assessment")).get("overall_score"))
        choices.append(f"{sample_id} | {band} | score {score}")
    return choices

def _zip_directory(source_dir: Path, target_path: Path) -> Optional[str]:
    if not source_dir.exists() or not source_dir.is_dir():
        return None
    with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
    return str(target_path) if target_path.exists() else None


def _format_compliance_summary(result: Mapping[str, Any]) -> str:
    summary = _as_dict(result.get("summary"))
    lines = [
        "### Compliance Assessment",
        "- Mode: Policy only",
        f"- Score: **{float(result.get('overall_score', 0.0)):.1f} / 100**",
        f"- Grade: **{result.get('grade', 'n/a')}**",
        f"- Status: **{result.get('status', 'n/a')}**",
        f"- Clauses analyzed: {summary.get('clauses_analyzed', 0)}",
    ]
    if "claims_generated" in summary:
        lines.append(f"- Claims generated: {summary.get('claims_generated', 0)}")

    bayesian_risk = _as_dict(result.get("bayesian_risk"))
    if bayesian_risk:
        overall = _as_dict(bayesian_risk.get("overall"))
        lines.append(f"- Bayesian risk score: {_format_optional_float(overall.get('primary_score'))}")
    return "\n".join(lines)


def _regulation_rows(result: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    summary = _as_dict(result.get("regulation_summary"))
    for regulation, values in sorted(summary.items()):
        value_map = _as_dict(values)
        rows.append([
            str(regulation).replace("_", " "),
            f"{float(value_map.get('compliance_pct', 0.0)):.1f}%",
            value_map.get("pass_count", 0),
            value_map.get("fail_count", 0),
            value_map.get("total_controls", 0),
        ])
    return rows


def _policy_check_rows(result: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for item in _as_list(result.get("policy_checks")):
        check = _as_dict(item)
        rows.append([
            check.get("title", ""),
            f"{float(check.get('score', 0.0)):.2f}/{float(check.get('max_score', 0.0)):.2f}",
            "Yes" if check.get("passed") else "No",
            ", ".join(str(keyword) for keyword in _as_list(check.get("matched_keywords"))[:6]),
        ])
    return rows


def _claim_rows(result: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    claims = _as_list(result.get("claims"))
    for claim in claims:
        claim_map = _as_dict(claim)
        verdicts = _as_list(claim_map.get("regulation_verdicts"))
        for verdict in verdicts:
            verdict_map = _as_dict(verdict)
            rows.append([
                f"#{int(claim_map.get('claim_index', 0)) + 1}",
                claim_map.get("check_title", ""),
                str(verdict_map.get("regulation", "")).replace("_", " "),
                verdict_map.get("control_id", ""),
                "Pass" if verdict_map.get("compliant") else "Fail",
                verdict_map.get("reason", ""),
            ])
            if len(rows) >= 120:
                return rows

    if rows:
        return rows

    for item in _as_list(result.get("policy_checks")):
        check = _as_dict(item)
        for evidence in _as_list(check.get("evidence")):
            rows.append(["", check.get("title", ""), "", "", "Evidence", str(evidence)])
    return rows[:120]


def _detail_markdown(result: Mapping[str, Any]) -> str:
    lines: List[str] = ["### Findings And Evidence"]
    failed_checks = [
        _as_dict(item)
        for item in _as_list(result.get("policy_checks"))
        if not _as_dict(item).get("passed", False)
    ]
    if failed_checks:
        lines.append("\nPriority policy gaps:")
        for check in failed_checks[:8]:
            lines.append(f"- {check.get('title', '')}: {float(check.get('score', 0.0)):.2f}/{float(check.get('max_score', 0.0)):.2f}")

    failing_verdicts: List[str] = []
    for claim in _as_list(result.get("claims")):
        claim_map = _as_dict(claim)
        for verdict in _as_list(claim_map.get("regulation_verdicts")):
            verdict_map = _as_dict(verdict)
            if verdict_map.get("compliant"):
                continue
            failing_verdicts.append(
                f"- {verdict_map.get('regulation', '')} {verdict_map.get('control_id', '')}: {verdict_map.get('remediation_advice', '')}"
            )
            if len(failing_verdicts) >= 8:
                break
        if len(failing_verdicts) >= 8:
            break
    if failing_verdicts:
        lines.append("\nRegulation remediation:")
        lines.extend(failing_verdicts)

    model_signal = _as_dict(result.get("model_signal"))
    if model_signal:
        lines.append("\nModel signal:")
        lines.append(f"- Score: {float(model_signal.get('score', 0.0)):.2f}/{float(model_signal.get('max_score', 5.0)):.2f}")
        for detail in _as_list(model_signal.get("details"))[:5]:
            lines.append(f"- {detail}")

    if len(lines) == 1:
        lines.append("No priority gaps were detected in the available checks.")
    return "\n".join(lines)


def _format_synthetic_summary(manifest: Mapping[str, Any]) -> str:
    output_files = _as_dict(manifest.get("output_files"))
    return "\n".join([
        "### Synthetic Dataset Generated",
        f"- Dataset: `{output_files.get('dataset', '')}`",
        f"- Manifest: `{output_files.get('manifest', '')}`",
        f"- Dictionary: `{output_files.get('dictionary', '')}`",
        f"- Seed: {manifest.get('seed', '')}",
    ])


def _synthetic_rows(manifest: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    score_summary = _as_dict(manifest.get("score_summary"))
    for band in ("low", "medium", "high"):
        band_summary = _as_dict(score_summary.get(band))
        rows.append([
            band,
            band_summary.get("count"),
            band_summary.get("in_target_band"),
            band_summary.get("minimum"),
            band_summary.get("mean"),
            band_summary.get("maximum"),
        ])
    return rows


def _format_benchmark_summary(registry: Mapping[str, Any]) -> str:
    rows = _as_list(registry.get("models"))
    ranked = _rank_benchmark_models(rows)
    best = _as_dict(ranked[0]) if ranked else {}
    unavailable = [row for row in rows if str(_as_dict(row).get("status", "available")) != "available"]
    notes = _as_list(registry.get("notes"))
    lines = [
        "### Model Benchmark Comparison",
        f"- Dataset: {registry.get('dataset', 'n/a')}",
        f"- Metric source: {registry.get('metric_source', 'stored published metrics')}",
        f"- Models listed: {len(rows)}",
    ]
    if best:
        metrics = _as_dict(best.get("metrics"))
        lines.append(
            f"- Best available test macro F1: **{best.get('display_name', '')}** ({_format_optional_float(metrics.get('test_macro_f1'))})"
        )
    if unavailable:
        lines.append(f"- Reference rows without stored metrics: {len(unavailable)}")
    for note in notes[:3]:
        lines.append(f"- {note}")
    return "\n".join(lines)

def _benchmark_leaderboard_rows(registry: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for rank, model in enumerate(_rank_benchmark_models(_as_list(registry.get("models"))), start=1):
        model_map = _as_dict(model)
        metrics = _as_dict(model_map.get("metrics"))
        rows.append(
            [
                rank if _model_has_primary_metrics(model_map) else "n/a",
                model_map.get("display_name", ""),
                model_map.get("model_family", ""),
                model_map.get("status", "available"),
                _format_optional_float(metrics.get("test_macro_f1")),
                _format_optional_float(metrics.get("test_accuracy")),
                _format_optional_float(metrics.get("validation_macro_f1")),
                _format_optional_float(metrics.get("validation_accuracy")),
                _format_optional_float(metrics.get("bayesian_primary_score")),
                _format_optional_float(metrics.get("calibration_test_ece")),
                model_map.get("notes", ""),
            ]
        )
    return rows

def _benchmark_per_class_rows(registry: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for model in _as_list(registry.get("models")):
        model_map = _as_dict(model)
        per_class = _as_dict(_as_dict(model_map.get("metrics")).get("test_per_class"))
        for label, values in sorted(per_class.items()):
            value_map = _as_dict(values)
            rows.append(
                [
                    model_map.get("display_name", ""),
                    label,
                    _format_optional_float(value_map.get("precision")),
                    _format_optional_float(value_map.get("recall")),
                    _format_optional_float(value_map.get("f1")),
                    value_map.get("support", ""),
                ]
            )
    return rows

def _rank_benchmark_models(models: List[Any]) -> List[Dict[str, Any]]:
    return sorted(
        [_as_dict(model) for model in models],
        key=lambda model: (
            _model_has_primary_metrics(model),
            _metric_float(model, "test_macro_f1"),
            _metric_float(model, "test_accuracy"),
        ),
        reverse=True,
    )

def _model_has_primary_metrics(model: Mapping[str, Any]) -> bool:
    metrics = _as_dict(model.get("metrics"))
    return metrics.get("test_macro_f1") is not None and metrics.get("test_accuracy") is not None

def _metric_float(model: Mapping[str, Any], name: str) -> float:
    try:
        return float(_as_dict(model.get("metrics")).get(name))
    except (TypeError, ValueError):
        return -1.0

def _load_benchmark_registry() -> Dict[str, Any]:
    if not BENCHMARKS_PATH.exists():
        return {
            "dataset": "n/a",
            "metric_source": "missing benchmark registry",
            "models": [],
            "notes": [f"Benchmark registry not found: {BENCHMARKS_PATH}"],
        }
    with BENCHMARKS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"models": [], "notes": ["Invalid benchmark registry format."]}


def _format_file_metadata(path: Path, root_dir: Path) -> str:
    try:
        relative = path.relative_to(root_dir).as_posix()
    except ValueError:
        relative = path.name
    return "\n".join([
        "### Artifact Preview",
        f"- File: `{relative}`",
        f"- Size: {_format_byte_size(path.stat().st_size)}",
        f"- Type: `{path.suffix.lower() or 'n/a'}`",
    ])


def _format_byte_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _format_optional_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _write_temp_json(name: str, payload: Mapping[str, Any]) -> str:
    output_dir = Path(tempfile.gettempdir()) / "prert-phase4-gradio"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _gradio_major_version() -> int:
    version_text = str(getattr(gr, "__version__", "0"))
    try:
        return int(version_text.split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def _blocks_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"title": "PrERT-CNM Compliance Studio"}
    if _gradio_major_version() < 6:
        kwargs["theme"] = theme
    return kwargs


def _default_server_port() -> int:
    port_value = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or "7860"
    try:
        return int(port_value)
    except (TypeError, ValueError):
        return 7860


def _running_in_space() -> bool:
    return any(
        os.getenv(variable_name)
        for variable_name in ("SPACE_ID", "SPACE_HOST", "HF_SPACE_ID", "SYSTEM")
    )


def _launch_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if _gradio_major_version() >= 6:
        kwargs["theme"] = theme
    if _running_in_space():
        kwargs["server_name"] = "0.0.0.0"
        kwargs["server_port"] = _default_server_port()
    return kwargs


def _event_kwargs() -> Dict[str, Any]:
    if _gradio_major_version() < 6:
        return {"show_api": False}
    return {}


def _available_regulation_choices() -> List[str]:
    if list_available_regulations is None:
        return ["GDPR", "NIST", "ISO_27701"]
    try:
        choices = [str(item) for item in list_available_regulations() if str(item).strip()]  # type: ignore[misc]
    except Exception:
        return ["GDPR", "NIST", "ISO_27701"]
    if not choices:
        return ["GDPR", "NIST", "ISO_27701"]
    return choices


theme = gr.themes.Soft(primary_hue="teal", neutral_hue="stone")

with gr.Blocks(**_blocks_kwargs()) as demo:
    gr.Markdown("# PrERT-CNM Compliance Studio")
    regulation_choices = _available_regulation_choices()

    with gr.Tabs():
        with gr.Tab("Compliance Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    policy_file = gr.File(label="Privacy policy", file_types=[".txt", ".md", ".pdf"], type="filepath")
                    policy_text = gr.Textbox(label="Policy text", lines=8, max_lines=18)
                    regulation_selector = gr.CheckboxGroup(
                        label="Frameworks to evaluate",
                        choices=regulation_choices,
                        value=regulation_choices,
                    )
                    assess_button = gr.Button("Analyze Compliance", variant="primary")
                with gr.Column(scale=2):
                    compliance_summary = gr.Markdown()
                    live_stage_markdown = gr.Markdown(value="### Stage Progress\n- Waiting for input")
                    live_graph_html = gr.HTML(value="<div style='padding:12px;border:1px solid #d6d6d6;border-radius:10px;background:#fafaf8;'>Awaiting events to form the evidence graph.</div>")
                    live_event_table = gr.Dataframe(
                        headers=["Stage", "Event", "Details"],
                        interactive=False,
                    )
                    with gr.Tabs():
                        with gr.Tab("Regulations"):
                            regulation_table = gr.Dataframe(
                                headers=["Regulation", "Compliance", "Pass", "Fail", "Controls"],
                                interactive=False,
                            )
                        with gr.Tab("Policy Checks"):
                            policy_check_table = gr.Dataframe(
                                headers=["Check", "Score", "Passed", "Keywords"],
                                interactive=False,
                            )
                        with gr.Tab("Claims"):
                            claim_table = gr.Dataframe(
                                headers=["Claim", "Check", "Regulation", "Control", "Status", "Reason"],
                                interactive=False,
                            )
                    details_markdown = gr.Markdown()
                    compliance_json = gr.JSON(label="Compliance report")
                    compliance_download = gr.File(label="Download compliance JSON")

            assess_button.click(
                run_compliance_assessment,
                inputs=[policy_file, policy_text, regulation_selector],
                outputs=[
                    compliance_summary,
                    regulation_table,
                    policy_check_table,
                    claim_table,
                    details_markdown,
                    compliance_json,
                    compliance_download,
                    live_stage_markdown,
                    live_graph_html,
                    live_event_table,
                ],
                **_event_kwargs(),
            )

        with gr.Tab("Synthetic Data"):
            with gr.Row():
                with gr.Column(scale=1):
                    low_count = gr.Number(label="Low compliance samples", value=6, precision=0)
                    medium_count = gr.Number(label="Medium compliance samples", value=6, precision=0)
                    high_count = gr.Number(label="High compliance samples", value=6, precision=0)
                    synthetic_seed = gr.Number(label="Random seed", value=42, precision=0)
                    generate_button = gr.Button("Generate Synthetic Data", variant="primary")
                with gr.Column(scale=2):
                    synthetic_summary = gr.Markdown()
                    synthetic_table = gr.Dataframe(
                        headers=["Band", "Count", "In Target", "Min", "Mean", "Max"],
                        interactive=False,
                    )
                    synthetic_sample_rows = gr.State([])
                    synthetic_sample_choice = gr.Dropdown(label="Generated sample", choices=[])
                    with gr.Row():
                        synthetic_policy_text = gr.Textbox(label="Policy text", lines=14, max_lines=24)
                        synthetic_schema_text = gr.Textbox(label="SQL schema", lines=14, max_lines=24)
                    synthetic_sample_json = gr.JSON(label="Selected sample")
                    synthetic_json = gr.JSON(label="Synthetic manifest")
                    with gr.Row():
                        synthetic_dataset_download = gr.File(label="Download dataset JSONL")
                        synthetic_download = gr.File(label="Download manifest JSON")
                        synthetic_fixture_download = gr.File(label="Download upload fixtures ZIP")

            generate_button.click(
                run_synthetic_generation,
                inputs=[
                    low_count,
                    medium_count,
                    high_count,
                    synthetic_seed,
                ],
                outputs=[
                    synthetic_summary,
                    synthetic_table,
                    synthetic_sample_choice,
                    synthetic_policy_text,
                    synthetic_schema_text,
                    synthetic_sample_json,
                    synthetic_json,
                    synthetic_dataset_download,
                    synthetic_download,
                    synthetic_fixture_download,
                    synthetic_sample_rows,
                ],
                **_event_kwargs(),
            )
            synthetic_sample_choice.change(
                select_synthetic_sample,
                inputs=[synthetic_sample_choice, synthetic_sample_rows],
                outputs=[synthetic_policy_text, synthetic_schema_text, synthetic_sample_json],
                **_event_kwargs(),
            )

        with gr.Tab("Benchmark Validation"):
            with gr.Row():
                with gr.Column(scale=1):
                    benchmark_button = gr.Button("Refresh Model Results", variant="primary")
                with gr.Column(scale=2):
                    benchmark_summary = gr.Markdown()
                    benchmark_leaderboard = gr.Dataframe(
                        headers=["Rank", "Model", "Family", "Status", "Test Macro F1", "Test Accuracy", "Validation Macro F1", "Validation Accuracy", "Bayesian Score", "Calibration ECE", "Notes"],
                        interactive=False,
                    )
                    benchmark_per_class = gr.Dataframe(
                        headers=["Model", "Class", "Precision", "Recall", "F1", "Support"],
                        interactive=False,
                    )
                    benchmark_json = gr.JSON(label="Benchmark report")
                    benchmark_download = gr.File(label="Download benchmark JSON")

            benchmark_button.click(
                show_model_benchmarks,
                inputs=[],
                outputs=[benchmark_summary, benchmark_leaderboard, benchmark_per_class, benchmark_json, benchmark_download],
                **_event_kwargs(),
            )
            demo.load(
                show_model_benchmarks,
                inputs=[],
                outputs=[benchmark_summary, benchmark_leaderboard, benchmark_per_class, benchmark_json, benchmark_download],
                **_event_kwargs(),
            )

        with gr.Tab("Data Explorer"):
            with gr.Row():
                with gr.Column(scale=1):
                    explorer_root = gr.Textbox(label="Artifacts root directory", value="artifacts")
                    explorer_query = gr.Textbox(label="Filter")
                    explorer_limit = gr.Number(label="Max files", value=500, precision=0)
                    scan_button = gr.Button("Scan Artifacts", variant="primary")
                    preview_path = gr.Textbox(label="Preview file path")
                    preview_limit = gr.Number(label="Preview characters", value=12000, precision=0)
                    preview_button = gr.Button("Preview File")
                with gr.Column(scale=2):
                    explorer_summary = gr.Markdown()
                    explorer_table = gr.Dataframe(headers=["Path", "Type", "Size"], interactive=False)
                    preview_summary = gr.Markdown()
                    preview_text = gr.Textbox(label="Preview", lines=18, max_lines=32, interactive=False)
                    preview_download = gr.File(label="Download selected file")

            scan_button.click(
                scan_artifacts,
                inputs=[explorer_root, explorer_query, explorer_limit],
                outputs=[explorer_summary, explorer_table],
                **_event_kwargs(),
            )
            preview_button.click(
                preview_artifact,
                inputs=[explorer_root, preview_path, preview_limit],
                outputs=[preview_summary, preview_text, preview_download],
                **_event_kwargs(),
            )

        with gr.Tab("PrivacyBERT Classifier"):
            with gr.Row():
                classifier_input = gr.Textbox(
                    label="Text",
                    lines=8,
                    max_lines=16,
                    placeholder="Paste text to classify.",
                )
                classifier_scores = gr.Label(label="Scores", num_top_classes=8)
            classifier_json = gr.JSON(label="Raw scores")
            with gr.Row():
                classify_button = gr.Button("Classify", variant="primary")
                gr.ClearButton([classifier_input, classifier_scores, classifier_json], value="Clear")

            gr.Examples(
                examples=[
                    "We collect your email address to provide account notifications and support.",
                    "The service may share aggregated analytics with trusted partners.",
                    "Users can request deletion of their stored profile data.",
                ],
                inputs=classifier_input,
            )

            classify_button.click(
                classify_text,
                inputs=classifier_input,
                outputs=[classifier_scores, classifier_json],
                **_event_kwargs(),
            )
            classifier_input.submit(
                classify_text,
                inputs=classifier_input,
                outputs=[classifier_scores, classifier_json],
                **_event_kwargs(),
            )


if __name__ == "__main__":
    demo.queue().launch(**_launch_kwargs())