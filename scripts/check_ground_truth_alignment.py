from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import httpx
from dotenv import load_dotenv

from prert.extract.docx_reader import read_docx_text
from prert.extract.iso_sources import discover_iso_docx_sources


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def collection_name_for_key(key: str) -> str:
    if key == "gdpr":
        return "gdpr_controls"
    if key == "nistpf":
        return "nist_controls"
    return f"{key}_controls"


def load_iso_baseline(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"ISO baseline manifest not found: {path}. Add the canonical clause-ID baseline before validation."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    standards = payload.get("iso_standards")
    if not isinstance(standards, dict):
        raise ValueError("ISO baseline manifest must include an 'iso_standards' object.")

    return standards


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    art = root / "artifacts" / "phase-1"
    regs = root / "docs" / "Standards" / "Regulations"
    baseline_path = regs / "iso_clause_id_baseline.json"

    iso_sources = discover_iso_docx_sources(regs)
    iso_baseline = load_iso_baseline(baseline_path)

    files: Dict[str, Dict[str, Any]] = {
        "gdpr": {
            "controls": art / "controls_gdpr.jsonl",
            "chunks": art / "chunks_gdpr.jsonl",
            "source": regs / "GDPR-2016_679.docx",
        },
        "nistpf": {
            "controls": art / "controls_nistpf.jsonl",
            "chunks": art / "chunks_nistpf.jsonl",
            "source": regs / "NIST-1.1.docx",
        },
    }

    for source in iso_sources:
        files[source.output_stem] = {
            "controls": art / f"controls_{source.output_stem}.jsonl",
            "chunks": art / f"chunks_{source.output_stem}.jsonl",
            "source": source.path,
            "source_document_id": source.source_document_id,
            "regulation": source.regulation,
        }

    all_ok = True
    print("=== FILE LAYER CHECKS ===")

    for reg, cfg in files.items():
        if not cfg["controls"].exists() or not cfg["chunks"].exists():
            print(f"[{reg}] missing controls/chunks artifacts")
            all_ok = False
            continue

        controls = read_jsonl(cfg["controls"])
        chunks = read_jsonl(cfg["chunks"])

        source_text = ""
        if cfg["source"].exists():
            if cfg["source"].suffix.lower() == ".docx":
                source_text = normalize(read_docx_text(cfg["source"]))
            else:
                source_text = normalize(cfg["source"].read_text(encoding="utf-8", errors="ignore"))

        record_ids = [row["record_id"] for row in controls]
        normalized_ids = [row["normalized_id"] for row in controls]
        chunk_ids = [row["chunk_id"] for row in chunks]

        duplicate_record_ids = len(record_ids) - len(set(record_ids))
        duplicate_normalized_ids = len(normalized_ids) - len(set(normalized_ids))
        duplicate_chunk_ids = len(chunk_ids) - len(set(chunk_ids))

        controls_by_id = {row["normalized_id"]: row for row in controls}
        missing_chunk_refs = 0
        metadata_reg_mismatch = 0

        for chunk in chunks:
            control_id = chunk.get("control_id")
            if control_id not in controls_by_id:
                missing_chunk_refs += 1
            if chunk.get("metadata", {}).get("regulation", "").lower() != chunk.get("regulation", "").lower():
                metadata_reg_mismatch += 1

        sample = controls[:20]
        source_matches = 0
        for row in sample:
            probe = normalize(row.get("text", ""))[:120]
            if probe and probe in source_text:
                source_matches += 1

        id_pattern_ok = True
        if reg == "gdpr":
            id_pattern_ok = all(str(row.get("native_id", "")).startswith("Article ") for row in controls)
        elif reg.startswith("iso"):
            id_pattern_ok = all(
                re.match(r"^(?:[1-9]\d*|A)(?:\.[0-9]+|\.[a-z])*$", str(row.get("native_id", "")))
                for row in controls
            )
        elif reg == "nistpf":
            id_pattern_ok = all(re.match(r"^[A-Z]{2}\.[A-Z]{2}-P\d+$", str(row.get("native_id", ""))) for row in controls)

        iso_baseline_ok = True
        missing_ids: list[str] = []
        extra_ids: list[str] = []
        source_file_ok = True

        if reg.startswith("iso"):
            baseline_entry = iso_baseline.get(reg)
            if not isinstance(baseline_entry, dict):
                iso_baseline_ok = False
            else:
                expected_ids = {
                    str(item).strip()
                    for item in baseline_entry.get("expected_clause_ids", [])
                    if str(item).strip()
                }
                actual_ids = {
                    str(row.get("native_id", "")).strip()
                    for row in controls
                    if str(row.get("native_id", "")).strip()
                }

                missing_ids = sorted(expected_ids - actual_ids)
                extra_ids = sorted(actual_ids - expected_ids)
                iso_baseline_ok = not missing_ids and not extra_ids

                manifest_source = str(baseline_entry.get("source_file", "")).strip()
                source_file_ok = manifest_source == cfg["source"].name

        reg_ok = (
            duplicate_record_ids == 0
            and duplicate_chunk_ids == 0
            and missing_chunk_refs == 0
            and metadata_reg_mismatch == 0
            and id_pattern_ok
            and iso_baseline_ok
            and source_file_ok
        )
        all_ok = all_ok and reg_ok

        print(f"[{reg}] controls={len(controls)} chunks={len(chunks)}")
        print(
            f"  duplicate record_id={duplicate_record_ids}, duplicate normalized_id={duplicate_normalized_ids}, duplicate chunk_id={duplicate_chunk_ids}"
        )
        print(
            f"  chunk->control missing refs={missing_chunk_refs}, metadata regulation mismatch={metadata_reg_mismatch}"
        )
        print(f"  source sample match={source_matches}/{len(sample)}")
        print(f"  id pattern check={'PASS' if id_pattern_ok else 'FAIL'}")
        if reg.startswith("iso"):
            print(
                "  baseline exact match="
                f"{'PASS' if iso_baseline_ok else 'FAIL'} "
                f"missing={len(missing_ids)} extra={len(extra_ids)}"
            )
            print(f"  baseline source file check={'PASS' if source_file_ok else 'FAIL'}")
            if missing_ids:
                print(f"  missing ids sample={missing_ids[:10]}")
            if extra_ids:
                print(f"  extra ids sample={extra_ids[:10]}")

    print("\n=== CHROMA LAYER CHECKS ===")
    load_dotenv(root / ".env")

    base_url = os.getenv("CHROMA_HOST", "api.trychroma.com")
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = f"https://{base_url}"

    tenant = os.getenv("CHROMA_TENANT", "")
    db_config = os.getenv("CHROMA_DATABASE", "")
    api_key = os.getenv("CHROMA_API_KEY", "")

    headers = {"x-chroma-token": api_key}

    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        ident_resp = client.get("/api/v2/auth/identity")
        ident_resp.raise_for_status()
        identity = ident_resp.json()

        databases = [str(x) for x in identity.get("databases", [])]
        resolved_db = next((db for db in databases if db.lower() == db_config.lower()), db_config)

        print(f"tenant={identity.get('tenant')}")
        print(f"configured_db={db_config} resolved_db={resolved_db}")

        collections_resp = client.get(f"/api/v2/tenants/{tenant}/databases/{resolved_db}/collections")
        collections_resp.raise_for_status()
        collections = collections_resp.json()
        by_name = {item.get("name"): item for item in collections}

        expected_counts: Dict[str, int] = {}
        for reg, cfg in files.items():
            chunk_path = cfg["chunks"]
            if not chunk_path.exists():
                continue
            expected_counts[collection_name_for_key(reg)] = count_jsonl(chunk_path)

        for name, expected in expected_counts.items():
            col = by_name.get(name)
            if not col:
                print(f"[{name}] missing in cloud")
                all_ok = False
                continue

            collection_id = col.get("id")
            count_resp = client.get(
                f"/api/v2/tenants/{tenant}/databases/{resolved_db}/collections/{collection_id}/count"
            )
            count_resp.raise_for_status()
            actual = int(count_resp.json())
            status = "PASS" if actual == expected else "MISMATCH"
            if actual != expected:
                all_ok = False
            print(f"[{name}] expected={expected} actual={actual} status={status}")

    print("\nOVERALL ALIGNMENT STATUS:", "PASS" if all_ok else "CHECK REQUIRED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
