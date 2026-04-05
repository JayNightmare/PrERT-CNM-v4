from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    art = root / "artifacts" / "phase-1"

    files = {
        "gdpr": {
            "controls": art / "controls_gdpr.jsonl",
            "chunks": art / "chunks_gdpr.jsonl",
            "source": root / "docs/Standards/Regulations/TX/GDPR-2016_679.txt",
        },
        "iso27001": {
            "controls": art / "controls_iso27001.jsonl",
            "chunks": art / "chunks_iso27001.jsonl",
            "source": root / "docs/Standards/Regulations/TX/ISO_27001_Standard-1.txt",
        },
        "nistpf": {
            "controls": art / "controls_nistpf.jsonl",
            "chunks": art / "chunks_nistpf.jsonl",
            "source": art / "NIST-1.1.txt",
        },
    }

    all_ok = True
    print("=== FILE LAYER CHECKS ===")

    for reg, cfg in files.items():
        controls = read_jsonl(cfg["controls"])
        chunks = read_jsonl(cfg["chunks"])

        source_text = ""
        if cfg["source"].exists():
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
        elif reg == "iso27001":
            id_pattern_ok = all(
                re.match(r"^(?:[1-9]|10|A)(?:\.[0-9]+|\.[a-z])*$", str(row.get("native_id", "")))
                for row in controls
            )
        elif reg == "nistpf":
            id_pattern_ok = all(re.match(r"^[A-Z]{2}\.[A-Z]{2}-P\d+$", str(row.get("native_id", ""))) for row in controls)

        reg_ok = (
            duplicate_record_ids == 0
            and duplicate_chunk_ids == 0
            and missing_chunk_refs == 0
            and metadata_reg_mismatch == 0
            and id_pattern_ok
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

        expected_counts = {
            "gdpr_controls": sum(1 for _ in open(art / "chunks_gdpr.jsonl", encoding="utf-8")),
            "iso27001_controls": sum(1 for _ in open(art / "chunks_iso27001.jsonl", encoding="utf-8")),
            "nist_controls": sum(1 for _ in open(art / "chunks_nistpf.jsonl", encoding="utf-8")),
        }

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
