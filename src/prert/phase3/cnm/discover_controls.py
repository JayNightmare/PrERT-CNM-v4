"""Find and describe a Phase 1 controls file so `build` knows where to look.

Walks `artifacts/phase-1/` and reports every JSON/JSONL that plausibly
contains a controls list. Prints size, first entry keys, and an example
control_id + text. No mutation.

Usage:
    python -m prert.phase3.cnm.discover_controls
    python -m prert.phase3.cnm.discover_controls --search-root artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

CONTROL_KEYS = {
    "control_id",
    "id",
    "text",
    "control_text",
    "body",
    "source",
    "framework",
    "section",
}


def _plausible_control_file(path: Path) -> bool:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    text = text.strip()
    if not text:
        return False
    sample_line = text.splitlines()[0].strip()
    try:
        obj = json.loads(sample_line)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            obj = parsed[0]
        elif isinstance(parsed, dict):
            for k in ("controls", "entries", "items", "data"):
                if k in parsed and isinstance(parsed[k], list) and parsed[k]:
                    obj = parsed[k][0]
                    break
            else:
                return False
        else:
            return False
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    return bool(keys & CONTROL_KEYS) and any(
        k in keys for k in ("text", "control_text", "body")
    )


def _describe(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    size = path.stat().st_size
    entries: List[dict] = []
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines()[:5]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(parsed, list):
            entries = parsed[:5]
        elif isinstance(parsed, dict):
            for k in ("controls", "entries", "items", "data"):
                if k in parsed and isinstance(parsed[k], list):
                    entries = parsed[k][:5]
                    break
    print(f"  path: {path}")
    print(f"  size: {size} bytes")
    if entries:
        first = entries[0]
        print(f"  first keys: {sorted(first.keys())}")
        for k in ("control_id", "id", "source", "framework", "section"):
            if k in first:
                print(f"  first.{k}: {first[k]!r}")
        for k in ("text", "control_text", "body"):
            if k in first:
                snippet = str(first[k])[:160].replace("\n", " ")
                print(f"  first.text: {snippet}...")
                break
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover Phase 1 controls files.")
    parser.add_argument("--search-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    if not args.search_root.exists():
        print(f"Search root does not exist: {args.search_root}")
        return 1

    hits = []
    for path in args.search_root.rglob("*"):
        if path.is_file() and _plausible_control_file(path):
            hits.append(path)

    if not hits:
        print(f"No plausible controls files found under {args.search_root}.")
        print(
            "Try passing --search-root data or specifying the path directly to `prert-cnm-build --controls-path`."
        )
        return 2

    print(f"Found {len(hits)} candidate control file(s):\n")
    for path in sorted(hits, key=lambda p: (p.stat().st_size, str(p)), reverse=True):
        _describe(path)
    print("Suggested next step:")
    top = sorted(hits, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(
        f"  python -m prert.phase3.cnm.cli build "
        f"--controls-path {top} --output-dir artifacts/cnm-index"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
