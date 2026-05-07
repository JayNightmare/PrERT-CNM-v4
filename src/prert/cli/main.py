"""Top-level PrERT command runner for command discovery and execution."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, List


ENTRYPOINTS: Dict[str, str] = {
    "extract": "prert.cli.extract",
    "migrate": "prert.cli.migrate",
    "phase2": "prert.cli.phase2",
    "opp115": "prert.cli.opp115",
    "phase3": "prert.cli.phase3",
    "phase3-freeze": "prert.cli.phase3_freeze",
    "phase4": "prert.cli.phase4",
    "phase4-web": "prert.cli.phase4_web",
    "phase4-synth": "prert.cli.phase4_synthetic",
}

ALIASES: Dict[str, str] = {
    "phase1-extract": "extract",
    "phase1-migrate": "migrate",
}

REQUIRED_DOCX_INPUTS = (
    "docs/Standards/Regulations/GDPR-2016_679.docx",
    "docs/Standards/Regulations/NIST-1.1.docx",
)

REQUIRED_CHROMA_ENV_VARS = (
    "CHROMA_HOST",
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
)

OPTIONAL_ENV_VARS = (
    "CHROMA_COLLECTION_NAME",
    "HF_TOKEN",
)


def main() -> None:
    raise SystemExit(run())


def run(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        _print_help()
        return 0

    command = args[0]
    command_args = args[1:]

    if command in {"-h", "--help", "help"}:
        _print_help()
        return 0

    if command in {"doctor", "guide"}:
        if command == "doctor":
            return _run_doctor(command_args)
        return _run_guide(command_args)

    canonical = ALIASES.get(command, command)
    if canonical not in ENTRYPOINTS:
        print(f"Unknown command: {command}")
        print("Use 'prert help' to list supported commands.")
        return 2

    return _dispatch_to_entrypoint(canonical, command_args)


def _dispatch_to_entrypoint(command: str, command_args: List[str]) -> int:
    module_name = ENTRYPOINTS[command]
    module = importlib.import_module(module_name)
    entrypoint: Callable[[], None] = getattr(module, "main")

    previous_argv = list(sys.argv)
    try:
        sys.argv = [f"prert {command}", *command_args]
        entrypoint()
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0
    finally:
        sys.argv = previous_argv

    return 0


def _run_doctor(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate PrERT setup prerequisites")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to validate. Defaults to current working directory.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    failures = 0

    print(f"PrERT Doctor: {root}")
    failures += _check_python_version()
    failures += _check_docx_inputs(root)
    failures += _check_iso_inputs(root)
    failures += _check_env_file(root)

    if failures:
        print("\nResult: FAIL")
        print("Fix the items above and rerun: prert doctor")
        return 1

    print("\nResult: PASS")
    return 0


def _run_guide(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Show guided PrERT command sequences")
    parser.add_argument(
        "--goal",
        choices=("full", "phase1", "phase2", "phase3", "phase4", "validation"),
        default="full",
        help="Workflow goal to print recommended next commands.",
    )
    args = parser.parse_args(argv)

    print("PrERT Guided Commands")
    print(f"Goal: {args.goal}\n")

    if args.goal in {"full", "phase1"}:
        print("Phase 1")
        print("1. prert extract --chunk --output-dir artifacts/phase-1")
        print("2. prert migrate --input-dir artifacts/phase-1")
        if args.goal == "phase1":
            return 0
        print("")

    if args.goal in {"full", "phase2"}:
        print("Phase 2")
        print("1. prert phase2")
        print("2. Optional public mapping: prert opp115")
        print("3. Optional enrichment: prert phase2 --public-input data/processed/opp115_public_mapping.csv")
        if args.goal == "phase2":
            return 0
        print("")

    if args.goal in {"full", "phase3"}:
        print("Phase 3")
        print("1. prert phase3")
        print("2. Optional acceptance freeze: prert phase3-freeze --strict")
        if args.goal == "phase3":
            return 0
        print("")

    if args.goal in {"full", "phase4", "validation"}:
        print("Phase 4")
        print("1. Validate artifacts: prert phase4 --baseline-dir artifacts/phase-3-freeze")
        print("2. Launch web app: prert phase4-web --port 8501")
        print("3. Optional synthetic fixtures: prert phase4-synth --output-dir artifacts/phase-4/synthetic-compliance")

    return 0


def _check_python_version() -> int:
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) >= (3, 11):
        print(f"[PASS] Python version {major}.{minor}")
        return 0

    print(f"[FAIL] Python version {major}.{minor} (requires >= 3.11)")
    return 1


def _check_docx_inputs(root: Path) -> int:
    failures = 0
    for relative_path in REQUIRED_DOCX_INPUTS:
        candidate = root / relative_path
        if candidate.exists():
            print(f"[PASS] Found {relative_path}")
        else:
            print(f"[FAIL] Missing required input {relative_path}")
            failures += 1
    return failures


def _check_iso_inputs(root: Path) -> int:
    regulations_dir = root / "docs/Standards/Regulations"
    if not regulations_dir.exists():
        print("[FAIL] Missing docs/Standards/Regulations directory")
        return 1

    iso_files = sorted(
        path
        for path in regulations_dir.glob("*.docx")
        if path.name.startswith("ISO") or path.name.startswith("BS EN ISO") or path.name.startswith("BS ISO")
    )
    if iso_files:
        print(f"[PASS] ISO DOCX inputs discovered: {len(iso_files)}")
        return 0

    print("[FAIL] No ISO DOCX inputs found in docs/Standards/Regulations")
    return 1


def _check_env_file(root: Path) -> int:
    env_path = root / ".env"
    if not env_path.exists():
        print("[FAIL] Missing .env file")
        return 1

    values = _load_env_values(env_path)
    failures = 0
    for key in REQUIRED_CHROMA_ENV_VARS:
        if values.get(key, "").strip():
            print(f"[PASS] {key} is set")
        else:
            print(f"[FAIL] {key} is missing or empty")
            failures += 1

    for key in OPTIONAL_ENV_VARS:
        if values.get(key, "").strip():
            print(f"[PASS] {key} is set")
        else:
            print(f"[WARN] {key} is not set (optional)")

    return failures


def _load_env_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        values[key.strip()] = raw.strip().strip('"').strip("'")
    return values


def _print_help() -> None:
    print("PrERT Command Runner")
    print("")
    print("Usage:")
    print("  prert <command> [args]")
    print("")
    print("Core commands:")
    print("  extract        Run Phase 1 extraction")
    print("  migrate        Run Phase 1 Chroma migration")
    print("  phase2         Run Phase 2 metrics pipeline")
    print("  opp115         Preprocess OPP-115 public mapping inputs")
    print("  phase3         Run Phase 3 baseline training")
    print("  phase3-freeze  Run Phase 3 acceptance freeze")
    print("  phase4         Run Phase 4 validation benchmark")
    print("  phase4-web     Launch Phase 4 Streamlit web app")
    print("  phase4-synth   Generate synthetic policy/schema fixtures")
    print("")
    print("Helper commands:")
    print("  doctor         Validate environment, inputs, and credentials")
    print("  guide          Print recommended command order")
    print("")
    print("Examples:")
    print("  prert guide --goal full")
    print("  prert extract --chunk --output-dir artifacts/phase-1")
    print("  prert phase4-web --port 8501")


if __name__ == "__main__":
    main()
