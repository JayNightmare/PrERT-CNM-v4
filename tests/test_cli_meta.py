from __future__ import annotations

import sys
import types
from pathlib import Path

from prert.cli import main as cli_main
from prert.cli import app350 as cli_app350
from prert.cli import phase3 as cli_phase3
from prert.cli import phase3_freeze as cli_phase3_freeze
from prert.phase3.classifier import DEFAULT_PRIVACYBERT_MODEL_NAME


def test_prert_guide_phase2_outputs_expected_commands(capsys) -> None:
    code = cli_main.run(["guide", "--goal", "phase2"])
    assert code == 0

    output = capsys.readouterr().out
    assert "PrERT Guided Commands" in output
    assert "prert phase2" in output


def test_prert_dispatch_forwards_arguments(monkeypatch) -> None:
    captured_argv: list[str] = []

    def _stub_main() -> None:
        captured_argv.extend(cli_main.sys.argv)

    stub_module = types.SimpleNamespace(main=_stub_main)
    monkeypatch.setattr(cli_main.importlib, "import_module", lambda _name: stub_module)

    code = cli_main.run(["extract", "--chunk", "--output-dir", "artifacts/phase-1"])
    assert code == 0
    assert captured_argv[0] == "prert extract"
    assert captured_argv[1:] == ["--chunk", "--output-dir", "artifacts/phase-1"]


def test_prert_run_loads_dotenv_before_dispatch(monkeypatch) -> None:
    calls: list[Path | None] = []

    monkeypatch.setattr(cli_main, "load_dotenv_if_available", lambda env_path: calls.append(env_path))
    monkeypatch.setattr(cli_main, "_dispatch_to_entrypoint", lambda _command, _args: 0)

    code = cli_main.run(["phase3"])

    assert code == 0
    assert calls == [None]


def test_prert_doctor_passes_when_required_inputs_exist(tmp_path) -> None:
    regulations = tmp_path / "docs" / "Standards" / "Regulations"
    regulations.mkdir(parents=True)

    (regulations / "GDPR-2016_679.docx").write_text("stub", encoding="utf-8")
    (regulations / "NIST-1.1.docx").write_text("stub", encoding="utf-8")
    (regulations / "ISO_27001_Standard-1.docx").write_text("stub", encoding="utf-8")

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "CHROMA_HOST=api.trychroma.com",
                "CHROMA_API_KEY=token",
                "CHROMA_TENANT=tenant",
                "CHROMA_DATABASE=db",
            ]
        ),
        encoding="utf-8",
    )

    code = cli_main.run(["doctor", "--root", str(tmp_path)])
    assert code == 0


def test_prert_doctor_fails_without_env_file(tmp_path) -> None:
    regulations = tmp_path / "docs" / "Standards" / "Regulations"
    regulations.mkdir(parents=True)

    (regulations / "GDPR-2016_679.docx").write_text("stub", encoding="utf-8")
    (regulations / "NIST-1.1.docx").write_text("stub", encoding="utf-8")
    (regulations / "ISO_27001_Standard-1.docx").write_text("stub", encoding="utf-8")

    code = cli_main.run(["doctor", "--root", str(tmp_path)])
    assert code == 1


def test_prert_interactive_lists_phase1_options(capsys) -> None:
    code = cli_main.run(["interactive", "--goal", "phase1", "--select", "1"])
    assert code == 0

    output = capsys.readouterr().out
    assert "PrERT Interactive Runner" in output
    assert "Phase 1: Extract controls and chunks" in output
    assert "Use --execute to run selected commands directly." in output


def test_prert_interactive_execute_dispatches(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _stub_dispatch(command: str, command_args: list[str]) -> int:
        captured["command"] = command
        captured["args"] = list(command_args)
        return 0

    monkeypatch.setattr(cli_main, "_dispatch_to_entrypoint", _stub_dispatch)

    code = cli_main.run(["interactive", "--goal", "phase1", "--select", "1", "--execute"])
    assert code == 0
    assert captured["command"] == "extract"
    assert captured["args"] == ["--chunk", "--output-dir", "artifacts/phase-1"]


def test_prert_interactive_invalid_selection_fails() -> None:
    code = cli_main.run(["interactive", "--goal", "phase1", "--select", "99"])
    assert code == 2


def test_phase3_cli_defaults_to_privbert_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prert-phase3"])

    args = cli_phase3._parse_args()

    assert args.privacybert_model_name == DEFAULT_PRIVACYBERT_MODEL_NAME


def test_phase3_freeze_cli_defaults_to_privbert_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prert-phase3-freeze"])

    args = cli_phase3_freeze._parse_args()

    assert args.privacybert_model_name == DEFAULT_PRIVACYBERT_MODEL_NAME


def test_phase3_cli_accepts_auxiliary_labeled_input_path(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["prert-phase3", "--auxiliary-labeled-input-path", "auxiliary.jsonl"],
    )

    args = cli_phase3._parse_args()

    assert args.auxiliary_labeled_input_path == Path("auxiliary.jsonl")


def test_phase3_main_loads_dotenv_before_pipeline(monkeypatch) -> None:
    calls: list[Path | None] = []

    monkeypatch.setattr(cli_phase3, "load_dotenv_if_available", lambda env_path: calls.append(env_path))
    monkeypatch.setattr(
        cli_phase3,
        "run_phase3_pipeline",
        lambda **_kwargs: {
            "metrics": {
                "validation_macro_f1": 0.0,
                "test_macro_f1": 0.0,
                "validation_accuracy": 0.0,
                "test_accuracy": 0.0,
                "bayesian_primary_score": None,
            },
            "dataset_manifest": {"total_rows": 1},
        },
    )
    monkeypatch.setattr(sys, "argv", ["prert-phase3"])

    cli_phase3.main()

    assert calls == [None]


def test_phase3_freeze_main_loads_dotenv_before_pipeline(monkeypatch) -> None:
    calls: list[Path | None] = []

    monkeypatch.setattr(cli_phase3_freeze, "load_dotenv_if_available", lambda env_path: calls.append(env_path))
    monkeypatch.setattr(cli_phase3_freeze, "run_phase3_pipeline", lambda **_kwargs: {"phase": "phase-3"})
    monkeypatch.setattr(
        cli_phase3_freeze,
        "evaluate_phase3_acceptance",
        lambda **_kwargs: {"acceptance": {"passed": True}},
    )
    monkeypatch.setattr(
        cli_phase3_freeze,
        "write_phase3_acceptance_report",
        lambda _output_dir, _report: {"json": "report.json", "markdown": "report.md"},
    )
    monkeypatch.setattr(sys, "argv", ["prert-phase3-freeze"])

    cli_phase3_freeze.main()

    assert calls == [None]


def test_app350_cli_defaults_to_workspace_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prert-app350"])

    args = cli_app350._parse_args()

    assert args.input_path == tmp_path / "data/raw/APP-350_v1.1.zip"
    assert args.output_jsonl == tmp_path / "data/processed/app350_phase3_auxiliary.jsonl"
    assert args.output_manifest == tmp_path / "data/processed/app350_phase3_auxiliary_manifest.json"
    assert args.include_synthetic is False
