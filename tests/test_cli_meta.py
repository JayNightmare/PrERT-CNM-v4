from __future__ import annotations

import types

from prert.cli import main as cli_main


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
