from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_phase3_classifier_does_not_eager_load_pipeline() -> None:
    result = _run_python(
        "import sys; import prert.phase3.classifier; "
        "assert 'prert.phase3.pipeline' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr


def test_importing_phase4_run_surface_does_not_eager_load_compliance_assessor() -> None:
    result = _run_python(
        "import sys; from prert.phase4 import run_phase4_validation; "
        "assert callable(run_phase4_validation); "
        "assert 'prert.phase4.compliance_assessor' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr