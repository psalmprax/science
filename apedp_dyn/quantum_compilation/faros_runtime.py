"""Interpreter discovery and diagnostic subprocess boundary for FAROS."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def faros_python(python_executable: str | None) -> str:
    executable = python_executable or os.environ.get("APEDP_Q_PYTHON")
    if executable:
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(f"FAROS interpreter not executable: {executable}")
        return resolved
    repo_root = Path(__file__).resolve().parents[2]
    for environment in ("qvcc-pinned313", "qvcc-venv313"):
        candidate = repo_root / "build" / environment / "bin" / "python"
        resolved = shutil.which(str(candidate))
        if resolved is not None:
            return resolved
    raise FileNotFoundError(
        "FAROS needs the KickMix-compatible Python interpreter. "
        "Set APEDP_Q_PYTHON or create build/qvcc-venv313 with Python 3.13."
    )


def run_faros_script(executable: str, script: str, operation: str) -> str:
    try:
        result = subprocess.run(
            [executable, "-c", script], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FAROS {operation} failed (exit {exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    return result.stdout
