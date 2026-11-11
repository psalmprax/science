"""FAROS runtime failures must not become fabricated resource evidence."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from apedp_dyn.quantum_compilation import faros_isq as faros


def test_explicit_interpreter_overrides_environment(monkeypatch):
    monkeypatch.setenv("APEDP_Q_PYTHON", "/missing/python")
    assert faros._faros_python(sys.executable) == sys.executable


def test_invalid_interpreter_is_not_silently_replaced(monkeypatch):
    monkeypatch.setenv("APEDP_Q_PYTHON", "/missing/python")
    with pytest.raises(FileNotFoundError, match="not executable"):
        faros._faros_python(None)


def test_repository_local_interpreter_is_discovered(monkeypatch):
    monkeypatch.delenv("APEDP_Q_PYTHON", raising=False)
    root = Path(faros.__file__).resolve().parents[2]
    local = str(root / "build/qvcc-venv313/bin/python")
    monkeypatch.setattr(faros.shutil, "which", lambda path: path if path == local else None)
    assert faros._faros_python(None) == local


def test_missing_interpreter_explains_setup(monkeypatch):
    monkeypatch.delenv("APEDP_Q_PYTHON", raising=False)
    monkeypatch.setattr(faros.shutil, "which", lambda path: None)
    with pytest.raises(FileNotFoundError, match="Set APEDP_Q_PYTHON"):
        faros._faros_python(None)


@pytest.mark.parametrize("operation", ["build", "measurement", "calibration"])
def test_subprocess_failure_preserves_diagnostic(operation):
    with pytest.raises(RuntimeError, match="missing native dependency"):
        faros._run_faros_script(
            sys.executable,
            "import sys; sys.stderr.write('missing native dependency'); sys.exit(2)",
            operation,
        )


def test_subprocess_success_returns_actual_output():
    assert faros._run_faros_script(sys.executable, "print('measured')", "measurement") == (
        "measured\n"
    )


def test_resources_use_subprocess_measurements(tmp_path, monkeypatch):
    artifact = tmp_path / "candidate.kmb"
    artifact.write_bytes(b"unit-test artifact; not a real circuit")
    measurements = {"num_qubits": 80, "static_toffoli": 1000, "t_like": 2, "rotations": 30}
    monkeypatch.setattr(
        faros.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(measurements)
        ),
    )
    metadata = faros.materialize_faros_circuit(
        output_path=artifact, python_executable=sys.executable
    )
    assert all(metadata[key] == value for key, value in measurements.items())
    saved = json.loads(artifact.with_suffix(".kmb.resources.json").read_text())
    assert all(saved[key] == value for key, value in measurements.items())


@pytest.mark.parametrize("response", ["failure", "malformed", "missing-fields"])
def test_resource_failure_does_not_write_success_metadata(tmp_path, monkeypatch, response):
    artifact = tmp_path / "candidate.kmb"
    artifact.write_bytes(b"unit-test artifact; not a real circuit")

    def run(*args, **kwargs):
        if response == "failure":
            raise subprocess.CalledProcessError(1, args[0], stderr="cannot load kickmix")
        return subprocess.CompletedProcess(
            args[0], 0, stdout="not json" if response == "malformed" else "{}"
        )

    monkeypatch.setattr(faros.subprocess, "run", run)
    error = {
        "failure": RuntimeError,
        "malformed": json.JSONDecodeError,
        "missing-fields": KeyError,
    }[response]
    with pytest.raises(error):
        faros.materialize_faros_circuit(
            output_path=artifact, python_executable=sys.executable
        )
    assert not artifact.with_suffix(".kmb.resources.json").exists()
