"""Pipeline reports must reflect actual execution and the final artifact."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apedp_dyn.quantum_compilation import faros_isq as faros

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    module = load_script("build_faros_candidate")
    monkeypatch.setattr(module, "load_chebyshev_minimax_tables", lambda: (None, {}, 0.0))
    monkeypatch.setattr(module, "exhaustive_integer_certificate", lambda *a, **k: {
        "pass": True, "tested_inputs": 3_133_452,
    })
    calls = []

    def materialize(**kwargs):
        calls.append(kwargs)
        return {"sha256": "final-digest", "cost_evaluation": {"cost": 999999}}

    monkeypatch.setattr(module, "materialize_faros_circuit", materialize)
    monkeypatch.setattr(module, "calibrate_faros_circuit", lambda *a, **k: {
        "lp_converged": None, "margin_verified": False,
    })
    return module, calls, tmp_path


def test_build_without_verify_never_claims_verified(pipeline, monkeypatch, capsys):
    module, calls, directory = pipeline
    monkeypatch.setattr(module, "verify_faros_candidate", lambda *a, **k: pytest.fail(
        "Verifier must not execute without --verify"
    ))
    assert module.main(["--output-dir", str(directory), "--candidate-id", "test"]) == 0
    summary = json.loads((directory / "test.pipeline.json").read_text())
    assert summary["status"] == "built_not_verified"
    assert summary["verification"] is None
    assert calls[0]["force_rebuild"] is True
    assert len(calls) == 2  # Final artifact is remeasured after calibration.
    assert "Expected Mean Toffoli" not in capsys.readouterr().out


@pytest.mark.parametrize("status,exit_code,expected", [("PASS", 0, 0), ("FAIL", 0, 1),
                                                     ("PASS", 1, 1)])
def test_verify_executes_and_propagates_outcome(pipeline, monkeypatch, status, exit_code, expected):
    module, _, directory = pipeline
    seen = []

    def verify(path, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            status=status, exit_code=exit_code, artifact_sha256="final-digest",
            to_dict=lambda: {"status": status, "verification_scope": "development-prefix"},
        )

    monkeypatch.setattr(module, "verify_faros_candidate", verify)
    assert module.main(["--output-dir", str(directory), "--candidate-id", "test",
                        "--verify", "--max-x", "4096"]) == expected
    summary = json.loads((directory / "test.pipeline.json").read_text())
    assert summary["status"] == ("verifier_passed" if expected == 0 else "verifier_failed")
    assert summary["verification"]["verification_scope"] == "development-prefix"
    assert seen[0]["max_x"] == 4096


def test_mismatched_receipt_is_rejected(pipeline, monkeypatch):
    module, _, directory = pipeline
    monkeypatch.setattr(module, "verify_faros_candidate", lambda *a, **k: SimpleNamespace(
        artifact_sha256="other-circuit"
    ))
    with pytest.raises(RuntimeError, match="not bound"):
        module.main(["--output-dir", str(directory), "--verify"])
    report = next(directory.glob("*.pipeline.json"))
    assert json.loads(report.read_text())["status"] == "in_progress"


def test_failed_model_stops_before_materialization(pipeline, monkeypatch):
    module, calls, directory = pipeline
    monkeypatch.setattr(module, "exhaustive_integer_certificate", lambda *a, **k: {
        "pass": False, "tested_inputs": 3_133_452,
    })
    with pytest.raises(RuntimeError, match="model certificate failed"):
        module.main(["--output-dir", str(directory)])
    assert not calls


def test_calibration_replays_supplied_data_without_claiming_lp(tmp_path, monkeypatch):
    artifact = tmp_path / "circuit.kmb"
    artifact.write_bytes(b"before calibration")
    original_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    supplied = {"marginUnits": 12345, "droppedOriginalRotationIndices": [2],
                "deltasPhaseUnits": [11, 22, 33]}
    scripts = []

    def run(executable, script, operation):
        scripts.append(script)
        artifact.write_bytes(b"after calibration")
        return json.dumps({"rotations": 30, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})

    monkeypatch.setattr(faros, "_faros_python", lambda value: "python")
    monkeypatch.setattr(faros, "_run_faros_script", run)
    report = faros.calibrate_faros_circuit(artifact, fixed_deltas=supplied)
    assert report["source_circuit_sha256"] == original_digest
    assert report["source_circuit_sha256"] != report["calibrated_circuit_sha256"]
    assert report["lp_converged"] is None
    assert report["margin_verified"] is False
    assert report["margin_units"] == 12345
    assert json.dumps(supplied) in scripts[0]
    assert "src/invsqrt_calibration.json" not in scripts[0]


def test_calibration_rejects_missing_evidence(tmp_path, monkeypatch):
    artifact = tmp_path / "circuit.kmb"
    artifact.write_bytes(b"not a real circuit")
    monkeypatch.setattr(faros, "_faros_python", lambda value: "python")
    with pytest.raises(KeyError):
        faros.calibrate_faros_circuit(artifact, fixed_deltas={})


def test_preflight_refuses_missing_and_unapproved_extension(tmp_path, monkeypatch):
    module = load_script("check_verifier_environment")
    assert not module.check_environment("missing-python", tmp_path / "missing.so")["ready"]
    extension = tmp_path / "kickmix.so"
    extension.write_bytes(b"unapproved")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail(
        "Unapproved binary must not be executed"
    ))
    report = module.check_environment("missing-python", extension)
    assert report["ready"] is False
    assert report["extension_sha256"] == hashlib.sha256(b"unapproved").hexdigest()


@pytest.mark.parametrize("spec", [None, SimpleNamespace(loader=None)])
def test_preflight_fails_explicitly_if_pin_module_cannot_be_loaded(monkeypatch, spec):
    module = load_script("check_verifier_environment")
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *a, **k: spec)
    with pytest.raises(ImportError, match="Cannot load KickMix build pins"):
        module._load_pins()


@pytest.mark.parametrize("runtime_kind", ["ready", "wrong-python", "wrong-extension", "missing-api"])
def test_preflight_requires_correct_runtime_and_import(tmp_path, monkeypatch, runtime_kind):
    module = load_script("check_verifier_environment")
    extension = tmp_path / "kickmix.so"
    extension.write_bytes(b"test pinned bytes")
    other = tmp_path / "other.so"
    other.write_bytes(b"different binary")
    monkeypatch.setattr(module, "_load_pins", lambda: SimpleNamespace(
        identify=lambda path: SimpleNamespace(name="test-pin", allowed_for_verifier=True)
    ))
    runtime = {
        "extension": str(other if runtime_kind == "wrong-extension" else extension),
        "python": "3.12.3" if runtime_kind == "wrong-python" else "3.13.14",
        "has_circuit_builder": runtime_kind != "missing-api",
    }
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps(runtime)
    ))
    report = module.check_environment("python", extension)
    assert bool(report["ready"]) == (runtime_kind == "ready")
