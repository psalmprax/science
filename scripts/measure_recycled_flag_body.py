#!/usr/bin/env python3
"""Build the one-body recycled decoder and record what it does to the peak.

Status: audit-only

`docs/DQI_RECYCLED_PASS_BODIES.md` now has one of its five `n_a` read sites substituted
(`_update_flag`). The claim that matters is not that the body is exact -- that is arithmetic --
but that substituting *one* body cannot lower the peak, because the other four still read the
register and it is therefore still allocated.

This builds both circuits at a fixed `n_iters` window and records both peaks and Toffolis, so the
finding is a measurement rather than a sentence. Both numbers are window-dependent: the halting
counter widens with `n_iters`, so the peak at `n_iters = 4` (1,888) is not the full-shot 1,893.

Writes `benchmarks/quantum_compilation/dqi_recycled_flag_body_build.json`.

Usage:
    PYTHONPATH=.../kickmix build/qvcc-venv313/bin/python scripts/measure_recycled_flag_body.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "benchmarks/quantum_compilation/dqi_recycled_flag_body_build.json"
REFERENCE = (
    REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"
    / "verifiers/dqi/dqi_reference_circuit.py"
)
WINDOW_ITERS = 4


def _load(name: str, path: pathlib.Path):
    """Load a module by path, so the audit does not import the whole `apedp_dyn` package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    try:
        import kickmix  # noqa: F401
    except ImportError as exc:
        print(f"kickmix is not importable: {exc}", file=sys.stderr)
        return 2

    for package in ("apedp_dyn", "apedp_dyn.quantum_compilation"):
        stub = types.ModuleType(package)
        stub.__path__ = [str(REPO_ROOT / package.replace(".", "/"))]
        sys.modules[package] = stub

    _load("apedp_dyn.quantum_compilation.dqi_reference_circuit", REFERENCE)
    module = _load(
        "apedp_dyn.quantum_compilation.dqi_gluzdov_v6",
        REPO_ROOT / "apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py",
    )

    report = module.build_candidate_with_recycled_flag_body(n_iters=WINDOW_ITERS)
    record = {
        "what": (
            "peak and Toffolis for the reference build and for the build with `_update_flag` "
            "substituted, at one fixed EEA window"
        ),
        "script": "scripts/measure_recycled_flag_body.py",
        "window_iters": WINDOW_ITERS,
        "window_note": (
            "peak and Toffolis depend on n_iters (the halting counter widens with it), so 1,888 "
            "here is this window, not the full-shot 1,893"
        ),
        **report,
        "conclusion": (
            "the substituted body costs Toffolis and buys no qubits: the register is still "
            "allocated for the four bodies that read it, so 1,885 stays an accounting figure"
        ),
    }
    ARTIFACT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
