#!/usr/bin/env python3
"""Measure the recycled DQI builds at the window that is actually scored.

Status: audit-only

Two builds are compared against the reference at two windows: the one-body prototype
(`_update_flag` only) and the all-sites decoder, where `n_a` is not allocated at all.

The window matters. At `n_iters = 4` the halting counter is narrow, so the reference peaks at
1,888; the scored circuit runs `n_iters = 222`, where it peaks at 1,893. A Toffoli delta quoted
at 4 iterations is therefore not the delta the score sees -- it is about a third of it -- and the
earlier record only had the short window.

Writes `benchmarks/quantum_compilation/dqi_recycled_degrees_window.json`.

Usage:
    PYTHONPATH=.../google-verifiable-quantum-circuit-compilation/kickmix \
        /dev/shm/qvcc-venv313/bin/python scripts/measure_recycled_degrees.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "benchmarks/quantum_compilation/dqi_recycled_degrees_window.json"
REFERENCE = (
    REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"
    / "verifiers/dqi/dqi_reference_circuit.py"
)
WINDOWS = (4, 222)


def _load(name: str, path: pathlib.Path):
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

    windows = []
    for iters in WINDOWS:
        one_body = module.build_candidate_with_recycled_flag_body(n_iters=iters)
        all_sites = module.build_candidate_with_recycled_degrees(n_iters=iters)
        windows.append(
            {
                "n_iters": iters,
                "reference_peak_qubits": one_body["reference_peak_qubits"],
                "reference_toffolis": one_body["reference_toffolis"],
                "one_body": {
                    "peak_qubits": one_body["recycled_flag_peak_qubits"],
                    "toffolis": one_body["recycled_flag_toffolis"],
                    "toffoli_delta": one_body["toffoli_delta"],
                },
                "all_sites": {
                    "peak_qubits": all_sites["recycled_degrees_peak_qubits"],
                    "toffolis": all_sites["recycled_degrees_toffolis"],
                    "toffoli_delta": all_sites["toffoli_delta"],
                    "matches_accounting": all_sites["matches_accounting"],
                },
            }
        )

    full = next(w for w in windows if w["n_iters"] == 222)
    record = {
        "what": (
            "peak and Toffolis for the reference build, the one-body recycled build and the "
            "all-sites recycled build, at a short window and at the scored 222-iteration window"
        ),
        "script": "scripts/measure_recycled_degrees.py",
        "windows": windows,
        "accounting_peak_qubits": 1885,
        "verdict": (
            f"at the scored window the all-sites build peaks at "
            f"{full['all_sites']['peak_qubits']} against the reference's "
            f"{full['reference_peak_qubits']} and costs "
            f"{full['all_sites']['toffoli_delta']:,} more Toffolis; "
            f"the one-body build costs {full['one_body']['toffoli_delta']:,} and buys nothing"
        ),
    }
    ARTIFACT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
