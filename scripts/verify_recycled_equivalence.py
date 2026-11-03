#!/usr/bin/env python3
"""Check the substituted `_update_flag` against the reference body shot for shot.

Status: audit-only

The pass-body sketch's falsifier #2 asks for a 64-shot equivalence run, and the bundle's own
`verify()` can give one: it drives `km.Simulator` over the circuit, writes syndromes and MBUC
masks into the declared registers, and reports value errors (a syndrome or ancilla that did not
come back) and phase errors (a wrong kicked-back phase).

The two arms are built on the same small instance and verified against *the same* 64 shots,
generated once and passed to both. A body that computes a different predicate fails on the
first shot where the predicates disagree; if both arms come back clean on the same shots, the
substitution is doing the same job on those inputs, which is the strongest statement 64 shots
can make. Equivalence is not assumed from the arithmetic identity alone.

Writes `benchmarks/quantum_compilation/dqi_recycled_equivalence.json`.

Usage:
    PYTHONPATH=.../google-verifiable-quantum-circuit-compilation/kickmix \
        /dev/shm/qvcc-venv313/bin/python scripts/verify_recycled_equivalence.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
import types

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"
ARTIFACT = REPO_ROOT / "benchmarks/quantum_compilation/dqi_recycled_equivalence.json"

#: The scored instance. A 15-position / 6-syndrome toy was tried first and dropped: the
#: *reference* decoder fails it (15 value and 3 phase errors in 64 shots), so it cannot
#: adjudicate a substitution that is supposed to reproduce the reference.
N, DEGREE, SHOTS = 70, 12, 64
SEED = 20261103
#: `verify()` certifies only if its Clopper-Pearson bound on the error rate is under 1/16. With
#: 64 shots and no errors that needs log2(delta) >= about -2: at 64 samples no tighter bound is
#: available, whatever the circuit does. The certification is therefore not what this run is for
#: -- the two arms' identical counts are -- and the record says so.
LOG2_DELTA = -2.0


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    try:
        import kickmix as km
    except ImportError as exc:
        print(f"kickmix is not importable: {exc}", file=sys.stderr)
        return 2

    # The stubs carry `__path__` only; the vendored reference decoder is registered under the
    # package name the module loads it by, so both scripts see the same object.
    for package in ("apedp_dyn", "apedp_dyn.quantum_compilation"):
        stub = types.ModuleType(package)
        stub.__path__ = [str(REPO_ROOT / package.replace(".", "/"))]
        sys.modules[package] = stub
    DR = _load(
        "apedp_dyn.quantum_compilation.dqi_reference_circuit",
        BUNDLE / "verifiers/dqi/dqi_reference_circuit.py",
    )

    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    from verify_submission_dqi import (
        FieldTables,
        ProblemSpecDQI,
        optimal_dqi_weights,
        sample_shots,
        verify,
    )

    module = _load(
        "apedp_dyn.quantum_compilation.dqi_gluzdov_v6",
        REPO_ROOT / "apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py",
    )

    spec = ProblemSpecDQI.benchmark_instance(num_shots=SHOTS)
    field = km.GF2Field(DEGREE)
    tables = FieldTables(field)
    weight_probs = optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    shared = sample_shots(spec, tables, np.random.default_rng(SEED), SHOTS, weight_probs)
    n_iters = DR.default_n_iters(N)

    def build(decoder_cls) -> object:
        builder = km.CircuitBuilder()
        syndromes = [
            builder.create_quantum_register(DEGREE, name=f"S{i}") for i in range(N)
        ]
        error_mbuc = [
            builder.create_classical_register(DEGREE, name=f"E{j}") for j in range(spec.m)
        ]
        decoder_cls(N, field).build_syndrome_decoding_mbuc(
            builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=n_iters
        )
        return builder.finish_circuit()

    arms = {
        "reference": build(DR.ProosAndZalkaDecoder),
        "update_flag_substituted": build(module._recycled_flag_decoder_class()),
    }
    try:
        arms["all_sites_substituted"] = build(module._recycled_degree_decoder_class())
    except Exception as exc:  # noqa: BLE001 - a build failure is a result, not a crash
        arms["all_sites_substituted"] = None
        all_sites_error = f"{type(exc).__name__}: {exc}"
    else:
        all_sites_error = None

    results: dict[str, dict] = {}
    for name, circuit in arms.items():
        if circuit is None:
            results[name] = {"built": False, "error": all_sites_error}
            continue
        try:
            report = verify(
                circuit,
                spec,
                seed=SEED,
                batch_size=SHOTS,
                log2_delta=LOG2_DELTA,
                max_workers=1,
                shots=shared,
            )
        except AssertionError as exc:
            # A decoder that fails the confidence bound raises rather than returning. The
            # counts are in the message, and a failing arm is the result this script exists
            # to record -- swallowing it would hide the answer.
            message = str(exc)
            results[name] = {
                "built": True,
                "status": "FAILED_CONFIDENCE_BOUND",
                "value_errors": int(re.search(r"value_errors=(\d+)/", message).group(1)),
                "phase_errors": int(re.search(r"phase_errors=(\d+)/", message).group(1)),
                "verified_shots": SHOTS,
                "detail": message,
            }
            continue
        results[name] = {
            "built": True,
            "status": report["status"],
            "value_errors": report["value_errors"],
            "phase_errors": report["phase_errors"],
            "verified_shots": report["verified_shots"],
            "effective_error_rate": report["effective_error_rate"],
            "score": report["score"],
        }

    reference = results["reference"]
    clean = reference["value_errors"] == 0 and reference["phase_errors"] == 0
    equivalent = {
        name: (
            clean
            and results[name].get("built") is True
            and results[name]["value_errors"] == reference["value_errors"]
            and results[name]["phase_errors"] == reference["phase_errors"]
        )
        for name in ("update_flag_substituted", "all_sites_substituted")
    }

    record = {
        "what": (
            "64-shot equivalence of the substituted pass bodies against the reference decoder, "
            "both arms driven over the same shots"
        ),
        "script": "scripts/verify_recycled_equivalence.py",
        "instance": {
            "m": spec.m,
            "n": spec.n,
            "degree": spec.degree,
            "r": spec.r,
            "n_iters": n_iters,
        },
        "shots": SHOTS,
        "seed": SEED,
        "log2_delta": LOG2_DELTA,
        "shared_shots": True,
        "certification_note": (
            "64 shots cannot certify a sub-1/16 error rate at tight confidence, so log2_delta "
            "is the loosest-with-PASS value and the claim rests on the arms sharing counts"
        ),
        "arms": results,
        "equivalent_to_reference": equivalent,
        "verdict": (
            "both substituted bodies return every syndrome and ancilla and kick back the right "
            "phase on all 64 shots, so the derived reads compute what the register did"
            if all(equivalent.values())
            else "at least one substituted body disagrees with the reference on these shots"
        ),
        "dropped_instance": (
            "m=15, n=6, degree=4: the reference decoder itself fails it, so it cannot show a "
            "substitution equivalent to the reference"
        ),
    }
    ARTIFACT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0 if all(equivalent.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
