#!/usr/bin/env python3
"""Reconcile the official DQI reference row with the locally measured one.

Three numbers are in circulation for the *same* decoder, and they must not be
compared as if they were the same measurement:

* ``Q = 1,893`` / ``T_mean = 5,725,108.382109375`` — the locked official
  reference row, a constant in ``apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py``
  and in the photonic-boundary documents.
* ``Q = 1,893`` / ``T_mean = 5,725,071.609375`` — this repository's own
  measurement of the same reference circuit, on the locally rebuilt extension,
  averaged over the 64-off sample this project's measurement scripts use.
* ``Q = 1,912`` / ``T_mean = 5,201,222.53125`` — the same reference decoder with
  the native 54-AND product policy and the reduced workspace.

This script does the three things the discrepancy needs:

1. **Sample size, from the numbers themselves.** A mean over ``k`` shots has a
   fractional part that is a multiple of ``1/k``. Both figures are tested for
   the shot count they can have come from; the official one is a mean over the
   verifier's default 102,400 shots, this project's over 64.
2. **Build neutrality.** The reference circuit is built and simulated on the
   pinned competition binary and on the locally rebuilt one, over the same 64
   sampled syndromes, and the per-shot Toffoli arrays are compared. If they are
   identical, the ``MR`` patch does not change the count for a circuit that
   does not use ``MR``, so the local number is the official metric on another
   sample rather than another metric.
3. **Receipt eligibility.** The six-lever arm is built on the pinned binary and
   the failure is recorded verbatim: the arm's measurement-based clears need an
   opcode that binary does not have, so it cannot be compiled into a submission
   and the official verifier cannot score it.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/reconcile_dqi_resource_baseline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
PINNED_DIR = BUNDLE / "kickmix"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                  / "dqi_resource_reconciliation.json")

#: The locked official reference row, as recorded in the repository.
OFFICIAL = {"qubits": 1893, "t_mean": 5_725_108.382109375,
            "source": "apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py:T_MEAN_REFERENCE"}
#: The verifier's default shot count; the official row is a mean over this many.
VERIFIER_SHOTS = 400 * 256

SIX_LEVERS = {"gate_forney": True, "clear_eea": True, "gate_sweep": True,
              "gate_swap": True, "gate_shift": True, "gate_quotient": True}


def mean_over(mean: float, shots: int) -> tuple[bool, int]:
    """Whether `mean` can be an average over exactly `shots` counts, and that sum.

    A wrong shot count misses integrality by O(1) (the fractional part of the sum),
    which is far above float64's spacing at these magnitudes (~1e-4 here), so a 1e-3
    tolerance tests the hypothesis rather than the floating point.
    """
    scaled = mean * shots
    integral = abs(scaled - round(scaled)) < 1e-3
    return integral, int(round(scaled))


def load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mr_gated_arms_kickmix", ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arms(km, runner):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mr_gated_arms", ROOT / "scripts" / "run_mr_gated_arms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._all_qubits = lambda registers: runner.all_qubits(km, registers)
    return module


def sample_syndromes(km, m: int, n: int, degree: int, shots: int, seed: int):
    """The measurement scripts' own shot sample, so the build comparison is exact."""
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree, num_shots=shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(seed)
    return V.sample_shots(spec, tables, rng, shots, weights)


def measure(build_dir: Path, m: int, n: int, degree: int, shots: int, seed: int) -> dict:
    """Build and run the reference on one extension, and try the six-lever arm."""
    sys.path.insert(0, str(build_dir))
    import kickmix as km

    runner = load_runner()
    arms = load_arms(km, runner)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "measure_gf54_native_opcode", ROOT / "scripts" / "measure_gf54_native_opcode.py")
    measure_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure_module)

    builder = runner.builder_class(km)
    shim = runner.shim_for(km, builder)
    arms.DR.km = shim
    syndromes_in, mbuc_in, _phase = sample_syndromes(km, m, n, degree, shots, seed)
    import verify_submission_dqi as V

    reference_inputs = V.to_bits(syndromes_in, degree)

    circuit, syn, mbuc = measure_module.build_reference(km, arms, shim, n, degree, m)
    out = measure_module.simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in,
                                  degree, shots)
    reference_out = out
    result = {
        "build": str(build_dir),
        "extension": str(Path(km.__file__).resolve()),
        "extension_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "has_mr": bool(hasattr(km.CircuitBuilder, "mr")),
        "has_gf_product": bool(hasattr(km.CircuitBuilder, "gf_product")),
        "reference": {
            "qubits": circuit.num_qubits,
            "mean_toffoli_like": float(np.mean(out["toffoli_per_shot"])),
            "toffoli_per_shot": out["toffoli_per_shot"],
            "ccx_per_shot": out["ccx_per_shot"],
            "syndromes_restored": bool(np.array_equal(out["syndromes"], reference_inputs)),
            "junk_clean": bool(not out["junk"].any()),
        },
        "shot_sample_sha256": hashlib.sha256(reference_inputs.tobytes()).hexdigest(),
    }

    # The arm is the one candidate that would need a receipt. Build it on this
    # extension and record exactly what happens. Where the extension has the
    # product policy, use it, so the arm recorded here is the same candidate the
    # reachability note quotes (1,221,466.8 rather than the built-in 1,325,230.0).
    arm_builder = builder
    policy = "karatsuba"
    if hasattr(km.CircuitBuilder, "gf_product"):
        class ArmBuilder(builder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.gf_product = "and54"

        arm_builder = ArmBuilder
        policy = "and54"
    arm_shim = runner.shim_for(km, arm_builder)
    arms.DR.km = arm_shim
    try:
        circuit, syn, mbuc = measure_module.build_arm(km, arms, arm_shim, SIX_LEVERS, n,
                                                      degree, m)
        out = measure_module.simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in,
                                      degree, shots)
        result["arm"] = {
            "built": True,
            "qubits": circuit.num_qubits,
            "mean_toffoli_like": float(np.mean(out["toffoli_per_shot"])),
            "correct": measure_module.is_correct(out, reference_out,
                                                 reference_inputs),
            "product_policy": policy,
        }
    except Exception as exc:  # noqa: BLE001 - the failure mode is the result
        result["arm"] = {
            "built": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "product_policy": policy,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--pinned-build", type=Path, default=PINNED_DIR)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--measure-build", type=Path,
                        help="internal: measure one extension and print its JSON")
    args = parser.parse_args()

    if args.measure_build is not None:
        print(json.dumps(measure(args.measure_build, args.m, args.n, args.degree,
                                 args.shots, args.seed)))
        return 0

    runs = {}
    for label, build in (("pinned", args.pinned_build), ("local", args.kickmix_build)):
        proc = subprocess.run(  # noqa: S603 - this interpreter, fixed argv
            [sys.executable, __file__, "--measure-build", str(build),
             "--m", str(args.m), "--n", str(args.n), "--degree", str(args.degree),
             "--shots", str(args.shots), "--seed", str(args.seed)],
            capture_output=True, text=True, check=True,
        )
        runs[label] = json.loads(proc.stdout)

    pinned = runs["pinned"]
    local = runs["local"]

    official_integral, official_sum = mean_over(OFFICIAL["t_mean"], VERIFIER_SHOTS)
    official_at_64, official_sum_64 = mean_over(OFFICIAL["t_mean"], args.shots)
    local_integral, local_sum = mean_over(local["reference"]["mean_toffoli_like"], args.shots)

    per_shot_pinned = np.array(pinned["reference"]["toffoli_per_shot"])
    per_shot_local = np.array(local["reference"]["toffoli_per_shot"])
    identical = bool(np.array_equal(per_shot_pinned, per_shot_local))
    spread = float(np.std(per_shot_pinned, ddof=1))
    sem = spread / np.sqrt(args.shots)
    delta_local_official = local["reference"]["mean_toffoli_like"] - OFFICIAL["t_mean"]
    delta_pinned_official = pinned["reference"]["mean_toffoli_like"] - OFFICIAL["t_mean"]

    arm_local = local["arm"]
    arm_record = None
    arm_record_path = (ROOT / "benchmarks" / "quantum_compilation"
                       / "gf54_native_opcode.json")
    if arm_record_path.exists():
        recorded = json.loads(arm_record_path.read_text())
        if recorded.get("kickmix_sha256") == local["extension_sha256"]:
            arm_record = recorded["arm"]["and54"]

    record = {
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": args.seed},
        "official_reference_row": {
            **OFFICIAL,
            "mean_over_shots": VERIFIER_SHOTS,
            "is_an_integer_sum_at_that_shot_count": official_integral,
            "sum_over_those_shots": official_sum,
            "is_a_mean_over_this_scripts_shots": official_at_64,
            "sum_at_this_scripts_shot_count": official_sum_64,
            "note": "the fractional part is a multiple of 1/102,400 and not of 1/64, "
                    "so this figure is the verifier's default-shot mean, not a "
                    "64-shot mean",
        },
        "builds": {
            "pinned": {k: v for k, v in pinned.items() if k not in
                       ("reference", "arm")},
            "local": {k: v for k, v in local.items() if k not in
                      ("reference", "arm")},
        },
        "reference": {
            "pinned": {k: v for k, v in pinned["reference"].items()
                       if k != "toffoli_per_shot"},
            "local": {k: v for k, v in local["reference"].items()
                      if k != "toffoli_per_shot"},
            "per_shot_arrays_identical_across_builds": identical,
            "shot_sample_sha256": local["shot_sample_sha256"],
            "same_shot_sample_across_builds": (
                pinned["shot_sample_sha256"] == local["shot_sample_sha256"]),
            "per_shot_stddev": spread,
            "standard_error_of_this_scripts_mean": sem,
            "local_minus_official": delta_local_official,
            "pinned_minus_official": delta_pinned_official,
            "local_delta_in_standard_errors": delta_local_official / sem if sem else None,
        },
        "arm": {
            "pinned_build": pinned["arm"],
            "local_build": arm_local,
            "recorded_local_measurement": arm_record,
        },
        "verdicts": {
            "official_and_local_are_the_same_circuit_and_metric": identical,
            "difference_is_a_shot_sample_not_a_metric": {
                "explained": bool(identical and official_integral
                                  and abs(delta_local_official) < 4 * sem),
                "difference_per_shot": delta_local_official,
                "standard_error": sem,
            },
            "arm_can_be_compiled_on_the_pinned_binary": bool(
                pinned["arm"].get("built")),
            "arm_receipt_possible": False,
            "why_no_receipt": (
                "the arm clears its AND work with MR, and the pinned competition "
                "extension has no MR opcode: the build fails there, and "
                "apedp_dyn/quantum_compilation/contracts.py pins the official verifier "
                "to that extension's hash, so no other build may produce a receipt"),
            "what_would_be_needed": (
                "an upstream release whose instruction set includes measurement-based "
                "reset, or a candidate that proves its gates without it"),
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"instance m={args.m} n={args.n} GF(2^{args.degree}), {args.shots} shots, "
          f"seed {args.seed}")
    print(f"  official   Q={OFFICIAL['qubits']} T={OFFICIAL['t_mean']:.6f} "
          f"(integer sum over {VERIFIER_SHOTS} shots: {official_integral})")
    print(f"  pinned     Q={pinned['reference']['qubits']} "
          f"T={pinned['reference']['mean_toffoli_like']:.6f} "
          f"({delta_pinned_official:+.2f} vs official)")
    print(f"  local      Q={local['reference']['qubits']} "
          f"T={local['reference']['mean_toffoli_like']:.6f} "
          f"({delta_local_official:+.2f} vs official, "
          f"{delta_local_official / sem:.2f} standard errors)")
    print(f"  per-shot arrays identical across builds: {identical}")
    print(f"  arm on pinned: built={pinned['arm'].get('built')} "
          f"{pinned['arm'].get('error_type', '')}")
    print(f"record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
