#!/usr/bin/env python3
"""Experiment F: what the competition's verifier scores the MR-gated arm.

Experiments D, D2 and E (`docs/DQI_OFFICIAL_COMPATIBILITY.md`) settled what the
*pinned* competition binary can compile and what the verifier assigns it: a correct
5.0946% improvement, 2.319438e13 at 102,400 shots. They also left the other half of
the frontier as an arithmetic claim: the six-lever arm executes 1,221,466.8 Toffolis
per shot at `Q` = 1,913, so its score *would be* `R * Q^2 * T` -- a product, not a
measurement, and on a build the pinned one cannot reproduce.

This experiment turns that product into a verifier number. It builds the arm on the
local `MR`-capable rebuild (`scripts/build_kickmix_mr.py`, the vendored sources plus
`scripts/kickmix_mr_opcode.patch`), runs the competition's own `verify()` on it at the
competition's own budget -- 102,400 shots, `log2_delta = -128` -- and records the
status, both error counts, `eps_plus`, `R`, `Q`, `T` and the score, beside the pinned
reference row from `dqi_submission_score.json` measured at the same shot count.

Three things it settles, and one it explicitly does not:

1. **The arm is correct at the verifier's own standard.** Value *and* phase errors are
   counted on the sampled shots, with the arm's classical-condition reads in the
   stream, so this is the first correctness statement about the arm that is not a
   hand-applied rule.
2. **Its score, measured.** The arm's `R(eps_plus)` is measured from its own failures
   rather than assumed, and `Q` and `T` come from the verifier's own walk of the
   circuit, so the ratio to the pinned reference needs no arithmetic of ours.
3. **How far the frontier is**, in the competition's unit: the ratio to
   `R * 1,789^2 * 445,229 = 1.6975e12`, the derived Benchmark II target.

What it does not do: **it is not a receipt.** `apedp_dyn/quantum_compilation/contracts.py`
allows only the pinned extension to produce one, and the pinned extension has no `MR`.
The record says so in `status`, in `kickmix_role` and in `scope`, and no submission is
claimed.

Examples::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_arm_submission_score.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_arm_submission_score.py \\
        --shots 1536 --record /tmp/arm_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                  / "dqi_arm_submission_score.json")
REFERENCE_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                    / "dqi_submission_score.json")

#: The competition's own defaults (`verify_submission_dqi`).
OFFICIAL_NUM_SHOTS = 400 * 256
OFFICIAL_BATCH_SIZE = 256
OFFICIAL_LOG2_DELTA = -128.0

#: The arm's recorded row, from `gf54_native_opcode.json` and re-measured as the
#: schedule campaign's control: six MR-gated levers, 54-AND product policy.
ARM_T_MEAN = 1_221_466.8
ARM_QUBITS = 1913
#: The executed mean is shot-dependent (`T` is a mean over sampled syndromes), so the
#: row is reproduced within a tolerance rather than exactly.
T_TOLERANCE = 1.5e-3

#: Benchmark II's derived target: `R(eps_plus) * 1,789^2 * 445,229`.
TARGET_SCORE = 1.6975e12
TARGET_QUBITS = 1789
TARGET_T_MEAN = 445_229


def load_submission_module():
    """Experiment E's harness: the verifier loader, the kept fields, `R`/`eps_plus`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dqi_submission_score", ROOT / "scripts" / "measure_dqi_submission_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arm(km, E, m: int, n: int, degree: int, arm: str) -> dict:
    """The requested arm, built with the real builder and the 54-AND policy."""
    sys.path.insert(0, str(ROOT / "scripts"))
    runner = E.load_module("mr_gated_arms_kickmix",
                           ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    ledger = E.load_module("region_ledger",
                           ROOT / "scripts" / "measure_dqi_region_ledger.py")
    arms = ledger.load_arms(km, runner)

    # `and54`: the arm's recorded figures are priced with the product policy as a
    # builder attribute, which the local rebuild has and the pinned build does not.
    shim = runner.shim_for(km, ledger.native_builder(runner.builder_class(km), "and54"))
    original = arms.DR.km
    arms.DR.km = shim
    try:
        started = time.time()
        circuit, _, _ = runner.build(km, arms, shim, arm, "mr", n, degree, m)
        return {"circuit": circuit, "build_seconds": round(time.time() - started, 2)}
    finally:
        arms.DR.km = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--arm", default="full",
                        help="which gates to enable: 'full' is all six levers")
    parser.add_argument("--shots", type=int, default=OFFICIAL_NUM_SHOTS,
                        help="shots to verify; a multiple of --batch-size "
                             "(default: the competition's 102,400)")
    parser.add_argument("--batch-size", type=int, default=OFFICIAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD,
                        help="lib directory of the MR-capable extension")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    if args.shots % args.batch_size:
        raise SystemExit(f"--shots must be a multiple of --batch-size={args.batch_size}")
    if not list(args.kickmix_build.glob("kickmix*.so")):
        raise SystemExit(
            f"no extension in {args.kickmix_build}: build it with "
            f"`{sys.executable} scripts/build_kickmix_mr.py`")

    sys.path.insert(0, str(args.kickmix_build))
    import kickmix as km  # noqa: PLC0415 - the build under test

    if not hasattr(km.CircuitBuilder, "mr"):
        raise SystemExit(f"{km.__file__} has no `mr`: this must be the MR rebuild")
    if not hasattr(km.CircuitBuilder, "gf_product"):
        raise SystemExit(f"{km.__file__} has no `gf_product`: the arm's row is priced "
                         f"with the 54-AND policy")

    E = load_submission_module()
    with tempfile.TemporaryDirectory() as verifier_dir:
        V, verifier_patch = E.load_verifier(Path(verifier_dir))

        print(f"[arm {args.arm}] building ...", flush=True)
        built = build_arm(km, E, args.m, args.n, args.degree, args.arm)
        circuit = built["circuit"]
        print(f"[arm {args.arm}] Q={circuit.num_qubits} ops={len(circuit):,} "
              f"({built['build_seconds']}s)", flush=True)

        spec = V.ProblemSpecDQI.benchmark_instance(
            m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
        print(f"[arm {args.arm}] verifying {args.shots} shots on {args.workers} "
              f"workers ...", flush=True)
        started = time.time()
        result = V.verify(circuit, spec, batch_size=args.batch_size,
                          log2_delta=OFFICIAL_LOG2_DELTA, max_workers=args.workers)
        seconds = round(time.time() - started, 1)

    verified = {name: result[name] for name in E.KEPT_FIELDS}
    print(f"[arm {args.arm}] {result['status']}: score={result['score']:.6e} "
          f"Q={result['num_qubits']} T={result['toffoli_count']:.6f} "
          f"value_err={result['value_errors']} phase_err={result['phase_errors']} "
          f"R={result['repetition_factor']:.6f} ({seconds}s)", flush=True)

    reference = json.loads(REFERENCE_RECORD.read_text())["verified"]["reference"]
    # `R` depends on the shot count (`eps_plus = 1 - 2^(-128/n)` at zero errors), so a
    # ratio between runs at different budgets compares two different penalties.
    like_for_like = args.shots == reference["verified_shots"]
    ratio_reference = verified["score"] / reference["score"] if like_for_like else None
    ratio_target = verified["score"] / TARGET_SCORE
    t_gap = verified["toffoli_count"] / TARGET_T_MEAN
    q_ratio_squared = (verified["num_qubits"] / TARGET_QUBITS) ** 2
    t_needed = TARGET_T_MEAN * q_ratio_squared
    zero_error = E.score_budget(args.shots, 0, V=V)

    checks = [
        {
            "check": "the arm passes the competition's verifier",
            "requirement": "status PASS, zero value and zero phase errors, "
                           "eps_plus < 1/16",
            "measured": {"status": verified["status"],
                         "value_errors": verified["value_errors"],
                         "phase_errors": verified["phase_errors"],
                         "effective_error_rate": verified["effective_error_rate"]},
            "passed": (verified["status"] == "PASS"
                       and verified["value_errors"] == 0
                       and verified["phase_errors"] == 0
                       and verified["effective_error_rate"] < 1 / 16),
        },
        {
            "check": "the arm reproduces the row the opcode and campaign records carry",
            "requirement": f"T = {ARM_T_MEAN:,.1f} per shot (within {T_TOLERANCE:g} "
                           f"relative) at Q = {ARM_QUBITS}",
            "measured": {"toffoli_count": verified["toffoli_count"],
                         "num_qubits": verified["num_qubits"]},
            "passed": (abs(verified["toffoli_count"] / ARM_T_MEAN - 1) < T_TOLERANCE
                       and verified["num_qubits"] == ARM_QUBITS),
        },
        {
            "check": "the score is R * Q^2 * T from the verifier's own numbers",
            "requirement": "the reported score equals R(eps_plus) * Q^2 * T to 1e-12",
            "measured": {"score": verified["score"],
                         "repetition_factor": verified["repetition_factor"],
                         "num_qubits": verified["num_qubits"],
                         "toffoli_count": verified["toffoli_count"]},
            "passed": abs(verified["score"] - verified["repetition_factor"]
                          * verified["num_qubits"] ** 2
                          * verified["toffoli_count"]) / verified["score"] < 1e-12,
        },
        {
            "check": "the arm is still above the derived Benchmark II target",
            "requirement": "the record states the measured gap; this check fails only "
                           "if the arm is *inside* the target, which would make the "
                           "note's verdict wrong",
            "measured": {"ratio_to_target": ratio_target,
                         "toffoli_ratio_to_target": t_gap,
                         "qubit_ratio_squared_to_target": q_ratio_squared,
                         "toffolis_needed_at_this_Q": t_needed},
            "passed": ratio_target > 1.0,
        },
    ]

    payload = {
        "experiment": "F",
        "question": "what does the competition's own verifier score the MR-gated "
                    "six-lever arm, measured rather than multiplied out?",
        "declared_in": "docs/DQI_OFFICIAL_COMPATIBILITY.md",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "batch_size": args.batch_size,
                     "log2_delta": OFFICIAL_LOG2_DELTA, "arm": args.arm,
                     "clear_with": "mr", "product_policy": "and54"},
        "status": "measured on the local MR rebuild; NOT a receipt",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(
            Path(km.__file__).resolve().read_bytes()).hexdigest(),
        "kickmix_role": ("the local rebuild from `scripts/build_kickmix_mr.py` "
                         "(vendored sources + `scripts/kickmix_mr_opcode.patch`). It "
                         "has `mr` and `gf_product`; the pinned competition binary has "
                         "neither, so only this build can execute the arm and only the "
                         "pinned build may produce a receipt."),
        "build_seconds": built["build_seconds"],
        "verify_seconds": seconds,
        "verifier_patch": verifier_patch,
        "verified": verified,
        "budget": {"verified_shots": args.shots,
                   "official_num_shots": OFFICIAL_NUM_SHOTS,
                   "fraction_of_official": args.shots / OFFICIAL_NUM_SHOTS,
                   "zero_error_bound_at_this_budget": zero_error},
        "comparisons": {
            "pinned_reference": {
                "score": reference["score"],
                "toffoli_count": reference["toffoli_count"],
                "num_qubits": reference["num_qubits"],
                "verified_shots": reference["verified_shots"],
                "record": str(REFERENCE_RECORD.relative_to(ROOT)),
            },
            "ratio_to_pinned_reference": ratio_reference,
            "ratio_is_like_for_like": like_for_like,
            "target": {"score": TARGET_SCORE, "qubits": TARGET_QUBITS,
                       "toffoli_mean": TARGET_T_MEAN},
            "ratio_to_target": ratio_target,
            "toffoli_ratio_to_target": t_gap,
            "qubit_ratio_squared_to_target": q_ratio_squared,
            "toffolis_needed_at_this_Q": t_needed,
        },
        "checks": checks,
        "failed_checks": [check["check"] for check in checks if not check["passed"]],
        "finding": (
            f"on the local MR rebuild, the competition's `verify()` passes the "
            f"six-lever arm at {args.shots:,} shots with {verified['value_errors']} "
            f"value and {verified['phase_errors']} phase errors and scores "
            f"{verified['score']:.6e} (Q={verified['num_qubits']}, "
            f"T={verified['toffoli_count']:.6f}, R={verified['repetition_factor']:.6f}). "
            + (f"That is {ratio_reference:.4f}x the pinned reference's "
               f"{reference['score']:.6e} at the same shot count, "
               if like_for_like else
               f"Its score is not comparable to the pinned reference's "
               f"{reference['score']:.6e}: those were measured at "
               f"{reference['verified_shots']:,} shots and this run at "
               f"{args.shots:,}, and `R` differs. ")
            + f"and {ratio_target:.3f}x the derived Benchmark II target of "
            f"{TARGET_SCORE:.4e} "
            f"-- so the arm does not reach the target either. The gap is not the "
            f"multiply: at Q={verified['num_qubits']} the target's 445,229 Toffolis "
            f"would have to be {t_needed:,.0f}, and the arm executes "
            f"{verified['toffoli_count']:,.1f}. No circuit this repository can build "
            f"on any build reaches the target; the pinned binary's own best correct "
            f"circuit is the 2.319438e13 of Experiment E."
        ),
        "scope": {
            "measured": f"the competition's verifier on the arm at {args.shots:,} "
                        f"shots (every scoring line unmodified; only `reaction_depth` "
                        f"is stubbed, see verifier_patch): status, value and phase "
                        f"errors, `eps_plus`, `R`, score, Q and T",
            "not_measured": "a receipt, or anything about the pinned binary: the arm "
                            "needs `mr`, which the pinned build does not have, so this "
                            "row cannot be submitted and is not a submission",
        },
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version.split()[0],
            "git_rev": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=False, capture_output=True,
                                      text=True).stdout.strip(),
        },
    }

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"ratio to pinned reference "
          f"{ratio_reference:.4f}" if ratio_reference is not None
          else "ratio to pinned reference: not like-for-like (different budgets)")
    print(f"ratio to target {ratio_target:.3f} "
          f"(T {t_gap:.3f}x, Q^2 {q_ratio_squared:.3f}x)")
    print(f"failed checks: {payload['failed_checks'] or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not payload["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
