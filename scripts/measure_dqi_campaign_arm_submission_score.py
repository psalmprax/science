#!/usr/bin/env python3
"""Experiment G: the campaign arm's score, measured by the competition's verifier.

Experiment F scored the six-lever arm: 5.324728e12 at `Q` = 1,913, 3.137x the derived
Benchmark II target. The best circuit this repository has measured is one rung further
-- the same arm with the five schedule/control gates stacked on it,
**686,014.8** Toffolis per shot at `Q` = 1,913 -- and its score has only ever been the
`Q^2 T` product of those two numbers.

This script builds that circuit exactly as `scripts/measure_dqi_schedule_campaign.py`
builds its `stacked` arm (the campaign's `ScheduleDecoder`, `mode="stacked"`, the two
promoted mixins, the six levers, the 54-AND product policy, cleared with `mr`) and runs
the competition's own `verify()` on it at the competition's own budget: 102,400 shots,
`log2_delta = -128`.

It is **not a receipt**. The arm needs `mr` and `gf_product`, which the pinned
competition binary does not have; `apedp_dyn/quantum_compilation/contracts.py` is
unchanged, and the record says so in `status`, `kickmix_role` and `scope`.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_campaign_arm_submission_score.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                  / "dqi_campaign_arm_submission_score.json")
REFERENCE_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                    / "dqi_submission_score.json")
ARM_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
              / "dqi_arm_submission_score.json")

OFFICIAL_NUM_SHOTS = 400 * 256
OFFICIAL_BATCH_SIZE = 256
OFFICIAL_LOG2_DELTA = -128.0

#: The campaign's own row for this circuit (`dqi_schedule_campaign.json`, `stacked`).
STACKED_T_MEAN = 686_014.8
STACKED_QUBITS = 1913
T_TOLERANCE = 1.5e-3

TARGET_SCORE = 1.6975e12
TARGET_QUBITS = 1789
TARGET_T_MEAN = 445_229


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_stacked(km, E, m: int, n: int, degree: int) -> dict:
    """The campaign's `stacked` arm, built by the campaign's own code path."""
    runner = E.load_module("mr_gated_arms_kickmix",
                           ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    ledger = E.load_module("region_ledger",
                           ROOT / "scripts" / "measure_dqi_region_ledger.py")
    arms = ledger.load_arms(km, runner)
    campaign = load_module("campaign", ROOT / "scripts" / "measure_dqi_schedule_campaign.py")
    halt = load_module("eea_oracle_ledger",
                       ROOT / "scripts" / "measure_eea_oracle_ledger.py")
    reuse = load_module("flag_predicate_reuse",
                        ROOT / "scripts" / "measure_flag_predicate_reuse.py")

    decoder_type = campaign.make_decoder(
        arms, "stacked", (halt.HaltGatedDecoder, reuse.ReusedFlagPredicateDecoder))
    shim = runner.shim_for(km, ledger.native_builder(runner.builder_class(km), "and54"))
    original = arms.DR.km
    arms.DR.km = shim
    try:
        started = time.time()
        builder = shim.CircuitBuilder()
        syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
        error_mbuc = [builder.create_classical_register(degree, name=f"E{j}")
                      for j in range(m)]
        decoder_type(n, km.GF2Field(degree), **campaign.SIX_LEVERS).\
            build_syndrome_decoding_mbuc(
                builder, syndromes=syndromes, error_mbuc=error_mbuc,
                n_iters=arms.DR.default_n_iters(n))
        return {"circuit": builder.finish_circuit(),
                "build_seconds": round(time.time() - started, 2),
                "gates": {key: bool(value) for key, value in campaign.SIX_LEVERS.items()},
                "mixins": [halt.HaltGatedDecoder.__name__,
                           reuse.ReusedFlagPredicateDecoder.__name__]}
    finally:
        arms.DR.km = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=OFFICIAL_NUM_SHOTS,
                        help="shots to verify; a multiple of --batch-size "
                             "(default: the competition's 102,400)")
    parser.add_argument("--batch-size", type=int, default=OFFICIAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    if args.shots % args.batch_size:
        raise SystemExit(f"--shots must be a multiple of --batch-size={args.batch_size}")
    if not list(args.kickmix_build.glob("kickmix*.so")):
        raise SystemExit(f"no extension in {args.kickmix_build}: build it with "
                         f"`{sys.executable} scripts/build_kickmix_mr.py`")

    sys.path.insert(0, str(args.kickmix_build))
    import kickmix as km  # noqa: PLC0415 - the build under test

    if not hasattr(km.CircuitBuilder, "mr") or not hasattr(km.CircuitBuilder, "gf_product"):
        raise SystemExit(f"{km.__file__} needs `mr` and `gf_product`: this is the "
                         f"MR rebuild's row, not the pinned binary's")

    E = load_module("dqi_submission_score",
                    ROOT / "scripts" / "measure_dqi_submission_score.py")
    with tempfile.TemporaryDirectory() as verifier_dir:
        V, verifier_patch = E.load_verifier(Path(verifier_dir))

        print("[stacked] building ...", flush=True)
        built = build_stacked(km, E, args.m, args.n, args.degree)
        circuit = built["circuit"]
        print(f"[stacked] Q={circuit.num_qubits} ops={len(circuit):,} "
              f"({built['build_seconds']}s)", flush=True)

        spec = V.ProblemSpecDQI.benchmark_instance(
            m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
        print(f"[stacked] verifying {args.shots} shots on {args.workers} workers ...",
              flush=True)
        started = time.time()
        result = V.verify(circuit, spec, batch_size=args.batch_size,
                          log2_delta=OFFICIAL_LOG2_DELTA, max_workers=args.workers)
        seconds = round(time.time() - started, 1)

    verified = {name: result[name] for name in E.KEPT_FIELDS}
    print(f"[stacked] {result['status']}: score={result['score']:.6e} "
          f"Q={result['num_qubits']} T={result['toffoli_count']:.6f} "
          f"value_err={result['value_errors']} phase_err={result['phase_errors']} "
          f"R={result['repetition_factor']:.6f} ({seconds}s)", flush=True)

    reference = json.loads(REFERENCE_RECORD.read_text())["verified"]["reference"]
    six_lever = json.loads(ARM_RECORD.read_text())
    like_for_like = args.shots == reference["verified_shots"]
    ratio_reference = verified["score"] / reference["score"] if like_for_like else None
    ratio_six_lever = (verified["score"] / six_lever["verified"]["score"]
                       if like_for_like else None)
    ratio_target = verified["score"] / TARGET_SCORE
    t_gap = verified["toffoli_count"] / TARGET_T_MEAN
    q_ratio_squared = (verified["num_qubits"] / TARGET_QUBITS) ** 2
    t_needed = TARGET_T_MEAN * q_ratio_squared
    zero_error = E.score_budget(args.shots, 0, V=V)

    checks = [
        {
            "check": "the campaign arm passes the competition's verifier",
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
            "check": "the circuit reproduces the campaign's stacked row",
            "requirement": f"T = {STACKED_T_MEAN:,.1f} per shot (within "
                           f"{T_TOLERANCE:g} relative) at Q = {STACKED_QUBITS}",
            "measured": {"toffoli_count": verified["toffoli_count"],
                         "num_qubits": verified["num_qubits"]},
            "passed": (abs(verified["toffoli_count"] / STACKED_T_MEAN - 1) < T_TOLERANCE
                       and verified["num_qubits"] == STACKED_QUBITS),
        },
        {
            "check": "the score is R * Q^2 * T from the verifier's own numbers",
            "requirement": "the reported score equals R(eps_plus) * Q^2 * T to 1e-12",
            "measured": {"score": verified["score"],
                         "repetition_factor": verified["repetition_factor"]},
            "passed": abs(verified["score"] - verified["repetition_factor"]
                          * verified["num_qubits"] ** 2
                          * verified["toffoli_count"]) / verified["score"] < 1e-12,
        },
        {
            "check": "the campaign arm is still above the derived target",
            "requirement": "the record states the measured gap; this fails only if the "
                           "arm is *inside* the target",
            "measured": {"ratio_to_target": ratio_target,
                         "toffoli_ratio_to_target": t_gap,
                         "toffolis_needed_at_this_Q": t_needed},
            "passed": ratio_target > 1.0,
        },
    ]
    if like_for_like:
        # A ratio between runs at different budgets compares two different `R`s, so
        # this check only exists when the comparison is with Experiment F's own sample.
        checks.insert(2, {
            "check": "the campaign arm beats the six-lever arm it stacks on",
            "requirement": "a lower score than Experiment F's row, at the same shot "
                           "count and the same Q",
            "measured": {"score": verified["score"],
                         "six_lever_score": six_lever["verified"]["score"],
                         "ratio": ratio_six_lever},
            "passed": (verified["score"] < six_lever["verified"]["score"]
                       and verified["num_qubits"] == six_lever["verified"]["num_qubits"]),
        })

    payload = {
        "experiment": "G",
        "question": "what does the competition's own verifier score the best circuit "
                    "this repository has built -- the campaign's stacked arm?",
        "declared_in": "docs/BENCHMARK2_TARGET_REACHABILITY.md",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "batch_size": args.batch_size,
                     "log2_delta": OFFICIAL_LOG2_DELTA, "arm": "stacked",
                     "clear_with": "mr", "product_policy": "and54",
                     "gates": built["gates"], "mixins": built["mixins"]},
        "status": "measured on the local MR rebuild; NOT a receipt",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(
            Path(km.__file__).resolve().read_bytes()).hexdigest(),
        "kickmix_role": ("the local rebuild from `scripts/build_kickmix_mr.py`. It has "
                         "`mr` (the arm's classical-condition reads) and `gf_product` "
                         "(the 54-AND policy); the pinned competition binary has "
                         "neither, so this circuit cannot be submitted and only the "
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
            "pinned_reference": {"score": reference["score"],
                                 "toffoli_count": reference["toffoli_count"],
                                 "num_qubits": reference["num_qubits"],
                                 "verified_shots": reference["verified_shots"]},
            "six_lever_arm": {"score": six_lever["verified"]["score"],
                              "toffoli_count": six_lever["verified"]["toffoli_count"],
                              "verified_shots": six_lever["verified"]["verified_shots"]},
            "ratio_to_pinned_reference": ratio_reference,
            "ratio_to_six_lever_arm": ratio_six_lever,
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
            f"the campaign's stacked arm passes the competition's `verify()` at "
            f"{args.shots:,} shots with {verified['value_errors']} value and "
            f"{verified['phase_errors']} phase errors and scores "
            f"{verified['score']:.6e} (Q={verified['num_qubits']}, "
            f"T={verified['toffoli_count']:.6f}, R={verified['repetition_factor']:.6f}). "
            f"Its `Q^2 T` product had put it at 1.541x the derived target on `T` alone; "
            f"the measured score is {ratio_target:.3f}x the target once the `Q` term "
            f"(1.143x) is included, and at Q={verified['num_qubits']} the target's "
            f"445,229 would have to be {t_needed:,.0f} against the measured "
            f"{verified['toffoli_count']:,.1f}. This is the frontier: no circuit this "
            f"repository can build on any extension reaches 1.6975e12, and the share it "
            f"is missing is decoder algorithm, not an opcode."
        ),
        "scope": {
            "measured": f"the competition's verifier on the stacked arm at "
                        f"{args.shots:,} shots (every scoring line unmodified; only "
                        f"`reaction_depth` is stubbed, see verifier_patch): status, "
                        f"value and phase errors, `eps_plus`, `R`, score, Q and T",
            "not_measured": "a receipt, or anything about the pinned binary: the arm "
                            "needs `mr` and `gf_product`, so this row cannot be "
                            "submitted and is not a submission",
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
    print(f"ratio to reference {ratio_reference}; ratio to six-lever arm "
          f"{ratio_six_lever}; ratio to target {ratio_target:.3f}")
    print(f"failed checks: {payload['failed_checks'] or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not payload["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
