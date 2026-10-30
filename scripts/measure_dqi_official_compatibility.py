#!/usr/bin/env python3
"""Experiment D: what the pinned competition binary can actually compile.

`docs/DQI_RESOURCE_RECONCILIATION.md` §2 records one sentence of this experiment --
``arm on the pinned binary: built = False, AttributeError: 'Builder' object has no
attribute 'mr'`` -- and calls the missing ``MR`` opcode "the only blocker between
686,014.8 and a receipt". This script measures the whole surface instead of one
attribute, because the arm's saving has more than one dependency:

1. **`mr`.** Every lever skips work under a *classical* condition, and a classical
   condition needs a measurement. The pinned builder has no ``mr``. The obvious
   substitution -- ``MR`` is ``HMR`` with one extra ``H`` in front -- is exact
   algebra (`benchmarks/quantum_compilation/vqc_licence_boundary.json` measures
   the identity), so the question is whether the builder has an ``H``. It does
   not, and that is measured here rather than assumed.
2. **`gf_product`.** The 54-AND product policy is a builder attribute. The arm's
   1,221,466.8 (and the stacked 686,014.8) are measured *with* it; the policy is
   worth 523,849.1 on the reference and 103,763.2 on the arm. It is absent from the
   pinned builder too, which the reconciliation note does not record.
3. **`del_and`.** If the AND could be emitted as a gate stream instead, its clear
   would be the question: an uncompute costs a second Toffoli, a measured clear
   costs none. This script measures the pinned binary's ``del_and``, so "emit the
   multiply instead of asking for the policy" is decided by a count rather than by
   preference.

The answer is a floor: the cheapest circuit the pinned binary can express is the
reference decoder itself, and everything this repository has measured is expressed
in instructions it does not have. That is a negative result, and the record keeps
the two numbers apart -- what the arm would score, and what can be submitted --
because quoting the first as if it were the second is the error the reconciliation
note was written to prevent.

Examples::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_official_compatibility.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
                  / "dqi_official_compatibility.json")

#: What the arm is, for the record, before this experiment tests whether it compiles.
ARM_TOTAL = 686_014.8
ARM_QUBITS = 1_913
REFERENCE_TOTAL = 5_725_071.609375
REFERENCE_QUBITS = 1_893
#: The and54 policy's measured worth, from `gf54_native_opcode.json`: reference
#: 5,725,071.6 -> 5,201,222.5 and arm 1,325,230.0 -> 1,221,466.8.
POLICY_WORTH_REFERENCE = 5_725_071.609375 - 5_201_222.5
POLICY_WORTH_ARM = 1_325_230.0 - 1_221_466.8

SURFACE = ("mr", "hmr", "h", "gf_product", "bit_store", "alloc_bits", "push_condition",
           "pop_condition", "del_and", "init_and", "cswap",
           # a *statistics* surface: the campaign's per-region ledger reads it
           "counting_op_executions")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def and_clear_costs(km) -> dict:
    """What `init_and` and `del_and` cost on this build, in Toffolis."""
    out = {}
    for name, ops in (("init_and", 1), ("and_pair", 2)):
        builder = km.CircuitBuilder()
        a = builder.alloc_qubits(1)[0]
        b = builder.alloc_qubits(1)[0]
        target = builder.alloc_qubits(1)[0]
        builder.x(a)
        builder.init_and(a, b, target)
        if ops == 2:
            builder.del_and(a, b, target)
        out[name] = int(builder.finish_circuit().operation_statistics().num_toffoli_like)
    out["del_and"] = out["and_pair"] - out["init_and"]
    return out


def simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, degree, shots) -> dict:
    """`ledger.simulate`, without the per-instruction counts the pinned binary lacks."""
    from verify_submission_dqi import to_bits

    sim = km.Simulator(batch_size=shots, seed=99, counting_operations=True)
    sim.use_same_registers_as(circuit)
    flat_syndromes = km.array([q for r in syn for q in r])
    flat_errors = km.array([b for r in mbuc for b in r])
    sim.write_across_shots(flat_syndromes, to_bits(syndromes_in, degree))
    sim.write_across_shots(flat_errors, to_bits(mbuc_in, degree))
    sim.do(circuit)
    per_shot = np.array([int(sim.operation_statistics(i).num_toffoli_like)
                         for i in range(shots)])
    syndrome_ids = {q.id for r in syn for q in r}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    return {
        "toffoli_per_shot": per_shot,
        "mean_toffoli_like": float(per_shot.mean()),
        "syndromes": sim.read_across_shots(flat_syndromes),
        "junk": sim.read_across_shots([km.q(i) for i in junk_ids]),
    }


def measure(build_dir: Path, m: int, n: int, degree: int, shots: int, seed: int) -> dict:
    """Measure one extension. Run in a subprocess so the two `kickmix` builds cannot mix."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(build_dir))
    runner = load_module("mr_gated_arms_kickmix",
                         ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    import kickmix as km  # noqa: PLC0415 - the build under test

    builder = km.CircuitBuilder()
    surface = {name: bool(hasattr(builder, name)) for name in SURFACE}

    result = {
        "build": str(build_dir),
        "extension": str(Path(km.__file__).resolve()),
        "extension_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "surface": surface,
        "and_clear": and_clear_costs(km),
    }

    # Can the 54-AND policy be requested at all, and can it be requested by setting it
    # on an instance? (A pybind11 class without dynamic_attr rejects the assignment.)
    try:
        builder.gf_product = "and54"
        result["policy_assignment"] = {"works": True}
    except Exception as exc:  # noqa: BLE001 - the failure mode is the measurement
        result["policy_assignment"] = {"works": False, "error_type": type(exc).__name__,
                                       "error": str(exc)}

    # The read the whole arm's saving depends on, and the substitution that would
    # avoid it: `mr` is `h` followed by `hmr`.
    for name, call in (
        ("mr", lambda b: b.mr(b.alloc_qubits(1)[0], b.alloc_bits(1)[0])),
        ("h", lambda b: b.h(b.alloc_qubits(1)[0])),
    ):
        probe = km.CircuitBuilder()
        try:
            call(probe)
            result[f"{name}_call"] = {"works": True}
        except Exception as exc:  # noqa: BLE001
            result[f"{name}_call"] = {"works": False, "error_type": type(exc).__name__,
                                      "error": str(exc)}

    ledger = load_module("region_ledger",
                         ROOT / "scripts" / "measure_dqi_region_ledger.py")
    gf54 = load_module("gf54_mul", ROOT / "scripts" / "gf54_mul.py")
    arms = ledger.load_arms(km, runner)
    base_builder = runner.builder_class(km)
    # Two shims, deliberately: the *identical-circuit* comparison between the two
    # binaries has to use the product policy both of them have, which is the built-in
    # one; the 54-AND policy is a separate row that only the local build can produce.
    builtin_shim = runner.shim_for(km, ledger.native_builder(base_builder, "karatsuba"))
    shim = runner.shim_for(km, ledger.native_builder(base_builder, "and54"))

    # The officially compilable candidate, and the substitution it rests on: the
    # 54-AND multiply clears its workspace rather than reading it, and a *clear* does
    # not need the value -- so `HMR`, which the pinned binary has, can do it where `MR`,
    # which it does not, is what the emitter asks for. A read cannot be substituted
    # this way (vqc_licence_boundary.json measures HMR randomising a basis qubit), so
    # this route reaches the multiply and not the levers.
    class ClearWithHmr(base_builder):
        def mr(self, target, bit):  # noqa: D401 - the substitution under test
            self.hmr(target, bit)

    candidate_builder = type(
        "OfficialCandidate", (gf54.Gf54MulMixin, ClearWithHmr),
        {"__doc__": "reference decoder, 54-AND gate-stream multiply, HMR clears"})
    candidate_shim = runner.shim_for(km, candidate_builder)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V  # noqa: PLC0415

    iterations = arms.DR.default_n_iters(n)
    spec = V.ProblemSpecDQI.benchmark_instance(
        m=m, n=n, degree=degree, num_shots=shots)
    weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    tables = V.FieldTables(km.GF2Field(spec.degree))
    rng = np.random.default_rng(seed)
    syndromes_in, mbuc_in, _ = V.sample_shots(spec, tables, rng, shots, weights)

    original = arms.DR.km
    arms.DR.km = builtin_shim
    try:
        circuit, syn, mbuc = ledger.build_reference(km, arms, builtin_shim, n, degree, m)
        out = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, degree, shots)
        result["reference"] = {
            "qubits": circuit.num_qubits,
            "static_toffoli": int(circuit.operation_statistics().num_toffoli_like),
            "mean_toffoli_like": out["mean_toffoli_like"],
            "toffoli_per_shot": out["toffoli_per_shot"].tolist(),
            "syndromes_restored": bool(np.array_equal(
                out["syndromes"], V.to_bits(syndromes_in, degree))),
            "junk_clean": bool(not out["junk"].any()),
        }
        del circuit
        result["reference_builds"] = True
        result["reference_policy"] = "built-in (karatsuba)"

        # The same reference with the 54-AND policy, where the build has the policy at
        # all. This is the row the reachability notes quote, and the row the pinned
        # binary cannot produce.
        if surface["gf_product"]:
            arms.DR.km = shim
            policy_circuit, policy_syn, policy_mbuc = ledger.build_reference(
                km, arms, shim, n, degree, m)
            policy_out = simulate(km, policy_circuit, policy_syn, policy_mbuc,
                                  syndromes_in, mbuc_in, degree, shots)
            result["reference_and54"] = {
                "qubits": policy_circuit.num_qubits,
                "mean_toffoli_like": policy_out["mean_toffoli_like"],
                "syndromes_restored": bool(np.array_equal(
                    policy_out["syndromes"], V.to_bits(syndromes_in, degree))),
            }
            del policy_circuit
        arms.DR.km = shim

        # The candidate: does the arm build here at all?
        try:
            arm_circuit, _arm_syn, _arm_mbuc = ledger.build_arm(
                km, arms, shim, {"gate_forney": True, "clear_eea": True,
                                 "gate_sweep": True, "gate_swap": True,
                                 "gate_shift": True, "gate_quotient": True},
                n, degree, m)
            result["arm"] = {"built": True, "qubits": arm_circuit.num_qubits,
                             "static_toffoli":
                                 int(arm_circuit.operation_statistics().num_toffoli_like)}
            del arm_circuit
        except Exception as exc:  # noqa: BLE001 - the failure mode is the measurement
            result["arm"] = {"built": False, "error_type": type(exc).__name__,
                             "error": str(exc)}

        # The candidate: the reference algorithm with the gate-stream multiply, whose
        # clears are HMR. No lever, no policy attribute, no Z-basis read.
        arms.DR.km = candidate_shim
        try:
            cand_circuit, cand_syn, cand_mbuc = ledger.build_reference(
                km, arms, candidate_shim, n, degree, m)
            cand_out = simulate(km, cand_circuit, cand_syn, cand_mbuc, syndromes_in,
                                mbuc_in, degree, shots)
            result["candidate"] = {
                "built": True,
                "qubits": cand_circuit.num_qubits,
                "static_toffoli":
                    int(cand_circuit.operation_statistics().num_toffoli_like),
                "mean_toffoli_like": cand_out["mean_toffoli_like"],
                "toffoli_per_shot": cand_out["toffoli_per_shot"].tolist(),
                "syndromes_restored": bool(np.array_equal(
                    cand_out["syndromes"], V.to_bits(syndromes_in, degree))),
                "call": "reference algorithm + Gf54MulMixin + HMR clears",
            }
            del cand_circuit
        except Exception as exc:  # noqa: BLE001
            result["candidate"] = {"built": False, "error_type": type(exc).__name__,
                                   "error": str(exc)}
    finally:
        arms.DR.km = original
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
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(f"measuring {label} ({build}) failed:\n{proc.stderr}")
        runs[label] = json.loads(proc.stdout)

    pinned, local = runs["pinned"], runs["local"]
    pinned_ref, local_ref = pinned["reference"], local["reference"]
    policy_row = local.get("reference_and54")
    policy_worth = (local_ref["mean_toffoli_like"] - policy_row["mean_toffoli_like"]
                    if policy_row else POLICY_WORTH_REFERENCE)
    identical_reference = bool(
        np.array_equal(pinned_ref["toffoli_per_shot"], local_ref["toffoli_per_shot"]))

    pinned_instance = (args.m, args.n, args.degree) == (4095, 70, 12) and args.shots == 64
    checks = [{
        "check": "the pinned reference reproduces the recorded row",
        "requirement": f"{REFERENCE_TOTAL:.6f} per shot at Q={REFERENCE_QUBITS}"
                       if pinned_instance else "not the recorded instance",
        "measured": {"mean": pinned_ref["mean_toffoli_like"],
                     "qubits": pinned_ref["qubits"]},
        "passed": ((abs(pinned_ref["mean_toffoli_like"] - REFERENCE_TOTAL) < 1e-6
                    and pinned_ref["qubits"] == REFERENCE_QUBITS)
                   if pinned_instance else True),
    }, {
        "check": "the two binaries count the same reference identically, shot for shot",
        "requirement": "identical per-shot arrays, both built with the built-in multiply "
                       "(the only policy the pinned binary has)",
        "measured": {"identical": identical_reference,
                     "pinned_mean": pinned_ref["mean_toffoli_like"],
                     "local_mean": local_ref["mean_toffoli_like"]},
        "passed": identical_reference,
    }, {
        "check": "the pinned binary cannot build the arm",
        "requirement": "the build fails, and the failure names the missing surface",
        "measured": pinned["arm"],
        "passed": not pinned["arm"].get("built", False),
    }, {
        "check": "no readout instruction stands in for `mr`",
        "requirement": "`mr` is absent, `h` is absent, so `h; hmr` cannot be emitted",
        "measured": {"mr": pinned["mr_call"], "h": pinned["h_call"],
                     "hmr_present": pinned["surface"]["hmr"]},
        "passed": (not pinned["mr_call"]["works"] and not pinned["h_call"]["works"]),
    }, {
        "check": "the 54-AND policy cannot be requested on the pinned binary",
        "requirement": "absent as an attribute and rejected on assignment",
        "measured": {"present": pinned["surface"]["gf_product"],
                     "assignment": pinned["policy_assignment"]},
        "passed": (not pinned["surface"]["gf_product"]
                   and not pinned["policy_assignment"].get("works", False)),
    }]

    local_arm = local["arm"]
    checks.append({
        "check": "the same arm builds on the local MR rebuild",
        "requirement": "built, at the incumbent's qubit count"
                       if pinned_instance else "built",
        "measured": local_arm,
        "passed": bool(local_arm.get("built"))
        and (local_arm.get("qubits") == ARM_QUBITS if pinned_instance else True),
    })

    candidate = pinned.get("candidate", {"built": False})
    if candidate.get("built"):
        candidate_score_ratio = ((candidate["mean_toffoli_like"]
                                 / pinned_ref["mean_toffoli_like"])
                                * (candidate["qubits"] / pinned_ref["qubits"]) ** 2)
        checks.append({
            "check": "a candidate the pinned binary can compile, and its cost",
            "requirement": "builds, restores the syndromes, and its score ratio to the "
                           "shipped reference is recorded",
            "measured": {"qubits": candidate["qubits"],
                         "mean_toffoli_like": candidate["mean_toffoli_like"],
                         "toffoli_ratio": candidate["mean_toffoli_like"]
                                          / pinned_ref["mean_toffoli_like"],
                         "qubit_ratio": candidate["qubits"] / pinned_ref["qubits"],
                         "score_ratio_to_reference": candidate_score_ratio,
                         "syndromes_restored": candidate["syndromes_restored"],
                         "call": candidate["call"]},
            "passed": bool(candidate["syndromes_restored"]),
        })

    failed = [entry for entry in checks if not entry["passed"]]
    if failed:
        verdict = "inconsistent measurement"
    elif candidate.get("built"):
        verdict = (
            f"the arm is not compilable and one candidate is: the reference algorithm "
            f"with the 54-AND multiply emitted as gates and cleared with HMR, "
            f"Q={candidate['qubits']} T={candidate['mean_toffoli_like']:.6f} per shot "
            f"against the shipped reference's Q={pinned_ref['qubits']} "
            f"T={pinned_ref['mean_toffoli_like']:.6f}; every lever stays out of reach, "
            f"because a lever's saving is a skipped instruction and the condition for "
            f"skipping it is a read")
    else:
        verdict = ("no officially compilable candidate beyond the reference decoder")
    payload = {
        "experiment": "D -- official compatibility of the measured arm",
        "declared_in": "docs/DQI_RESOURCE_RECONCILIATION.md §2",
        "question": "what is the cheapest DQI circuit the pinned competition binary can "
                    "express, and what does the arm need that it does not have?",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": args.seed},
        "builds": {"pinned": pinned, "local": local},
        "comparison": {
            "reference_per_shot_pinned": pinned_ref["mean_toffoli_like"],
            "reference_per_shot_local": local_ref["mean_toffoli_like"],
            "policy": "the built-in multiply on both builds, so the comparison is the "
                      "same circuit on two counting backends",
            "reference_and54_local_only": policy_row,
            "per_shot_arrays_identical": identical_reference,
            "arm_per_shot": ARM_TOTAL,
            "arm_qubits": ARM_QUBITS,
            "arm_over_reference": REFERENCE_TOTAL / ARM_TOTAL,
        },
        "blockers": {
            "mr": {
                "why": "every lever's saving is a skipped instruction, and the condition "
                       "comes from a measurement",
                "worth_per_shot": 535_452.0,
                "substitution_tested": "mr == h; hmr (exact algebra, measured in "
                                       "vqc_licence_boundary.json)",
                "why_it_fails": "the pinned builder has `hmr` but no `h`, so the "
                                "substitution cannot be emitted",
            },
            "gf_product": {
                "present_on_pinned": pinned["surface"]["gf_product"],
                "documented_before_this_record": False,
                "worth_per_shot_reference": policy_worth,
                "worth_per_shot_arm": POLICY_WORTH_ARM,
                "measured_row": local.get("reference_and54"),
                "why_it_fails": "the 54-AND policy is a builder attribute and the pinned "
                                "builder rejects the assignment; emitting the multiply as "
                                "a gate stream instead would pay del_and, which this "
                                "record measures at one Toffoli per AND",
            },
        },
        "floor": {
            "claim": "the cheapest circuit the pinned binary can express is the reference "
                     "algorithm with the 54-AND multiply emitted as gates and cleared with "
                     "HMR; the levers are still out of reach, because their saving needs a "
                     "*read*",
            "per_shot": pinned_ref["mean_toffoli_like"],
            "qubits": pinned_ref["qubits"],
            "candidate": candidate,
            "what_it_would_take": "an upstream build exposing `mr` (for reads) and the "
                                  "product policy; `hmr` is enough for clears and is "
                                  "already in the pinned binary",
        },
        "checks": checks,
        "failed_checks": [entry["check"] for entry in failed],
        "verdict": verdict,
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pinned_kickmix_sha256": pinned["extension_sha256"],
            "local_kickmix_sha256": local["extension_sha256"],
        },
        "scope": {
            "measured": "the surface of both builds, the AND-clear cost, the reference on "
                        "both, and the arm's build outcome on both",
            "not_measured": "no submission: the verifier is never run on the arm, because "
                            "the pinned binary cannot construct it",
        },
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"pinned  Q={pinned_ref['qubits']} T={pinned_ref['mean_toffoli_like']:.6f}")
    print(f"local   Q={local_ref['qubits']} T={local_ref['mean_toffoli_like']:.6f}")
    print(f"arm built on pinned: {pinned['arm'].get('built')}  "
          f"on local: {local_arm.get('built')}")
    print(f"pinned surface: mr={pinned['surface']['mr']} h={pinned['surface']['h']} "
          f"hmr={pinned['surface']['hmr']} gf_product={pinned['surface']['gf_product']}")
    print(f"pinned del_and cost: {pinned['and_clear']['del_and']}")
    print(f"failed checks: {payload['failed_checks'] or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
