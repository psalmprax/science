#!/usr/bin/env python3
"""Experiment E: the score the competition's own verifier assigns, and where it comes from.

Experiments D and D2 (`docs/DQI_OFFICIAL_COMPATIBILITY.md`) measured Toffoli counts per
shot and checked correctness with a hand-written copy of the verifier's per-shot semantics.
Both stopped short of the thing the competition scores: `verify()` in
`verifiers/dqi/verify_submission_dqi.py`, which samples its own shots, counts value *and*
phase errors, converts the error count into a one-sided Clopper-Pearson bound
`eps_plus`, refuses the circuit outright if `eps_plus >= 1/16`, and reports

    score = R(eps_plus) * Q^2 * T,   R(eps) = ln(2) / -ln(1/2 + 2 * sqrt(eps)).

This experiment runs that function, unmodified, on the reference circuit and on the
phase-safe 54-AND candidate, at the same shot budget, and reports the two scores and
their ratio.

Three things it establishes, and one it cannot:

1. **A pass, at the official budget.** `eps_plus` at the competition's default
   `log2_delta = -128` is `1 - 2^(-128/n)`, so a *zero-error* run at `n` shots still
   fails the `1/16` gate below `n ~ 1375`; the default budget here is therefore the
   competition's own 102,400 shots (400 batches of 256). `--shots` can lower it, and
   the record names whatever was actually verified.

2. **The ratio is budget-free.** Both circuits have zero errors at the same `n`, so
   they share `eps_plus` and therefore `R`; the score ratio is exactly the
   `Q^2 * T` ratio. That is why the ratio answers the question D asked -- is the
   candidate cheaper *by the official metric* -- even where the absolute score is
   dominated by how wide `R` is at a reduced budget.

3. **The exact decomposition of the saving.** The candidate replaces only
   `ixor_gf2_mul`: 66 Toffoli-like gates per call (54 ANDs + the 12-Toffoli add)
   against the built-in 75 (63 + 12), so each emitted call should save exactly 9. The
   script counts the calls the reference emits and checks `9 * calls` against the
   measured per-shot difference to the gate, and against D2's recorded counts.

4. **Why the levers still cannot compile.** Their condition is a *read* of qubit data
   into a classical bit. Rather than assert the pinned builder cannot express that,
   the script introspects its surface -- the pybind doc signatures it ships -- lists
   every method that mentions a `km.b`, and classifies them by whether the bit is an
   `output`, a `target` or a control. It then measures the `output` ones on |1>: a
   primitive that reads a qubit writes the bit the qubit holds.

The two circuits are built and verified in separate subprocesses: together they hold
about 3 GB of instruction stream, and forking the verifier's batch workers over both
at once has been observed to exhaust this machine.

Examples::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_submission_score.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_submission_score.py --shots 1536
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
PINNED_DIR = BUNDLE / "kickmix"
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                  / "dqi_submission_score.json")

#: Rows these numbers are checked against, from the records Experiment E extends.
REFERENCE_TOTAL = 5_725_071.609375
REFERENCE_QUBITS = 1893
CANDIDATE_TOTAL = 5_433_399.609375
CANDIDATE_QUBITS = 1893

#: The competition's own defaults (`verify_submission_dqi`).
OFFICIAL_NUM_SHOTS = 400 * 256
OFFICIAL_BATCH_SIZE = 256
OFFICIAL_LOG2_DELTA = -128.0
OFFICIAL_MAX_ERROR_RATE = 1 / 16

#: Toffoli-like gates one controlled multiply costs, both arms.
REFERENCE_TOFFOLIS_PER_CALL = 63 + 12
CANDIDATE_TOFFOLIS_PER_CALL = 54 + 12

#: How close the executed mean Toffoli count has to land on D2's recorded one. The
#: decoder has classically conditioned Toffolis, so the mean over shots is a random
#: variable even for a fixed circuit; 64 shots of D2 and `--shots` here differ by ~1e-5.
T_TOLERANCE = 1e-4


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The one line of `verify()` this experiment does not run. `reaction_depth` is a
#: reported diagnostic -- the score is `R * Q^2 * T` -- and computing it on the 86M-op
#: reference costs a 13.9 GB peak and 136 s, which the kernel kills on this machine.
REACTION_DEPTH_LINE = "'reaction_depth': circuit.reaction_depth(),"
REACTION_DEPTH_STUB = "'reaction_depth': None,  # not computed by Experiment E"


def load_verifier(workdir: Path):
    """The competition's verifier, with one documented reporting line stubbed out."""
    source_path = BUNDLE / "verifiers" / "dqi" / "verify_submission_dqi.py"
    source = source_path.read_text()
    if source.count(REACTION_DEPTH_LINE) != 1:
        raise SystemExit(f"expected one {REACTION_DEPTH_LINE!r} in {source_path}")
    patched = source.replace(REACTION_DEPTH_LINE, REACTION_DEPTH_STUB)
    path = workdir / "verify_submission_dqi.py"
    path.write_text(patched)
    sys.path.insert(0, str(workdir))
    # `@dataclass` resolves its own annotations through `sys.modules`, so the module
    # has to be registered before it executes.
    spec = importlib.util.spec_from_file_location("verify_submission_dqi", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_submission_dqi"] = module
    spec.loader.exec_module(module)
    patch = {
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "replaced": REACTION_DEPTH_LINE,
        "with": REACTION_DEPTH_STUB,
        "why": "`reaction_depth` is a reported diagnostic, not part of the score; on "
               "the 86M-op reference it peaked at 13.9 GB and took 136 s, which gets "
               "the process killed here. Every scoring line is the shipped one.",
    }
    return module, patch


# ------------------------------------------------------------- the pinned surface

def _signature_text(method) -> str:
    """The `@signature` line pybind generated for `method`, on one line."""
    doc = getattr(method, "__doc__", "") or ""
    start = doc.find("@signature")
    if start < 0:
        return ""
    end = doc.find("->", start)
    return re.sub(r"\s+", " ", doc[start:end if end > 0 else len(doc)])


def bit_surface(km) -> dict[str, dict]:
    """Every pinned builder method whose signature mentions a classical bit.

    Read off the extension's own docstrings, so the list is what the binary exposes,
    not what a header claims. Each method is classified by the *role* the bit plays:

    `output`  -- the method writes a bit the caller supplied, from a qubit.
    `target`  -- the bit is a target, but not of qubit data (`bit_store` writes a
                 bool, `x`/`z` flip it, `free`/`swap`/`debug_print` are plumbing).
    `control` -- the bit is read as a condition; it is an input, not a product.
    """
    surface: dict[str, dict] = {}
    for name in sorted(dir(km.CircuitBuilder)):
        if name.startswith("_"):
            continue
        method = getattr(km.CircuitBuilder, name, None)
        if not callable(method):
            continue
        signature = _signature_text(method)
        if not signature or "km.b" not in signature:
            continue
        if re.search(r"\boutputs?:\s*[^,)]*km\.b", signature):
            role = "output"
        elif re.search(r"\btargets?:\s*[^,)]*km\.b", signature):
            role = "target"
        else:
            role = "control"
        surface[name] = {"role": role, "signature": signature,
                         "parameters": re.findall(r"(\w+):", signature)}
    return surface


def probe_bit_outputs(km, surface: dict[str, dict], shots: int = 64) -> dict[str, dict]:
    """Call each `output`-role method on |1> and look at the bit it writes.

    A method that *reads* a qubit writes the qubit's value every time; anything else
    writes something that is not the input. The tracked phase is recorded too, since
    an X-basis measurement of |1> flips it.
    """
    rows = {}
    for name, info in surface.items():
        if info["role"] != "output":
            continue
        builder = km.CircuitBuilder()
        qubit = builder.alloc_qubits(1)
        bit = builder.alloc_bits(1)
        builder.x(qubit)
        try:
            getattr(builder, name)(km.array([qubit[0]]), km.array([bit[0]]))
        except Exception as error:  # noqa: BLE001 - report, don't hide the surface
            rows[name] = {"callable": False, "error": repr(error)}
            continue
        circuit = builder.finish_circuit()
        sim = km.Simulator(batch_size=shots, seed=13, counting_operations=True)
        sim.use_same_registers_as(circuit)
        sim.do(circuit)
        values = np.asarray(sim.read_across_shots(bit[0])).ravel().astype(bool)
        phase = np.asarray(sim.read_phase_flipped_across_shots()).ravel()
        rows[name] = {
            "callable": True,
            "shots": int(shots),
            "target_prepared": "|1>",
            "bit_ones": int(values.sum()),
            "bit_zeros": int((~values).sum()),
            "reads_qubit": bool(values.all()),
            "writes_random": bool(values.any() and not values.all()),
            "phase_flips": int(phase.sum()),
        }
    return rows


# --------------------------------------------------------------------- the circuits

def counting_mixin(base):
    """A builder that counts `ixor_gf2_mul` emissions, before any override runs."""

    class Counted(base):
        calls = 0

        def ixor_gf2_mul(self, *args, **kwargs):
            type(self).calls += 1
            return super().ixor_gf2_mul(*args, **kwargs)

    return Counted


def build_circuit(km, m: int, n: int, degree: int, role: str) -> dict:
    """One arm's circuit, with the number of multiply calls it emitted."""
    sys.path.insert(0, str(ROOT / "scripts"))
    runner = load_module("mr_gated_arms_kickmix",
                         ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    ledger = load_module("region_ledger", ROOT / "scripts" / "measure_dqi_region_ledger.py")
    gf54_reduced = load_module("gf54_mul_reduced", ROOT / "scripts" / "gf54_mul_reduced.py")
    arms = ledger.load_arms(km, runner)

    base = runner.builder_class(km)
    builders = {
        "reference": counting_mixin(ledger.native_builder(base, "karatsuba")),
        "candidate": counting_mixin(
            type("PhaseSafe", (gf54_reduced.Gf54MulReducedMixin, base),
                 {"__doc__": "reduced-window 54-AND multiply (del_and + del_gf2_mul)"})),
    }
    builder = builders[role]
    shim = runner.shim_for(km, builder)
    original = arms.DR.km
    try:
        arms.DR.km = shim
        started = time.time()
        circuit, _, _ = ledger.build_reference(km, arms, shim, n, degree, m)
        return {"circuit": circuit, "multiply_calls": int(getattr(builder, "calls", 0)),
                "build_seconds": round(time.time() - started, 2)}
    finally:
        arms.DR.km = original


# ----------------------------------------------------------------------- the score

def score_budget(spec_shots: int, total: int, *, build: Path | None = None,
                 V=None) -> dict:
    """The Clopper-Pearson bound and `R` a run of `total` errors at this budget earns."""
    if V is None:
        sys.path.insert(0, str(build or PINNED_DIR))
        sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
        import verify_submission_dqi as V  # noqa: PLC0415 - the competition's module

    eps_plus = V.clopper_pearson_upper_bound(total, spec_shots,
                                             log2_delta=OFFICIAL_LOG2_DELTA)
    passes = bool(eps_plus < OFFICIAL_MAX_ERROR_RATE)
    # `repetition_overhead` refuses `eps >= 1/16` -- the same gate the verifier applies.
    return {"shots": spec_shots, "errors": total, "effective_error_rate": eps_plus,
            "repetition_factor": V.repetition_overhead(eps_plus) if passes else None,
            "passes": passes}


def smallest_passing_budget(build: Path) -> int:
    """The fewest batch-aligned shots a zero-error run can have and still pass."""
    for shots in range(OFFICIAL_BATCH_SIZE, OFFICIAL_NUM_SHOTS + 1, OFFICIAL_BATCH_SIZE):
        if score_budget(shots, 0, build=build)["passes"]:
            return shots
    raise SystemExit("no batch-aligned budget up to the official one passes at zero errors")


KEPT_FIELDS = ("status", "score", "verified_shots", "value_errors", "phase_errors",
               "empirical_error_rate", "effective_error_rate", "repetition_factor",
               "toffoli_count", "max_toffoli_count", "num_qubits", "num_bits",
               "reaction_depth", "num_ops", "mean_error_weight", "circuit_sha256")


def run_role(args, role: str) -> dict:
    """Build and verify one circuit. Run in its own process for the memory."""
    sys.path.insert(0, str(args.build))
    import kickmix as km  # noqa: PLC0415 - the build under test

    with tempfile.TemporaryDirectory() as verifier_dir:
        V, patch = load_verifier(Path(verifier_dir))
        out = run_role_with(args, role, km, V)
    out["verifier_patch"] = patch
    return out


def run_role_with(args, role: str, km, V) -> dict:
    """Build and verify one circuit against a loaded verifier module."""

    out = {"role": role, "build": str(args.build),
           "extension": str(Path(km.__file__).resolve()),
           "extension_sha256": hashlib.sha256(Path(km.__file__).resolve().read_bytes())
           .hexdigest(),
           "has_mr": bool(hasattr(km.CircuitBuilder, "mr"))}
    if role == "reference":
        # The pinned bit surface, read off the same extension.
        surface = bit_surface(km)
        out["bit_surface"] = surface
        out["probes"] = probe_bit_outputs(km, surface)

    print(f"[{role}] building ...", flush=True)
    built = build_circuit(km, args.m, args.n, args.degree, role)
    circuit = built["circuit"]
    out["multiply_calls"] = built["multiply_calls"]
    out["build_seconds"] = built["build_seconds"]
    print(f"[{role}] {built['multiply_calls']:,} multiply calls, "
          f"{len(circuit):,} ops, {built['build_seconds']}s", flush=True)

    spec = V.ProblemSpecDQI.benchmark_instance(
        m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
    started = time.time()
    print(f"[{role}] verifying {args.shots} shots on {args.workers} workers ...",
          flush=True)
    result = V.verify(circuit, spec, batch_size=args.batch_size,
                      log2_delta=OFFICIAL_LOG2_DELTA, max_workers=args.workers)
    out["seconds"] = round(time.time() - started, 1)
    out["verified"] = {name: result[name] for name in KEPT_FIELDS}
    out["shot_operation_counts"] = result["shot_operation_counts"]
    print(f"[{role}] {result['status']}: score={result['score']:.6e} "
          f"Q={result['num_qubits']} T={result['toffoli_count']:.6f} "
          f"value_err={result['value_errors']} phase_err={result['phase_errors']} "
          f"R={result['repetition_factor']:.6f} ({out['seconds']}s)", flush=True)
    return out


def combine(rows: dict[str, dict], args) -> dict:
    """The record: two verified runs, their ratio, its decomposition, the checks."""
    reference, candidate = rows["reference"]["verified"], rows["candidate"]["verified"]
    t_ratio = candidate["toffoli_count"] / reference["toffoli_count"]
    q_ratio = candidate["num_qubits"] / reference["num_qubits"]
    score_ratio = candidate["score"] / reference["score"]

    calls = rows["reference"]["multiply_calls"]
    per_call = REFERENCE_TOFFOLIS_PER_CALL - CANDIDATE_TOFFOLIS_PER_CALL
    predicted = per_call * calls
    # Two different quantities: `max_toffoli_count` is the static instruction count,
    # exact for any number of shots; `toffoli_count` is the mean that actually *executed*
    # per shot, and the decoder has classically conditioned Toffolis whose firing depends
    # on the state, so it is a random variable on top of the 9-per-call difference.
    static_delta = reference["max_toffoli_count"] - candidate["max_toffoli_count"]
    measured = reference["toffoli_count"] - candidate["toffoli_count"]
    op_delta = {
        op: rows["reference"]["shot_operation_counts"].get(op, 0)
            - rows["candidate"]["shot_operation_counts"].get(op, 0)
        for op in sorted(set(rows["reference"]["shot_operation_counts"])
                         | set(rows["candidate"]["shot_operation_counts"]))
    }
    op_delta = {op: delta for op, delta in op_delta.items() if delta}

    surface = rows["reference"]["bit_surface"]
    probes = rows["reference"]["probes"]
    qubit_to_bit = {name: info for name, info in surface.items()
                    if info["role"] == "output"}
    reads = {name: row for name, row in probes.items() if row.get("reads_qubit")}

    zero_error = score_budget(args.shots, 0, build=args.build)
    official = score_budget(OFFICIAL_NUM_SHOTS, 0, build=args.build)

    checks = [
        {
            "check": "the reference reproduces the row D2 recorded",
            "requirement": f"{REFERENCE_TOTAL:.6f} Toffolis per shot at Q={REFERENCE_QUBITS} "
                           f"(within {T_TOLERANCE:g} relative: the executed mean is "
                           f"shot-dependent)",
            "measured": {"toffoli_count": reference["toffoli_count"],
                         "num_qubits": reference["num_qubits"]},
            "passed": (abs(reference["toffoli_count"] / REFERENCE_TOTAL - 1) < T_TOLERANCE
                       and reference["num_qubits"] == REFERENCE_QUBITS),
        },
        {
            "check": "the candidate reproduces the row D2 recorded",
            "requirement": f"{CANDIDATE_TOTAL:.6f} at Q={CANDIDATE_QUBITS} "
                           f"(within {T_TOLERANCE:g} relative)",
            "measured": {"toffoli_count": candidate["toffoli_count"],
                         "num_qubits": candidate["num_qubits"]},
            "passed": (abs(candidate["toffoli_count"] / CANDIDATE_TOTAL - 1) < T_TOLERANCE
                       and candidate["num_qubits"] == CANDIDATE_QUBITS),
        },
        {
            "check": "the competition's verifier passes both circuits",
            "requirement": f"status PASS, zero value and phase errors, and "
                           f"eps_plus < 1/16 at {args.shots} shots",
            "measured": {key: {"status": row["verified"]["status"],
                               "value_errors": row["verified"]["value_errors"],
                               "phase_errors": row["verified"]["phase_errors"],
                               "effective_error_rate":
                                   row["verified"]["effective_error_rate"]}
                         for key, row in rows.items()},
            "passed": (all(row["verified"]["status"] == "PASS"
                           and row["verified"]["value_errors"] == 0
                           and row["verified"]["phase_errors"] == 0
                           for row in rows.values())
                       and all(row["verified"]["effective_error_rate"] < 1 / 16
                               for row in rows.values())),
        },
        {
            "check": "the score ratio is the Toffoli ratio, exactly",
            "requirement": "no R and no Q penalty: score_ratio == T_ratio to 1e-12",
            "measured": {"score_ratio": score_ratio, "t_ratio": t_ratio,
                         "q_ratio": q_ratio},
            "passed": abs(score_ratio - t_ratio) < 1e-12 and q_ratio == 1.0,
        },
        {
            "check": "the saving is exactly the multiply's 9 Toffolis per call",
            "requirement": f"static delta {static_delta} == {per_call} x {calls:,} calls "
                           f"= {predicted:,}, and the executed delta within 1% of it",
            "measured": {"multiply_calls": calls, "toffolis_per_call_delta": per_call,
                         "predicted_delta": predicted, "static_delta": static_delta,
                         "executed_delta": measured,
                         "operation_count_deltas": op_delta},
            "passed": (static_delta == predicted
                       and abs(measured / predicted - 1) < 0.01),
        },
        {
            "check": "no pinned primitive reads a qubit's value into a bit",
            "requirement": "every qubit->bit writer in the pinned surface writes "
                           "something that is not the qubit's value",
            "measured": {"qubit_to_bit_methods": sorted(qubit_to_bit),
                         "bit_writers_that_read_the_qubit": sorted(reads),
                         "probes": probes,
                         "has_mr": rows["reference"]["has_mr"]},
            "passed": not reads,
        },
    ]
    failed = [check for check in checks if not check["passed"]]
    saving = measured / reference["toffoli_count"]
    static_saving = static_delta / reference["max_toffoli_count"]
    multiply_share = REFERENCE_TOFFOLIS_PER_CALL * calls / reference["max_toffoli_count"]
    probe_text = ("none of them writes the qubit's value"
                  if not probes else ", ".join(
                      f"`{name}` wrote the value on {row['bit_ones']}/{row['shots']} "
                      f"shots" for name, row in probes.items()))

    return {
        "experiment": "E -- the official score, its exact decomposition, and why the "
                      "levers still cannot compile",
        "declared_in": "docs/DQI_OFFICIAL_COMPATIBILITY.md §7",
        "question": "D and D2 measured Toffoli counts and hand-checked correctness. "
                    "What score does the competition's own verifier assign, where in "
                    "the reference's gate stream does that saving come from, and can a "
                    "pinned primitive read a qubit into a bit?",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "batch_size": args.batch_size,
                     "log2_delta": OFFICIAL_LOG2_DELTA},
        "budget": {
            "verified_shots": args.shots,
            "official_num_shots": OFFICIAL_NUM_SHOTS,
            "fraction_of_official": args.shots / OFFICIAL_NUM_SHOTS,
            "zero_error_bound_at_this_budget": zero_error,
            "zero_error_bound_at_official_budget": official,
            "smallest_passing_budget": smallest_passing_budget(args.build),
        },
        "build": {key: rows["reference"][key] for key in
                  ("build", "extension", "extension_sha256", "has_mr")},
        "verifier_patch": rows["reference"]["verifier_patch"],
        "verified": {key: row["verified"] for key, row in rows.items()},
        "timings": {key: {"build_seconds": row["build_seconds"],
                          "verify_seconds": row["seconds"]}
                    for key, row in rows.items()},
        "score_ratio": {"score": score_ratio, "toffoli": t_ratio, "qubits": q_ratio,
                        "t_ratio_squared": t_ratio * q_ratio ** 2},
        "decomposition": {
            "multiply_calls": calls,
            "toffolis_per_call": {"reference": REFERENCE_TOFFOLIS_PER_CALL,
                                  "candidate": CANDIDATE_TOFFOLIS_PER_CALL,
                                  "delta": per_call},
            "predicted_delta": predicted,
            "static_delta": static_delta,
            "executed_delta": measured,
            "reference_toffoli_count": reference["toffoli_count"],
            "reference_static_toffoli_count": reference["max_toffoli_count"],
            "saving_fraction_of_executed": saving,
            "saving_fraction_of_static": static_saving,
            "multiply_share_of_static": multiply_share,
            "operation_count_deltas": op_delta,
        },
        "lever_primitives": {"bit_surface": surface, "probes": probes},
        "checks": checks,
        "failed_checks": [check["check"] for check in failed],
        "finding": (
            f"under the competition's own `verify()` at {args.shots:,} shots the "
            f"reference passes with {reference['value_errors']} value and "
            f"{reference['phase_errors']} phase errors and scores "
            f"{reference['score']:.6e} (Q={reference['num_qubits']}, "
            f"T={reference['toffoli_count']:.6f}); the phase-safe 54-AND candidate "
            f"passes with {candidate['value_errors']} and "
            f"{candidate['phase_errors']} and scores {candidate['score']:.6e}, a ratio "
            f"of {score_ratio:.6f}. That is exactly its Toffoli ratio "
            f"({t_ratio:.6f}) at equal Q, because both runs share the same zero-error "
            f"`eps_plus` and therefore the same `R` -- the comparison carries no "
            f"error-rate or qubit penalty in either direction. The saving is the "
            f"multiply's alone: {calls:,} calls x {per_call} = {predicted:,} static "
            f"Toffolis per shot, exactly the measured static delta ({static_delta}); "
            f"executed, the two circuits differ by {measured:.3f} per shot, the rest "
            f"being the classically conditioned Toffolis whose firing is "
            f"shot-dependent. The multiply is {multiply_share:.2%} of the reference's "
            f"static Toffoli stream and the candidate removes {static_saving:.4%} of "
            f"it ({saving:.4%} of the executed mean). The levers' obstacle is "
            f"measured here rather than assumed: of {len(surface)} pinned methods whose "
            f"signature mentions a classical bit, {len(qubit_to_bit)} writes a bit from "
            f"qubit data ({', '.join(sorted(qubit_to_bit)) or 'none'}), and on |1> "
            f"{probe_text}, so nothing in the pinned binary can express a read."
        ),
        "scope": {
            "measured": f"the competition's verifier on both circuits at "
                        f"{args.shots:,} shots (every scoring line unmodified; only "
                        f"`reaction_depth` is stubbed, see verifier_patch): status, "
                        f"value and phase errors, `eps_plus`, `R`, score, Q, static and "
                        f"executed T; the emitted multiply-call count and the full "
                        f"operation-count deltas; and every pinned bit-writing "
                        f"primitive probed on |1>",
            "not_measured": "nothing about the levers' Toffoli counts here -- they "
                            "remain uncompilable, so they have no score; and the "
                            "`reaction_depth` diagnostic, which the verifier computes "
                            "on a 13.9 GB peak and which is not part of the score",
        },
        "reproducibility": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kickmix_sha256": rows["reference"]["extension_sha256"],
            "python": sys.version.split()[0],
            "git_rev": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=False, capture_output=True,
                                      text=True).stdout.strip(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=OFFICIAL_NUM_SHOTS,
                        help="shots to verify; must be a multiple of --batch-size "
                             "(default: the competition's 102,400)")
    parser.add_argument("--batch-size", type=int, default=OFFICIAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--build", type=Path, default=PINNED_DIR)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--role", choices=("both", "reference", "candidate"),
                        default="both", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.shots % args.batch_size:
        raise SystemExit(f"--shots must be a multiple of --batch-size={args.batch_size}")

    if args.role != "both":
        payload = run_role(args, args.role)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        return 0

    rows: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for role in ("reference", "candidate"):
            partial = Path(tmp) / f"{role}.json"
            status = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--role", role,
                 "--out", str(partial), "--shots", str(args.shots),
                 "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                 "--m", str(args.m), "--n", str(args.n),
                 "--degree", str(args.degree), "--build", str(args.build)],
                check=False)
            if status.returncode != 0 or not partial.exists():
                raise SystemExit(f"{role} verification failed "
                                 f"(exit {status.returncode})")
            rows[role] = json.loads(partial.read_text())

    payload = combine(rows, args)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n")

    ratio = payload["score_ratio"]
    decomposition = payload["decomposition"]
    print(f"score ratio {ratio['score']:.6f} (T ratio {ratio['toffoli']:.6f}, "
          f"Q ratio {ratio['qubits']})")
    print(f"decomposition: {decomposition['multiply_calls']:,} calls x "
          f"{decomposition['toffolis_per_call']['delta']} = "
          f"{decomposition['predicted_delta']:,} vs static "
          f"{decomposition['static_delta']:,} and executed "
          f"{decomposition['executed_delta']:.3f}")
    print(f"qubit->bit writers: {sorted(payload['lever_primitives']['probes']) or 'none'}")
    print(f"failed checks: {payload['failed_checks'] or 'none'}")
    print(f"wrote {args.record}")
    return 0 if not payload["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
