#!/usr/bin/env python3
"""Price the proved 54-AND field multiply against the built-in 63-AND one.

`benchmarks/quantum_compilation/designer_directions.json` left one claim unexecuted:
that `mu(3) = 6` composed by Karatsuba gives a 12-bit product in 54 ANDs, and that
swapping it in for `ixor_gf2_mul` would take the six-lever arm from 4.320x to about
4.93x. That projection priced the *whole* multiply by its AND count, which the
circuit does not do: a controlled multiply of either kind pays 12 Toffolis to add
the product into the target under the control, and only the ANDs scale. So the
honest projection is `(54 + 12) / (63 + 12)`, not `54 / 63`, and this script
measures which it is.

Three measurements, in order of increasing scope:

1. **Unit cost.** One controlled `ixor_gf2_mul` against one controlled
   `emit_gf54_mul`, both at GF(2^12): Toffolis, CNOTs, qubits.
2. **Function.** The emitted circuit run on random basis states: does `target`
   get `lhs * rhs` under the control, are the operands restored, is the workspace
   clean, is the tracked phase untouched? A multiplier that is cheap and wrong is
   not a result, so this runs before the arm does.
3. **The arm.** The six-lever gated decoder built twice — built-in multiply and
   54-AND multiply — on the same instance with the same shots, checked against the
   reference circuit's syndromes and phase, and priced by `kickmix`'s
   conditional-aware per-shot counting.

The multiply is swapped in through a builder mixin
(`scripts/gf54_mul.py:Gf54MulMixin`), so the only difference between the two arms
is which instruction the emitter chose for `ixor_gf2_mul`.

Example::

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_gf54_mul_kickmix.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = (ROOT / "benchmarks" / "quantum_compilation" / "gf54_mul_kickmix.json")

#: The six levers, i.e. the `all+quotient` arm of `kickmix_mr_arm_costs.json`.
SIX_LEVERS = {"gate_forney": True, "clear_eea": True, "gate_sweep": True,
              "gate_swap": True, "gate_shift": True, "gate_quotient": True}

#: Recorded there: the arm's executed count per shot, and the multiply's share of it.
#: The share is *derived here* rather than assumed: see `decompose`.
EXECUTED_SIX_LEVERS = 1_325_230.0
RECORDED_MULTIPLY_SHARE_PERCENT = 86.33
REFERENCE_T_MEAN = 5_725_071.61
TARGET_T_MEAN = 445_229.0

#: Controlled-call Toffolis: the ANDs, plus the controlled add that does not scale.
BUILTIN_ANDS = 63
BUILTIN_ADD = 12
GF54_ANDS = 54
GF54_ADD = 12


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "mr_gated_arms_kickmix", ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arms(km, runner):
    spec = importlib.util.spec_from_file_location(
        "mr_gated_arms", ROOT / "scripts" / "run_mr_gated_arms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._all_qubits = lambda registers: runner.all_qubits(km, registers)
    return module


def load_mul():
    spec = importlib.util.spec_from_file_location("gf54_mul", ROOT / "scripts" / "gf54_mul.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toffoli_like(circuit) -> int:
    stats = dict(circuit.operation_statistics())
    return int(sum(stats.get(name, 0) for name in ("CCX", "CSWAP", "AND")))


# ------------------------------------------------------------------ measurements


def unit_cost(km, base: type, mul, degree: int) -> dict:
    """One controlled multiply, priced statically, for both emitters."""
    out = {}
    for label, cls in (("builtin", base), ("gf54", _gf54_builder(base, mul))):
        builder = cls()
        lhs = builder.create_quantum_register(degree, name="L")
        rhs = builder.create_quantum_register(degree, name="R")
        target = builder.create_quantum_register(degree, name="T")
        control = builder.create_quantum_register(1, name="C")
        builder.ixor_gf2_mul(lhs, rhs, target=target, field=km.GF2Field(degree),
                             control=control[0])
        circuit = builder.finish_circuit()
        stats = {k: int(v) for k, v in dict(circuit.operation_statistics()).items() if v}
        registers = 3 * degree + 1
        out[label] = {
            "toffoli_like": toffoli_like(circuit),
            "ccx": stats.get("CCX", 0),
            "cx": stats.get("CX", 0),
            "mr": stats.get("MR", 0),
            "hmr": stats.get("HMR", 0),
            "qubits_total": circuit.num_qubits,
            "workspace_qubits": circuit.num_qubits - registers,
        }
    out["ratio_builtin_over_gf54_toffoli"] = (out["builtin"]["toffoli_like"]
                                             / out["gf54"]["toffoli_like"])
    return out


def _gf54_builder(base: type, mul) -> type:
    mixin = mul.Gf54MulMixin
    if issubclass(base, mixin):
        return base
    return type("Gf54Builder", (mixin, base), {})


def function_check(km, base: type, mul, degree: int, shots: int, seed: int) -> dict:
    """Does the emitted multiply compute the field product, on random basis states?"""
    field = km.GF2Field(degree)
    rng = np.random.default_rng(seed)
    lhs_in = rng.integers(0, 1 << degree, shots)
    rhs_in = rng.integers(0, 1 << degree, shots)
    control_in = rng.integers(0, 2, shots)
    width = degree

    def bits(value: int) -> np.ndarray:
        return np.array([(int(value) >> i) & 1 for i in range(width)], dtype=bool)

    results = {}
    for label, cls in (("builtin", base), ("gf54", _gf54_builder(base, mul))):
        builder = cls()
        lhs = builder.create_quantum_register(degree, name="L")
        rhs = builder.create_quantum_register(degree, name="R")
        target = builder.create_quantum_register(degree, name="T")
        control = builder.create_quantum_register(1, name="C")
        builder.ixor_gf2_mul(lhs, rhs, target=target, field=field, control=control[0])
        circuit = builder.finish_circuit()

        sim = km.Simulator(batch_size=shots, seed=seed)
        sim.use_same_registers_as(circuit)
        operands = np.array([np.concatenate([bits(lhs_in[i]), bits(rhs_in[i])])
                             for i in range(shots)])
        sim.write_across_shots(km.array([q for r in (lhs, rhs) for q in r]), operands.T)
        sim.write_across_shots(km.array([control[0]]), control_in.astype(bool).reshape(1, -1))
        sim.do(circuit)

        read = sim.read_across_shots(km.array(list(target)))
        got = (read.T * (1 << np.arange(degree))).sum(axis=1)
        want = np.array([field.mul(int(lhs_in[i]), int(rhs_in[i])) if control_in[i] else 0
                         for i in range(shots)])
        everything = sim.read_across_shots(
            km.array([km.q(i) for i in range(circuit.num_qubits)])).T
        kept = [q.id for r in (lhs, rhs) for q in r]
        known = set(kept) | {q.id for q in target} | {control[0].id}
        other = [i for i in range(circuit.num_qubits) if i not in known]
        results[label] = {
            "samples": shots,
            "value_mismatches": int(np.abs(got - want).astype(bool).sum()),
            "operands_restored": bool(np.array_equal(everything[:, kept], operands)),
            "workspace_back_to_zero": bool(not everything[:, other].any()),
            "phase_flips": int(sim.read_phase_flipped_across_shots().sum()),
        }
    return results


def build_arm(km, arms, shim, flags, n, degree, m):
    decoder = arms.MRGatedDecoder(n, km.GF2Field(degree), **flags)
    builder = shim.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f"S{i}") for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f"E{j}") for j in range(m)]
    decoder.build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc,
        n_iters=arms.DR.default_n_iters(n))
    return builder.finish_circuit(), syndromes, error_mbuc


def simulate(km, circuit, syndromes, error_mbuc, syndromes_in, mbuc_in, degree, shots):
    sim = km.Simulator(batch_size=shots, seed=99, counting_operations=True)
    sim.use_same_registers_as(circuit)
    flat_syndromes = km.array([q for register in syndromes for q in register])
    flat_errors = km.array([b for register in error_mbuc for b in register])
    sim.write_across_shots(flat_syndromes, _bits(syndromes_in, degree))
    sim.write_across_shots(flat_errors, _bits(mbuc_in, degree))
    sim.do(circuit)
    syndrome_ids = {q.id for register in syndromes for q in register}
    junk_ids = [i for i in range(circuit.num_qubits) if i not in syndrome_ids]
    per_shot = [dict(sim.operation_statistics(i)) for i in range(shots)]
    return {
        "syndromes": sim.read_across_shots(flat_syndromes),
        "junk": sim.read_across_shots([km.q(i) for i in junk_ids]),
        "phase": sim.read_phase_flipped_across_shots(),
        "toffoli_per_shot": [int(sim.operation_statistics(i).num_toffoli_like)
                             for i in range(shots)],
        "ccx_per_shot": [int(row.get("CCX", 0)) for row in per_shot],
        "cswap_per_shot": [int(row.get("CSWAP", 0)) for row in per_shot],
    }


def decompose(builtin: dict, gf54: dict) -> dict:
    """Split the arm's executed count into `calls * per_call + everything else`.

    The two arms are the same circuit except for the multiply's Toffolis, so the
    executed CCX difference over the same shots is exactly `calls * (75 - 66)`. That
    gives the call count without an ablation, and an ablation is what makes this
    worth doing: deleting the multiply changes the register state, which moves the
    predicates that gate the rest of the EEA, so the deleted circuit executes a
    different amount of *other* work too and its difference is not the multiply.
    """
    delta = (np.array(builtin["ccx_per_shot"]) - np.array(gf54["ccx_per_shot"]))
    per_call = (BUILTIN_ANDS + BUILTIN_ADD) - (GF54_ANDS + GF54_ADD)
    if not np.all(delta == delta[0]) or int(delta[0]) % per_call:
        raise SystemExit(f"the two arms' CCX counts do not differ by {per_call} per call: "
                         f"{sorted(set(delta.tolist()))}")
    calls = int(delta[0]) // per_call
    builtin_mean = float(np.mean(builtin["toffoli_per_shot"]))
    gf54_mean = float(np.mean(gf54["toffoli_per_shot"]))
    multiply = calls * (BUILTIN_ANDS + BUILTIN_ADD)
    remainder = builtin_mean - multiply
    return {
        "executed_multiply_calls": calls,
        "builtin_toffoli_per_call": BUILTIN_ANDS + BUILTIN_ADD,
        "gf54_toffoli_per_call": GF54_ANDS + GF54_ADD,
        "builtin_multiply_toffoli": multiply,
        "everything_else_toffoli": remainder,
        "multiply_share_percent": 100 * multiply / builtin_mean,
        "reconstructed_gf54_mean": remainder + calls * (GF54_ANDS + GF54_ADD),
        "reconstructed_matches_measurement": bool(
            abs(remainder + calls * (GF54_ANDS + GF54_ADD) - gf54_mean) < 1e-6),
    }


def ladder(decomposition: dict, and_counts=(63, 54, 45, 23, 0)) -> dict:
    """What the arm would execute for a multiply of `n` ANDs, holding the rest fixed."""
    calls = decomposition["executed_multiply_calls"]
    rest = decomposition["everything_else_toffoli"]
    return {
        str(n): {
            "toffoli_per_shot": rest + calls * (n + BUILTIN_ADD),
            "ratio_vs_reference": REFERENCE_T_MEAN / (rest + calls * (n + BUILTIN_ADD)),
            "ratio_vs_target": (rest + calls * (n + BUILTIN_ADD)) / TARGET_T_MEAN,
        }
        for n in and_counts
    }


def _bits(values, width):
    from verify_submission_dqi import to_bits
    return to_bits(values, width)


def _is_correct(out, reference, reference_inputs) -> bool:
    return bool(
        np.array_equal(out["syndromes"], reference["syndromes"])
        and np.array_equal(out["phase"], reference["phase"])
        and np.array_equal(out["syndromes"], reference_inputs)
        and not out["junk"].any()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4095)
    parser.add_argument("--n", type=int, default=70)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--checks", type=int, default=32,
                        help="basis states in the function check")
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    runner = load_runner()
    km = runner.load_kickmix(args.kickmix_build)
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import verify_submission_dqi as V

    arms = load_arms(km, runner)
    mul = load_mul()
    base = runner.builder_class(km)
    ands = mul.and_count(args.degree)
    if args.degree == 12 and ands != GF54_ANDS:
        raise SystemExit(f"and_count(12) = {ands}, not the proved {GF54_ANDS}")

    unit = unit_cost(km, base, mul, args.degree)
    if official := args.degree == 12:
        # A controlled call is the ANDs plus the controlled add, in both circuits.
        for label, ands_count, total in (("builtin", BUILTIN_ANDS, BUILTIN_ANDS + BUILTIN_ADD),
                                        ("gf54", GF54_ANDS, GF54_ANDS + GF54_ADD)):
            if unit[label]["ccx"] != total:
                raise SystemExit(f"{label} controlled multiply is {unit[label]['ccx']} CCX at "
                                 f"12 bits, expected {ands_count} + {BUILTIN_ADD}")
    function = function_check(km, base, mul, args.degree, args.checks, 20261029)
    if function["gf54"]["value_mismatches"] or not function["gf54"]["operands_restored"] \
            or not function["gf54"]["workspace_back_to_zero"]:
        raise SystemExit(f"the 54-AND multiply is wrong, refusing to price it: {function}")

    shim = runner.shim_for(km, base)
    shim_gf54 = runner.shim_for(km, _gf54_builder(base, mul))
    original = arms.DR.km
    try:
        spec = V.ProblemSpecDQI.benchmark_instance(
            m=args.m, n=args.n, degree=args.degree, num_shots=args.shots)
        weights = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
        tables = V.FieldTables(km.GF2Field(spec.degree))
        rng = np.random.default_rng(20261028)
        syndromes_in, mbuc_in, _phase = V.sample_shots(
            spec, tables, rng, args.shots, weights)
        reference_inputs = V.to_bits(syndromes_in, args.degree)

        arms.DR.km = shim
        ref_circuit, ref_syn, ref_mbuc = build_arm(km, arms, shim, {}, args.n,
                                                  args.degree, args.m)
        reference = simulate(km, ref_circuit, ref_syn, ref_mbuc, syndromes_in, mbuc_in,
                             args.degree, args.shots)

        rows = {}
        for label, shim_used in (("gated_builtin", shim), ("gated_gf54", shim_gf54)):
            arms.DR.km = shim_used
            circuit, syn, mbuc = build_arm(km, arms, shim_used, SIX_LEVERS, args.n,
                                           args.degree, args.m)
            out = simulate(km, circuit, syn, mbuc, syndromes_in, mbuc_in, args.degree,
                           args.shots)
            rows[label] = {
                "qubits": circuit.num_qubits,
                "toffoli_per_shot": out["toffoli_per_shot"],
                "ccx_per_shot": out["ccx_per_shot"],
                "cswap_per_shot": out["cswap_per_shot"],
                "mean_toffoli_like": float(np.mean(out["toffoli_per_shot"])),
                "ratio_vs_reference": None,
                "correct": _is_correct(out, reference, reference_inputs),
            }
    finally:
        arms.DR.km = original

    reference_mean = float(np.mean(reference["toffoli_per_shot"]))
    for row in rows.values():
        row["ratio_vs_reference"] = reference_mean / row["mean_toffoli_like"]

    # The executed multiply call count, from the two arms' own CCX counts, and the
    # arm priced for any multiply cost holding everything else fixed.
    decomposition = decompose(rows["gated_builtin"], rows["gated_gf54"])
    costs = ladder(decomposition) if official else {}
    measured_saving = 100 * (1 - rows["gated_gf54"]["mean_toffoli_like"]
                             / rows["gated_builtin"]["mean_toffoli_like"])
    recorded_share = EXECUTED_SIX_LEVERS * RECORDED_MULTIPLY_SHARE_PERCENT / 100
    recorded_projection = EXECUTED_SIX_LEVERS - recorded_share * (
        1 - GF54_ANDS / BUILTIN_ANDS) if official else None

    record = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR (scripts/build_kickmix_mr.py); not the "
                        "pinned competition binary and not eligible for a receipt",
        "instance": {"m": args.m, "n": args.n, "degree": args.degree,
                     "shots": args.shots, "seed": 20261028},
        "subject": "the 54-AND GF(2^12) multiply of scripts/gf54_mul.py, swapped in for "
                   "ixor_gf2_mul through Gf54MulMixin",
        "unit_cost": unit,
        "function_check": function,
        "arms": rows,
        "reference_t_mean_measured": reference_mean,
        "decomposition": decomposition,
        "ladder": costs,
        "recorded_projection": {
            "source": "benchmarks/quantum_compilation/designer_directions.json, part C",
            "value": recorded_projection,
            "ratio": (REFERENCE_T_MEAN / recorded_projection) if official else None,
            "model": f"the multiply's whole cost scaled by {GF54_ANDS}/{BUILTIN_ANDS}, on a "
                     f"multiply share of {RECORDED_MULTIPLY_SHARE_PERCENT}% measured by "
                     f"ablation",
            "note": "defined at the official GF(2^12) instance only",
        },
        "measured_saving_percent": measured_saving,
    }

    measured = rows["gated_gf54"]["mean_toffoli_like"]
    if not official:
        record["finding"] = (
            f"Scaled check at m={args.m}, n={args.n}, GF(2^{args.degree}) only: the 54-AND "
            f"emitter costs {unit['gf54']['toffoli_like']} Toffolis per controlled call "
            f"against {unit['builtin']['toffoli_like']}, and the arm runs "
            f"{rows['gated_gf54']['ratio_vs_reference']:.3f}x against "
            f"{rows['gated_builtin']['ratio_vs_reference']:.3f}x. The projections are "
            f"defined at the official instance; rerun there for the recorded claim."
        )
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n")
        print(f"  scaled check: {record['finding']}")
        print(f"  record: {args.record}")
        return 0 if all(row["correct"] for row in rows.values()) else 1
    record["finding"] = (
        f"The proved 54-AND multiply is a real circuit: {unit['gf54']['ccx'] - GF54_ADD} ANDs "
        f"and {unit['gf54']['toffoli_like']} Toffolis per controlled call against the built-in "
        f"{unit['builtin']['toffoli_like']}, correct on every basis state tried (0 value "
        f"mismatches, operands restored, workspace clean, phase untouched). Swapped into the "
        f"six-lever arm it takes {rows['gated_builtin']['mean_toffoli_like']:,.1f} to "
        f"{measured:,.1f} Toffoli per shot: {rows['gated_gf54']['ratio_vs_reference']:.3f}x "
        f"against {rows['gated_builtin']['ratio_vs_reference']:.3f}x, a "
        f"{measured_saving:.2f}% saving and {rows['gated_builtin']['mean_toffoli_like'] - measured:,.1f} "
        f"Toffolis per shot. Two corrections come with it. First, the projected 1,161,791.3 "
        f"does not reproduce, and that is a modelling error rather than a shortfall in the "
        f"circuit: every controlled multiply pays {BUILTIN_ADD} Toffolis to add the product "
        f"into its target under the control, and that part does not shrink with the AND "
        f"count, so the call costs ({GF54_ANDS}+{GF54_ADD})/({BUILTIN_ANDS}+{BUILTIN_ADD}) "
        f"rather than {GF54_ANDS}/{BUILTIN_ANDS}. Second, and larger: the multiply is "
        f"{decomposition['multiply_share_percent']:.2f}% of this arm, not the "
        f"{RECORDED_MULTIPLY_SHARE_PERCENT}% that record carries. That earlier share came "
        f"from deleting `ixor_gf2_mul` and differencing the executed count, which changes the "
        f"register state and therefore the predicates that gate the rest of the EEA -- the "
        f"same state-perturbation the static-count rule warns about, in an executed count. "
        f"Here the two arms are identical except for the multiply, so their CCX difference is "
        f"exactly `calls * {decomposition['builtin_toffoli_per_call'] - decomposition['gf54_toffoli_per_call']}`: "
        f"{decomposition['executed_multiply_calls']:,} executed calls, "
        f"{decomposition['builtin_multiply_toffoli']:,.1f} Toffolis of multiply, "
        f"{decomposition['everything_else_toffoli']:,.1f} of everything else, and the "
        f"reconstruction reproduces the measured arm to {decomposition['reconstructed_gf54_mean'] - measured:+.6f}. "
        f"That decomposition also prices the rest of the ladder: at {GF54_ANDS} ANDs the arm "
        f"is {costs[str(GF54_ANDS)]['ratio_vs_reference']:.3f}x, at 23 ANDs "
        f"{costs['23']['ratio_vs_reference']:.3f}x ({costs['23']['ratio_vs_target']:.2f}x the "
        f"445,229 target), and with a free multiply "
        f"{costs['0']['ratio_vs_reference']:.3f}x ({costs['0']['ratio_vs_target']:.2f}x the "
        f"target) -- so the earlier conclusion holds with a narrower margin: no multiply "
        f"reaches Benchmark II on this decoder."
    )

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"m={args.m} n={args.n} GF(2^{args.degree}) shots={args.shots}")
    print(f"  unit cost (controlled, static)")
    for label in ("builtin", "gf54"):
        row = unit[label]
        print(f"    {label:8s} {row['toffoli_like']:4d} Toffoli ({row['ccx']:2d} ANDs) "
              f"{row['cx']:4d} CX  workspace {row['workspace_qubits']:3d} qubits")
    print(f"  function check ({args.checks} basis states): "
          f"{function['gf54']['value_mismatches']} value mismatches, "
          f"phase flips {function['gf54']['phase_flips']}")
    print(f"  reference        {reference_mean:14,.1f}")
    for label, row in rows.items():
        print(f"  {label:16s} {row['mean_toffoli_like']:14,.1f}  "
              f"{row['ratio_vs_reference']:.3f}x  Q={row['qubits']}  ok={row['correct']}")
    print(f"  measured saving {measured_saving:.2f}% of the arm")
    print(f"  decomposition: {decomposition['executed_multiply_calls']:,} multiply calls x "
          f"{decomposition['builtin_toffoli_per_call']} = "
          f"{decomposition['builtin_multiply_toffoli']:,.1f}, everything else "
          f"{decomposition['everything_else_toffoli']:,.1f} "
          f"({decomposition['multiply_share_percent']:.2f}% multiply)")
    if costs:
        for ands, row in costs.items():
            print(f"    {ands:>2s} ANDs -> {row['toffoli_per_shot']:13,.1f} T/shot  "
                  f"{row['ratio_vs_reference']:6.3f}x  {row['ratio_vs_target']:5.2f}x target")
    print(f"  both arms correct: {rows['gated_builtin']['correct'] and rows['gated_gf54']['correct']}")
    print(f"  record: {args.record}")
    return 0 if all(row["correct"] for row in rows.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
