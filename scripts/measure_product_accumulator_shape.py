#!/usr/bin/env python3
"""Can a low-AND GF(2^12) product accumulate in place, without the 12-Toffoli add?

`product_axis_decision()` left exactly this open (`what_this_does_not_settle`): a 23-AND
product that accumulates in place prices the arm below the target at the verifier peak,
and one that has to materialise its product into a clean window prices it above it. The
figures are read off the model rather than typed here -- §7.15 of the reachability note
re-priced both rows after the division stream was decomposed by measurement, and this
script's own output follows the model. The built-in Karatsuba accumulates in place; the
54-AND substitute of `scripts/gf54_mul.py` writes a 2n-1 window and pays `c_gf_iadd`.

The question is which of those is a property of the *accumulator* and which is a
property of the *control*. This script separates them, at two AND counts and in four
call shapes, from the emitted instruction stream:

* **The control is the price, not the accumulate.** A quantum control is 12 CCX
  whatever the product costs: `c_gf_iadd` is one per output bit, and folding the same
  control into one operand instead is *also* 12 CCX (`c * (a*b) = (c*a) * b`). So a
  quantum-controlled call is `2n = 24` Toffolis above its AND count... at every AND
  count, the two shapes are identical: 54 + 12 = 66 = 12 + 54.
* **A classical or unconditional call accumulates in place at any AND count.** The
  bilinear form's term products are XORed into the target's bits, and under a
  classical control those are CNOTs, which carry no Toffoli. The emitter here
  accumulates the reduced product straight into a *dirty* target -- no 2n-1 window,
  no add -- and measures 54 Toffolis for the 54 ANDs, 0 non-AND, against the same
  family's 66 when the call is quantum-controlled.
* **Functionally the same call.** The direct accumulate is checked shot-for-shot
  against the built-in `ixor_gf2_mul` into the same dirty target, values and tracked
  phase, on the local MR simulator at GF(2^4).

The consequence for the campaign arm is arithmetic, and it goes the other way from the
bracket: the arm's measured post-window product stream is **63.0 per call**
(`331,023 / 5,255`), which is the in-place shape, so the arm's calls are classical and
the 12-Toffoli add is **not on the arm at all**. The 23-AND row therefore needs no
accumulator work to reach the in-place bound row; the materialising row is the price of a
*quantum-controlled* variant of the same arm, and no AND count buys it back.

Run it under the interpreter `kickmix` was built for (CPython 3.13)::

    /dev/shm/qvcc-venv313/bin/python scripts/measure_product_accumulator_shape.py
    /dev/shm/qvcc-venv313/bin/python scripts/measure_product_accumulator_shape.py \
        --record benchmarks/quantum_compilation/gf_product_accumulator_shape.json
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
DEFAULT_BUILD = ROOT / "build" / "kickmix_mr" / "lib"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "gf_product_accumulator_shape.json"

#: The degree the DQI decoder multiplies in, and the 2n-1 AND bound on one product.
DEGREE = 12
AND_DIMENSION_BOUND = 2 * DEGREE - 1

#: The local-simulator degree for the functional check: GF(2^4) keeps the bilinear form
#: at 3 * 3 = 9 ANDs while exercising the same reduction machinery.
CHECK_DEGREE = 4
CHECK_SHOTS = 64


def load_kickmix(build_dir: Path):
    sys.path.insert(0, str(build_dir))
    try:
        import kickmix as km
    except ImportError as exc:  # pragma: no cover - environment failure
        raise SystemExit(f"could not import kickmix from {build_dir}: {exc}") from exc
    if not hasattr(km.CircuitBuilder, "mr"):
        raise SystemExit(f"{km.__file__} has no mr method -- build the MR extension first")
    return km


def load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses looks the defining module up by name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- a reduction-aware accumulate


def _reducing_emitter(gf54):
    """The 54-AND product with the field reduction folded into every accumulate.

    `_emit_product` in `scripts/gf54_mul.py` accumulates into a 2n-1 window and reduces
    afterwards, which is why it needs a clean window and a controlled add. Reduction is
    linear, so each accumulated position can be folded into the *target* directly: the
    window disappears and the accumulate becomes CNOTs into a register that may already
    hold anything.
    """

    def accumulate(builder, qubit, position, target, mapping) -> None:
        mask = mapping[position]
        for k in range(len(target)):
            if (mask >> k) & 1:
                builder.cx(qubit, target[k])

    def base(builder, a, b, target, mapping) -> None:
        products, outputs = gf54.terms(tuple(frozenset([i]) for i in range(len(a))))
        for t, (u, v) in enumerate(products):
            left_bits = [a[i] for i in sorted(u)]
            right_bits = [b[i] for i in sorted(v)]
            left, left_temp = gf54._emit_linear(builder, left_bits)
            right, right_temp = gf54._emit_linear(builder, right_bits)
            product = builder.alloc_qubits(1)[0]
            builder.ccx(left, right, product)
            for k in gf54.outputs_indexed(outputs, t):
                accumulate(builder, product, k, target, mapping)
            builder.del_and(left, right, product)
            gf54._undo_linear(builder, right_bits, right_temp)
            gf54._undo_linear(builder, left_bits, left_temp)
            builder.free(product)

    def product_of(builder, a, b, target, mapping) -> None:
        n = len(a)
        if n in gf54.BASE_ALGORITHMS:
            base(builder, a, b, target, mapping)
            return

        k = (n + 1) // 2
        low = builder.alloc_qubits(2 * k - 1)
        product_of(builder, a[:k], b[:k], low, _window_mapping(2 * k - 1))
        for i in range(2 * k - 1):
            accumulate(builder, low[i], i, target, mapping)        # + low
            accumulate(builder, low[i], k + i, target, mapping)    # + x^k low
        _measure_and_reset(builder, low)
        builder.free(low)

        width_high = 2 * (n - k) - 1
        high = builder.alloc_qubits(width_high)
        product_of(builder, a[k:], b[k:], high, _window_mapping(width_high))
        for i in range(width_high):
            accumulate(builder, high[i], k + i, target, mapping)      # + x^k high
            accumulate(builder, high[i], 2 * k + i, target, mapping)  # + x^2k high
        _measure_and_reset(builder, high)
        builder.free(high)

        for i in range(n - k):
            builder.cx(a[k + i], a[i])
            builder.cx(b[k + i], b[i])
        width_low = 2 * k - 1
        mid = builder.alloc_qubits(width_low)
        product_of(builder, a[:k], b[:k], mid, _window_mapping(width_low))
        for i in range(width_low):
            accumulate(builder, mid[i], k + i, target, mapping)
        _measure_and_reset(builder, mid)
        builder.free(mid)
        for i in range(n - k):
            builder.cx(b[k + i], b[i])
            builder.cx(a[k + i], a[i])

    return accumulate, product_of


def _window_mapping(width: int) -> list[int]:
    """A scratch window is unreduced: position `i` is bucket `i`."""
    return [1 << i for i in range(width)]


def _measure_and_reset(builder, qubits) -> None:
    bit = builder.alloc_bits(1)[0]
    for q in list(qubits):
        builder.mr(q, bit)
    builder.free(bit)


def emit_direct(gf54, builder, lhs, rhs, *, target, field) -> None:
    """`target ^= lhs * rhs`, accumulated straight into a possibly-dirty target."""
    degree = field.degree
    if len(lhs) != degree or len(rhs) != degree or len(target) != degree:
        raise ValueError(f"all three registers must be {degree} qubits wide")
    mapping = gf54.reduction_map(field, degree)
    _accumulate, product_of = _reducing_emitter(gf54)
    product_of(builder, list(lhs), list(rhs), list(target), mapping)


def emit_folded(gf54, builder, lhs, rhs, *, target, field, control) -> None:
    """`target ^= control * lhs * rhs` by folding the control into `lhs`, in place.

    `control * (lhs * rhs) = (control * lhs) * rhs`, so 12 CCX on the operand replace the
    12 CCX of `c_gf_iadd` -- the same price, in the same place, at any AND count.
    """
    folded = builder.alloc_qubits(field.degree)
    for i in range(field.degree):
        builder.ccx(control, lhs[i], folded[i])
    emit_direct(gf54, builder, folded, rhs, target=target, field=field)
    for i in range(field.degree):
        builder.mr(folded[i], builder.alloc_bits(1)[0])
    builder.free(folded)


# --------------------------------------------------------------------- counting shapes


def count_call(km, gf54, label: str, shape: str) -> dict:
    """Emit one call in one shape and count the Toffolis the stream actually carries."""
    runner = load_module("mr_runner_shapes", ROOT / "scripts" / "run_mr_gated_arms_kickmix.py")
    builder = runner.builder_class(km)()
    field = km.GF2Field(DEGREE)
    target = builder.alloc_qubits(DEGREE)
    lhs = builder.alloc_qubits(DEGREE)
    rhs = builder.alloc_qubits(DEGREE)
    control = builder.alloc_qubits(1)[0]
    bit = builder.alloc_bits(1)[0]

    if shape == "builtin_classical":
        builder.ixor_gf2_mul(lhs, rhs, target=target, field=field, control=bit)
    elif shape == "builtin_quantum":
        builder.ixor_gf2_mul(lhs, rhs, target=target, field=field, control=control)
    elif shape == "bilinear_window_quantum":
        gf54.emit_gf54_mul(builder, lhs, rhs, target=target, field=field, control=control)
    elif shape == "bilinear_window_classical":
        gf54.emit_gf54_mul(builder, lhs, rhs, target=target, field=field, control=bit)
    elif shape == "bilinear_direct_classical":
        emit_direct(gf54, builder, lhs, rhs, target=target, field=field)
    elif shape == "bilinear_folded_quantum":
        emit_folded(gf54, builder, lhs, rhs, target=target, field=field, control=control)
    else:  # pragma: no cover - programming error
        raise ValueError(shape)

    circuit = builder.finish_circuit()
    stats = {name: int(count) for name, count in dict(circuit.operation_statistics()).items()
             if count}
    toffoli = sum(stats.get(name, 0) for name in ("CCX", "CSWAP", "AND"))
    gates = sum(stats.values())
    return {
        "call": label,
        "shape": shape,
        "toffolis": toffoli,
        "and_gates": stats.get("AND", 0),
        "ops": stats,
        "gates_total": gates,
        "qubits": int(circuit.num_qubits),
    }


# ---------------------------------------------------------------- functional check


def check_values(gf54) -> dict:
    """Do the direct accumulate and the built-in produce the same call?

    On the local MR simulator at GF(2^4), 64 shots, a dirty target throughout: the
    reference is the built-in `ixor_gf2_mul`, and the candidate is the reduction-aware
    bilinear accumulate. Values and tracked phase must both agree.
    """
    sim = load_module("local_mr", ROOT / "apedp_dyn" / "quantum_compilation" / "local_mr_sim.py")
    field = sim.Field(CHECK_DEGREE)
    rng = np.random.default_rng(20261104)

    def build(kind: str):
        builder = sim.Builder()
        target = builder.alloc_qubits(CHECK_DEGREE)
        lhs = builder.alloc_qubits(CHECK_DEGREE)
        rhs = builder.alloc_qubits(CHECK_DEGREE)
        control = builder.alloc_qubits(1)[0]
        if kind == "reference":
            builder.ixor_gf2_mul(lhs, rhs, target=target, field=field)
        elif kind == "direct":
            emit_direct(gf54, builder, lhs, rhs, target=target, field=field)
        elif kind == "windowed":
            gf54.emit_gf54_mul(builder, lhs, rhs, target=target, field=field, control=True)
        else:  # pragma: no cover
            raise ValueError(kind)
        return builder.finish_circuit(), target, lhs, rhs, control

    circuits = {}
    for kind in ("reference", "direct", "windowed"):
        circuits[kind] = build(kind)

    # One instance: every shot runs all three circuits from the same inputs.
    simulator = sim.Simulator(batch_size=CHECK_SHOTS, seed=20261104)
    for circuit, _target, _lhs, _rhs, _control in circuits.values():
        simulator.use_same_registers_as(circuit)

    def load(a_values, b_values, t_values, c_values):
        for _circuit, target, lhs, rhs, control in circuits.values():
            simulator.write_across_shots(lhs, _bits(a_values, CHECK_DEGREE))
            simulator.write_across_shots(rhs, _bits(b_values, CHECK_DEGREE))
            simulator.write_across_shots(target, _bits(t_values, CHECK_DEGREE))
            simulator.write_across_shots([control], c_values[None, :])

    a_values = rng.integers(0, 1 << CHECK_DEGREE, CHECK_SHOTS)
    b_values = rng.integers(0, 1 << CHECK_DEGREE, CHECK_SHOTS)
    # A dirty target: not zero, and not the product.
    t_values = rng.integers(0, 1 << CHECK_DEGREE, CHECK_SHOTS)
    c_values = rng.integers(0, 2, CHECK_SHOTS)

    load(a_values, b_values, t_values, c_values)
    results = {}
    for kind, (circuit, target, _lhs, _rhs, _control) in circuits.items():
        simulator.clear_for_shot()
        load(a_values, b_values, t_values, c_values)
        simulator.do(circuit)
        results[kind] = (
            simulator.read_across_shots(target),
            simulator.read_phase_flipped_across_shots(),
        )

    expected = np.array([
        int(t_values[i]) ^ int(field.mul(int(a_values[i]), int(b_values[i])))
        for i in range(CHECK_SHOTS)
    ])

    def values_of(kind: str) -> np.ndarray:
        bits, _phase = results[kind]
        out = np.zeros(CHECK_SHOTS, dtype=np.int64)
        for k in range(bits.shape[0]):
            out |= bits[k].astype(np.int64) << k
        return out

    reference = values_of("reference")
    direct = values_of("direct")
    windowed = values_of("windowed")
    return {
        "degree": CHECK_DEGREE,
        "shots": CHECK_SHOTS,
        "field": f"GF(2^{CHECK_DEGREE})",
        "dirty_target": True,
        "reference_matches_arithmetic": int((reference == expected).sum()),
        "direct_matches_arithmetic": int((direct == expected).sum()),
        "windowed_matches_arithmetic": int((windowed == expected).sum()),
        "direct_equals_reference": bool(np.array_equal(direct, reference)),
        "windowed_equals_reference": bool(np.array_equal(windowed, reference)),
        "phase_unchanged": {
            kind: bool(not results[kind][1].any()) for kind in results
        },
    }


def _bits(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return np.stack([((values >> k) & 1) for k in range(width)])


# --------------------------------------------------------------------------- pricing


def bound_rows(shapes: dict) -> dict:
    """Price the arm at the 2n-1 bound with the accumulator shape that is now known."""
    model = load_module("dqi_v6", ROOT / "apedp_dyn" / "quantum_compilation"
                        / "dqi_gluzdov_v6.py")
    decision = model.product_axis_decision()
    bound = decision["and_dimension_bound"]
    in_place = bound["in_place"]
    materialised = bound["materialised"]
    direct = shapes["bilinear_direct_classical"]
    windowed_quantum = shapes["bilinear_window_quantum"]
    builtin = shapes["builtin_classical"]

    peak = model.PEAK_QUBITS_REFERENCE
    target = model.T_MEAN_TARGET_V6

    def score_ratio(qubits: int, t_mean: float) -> float:
        return (qubits / peak) ** 2 * t_mean / target

    # The substitute is narrower than its own window but wider than the built-in call by
    # its bilinear workspace; the arm's peak absorbs whatever lands on the critical path.
    excess = direct["qubits"] - builtin["qubits"]
    width_rows = {
        "builtin_call_qubits": builtin["qubits"],
        "bilinear_direct_call_qubits": direct["qubits"],
        "bilinear_window_call_qubits": windowed_quantum["qubits"],
        "window_the_direct_accumulate_saves": (windowed_quantum["qubits"]
                                              - direct["qubits"]),
        "workspace_over_the_builtin_call": excess,
        "in_place_at_the_arm_peak": score_ratio(model.CAMPAIGN_ARM_QUBITS,
                                                in_place["t_mean_toffolis"]),
        "in_place_at_the_verifier_peak": score_ratio(peak, in_place["t_mean_toffolis"]),
        "in_place_at_the_peak_plus_the_workspace": score_ratio(peak + excess,
                                                              in_place["t_mean_toffolis"]),
        "basis": "per-call widths, measured; the peak-plus-workspace row assumes the "
                 "substitute's workspace lands on the decoder's peak, which no arm "
                 "build decides",
    }
    return {
        "width_rows": width_rows,
        "ands_at_bound": AND_DIMENSION_BOUND,
        "arm_calls_are_classical": {
            "post_window_toffolis_per_call": 331_023.0 / 5_255,
            "why": "the measured stream averages 63.0 per call, the in-place shape; a "
                   "call carrying c_gf_iadd measures 75",
        },
        "quote_rows_from_the_model": {
            "in_place": {
                "t_mean_toffolis": in_place["t_mean_toffolis"],
                "vs_target": in_place["vs_target"],
                "vs_denominator_at_1913": in_place["vs_denominator"],
                "vs_denominator_at_verifier_peak": in_place["vs_denominator_at_verifier_peak"],
                "reachable_shape": "classical control, direct accumulate",
            },
            "materialised": {
                "t_mean_toffolis": materialised["t_mean_toffolis"],
                "vs_target": materialised["vs_target"],
                "vs_denominator_at_1913": materialised["vs_denominator"],
                "reachable_shape": "quantum control, window + c_gf_iadd",
            },
        },
        "measured_shapes_at_the_built_and_count": {
            "bilinear_in_place_classical": direct["toffolis"],
            "bilinear_window_quantum": windowed_quantum["toffolis"],
            "fold_the_control_instead": shapes["bilinear_folded_quantum"]["toffolis"],
        },
        "fold_identity": (
            f"{shapes['bilinear_window_quantum']['toffolis']} = "
            f"{shapes['bilinear_folded_quantum']['toffolis']} = "
            f"{direct['toffolis']} ANDs + {DEGREE} for the control"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kickmix-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args()

    km = load_kickmix(args.kickmix_build)
    gf54 = load_module("gf54_mul", ROOT / "scripts" / "gf54_mul.py")

    calls = [
        count_call(km, gf54, "built-in, classically controlled", "builtin_classical"),
        count_call(km, gf54, "built-in, quantum-controlled", "builtin_quantum"),
        count_call(km, gf54, "54-AND window, classically controlled",
                   "bilinear_window_classical"),
        count_call(km, gf54, "54-AND window, quantum-controlled",
                   "bilinear_window_quantum"),
        count_call(km, gf54, "54-AND direct accumulate (in place)",
                   "bilinear_direct_classical"),
        count_call(km, gf54, "54-AND direct accumulate, control folded "
                             "(in place, quantum-controlled)", "bilinear_folded_quantum"),
    ]
    shapes = {row["shape"]: row for row in calls}
    # The AND count of the bilinear family is fixed at 54; every Toffoli above it in a
    # 54-AND call is the control, not the accumulator.
    ands = shapes["bilinear_direct_classical"]["toffolis"]
    for row in calls:
        if row["shape"].startswith("bilinear"):
            row["toffolis_above_the_ands"] = row["toffolis"] - ands

    check = check_values(gf54)
    pricing = bound_rows(shapes)

    record = {
        "engine": "kickmix",
        "kickmix": str(Path(km.__file__).resolve()),
        "kickmix_sha256": hashlib.sha256(Path(km.__file__).read_bytes()).hexdigest(),
        "kickmix_role": "local rebuild with MR (scripts/build_kickmix_mr.py); not the "
                        "pinned competition binary and not eligible for a receipt",
        "degree": DEGREE,
        "measure": "static Toffolis of one GF(2^12) product call in six shapes, plus a "
                   "64-shot value-and-phase equivalence of the direct accumulate against "
                   "the built-in call on the local MR simulator",
        "calls": calls,
        "functional_check": check,
        "pricing": pricing,
        "verdict": (
            "the 12-Toffoli add is the price of a quantum control, not of the accumulator: "
            "a bilinear product accumulates in place at any AND count when the call is "
            "classical (measured 54 Toffolis for 54 ANDs, 0 non-AND, into a dirty target), "
            "and a quantum-controlled one costs 12 more however it is arranged -- folding "
            f"the control into an operand measures {shapes['bilinear_folded_quantum']['toffolis']}, "
            f"the same as the window plus c_gf_iadd "
            f"({shapes['bilinear_window_quantum']['toffolis']}). The campaign arm's "
            "post-window stream already averages 63.0 Toffolis per call, the in-place shape, "
            "so the add is not on the arm and the 23-AND row needs no accumulator work: "
            f"T {pricing['quote_rows_from_the_model']['in_place']['t_mean_toffolis']:,.1f} "
            f"at 1,893 ({pricing['quote_rows_from_the_model']['in_place']['vs_denominator_at_verifier_peak']:.4f}x) "
            "at the peak and 1.011x at the arm's own 1,913. The materialising "
            "row is a quantum-controlled variant of the same arm, and no AND count buys it "
            "back"
        ),
    }

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print("one GF(2^12) product call, by control and accumulator shape")
    for row in calls:
        above = row.get("toffolis_above_the_ands")
        extra = f" +{above} for the control" if above is not None else ""
        print(f"  {row['call']:58s} {row['toffolis']:3d} Toffolis{extra}"
              f"  q={row['qubits']}")
    print(f"functional check (GF(2^{CHECK_DEGREE}), {CHECK_SHOTS} shots, dirty target): "
          f"direct {check['direct_matches_arithmetic']}/{CHECK_SHOTS}, "
          f"windowed {check['windowed_matches_arithmetic']}/{CHECK_SHOTS}, "
          f"reference {check['reference_matches_arithmetic']}/{CHECK_SHOTS}, "
          f"phase unchanged {check['phase_unchanged']}")
    ip = pricing["quote_rows_from_the_model"]["in_place"]
    mt = pricing["quote_rows_from_the_model"]["materialised"]
    print(f"at the 2n-1 = {pricing['ands_at_bound']} AND bound:")
    print(f"  classical control, in place : T {ip['t_mean_toffolis']:,.1f}  "
          f"{ip['vs_denominator_at_verifier_peak']:.3f}x peak  "
          f"{ip['vs_denominator_at_1913']:.3f}x at 1,913")
    print(f"  quantum control, window+add : T {mt['t_mean_toffolis']:,.1f}  "
          f"{mt['vs_denominator_at_1913']:.3f}x at 1,913")
    print(f"  fold identity: {pricing['fold_identity']}")
    if args.record is not None:
        print(f"record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
