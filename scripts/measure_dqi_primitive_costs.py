#!/usr/bin/env python3
"""Measure where Benchmark II's Toffoli budget actually sits, primitive by primitive.

Why this exists: the DQI target row claims a 12.86x cut in T_mean and 104 fewer
qubits. Neither is checkable from the projection, and kickmix exposes no
per-stage accounting, so this script recovers the cost structure by measurement:

  1. Per-call Toffoli cost of each field primitive, by differencing k calls.
  2. Per-iteration (EEA) and per-position (Chien/Forney) slopes at the benchmark
     widths, by differencing n_iters and the number of error_mbuc registers.
  3. The ceilings those numbers put on any optimisation of each phase.
  4. Whether Montgomery batch inversion of the Forney denominators pays, priced
     at this repository's own marginal qubit price 2*T/Q.

Requires the kickmix extension built for the running interpreter:
    /dev/shm/qvcc-venv313/bin/python scripts/measure_dqi_primitive_costs.py
A mismatched interpreter fails at import with `undefined symbol:
PyObject_ClearManagedDict`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
OUT = ROOT / "benchmarks" / "quantum_compilation" / "dqi_primitive_costs.json"

#: Benchmark II reference row (sealed receipt) and the projection's target row.
Q_REF, T_REF = 1893, 5725108.38
Q_TGT, T_TGT = 1789, 445229.0
DEG, N_SYNDROMES, N_POSITIONS = 12, 70, 4095


def _imports():
    sys.path.insert(0, str(BUNDLE / "kickmix"))
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    import kickmix as km  # noqa: F401
    import dqi_reference_circuit as DR  # noqa: F401

    return km, DR


def _toffoli(km, build, *args):
    builder = km.CircuitBuilder()
    build(builder, *args)
    circuit = builder.finish_circuit()
    return circuit.num_qubits, circuit.operation_statistics().num_toffoli_like


def measure_primitives(km) -> dict[str, float]:
    """Per-call Toffoli cost at GF(2**DEG), measured by differencing k calls."""
    gf = km.GF2Field(DEG)

    def slope(build, ks=(1, 4)):
        vals = [_toffoli(km, build, k)[1] for k in ks]
        return (vals[-1] - vals[0]) / (ks[-1] - ks[0])

    def imul(b, k):
        b.alloc_qubits(DEG)
        y = b.alloc_qubits(DEG)
        for _ in range(k):
            b.gf2_imul(3, target=y, field=gf)

    def idiv(b, k):
        b.alloc_qubits(DEG)
        y = b.alloc_qubits(DEG)
        for _ in range(k):
            b.gf2_idiv(3, target=y, field=gf)

    def qmul(b, k):
        x = b.alloc_qubits(DEG)
        y = b.alloc_qubits(DEG)
        t = b.alloc_qubits(DEG)
        for _ in range(k):
            b.ixor_gf2_mul(x, y, target=t, field=gf)

    def inv_scaffold(b, k):
        for _ in range(k):
            x = b.alloc_qubits(DEG)
            inv, sc = b.init_gf2_inverse_with_scaffold(x, field=gf)
            b.del_gf2_inverse_with_scaffold(x, target=inv, scaffold=sc, field=gf)
            b.free(sc + inv + x)

    def div_scaffold(b, k):
        for _ in range(k):
            x = b.alloc_qubits(DEG)
            y = b.alloc_qubits(DEG)
            q, sc = b.init_gf2_div_with_scaffold(x, y, field=gf)
            b.del_gf2_div_with_scaffold(x, y, target=q, scaffold=sc, field=gf)
            b.free(sc + q + x + y)

    return {
        "multiply_by_constant": slope(imul),
        "divide_by_constant": slope(idiv),
        "quantum_multiply": slope(qmul),
        "inverse_scaffold_pair": slope(inv_scaffold),
        "divide_scaffold_pair": slope(div_scaffold),
    }


def measure_slopes(km, DR) -> dict[str, float]:
    """EEA cost per iteration and Chien/Forney cost per position, at benchmark widths."""

    def build(builder, n_err, n_iters):
        syn = [builder.create_quantum_register(DEG, name=f"S{i}") for i in range(N_SYNDROMES)]
        err = [builder.create_classical_register(DEG, name=f"E{j}") for j in range(n_err)]
        DR.ProosAndZalkaDecoder(N_SYNDROMES, km.GF2Field(DEG)).build_syndrome_decoding_mbuc(
            builder, syndromes=syn, error_mbuc=err, n_iters=n_iters
        )

    _, t_2_255 = _toffoli(km, build, 255, 2)
    _, t_4_255 = _toffoli(km, build, 255, 4)
    _, t_2_511 = _toffoli(km, build, 511, 2)
    per_iter = (t_4_255 - t_2_255) / 2
    per_pos = (t_2_511 - t_2_255) / (511 - 255)
    return {
        "toffoli_iter2_err255": t_2_255,
        "toffoli_iter4_err255": t_4_255,
        "toffoli_iter2_err511": t_2_511,
        "per_iteration": per_iter,
        "per_position": per_pos,
        "qubits_at_benchmark_widths": _toffoli(km, build, 255, 2)[0],
    }


def derive(primitives: dict[str, float], slopes: dict[str, float]) -> dict:
    """Ceilings and the batch-inversion cost/benefit, priced at 2T/Q per qubit."""
    inv = primitives["inverse_scaffold_pair"]
    qmul = primitives["quantum_multiply"]
    price_q = 2 * T_REF / Q_REF  # Toffoli-equivalents per qubit

    def phase_factor(new_per_position: float) -> float:
        return T_REF / (T_REF - (slopes["per_position"] - new_per_position) * N_POSITIONS)

    batch = []
    for b in (2, 4, 8, 16, 32, 64, 128):
        per_el = 3 * qmul + inv / b
        t_saved = (inv - per_el) * N_POSITIONS
        q_added = DEG * b
        net = t_saved - q_added * price_q
        score_ratio = ((Q_REF + q_added) / Q_REF) ** 2 * ((T_REF - t_saved) / T_REF)
        batch.append(
            {
                "block": b,
                "toffoli_per_position": round(per_el, 1),
                "toffoli_saved": round(t_saved, 1),
                "qubits_added": q_added,
                "qubit_cost": round(q_added * price_q, 1),
                "net_toffoli_equivalents": round(net, 1),
                "score_ratio": round(score_ratio, 5),
                "pays": net > 0,
            }
        )

    return {
        "forney_inversion_total": {
            "toffoli": round(inv * N_POSITIONS, 1),
            "share_of_T_mean_percent": round(100 * inv * N_POSITIONS / T_REF, 2),
            "multiple_of_entire_target_T": round(inv * N_POSITIONS / T_TGT, 2),
        },
        "ceilings_if_phase_became": {
            "one_multiply_per_position": round(phase_factor(qmul), 4),
            "free": round(phase_factor(0.0), 4),
        },
        "target_factor_required": round(T_REF / T_TGT, 4),
        "batch_inversion": batch,
        "batch_inversion_ever_pays": any(row["pays"] for row in batch),
        "qubit_price_toffoli_equivalents": round(price_q, 2),
        "qubit_floor": {
            "persistent_registers": 2 * (N_SYNDROMES + 3) * DEG,
            "measured": slopes["qubits_at_benchmark_widths"],
            "transient_headroom": slopes["qubits_at_benchmark_widths"] - 2 * (N_SYNDROMES + 3) * DEG,
            "target_reduction": Q_REF - Q_TGT,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="write the artifact")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    km, DR = _imports()
    primitives = measure_primitives(km)
    slopes = measure_slopes(km, DR)
    result = {
        "source": "measured from the reference Proos-Zalka decoder via kickmix",
        "instance": {"n_syndromes": N_SYNDROMES, "degree": DEG, "positions": N_POSITIONS},
        "primitives_per_call": primitives,
        "slopes": slopes,
        "derived": derive(primitives, slopes),
    }

    print(f"primitives (Toffoli/call, GF(2^{DEG})): {primitives}")
    print(f"per iteration={slopes['per_iteration']:.0f}  per position={slopes['per_position']:.0f}")
    d = result["derived"]
    print(f"Forney inversions: {d['forney_inversion_total']['toffoli']:,.0f} Toffoli "
          f"({d['forney_inversion_total']['share_of_T_mean_percent']}% of T_mean, "
          f"{d['forney_inversion_total']['multiple_of_entire_target_T']}x the whole target)")
    print(f"ceilings: {d['ceilings_if_phase_became']}  (need {d['target_factor_required']}x)")
    print(f"batch inversion ever pays: {d['batch_inversion_ever_pays']}")
    print(f"qubit floor: {d['qubit_floor']}")

    if args.json:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
