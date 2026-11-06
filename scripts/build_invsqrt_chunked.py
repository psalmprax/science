#!/usr/bin/env python3
"""Build the champion with a segmented adder carry chain and score it.

Status: experimental

`docs/INVSQRT_SUB62_DESIGN.md` §2 prices segmenting the 21-bit accumulate add:
`folded_gadd` allocates 18 carries, which is what takes the champion from 71 live
registers to 89, so a chunk size of 7 should leave 5 carries plus one hand-off
qubit live. This script builds that variant through the same code path as
`scripts/test_shared_pddd_circuit.py` (same tables, same shared PDDD plan, same
rotation calibration) and runs the competition verifier on it, so the width is
reported by the verifier rather than by a model.

The rotation calibration is the champion's own: segmenting the adder changes
which gates are emitted, not which rotations are, so the 44 angles and their
order are unchanged. If that assumption ever breaks, the verifier fails rather
than the script silently scoring a different circuit.

Usage:
    .venv-kickmix/bin/python scripts/build_invsqrt_chunked.py --chunk 7 [--shots 25600] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from test_shared_pddd_circuit import (  # noqa: E402
    ProblemSpecInvSqrt,
    build_candidate,
    build_shared_plan,
    cal,
    core,
    domain_compiler,
    km,
    uf,
    verify,
)


def build(chunk: int) -> tuple[object, dict, dict]:
    cfg, tables, _ = core.load_or_build_tables()
    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray = cv_curve ^ (cv_curve >> 1)
    cand0 = json.loads((REPO_ROOT / "src/shared_davio_results.json").read_text())["candidates"][0]
    plan = build_shared_plan(
        {"curve4": (gray >> 4) & 1, "curve5": (gray >> 5) & 1, "curve6": (gray >> 6) & 1},
        order=cand0["order"],
        polarity=cand0["polarity"],
    )
    c6_choice = json.loads((REPO_ROOT / "src/boolean_choices90.json").read_text())["coarse6"]

    raw, raw_meta = build_candidate(
        cfg, tables, coarse_choice=c6_choice, shared_curve_plan=plan, chunk=chunk
    )
    opt, _ = domain_compiler.optimize_circuit(raw, linear=True)

    fixed = json.loads((REPO_ROOT / "src/invsqrt_calibration.json").read_text())
    lines, rots = cal.parse_kmx(cal._serialize(opt, "kmx").decode())
    values = fixed.get("deltasPhaseUnits", [])
    dropped = set(fixed.get("droppedOriginalRotationIndices", [18]))
    rewritten = uf.rewrite(
        lines, rots, {j: int(v) for j, v in enumerate(values) if j not in dropped}, dropped
    )
    circuit = km.Circuit(rewritten)
    return circuit, raw_meta, {"rotations": len(rots), "calibrated_rotation_count": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, default=7,
                        help="carry-chain segment width in bits (0 = shipped, unsegmented)")
    parser.add_argument("--shots", type=int, default=25600,
                        help="verified shots (the champion rebuild uses 25,600)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    circuit, raw_meta, rots = build(args.chunk or None)
    stats = circuit.operation_statistics()
    print(f"chunk={args.chunk}: raw Q={raw_meta['qubits']} raw T={raw_meta['static_toffoli']}")
    print(f"              compiled Q={circuit.num_qubits} static T={int(stats.num_toffoli_like)} "
          f"rotations={int(stats.num_z_pow(min_clifford_hierarchy_level=4))} "
          f"({rots['rotations']} parsed)")

    spec = ProblemSpecInvSqrt.for_n_p(10, shots_per_x=1)
    spec = ProblemSpecInvSqrt(
        width=spec.width, max_x=args.shots, beta=spec.beta,
        precision_bits=spec.precision_bits, shots_per_x=1, system_qubits=spec.system_qubits,
    )
    receipt = verify(circuit, spec)
    receipt["build_seconds"] = round(time.perf_counter() - t0, 3)
    receipt["chunk"] = args.chunk
    print(json.dumps(receipt, indent=2))
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
