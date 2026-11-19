#!/usr/bin/env python3
"""Where the champion's 89 qubits sit, adder by adder.

Status: audit-only

`docs/INVSQRT_SUB62_DESIGN.md` §1 prices the width of the 89-qubit champion as
`71` live named registers plus `18` carries allocated by `folded_gadd` on the
21-bit accumulate add. That decomposition decides both width levers in the
design note, so it is measured here rather than asserted: the builder is wrapped
to track the live qubit count by id, and `scheduled_arithmetic90.folded_gadd` is
wrapped to record the count on entry and with its carry chain allocated.

Prints, per (target width, live on entry, live with carries) triple, how many
calls share it, and the maximum of each. The maximum with carries must equal the
circuit's own `num_qubits`; if it does not, the wrapper is missing an allocation
path and the note's decomposition is wrong.

Usage:
    .venv-kickmix/bin/python scripts/diagnose_invsqrt_adder_liveness.py [--json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(
    0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix")
)

import numpy as np  # noqa: E402
import kickmix as km  # noqa: E402
import signed_digit_invsqrt_v3 as core  # noqa: E402
import scheduled_arithmetic90 as H  # noqa: E402
import test_shared_pddd_circuit as builder  # noqa: E402

_live = 0
_peak = 0
_adds: list[tuple[int, int, int]] = []


def _size(obj: object) -> int:
    try:
        return len(obj)  # type: ignore[arg-type]
    except TypeError:
        return 1


def instrument(cls: type) -> None:
    orig_alloc = cls.alloc_qubits
    orig_create = cls.create_quantum_register
    orig_free = cls.free

    def alloc_qubits(self, n, *args, **kwargs):
        global _live, _peak
        out = orig_alloc(self, n, *args, **kwargs)
        _live += _size(out)
        _peak = max(_peak, _live)
        return out

    def create_quantum_register(self, n, *args, **kwargs):
        global _live, _peak
        out = orig_create(self, n, *args, **kwargs)
        _live += _size(out)
        _peak = max(_peak, _live)
        return out

    def free(self, reg, *args, **kwargs):
        global _live
        _live -= _size(reg)
        return orig_free(self, reg, *args, **kwargs)

    cls.alloc_qubits = alloc_qubits
    cls.create_quantum_register = create_quantum_register
    cls.free = free


def main() -> int:
    instrument(km.CircuitBuilder)

    orig_gadd = H.folded_gadd

    def spy(b, a, t, cin=None, sub=False, borrowed=()):
        carries = max(0, len(t) - 2 - len(borrowed))
        _adds.append((len(t), _live, _live + carries))
        return orig_gadd(b, a, t, cin, sub, borrowed)

    H.folded_gadd = spy

    cfg, tables, _ = core.load_or_build_tables()
    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray = cv_curve ^ (cv_curve >> 1)
    cand0 = json.loads((REPO_ROOT / "src/shared_davio_results.json").read_text())["candidates"][0]
    plan = builder.build_shared_plan(
        {"curve4": (gray >> 4) & 1, "curve5": (gray >> 5) & 1, "curve6": (gray >> 6) & 1},
        order=cand0["order"],
        polarity=cand0["polarity"],
    )
    c6_choice = json.loads((REPO_ROOT / "src/boolean_choices90.json").read_text())["coarse6"]
    circuit, meta = builder.build_candidate(cfg, tables, coarse_choice=c6_choice, shared_curve_plan=plan)

    rows = Counter(_adds)
    entry = max(a[1] for a in _adds)
    with_carries = max(a[2] for a in _adds)

    print(f"circuit Q={circuit.num_qubits}  instrumented peak={_peak}  static T={meta['static_toffoli']}")
    print(f"adder calls={len(_adds)}  max live on entry={entry}  max live with carries={with_carries}")
    print(f"{'target width':>12} {'live in':>8} {'with carries':>13} {'calls':>6}")
    for (width, before, inside), count in rows.most_common():
        print(f"{width:>12} {before:>8} {inside:>13} {count:>6}")

    ok = with_carries == circuit.num_qubits
    print(f"\npeak accounted for by an adder: {ok} "
          f"({with_carries} vs Q={circuit.num_qubits})")
    if "--json" in sys.argv:
        print(json.dumps({
            "qubits": circuit.num_qubits,
            "instrumented_peak": _peak,
            "static_toffoli": meta["static_toffoli"],
            "max_live_on_entry": entry,
            "max_live_with_carries": with_carries,
            "calls": [{"target_width": w, "live_in": b, "live_with_carries": i, "count": c}
                      for (w, b, i), c in rows.most_common()],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
