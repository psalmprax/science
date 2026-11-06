#!/usr/bin/env python3
"""Attribute the champion's Toffoli stream to the region that emitted it.

Status: audit-only

The champion reports 1,466 static Toffolis in total, which says nothing about
which part of the arithmetic to attack. This script wraps the builder's Toffoli-
like primitives (`init_and`, `del_and`, `ccx`, `cswap`, `multi_controlled_x` —
the same five `num_toffoli_like` sums) and the builder's region functions, and
attributes each gate to the innermost region on the stack.

Regions are the build's own call sites: the table load (`init_anf`), the record
phasing (`phase_from_records`), the PDDD restores (`restore_boolean`,
`restore_shared_boolean`), normalization and de-normalization, and the adders —
which are split by target width, because the 12-bit adds come from the 8
correction rows (`inner`) and the 21-bit adds from the 14 outer rows.

The sum must equal `operation_statistics().num_toffoli_like`; if it does not,
some primitive emits gates outside the five wrapped ones and the percentages are
not to be trusted.

Usage:
    .venv-kickmix/bin/python scripts/attribute_invsqrt_toffolis.py [--json]
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

# `del_and` is deliberately absent: it is a measurement plus a classically
# conditioned fixup, and `num_toffoli_like` does not charge it. Counting it here
# over-counted the stream by 936 gates on the first run.
TOFFOLI_LIKE = ("init_and", "ccx", "cswap", "multi_controlled_x")
_counts: Counter[str] = Counter()
_stack: list[str] = []


def _region() -> str:
    return _stack[-1] if _stack else "unattributed"


def instrument_builder(cls: type) -> None:
    for name in TOFFOLI_LIKE:
        original = getattr(cls, name, None)
        if original is None:
            continue

        def make(orig):
            def wrapper(self, *args, **kwargs):
                _counts[_region()] += 1
                return orig(self, *args, **kwargs)
            return wrapper

        setattr(cls, name, make(original))


def instrument_region(module, name: str, label: str) -> None:
    original = getattr(module, name, None)
    if original is None:
        return

    def wrapper(*args, **kwargs):
        _stack.append(label)
        try:
            return original(*args, **kwargs)
        finally:
            _stack.pop()

    setattr(module, name, wrapper)


def main() -> int:
    instrument_builder(km.CircuitBuilder)
    instrument_region(builder, "init_anf", "table load (init_anf)")
    instrument_region(builder, "phase_from_records", "record phasing")
    instrument_region(builder, "restore_boolean", "single-output PDDD restore")
    instrument_region(builder, "restore_shared_boolean", "shared PDDD restore")
    instrument_region(H, "normalize", "normalize")
    instrument_region(H, "unnormalize", "unnormalize")

    orig_gadd = H.folded_gadd

    def gadd(b, a, t, cin=None, sub=False, borrowed=()):
        _stack.append(f"adder ({len(t)}-bit target)")
        try:
            return orig_gadd(b, a, t, cin, sub, borrowed)
        finally:
            _stack.pop()

    H.folded_gadd = gadd

    cfg, tables, _ = core.load_or_build_tables()
    cv = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray = cv ^ (cv >> 1)
    cand0 = json.loads((REPO_ROOT / "src/shared_davio_results.json").read_text())["candidates"][0]
    plan = builder.build_shared_plan(
        {"curve4": (gray >> 4) & 1, "curve5": (gray >> 5) & 1, "curve6": (gray >> 6) & 1},
        order=cand0["order"], polarity=cand0["polarity"],
    )
    c6 = json.loads((REPO_ROOT / "src/boolean_choices90.json").read_text())["coarse6"]
    circuit, meta = builder.build_candidate(cfg, tables, coarse_choice=c6, shared_curve_plan=plan)

    total = sum(_counts.values())
    reported = int(circuit.operation_statistics().num_toffoli_like)
    print(f"raw circuit: Q={circuit.num_qubits}  num_toffoli_like={reported}  attributed={total}")
    print(f"{'region':<34} {'Toffolis':>9} {'share':>8}")
    for label, count in _counts.most_common():
        print(f"{label:<34} {count:>9} {100*count/total:7.1f}%")
    if total != reported:
        raise SystemExit(
            f"attributed {total} gates but the circuit reports {reported}: "
            f"{reported - total:+d} are outside the wrapped primitives"
        )
    if "--json" in sys.argv:
        print(json.dumps({"raw_total": reported, "attributed": dict(_counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
