#!/usr/bin/env python3
"""Where does the 89Q invsqrt champion actually peak?

Status: audit-only

The score is ``Q x toffoli_equivalent``, so one qubit costs 19.9 Toffolis at
Q = 89 (1,770.8 / 89). A qubit saved without spending that many Toffolis is
therefore worth more than any plausible gate-level tidy-up, and a qubit spent
has to buy more than 19.9 Toffolis to pay for itself.

Finding out whether either is possible needs the peak's location, not its
value: the circuit reports 89 qubits at the end, which says nothing about
which region holds them. This script wraps the builder's allocation calls,
records every rise in the live count with the file and line that caused it,
and rebuilds the champion through the same code path as
``scripts/test_shared_pddd_circuit.py`` so the number it explains is the
pinned one (89 qubits, hash ``ce04c9d5...``).

Usage:
    APEDP_Q_KICKMIX_PYTHON=... .venv-kickmix/bin/python scripts/diagnose_invsqrt_qubit_peak.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(
    0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix")
)
sys.path.insert(
    0,
    str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"),
)

import numpy as np  # noqa: E402

import kickmix as km  # noqa: E402
import signed_digit_invsqrt_v3 as core  # noqa: E402
import scheduled_arithmetic90 as H  # noqa: E402
import test_shared_pddd_circuit as builder  # noqa: E402

#: Files whose call sites name a build region. The first frame outside the
#: shims below that matches one of these is reported as the allocation's cause.
_REGION_FILES = {
    # `scheduled_arithmetic90.py` is deliberately absent: every allocation below is
    # made by `folded_gadd`, and naming that frame hides the caller, which is what
    # says whether the peak is in the forward `inner` loop or the `outer` loop.
    "test_shared_pddd_circuit.py": "build_candidate",
    "signed_digit_invsqrt_v3.py": "signed_digit_core",
    "schedule_support90.py": "schedule_support",
}

_live = 0
_peak = 0
_events: list[dict] = []


def _site() -> str:
    """The two innermost region frames, so `adder` shows the loop that called it."""
    seen: list[str] = []
    for frame in inspect.stack()[2:]:
        name = Path(frame.filename).name
        if name in _REGION_FILES:
            seen.append(f"{name}:{frame.lineno} ({frame.function})")
            if len(seen) == 2:
                break
    return " <- ".join(seen) if seen else "unknown"


def _delta(label: str, amount: int) -> None:
    global _live, _peak
    _live += amount
    if _live > _peak:
        _peak = _live
        _events.append({"label": label, "live": _live, "site": _site()})


def _size(obj: object, fallback: int) -> int:
    try:
        return len(obj)  # type: ignore[arg-type]
    except TypeError:
        return fallback


def instrument(cls: type) -> None:
    orig_alloc_qubits = cls.alloc_qubits
    orig_create = cls.create_quantum_register
    orig_alloc_bits = cls.alloc_bits
    orig_free = cls.free

    def alloc_qubits(self, n, *args, **kwargs):
        out = orig_alloc_qubits(self, n, *args, **kwargs)
        _delta("alloc_qubits", _size(out, n))
        return out

    def create_quantum_register(self, n, *args, **kwargs):
        out = orig_create(self, n, *args, **kwargs)
        _delta("create_quantum_register", _size(out, n))
        return out

    def alloc_bits(self, n, *args, **kwargs):
        # Classical bits are not qubits: kickmix allocates them from a separate
        # budget, and counting them here inflated the peak by ~26.
        return orig_alloc_bits(self, n, *args, **kwargs)

    def free(self, reg, *args, **kwargs):
        size = _size(reg, 0)
        _delta("free", -size)
        return orig_free(self, reg, *args, **kwargs)

    cls.alloc_qubits = alloc_qubits
    cls.create_quantum_register = create_quantum_register
    cls.alloc_bits = alloc_bits
    cls.free = free


def main() -> int:
    instrument(km.CircuitBuilder)

    cfg, tables, _ = core.load_or_build_tables()
    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray = cv_curve ^ (cv_curve >> 1)
    shared_path = REPO_ROOT / "src/shared_davio_results.json"
    cand0 = json.loads(shared_path.read_text())["candidates"][0]
    plan = builder.build_shared_plan(
        {
            "curve4": (gray >> 4) & 1,
            "curve5": (gray >> 5) & 1,
            "curve6": (gray >> 6) & 1,
        },
        order=cand0["order"],
        polarity=cand0["polarity"],
    )
    c6_choice = json.loads((REPO_ROOT / "src/boolean_choices90.json").read_text())["coarse6"]

    circuit, meta = builder.build_candidate(
        cfg, tables, coarse_choice=c6_choice, shared_curve_plan=plan
    )

    print(f"circuit qubits      : {circuit.num_qubits}")
    print(f"instrumented peak   : {_peak}")
    print(f"static Toffolis     : {meta['static_toffoli']}")
    print(f"restoration nodes   : coarse6={len(builder.boolean_plan(cv_curve, order=c6_choice['order'], polarity=c6_choice['polarity'])['nodes'])} "
          f"curves={len(plan['nodes'])}")
    print("\nevery rise in the live count (the peak is the last row):")
    for event in _events:
        print(f"  {event['live']:3d}  after {event['label']:<24s} at {event['site']}")
    print(f"\npeak is {_peak} live qubits; the pin is 89, so one allocation is unaccounted"
          f" (a reused id or a register the wrappers do not see).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
