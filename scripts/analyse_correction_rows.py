#!/usr/bin/env python3
"""What each of the 8 correction rows buys, in phase error and in Toffolis.

Status: audit-only

`docs/INVSQRT_SUB62_DESIGN.md` §4b notes that a cheaper correction is the remaining
lever on Benchmark I: 8 signed-digit rows (`cfg.rrow_list()`) plus 14 outer rows
are where the 1,466 Toffolis go. This script answers whether a row can be
dropped at the same 24-bit precision by evaluating the emitted integer model
against the verifier's own target and tolerance.

For every `x` in the competition domain it computes the model phase with the full
row set and with each single row removed, and reports the worst
`|error| / tol(x)` plus the residual's maximum, which sets the working width.
The full-row run is the control: it must reproduce the champion's 0.99962.

Row `i` contributes `sig_i * (K2 >> sh2[i])` to the residual, so dropping it is
not a rounding change — it deletes one signed digit of the C2 * delta product.
The coefficients were calibrated with all 8 rows present, so these errors are an
upper bound: re-calibrating K2/K1/K0 against the reduced row set could recover
some of it, which `--recalibrate` does for the single most promising drop.

Usage:
    .venv-kickmix/bin/python scripts/analyse_correction_rows.py [--recalibrate]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(
    0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix")
)
sys.path.insert(
    0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt")
)

import numpy as np  # noqa: E402

import signed_digit_invsqrt_v3 as core  # noqa: E402
import scheduled_arithmetic90 as H  # noqa: E402
from verify_submission_invsqrt import ProblemSpecInvSqrt, reference_values  # noqa: E402


def rd_vec(v: np.ndarray, s: int) -> np.ndarray:
    if s < 0:
        return v << (-s)
    out = v >> s
    if s >= 1:
        out = out + ((v >> (s - 1)) & 1)
    return out


def domain(cfg) -> dict:
    """Normalized address, delta and shift for every supported x."""
    xs = np.arange(1, core.MAX_X + 1, dtype=np.int64)
    exp = np.frexp(xs.astype(np.float64))[1].astype(np.int64)
    shifts = 21 - (exp - 1)
    norm = xs << shifts
    parity = shifts & 1
    index = ((norm >> 14) & 127) + 128 * parity
    return {"xs": xs, "shifts": shifts, "index": index, "delta": norm & ((1 << core.DB) - 1)}


def evaluate(cfg, tables, dom, rows) -> dict:
    keys = [(e, a) for e in range(2) for a in range(128)]
    arr = np.asarray([tables[k] for k in keys], dtype=np.int64)
    K2, K1, K0 = (arr[dom["index"], j] for j in range(3))
    d = dom["delta"]
    R = K1.copy()
    for i in rows:
        sig = 2 * ((d >> i) & 1) - 1
        R = R - sig * rd_vec(K2, cfg.sh2[i])
    Q = np.zeros_like(R)
    for i in reversed(range(core.DB)):
        sig = 2 * ((d >> i) & 1) - 1
        Q = Q + sig * rd_vec(R, cfg.sh1[i])
    Y = ((K0 - Q) >> cfg.gy) << cfg.gy
    phase = Y << (dom["shifts"] >> 1)          # units of 2^-(fy+10) turns
    return {"R": R, "Rmax": int(R.max()), "Rmin": int(R.min()), "phase": phase}


def ratios(cfg, phase, spec, dom) -> dict:
    target, tol = reference_values(spec, dom["xs"].astype(np.uint64))
    # The verifier compares phases as uint64 in units of 2^-64 turns mod 1 turn, so the
    # difference has to wrap: at x=1 the phase is a whole turn and an unwrapped
    # subtraction reports 2^64 instead of 0.
    scale = np.uint64(1 << (64 - cfg.fy - 10))
    with np.errstate(over="ignore"):
        units = phase.astype(np.uint64) * scale
    err = np.abs((units - target).astype(np.int64).astype(np.float64))
    tol_f = tol.astype(np.float64)
    ratio = np.divide(err, tol_f, out=np.full_like(err, np.inf), where=tol_f > 0)
    worst = int(np.argmax(ratio))
    return {
        "worst_ratio": float(ratio[worst]),
        "worst_x": int(dom["xs"][worst]),
        "over_tolerance": int((ratio > 1.0).sum()),
    }


def main() -> int:
    cfg, tables, _ = core.load_or_build_tables()
    rows = cfg.rrow_list()
    spec = ProblemSpecInvSqrt.for_n_p(10, shots_per_x=1)
    dom = domain(cfg)

    print(f"rows = {rows}   (shifts {[cfg.sh2[i] for i in rows]})   tolerance bits = {spec.precision_bits}")
    base = evaluate(cfg, tables, dom, rows)
    base_r = ratios(cfg, base["phase"], spec, dom)
    print(f"\ncontrol (all {len(rows)} rows): worst error/tol = {base_r['worst_ratio']:.6f} "
          f"at x={base_r['worst_x']}, Rmax={base['Rmax']} (residual width {base['Rmax'].bit_length()})")

    print(f"\n{'dropped row':>12} {'shift':>6} {'worst err/tol':>14} {'worst x':>10} "
          f"{'x over tol':>11} {'Rmax':>6} {'width':>6}")
    results = []
    for i in rows:
        kept = [j for j in rows if j != i]
        got = evaluate(cfg, tables, dom, kept)
        r = ratios(cfg, got["phase"], spec, dom)
        results.append((i, r, got["Rmax"]))
        print(f"{i:>12} {cfg.sh2[i]:>6} {r['worst_ratio']:>14.3f} {r['worst_x']:>10} "
              f"{r['over_tolerance']:>11} {got['Rmax']:>6} {got['Rmax'].bit_length():>6}")

    print("\nA 12-bit inner add costs 9-11 Toffolis (`folded_gadd` with one borrowed carry),")
    print("so a dropped row is worth about 18-22 Toffolis across its compute and uncompute passes.")

    if "--recalibrate" in sys.argv:
        best = min(results, key=lambda t: t[1]["worst_ratio"])[0]
        print(f"\nre-calibrating the coefficient tables without row {best} ...", flush=True)
        reduced = core.Cfg(core.F2, core.F1, core.FYR, core.GY, drop2=(best,))
        tabs, worst = core.build_tables(reduced, search=2)
        got = evaluate(reduced, tabs, dom, reduced.rrow_list())
        r = ratios(reduced, got["phase"], spec, dom)
        print(f"re-calibrated without row {best}: worst error/tol = {r['worst_ratio']:.6f} "
              f"at x={r['worst_x']}, Rmax={got['Rmax']} "
              f"(calibration's own worst ratio {worst:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
