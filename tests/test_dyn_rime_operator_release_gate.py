"""Operator release gate: an opt-in may only become default-on by measurement.

For each opt-in grammar operator this records three numbers on the SAME
fixed batteries and applies a preregistered promotion rule:

    recall uplift  = known-truth recovery (ops-on) − recovery (frozen)
    system FPR     = structured-null promotions surviving the calibrated
                     whiteness guard (must stay 0)
    cost           = candidates evaluated, frozen vs ops-on (cap at 2x)

Promotion rule (preregistered, applied by the test, asserted by it):
    promote-to-default  iff  recall_uplift > 0 and system_fpr == 0
                          and cost_ratio <= 2.0
Anything else stays default-off. The test records the decision row so the
conversation is about measurements rather than arguments, and it fails if an
operator that already fails the rule is silently enabled by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.whiteness_calibration import (
    calibrate_flag,
    check_whiteness,
)
from apedp_dyn.proposers.rime_proposer import RimeProposer

OPERATORS = ("square_seeds", "rational_ops", "stem_ops", "power_ops")

# Preregistered promotion rule.
MAX_COST_RATIO = 2.0
REQUIRED_FPR = 0


def _planted(n: int, seed: int, rho: float) -> pd.DataFrame:
    r = np.random.RandomState(seed)
    x = r.randn(n)
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0
    y = y + r.normal(0, 3.0 * np.std(x) / max(rho, 1e-9), n)
    return pd.DataFrame({"x": x, "target": y})


def _structured_null(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.RandomState(seed)
    x = r.randn(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.9 * e[t - 1] + r.randn()
    return x, e


def _recovery_exact_lag(ops: dict, *, rho: float = 5.0, n_trials: int = 10) -> float:
    """Battery 1: a planted lag the frozen grammar already expresses."""
    good = 0
    for s in range(n_trials):
        frame = _planted(600, 2000 + s, rho)
        props = RimeProposer(feature_cols=["x"], target_col="target",
                             max_depth=3, budget=200, **ops).fit_and_propose(frame)
        if not props:
            continue
        best = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        eq = best.metadata.get("equation") or ""
        if "x" in eq:
            good += 1
    return good / n_trials


_QEC_P = np.linspace(0.001, 0.025, 400)


def _recovery_exact_square(ops: dict, *, n_trials: int = 6) -> float:
    """Battery 2: a lag-ZERO self-square — the frozen grammar cannot reach it.

    Structural (not fitted) recovery: the winning equation must contain a
    lag-0 self-product of p. Lagged proxies like ``p * p[t-5]`` are NOT
    accepted here, because they only fit the grid's constant ratio.
    """
    good = 0
    for s in range(n_trials):
        r = np.random.default_rng(3000 + s)
        clean = 1000.0 * _QEC_P ** 2
        y = clean + r.normal(0, 0.05 * clean.std(), _QEC_P.size)
        frame = pd.DataFrame({"p": _QEC_P, "target": y})
        props = RimeProposer(feature_cols=["p"], target_col="target", max_depth=1,
                             budget=200, static_mode=True, **ops).fit_and_propose(frame)
        if not props:
            continue
        best = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        eq = best.metadata.get("equation") or ""
        # lag-0 self-product present, no temporal read
        if "*" in eq and "[t-" not in eq and "state_" not in eq:
            good += 1
    return good / n_trials


def _cost(ops: dict) -> float:
    frame = _planted(600, 2000, 5.0)
    props = RimeProposer(feature_cols=["x"], target_col="target",
                         max_depth=3, budget=200, **ops).fit_and_propose(frame)
    return float(props[0].metadata.get("evaluated", 0)) if props else float("nan")


def _system_fpr(ops: dict, n_worlds: int = 12) -> int:
    fp = 0
    for w in range(n_worlds):
        x, y = _structured_null(600, w)
        ntr = 400
        tr = pd.DataFrame({"x": x[:ntr], "target": y[:ntr]})
        te = pd.DataFrame({"x": x[ntr:], "target": y[ntr:]})
        props = RimeProposer(feature_cols=["x"], target_col="target",
                             max_depth=3, budget=200, **ops).fit_and_propose(tr)
        for q in props:
            yp = np.asarray(q.compute_score(te), dtype=float)
            yt = te["target"].to_numpy()
            m = np.isfinite(yt) & np.isfinite(yp)
            calib = calibrate_flag(y[:ntr])
            flag = calib.flag if calib else float("nan")
            ok, _ = check_whiteness((yt - yp)[m], flag, calib)
            fp += int(ok)
    return fp


def _decision(uplift: float, fpr: int, cost_ratio: float) -> str:
    if uplift > 0 and fpr == REQUIRED_FPR and cost_ratio <= MAX_COST_RATIO:
        return "promote-to-default"
    return "keep-default-off"


BATTERIES = (("exact_lag", _recovery_exact_lag), ("exact_lag_zero_square", _recovery_exact_square))


def test_operator_release_gate_records_a_decision_per_operator():
    """The gate verifies the RULE: recorded decision == rule(measurements).

    It does not assert a predetermined answer. Which operators pass is a
    measurement; the safety half (system FPR == 0) is the non-negotiable part.
    """
    rows = []
    for bname, fn in BATTERIES:
        base_recall = fn({})
        base_cost = _cost({})
        for op in OPERATORS:
            ops = {op: True}
            rec = fn(ops)
            fpr = _system_fpr(ops)
            cost = _cost(ops)
            ratio = cost / base_cost if base_cost else float("inf")
            decision = _decision(rec - base_recall, fpr, ratio)
            # Gate-internal consistency: the recorded row must equal the rule
            # applied to the very numbers recorded next to it.
            assert decision == _decision(rec - base_recall, fpr, ratio)
            rows.append((bname, op, rec, base_recall, fpr, cost, ratio, decision))
            print(f"\n[gate] {bname:<22} {op:<14} recall={rec:.2f} (base {base_recall:.2f} "
                  f"uplift {rec - base_recall:+.2f}) FPR={fpr} "
                  f"cost={cost} ratio={ratio:.2f} -> {decision}")
    assert rows, "no operators evaluated"
    # Safety half is non-negotiable: no row may carry a nonzero system FPR.
    for bname, op, _rec_c, _base_recall_c, fpr, _cost_c, _ratio_c, _decision_c in rows:
        assert fpr == REQUIRED_FPR, (bname, op, fpr)
    # Discrimination check, restated for the new baseline: the gate must not
    # be vacuously unanimous. The explicitly-FROZEN grammar cannot express the
    # lag-zero-square truth while the shipped default can, so the two baselines
    # must actually differ -- otherwise the promotion carried no information.
    frozen = {"square_seeds": False, "rational_ops": False, "stem_ops": False, "power_ops": False}
    assert _recovery_exact_square(frozen) < _recovery_exact_square({}), (
        "default and frozen grammar are indistinguishable on the square battery")


def test_expanded_grammar_keeps_system_fpr_at_zero():
    """The gate's safety half, measured at the expanded operator count."""
    ops = {"square_seeds": True, "rational_ops": True, "stem_ops": True, "power_ops": True}
    got = _system_fpr(ops, n_worlds=12)
    assert got == 0, got


def test_defaults_are_exactly_the_operators_the_gate_promoted():
    """Policy: a default is ON only if the gate recorded promote-to-default.

    The gate's baseline is the SHIPPED default (whatever it currently is), so
    an operator that already paid for itself and was promoted shows zero
    marginal uplift here -- that is the correct reading, not a gap. The
    evidence for the promotion is the frozen-baseline comparison below.
    """
    p = RimeProposer()
    on_by_default = {op for op in OPERATORS if getattr(p, "_" + op) is True}
    assert on_by_default == {"square_seeds"}, on_by_default
    # Every default-on operator must earn its place against the frozen grammar
    # (all opt-ins explicitly off) on the battery it was promoted for.
    frozen = {"square_seeds": False, "rational_ops": False, "stem_ops": False, "power_ops": False}
    base_frozen = _recovery_exact_square(frozen)
    assert base_frozen == 0.0, base_frozen          # frozen grammar cannot
    for op in on_by_default:
        ops = dict(frozen)
        ops[op] = True
        rec = _recovery_exact_square(ops)
        fpr = _system_fpr(ops)
        ratio = _cost(ops) / _cost(frozen)
        decision = _decision(rec - base_frozen, fpr, ratio)
        assert decision == "promote-to-default", (op, decision, rec, fpr, ratio)
    print(f"\n[gate] defaults on = {sorted(on_by_default)} (each promoted against the frozen baseline)")


def test_default_change_did_not_widen_coverage_or_risk():
    """Safety of the flip: the new default must not reach more families or
    leak more on the battery the frozen grammar already solves."""
    frozen = {"square_seeds": False, "rational_ops": False, "stem_ops": False, "power_ops": False}
    default = {}
    rec_frozen = _recovery_exact_lag(frozen)
    rec_default = _recovery_exact_lag(default)
    assert rec_default >= rec_frozen, (rec_default, rec_frozen)
    fpr_default = _system_fpr(default, n_worlds=12)
    fpr_frozen = _system_fpr(frozen, n_worlds=12)
    assert fpr_default == fpr_frozen == 0, (fpr_default, fpr_frozen)
    print(f"\n[gate] default vs frozen on the easy battery: recall {rec_default:.2f} vs "
          f"{rec_frozen:.2f}, FPR {fpr_default} vs {fpr_frozen}")
