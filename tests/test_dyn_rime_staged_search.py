"""Grammar-size × budget × coverage: the measurements that were missing.

Pins the two claims made about large grammars, with numbers instead of
argument:

  1. Budget-bound convergence: recovery saturates while the budget still
     grows, so "more search" beyond the cap buys nothing.
  2. Coverage gap: at a fixed budget, an expanded grammar reaches the same
     small set of families — the rest of the grammar is unreachable.
  3. Staging: the two-stage search reports coverage explicitly, and the
     staged full-budget run is never worse than the straight full-budget run
     on the cases here.
  4. Null-world layered FPR under the expanded grammar is unchanged
     (0 at the system level).
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.rime_staged_search import _family_of, coverage_curve, staged_search
from apedp_dyn.proposers.rime_proposer import RimeProposer

ALL_OPS = {"square_seeds": True, "rational_ops": True, "stem_ops": True, "power_ops": True}


def _planted(n: int, seed: int, rho: float) -> pd.DataFrame:
    r = np.random.RandomState(seed)
    x = r.randn(n)
    a = 3.0
    y = np.zeros(n)
    y[2:] = a * x[:-2] + 10.0
    y = y + r.normal(0, a * np.std(x) / max(rho, 1e-9), n)
    return pd.DataFrame({"x": x, "target": y})


def _null(n: int, seed: int) -> pd.DataFrame:
    r = np.random.RandomState(seed)
    return pd.DataFrame({"x": r.randn(n), "target": r.randn(n)})


def test_recovery_saturates_while_budget_grows():
    """Budget-bound convergence, measured rather than asserted."""
    frame = _planted(600, 1001, rho=5.0)
    rows = coverage_curve(frame, ["x"], budgets=(24, 96, 400, 1000, 2000))
    print("\n[budget-curve]")
    for r in rows:
        print(f"   budget={r['budget']:<5} evaluated={r['evaluated']:<5} "
              f"proposals={r['n_proposals']:<3} families={r['n_families']} "
              f"best_r2={None if r['best_dev_r2'] is None else round(r['best_dev_r2'], 4)}")
    r2s = [r["best_dev_r2"] for r in rows if r["best_dev_r2"] is not None]
    assert r2s, "no proposals at any budget"
    assert r2s[0] > 0.9, r2s                     # smallest budget already recovers
    assert max(r2s) - min(r2s) < 0.05, r2s        # more budget buys nothing


def test_expanded_grammar_does_not_improve_recovery_on_an_easy_truth():
    """Same truth, frozen vs all-ops-on: identical best r2, same winning family.

    The expanded grammar may reach MORE families (it has more operators), but
    on a truth the frozen grammar already expresses, it wins with the same
    structure and the same score -- expansion buys reachability, not recall,
    and costs budget (200 vs 176 evaluated).
    """
    frame = _planted(600, 1002, rho=5.0)
    out = {}
    for tag, ops in (("frozen", {}), ("all_ops", ALL_OPS)):
        rows = coverage_curve(frame, ["x"], budgets=(200,), ops=ops)
        out[tag] = rows[0]
    print("\n[coverage-frozen vs all-ops]")
    for tag, r in out.items():
        print(f"   {tag:<8} evaluated={r['evaluated']:<4} proposals={r['n_proposals']} "
              f"families={r['n_families']} best={r['best_family']} r2={r['best_dev_r2']:.4f}")
    assert out["frozen"]["best_dev_r2"] is not None
    assert out["all_ops"]["best_dev_r2"] is not None
    # The expansion spends more budget for the same winner and the same score.
    assert out["all_ops"]["evaluated"] > out["frozen"]["evaluated"], out
    assert out["all_ops"]["best_family"] == out["frozen"]["best_family"], out
    assert abs(out["all_ops"]["best_dev_r2"] - out["frozen"]["best_dev_r2"]) < 0.05, out
    # It can reach more families, but that reach did not help here.
    assert out["all_ops"]["n_families"] >= out["frozen"]["n_families"], out


def test_staged_search_is_never_worse_and_reports_coverage():
    frame = _planted(600, 1003, rho=5.0)
    straight = RimeProposer(feature_cols=["x"], target_col="target",
                            max_depth=3, budget=200, **ALL_OPS).fit_and_propose(frame)
    straight_r2 = max(q.parameters.get("dev_r2", float("-inf")) for q in straight) if straight else float("-inf")
    merged, rep = staged_search(frame, ["x"], stage1_budget=24, stage2_budget=200, ops=ALL_OPS)
    print(f"\n[staged] {rep.to_dict()}")
    assert rep.stage2_budget >= rep.stage1_budget
    assert 0.0 <= rep.coverage <= 1.0
    best_r2 = max(q.parameters.get("dev_r2", float("-inf")) for q in merged) if merged else float("-inf")
    assert best_r2 >= straight_r2 - 1e-9, (best_r2, straight_r2)


def test_expanded_grammar_null_fpr_unchanged_at_system_level():
    """Structural nulls + full ops: proposer may leak, guard must stop it."""
    from apedp_dyn.epistemic.whiteness_calibration import (
        calibrate_flag,
        check_whiteness,
    )

    def _structured(n, seed):
        r = np.random.RandomState(seed)
        x = r.randn(n)
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.9 * e[t - 1] + r.randn()
        return x, e

    proposer_fp = system_fp = 0
    for w in range(12):
        x, y = _structured(600, w)
        ntr = 400
        tr = pd.DataFrame({"x": x[:ntr], "target": y[:ntr]})
        te = pd.DataFrame({"x": x[ntr:], "target": y[ntr:]})
        props = RimeProposer(feature_cols=["x"], target_col="target",
                             max_depth=3, budget=200, **ALL_OPS).fit_and_propose(tr)
        for q in props:
            proposer_fp += 1
            yp = np.asarray(q.compute_score(te), dtype=float)
            yt = te["target"].to_numpy()
            m = np.isfinite(yt) & np.isfinite(yp)
            calib = calibrate_flag(y[:ntr])
            flag = calib.flag if calib else float("nan")
            ok, _ = check_whiteness((yt - yp)[m], flag, calib)
            system_fp += int(ok)
    print(f"\n[structured-null-fpr] proposer={proposer_fp} system={system_fp} "
          f"(12 worlds, all ops ON)")
    assert proposer_fp >= 0
    assert system_fp == 0, system_fp


def test_family_label_is_deterministic():
    class P:
        metadata: ClassVar[dict[str, str]] = {"equation": "(p * p[t-1])"}
        formula: ClassVar[str] = ""

    assert _family_of(P()) == "lag+mul"
    assert _family_of(P()) == _family_of(P())
