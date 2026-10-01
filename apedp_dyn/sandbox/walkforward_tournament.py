"""Walk-forward APEDP trading tournament (prediction-first, §18/19/23).

Not "does APEDP make money" but: do APEDP epistemic states predict
prediction reliability on genuinely unseen data?

Temporal discipline (§21/22): per window, structures are fit on train,
adjudicated on a validation holdout, then FROZEN; the frozen winner is
scored on the next forward window it never saw. Grammars never evolve
from future results. Strategies compared on predictive metrics first:
  A: always trade best-RMSE model
  B: trade only UNIQUE_SUPPORT
  C: trade only UNIQUE + transfer PASS
  D: trade only on multi-model directional agreement
  E: trade only on UNIQUE + transfer PASS + agreement
P&L is recorded secondarily; the primary table is future error by
epistemic bucket. Fixed narrow structures (§29): volatility-momentum
interaction, plain AR(1), constant. Transfer per window: prior-window
winner re-scored on current holdout (temporal PASS/FAIL).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import (  # noqa: E402
    DataOrigin,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import (  # noqa: E402
    CandidateDynamics,
    ScoredCandidate,
    Term,
)

NTR, NVAL, NFWD, STEP = 120, 30, 20, 20


def load_spx(path=None):
    import tempfile
    path = path or f"{tempfile.gettempdir()}/sp500.csv"
    dates, px = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["SP500"])
                if v > 0:
                    dates.append(row["observation_date"])
                    px.append(v)
            except ValueError:
                pass
    return dates, np.array(px)


def _mk(cid, terms, rmse, resid, obs_rms, ns, n):
    g4 = {"passed": True, "holdout_obs_rms": obs_rms,
          "holdout_noise_sigma": ns, "holdout_n_samples": n,
          "holdout_residuals": [float(v) for v in resid]}
    c = CandidateDynamics(lhs="r_next",
                          terms=tuple(Term(t, f"a{i}") for i, t in enumerate(terms)),
                          candidate_id=cid)
    return ScoredCandidate(candidate=c, provenance={"gate4_status": g4},
                           train_loss=rmse, holdout_rmse=rmse)


def fit_predict_structures(r_tr, r_va):
    """Fit 3 fixed structures on train; return dict cid -> (w, val_rmse, val_resid, val_pred_fn)."""
    y_tr = r_tr[1:]
    out = {}
    # momentum interaction: r = a + b*(r_prev/vol), vol = trailing-10 std
    v_tr = np.array([max(float(np.std(r_tr[max(0, i - 10):i + 1])) if i > 1 else 0.015, 0.005)
                     for i in range(len(r_tr))])
    f_tr = r_tr[:-1] / v_tr[:-1]
    M = np.column_stack([np.ones_like(f_tr), f_tr])
    w, *_ = np.linalg.lstsq(M, y_tr, rcond=None)
    out["momentum"] = (w, ("one", "mom"))
    # plain AR(1)
    M2 = np.column_stack([np.ones(len(r_tr) - 1), r_tr[:-1]])
    w2, *_ = np.linalg.lstsq(M2, y_tr, rcond=None)
    out["ar1"] = (w2, ("one", "ar1"))
    # constant
    out["const"] = (np.array([float(np.mean(y_tr))]), ("one",))
    return out


def predict_with(struct, w, r_hist):
    if struct == "momentum":
        v = max(float(np.std(r_hist[-10:])) if len(r_hist) >= 2 else 0.015, 0.005)
        f = r_hist[-1] / v
        return float(w[0] + w[1] * f)
    if struct == "ar1":
        return float(w[0] + w[1] * r_hist[-1])
    return float(w[0])


def run_tournament():
    dates, px = load_spx()
    rets = np.diff(np.log(px))
    rows = []
    prev_winner = None
    i = 0
    while i + NTR + NVAL + NFWD <= len(rets):
        r_tr = rets[i:i + NTR]
        r_va = rets[i + NTR:i + NTR + NVAL]
        r_fw = rets[i + NTR + NVAL:i + NTR + NVAL + NFWD]
        structs = fit_predict_structures(r_tr, r_va)
        # adjudicate on validation holdout
        cands = []
        for cid, (w, terms) in structs.items():
            if cid == "momentum":
                v_va = np.array([max(float(np.std(r_va[max(0, j - 10):j + 1])) if j > 1 else 0.015, 0.005)
                                 for j in range(len(r_va))])
                pred = w[0] + w[1] * (r_va[:-1] / v_va[:-1])
                yy = r_va[1:]
            elif cid == "ar1":
                pred = w[0] + w[1] * r_va[:-1]
                yy = r_va[1:]
            else:
                pred = np.full_like(r_va, w[0])
                yy = r_va
            rm = float(np.sqrt(np.mean((yy - pred) ** 2)))
            resid = yy - pred
            cands.append(_mk(cid, terms, rm, resid, float(np.sqrt(np.mean(yy ** 2))),
                             float(np.std(resid)), len(yy)))
        ranked = sorted([s for s in cands if np.isfinite(s.train_loss)], key=lambda s: s.train_loss)
        adj = adjudicate(
            cands, resid_best=ranked[0].provenance["gate4_status"]["holdout_residuals"],
            resid_rival=ranked[1].provenance["gate4_status"]["holdout_residuals"],
            bootstrap=True, bootstrap_n_boot=500, bootstrap_seed=7,
            data_origin=DataOrigin.REAL_OBSERVATIONAL)
        # transfer: prior-window winner re-scored on current validation
        transfer = "NOT_TESTED"
        if prev_winner is not None and prev_winner in structs:
            w0, _ = structs[prev_winner]
            if prev_winner == "momentum":
                v_va = np.array([max(float(np.std(r_va[max(0, j - 10):j + 1])) if j > 1 else 0.015, 0.005)
                                 for j in range(len(r_va))])
                p0 = w0[0] + w0[1] * (r_va[:-1] / v_va[:-1])
                yy0 = r_va[1:]
            elif prev_winner == "ar1":
                p0 = w0[0] + w0[1] * r_va[:-1]
                yy0 = r_va[1:]
            else:
                p0 = np.full_like(r_va, w0[0])
                yy0 = r_va
            rm0 = float(np.sqrt(np.mean((yy0 - p0) ** 2)))
            cur_best = min(float(s.holdout_rmse) for s in cands)
            transfer = "PASS" if rm0 <= 1.5 * cur_best else "FAIL"
        # blind forward scoring with FROZEN structures
        fw = {}
        for cid, (w, _) in structs.items():
            preds = np.array([predict_with(cid, w, np.concatenate([r_tr, r_va, r_fw[:k]]))
                              for k in range(len(r_fw))])
            fw[cid] = (float(np.sqrt(np.mean((r_fw - preds) ** 2))),
                       float(np.mean(np.sign(r_fw) == np.sign(preds))))
        best_cid = min(structs, key=lambda c: next(
            s.holdout_rmse for s in cands
            if s.candidate.candidate_id == c))
        prev_winner = best_cid
        # agreement among the three frozen forward predictions
        signs = []
        for cid, (w, _) in structs.items():
            p1 = predict_with(cid, w, np.concatenate([r_tr, r_va]))
            signs.append(np.sign(p1))
        agree = len(set(signs)) == 1
        rows.append({"date": dates[i + NTR + NVAL], "status": adj.status.value,
                     "branch": getattr(adj, "bootstrap_branch", ""),
                     "transfer": transfer, "agree": agree,
                     "best": best_cid, "fw": fw})
        i += STEP
    return rows


def report(rows):
    from collections import Counter
    print(f"windows: {len(rows)}")
    # Strategy comparison on blind-forward RMSE
    strat = {"A_always_best": [], "B_unique": [], "C_unique_transfer": [],
             "D_agree": [], "E_all": []}
    for r in rows:
        errs = {c: v[0] for c, v in r["fw"].items()}
        best = r["best"]
        strat["A_always_best"].append(errs[best])
        if r["status"] == "unique_support":
            strat["B_unique"].append(errs[best])
            if r["transfer"] == "PASS":
                strat["C_unique_transfer"].append(errs[best])
                if r["agree"]:
                    strat["E_all"].append(errs[best])
        if r["agree"]:
            # agreement: mean prediction error approx by best-agreeing? use mean of three
            strat["D_agree"].append(float(np.mean(list(errs.values()))))
    for k, v in strat.items():
        a = np.array(v)
        cov = f"{len(v)}/{len(rows)}"
        print(f"  {k:16s} n={cov:9s} mean RMSE={a.mean():.5f}" if len(a) else f"  {k:16s} n=0")
    # Reliability table: future error by epistemic bucket
    print("\n  future RMSE by epistemic bucket:")
    buckets: dict = {}
    for r in rows:
        key = (r["status"], r["transfer"])
        buckets.setdefault(key, []).append(r["fw"][r["best"]][0])
    for k in sorted(buckets):
        a = np.array(buckets[k])
        print(f"    {k[0]:22s} transfer={k[1]:12s} n={len(a):3d} mean RMSE={a.mean():.5f}")
    print("\n  coverage:", dict(Counter(r["status"] for r in rows)))


if __name__ == "__main__":
    report(run_tournament())
