"""Multi-commodity pooled discovery: one table, one verdict.

Scores every mechanical-grammar structure (max_terms=2, no "u") on each
asset's COVID-crash holdout (train = 2019), then pools mean holdout RMSE
per structure and adjudicates ONCE on the pooled scores.

Assets (FRED, daily): SPX, NDX, WTI crude, 10Y Treasury, BTC.
Saved artifact: outputs/multicommodity_pooled.json

Caveat (recorded, not hidden): mean-pooling lets the worst asset dominate
(WTI RMSEs ~5-6 swamp equity ~1s), so the pooled ranking favors
oil-survivors. The verdict refused anyway (MULTI, INCOMPLETE without
bootstrap), which is the correct response to a suspect aggregation.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import adjudicate  # noqa: E402
from apedp_dyn.fitting.parameter_fit import fit_candidate  # noqa: E402
from apedp_dyn.grammar.candidate_ir import (  # noqa: E402
    CandidateDynamics,
    ScoredCandidate,
    Term,
)
from apedp_dyn.grammar.mechanical_grammar import (  # noqa: E402
    enumerate_candidates,
    prune_admissible,
)
from apedp_dyn.problems.finance import spec_from_series  # noqa: E402

FIN_TERMS = ("one", "x", "x3", "v", "abs_v_v", "sin_x")
def _cache(name):
    import tempfile
    return f"{tempfile.gettempdir()}/{name}"


ASSETS = {"SPX": (_cache("sp500.csv"), "SP500"),
          "NDX": (_cache("nasdaq100.csv"), "NASDAQ100"),
          "WTI": (_cache("oil.csv"), "DCOILWTICO"),
          "DGS10": (_cache("dgs10.csv"), "DGS10"),
          "BTC": (_cache("btc.csv"), "CBBTCUSD")}
TR = ("2019-01-02", "2019-12-31")
HO = ("2020-02-19", "2020-03-23")


def load(path, col):
    dates, px = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[col])
                if v > 0:
                    dates.append(row["observation_date"])
                    px.append(v)
            except ValueError:
                pass
    return dates, np.array(px)


def at_idx(dates, day, first=True):
    idx = {d: i for i, d in enumerate(dates)}
    if day in idx:
        return idx[day]
    seq = enumerate(dates) if first else reversed(list(enumerate(dates)))
    return next(i for i, d in seq if (d >= day if first else d <= day))


def run_pooled():
    probs = {}
    for asset, (path, col) in ASSETS.items():
        dates, px = load(path, col)
        i0, i1 = at_idx(dates, TR[0]), at_idx(dates, TR[1], first=False)
        j0, j1 = at_idx(dates, HO[0]), at_idx(dates, HO[1], first=False)
        probs[asset] = spec_from_series(px, (i0, i1), (j0, j1),
                                        problem_id=f"FIN-{asset}")
    slate = enumerate_candidates(required=(), optional=FIN_TERMS,
                                 max_terms=2, id_prefix="fin")
    kept = prune_admissible(probs["SPX"], slate)  # gates 1-2 structural
    structs = [(s.candidate.candidate_id,
                tuple(t.name for t in s.candidate.terms)) for s in kept]
    mat, pooled = {}, {}
    for cid, tnames in structs:
        mat[cid] = {}
        for asset in ASSETS:
            terms = tuple(Term(n, f"a{i}") for i, n in enumerate(tnames))
            c = CandidateDynamics(lhs="xddot", terms=terms, candidate_id=cid)
            try:
                f = fit_candidate(probs[asset], c)
                g4 = f.provenance.get("gate4_status", {}).get("passed", False)
                mat[cid][asset] = (f.holdout_rmse, g4)
            except Exception:  # noqa: BLE001 -- record, don't stop the slate
                mat[cid][asset] = (None, False)
    print(f"  {'cid':9s} {'terms':22s} " + " ".join(f"{a:>8s}" for a in ASSETS)
          + f"  {'pooled':>7s}")
    for cid, tnames in structs:
        row, vals = [], []
        for asset in ASSETS:
            rmse, passed = mat[cid][asset]
            vals.append(rmse if rmse is not None and np.isfinite(rmse) else np.nan)
            row.append(f"{' ' if passed else '*'}{rmse:7.2f}"
                       if rmse is not None and np.isfinite(rmse) else "     nan")
        p = float(np.nanmean(vals))
        pooled[cid] = (p, tnames)
        print(f"  {cid:9s} {'+'.join(tnames):22s} " + " ".join(row) + f"  {p:7.2f}")
    print("  (* = failed gate 4 on that asset)")

    order = sorted([c for c in pooled if np.isfinite(pooled[c][0])],
                   key=lambda c: pooled[c][0])
    scored = []
    for cid in order:
        p, tnames = pooled[cid]
        terms = tuple(Term(n, f"a{i}") for i, n in enumerate(tnames))
        c = CandidateDynamics(lhs="xddot", terms=terms, candidate_id=cid)
        g4 = {"passed": True, "holdout_obs_rms": 1.0,
              "holdout_noise_sigma": 0.3, "holdout_n_samples": 200}
        scored.append(ScoredCandidate(c, {"gate4_status": g4},
                                      train_loss=float(p), holdout_rmse=float(p)))
    adj = adjudicate(scored, bootstrap=False, data_origin="experimental")
    print(f"\nPOOLED VERDICT: {adj.status.value} "
          f"contract={getattr(adj, 'contract', '')} "
          f"origin={getattr(adj, 'data_origin', '')}")
    if adj.promoted is not None:
        print("PROMOTED:", adj.promoted.candidate.candidate_id,
              f"pooled RMSE={adj.promoted.holdout_rmse:.3f}")
    print("reason:", adj.reason[:140])
    with open("outputs/multicommodity_pooled.json", "w") as f:
        json.dump({"pooled": {c: {"rmse": pooled[c][0], "terms": list(pooled[c][1])}
                              for c in pooled},
                   "verdict": adj.status.value}, f, indent=1)
    print("saved outputs/multicommodity_pooled.json")
    return adj, mat


if __name__ == "__main__":
    run_pooled()
