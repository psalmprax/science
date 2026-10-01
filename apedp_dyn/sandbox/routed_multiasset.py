"""Routed ensemble (vol-gated 007/017) across 5 assets, 2019-2023.

Per asset: freeze trailing-20d-vol 80th percentile on 2018, then
walk-forward (train 60d / holdout 20d, step 20): high-vol windows route
to fin-0007 (drift+reversion), the rest to fin-0017 (cubic+damping).
Baselines: each leg alone. All scores are unseen-holdout RMSE.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.fitting.parameter_fit import fit_candidate  # noqa: E402
from apedp_dyn.grammar.candidate_ir import (  # noqa: E402
    CandidateDynamics,
    Term,
)
from apedp_dyn.problems.finance import spec_from_series  # noqa: E402

ASSETS = {"SPX": "SP500", "NDX": "NASDAQ100", "WTI": "DCOILWTICO",
          "DGS10": "DGS10", "BTC": "CBBTCUSD"}
FORMS = {"fin-0007": ("one", "x"), "fin-0017": ("x3", "abs_v_v")}
NTR, NHO, STEP = 60, 20, 20


def load(name):
    import tempfile
    files = {"SPX": "sp500", "NDX": "nasdaq100", "WTI": "oil",
             "DGS10": "dgs10", "BTC": "btc"}
    path = f"{tempfile.gettempdir()}/{files[name]}.csv"
    col = ASSETS[name]
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


def ok(x):
    return x is not None and np.isfinite(x)


def run_asset(name):
    dates, px = load(name)
    rets = np.diff(np.log(px))
    vols = np.array([rets[max(0, i - 20):i].std() if i >= 2 else np.nan
                     for i in range(len(rets))])
    m18 = np.array([d.startswith("2018") for d in dates[1:]])
    TH = float(np.quantile(vols[m18][np.isfinite(vols[m18])], 0.80))
    res = {"routed": [], "007": [], "017": []}
    n_route = {"fin-0007": 0, "fin-0017": 0}
    i = next(i for i, d in enumerate(dates) if d >= "2019-01-01")
    while i + NTR + NHO <= len(px):
        tr, ho = (i, i + NTR), (i + NTR, i + NTR + NHO)
        try:
            prob = spec_from_series(px, tr, ho, problem_id="FIN-W")
            out = {}
            for cid, tnames in FORMS.items():
                terms = tuple(Term(n, f"a{j}") for j, n in enumerate(tnames))
                f = fit_candidate(
                    prob, CandidateDynamics(lhs="xddot", terms=terms, candidate_id=cid))
                out[cid] = f.holdout_rmse
            if not all(ok(v) for v in out.values()):
                i += STEP
                continue
        except Exception:  # noqa: BLE001 -- skip failed windows, keep walking
            i += STEP
            continue
        v = vols[tr[0]:tr[1]].mean()
        leg = "fin-0007" if (np.isfinite(v) and v > TH) else "fin-0017"
        n_route[leg] += 1
        res["routed"].append(out[leg])
        res["007"].append(out["fin-0007"])
        res["017"].append(out["fin-0017"])
        i += STEP
    return TH, n_route, res


def main():
    print(f"  {'asset':6s} {'n':>3s} {'to007':>5s} "
          f"{'routed':>7s} {'007':>7s} {'017':>7s}  (mean holdout RMSE)")
    for asset in ASSETS:
        TH, n_route, res = run_asset(asset)
        n = len(res["routed"])
        line = f"  {asset:6s} {n:>3d} {n_route['fin-0007']:>5d} "
        for k in ("routed", "007", "017"):
            a = np.array(res[k])
            line += f"{a.mean():>7.3f} " if len(a) else f"{'nan':>7s} "
        print(line + f" TH={TH:.4f}")


if __name__ == "__main__":
    main()
