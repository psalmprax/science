"""Grammar-driven discovery on real market data (FIN-PRICE).

Unlike the hand-built 3-candidate sandboxes, this wires the actual
dynamic-systems pipeline: enumerate (mechanical grammar, no "u" term) ->
prune_admissible (gates 1-2) -> fit_candidate (gates 3-4) -> adjudicate
(bootstrap tiebreak + residual series), exactly the DUFF/COUP/STAB path.

Two experiments on real S&P 500 (FRED):
  A. calm -> calm     (train 2019H1, holdout 2019H2): same-regime transfer
  B. calm -> crash    (train 2019 full, holdout COVID crash): regime break,
     the intervention analog — expect gate-4 refusal (honest abstention).

The promoted term set, if any, is a genuinely discovered equation:
selected from 22 grammar structures, not hand-picked.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.epistemic.bootstrap import holdout_se_series
from apedp_dyn.fitting.parameter_fit import fit_candidate
from apedp_dyn.grammar.mechanical_grammar import enumerate_candidates, prune_admissible
from apedp_dyn.problems.finance import spec_from_series

FIN_TERMS = ("one", "x", "x3", "v", "abs_v_v", "sin_x")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "market"


def load_sp500(path: str | Path = DATA_DIR / "sp500.csv"):
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


def run_experiment(name, prob):
    print(f"--- Experiment {name} ---")
    cands = enumerate_candidates(required=(), optional=FIN_TERMS,
                                 max_terms=2, id_prefix="fin")
    print(f"  enumerated: {len(cands)} grammar structures")
    kept = prune_admissible(prob, cands)
    rej = len(cands) - len(kept)
    print(f"  gates 1-2 survivors: {len(kept)} (rejected: {rej})")
    fitted = []
    for s in kept:
        try:
            f = fit_candidate(prob, s.candidate)
        except Exception as e:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            print(f"    fit failed {s.candidate.candidate_id}: {type(e).__name__}")
            continue
        fitted.append(f)
    g4 = [s for s in fitted
          if s.provenance.get("gate4_status", {}).get("passed", False)]
    print(f"  gate-4 survivors: {len(g4)} / {len(fitted)} fitted")

    def series(sc):
        try:
            if sc.fit is None:
                return None
            return holdout_se_series(prob, sc.candidate, sc.fit.param_values)
        except (RuntimeError, ValueError, TypeError):
            return None

    ranked = sorted(
        [s for s in g4 if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: float(s.train_loss))
    se = {id(s): series(s) for s in ranked}
    best_se = se.get(id(ranked[0])) if ranked else None
    rival_se = se.get(id(ranked[1])) if len(ranked) > 1 else None
    try:
        adj = adjudicate(
            ranked, resid_best=best_se, resid_rival=rival_se,
            resid_sole=best_se, bootstrap=True, bootstrap_n_boot=1000,
            bootstrap_seed=7000, data_origin="experimental")
    except TypeError:
        adj = adjudicate(ranked)
    print(f"  verdict: {adj.status.value}")
    if adj.promoted is not None:
        terms = sorted(t.name for t in adj.promoted.candidate.terms)
        print(f"  DISCOVERED: {adj.promoted.candidate.candidate_id} "
              f"terms={terms} train_loss={adj.promoted.train_loss:.4g} "
              f"holdout_rmse={adj.promoted.holdout_rmse:.4g}")
    print(f"  reason: {adj.reason[:110]}")
    print(f"  bootstrap_branch={getattr(adj, 'bootstrap_branch', '')} "
          f"contract={getattr(adj, 'contract', '')} "
          f"origin={getattr(adj, 'data_origin', '')}")
    return adj


def main():
    dates, px = load_sp500()
    idx = {d: i for i, d in enumerate(dates)}
    print(f"S&P 500 real: {len(px)} closes {dates[0]}..{dates[-1]}")
    # A: calm -> calm
    run_experiment("A calm->calm",
                   spec_from_series(px, (idx["2019-01-02"], idx["2019-07-01"]),
                                    (idx["2019-07-02"], idx["2019-12-31"]),
                                    problem_id="FIN-CALM"))
    print()
    # B: calm -> crash (regime break)
    run_experiment("B calm->crash",
                   spec_from_series(px, (idx["2019-01-02"], idx["2019-12-31"]),
                                    (idx["2020-02-19"], idx["2020-03-23"]),
                                    problem_id="FIN-CRASH"))


if __name__ == "__main__":
    main()
