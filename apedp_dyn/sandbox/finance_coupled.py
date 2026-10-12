"""Joint SPX/NDX discovery with coupling topology (FIN-COUPLED).

Maps two assets onto the 2-DOF coupled machinery: states (x_s, v_s, x_n, v_n)
= standardized log-price and velocity for S&P 500 and Nasdaq-100. The "xr"
grammar term in equation 1 reads Nasdaq's position (and vice versa), so
cross-terms are contagion hypotheses competing genuinely against the
uncoupled rival (no xr terms), which is a first-class citizen of the slate.

Coupling topology is discovered, not assumed: if cross-asset terms never
survive gates/adjudication, the verdict says the assets are dynamically
decoupled on the tested regimes.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import adjudicate  # noqa: E402
from apedp_dyn.fitting.parameter_fit import fit_candidate  # noqa: E402
from apedp_dyn.grammar.coupled import enumerate_coupled  # noqa: E402
from apedp_dyn.grammar.mechanical_grammar import prune_admissible  # noqa: E402
from apedp_dyn.model import (  # noqa: E402
    DynamicalEquation,
    InputVariable,
    StateVariable,
)
from apedp_dyn.problems.base import DynProblem, make_signal  # noqa: E402

COUP_TERMS = ("x", "v", "xr")


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


def _standardize(tr, ho):
    mu, sg = float(np.log(tr).mean()), float(np.log(tr).std())
    def conv(p):
        x = (np.log(np.asarray(p, dtype=float)) - mu) / sg
        return np.column_stack([x, np.gradient(x, 1.0)])
    return conv(tr), conv(ho)


def build_joint_problem(px_s, px_n, tr, ho, problem_id="FIN-COUPLED"):
    """Joint 4-state problem: (x_s, v_s, x_n, v_n), two zero drives."""
    Ys_tr, Ys_ho = _standardize(px_s[tr[0]:tr[1]], px_s[ho[0]:ho[1]])
    Yn_tr, Yn_ho = _standardize(px_n[tr[0]:tr[1]], px_n[ho[0]:ho[1]],
                                )
    # NOTE: each asset standardized on its OWN train stats (no cross leakage).
    Ytr = np.column_stack([Ys_tr[:, 0], Ys_tr[:, 1], Yn_tr[:, 0], Yn_tr[:, 1]])
    Yho = np.column_stack([Ys_ho[:, 0], Ys_ho[:, 1], Yn_ho[:, 0], Yn_ho[:, 1]])
    states = (StateVariable("xs", "dimensionless", float(Ytr[0, 0])),
              StateVariable("vs", "1/day", float(Ytr[0, 1])),
              StateVariable("xn", "dimensionless", float(Ytr[0, 2])),
              StateVariable("vn", "1/day", float(Ytr[0, 3])))
    inputs = (InputVariable("u1", "dimensionless"),
              InputVariable("u2", "dimensionless"))
    truth = (DynamicalEquation("xs", "vs"),
             DynamicalEquation("vs", "0"),
             DynamicalEquation("xn", "vn"),
             DynamicalEquation("vn", "0"))
    train = {"t_span": (0.0, float(len(Ytr) - 1)),
             "u_of_t": make_signal({"kind": "vector", "signals": [
                 {"kind": "zero"}, {"kind": "zero"}]}),
             "x0": tuple(float(v) for v in Ytr[0]),
             "truth_t": np.arange(len(Ytr), dtype=float), "truth_y": Ytr}
    holdout = {"t_span": (0.0, float(len(Yho) - 1)),
               "u_of_t": make_signal({"kind": "vector", "signals": [
                   {"kind": "zero"}, {"kind": "zero"}]}),
               "x0": tuple(float(v) for v in Yho[0]),
               "truth_t": np.arange(len(Yho), dtype=float), "truth_y": Yho,
               "tolerance": "auto"}
    return DynProblem(problem_id, "finance-coupled", states, inputs,
                      (), truth, train, holdout)


def run_coupled():
    import tempfile
    cache = tempfile.gettempdir()
    ds, px_s = load(f"{cache}/sp500.csv", "SP500")
    _, px_n = load(f"{cache}/nasdaq100.csv", "NASDAQ100")
    idx = {d: i for i, d in enumerate(ds)}
    tr = (idx["2019-01-02"], idx["2019-12-31"] + 1)
    ho = (idx["2020-02-19"], idx["2020-03-23"] + 1)
    prob = build_joint_problem(px_s, px_n, tr, ho)
    slate = enumerate_coupled(required=(), optional=COUP_TERMS,
                              max_terms=2, id_prefix="fx")
    print(f"enumerated: {len(slate)} coupled systems")
    kept = prune_admissible(prob, slate)
    print(f"gates 1-2 survivors: {len(kept)}")
    fitted = []
    for s in kept:
        try:
            fitted.append(fit_candidate(prob, s.candidate))
        except Exception as e:  # noqa: BLE001 -- record, don't stop
            print(f"  fit failed: {type(e).__name__}")
    g4 = [f for f in fitted
          if f.provenance.get("gate4_status", {}).get("passed", False)]
    print(f"gate-4 survivors: {len(g4)} / {len(fitted)}")
    ranked = sorted(
        [s for s in g4 if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: float(s.train_loss))
    adj = adjudicate(ranked, bootstrap=False, data_origin="experimental")
    print(f"verdict: {adj.status.value} "
          f"contract={getattr(adj, 'contract', '')}")
    if adj.promoted is not None:
        for i, eq in enumerate(adj.promoted.candidate.equations):
            print(f"  promoted eq{i}: {eq}")
    print("reason:", adj.reason[:150])
    # Coupling census regardless of verdict: how many survivors use xr?
    xr_n = sum(1 for s in g4
               if any("xr" in t.name for e in s.candidate.equations
                      for t in e.terms))
    print(f"coupled (xr-bearing) survivors: {xr_n} / {len(g4)}")
    return adj


if __name__ == "__main__":
    run_coupled()
