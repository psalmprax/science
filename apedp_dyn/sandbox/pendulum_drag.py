"""Damped pendulum with quadratic drag: transfer test for fin-0017's family.

Truth (hidden): theta'' = -(g/L)*sin(theta) - c*|omega|*omega
(drag quadratic in angular velocity, the |v|v motif). Small-angle
oscillations around theta=0 with release from several amplitudes;
holdout releases from unseen amplitudes (intervention analog).

Question: does the x3+abs_v_v family (cubic + quadratic damping) that
won ordinary months in markets get promoted where the physics actually
contains it? Grammar slate via mechanical_grammar (no "u": autonomous),
prune -> fit -> adjudicate, same DUFF/COUP/STAB path as finance_grammar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.fitting.parameter_fit import fit_candidate
from apedp_dyn.grammar.mechanical_grammar import (
    enumerate_candidates,
    prune_admissible,
)
from apedp_dyn.model import (
    DynamicalEquation,
    InputVariable,
    StateVariable,
)
from apedp_dyn.problems.base import DynProblem, make_signal

G, L, DRAG = 9.81, 1.0, 0.6
PEND_TERMS = ("one", "x", "x3", "v", "abs_v_v", "sin_x")


def simulate_pendulum(theta0, t_end=10.0, dt=0.02, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(t_end / dt) + 1
    th = np.empty(n)
    om = np.empty(n)
    th[0], om[0] = theta0, 0.0
    for t in range(1, n):
        acc = -(G / L) * np.sin(th[t - 1]) - DRAG * abs(om[t - 1]) * om[t - 1]
        om[t] = om[t - 1] + acc * dt
        th[t] = th[t - 1] + om[t] * dt
    if noise > 0:
        th = th + rng.normal(0, noise, n)
        om = om + rng.normal(0, noise, n)
    return np.arange(n) * dt, th, om


def build_problem(dt=0.02, noise=0.01, tr_amp=0.7, ho_amp=0.6, problem_id="PEND-DRAG"):
    t_tr, th_tr, om_tr = simulate_pendulum(tr_amp, t_end=10.0, dt=dt, noise=noise, seed=11)
    t_ho, th_ho, om_ho = simulate_pendulum(ho_amp, t_end=10.0, dt=dt, noise=noise, seed=23)
    Ytr = np.column_stack([th_tr, om_tr])
    Yho = np.column_stack([th_ho, om_ho])
    states = (StateVariable("x", "radian", float(Ytr[0, 0])),
              StateVariable("v", "1/second", float(Ytr[0, 1])))
    inputs = (InputVariable("u", "dimensionless"),)
    truth = (DynamicalEquation("x", "v"), DynamicalEquation("v", "0"))
    train = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "zero", "n": 1}),
        "x0": (float(th_tr[0]), float(om_tr[0])),
        "truth_t": t_tr,
        "truth_y": Ytr,
    }
    holdout = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "zero", "n": 1}),
        "x0": (float(th_ho[0]), float(om_ho[0])),
        "truth_t": t_ho,
        "truth_y": Yho,
        "tolerance": "auto",
    }
    return DynProblem(problem_id, "pendulum-drag", states, inputs,
                      (), truth, train, holdout)


def run_pendulum(noise=0.01):
    prob = build_problem(dt=0.02, noise=noise, tr_amp=0.7, ho_amp=0.6)
    slate = enumerate_candidates(required=(), optional=PEND_TERMS,
                                 max_terms=2, id_prefix="pd")
    kept = prune_admissible(prob, slate)
    print(f"enumerated: {len(slate)} | gates 1-2 survivors: {len(kept)}")
    fitted = []
    for s in kept:
        try:
            fitted.append(fit_candidate(prob, s.candidate))
        except Exception as e:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            print(f"  fit failed {s.candidate.candidate_id}: {type(e).__name__}")
    g4 = [f for f in fitted
          if f.provenance.get("gate4_status", {}).get("passed", False)]
    print(f"gate-4 survivors: {len(g4)} / {len(fitted)}")
    ranked = sorted(
        [s for s in g4 if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: float(s.train_loss))
    adj = adjudicate(ranked, bootstrap=False, data_origin="simulation")
    print(f"verdict: {adj.status.value} "
          f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None}")
    for s in sorted(g4, key=lambda c: c.holdout_rmse if c.holdout_rmse is not None else 1e18)[:6]:
        terms = "+".join(t.name for t in s.candidate.terms)
        print(f"  {s.candidate.candidate_id} {terms:18s} holdout RMSE={s.holdout_rmse:.4f}")
    print("reason:", adj.reason[:130])
    return adj, g4


if __name__ == "__main__":
    run_pendulum()
