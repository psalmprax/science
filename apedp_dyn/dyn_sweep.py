"""DUFF/COUP/STAB four-noise sweep (frozen).

Identical seeds, splits, intervention definitions, and gate thresholds
across noise levels (paired). Noise = Gaussian at pct% of per-channel
train/holdout std, rng seeded per (problem, rep, pct); base truth fixed.

Cells (frozen):
  DUFF: duffing.load() (train A=0.3, holdout A=2.5, decisive OOD).
        slate truth {u,v,x,x3} vs linear {u,v,x}.
  COUP: coupled.load(). slate coupled truth vs uncoupled rival.
  STAB: duffing.spec with holdout A=1.2 (near-boundary probe).
        slate truth {u,v,x,x3} vs linear {u,v,x}.

Slates are pruned ONCE on the clean problem (gates 1-2 are structural,
noise-independent); only fitting/adjudication rerun per cell.
Thresholds: gate-4 tolerance "auto" (as specced); adjudication via the
frozen holdout-scale resolution floor. False promotion is REPORTED.

Recorded per cell: status, promoted, delta/tau, holdout RMSEs,
n_candidates, runtime. Curves: Recovery(sigma), FalsePromotion(sigma),
Abstention(sigma) — answering whether noise causes structure loss,
correct abstention, or surrogate promotion.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from apedp_dyn import (
    adjudicate,
    enumerate_candidates,
    enumerate_coupled,
    fit_candidate,
    make_signal,
    prune_admissible,
)
from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_dyn.problems import coupled, duffing
from apedp_dyn.problems.base import finalize

SWEEP_VERSION = "1.1.0-bootdiag"
SIGMA_PCTS = (1, 3, 5, 10)
REPS = (0, 1, 2)


def cell_seed(problem_name: str, rep: int, pct: float) -> int:
    """Deterministic per-cell seed (crc32, NOT hash(): hash() of str is
    salted per process and made v1.0.0 sweeps irreproducible)."""
    import zlib

    return 1_000_000 + int(zlib.crc32(f"{problem_name}|{rep}|{pct}".encode())) % 900_000


def _with_noise(prob, pct: float, seed: int):
    rng = np.random.default_rng(seed)
    train = dict(prob.train)
    holdout = dict(prob.holdout)
    for reg in (train, holdout):
        Y = np.asarray(reg["truth_y"], dtype=float)
        scale = np.std(Y, axis=0)
        train_noisy = Y + rng.normal(0.0, pct / 100.0 * scale, Y.shape)
        reg["truth_y"] = train_noisy
    return dataclasses.replace(prob, train=train, holdout=holdout)


def _duff_stab_base(stab: bool):
    if not stab:
        return duffing.load()
    base = duffing.spec()
    holdout = dict(base.holdout)
    holdout["u_of_t"] = make_signal({"kind": "sine", "amplitude": 1.2, "frequency": 1.0})
    return finalize(dataclasses.replace(base, holdout=holdout))


def _duff_slate(prob):
    cands = enumerate_candidates(required=("u",), optional=("x", "x3", "v"), max_terms=4)
    kept = {tuple(sorted(s.candidate.term_set())): s for s in prune_admissible(prob, cands)}
    return kept[("u", "v", "x", "x3")].candidate, kept[("u", "v", "x")].candidate


def _coup_slate(prob):
    kept = prune_admissible(prob, enumerate_coupled(
        required=("u",), optional=("x", "v", "xr"), max_terms=4))

    def get(sets):
        return next(s for s in kept if [set(e) for e in s.candidate.term_sets()] == sets)

    truth = get([{"u", "x", "v", "xr"}, {"u", "x", "v", "xr"}]).candidate
    rival = get([{"u", "x", "v"}, {"u", "x", "v"}]).candidate
    return truth, rival


def run_cell(problem_name: str, pct: float, rep: int,
             slate: dict[str, dict]) -> dict:
    t0 = time.perf_counter()
    prob = _with_noise(slate[problem_name]["prob_clean"], pct,
                       seed=cell_seed(problem_name, rep, pct))
    from apedp_dyn.epistemic.bootstrap import holdout_se_series

    def _series(sc, ir):
        try:
            if sc.fit is None:
                return None
            return holdout_se_series(prob, ir, sc.fit.param_values)
        except (RuntimeError, ValueError, TypeError):
            return None

    truth_c, rival_c = slate[problem_name]["truth"], slate[problem_name]["rival"]
    rt = fit_candidate(prob, truth_c)
    rr = fit_candidate(prob, rival_c)
    # Rank mirrors adjudicate(): lowest finite train loss first, so the
    # residual series land on the right roles (best/rival/sole).
    ranked = [s for s in (rt, rr)
              if s.train_loss is not None and np.isfinite(s.train_loss)]

    def _train_key(s) -> float:
        assert s.train_loss is not None
        return float(s.train_loss)

    ordered = sorted(ranked, key=_train_key)
    se = {id(s): _series(s, c) for s, c in ((rt, truth_c), (rr, rival_c))}
    best_se = se.get(id(ordered[0])) if ordered else None
    rival_se = se.get(id(ordered[1])) if len(ordered) > 1 else None
    # Overlap: the pair path carries the bootstrap tiebreak AND a
    # winner-whiteness diagnostic (recorded, non-decisive on this path).
    # Stand-down fallback: v1 tree without the hardened kwargs.
    try:
        adj = adjudicate([rt, rr], resid_best=best_se, resid_rival=rival_se,
                         resid_sole=best_se, bootstrap=True, bootstrap_n_boot=1000,
                         bootstrap_seed=cell_seed(problem_name, rep, pct))
    except TypeError:
        adj = adjudicate([rt, rr])
    from apedp_dyn.epistemic.bootstrap import lag1_autocorrelation

    def _gate4(sc):
        return bool(sc.provenance.get("gate4_status", {}).get("passed", False))

    winner_se = None
    if adj.promoted is not None:
        winner_se = se.get(id(adj.promoted), None)
    elif ordered:
        winner_se = best_se
    winner_rho1 = None
    if winner_se is not None:
        try:
            rho = float(lag1_autocorrelation(winner_se))
            winner_rho1 = rho if np.isfinite(rho) else None
        except (ValueError, TypeError):
            winner_rho1 = None
    promoted = adj.promoted.candidate.candidate_id if adj.promoted is not None else None
    is_truth = promoted == truth_c.candidate_id if promoted else False
    return {
        "problem": problem_name, "pct": pct, "rep": rep,
        "status": adj.status.value, "promoted_is_truth": bool(is_truth),
        "truth_gate4": _gate4(rt), "rival_gate4": _gate4(rr),
        "winner_rho1": winner_rho1,
        "bootstrap_branch": getattr(adj, "bootstrap_branch", None) or None,
        "recovery": bool(adj.status == EpistemicStatus.UNIQUE_SUPPORT and is_truth),
        "false_promotion": bool(adj.status == EpistemicStatus.UNIQUE_SUPPORT and not is_truth),
        "abstention": bool(adj.status in (EpistemicStatus.MULTI_HYPOTHESIS,
                                          EpistemicStatus.INSUFFICIENT_EVIDENCE)),
        "delta_rmse": adj.delta_rmse, "tau_epi": adj.tau_epi,
        "truth_holdout_rmse": rt.holdout_rmse, "rival_holdout_rmse": rr.holdout_rmse,
        "n_candidates": 2, "runtime_s": time.perf_counter() - t0,
    }


def build_slates() -> dict[str, dict]:
    duff = _duff_stab_base(False)
    stab = _duff_stab_base(True)
    coup = coupled.load()
    dt, dr = _duff_slate(duff)
    st, sr = _duff_slate(stab)
    ct, cr = _coup_slate(coup)
    return {
        "DUFF": {"prob_clean": duff, "truth": dt, "rival": dr,
                 "desc": "train A=0.3, holdout A=2.5 (decisive OOD)"},
        "COUP": {"prob_clean": coup, "truth": ct, "rival": cr,
                 "desc": "2-DOF coupling topology transfer"},
        "STAB": {"prob_clean": stab, "truth": st, "rival": sr,
                 "desc": "train A=0.3, holdout A=1.2 (near-boundary probe)"},
    }


def run_sweep(problems: tuple[str, ...] = ("DUFF", "COUP", "STAB"),
              pcts: tuple[float, ...] = SIGMA_PCTS,
              reps: tuple[int, ...] = REPS) -> dict:
    slate = build_slates()
    cells = [run_cell(p, pct, rep, slate) for p in problems for pct in pcts for rep in reps]
    curves: dict[str, dict] = {}
    for p in problems:
        curves[p] = {}
        for pct in pcts:
            sub = [c for c in cells if c["problem"] == p and c["pct"] == pct]
            curves[p][str(pct)] = {
                "n": len(sub),
                "recovery": sum(c["recovery"] for c in sub) / len(sub),
                "false_promotion": sum(c["false_promotion"] for c in sub) / len(sub),
                "abstention": sum(c["abstention"] for c in sub) / len(sub),
                "statuses": sorted({c["status"] for c in sub}),
                "mean_runtime_s": float(np.mean([c["runtime_s"] for c in sub])),
            }
    try:
        from apedp_dyn.epistemic.adjudicate import (
            DYNAMIC_EPISTEMIC_VERSION,
            RESID_RHO_FLAG,
            SOLE_SURVIVOR_GAMMA,
        )
        adjudication = {"version": DYNAMIC_EPISTEMIC_VERSION,
                        "sole_gamma": SOLE_SURVIVOR_GAMMA,
                        "resid_rho_flag": RESID_RHO_FLAG,
                        "residuals": "measured holdout SE series",
                        "bootstrap": True, "bootstrap_n_boot": 1000}
    except ImportError:
        # Stand-down: v1 tree without hardening constants.
        adjudication = {"version": "point-estimate-v1 (stand-down fallback)",
                        "residuals": "measured holdout SE series (diagnostic only)",
                        "bootstrap": False}
    return {"sweep_version": SWEEP_VERSION, "problems": list(problems),
            "pcts": list(pcts), "reps": list(reps),
            "slate_desc": {k: v["desc"] for k, v in slate.items()},
            "adjudication": adjudication,
            "cells": cells, "curves": curves}
