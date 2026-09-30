"""DUFF 3% adversarial tournament (frozen regression boundary, v1).

Additive harness only: builds an 8-candidate slate, runs it through the
UNMODIFIED pipeline (prune -> fit -> gates -> adjudicate), and records per
candidate WHERE it died plus an audit-only RMSE_rel. It never alters
adjudication behavior and never feeds RMSE_rel back into any decision.

Slate (frozen; hashed):
  C1 ground truth      {u,x,x3,v}                -> promotion expected
  C2 linear surrogate  {u,x,v}                   -> gate 4 expected
  C3 even-polynomial   {u,x,x2,v}                -> gate 4 expected
  C4 over-parameterized {u,x,x2,x3,v,abs_v_v}    -> adjudication expected
  C5 dimensionally invalid {xr,x}                -> gate 1 (materialize raises)
  C6 unstable/divergent {u,x3}, a1 in (50,100)   -> gate 3 (train blowup)
  C7 wrong-regime      linear fitted on A=2.5-train variant -> gate 4 expected
  C8 under-identified  {u,x,x3,v,x_v}            -> adjudication expected

eliminated_at rule (first failing layer, pipeline order):
  prune-reject with dimensional cause -> "gate1"
  fit: gate1 fail -> "gate1"; gate2 -> "gate2"; gate3 -> "gate3";
       gate4 fail -> "gate4"; all pass -> "adjudication" (promoted or rival).
RMSE_rel = holdout_rmse / holdout_obs_rms when holdout is reached, else None.
EPS_PHYS = 0.05 is a frozen audit constant; adjudication never sees RMSE_rel.

Deviation log (live tree vs naive expectation — recorded, not hidden):
  D1. At 3% noise the fitted truth itself can fail gate 4 (rep-dependent), so
      C1 promotion is NOT guaranteed; the contract pins OBSERVED per-rep
      traces and fails on any CHANGE, including attribution changes with an
      unchanged final status.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time

import numpy as np

from apedp_dyn import dyn_sweep as ds
from apedp_dyn import fit_candidate, make_signal, prune_admissible
from apedp_dyn.epistemic.adjudicate import adjudicate

try:
    from apedp_dyn.epistemic.adjudicate import DYNAMIC_EPISTEMIC_VERSION
except ImportError:  # stand-down: v1 tree without hardening constants
    DYNAMIC_EPISTEMIC_VERSION = "point-estimate-v1 (stand-down fallback)"
from apedp_dyn.fitting.parameter_fit import _with_values
from apedp_dyn.gates import dimensional_validity, intervention_holdout
from apedp_dyn.grammar import materialize
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, Term
from apedp_dyn.problems import duffing
from apedp_dyn.problems.base import finalize

TOURNAMENT_VERSION = "1.0.0-duff3"
PCT = 3
REPS = (0, 1, 2)
EPS_PHYS = 0.05

# cid -> (term names, parameter bounds, fit x0 or None)
SLATE: dict[str, tuple[tuple[str, ...], dict[str, tuple[float, float]], list[float] | None]] = {
    "C1": (("u", "x", "x3", "v"), {}, None),
    "C2": (("u", "x", "v"), {}, None),
    "C3": (("u", "x", "x2", "v"), {}, None),
    "C4": (("u", "x", "x2", "x3", "v", "abs_v_v"), {}, None),
    "C5": (("xr", "x"), {}, None),
    "C6": (("u", "x3"), {"a1": (50.0, 100.0)}, [0.0, 75.0]),
    "C7": (("u", "x", "v"), {}, None),  # fitted on the big-train variant (below)
    "C8": (("u", "x", "x3", "v", "x_v"), {}, None),
}


def slate_hash() -> str:
    body = json.dumps(
        {k: {"terms": list(v[0]), "bounds": v[1], "x0": v[2]} for k, v in SLATE.items()},
        sort_keys=True,
    )
    return hashlib.sha256(body.encode()).hexdigest()


def _make_ir(cid: str) -> CandidateDynamics:
    names, bounds, _ = SLATE[cid]
    terms = tuple(Term(n, f"a{i}") for i, n in enumerate(names))
    return CandidateDynamics(lhs="xddot", terms=terms, candidate_id=cid,
                             parameter_bounds=dict(bounds))


def _attribute_prune_reject(clean, ir: CandidateDynamics) -> tuple[str, str]:
    try:
        model = materialize(clean, ir)
    except (TypeError, ValueError) as e:
        return "gate1", f"materialize rejected: {type(e).__name__}: {e}"
    g1 = dimensional_validity(model)
    if not g1.passed:
        return "gate1", f"dimensional: {g1.detail}"
    return "gate2", "pruned at admissibility (see gate2 detail)"


def _big_train_variant():
    base = duffing.spec()
    tr = dict(base.train)
    tr["u_of_t"] = make_signal({"kind": "sine", "amplitude": 2.5, "frequency": 1.0})
    return finalize(dataclasses.replace(base, train=tr))


def run_rep(rep: int) -> dict:
    t0 = time.perf_counter()
    slate = ds.build_slates()
    clean = slate["DUFF"]["prob_clean"]
    prob = ds._with_noise(clean, PCT, seed=ds.cell_seed("DUFF", rep, PCT))
    obs = np.asarray(prob.holdout["truth_y"], dtype=float)
    obs_rms = float(np.sqrt(np.mean(obs ** 2)))

    rows: dict[str, dict] = {}
    survivors = []
    for cid in SLATE:
        ir = _make_ir(cid)
        kept = prune_admissible(clean, [ir])
        if not kept:
            layer, detail = _attribute_prune_reject(clean, ir)
            rows[cid] = {"eliminated_at": layer, "rmse_rel": None,
                         "holdout_rmse": None, "detail": detail}
            continue
        if cid == "C7":
            big = ds._with_noise(_big_train_variant(), PCT,
                                 seed=ds.cell_seed("DUFF_C7BIG", rep, PCT))
            fit = fit_candidate(big, ir)
            if fit.fit is None:
                rows[cid] = {"eliminated_at": "gate2", "rmse_rel": None,
                             "holdout_rmse": None,
                             "detail": "wrong-regime fit failed on variant"}
                continue
            model = _with_values(materialize(clean, ir),
                                 {k: float(v) for k, v in fit.fit.param_values.items()})
            g4 = intervention_holdout(model, prob)
            rmse = g4.metrics.get("rmse")
            rmse = (float(rmse) if rmse is not None and np.isfinite(rmse) else None)
            rows[cid] = {
                "eliminated_at": "adjudication" if g4.passed else "gate4",
                "rmse_rel": (rmse / obs_rms) if rmse is not None else None,
                "holdout_rmse": rmse, "detail": f"wrong-regime transfer: {g4.detail}"}
            continue
        x0 = SLATE[cid][2]
        s = fit_candidate(prob, ir,
                          x0=np.array(x0, dtype=float) if x0 else None)
        g = s.provenance
        if not g.get("gate1_status", {}).get("passed", False):
            layer = "gate1"
        elif not g.get("gate2_status", {}).get("passed", False):
            layer = "gate2"
        elif not g.get("gate3_status", {}).get("passed", False):
            layer = "gate3"
        elif not g.get("gate4_status", {}).get("passed", False):
            layer = "gate4"
        else:
            layer = "adjudication"
            survivors.append(s)
        rmse = s.holdout_rmse
        rows[cid] = {
            "eliminated_at": layer,
            "rmse_rel": (float(rmse) / obs_rms) if rmse is not None and np.isfinite(rmse) else None,
            "holdout_rmse": (float(rmse) if rmse is not None and np.isfinite(rmse) else None),
            "detail": (g.get("rejection_reason")
                       or g.get("gate4_status", {}).get("detail", "")),
        }

    adj = adjudicate(survivors)
    promoted = adj.promoted.candidate.candidate_id if adj.promoted is not None else None
    if promoted is not None and promoted in rows:
        rows[promoted]["eliminated_at"] = "promotion"
    return {
        "rep": rep, "pct": PCT, "rows": rows,
        "adjudication": adj.status.value, "promoted": promoted,
        "n_survivors": len(survivors),
        "slate_hash": slate_hash(), "eps_phys": EPS_PHYS,
        "adjudicator": DYNAMIC_EPISTEMIC_VERSION,
        "runtime_s": time.perf_counter() - t0,
    }


def run_tournament(reps: tuple[int, ...] = REPS) -> dict:
    traces = [run_rep(r) for r in reps]
    return {
        "tournament": TOURNAMENT_VERSION, "pct": PCT, "reps": list(reps),
        "slate_hash": slate_hash(), "eps_phys": EPS_PHYS,
        "adjudicator": DYNAMIC_EPISTEMIC_VERSION,
        "traces": traces,
    }
