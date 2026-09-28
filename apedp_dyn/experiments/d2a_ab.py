"""D2a controlled A/B: spline/pointwise-derivative vs weak/integral estimation.

Same physical problems, same noise realizations, both arms structure-fixed on
grammar-generated IR. Noise sweep 0/0.25/0.5/1/2/5%. Metrics per arm/level:
estimator param error, template-threshold support (false/missed terms), refined
trajectory-fit train loss + holdout RMSE + gate pass, runtime.

Hypothesis: weak-form reduces noise-induced derivative artifacts and improves
structural recovery as observation noise increases. Effect size is measured,
not promised. Weak-form is an estimation mechanism inside the gates, never a
bypass: refined models run gates 1-4 identically in both arms.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn import (  # noqa: E402
    CandidateDynamics,
    Term,
    enumerate_candidates,
    fit_candidate,
    materialize,
    prune_admissible,
)
from apedp_dyn.estimation import spline_coefficients, weak_coefficients  # noqa: E402
from apedp_dyn.problems import mass_spring_damper, pendulum  # noqa: E402

NOISE_LEVELS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
REFINE_LEVELS = (0.0, 1.0, 5.0)
SEED = 11
SUPPORT_THRESHOLD = 0.01  # relative to max |coef|
# Window-scale sensitivity: fixed radius=1.0 is the headline arm; radius=0.5
# tests whether the pendulum gap is a timescale mismatch (period ~2 s).
SENSITIVITY_RADII = {"D1-MSD": (1.0,), "D1-PEND": (1.0, 0.5)}

TRUTH = {
    # problem_id: (template term names, {term: true value})
    "D1-MSD": (("one", "x", "x3", "v", "abs_v_v", "u"),
               {"x": -4.0, "v": -0.2, "u": 1.0}),
    "D1-PEND": (("one", "x", "x3", "v", "abs_v_v", "sin_x", "u"),
                {"sin_x": -9.81, "v": -0.1, "u": 1.0}),
}


def _template(problem_id: str) -> CandidateDynamics:
    names, _ = TRUTH[problem_id]
    terms = tuple(Term(n, f"b{i}") for i, n in enumerate(names))
    return CandidateDynamics(
        lhs="xddot", terms=terms,
        metadata={"family": "mechanical_2nd_order", "order": 2},
        candidate_id=f"d2a-template-{problem_id}",
        parameter_bounds={f"b{i}": (-1e6, 1e6) for i in range(len(names))},
        generation_seed=SEED)


def _truth_candidate(problem, problem_id: str):
    names = sorted(TRUTH[problem_id][1])
    pool = enumerate_candidates(required=(), optional=tuple(names),
                                max_terms=len(names), id_prefix="d2a")
    want = set(names)
    c = next(x for x in pool if set(x.term_set()) == want)
    kept = list(prune_admissible(problem, [c]))
    assert len(kept) == 1
    return kept[0].candidate


def _observe(Y: np.ndarray, pct: float):
    if pct == 0.0:
        return Y
    rng = np.random.default_rng(SEED)
    return Y + rng.normal(0, pct / 100 * np.std(Y, axis=0), Y.shape)


def _max_rel_error(est: dict[str, float], true: dict[str, float]) -> float:
    return max(abs(est[k] - v) / max(abs(v), 1e-12)
               for k, v in true.items() if k in est)


def _term_columns(problem, candidate, t: np.ndarray, Y: np.ndarray,
                  U: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Materialized term columns for normalized (variance-aware) thresholding."""
    import sympy as sp

    model = materialize(problem, candidate)
    order = [s.name for s in model.states]
    inames = [u.name for u in model.inputs]
    syms = [sp.Symbol(n) for n in order] + [sp.Symbol(n) for n in inames]
    args = [Y[:, i] for i in range(len(order))] + [U[:, i] for i in range(len(inames))]
    vel = order[1]
    accel_eq = next(e.sympy_expr for e in model.equations if e.state == vel)
    labels, cols = [], []
    for p in model.parameters:
        term = sp.diff(accel_eq, sp.Symbol(p.name))
        f = sp.lambdify(syms, term, modules=["numpy"])
        col = np.asarray(f(*args), dtype=float).ravel()
        if col.size == 1 and len(t) > 1:
            col = np.full(len(t), col[0])
        labels.append(p.name)
        cols.append(col)
    return labels, np.column_stack(cols)


def _normalized_support(full: dict, labels: list[str], cols: np.ndarray,
                        template_names: tuple) -> list[str]:
    """Threshold |coef|*||column|| (explained-variance scale), not raw magnitude."""
    pnames = [f"b{i}" for i in range(len(template_names))]
    strength = {n: abs(full[p]) * float(np.linalg.norm(cols[:, i]))
                for i, (n, p) in enumerate(zip(template_names, pnames, strict=True))}
    m = max(strength.values())
    return sorted(n for n, s in strength.items() if s >= SUPPORT_THRESHOLD * m)


def run() -> dict:
    report: dict = {"noise_levels": list(NOISE_LEVELS), "seed": SEED,
                    "support_threshold": SUPPORT_THRESHOLD, "problems": {}}
    for prob in (mass_spring_damper.load(), pendulum.load()):
        pid = prob.problem_id
        template_names, true_vals = TRUTH[pid]
        template = _template(pid)
        truth_cand = _truth_candidate(prob, pid)
        # Param names per term for the truth candidate.
        truth_pnames = {t.name: t.parameter for t in truth_cand.terms}
        true_by_pname = {truth_pnames[t]: v for t, v in true_vals.items()
                         if t in truth_pnames}
        reg = prob.train
        t = np.asarray(reg["truth_t"])
        Y = np.asarray(reg["truth_y"])
        U = np.array([reg["u_of_t"](ti) for ti in t]).reshape(-1, 1)
        pres: dict = {"levels": {}}
        for pct in NOISE_LEVELS:
            Yn = _observe(Y, pct)
            row: dict = {}
            ests = {}
            for arm, fn in (("spline", spline_coefficients), ("weak", weak_coefficients)):
                t0 = time.time()
                est = fn(prob, truth_cand, t, Yn, U)
                dt = time.time() - t0
                ests[arm] = est
                row[arm] = {"param_max_rel_error": _max_rel_error(est, true_by_pname),
                            "estimator_seconds": round(dt, 3)}
            # Template-threshold support study (selection among grammar terms).
            tmpl_labels, tmpl_cols = _term_columns(prob, template, t, Yn, U)
            for arm in ("spline", "weak"):
                t0 = time.time()
                full = (spline_coefficients if arm == "spline" else weak_coefficients)(
                    prob, template, t, Yn, U)
                sup = {n for n, p in
                       zip(template_names,
                           [f"b{i}" for i in range(len(template_names))], strict=True)
                       if abs(full[p]) >= SUPPORT_THRESHOLD * max(abs(v) for v in full.values())}
                nsup = set(_normalized_support(full, tmpl_labels, tmpl_cols, template_names))
                row[arm]["support"] = sorted(sup)
                row[arm]["false_terms"] = sorted(sup - set(true_vals))
                row[arm]["missed_terms"] = sorted(set(true_vals) - sup)
                row[arm]["support_norm"] = sorted(nsup)
                row[arm]["false_terms_norm"] = sorted(nsup - set(true_vals))
                row[arm]["missed_terms_norm"] = sorted(set(true_vals) - nsup)
                row[arm]["support_seconds"] = round(time.time() - t0, 3)
            # Refined trajectory fits at subset of levels (identical LS, own x0).
            if pct in REFINE_LEVELS:
                model0 = materialize(prob, truth_cand)
                order = [p.name for p in model0.parameters]
                for arm in ("spline", "weak"):
                    x0 = np.array([ests[arm][n] for n in order])
                    t0 = time.time()
                    res = fit_candidate(prob, truth_cand, x0=x0)
                    row[arm]["refined"] = {
                        "train_loss": res.train_loss,
                        "holdout_rmse": res.holdout_rmse,
                        "gates_pass": [g.passed for g in res.gates],
                        "fit_seconds": round(time.time() - t0, 1)}
            pres["levels"][str(pct)] = row
            print(f"[{pid} {pct}%] " + " ".join(
                f"{a}:err={row[a]['param_max_rel_error']:.4f} "
                f"sup={row[a]['support']} norm={row[a]['support_norm']}" for a in ("spline", "weak")), flush=True)
        # Window-scale sensitivity (estimator-only, truth structure).
        sens: dict = {}
        truth_cand = _truth_candidate(prob, pid)
        true_by_pname = _true_by_pname(prob, pid, truth_cand)
        for radius in SENSITIVITY_RADII[pid]:
            errs = []
            for pct in NOISE_LEVELS:
                Yn = _observe(Y, pct)
                est = weak_coefficients(prob, truth_cand, t, Yn, U, radius=radius)
                errs.append(round(_max_rel_error(est, true_by_pname), 5))
            sens[str(radius)] = errs
        pres["sensitivity_radius_param_err"] = sens
        report["problems"][pid] = pres
    return report


def _true_by_pname(prob, problem_id: str, truth_cand) -> dict:
    _, true_vals = TRUTH[problem_id]
    mapping = {t.name: t.parameter for t in truth_cand.terms}
    return {mapping[t]: v for t, v in true_vals.items() if t in mapping}


if __name__ == "__main__":
    rep = run()
    out = ROOT / "outputs" / "d2a_ab_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {out}")
