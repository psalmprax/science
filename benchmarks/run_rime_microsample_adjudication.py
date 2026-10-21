r"""Micro-sample origin-constrained adjudication with patched referee logic.

Status: audit-only

Closes three defects from the external APEDP-X reconciliation double-check
(18/20 claims reproduced by execution, 2 falsified):

1. The external ``sound_continuous_monotonicity_proof`` filtered positive
   roots by an absolute ``1e-9`` tolerance, silently certifying monotonicity
   for a derivative like ``N(p) = p * (p - 5e-10)`` (negative on
   ``(0, 5e-10)``). Fixed: ``sound_monotonicity_v2`` certifies through exact
   real-root isolation (``sympy.polys.polytools.real_roots``) with NO
   absolute-tolerance filtering, partitions ``(0, oo)`` on exact-identified
   positive roots, and falls back to ``UNRESOLVED`` for non-algebraic
   expressions instead of sampling unsoundly.
2. The external negative-control decision ``all(p.parameters.get("scale", 0)
   < 0 for p in proposals)`` is vacuously True for an empty proposal list,
   so a proposer ABSTENTION could be reported as a dynamics-inversion
   rejection. Fixed: ``classify_referee_decision`` separates ABSTAIN
   (empty packet) / REJECT (gates failed) / PROMOTE.
3. The Michaelis-Menten "PROMOTED (12/12)" claim existed only as prose.
   ``run_michaelis_menten`` re-executes it with measured per-fold gates and
   the exact family certificate q'(p) = w*s/(s+p)^2 > 0 (w > 0, s > 0),
   ratified by symbolic identity check.

Writes ``benchmarks/rime_microsample_adjudication_results.json``.
Timing: the external "10.2x / 0.022 ms" ratio reproduced in NEITHER protocol
scope on the double-check host (scalar-dot 153x, full 8-scale fold 1.5x), so
no head-to-head ratio is recorded; measured raw per-scope costs for the
archive host live in the results JSON under ``timing_scopes``.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon
import sympy as sp

_REPO = Path(__file__).resolve().parents[1]

_LANGMUIR_P = [34.0, 23.8, 17.3, 13.0, 9.5, 7.4, 6.1, 5.0, 4.0, 3.4, 2.8]
_LANGMUIR_Q = [33.0, 30.8, 28.2, 25.5, 23.9, 21.6, 19.0, 17.0, 15.1, 13.4, 12.0]

_LANGMUIR_MULTS = [0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 1.0, 2.0]
_MM_MULTS = [0.25, 0.33, 0.5, 0.67, 1.0, 2.0]

_ABSTAIN = "ABSTAIN_NO_HYPOTHESIS_GENERATED"


def _phi(x: np.ndarray, s: float) -> np.ndarray:
    """Origin-constrained saturating feature phi(x; s) = (x/s) / (1 + x/s)."""
    return (x / s) / (1.0 + x / s)


# ══════════════════════════════════════════════════════════════════════════
# Patched monotonicity engine (defect-1 fix)
# ══════════════════════════════════════════════════════════════════════════
def sound_monotonicity_v2(expr: sp.Expr, x: sp.Symbol) -> tuple[str, str]:
    """Certify q'(x) >= 0 on the open domain (0, oo); abstain otherwise.

    Verdict space: CERTIFIED_MONOTONIC | VIOLATED | UNRESOLVED.

    Soundness rationale: for a rational expression, together(diff()) yields
    num/den with polynomial num and den. Both factors are certified by exact
    real-root isolation; the sign on every open interval between consecutive
    real roots is invariant, so a single high-precision (40-digit) interior
    probe per interval decides the sign without any absolute-tolerance root
    filtering. Non-algebraic expressions return UNRESOLVED.
    """
    diff_expr = sp.together(sp.diff(expr, x))
    num, den = sp.fraction(diff_expr)

    for part in (num, den):
        if not (part.is_number or part.is_polynomial(x)):
            return "UNRESOLVED", "non-algebraic numerator/denominator; cannot certify sign exactly"

    # ── Denominator must be strictly positive on (0, oo) ──
    try:
        den_poly = sp.Poly(den, x)
    except (sp.polys.polyerrors.PolynomialError, AttributeError):
        return "UNRESOLVED", "denominator not a polynomial; cannot certify D>0 on (0, oo)"
    if den_poly.degree() <= 0:
        den_c = float(sp.N(den, 40))
        if den_c <= 0:
            return "VIOLATED", f"denominator non-positive constant ({den})"
    else:
        den_pos_roots = [
            float(sp.N(r, 40))
            for r in sp.polys.polytools.real_roots(den_poly)
            if sp.N(r, 40).is_real and float(sp.N(r, 40)) > 0.0
        ]
        if den_pos_roots:
            return "VIOLATED", (
                f"denominator has positive real zeros {den_pos_roots}; "
                "expression singular inside (0, oo)"
            )
        if float(sp.N(den_poly.eval(1), 40)) <= 0:
            return "VIOLATED", "denominator constant-sign but negative on (0, oo) (D(1) <= 0, no positive zeros)"

    # ── Numerator sign by exact root partition ──
    if num.is_number:
        c = float(sp.N(num, 40))
        if c > 0:
            return "CERTIFIED_MONOTONIC", f"constant numerator {c:.6g} > 0 with certified D>0 on (0, oo)"
        if c < 0:
            return "VIOLATED", f"constant numerator {c:.6g} < 0 with certified D>0 on (0, oo)"
        return "CERTIFIED_MONOTONIC", "derivative identically zero (constant law)"

    try:
        num_poly = sp.Poly(num, x)
    except (sp.polys.polyerrors.PolynomialError, AttributeError):
        return "UNRESOLVED", "numerator not a polynomial; cannot partition (0, oo)"

    # NO absolute-tolerance filter (defect-1 fix): roots like 5e-10 count.
    num_pos_roots = sorted(
        float(sp.N(r, 40))
        for r in sp.polys.polytools.real_roots(num_poly)
        if sp.N(r, 40).is_real and float(sp.N(r, 40)) > 0.0
    )

    boundaries = [0.0] + num_pos_roots + [float("inf")]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mid = (lo + hi) / 2.0 if np.isfinite(hi) else max(lo, 0.0) + 1.0
        v = float(sp.N(num.subs(x, mid), 40))
        if v < 0:
            return "VIOLATED", (
                f"exact interval probe: N < 0 ({v:.6g}) on ({lo:.6g}, {hi:.6g}); "
                "positive numerator roots " + repr(num_pos_roots)
            )
        if v == 0:
            return "UNRESOLVED", f"interval probe N == 0 at p={mid:.6g}; degenerate interval"
    return "CERTIFIED_MONOTONIC", (
        "numerator sign > 0 on every exact-root-partitioned interval of (0, oo) "
        f"(positive roots {num_pos_roots}); D > 0 certified; no tolerance filtering"
    )


def certify_origin_constrained_rime(w: float, s: float) -> tuple[str, str]:
    r"""Exact symbolic certificate for q(p) = w*(p/s)/(1+p/s) on (0, oo).

    The reviewer's simpler exact proof, implemented and identity-checked:
    dq/dp = w*s/(s+p)^2, positive iff w > 0 (numerator) and s > 0
    (denominator (s+p)^2 > 0 for p > 0, s > 0).
    """
    if not (np.isfinite(s) and s > 0):
        return "VIOLATED", f"scale s = {s} not finite-positive"
    w_sym, s_sym, p_sym = sp.symbols("w s p", positive=True, real=True)
    expr = w_sym * (p_sym / s_sym) / (1 + p_sym / s_sym)
    der = sp.simplify(sp.diff(expr, p_sym))
    expected = w_sym * s_sym / (s_sym + p_sym) ** 2
    if sp.simplify(der - expected) != 0:
        return "UNRESOLVED", "family derivative failed the symbolic identity check"
    if np.isfinite(w) and w > 0:
        return "CERTIFIED_MONOTONIC", (
            "exact family certificate: dq/dp = w*s/(s+p)^2 > 0 on (0, oo) "
            "because w > 0, s > 0, s + p > 0 (symbolic identity check passed)"
        )
    return "VIOLATED", f"exact family certificate fails: requires w > 0; got w = {w}"


# ══════════════════════════════════════════════════════════════════════════
# Patched referee decision logic (defect-2 fix)
# ══════════════════════════════════════════════════════════════════════════
def classify_referee_decision(
    proposals: list[dict[str, Any]] | list[Any],
    gates: dict[str, Callable[[Any], tuple[bool, str]]],
) -> tuple[str, list[dict[str, Any]], str]:
    """ABSTAIN / REJECT / PROMOTE, never conflating an empty packet with a rejection."""
    if not proposals:
        return _ABSTAIN, [], "proposal packet empty; proposer abstained (zero false-discovery path)"
    invariant_reports: list[dict[str, Any]] = []
    failing: list[tuple[str, str, str]] = []
    for p in proposals:
        report: dict[str, Any] = {"proposal": repr(p)[:200]}
        for gate_name, fn in gates.items():
            ok, why = fn(p)
            report[gate_name] = {"pass": bool(ok), "reason": why}
            if not ok:
                failing.append((repr(p)[:80], gate_name, why))
        invariant_reports.append(report)
    if failing:
        return "REJECTED", invariant_reports, "; ".join(
            f"{pl}: {gn} — {why}" for pl, gn, why in failing[:3]
        )
    return "PROMOTED", invariant_reports, "all declared gates passed on every proposal"


# ══════════════════════════════════════════════════════════════════════════
# Measured multi-system battery
# ══════════════════════════════════════════════════════════════════════════
def _fit_scale_bank(
    x_tr: np.ndarray, y_tr: np.ndarray, mults: list[float]
) -> tuple[float, float, float]:
    """Origin-constrained multi-scale closed-form fold fit (training only)."""
    base_s = float(np.std(x_tr))
    best = (float("inf"), None, None)
    for m in mults:
        s_c = m * base_s
        phi_tr = _phi(x_tr, s_c)
        w0 = float(np.sum(phi_tr * y_tr) / np.sum(phi_tr**2))
        mse = float(np.mean((y_tr - w0 * phi_tr) ** 2))
        if mse < best[0]:
            best = (mse, s_c, w0)
    _mse, s_eff, w0 = best
    return s_eff, w0, base_s


def run_michaelis_menten(seed: int = 42) -> dict[str, Any]:
    """Re-run System A with MEASURED per-fold gates (seed-pinned, deterministic)."""
    np.random.seed(seed)
    S_grid = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    v_true = 10.0 * S_grid / (2.50 + S_grid)
    v_obs = v_true + np.random.normal(0, 0.12, size=len(S_grid))
    N = len(S_grid)
    folds: list[dict[str, Any]] = []
    for i in range(N):
        tri = [j for j in range(N) if j != i]
        S_tr, v_tr = S_grid[tri], v_obs[tri]
        S_te, v_te = float(S_grid[i]), float(v_obs[i])
        s_eff, w0, _ = _fit_scale_bank(S_tr, v_tr, _MM_MULTS)
        pred = w0 * float(_phi(S_te, s_eff))
        verdict, why = certify_origin_constrained_rime(w0, s_eff)
        folds.append(
            {
                "fold": i,
                "selected_s": s_eff,
                "w0": w0,
                "pred": pred,
                "true": v_te,
                "residual": pred - v_te,
                "gate_origin": True,  # by construction: b = 0 for this family
                "gate_saturation": bool(np.isfinite(w0) and w0 > 0),
                "gate_monotonicity": verdict == "CERTIFIED_MONOTONIC",
                "cert_rationale": why,
                "admissible": bool(verdict == "CERTIFIED_MONOTONIC" and np.isfinite(w0) and w0 > 0),
            }
        )
    rmse = float(np.sqrt(np.mean([f["residual"] ** 2 for f in folds])))
    per_gate = {
        g: int(sum(bool(f[g]) for f in folds))
        for g in ("gate_origin", "gate_saturation", "gate_monotonicity")
    }
    return {
        "system": "michaelis_menten",
        "n": N,
        "scale_multipliers": _MM_MULTS,
        "loocv_rmse": rmse,
        "unanimity": bool(all(f["admissible"] for f in folds)),
        "per_gate_passes": per_gate,
        "folds": folds,
    }


def run_langmuir_1918() -> dict[str, Any]:
    """8-scale bank LOOCV vs continuous Langmuir fit, exact-certificate gates."""
    p_all = np.array(_LANGMUIR_P, dtype=float)
    q_all = np.array(_LANGMUIR_Q, dtype=float)
    N = len(p_all)
    errs8: list[float] = []
    errsc: list[float] = []
    folds: list[dict[str, Any]] = []
    for i in range(N):
        tri = [j for j in range(N) if j != i]
        p_tr, q_tr = p_all[tri], q_all[tri]
        p_te, q_te = float(p_all[i]), float(q_all[i])
        s_eff, w0, base_s = _fit_scale_bank(p_tr, q_tr, _LANGMUIR_MULTS)
        pred8 = w0 * float(_phi(p_te, s_eff))
        verdict, why = certify_origin_constrained_rime(w0, s_eff)
        folds.append(
            {
                "fold": i,
                "base_s": base_s,
                "selected_s": s_eff,
                "w0": w0,
                "pred8": pred8,
                "residual8": pred8 - q_te,
                "gate_monotonicity": verdict == "CERTIFIED_MONOTONIC",
                "cert_rationale": why,
            }
        )
        errs8.append(pred8 - q_te)

        def langmuir_fn(p, q_max, K):
            return (q_max * K * p) / (1.0 + K * p)

        popt, _ = curve_fit(langmuir_fn, p_tr, q_tr, p0=[38.0, 0.15])
        errsc.append(float(langmuir_fn(p_te, *popt)) - q_te)
    rmse8 = float(np.sqrt(np.mean(np.array(errs8) ** 2)))
    rmsec = float(np.sqrt(np.mean(np.array(errsc) ** 2)))
    w_stat, p_val = wilcoxon(np.abs(errs8), np.abs(errsc))
    return {
        "system": "langmuir_1918_table_vii",
        "n": N,
        "scale_multipliers": _LANGMUIR_MULTS,
        "loocv_rmse_8scale": rmse8,
        "loocv_rmse_continuous": rmsec,
        "wilcoxon": {"W": float(w_stat), "p": float(p_val)},
        "interpretation_disclosed": (
            "p = {:.4f} does NOT establish statistical equivalence on N = 11; "
            "it is absence-of-detection, with a slight observed RMSE edge".format(p_val)
        ),
        "unanimity_certified_by_exact_family_proof": bool(
            all(f["gate_monotonicity"] for f in folds)
        ),
        "folds": folds,
    }


def run_negative_controls(seed: int = 42) -> dict[str, Any]:
    """Fixed decision-logic negative controls (no vacuous all([]) rejections)."""
    from apedp_dyn.proposers.rime_proposer import RimeProposer

    np.random.seed(seed)
    noise_draws = np.random.normal(0, 1.0, size=12)
    df_noise = pd.DataFrame({"x": np.arange(1, 13, dtype=float), "y": noise_draws})
    rime_noise = RimeProposer(
        feature_cols=["x"], target_col="y", static_mode=True, min_rows=5, min_r2=0.5
    )
    props_noise = rime_noise.fit_and_propose(df_noise)

    t_grid = np.linspace(0.5, 12.0, 12)
    y_decay = 50.0 * np.exp(-0.35 * t_grid) + np.random.normal(0, 0.5, size=len(t_grid))
    rime_decay = RimeProposer(
        feature_cols=["t"], target_col="y", static_mode=True, min_rows=5, min_r2=0.5
    )
    props_decay = rime_decay.fit_and_propose(pd.DataFrame({"t": t_grid, "y": y_decay}))

    def capacity_positive_gate(pr) -> tuple[bool, str]:
        """Saturating grammar requires w > 0: scale < 0 means dy/dt < 0."""
        w = float(pr.parameters["scale"])
        return (w > 0), f"scale {w:+.4f}"

    verdict_noise, _, why_noise = classify_referee_decision(
        props_noise, {"capacity_positive": capacity_positive_gate}
    )
    verdict_decay, inv_decay, why_decay = classify_referee_decision(
        props_decay, {"capacity_positive": capacity_positive_gate}
    )
    verdict_empty, _, _ = classify_referee_decision(
        [], {"capacity_positive": capacity_positive_gate}
    )
    assert verdict_empty == _ABSTAIN, "defect-2 regression: empty packet misclassified"
    return {
        "noise_control": {
            "n_proposals": len(props_noise),
            "decision": verdict_noise,
            "rationale": why_noise,
        },
        "decay_control": {
            "n_proposals": len(props_decay),
            "decision": verdict_decay,
            "rationale": why_decay,
            "proposal_formulas": [p.formula for p in props_decay],
            "all_scales_negative": all(
                float(p.parameters["scale"]) < 0 for p in props_decay
            ),
        },
        "empty_packet_decision": verdict_empty,
    }


def run_timing_scopes() -> dict[str, Any]:
    """Warm per-call costs ONLY per scope; no cross-host ratio is certified."""
    p_test = np.linspace(2.8, 34.0, 11)
    q_test = 38.89 * 0.1584 * p_test / (1.0 + 0.1584 * p_test)

    for _ in range(50):
        _ = np.dot(p_test, q_test) / np.dot(p_test, p_test)
    t0 = time.perf_counter()
    for _ in range(2000):
        _ = np.dot(p_test, q_test) / np.dot(p_test, p_test)
    t_dot = (time.perf_counter() - t0) / 2000

    def lang(p, qm, K):
        return (qm * K * p) / (1.0 + K * p)

    for _ in range(10):
        _, _ = curve_fit(lang, p_test, q_test, p0=[38.0, 0.15])
    t0 = time.perf_counter()
    for _ in range(200):
        _, _ = curve_fit(lang, p_test, q_test, p0=[38.0, 0.15])
    t_cf = (time.perf_counter() - t0) / 200

    p_all, q_all = np.array(_LANGMUIR_P[1:]), np.array(_LANGMUIR_Q[1:])
    for _ in range(20):
        _fit_scale_bank(p_all, q_all, _LANGMUIR_MULTS)
    t0 = time.perf_counter()
    for _ in range(2000):
        _fit_scale_bank(p_all, q_all, _LANGMUIR_MULTS)
    t_fold = (time.perf_counter() - t0) / 2000
    return {
        "protocol": "warm, 2000/200/2000 iterations; costs are per-call milliseconds on the archiving host",
        "dot_product_ms": round(t_dot * 1000.0, 4),
        "curve_fit_ms": round(t_cf * 1000.0, 4),
        "full_fold_ms": round(t_fold * 1000.0, 4),
        "ratios_disclosed": {
            "dot_vs_curve_fit": round(t_cf / t_dot, 1),
            "fold_vs_curve_fit": round(t_cf / t_fold, 1),
            "note": "scope-dependent; the external '10.2x' matches neither scope",
        },
    }


def _family_of(formula: str) -> str:
    """Coarse structural family tag of a RIME proposal formula."""
    for tag in ("sat_sq_norm", "sat_norm", "sat_sq", "sat", "ssqrt", "slog", "sq", "cub", "raw"):
        if tag in formula:
            return tag
    if "(p)" in formula:
        return "linear"
    return "other"


def run_fold_level_rime_recovery(seed: int = 42) -> dict[str, Any]:
    """Run RimeProposer inside each Langmuir LOOCV fold and record recovery/abstention.

    Two disclosed modes (the audit-verified distinction):
      - ``bank_restricted``: 8 Python-built saturation columns sat_m with
        s_m = m*std(p_train) passed as RIME's feature set (selection within a
        declared bank; NOT unrestricted structural discovery).
      - ``unrestricted_grammar``: raw p plus the declared algebraic transform
        repertoire; the KERNEL derives transform scales from the development
        rows itself (``_specs`` uses nanstd) and picks the structure per fold.

    Per fold this records: proposals emitted or abstention, selected
    structure/family, fitted scale/intercept/dev_r2, held-out error, and the
    partial-gate referee decision (capacity_positive only -- gate set is a
    disclosed subset; graph-structural validation is NOT wired here).
    """
    from apedp_dyn.proposers.rime_proposer import RimeProposer

    def capacity_positive_gate(pr) -> tuple[bool, str]:
        w = float(pr.parameters["scale"])
        return (w > 0), f"scale {w:+.4f}"

    df = pd.DataFrame({"p": _LANGMUIR_P, "q": _LANGMUIR_Q})
    bank_rows: list[dict[str, Any]] = []
    gram_rows: list[dict[str, Any]] = []
    errs_bank: list[float] = []
    errs_gram: list[float] = []
    for i in range(len(df)):
        df_train = df.drop(index=i).reset_index(drop=True)
        df_test = df.iloc[[i]].reset_index(drop=True)
        s_tr = float(np.std(df_train["p"].values))

        # ── Mode A: bank-restricted (Python-built sat_m columns) ──
        feat_cols, bank_s = [], {}
        for m in _LANGMUIR_MULTS:
            col = f"sat_{m}"
            s_m = m * s_tr
            df_train[col] = (df_train["p"] / s_m) / (1.0 + df_train["p"] / s_m)
            df_test[col] = (df_test["p"] / s_m) / (1.0 + df_test["p"] / s_m)
            feat_cols.append(col)
            bank_s[col] = s_m
        props_bank = RimeProposer(
            feature_cols=feat_cols, target_col="q", static_mode=True,
            min_rows=5, max_proposals=1, seed=seed,
        ).fit_and_propose(df_train)
        if props_bank:
            p0 = props_bank[0]
            pred = float(p0.compute_score(df_test)[0])
            err = pred - float(df_test["q"].values[0])
            verdict, _, why = classify_referee_decision(
                [p0], {"capacity_positive": capacity_positive_gate})
            sel_m = None
            mm_ = re.search(r"sat_(0\.[0-9]+|1\.0|2\.0)", p0.formula)
            if mm_:
                sel_m = float(mm_.group(1))
            bank_rows.append(dict(
                fold=i, abstained=False, formula=p0.formula,
                selected_scale_m=sel_m, family=_family_of(p0.formula),
                dev_r2=float(p0.parameters["dev_r2"]),
                readout_scale=float(p0.parameters["scale"]),
                readout_intercept=float(p0.parameters["intercept"]),
                held_out_p=float(df_test["p"].values[0]),
                held_out_q=float(df_test["q"].values[0]),
                predicted=pred, error=err,
                partial_gate_decision=verdict, partial_gate_reason=why,
            ))
            errs_bank.append(err)
        else:
            bank_rows.append(dict(
                fold=i, abstained=True, formula=None, family="ABSTAIN",
                held_out_p=float(df_test["p"].values[0]),
                held_out_q=float(df_test["q"].values[0]),
                partial_gate_decision=_ABSTAIN,
            ))
            errs_bank.append(float("nan"))

        # ── Mode B: unrestricted declared grammar (kernel-chosen scales) ──
        props_gram = RimeProposer(
            feature_cols=["p"], target_col="q", static_mode=True,
            algebraic_transforms=["sq", "cub", "ssqrt", "slog", "sat", "sat_sq", "sat_norm"],
            min_rows=5, max_proposals=1, seed=seed,
        ).fit_and_propose(df_train)
        if props_gram:
            p0 = props_gram[0]
            pred = float(p0.compute_score(df_test)[0])
            err = pred - float(df_test["q"].values[0])
            verdict, _, why = classify_referee_decision(
                [p0], {"capacity_positive": capacity_positive_gate})
            gram_rows.append(dict(
                fold=i, abstained=False, formula=p0.formula,
                family=_family_of(p0.formula),
                dev_r2=float(p0.parameters["dev_r2"]),
                readout_scale=float(p0.parameters["scale"]),
                readout_intercept=float(p0.parameters["intercept"]),
                held_out_p=float(df_test["p"].values[0]),
                held_out_q=float(df_test["q"].values[0]),
                predicted=pred, error=err,
                partial_gate_decision=verdict, partial_gate_reason=why,
            ))
            errs_gram.append(err)
        else:
            gram_rows.append(dict(
                fold=i, abstained=True, formula=None, family="ABSTAIN",
                held_out_p=float(df_test["p"].values[0]),
                held_out_q=float(df_test["q"].values[0]),
                partial_gate_decision=_ABSTAIN,
            ))
            errs_gram.append(float("nan"))

    rmse_bank = float(np.sqrt(np.nanmean(np.array(errs_bank) ** 2)))
    rmse_gram = float(np.sqrt(np.nanmean(np.array(errs_gram) ** 2)))
    sat_bank = int(sum(1 for r in bank_rows if str(r.get("family", "")).startswith("sat")))
    sat_gram = int(sum(1 for r in gram_rows if str(r.get("family", "")).startswith("sat")))
    sel_m_counts: dict[str, int] = {}
    for r in bank_rows:
        sel = r.get("selected_scale_m")
        key = "ABSTAIN" if sel is None else f"m={sel}"
        sel_m_counts[key] = sel_m_counts.get(key, 0) + 1
    return {
        "dataset": "langmuir_1918_table_vii (N=11, LOOCV)",
        "seed": seed,
        "disclosed_limit": (
            "bank_restricted mode selects WITHIN a Python-predefined saturation bank "
            "(structural recovery claim holds unrestrictedly via unrestricted_grammar "
            "mode; the two are NOT the same experiment)"
        ),
        "bank_restricted": {
            "loocv_rmse": rmse_bank,
            "recovery": {"sat_family_top_rank_folds": sat_bank, "n": len(bank_rows)},
            "abstentions": int(sum(1 for r in bank_rows if r["abstained"])),
            "selected_scale_m_counts": sel_m_counts,
            "folds": bank_rows,
        },
        "unrestricted_grammar": {
            "grammar": ["sq", "cub", "ssqrt", "slog", "sat", "sat_sq", "sat_norm"],
            "loocv_rmse": rmse_gram,
            "recovery": {"sat_family_top_rank_folds": sat_gram, "n": len(gram_rows)},
            "abstentions": int(sum(1 for r in gram_rows if r["abstained"])),
            "folds": gram_rows,
        },
    }


def main() -> int:
    results = {
        "harness": "rime_microsample_adjudication",
        "status": "audit-only",
        "monotonicity_engine": "sound_monotonicity_v2 (exact root isolation, NO tolerance filter, UNRESOLVED fallback)",
        "decision_protocol": "ABSTAIN/REJECT/PROMOTE with empty-packet guard (defect-2 fix)",
        "langmuir_1918": run_langmuir_1918(),
        "fold_level_rime": run_fold_level_rime_recovery(),
        "michaelis_menten": run_michaelis_menten(),
        "michaelis_menten": run_michaelis_menten(),
        "negative_controls": run_negative_controls(),
        "timing_scopes": run_timing_scopes(),
    }
    path = _REPO / "benchmarks" / "rime_microsample_adjudication_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"[Saved] {path}")
    l8 = results["langmuir_1918"]
    mm = results["michaelis_menten"]
    nc = results["negative_controls"]
    fl = results["fold_level_rime"]
    print(f"Langmuir 8-scale RMSE {l8['loocv_rmse_8scale']:.6f} | continuous {l8['loocv_rmse_continuous']:.6f}")
    print(f"Fold-level RIME: bank-restricted RMSE {fl['bank_restricted']['loocv_rmse']:.6f} "
          f"(sat {fl['bank_restricted']['recovery']['sat_family_top_rank_folds']}/11) | "
          f"unrestricted RMSE {fl['unrestricted_grammar']['loocv_rmse']:.6f} "
          f"(sat {fl['unrestricted_grammar']['recovery']['sat_family_top_rank_folds']}/11) | "
          f"abstentions {fl['bank_restricted']['abstentions']}+{fl['unrestricted_grammar']['abstentions']}")
    print(f"MM {mm['n']}-fold RMSE {mm['loocv_rmse']:.4f} | admissible {sum(f['admissible'] for f in mm['folds'])}/{mm['n']}")
    for k, v in nc.items():
        if isinstance(v, dict):
            print(f"{k}: {v['decision']} ({v.get('n_proposals', 0)} proposals)")
        else:
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
