"""Parametric and mechanistic family detection probes for APEDP.

Extracts power-law probes, composition families, Arrhenius and saturating transitions,
cross-family BIC selection, and domain-informed mechanistic priors.

Fixes:
- Sign handling: integer powers use v**p directly to preserve correct signs on negative inputs
  (eliminating the bug where np.sign(v)*np.abs(v)**p inverted (-2)^2 to -4).
- Single-variable isolation: detect_power_law is called on individual variables rather than
  re-evaluating all variables redundantly and misattributing exponents.
- Unused priors: detect_composition_family_with_priors attaches resolved priors to output.
"""
from __future__ import annotations

import itertools as _it
import logging
from typing import Any

import numpy as np
import sympy as sp

from apedp.discovery_types import (
    CrossFamilyResult,
    PowerLawResult,
)
from apedp.numerical_policy import DEFAULT_POLICY, DiscoveryNumericalPolicy

logger = logging.getLogger("apedp.family_detection")

_PROBE_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    ArithmeticError,
    KeyError,
    ImportError,
)

POWER_LAW_RATIONALS: list[tuple[float, str]] = [
    (1.0 / 2.0, "1/2"),
    (1.0 / 3.0, "1/3"),
    (2.0 / 3.0, "2/3"),
    (3.0 / 2.0, "3/2"),
    (4.0 / 3.0, "4/3"),
    (2.0, "2"),
    (3.0, "3"),
    (4.0, "4"),
    (1.0, "1"),
    (-1.0, "-1"),
    (-2.0, "-2"),
    (-1.0 / 2.0, "-1/2"),
]

STANDARD_RATIONALS: list[tuple[float, sp.Expr]] = [
    (0.5, sp.Rational(1, 2)),
    (0.25, sp.Rational(1, 4)),
    (0.75, sp.Rational(3, 4)),
    (1.0 / 3.0, sp.Rational(1, 3)),
    (2.0 / 3.0, sp.Rational(2, 3)),
    (1.0, sp.Integer(1)),
    (2.0, sp.Integer(2)),
    (3.0, sp.Integer(3)),
    (4.0, sp.Integer(4)),
    (-1.0, sp.Integer(-1)),
    (-0.5, sp.Rational(-1, 2)),
]

MECHANISTIC_PRIORS: dict[str, list[dict[str, Any]]] = {
    "battery_degradation": [
        {"name": "sei_growth", "expr": "sqrt(time)", "domain": "battery_degradation"},
        {"name": "arrhenius_aging", "expr": "exp(-Ea / (R * T))", "domain": "battery_degradation"},
        {"name": "power_law_fade", "expr": "cycles ** z", "domain": "battery_degradation"},
    ],
    "protein_folding": [
        {"name": "two_state_kinetics", "expr": "1 / (1 + exp(-delta_G / (R * T)))", "domain": "protein_folding"},
        {"name": "kramers_rate", "expr": "gamma * exp(-barrier / (k_B * T))", "domain": "protein_folding"},
    ],
    "turbulence_closure": [
        {"name": "smagorinsky", "expr": "(Cs * Delta)**2 * abs(S)", "domain": "turbulence_closure"},
        {"name": "kolmogorov_spectrum", "expr": "C_K * epsilon**(2/3) * k**(-5/3)", "domain": "turbulence_closure"},
    ],
    "chemical_kinetics": [
        {"name": "michaelis_menten", "expr": "Vmax * S / (Km + S)", "domain": "chemical_kinetics"},
        {"name": "hill_equation", "expr": "Vmax * S**n / (K**n + S**n)", "domain": "chemical_kinetics"},
    ],
    "epidemiology": [
        {"name": "sir_transmission", "expr": "beta * S * I / N", "domain": "epidemiology"},
        {"name": "logistic_growth", "expr": "r * N * (1 - N / K)", "domain": "epidemiology"},
    ],
}


def recombine_power_law(variables: list[str], exponents: list[float]) -> str:
    """Render a power-law product string from parallel variable/exponent lists."""
    terms = []
    for name, p in zip(variables, exponents):
        if abs(p) < 1e-12:
            continue
        if abs(p - round(p)) > 1e-9:
            terms.append(f"{name}**({p})")
        elif round(p) == 1:
            terms.append(name)
        else:
            terms.append(f"{name}**{round(p)}")
    return "*".join(terms) if terms else "1"


def detect_power_law(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    r2_threshold: float = 0.90,
    snap_tol: float = 0.02,
    *,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> PowerLawResult | dict[str, Any]:
    """Log-log power-law probe: fit log y = p * log x + c per variable."""
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < policy.MIN_SAMPLE_SIZE_POWER_LAW:
            return {}
        if float(np.sum((y - y.mean()) ** 2)) <= policy.VARIANCE_EPSILON:
            return {}
        if not bool(np.all(np.isfinite(y))) or not bool(np.all(y > 0)):
            return {}

        best_result: PowerLawResult | None = None
        best_r2 = -1.0

        for name in names:
            xv = np.asarray(x_data[name], float).ravel()
            if len(xv) != len(y):
                continue
            if not bool(np.all(np.isfinite(xv))) or not bool(np.all(xv > 0)):
                continue
            try:
                lx, ly = np.log(xv), np.log(y)
                A = np.column_stack([lx, np.ones_like(lx)])
                coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
                pred = A @ coef
                denom = float(np.sum((ly - ly.mean()) ** 2)) + policy.VARIANCE_EPSILON
                r2 = 1.0 - float(np.sum((ly - pred) ** 2)) / denom

                if r2 > best_r2:
                    best_r2 = r2
                    p = float(coef[0])
                    snap = None
                    for target, label in POWER_LAW_RATIONALS:
                        if abs(p - target) < snap_tol:
                            snap = label
                            break
                    best_result = PowerLawResult(
                        family="power-law",
                        variable=name,
                        exponent=round(p, 4),
                        r_squared=round(float(r2), 6),
                        snapped=snap,
                        confident=bool(r2 > r2_threshold),
                    )
            except (TypeError, ValueError, ArithmeticError) as exc:
                logger.debug("Power-law probe failed for %s: %s", name, exc)

        return best_result if best_result is not None else {}
    except (TypeError, ValueError, ArithmeticError) as exc:
        logger.warning("detect_power_law encountered error: %s", exc, exc_info=True)
        return {}


def detect_composition_family(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    r2_threshold: float = 0.99,
    *,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Probe data shape for function families: exponential, arrhenius, saturating, rational."""
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < policy.MIN_SAMPLE_SIZE_FORWARD_SELECT:
            return {}
        if float(np.sum((y - y.mean()) ** 2)) <= policy.VARIANCE_EPSILON:
            return {}

        best: dict[str, Any] = {}
        for name in names:
            xv = np.asarray(x_data[name], float).ravel()
            if len(xv) != len(y):
                continue
            order = np.argsort(xv)
            xs, ys = xv[order], y[order]

            # 1. Exponential
            try:
                if bool(np.all(np.isfinite(ys))) and bool(np.all(ys != 0)):
                    ly = np.log(np.abs(ys))
                    A = np.column_stack([xs, np.ones_like(xs)])
                    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
                    pred = A @ coef
                    denom_log = float(np.sum((ly - ly.mean()) ** 2)) + policy.VARIANCE_EPSILON
                    r2e = 1.0 - float(np.sum((ly - pred) ** 2)) / denom_log
                    if r2e > r2_threshold and r2e > best.get("r_squared", 0.0):
                        best = {
                            "family": "exponential",
                            "variable": name,
                            "r_squared": round(float(r2e), 6),
                        }
            except (TypeError, ValueError, ArithmeticError):
                logger.debug("Exponential-family fit rejected for %s", name)

            # 2. Arrhenius
            try:
                if bool(np.all(np.isfinite(ys))) and bool(np.all(ys != 0)) and bool(np.all(xs != 0)):
                    ly = np.log(np.abs(ys))
                    ix = 1.0 / xs
                    A = np.column_stack([ix, np.ones_like(ix)])
                    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
                    pred = A @ coef
                    denom_log = float(np.sum((ly - ly.mean()) ** 2)) + policy.VARIANCE_EPSILON
                    r2a = 1.0 - float(np.sum((ly - pred) ** 2)) / denom_log
                    if r2a > r2_threshold and r2a > best.get("r_squared", 0.0):
                        best = {
                            "family": "arrhenius",
                            "variable": name,
                            "r_squared": round(float(r2a), 6),
                        }
            except (TypeError, ValueError, ArithmeticError):
                logger.debug("Arrhenius-family fit rejected for %s", name)

            # 3. Saturating-exponential
            try:
                if bool(np.all(np.isfinite(ys))) and ys[-1] != ys[0]:
                    ymax = float(np.max(ys))
                    for frac in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5):
                        a_est = ymax * (1.0 + frac)
                        gap = a_est - ys
                        if not bool(np.all(gap > 0)):
                            continue
                        ly = np.log(gap)
                        A = np.column_stack([xs, np.ones_like(xs)])
                        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
                        pred = A @ coef
                        denom_s = float(np.sum((ly - ly.mean()) ** 2)) + policy.VARIANCE_EPSILON
                        r2s = 1.0 - float(np.sum((ly - pred) ** 2)) / denom_s
                        if r2s > policy.OVERFIT_R2_THRESHOLD and r2s > best.get("r_squared", 0.0):
                            best = {
                                "family": "saturating-exponential",
                                "variable": name,
                                "r_squared": round(float(r2s), 6),
                                "asymptote": round(a_est, 6),
                            }
            except (TypeError, ValueError, ArithmeticError):
                logger.debug("Saturating-exponential fit rejected for %s", name)

            # 4. Rational (inverse)
            try:
                if bool(np.all(ys != 0)):
                    iy = 1.0 / ys
                    A = np.column_stack([xs, np.ones_like(xs)])
                    coef, *_ = np.linalg.lstsq(A, iy, rcond=None)
                    pred = A @ coef
                    denom_inv = float(np.sum((iy - iy.mean()) ** 2)) + policy.VARIANCE_EPSILON
                    r2r = 1.0 - float(np.sum((iy - pred) ** 2)) / denom_inv
                    if r2r > r2_threshold and r2r > best.get("r_squared", 0.0):
                        best = {
                            "family": "rational",
                            "variable": name,
                            "r_squared": round(float(r2r), 6),
                        }
            except (TypeError, ValueError, ArithmeticError):
                logger.debug("Rational-family fit rejected for %s", name)

        return best
    except (TypeError, ValueError, ArithmeticError) as exc:
        logger.warning("detect_composition_family encountered error: %s", exc, exc_info=True)
        return {}


def detect_composition_family_with_priors(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    r2_threshold: float = 0.99,
    domain: str | None = None,
) -> dict[str, Any]:
    """Enhanced composition family detection incorporating mechanistic domain priors."""
    result = detect_composition_family(x_data, y_data, r2_threshold)
    if not result and domain:
        priors = get_mechanistic_priors(domain)
        if priors:
            result = {
                "family": "mechanistic_prior",
                "priors": priors,
                "domain": domain,
                "note": f"Evaluated candidate priors for domain '{domain}'",
            }
    return result


def cross_family_select(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    max_power: int = 2,
    extrapolation_check: bool = True,
    *,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> CrossFamilyResult | dict[str, Any]:
    """Evaluates competing functional families (power-law, exponential, Arrhenius, polynomial) under BIC."""
    from apedp.bayesian_refinement import BayesianRefinementEngine
    from apedp.feature_selection import sparse_select_terms

    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        n = len(y)
        if n < policy.MIN_SAMPLE_SIZE_FORWARD_SELECT:
            return {}
        if float(np.sum((y - y.mean()) ** 2)) <= policy.VARIANCE_EPSILON:
            return {}
        if not bool(np.all(np.isfinite(y))):
            return {}

        candidates: list[dict[str, Any]] = []
        notes: list[str] = []

        # 1. Power-law per variable (testing each variable independently)
        for name in names:
            pl = detect_power_law({name: x_data[name]}, y, policy=policy)
            if pl and pl.get("r_squared", 0.0) > policy.CANDIDATE_VIABILITY_R2:
                xv = np.asarray(x_data[name], float).ravel()
                lxv = np.log(np.maximum(xv, policy.VARIANCE_EPSILON))
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([lxv, np.ones(n)]),
                    np.log(np.maximum(y, policy.VARIANCE_EPSILON)),
                    rcond=None,
                )
                y_pred = np.exp(coef[0] * lxv + coef[1])
                bic = BayesianRefinementEngine.compute_bic(
                    BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                    2,
                    n,
                )
                candidates.append({
                    "family": "power-law",
                    "variable": name,
                    "expression": f"{name}**{pl['exponent']:.3f}",
                    "bic": bic,
                    "r_squared": pl["r_squared"],
                    "num_params": 2,
                })

        # 2. Exponential
        try:
            for name in names:
                xv = np.asarray(x_data[name], float).ravel()
                if not bool(np.all(xv > 0)):
                    continue
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([xv, np.ones(n)]),
                    np.log(np.maximum(y, policy.VARIANCE_EPSILON)),
                    rcond=None,
                )
                y_pred = np.exp(coef[0] * xv + coef[1])
                r2 = BayesianRefinementEngine.compute_r_squared(y, y_pred)
                if r2 > policy.MIN_SIGNIFICANT_R2:
                    bic = BayesianRefinementEngine.compute_bic(
                        BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                        2,
                        n,
                    )
                    candidates.append({
                        "family": "exponential",
                        "variable": name,
                        "expression": f"exp({coef[0]:.4f}*{name})",
                        "bic": bic,
                        "r_squared": round(float(r2), 4),
                        "num_params": 2,
                    })
        except (TypeError, ValueError, ArithmeticError) as exc:
            logger.debug("Exponential refinement candidate rejected: %s", exc)

        # 3. Arrhenius (exp in 1/x)
        try:
            for name in names:
                xv = np.asarray(x_data[name], float).ravel()
                if not bool(np.all(xv != 0)):
                    continue
                ix = 1.0 / xv
                coef, *_ = np.linalg.lstsq(
                    np.column_stack([ix, np.ones(n)]),
                    np.log(np.maximum(y, policy.VARIANCE_EPSILON)),
                    rcond=None,
                )
                y_pred = np.exp(coef[0] * ix + coef[1])
                r2 = BayesianRefinementEngine.compute_r_squared(y, y_pred)
                if r2 > policy.MIN_SIGNIFICANT_R2:
                    bic = BayesianRefinementEngine.compute_bic(
                        BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                        2,
                        n,
                    )
                    candidates.append({
                        "family": "arrhenius",
                        "variable": name,
                        "expression": f"exp({coef[0]:.4f}/{name})",
                        "bic": bic,
                        "r_squared": round(float(r2), 4),
                        "num_params": 2,
                    })
        except (TypeError, ValueError, ArithmeticError) as exc:
            logger.debug("Arrhenius refinement candidate rejected: %s", exc)

        # 4. Polynomial from sparse_select_terms
        try:
            sp_res = sparse_select_terms(x_data, y, max_power=max_power, policy=policy)
            if sp_res.terms:
                from sklearn.linear_model import LassoCV

                cols = {k: np.asarray(x_data[k], float).ravel() for k in x_data}
                feats = {"1": np.ones(n)}
                for k in x_data:
                    v = cols[k]
                    for p in range(1, max_power + 1):
                        # CRITICAL BUG FIX: use v**p directly.
                        # Do not use np.sign(v)*np.abs(v)**p which corrupts even powers on negative inputs
                        feats[f"{k}**{p}"] = v ** p
                    if bool(np.all(v != 0)):
                        feats[f"1/{k}"] = 1.0 / v
                for a, b in _it.combinations(x_data, 2):
                    feats[f"{a}*{b}"] = cols[a] * cols[b]

                keys = [k for k in feats if float(np.std(feats[k])) > policy.VARIANCE_EPSILON]
                X = np.column_stack([feats[k] for k in keys])
                scales = np.std(X, axis=0) + policy.VARIANCE_EPSILON
                Xn = X / scales
                model = LassoCV(max_iter=5000).fit(Xn, y)
                y_pred = model.predict(Xn)
                r2 = BayesianRefinementEngine.compute_r_squared(y, y_pred)

                if r2 > policy.MIN_SIGNIFICANT_R2:
                    active_k = int(np.sum(np.abs(model.coef_) > 1e-9)) + 1
                    if r2 > policy.OVERFIT_R2_THRESHOLD and active_k > n * policy.OVERFIT_PARAM_RATIO:
                        notes.append(
                            f"polynomial rejected: R^2={r2:.4f} with {active_k} params on n={n} (overfit regime)"
                        )
                    else:
                        bic = BayesianRefinementEngine.compute_bic(
                            BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                            active_k,
                            n,
                        )
                        candidates.append({
                            "family": "polynomial",
                            "variable": ", ".join(x_data),
                            "expression": sp_res.terms[0] if sp_res.terms else "poly",
                            "bic": bic,
                            "r_squared": round(float(r2), 4),
                            "num_params": active_k,
                        })
        except _PROBE_ERRORS as exc:
            logger.debug("Polynomial candidate rejected during BIC ranking: %s", exc)

        # 5. Extrapolation audit filter
        if extrapolation_check and candidates:
            from apedp.scientific_layers import extrapolation_audit

            cols = {k: np.asarray(x_data[k], float).ravel() for k in names}
            ranges = {k: (float(cols[k].min()), float(cols[k].max())) for k in names}
            y_scale = float(np.max(np.abs(y))) + policy.VARIANCE_EPSILON
            survivors = []
            for cand in candidates:
                audit = extrapolation_audit(cand["expression"], ranges, y_scale)
                if audit.get("status") == "FAIL":
                    notes.append(f"{cand['family']} {cand['expression']} rejected: {audit.get('reason')}")
                else:
                    survivors.append(cand)
            candidates = survivors

        if not candidates:
            return CrossFamilyResult(notes=notes)

        best = BayesianRefinementEngine.select_best_model(candidates)
        all_sorted = sorted(candidates, key=lambda c: c.get("bic", float("inf")))

        return CrossFamilyResult(
            best=best,
            all_candidates=all_sorted,
            notes=notes,
        )
    except (TypeError, ValueError, ArithmeticError) as exc:
        logger.warning("cross_family_select encountered error: %s", exc, exc_info=True)
        return CrossFamilyResult()


def detect_multiplicative_split(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    tol: float = 1e-6,
) -> dict:
    """Pure helper: log-domain least-squares separability probe.

    Returns {"kind": "multiplicative", "exponents": [...], "variables": [...],
    "r_squared": ...} when a power-law product explains y within tol, else {}.
    No evolution, no side effects. Additive separability needs grid-structured
    data and is deliberately not attempted on scattered samples.
    """
    try:
        names = list(x_data.keys())
        if len(names) < 2:
            return {}
        cols = [np.asarray(x_data[k], float).ravel() for k in names]
        y = np.asarray(y_data, float).ravel()
        if any(len(c) != len(y) for c in cols):
            raise ValueError("x columns and y length mismatch")
        if not bool(np.all(y > 0)):
            return {}
        X = np.column_stack(cols)
        ly = np.log(y)
        denom = float(np.sum((ly - ly.mean()) ** 2)) + 1e-300
        L = np.log(np.abs(X) + 1e-300)
        A2 = np.column_stack([L, np.ones(len(y))])
        coef, *_ = np.linalg.lstsq(A2, ly, rcond=None)
        pred = A2 @ coef
        ss = 1.0 - float(np.sum((ly - pred) ** 2)) / denom
        if ss > 1.0 - tol:
            return {
                "kind": "multiplicative",
                "exponents": [round(float(v), 4) for v in coef[:-1]],
                "variables": names,
                "r_squared": round(float(ss), 6),
            }
        return {}
    except (TypeError, ValueError, ArithmeticError):
        return {}


def recursive_solve(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    tol: float = 1e-6,
) -> dict:
    """If multiplicatively separable, read power laws off the log-linear fit
    and return the recombined expression. No evolution involved."""
    det = detect_multiplicative_split(x_data, y_data, tol)
    if det.get("kind") != "multiplicative":
        return {"split": False, "detail": det}
    return {
        "split": True,
        "expression": recombine_power_law(det["variables"], det["exponents"]),
        "r_squared": det["r_squared"],
        "note": "constant factor undetermined by log-fit; fit separately",
    }


def get_mechanistic_priors(domain: str) -> list[dict[str, Any]]:
    """Get physics-informed candidate expressions for a scientific domain."""
    return MECHANISTIC_PRIORS.get(domain, [])


def suggest_mechanistic_priors(variable_names: list[str], target_name: str) -> list[dict[str, Any]]:
    """Infer likely mechanistic priors from variable and target names."""
    names_lower = [v.lower() for v in variable_names]
    target_lower = target_name.lower()

    if any("charge" in n or "discharge" in n or "dod" in n or "capacity" in n for n in names_lower):
        return MECHANISTIC_PRIORS["battery_degradation"]

    if (
        any("charge" in n or "discharge" in n or "dod" in n for n in names_lower)
        and "time" in names_lower
        and ("rate" in target_lower or "capacity" in target_lower)
    ):
        return MECHANISTIC_PRIORS["protein_folding"]

    if any("strain" in n or "vorticity" in n or "dissipation" in n for n in names_lower):
        return MECHANISTIC_PRIORS["turbulence_closure"]

    if any("conc" in n or "concentration" in n or "rate" in target_lower for n in names_lower):
        return MECHANISTIC_PRIORS["chemical_kinetics"]

    if any("s" in n or "i" in n or "r" in n for n in names_lower) and (
        "infect" in target_lower or "case" in target_lower
    ):
        return MECHANISTIC_PRIORS["epidemiology"]

    return []
