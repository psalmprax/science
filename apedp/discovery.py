# Databricks notebook source
"""APEDP Discovery: Multi-Variable Symbolic Regression Engine & Bayesian Model Selection."""
from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import sympy as sp

from apedp.core import ClosedLoopAssuranceSystem
from apedp.gated_assurance import ParetoFrontierTracker, ScientificImprovementMetric
from apedp.safety import quiet_float_warnings

logger = logging.getLogger(__name__)

#: Failures a best-effort probe may legitimately hit: numeric/operational error
#: from numpy or sympy, malformed or absent input, a missing mapping key, or an
#: optional dependency that is not installed. ``numpy.linalg.LinAlgError`` and
#: ``sympy.SympifyError`` are subclasses of ``ValueError``, and
#: ``ZeroDivisionError``/``OverflowError`` are subclasses of ``ArithmeticError``,
#: so those need no separate entry. Anything outside this set is a defect and
#: must propagate instead of being reported as a failed probe (mirrors
#: ``gated_assurance._PROBE_ERRORS``; see docs/INDEPENDENT_REVIEW_2026-09-23.md
#: §4.2).
_PROBE_ERRORS = (RuntimeError, TypeError, ValueError, ArithmeticError,
                 KeyError, ImportError)


def bic_forward_select(columns: dict[str, np.ndarray],
                       y_data: np.ndarray,
                       max_terms: int | None = None) -> dict:
    """Greedy forward subset selection minimizing BIC (complexity-penalized).

    Starts from intercept-only and adds the single feature giving the largest
    BIC improvement per step; stops when nothing improves. Unlike LassoCV,
    every retained term must individually justify itself against the
    k*ln(n) penalty, which cures the small-n soup (9 params on 7 points can
    never survive: max_terms defaults to n-2 and each addition must pay).
    Near-ties (|dBIC| < 2, Kass-Raftery "weak evidence") defer to the
    structurally simpler term (plain powers/reciprocals before fractional
    powers before exp/log), so noise cannot crown ``x**1.5`` over ``x**1``.
    Deterministic. Raises no exception outward.
    """
    try:
        names = sorted(columns)
        y = np.asarray(y_data, float).ravel()
        n = len(y)
        if n < 3 or not names:
            return {"terms": [], "note": "insufficient data"}
        cols = {}
        for k in names:
            v = np.asarray(columns[k], float).ravel()
            if v.shape == (n,) and bool(np.all(np.isfinite(v))):
                cols[k] = v
        if not cols:
            return {"terms": [], "note": "no finite features"}
        names = sorted(cols)
        if max_terms is None:
            max_terms = max(1, n - 2)
        max_terms = max(1, min(int(max_terms), n - 2, len(names)))
        def _simplicity(name: str) -> tuple:
            if name.startswith(("exp(", "log(")):
                return (2, name)
            if "**(" in name:
                return (1, name)
            return (0, name)

        ssr0 = float(np.sum((y - y.mean()) ** 2)) + 1e-300
        best_bic = n * np.log(ssr0 / n) + 1 * np.log(n)
        selected: list = []
        while len(selected) < max_terms:
            gains: list = []
            for k in names:
                if k in selected:
                    continue
                trial = [*selected, k]
                X = np.column_stack([cols[c] for c in trial] + [np.ones(n)])
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                ssr = float(np.sum((y - X @ coef) ** 2)) + 1e-300
                bic = n * np.log(ssr / n) + (len(trial) + 1) * np.log(n)
                gains.append((best_bic - bic, k, coef))
            gains = [(g, k, c) for g, k, c in gains if g > 1e-9]
            if not gains:
                break
            top = max(g for g, _, _ in gains)
            # Kass-Raftery zone: contenders within 2 BIC points of the best
            # are evidentially tied -- pick the structurally simplest.
            tied = [(k, c) for g, k, c in gains if top - g < 2.0]
            best_k, _best_coef = min(tied, key=lambda kc: _simplicity(kc[0]))
            best_gain = next(g for g, k, _ in gains if k == best_k)
            selected.append(best_k)
            best_bic -= best_gain
        if not selected:
            return {"terms": [], "bic": round(best_bic, 4),
                    "note": "no feature improves on intercept-only BIC"}
        X = np.column_stack([cols[c] for c in selected] + [np.ones(n)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return {"terms": selected,
                "coef": [round(float(v), 6) for v in coef[:-1]],
                "intercept": round(float(coef[-1]), 6),
                "bic": round(float(best_bic), 4)}
    except _PROBE_ERRORS as e:
        return {"terms": [], "note": f"bic-forward failed: {e}"}


def check_data_adequacy(x_data: dict[str, np.ndarray],
                        y_data: np.ndarray,
                        min_range_ratio: float = 4.0) -> dict:
    """Data-adequacy gate for discovery runs (auto-enforcement hook).

    Rules, all verified failure modes (2026-09-21):
    - n < 5 or no y variance -> INSUFFICIENT (matches library minima).
    - n < 3*(p+1) observations per input -> THIN (BIC-forward caps at n-2
      terms, but selection among near-collinear features is still luck).
    - every all-positive input spans max/min < min_range_ratio -> THIN
      (power-law terms unidentifiable over narrow windows: the QD
      [1,6]-nm result; the go/no-go rule, now automatic).
    Returns verdict OK/THIN/INSUFFICIENT with per-variable range ratios
    and reasons. Advisory only -- callers decide; sparse_select_terms
    attaches it to every result under "adequacy".
    """
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        n = len(y)
        ratios = {}
        for k in names:
            v = np.asarray(x_data[k], float).ravel()
            if v.shape != (n,) or not bool(np.all(np.isfinite(v))):
                return {"verdict": "INSUFFICIENT", "n": n,
                        "reasons": [f"input '{k}' is not finite length-{n}"]}
            if bool(np.all(v > 0)):
                ratios[k] = round(float(v.max() / (v.min() + 1e-300)), 4)
        reasons = []
        if n < 5 or float(np.sum((y - y.mean()) ** 2)) <= 1e-300:
            reasons.append("n < 5 or no y variance")
            return {"verdict": "INSUFFICIENT", "n": n, "n_vars": len(names),
                    "range_ratios": ratios, "reasons": reasons}
        if n < 3 * (len(names) + 1):
            reasons.append(f"n={n} < 3*(p+1)={3 * (len(names) + 1)} "
                           f"for p={len(names)} inputs")
        if ratios and all(r < min_range_ratio for r in ratios.values()):
            reasons.append("all positive inputs span max/min < "
                           f"{min_range_ratio} (power-law unidentifiability)")
        return {"verdict": "THIN" if reasons else "OK", "n": n,
                "n_vars": len(names), "range_ratios": ratios,
                "reasons": reasons}
    except _PROBE_ERRORS as e:
        return {"verdict": "INSUFFICIENT", "reasons": [f"adequacy check failed: {e}"]}


def sparse_select_terms(x_data: dict[str, np.ndarray],
                        y_data: np.ndarray,
                        max_power: int = 2,
                        include_reciprocals: bool = True,
                        *,
                        corr_threshold: float | None = 0.999,
                        stability_reps: int | None = None,
                        stability_threshold: float = 0.6,
                        ambiguity_corr: float | None = None,
                        seed: int = 0,
                        selector: str = "bic_forward",
                        max_reciprocal_power: int = 2,
                        strict: bool = False) -> dict:
    """SINDy-style sparse term selection over a monomial library.

    Builds product/power/reciprocal-power features from inputs, selects by
    ``selector`` (default BIC-forward greedy subset search; legacy LassoCV
    via selector="lasso"), and returns active terms. Use as a pre-filter:
    tells GP *which* terms matter before it searches *how* they compose.
    Raises no exception outward.

    Collinearity filter (new): the power/reciprocal library contains exact
    duplicates (e.g. ``Rinv**2`` vs ``Rinv2**1``) and near-duplicates that
    make LassoCV return term soup. Before fitting, features are ranked by
    |correlation| with ``y`` and greedily kept only if their max |correlation|
    with an already-kept feature is below ``corr_threshold``. Dropped terms
    are reported under ``dropped_collinear`` with the keeper that blocked
    them, so nothing is silently lost. The default 0.999 removes only
    (near-)exact degeneracies -- e.g. ``Rinv**2`` vs ``Rinv2**1``, or
    ``log(Rinv)`` vs ``log(Rinv2)`` -- while leaving genuinely distinct but
    correlated physics terms (``1/R`` vs ``1/R^1.33``, corr ~0.997) to
    LassoCV. Lower thresholds filter more aggressively but risk dropping
    true terms shadowed by fractional-power impostors. Pass
    ``corr_threshold=None`` to restore the legacy unfiltered library.

    Stability selection (optional): with ``stability_reps`` set, Lasso is
    refit on seeded bootstrap resamples at the cross-validated alpha and
    only terms selected in at least ``stability_threshold`` fraction of
    reps are returned. Frequencies are reported under ``stability_freq``.
    Note its limit, verified on Brus data: when impostors fit as well as
    truth (``1/R^3`` vs ``1/R`` at 2% noise), the wrong terms are stably
    selected too -- stability measures selector consistency, not truth.

    Ambiguity groups (optional): with ``ambiguity_corr`` set, single-linkage
    clustering on pairwise |correlation| groups mutually indistinguishable
    features (``Rinv**1`` vs ``Rinv2**(2/3)``, corr 0.997). Each group means
    "at most one of these is identifiable -- try each downstream". This is
    the honest output under unidentifiability: the pre-filter cannot pick
    the winner, so it reports the ambiguity set for the GP/BIC stage.

    ``strict=True`` raises ValueError instead of proceeding when
    :func:`check_data_adequacy` reports THIN or INSUFFICIENT.
    """
    try:
        import itertools
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        adequacy = check_data_adequacy(x_data, y_data)
        if strict and adequacy["verdict"] != "OK":
            raise ValueError(
                "data gate: " + adequacy["verdict"] + ": "
                + "; ".join(adequacy.get("reasons", ["unspecified"]))
                + " (pass strict=False to override with an audit trail)")
        if len(y) < 5 or float(np.sum((y - y.mean()) ** 2)) <= 1e-300:
            return {"terms": [], "note": "insufficient variance",
                    "adequacy": adequacy}
        cols = {k: np.asarray(x_data[k], float).ravel() for k in names}
        feats: dict = {"1": np.ones_like(y)}
        for k in names:
            v = cols[k]
            for p in range(1, max_power + 1):
                # Integer powers use v**p directly: sign-preserving
                # sign(v)*|v|**p corrupts EVEN powers on signed inputs
                # (it negates x^2 for x<0). Fractional powers stay gated
                # on all-positive below.
                feats[f"{k}**{p}"] = v ** p
            if bool(np.all(v > 0)):
                for num, den in ((1, 2), (3, 2), (1, 3), (2, 3)):
                    feats[f"{k}**({num}/{den})"] = v ** (num / den)
            if include_reciprocals and bool(np.all(v != 0)):
                # Reciprocal powers (1/x, 1/x^2, ...): confinement-style
                # decays must be expressible without hand-engineering.
                for q in range(1, max(1, int(max_reciprocal_power)) + 1):
                    feats[f"1/{k}" if q == 1 else f"1/{k}**{q}"] = 1.0 / v ** q
            # Add exponential and logarithmic basis functions for positive variables
            # Scale input to prevent overflow: exp(x) -> exp((x - mean)/std)
            if bool(np.all(v > 0)):
                v_mean = float(np.mean(v))
                v_std = float(np.std(v)) + 1e-300
                v_scaled = (v - v_mean) / v_std
                feats[f"exp({k})"] = np.exp(np.clip(v_scaled, -20, 20))
                feats[f"log({k})"] = np.log(v)
        for a, b in itertools.combinations(names, 2):
            feats[f"{a}*{b}"] = cols[a] * cols[b]
        keys = [k for k in feats
                if float(np.std(feats[k])) > 1e-300]
        if not keys:
            return {"terms": [], "note": "no varying features",
                    "adequacy": adequacy}
        # Greedy collinearity filter: rank by |corr| with y, keep a feature
        # only if it is not near-duplicate of an already-kept one.
        dropped: list = []
        if corr_threshold is not None:
            if not np.isfinite(corr_threshold) or not 0.0 < corr_threshold <= 1.0:
                return {"terms": [], "note": "corr_threshold must be in (0, 1]"}
            y_centered = y - y.mean()
            y_norm = float(np.sqrt(np.sum(y_centered ** 2))) + 1e-300
            scored = []
            for k in keys:
                v = feats[k] - feats[k].mean()
                denom = float(np.sqrt(np.sum(v ** 2))) * y_norm + 1e-300
                scored.append((abs(float(np.sum(v * y_centered) / denom)), k))
            scored.sort(key=lambda item: (-item[0], item[1]))
            kept: list = []
            kept_cols: list = []
            for _, k in scored:
                v = feats[k]
                blocker = None
                for kk, col in zip(kept, kept_cols, strict=False):
                    w = col - col.mean()
                    denom = (float(np.sqrt(np.sum((v - v.mean()) ** 2)))
                             * float(np.sqrt(np.sum(w ** 2))) + 1e-300)
                    c = abs(float(np.sum((v - v.mean()) * w) / denom))
                    if c >= corr_threshold:
                        blocker = (kk, round(c, 6))
                        break
                if blocker is None:
                    kept.append(k)
                    kept_cols.append(v)
                else:
                    dropped.append({"dropped": k, "kept": blocker[0],
                                    "corr": blocker[1]})
            keys = kept
            if not keys:
                return {"terms": [], "note": "all features filtered as collinear",
                        "adequacy": adequacy,
                        "dropped_collinear": dropped,
                        "n_library_raw": len(scored)}
        X = np.column_stack([feats[k] for k in keys])
        scales = np.std(X, axis=0) + 1e-300
        Xn = X / scales
        from sklearn.linear_model import Lasso, LassoCV
        out: dict = {"n_library": len(keys)}
        alpha_for_stability: float | None = None
        if selector == "bic_forward":
            # Polynomial-branch parity: unscaled OLS selection; each term
            # must pay k*ln(n). Structural overfit cap lives in max_terms.
            fwd = bic_forward_select({k: feats[k] for k in keys}, y)
            if fwd.get("terms"):
                out.update({"terms": fwd["terms"],
                            "intercept": fwd["intercept"],
                            "bic": fwd["bic"],
                            "selector": "bic_forward"})
            else:
                out.update({"terms": [], "note": fwd.get("note", ""),
                            "selector": "bic_forward"})
            out["alpha"] = None
        elif selector == "lasso":
            model = LassoCV(max_iter=5000).fit(Xn, y)
            out.update({"terms": [keys[i] for i, c in enumerate(model.coef_)
                                  if abs(c) > 1e-9],
                        "intercept": round(float(model.intercept_), 6),
                        "alpha": float(model.alpha_),
                        "selector": "lasso"})
            alpha_for_stability = float(model.alpha_)
        else:
            return {"terms": [], "note": f"unknown selector '{selector}'",
                    "adequacy": adequacy}
        if corr_threshold is not None:
            out["dropped_collinear"] = dropped
            out["n_dropped"] = len(dropped)
        if stability_reps is not None:
            if (not isinstance(stability_reps, int) or stability_reps < 2
                    or not 0.0 < stability_threshold <= 1.0):
                return {"terms": [], "note": "stability_reps must be an int >= 2 "
                                             "and stability_threshold in (0, 1])",
                        "adequacy": adequacy}
            if alpha_for_stability is None:
                alpha_for_stability = float(
                    LassoCV(max_iter=5000).fit(Xn, y).alpha_)
            out["alpha"] = alpha_for_stability
            rng = np.random.default_rng(seed)
            n = Xn.shape[0]
            freq = np.zeros(len(keys))
            for _ in range(stability_reps):
                idx = rng.integers(0, n, n)
                boot = Lasso(alpha=alpha_for_stability,
                             max_iter=5000).fit(Xn[idx], y[idx])
                freq += (np.abs(boot.coef_) > 1e-9).astype(float)
            freq /= stability_reps
            stable = [keys[i] for i in range(len(keys))
                      if freq[i] >= stability_threshold]
            out["terms"] = stable
            out["stability_freq"] = {keys[i]: round(float(freq[i]), 3)
                                     for i in range(len(keys))}
            out["stability_reps"] = stability_reps
            out["stability_threshold"] = stability_threshold
        if ambiguity_corr is not None:
            if not np.isfinite(ambiguity_corr) or not 0.0 < ambiguity_corr < 1.0:
                return {"terms": [], "note": "ambiguity_corr must be in (0, 1)"}
            centered = [feats[k] - feats[k].mean() for k in keys]
            norms = [float(np.sqrt(np.sum(c ** 2))) + 1e-300 for c in centered]
            parent = list(range(len(keys)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    c = abs(float(np.sum(centered[i] * centered[j])
                                  / (norms[i] * norms[j])))
                    if c >= ambiguity_corr:
                        union(i, j)
            groups: dict = {}
            for i, k in enumerate(keys):
                groups.setdefault(find(i), []).append(k)
            out["ambiguity_groups"] = sorted(
                (sorted(members) for members in groups.values() if len(members) > 1),
                key=lambda members: (-len(members), members))
            out["ambiguity_corr"] = ambiguity_corr
        out["adequacy"] = adequacy
        return out
    except _PROBE_ERRORS as e:
        return {"terms": [], "note": f"sparse-select failed: {e}"}


def beam_search_expressions(variables: list[str], depth: int = 2,
                            width: int = 8,
                            seed: int | None = 42) -> list:
    """Breadth-capped construction search (MCTS-lite without rollouts).

    Expands expression trees level by level over {+, -, *, /} applied to
    variables and small integers, keeping the `width` structurally distinct
    candidates per level. Deterministic given seed. Returns expression
    strings, simplest first.
    """
    import random as _random
    rng = _random.Random(seed)
    ops = ["+", "-", "*", "/"]
    level = [*list(variables), "1", "2"]
    seen = set(level)
    out = list(level)
    for _ in range(max(0, depth)):
        nxt = []
        for _ in range(width * 4):
            a = rng.choice(level)
            b = rng.choice(level)
            op = rng.choice(ops)
            if op == "/" and b in ("0",):
                continue
            expr = f"({a}{op}{b})"
            if expr not in seen:
                seen.add(expr)
                nxt.append(expr)
        nxt = nxt[:width]
        out.extend(nxt)
        level = nxt
        if not level:
            break
    return out


def control_variable_probe(x_data: dict[str, np.ndarray],
                           y_data: np.ndarray,
                           tol: float = 1e-6) -> dict:
    """CVGP-style probe: drop-one linear R^2 gain per variable.

    Returns per-variable relevance plus the verdict of which variables the
    target actually depends on. Gains are relative: a variable counts only
    above both the absolute tol and 1% of the max gain (kills numerical
    dust). Constant targets short-circuit (nothing to explain).
    Nonlinear links read low by construction — documented, not hidden.
    """
    try:
        import numpy as np
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < 10:
            return {"relevant": [], "note": "insufficient data"}
        if float(np.sum((y - y.mean()) ** 2)) <= 1e-300:
            return {"relevant": [], "note": "constant target"}
        cols = {k: np.asarray(x_data[k], float).ravel() for k in names}

        def _r2(use: list) -> float:
            if not use:
                return 0.0
            A = np.column_stack([cols[k] for k in use]
                                + [np.ones(len(y))])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            denom = float(np.sum((y - y.mean()) ** 2)) + 1e-300
            return 1.0 - float(np.sum((y - pred) ** 2)) / denom

        try:
            full = _r2(names)
        except (TypeError, ValueError, ArithmeticError):
            return {"relevant": [], "note": "probe failed"}
        relevance = {}
        for name in names:
            try:
                rest = [k for k in names if k != name]
                gain = full - _r2(rest)
            except (TypeError, ValueError, ArithmeticError):
                gain = 0.0
            relevance[name] = round(float(max(gain, 0.0)), 4)
        floor = max(tol, 0.01 * max(relevance.values(), default=0.0))
        relevant = sorted([k for k, v in relevance.items() if v > floor],
                          key=lambda k: -relevance[k])
        return {"relevant": relevant, "relevance": relevance,
                "note": "drop-one linear R^2 gain; nonlinear links read low"}
    except (TypeError, ValueError, ArithmeticError):
        return {"relevant": [], "note": "probe failed"}


def _mlp_pipeline(seed: int):
    """Standardized MLP: scaling is mandatory, not optional.

    Unscaled inputs let wide-range variables saturate hidden units while
    narrow-range ones are ignored, corrupting every downstream importance
    or distillation. Returns (pipeline, converged_flag_holder).
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=5000,
                     random_state=seed))


def _mlp_convergence(model) -> dict:
    """Report optimizer convergence instead of silencing it.

    sklearn signals non-convergence via n_iter_ == max_iter (and a
    ConvergenceWarning we still suppress for output cleanliness); a
    non-converged fit gets flagged, never silently trusted.
    """
    mlp = model.named_steps["mlpregressor"]
    n_iter = int(getattr(mlp, "n_iter_", 0))
    converged = bool(n_iter < int(mlp.max_iter))
    return {"n_iter": n_iter, "converged": converged}


def neural_relevance(x_data: dict[str, np.ndarray],
                     y_data: np.ndarray,
                     seed: int = 42) -> dict:
    """MLP surrogate + permutation importance per variable. Agnostic.
 
    To use the JAX-based neural discovery system instead, set the
    environment variable NEURAL_DISCOVERY_ENGINE=1 or instantiate
    apedp.neural_discovery.integration.NeuralDiscoveryEngine directly.
    """
    try:
        import os
        import warnings

        import numpy as np
 
        # Check if we should use the new neural discovery engine
        use_new_engine = (
            os.environ.get("NEURAL_DISCOVERY_ENGINE", "0") == "1"
        )
 
        if use_new_engine:
            # Use the new JAX-based neural surrogate via integration module
            from apedp.neural_discovery.integration import neural_relevance as neural_relevance_new
            # `neural_discovery` is excluded from type checking (see pyproject),
            # so the surrogate entry points are statically opaque here.
            return cast(
                "dict[Any, Any]",
                neural_relevance_new(
                    x_data=x_data,
                    y_data=y_data,
                    seed=seed,
                ),
            )
 
        # Original sklearn-based implementation (fallback)
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < 30:
            return {"relevant": [], "note": "insufficient data for MLP"}
        X = np.column_stack([np.asarray(x_data[k], float).ravel()
                         for k in names])
        if not bool(np.all(np.isfinite(X))) or not bool(np.all(np.isfinite(y))):
            return {"relevant": [], "note": "non-finite data"}
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=seed)
        model = _mlp_pipeline(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, ytr)
        conv = _mlp_convergence(model)
        base = float(model.score(Xte, yte))
        report = {"surrogate_r2": round(base, 4),
                  "converged": conv["converged"], "n_iter": conv["n_iter"]}
        if base < 0.5:
            report.update({"relevant": [],
                           "note": f"held-out surrogate too weak (R^2={base:.3f})"
                                   + ("" if conv["converged"]
                                      else "; optimizer did not converge")
                           })
            return report
        rng = np.random.default_rng(seed)
        importance = {}
        for j, name in enumerate(names):
            drops = []
            for _ in range(5):
                Xp = Xte.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - float(model.score(Xp, yte)))
            importance[name] = round(float(np.mean(drops)), 4)
        floor = max(0.01, 0.05 * max(importance.values(), default=0.0))
        relevant = sorted([k for k, v in importance.items() if v > floor],
                          key=lambda k: -importance[k])
        report.update({"relevant": relevant, "relevance": importance,
                       "note": "held-out permutation importance; "
                              "deterministic seed"
                              + ("" if conv["converged"]
                                 else "; optimizer did not converge")
                       })
        return report
    except _PROBE_ERRORS as e:
        return {"relevant": [], "note": f"neural probe failed: {e}"}


def grammar_terminals_for(target_unit: str, variables: dict,
                          engine=None) -> dict:
    """Unit-ontology grammar: which variables may legally appear under
    log/exp/sqrt, and which binary combinations preserve dimensions.

    Returns allowed unary function applications plus dimensionally valid
    binary pairs. Without an engine, falls back to name-only heuristics
    and says so.
    """
    try:
        names = list(variables.keys())
        dimensionless = []
        units_checked = engine is not None
        if engine is not None:
            try:
                for k in names:
                    unit = engine.check_expression_consistency(sp.Symbol(k))
                    if unit.dimensionality == engine.registry(
                            "dimensionless").dimensionality:
                        dimensionless.append(k)
            except _PROBE_ERRORS as exc:
                logger.debug("unit-consistency probe failed: %s", exc)
                dimensionless = []
                units_checked = False
        return {"log_exp_allowed": sorted(dimensionless),
                "note": ("unit-checked" if units_checked
                         else "unit check unavailable: names unchecked"
                         if engine is not None
                         else "no engine: names only, unchecked")}
    except _PROBE_ERRORS as e:
        return {"log_exp_allowed": [], "note": f"grammar probe failed: {e}"}


def distill_via_surrogate(x_data: dict[str, np.ndarray],
                          y_data: np.ndarray, seed: int = 42,
                          n_dense: int = 200) -> dict:
    """Neural distill path: fit an MLP surrogate, sample it densely on a
    grid, then run differentiable symbolic regression on surrogate samples.
 
    Replaces the original MLP-based distillation with a JAX-based approach
    that uses differentiable symbolic regression instead of sparse term
    selection on surrogate predictions.
 
    To use the original sklearn-based distillation, set
    NEURAL_DISCOVERY_ENGINE=0 (default).
    """
    try:
        import os
        import warnings

        import numpy as np
 
        # Check if we should use the new neural discovery engine
        use_new_engine = (
            os.environ.get("NEURAL_DISCOVERY_ENGINE", "0") == "1"
        )
 
        if use_new_engine:
            # Use the new JAX-based neural distillation via integration module
            from apedp.neural_discovery.integration import (
                distill_via_surrogate as distill_via_surrogate_new,
            )
            return cast(
                "dict[Any, Any]",
                distill_via_surrogate_new(
                    x_data=x_data,
                    y_data=y_data,
                    seed=seed,
                    n_dense=n_dense,
                ),
            )
 
        # Original sklearn-based implementation (fallback)
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        X = np.column_stack([np.asarray(x_data[k], float).ravel()
                             for k in names])
        if len(y) < 30 or not bool(np.all(np.isfinite(X))):
            return {"terms": [], "note": "insufficient/non-finite data"}
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _mlp_pipeline(seed)
            model.fit(Xtr, ytr)
        conv = _mlp_convergence(model)
        r2 = float(model.score(Xte, yte))
        if r2 < 0.5:
            return {"terms": [],
                    "note": f"held-out surrogate too weak (R^2={r2:.3f})"
                            + ("" if conv["converged"]
                               else "; optimizer did not converge"),
                    "surrogate_r2": round(r2, 4),
                    "converged": conv["converged"], "n_iter": conv["n_iter"]}
        rng = np.random.default_rng(seed)
        # Near-manifold grid: bootstrap real rows + small jitter, clipped
        # to the data box. Independent-uniform grids query empty corners
        # where the surrogate hallucinates; those hallucinations used to
        # get distilled as terms.
        pick = rng.integers(0, len(Xtr), n_dense)
        spread = np.std(Xtr, axis=0) + 1e-300
        G = Xtr[pick] + rng.normal(0, 0.05, (n_dense, Xtr.shape[1])) * spread
        G = np.clip(G, Xtr.min(axis=0), Xtr.max(axis=0))
        grid = {k: G[:, j] for j, k in enumerate(names)}
        ys = model.predict(G)
        out = sparse_select_terms(grid, ys)
        out["surrogate_r2"] = round(r2, 4)
        out["converged"] = conv["converged"]
        out["n_iter"] = conv["n_iter"]
        out["note"] = "terms describe the surrogate, not raw data directly"
        return out
    except _PROBE_ERRORS as e:
        return {"terms": [], "note": f"distill failed: {e}"}
def recombine_power_law(variables: list[str], exponents: list[float]) -> str:
    """Render a power-law product string from parallel variable/exponent lists."""
    terms = []
    for name, p in zip(variables, exponents, strict=False):
        if abs(p) < 1e-12:
            continue
        if abs(p - round(p)) > 1e-9:
            terms.append(f"{name}**({p})")
        elif round(p) == 1:
            terms.append(name)
        else:
            terms.append(f"{name}**{round(p)}")
    return "*".join(terms) if terms else "1"


# Default nested-operator constraints (PySR-style): enclosing function name
# maps to forbidden direct-child function names. Prevents pathological
# nestings like exp(sin(exp(x))) while allowing exp(x), sin(x), log(x).
# Opt-in via nested_constraints=None (legacy) or a custom mapping.
DEFAULT_NESTED_CONSTRAINTS: dict[str, tuple] = {
    "exp": ("exp", "log", "sin", "cos"),
    "log": ("exp", "log", "sin", "cos"),
    "sin": ("sin", "cos", "exp"),
    "cos": ("sin", "cos", "exp"),
}


# Rational coefficients worth snapping fitted values to.
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


def detect_composition_family(x_data: dict[str, np.ndarray],
                              y_data: np.ndarray,
                              r2_threshold: float = 0.99) -> dict:
    """Probe data shape for function families: exponential (log y linear),
    arrhenius (log y linear in 1/x), saturating-exponential (log(A-y)
    linear for a scanned asymptote), rational/inverse (1/y linear).

    Scattered samples suffice (single-variable regressions only). Returns
    the best-matching family or {}. The saturating branch uses a 0.999 bar
    (not r2_threshold): its free asymptote can accidentally fit monotone
    non-saturating data, so it must be near-exact to fire.
    """
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < 5:
            return {}
        if float(np.sum((y - y.mean()) ** 2)) <= 1e-300:
            return {}
        best: dict = {}
        for name in names:
            xv = np.asarray(x_data[name], float).ravel()
            if len(xv) != len(y):
                continue
            order = np.argsort(xv)
            xs, ys = xv[order], y[order]
            try:
                if bool(np.all(np.isfinite(ys))) and bool(np.all(ys != 0)):
                    ly = np.log(np.abs(ys))
                    A = np.column_stack([xs, np.ones_like(xs)])
                    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
                    pred = A @ coef
                    denom_log = float(np.sum((ly - ly.mean()) ** 2)) + 1e-300
                    r2e = 1.0 - float(np.sum((ly - pred) ** 2)) / denom_log
                    if r2e > r2_threshold and r2e > best.get("r_squared", 0.0):
                        best = {"family": "exponential", "variable": name,
                                "r_squared": round(float(r2e), 6)}
            except (TypeError, ValueError, ArithmeticError):
                pass
            try:
                if (bool(np.all(np.isfinite(ys))) and bool(np.all(ys != 0))
                        and bool(np.all(xs != 0))):
                    ly = np.log(np.abs(ys))
                    ix = 1.0 / xs
                    A = np.column_stack([ix, np.ones_like(ix)])
                    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
                    pred = A @ coef
                    denom_log = (float(np.sum((ly - ly.mean()) ** 2))
                                 + 1e-300)
                    r2a = (1.0 - float(np.sum((ly - pred) ** 2))
                           / denom_log)
                    if (r2a > r2_threshold
                            and r2a > best.get("r_squared", 0.0)):
                        best = {"family": "arrhenius", "variable": name,
                                "r_squared": round(float(r2a), 6)}
            except (TypeError, ValueError, ArithmeticError):
                pass
            try:
                if bool(np.all(np.isfinite(ys))) and ys[-1] != ys[0]:
                    ymax = float(np.max(ys))
                    for frac in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4,
                                 0.8, 1.5):
                        a_est = ymax * (1.0 + frac)
                        gap = a_est - ys
                        if not bool(np.all(gap > 0)):
                            continue
                        ly = np.log(gap)
                        A = np.column_stack([xs, np.ones_like(xs)])
                        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
                        pred = A @ coef
                        denom_s = (float(np.sum((ly - ly.mean()) ** 2))
                                   + 1e-300)
                        r2s = (1.0 - float(np.sum((ly - pred) ** 2))
                               / denom_s)
                        if (r2s > 0.999
                                and r2s > best.get("r_squared", 0.0)):
                            best = {"family": "saturating-exponential",
                                    "variable": name,
                                    "r_squared": round(float(r2s), 6),
                                    "asymptote": round(a_est, 6)}
            except (TypeError, ValueError, ArithmeticError):
                pass
            try:
                if bool(np.all(ys != 0)):
                    iy = 1.0 / ys
                    A = np.column_stack([xs, np.ones_like(xs)])
                    coef, *_ = np.linalg.lstsq(A, iy, rcond=None)
                    pred = A @ coef
                    denom_inv = float(np.sum((iy - iy.mean()) ** 2)) + 1e-300
                    r2r = 1.0 - float(np.sum((iy - pred) ** 2)) / denom_inv
                    if r2r > r2_threshold and r2r > best.get("r_squared", 0.0):
                        best = {"family": "rational", "variable": name,
                                "r_squared": round(float(r2r), 6)}
            except (TypeError, ValueError, ArithmeticError):
                pass
        return best
    except (TypeError, ValueError, ArithmeticError):
        return {}


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


def detect_power_law(x_data: dict[str, np.ndarray],
                     y_data: np.ndarray,
                     r2_threshold: float = 0.90,
                     snap_tol: float = 0.02) -> dict:
    """Log-log power-law probe: fit log y = p * log x + c per variable.

    Always returns the best-matching variable with fitted exponent, R^2,
    and the nearby rational snap (e.g. 1.5 -> "3/2") when within snap_tol.
    The ``confident`` flag marks whether R^2 clears r2_threshold -- callers
    (or BIC selection) decide what to do with sub-threshold fits instead of
    the result being swallowed. Requires strictly positive data; needs only
    >= 3 points, so it fires on the tiny real datasets (n=8 Kepler) where
    the linear probes decline. Raises no exception outward.
    """
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < 3:
            return {}
        if float(np.sum((y - y.mean()) ** 2)) <= 1e-300:
            return {}
        if not bool(np.all(np.isfinite(y))) or not bool(np.all(y > 0)):
            return {}
        best: dict = {}
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
                denom = float(np.sum((ly - ly.mean()) ** 2)) + 1e-300
                r2 = 1.0 - float(np.sum((ly - pred) ** 2)) / denom
                if r2 > best.get("r_squared", 0.0):
                    p = float(coef[0])
                    snap = None
                    for target, label in POWER_LAW_RATIONALS:
                        if abs(p - target) < snap_tol:
                            snap = label
                            break
                    best = {"family": "power-law", "variable": name,
                            "exponent": round(p, 4),
                            "r_squared": round(float(r2), 6),
                            "snapped": snap,
                            "confident": bool(r2 > r2_threshold)}
            except (TypeError, ValueError, ArithmeticError):
                pass
        return best
    except (TypeError, ValueError, ArithmeticError):
        return {}


def cross_family_select(x_data: dict[str, np.ndarray],
                          y_data: np.ndarray,
                          max_power: int = 3,
                          extrapolation_check: bool = True) -> dict:
    """Cross-family model selection: fit candidates from multiple functional
    families and rank by BIC (lower is better).

    Families tested: power-law (detect_power_law), exponential,
    Arrhenius (exp in 1/x), and polynomial (sparse_select_terms).
    Returns the best candidate with its family, BIC, and a comparison
    table against all other families. A perfect-fit candidate
    (R^2 > 0.999) using more than n/2 parameters is rejected as overfit
    and reported under "notes" instead of winning. With
    ``extrapolation_check`` (default), candidates that go non-finite or
    blow up super-polynomially just outside the data range are likewise
    rejected (kills coincidental in-window fits like exp(+c/x) on decaying
    data). Raises no exception outward.
    """
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        n = len(y)
        if n < 5:
            return {}
        if float(np.sum((y - y.mean()) ** 2)) <= 1e-300:
            return {}
        if not bool(np.all(np.isfinite(y))):
            return {}
        candidates = []
        notes: list = []
        # Power-law per variable
        for name in names:
            pl = detect_power_law(x_data, y)
            if pl and pl.get("r_squared", 0) > 0.5:
                xv = np.asarray(x_data[name], float).ravel()
                lxv = np.log(np.maximum(xv, 1e-300))
                coef, *_ = np.linalg.lstsq(np.column_stack([lxv, np.ones(n)]), np.log(np.maximum(y, 1e-300)), rcond=None)
                y_pred = np.exp(coef[0]*lxv + coef[1])
                bic = BayesianRefinementEngine.compute_bic(
                    BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                    2, n)
                candidates.append({"family": "power-law", "variable": name,
                                   "expression": f"{name}**{pl['exponent']:.3f}",
                                   "bic": bic, "r_squared": pl["r_squared"],
                                   "num_params": 2})
        # Exponential
        try:
            for name in names:
                xv = np.asarray(x_data[name], float).ravel()
                if not bool(np.all(xv > 0)):
                    continue
                coef, *_ = np.linalg.lstsq(np.column_stack([xv, np.ones(n)]),
                                           np.log(np.maximum(y, 1e-300)), rcond=None)
                y_pred = np.exp(coef[0]*xv + coef[1])
                r2 = BayesianRefinementEngine.compute_r_squared(y, y_pred)
                if r2 > 0.3:
                    bic = BayesianRefinementEngine.compute_bic(
                        BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                        2, n)
                    candidates.append({"family": "exponential", "variable": name,
                                       "expression": f"exp({coef[0]:.4f}*{name})",
                                       "bic": bic, "r_squared": round(float(r2), 4),
                                       "num_params": 2})
        except (TypeError, ValueError, ArithmeticError):
            pass
        # Arrhenius (exp in 1/x)
        try:
            for name in names:
                xv = np.asarray(x_data[name], float).ravel()
                if not bool(np.all(xv != 0)):
                    continue
                ix = 1.0 / xv
                coef, *_ = np.linalg.lstsq(np.column_stack([ix, np.ones(n)]),
                                           np.log(np.maximum(y, 1e-300)), rcond=None)
                y_pred = np.exp(coef[0]*ix + coef[1])
                r2 = BayesianRefinementEngine.compute_r_squared(y, y_pred)
                if r2 > 0.3:
                    bic = BayesianRefinementEngine.compute_bic(
                        BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                        2, n)
                    candidates.append({"family": "arrhenius", "variable": name,
                                       "expression": f"exp({coef[0]:.4f}/{name})",
                                       "bic": bic, "r_squared": round(float(r2), 4),
                                       "num_params": 2})
        except (TypeError, ValueError, ArithmeticError):
            pass
        # Polynomial from sparse_select_terms
        try:
            import itertools as _it
            sp = sparse_select_terms(x_data, y, max_power=max_power)
            if sp.get("terms"):
                from sklearn.linear_model import LassoCV
                cols = {k: np.asarray(x_data[k], float).ravel() for k in x_data}
                feats = {"1": np.ones(n)}
                for k in x_data:
                    v = cols[k]
                    for p in range(1, max_power+1):
                        # Integer powers must use v**p directly: sign(v)*|v|^p
                        # inverts even powers on negative inputs ((-2)^2 -> -4).
                        feats[f"{k}**{p}"] = v**p
                    if bool(np.all(v != 0)):
                        feats[f"1/{k}"] = 1.0/v
                for a, b in _it.combinations(x_data, 2):
                    feats[f"{a}*{b}"] = cols[a]*cols[b]
                keys = [k for k in feats if float(np.std(feats[k])) > 1e-300]
                X = np.column_stack([feats[k] for k in keys])
                scales = np.std(X, axis=0) + 1e-300
                Xn = X/scales
                model = LassoCV(max_iter=5000).fit(Xn, y)
                y_pred = model.predict(Xn)
                r2 = BayesianRefinementEngine.compute_r_squared(y, y_pred)
                if r2 > 0.3:
                    active_k = int(np.sum(np.abs(model.coef_) > 1e-9)) + 1
                    if r2 > 0.999 and active_k > n / 2:
                        notes.append(
                            f"polynomial rejected: R^2={r2:.4f} with "
                            f"{active_k} params on n={n} (overfit regime)")
                    else:
                        bic = BayesianRefinementEngine.compute_bic(
                            BayesianRefinementEngine.gaussian_log_likelihood(y, y_pred)[0],
                            active_k, n)
                        candidates.append({"family": "polynomial", "variable": ", ".join(x_data),
                                           "expression": sp["terms"][0] if sp["terms"] else "poly",
                                           "bic": bic, "r_squared": round(float(r2), 4),
                                           "num_params": active_k})
        except _PROBE_ERRORS as exc:
            logger.debug("Polynomial candidate rejected during BIC ranking: %s", exc)
        if extrapolation_check and candidates:
            from apedp.scientific_layers import extrapolation_audit
            cols = {k: np.asarray(x_data[k], float).ravel() for k in names}
            ranges = {k: (float(cols[k].min()), float(cols[k].max()))
                      for k in names}
            y_scale = float(np.max(np.abs(y))) + 1e-300
            survivors = []
            for cand in candidates:
                audit = extrapolation_audit(cand["expression"], ranges,
                                            y_scale)
                if audit["status"] == "FAIL":
                    notes.append(f"{cand['family']} {cand['expression']} "
                                 f"rejected: {audit['reason']}")
                else:
                    survivors.append(cand)
            candidates = survivors
        if not candidates:
            return {"notes": notes} if notes else {}
        best = BayesianRefinementEngine.select_best_model(candidates)
        out = {"best": best, "all_candidates": sorted(candidates, key=lambda c: c.get("bic", float('inf')))}
        if notes:
            out["notes"] = notes
        return out
    except (TypeError, ValueError, ArithmeticError):
        return {}


@dataclass(frozen=True)
class ModelEvidence:
    """The marginal likelihood ``p(D|M)`` of one model, and how it was obtained.

    ``p(D|M) = ∫ p(D|θ,M) p(θ|M) dθ``: the likelihood is *integrated over the
    prior*, not evaluated at a point. A ratio of summed pointwise likelihoods has
    neither, so it is a different (weaker) statistic with a different home --
    :meth:`BayesianRefinementEngine.compute_likelihood_ratio`.

    ``method`` is the approximation used (``"quadrature"``, ``"laplace"``,
    ``"bic"``); ``prior_description`` records the prior the integral was taken
    against. Both travel with the number so a verdict can state its own basis.
    """

    log_evidence: float
    method: str
    n_params: int
    prior_description: str = "unspecified"

    def __str__(self) -> str:
        return (
            f"log p(D|M) = {self.log_evidence:.4f} "
            f"[{self.method}, {self.n_params} param(s), prior: {self.prior_description}]"
        )


def _log_integrand(
    log_likelihood_fn: Callable[[np.ndarray], float],
    prior_logpdf: Callable[[np.ndarray], float],
    point: np.ndarray | list[float],
) -> float:
    """``log p(D|θ) + log p(θ)`` at one parameter point."""
    p = np.asarray(point, dtype=float)
    return float(log_likelihood_fn(p)) + float(prior_logpdf(p))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Composite trapezoid rule on a uniform or non-uniform grid.

    Written out rather than calling ``np.trapz``, which NumPy 2 removed.
    """
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _log_integrate(log_values: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule on ``exp(log_values)`` without leaving log space.

    Non-finite values are treated as a zero integrand (a hard support boundary),
    not as an annihilation of their neighbourhood.
    """
    finite = np.isfinite(log_values)
    if not bool(np.any(finite)):
        return float("-inf")
    m = float(np.max(log_values[finite]))
    area = _trapezoid(np.where(finite, np.exp(log_values - m), 0.0), x)
    if area <= 0.0 or not np.isfinite(area):
        return float("-inf")
    return float(m + np.log(area))


def _finite_difference_hessian(
    f: Callable[[np.ndarray], float],
    theta: np.ndarray,
    rel_step: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``theta`` (step scaled per parameter)."""
    k = int(theta.size)
    h = rel_step * np.maximum(np.abs(theta), 1.0)
    hessian = np.zeros((k, k))
    f0 = float(f(theta))
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        hessian[i, i] = (f(theta + ei) - 2.0 * f0 + f(theta - ei)) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            hessian[i, j] = hessian[j, i] = (
                f(theta + ei + ej)
                - f(theta + ei - ej)
                - f(theta - ei + ej)
                + f(theta - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return hessian


def _snap_coefficient(value: float) -> sp.Expr:
    for target, rational_val in STANDARD_RATIONALS:
        if abs(value - target) < 1e-3:
            return rational_val
    return sp.Float(value)


def _dim_dict(unit) -> dict[str, float]:
    """Pint dimensionality as a plain {dimension: exponent} dict."""
    return {str(k): float(v) for k, v in unit.dimensionality.items() if float(v) != 0.0}


def _dim_add(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
        if out[k] == 0.0:
            del out[k]
    return out


def _dim_scale(a: dict[str, float], p: float) -> dict[str, float]:
    return {k: v * p for k, v in a.items() if v * p != 0.0}


class BayesianRefinementEngine:
    """Bayesian hypothesis comparison and model selection engine.

    .. note::
        ``compute_bayes_factor`` is a legacy heuristic (ratio of summed
        likelihoods). It is kept for backwards compatibility but must NOT
        be used for discovery decisions. Use ``gaussian_log_likelihood`` +
        ``compute_bic``/``compute_aic`` + ``select_best_model`` instead,
        which implement a real Gaussian-noise log-likelihood and standard
        information-criterion selection.
    """

    @staticmethod
    def gaussian_log_likelihood(
        y_true: np.ndarray, y_pred: np.ndarray, noise_std: float | None = None
    ) -> tuple[float, float]:
        """Gaussian log-likelihood with ML noise estimate when not supplied.

        Returns ``(log_likelihood, noise_std_used)``. ``log_likelihood`` is
        ``-n/2 * (log(2*pi*sigma^2) + 1)`` at the ML sigma, the standard
        form used for BIC/AIC comparison.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        resid = y_true - y_pred
        n = int(y_true.size)
        if n == 0:
            raise ValueError("Empty data for likelihood computation")
        if noise_std is None:
            sigma2 = float(np.mean(resid**2))
            sigma2 = max(sigma2, 1e-300)
        else:
            sigma2 = max(float(noise_std) ** 2, 1e-300)
        log_lik = float(-0.5 * n * (np.log(2.0 * np.pi * sigma2) + np.mean(resid**2) / sigma2))
        return log_lik, float(np.sqrt(sigma2))

    @staticmethod
    def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of determination. Returns ``nan`` for degenerate inputs."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size == 0 or not np.all(np.isfinite(y_pred)):
            return float("nan")
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 0 or not np.isfinite(ss_tot):
            return float("nan")
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def is_degenerate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        expression: sp.Expr | None = None,
        min_r_squared: float = 0.1,
    ) -> tuple[bool, str]:
        """Decide whether a candidate is degenerate and must be refused.

        A candidate is degenerate when predictions are non-finite, constant
        (zero variance while the target varies), R² is NaN, or R² is below
        ``min_r_squared``. Returns ``(is_degenerate, reason)``.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape != y_true.shape:
            return True, f"shape mismatch {y_pred.shape} vs {y_true.shape}"
        if y_true.size == 0:
            # ``np.std``/``np.var`` warn ("Degrees of freedom <= 0") and return
            # nan on empty input. Name the condition rather than computing
            # statistics over nothing, which is how this surfaced as a numeric
            # warning in the first place.
            return True, "no data points to score"
        if not np.all(np.isfinite(y_pred)):
            return True, "non-finite predictions"
        if float(np.std(y_pred)) <= 1e-12 and float(np.std(y_true)) > 1e-12:
            return True, "constant prediction for varying target"
        r2 = BayesianRefinementEngine.compute_r_squared(y_true, y_pred)
        if not np.isfinite(r2):
            return True, "R² is NaN/non-finite"
        if r2 < min_r_squared:
            return True, f"R²={r2:.4f} below threshold {min_r_squared}"
        if expression is not None and expression in (sp.Integer(0), sp.Float(0.0)):
            return True, "fallback zero expression"
        return False, f"OK (R²={r2:.4f})"

    @staticmethod
    def select_best_model(candidates: list[dict]) -> dict | None:
        """Select the best candidate by BIC (lower is better).

        Each candidate dict must carry ``bic`` (float, finite). Ties break
        on lower ``complexity``. Returns ``None`` when no finite-BIC
        candidate exists — the caller must then refuse to persist.
        """
        finite = [c for c in candidates if np.isfinite(float(c.get("bic", np.inf)))]
        if not finite:
            return None
        return min(finite, key=lambda c: (float(c["bic"]), int(c.get("complexity", 0))))

    # ------------------------------------------------------------------
    # Model evidence
    # ------------------------------------------------------------------

    @staticmethod
    def marginal_likelihood(
        log_likelihood_fn: Callable[[np.ndarray], float],
        prior_logpdf: Callable[[np.ndarray], float],
        *,
        bounds: np.ndarray | None = None,
        map_params: np.ndarray | None = None,
        grid_points: int = 401,
        prior_description: str = "unspecified",
    ) -> ModelEvidence:
        """Integrate the likelihood against the prior: ``log ∫ p(D|θ) p(θ) dθ``.

        This is the quantity a Bayes factor is a ratio of. Supply either
        ``bounds`` (shape ``(n_params, 2)``; 1-2 parameters, nested trapezoid
        quadrature) or ``map_params`` (the MAP estimate; Gaussian/Laplace
        approximation around it). The returned :class:`ModelEvidence` records
        which approximation was used, because the number is only as trustworthy
        as the method.

        Raises ``ValueError`` when neither is usable rather than guessing: an
        un-integrated likelihood is a different statistic, not this one.
        """
        if bounds is not None:
            b = np.asarray(bounds, dtype=float)
            if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] < 1:
                raise ValueError(f"bounds must have shape (n_params, 2), got {b.shape}")
            if bool(np.any(b[:, 1] <= b[:, 0])):
                raise ValueError(f"every bound must have upper > lower, got {b.tolist()}")
            n_params = int(b.shape[0])
            if n_params > 2:
                raise ValueError(
                    f"grid quadrature is implemented for 1-2 parameters, got {n_params}; "
                    "pass map_params for the Laplace approximation"
                )
            n = max(int(grid_points), 3)
            axes = [np.linspace(b[i, 0], b[i, 1], n) for i in range(n_params)]
            if n_params == 1:
                vals = np.array([_log_integrand(log_likelihood_fn, prior_logpdf, [x]) for x in axes[0]])
                log_ev = _log_integrate(vals, axes[0])
            else:
                grid = np.empty((n, n))
                for i, x in enumerate(axes[0]):
                    for j, y in enumerate(axes[1]):
                        grid[i, j] = _log_integrand(log_likelihood_fn, prior_logpdf, [x, y])
                inner = np.array([_log_integrate(grid[i], axes[1]) for i in range(n)])
                log_ev = _log_integrate(inner, axes[0])
            return ModelEvidence(log_ev, "quadrature", n_params, prior_description)

        if map_params is None:
            raise ValueError(
                "marginal_likelihood needs bounds (quadrature) or map_params (Laplace); "
                "a likelihood without a prior and an integral is not a marginal likelihood"
            )

        theta = np.asarray(map_params, dtype=float).ravel()
        if theta.size < 1:
            raise ValueError("map_params must be non-empty")

        def log_posterior(p: np.ndarray) -> float:
            return _log_integrand(log_likelihood_fn, prior_logpdf, p)

        f_here = log_posterior(theta)
        step = 1e-4 * np.maximum(np.abs(theta), 1.0)
        for i in range(theta.size):
            offset = np.zeros(theta.size)
            offset[i] = step[i]
            if log_posterior(theta + offset) > f_here or log_posterior(theta - offset) > f_here:
                raise ValueError(
                    "Laplace approximation needs map_params to be a mode of the log-posterior, but "
                    f"moving parameter {i} by {step[i]:g} from {theta.tolist()} increases it; "
                    "optimise first, or pass bounds for quadrature"
                )

        hessian = _finite_difference_hessian(log_posterior, theta)
        neg_det = float(np.linalg.det(-hessian))
        if not np.isfinite(neg_det) or neg_det <= 0.0:
            raise ValueError(
                "Laplace approximation needs a negative-definite log-posterior Hessian "
                f"at the MAP; det(-H)={neg_det:g} for {theta.tolist()} (is it a mode?)"
            )
        log_ev = log_posterior(theta) + 0.5 * theta.size * np.log(2.0 * np.pi) - 0.5 * np.log(neg_det)
        return ModelEvidence(float(log_ev), "laplace", int(theta.size), prior_description)

    @staticmethod
    def evidence_from_bic(
        log_likelihood_max: float,
        num_params: int,
        num_samples: int,
        prior_description: str = "flat over the parameter support (asymptotic)",
    ) -> ModelEvidence:
        """Asymptotic evidence proxy: ``log p(D|M) ≈ log L_max - (k/2) ln n``.

        Labelled ``"bic"`` so a verdict resting on it can say so. Asymptotic in
        ``n``; not valid for small samples.
        """
        if num_params < 0:
            raise ValueError(f"num_params must be >= 0, got {num_params}")
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        lam = 0.5 * float(num_params) * float(np.log(float(num_samples)))
        return ModelEvidence(float(log_likelihood_max) - lam, "bic", int(num_params), prior_description)

    @staticmethod
    def kass_raftery_verdict(two_ln_bf: float, method_note: str = "") -> str:
        """Kass-Raftery grade for ``2 ln BF``, with the evidence method appended.

        The scale applies to a Bayes factor. It is not applied to a
        summed-likelihood ratio (see :meth:`compute_likelihood_ratio`).
        """
        if two_ln_bf > 10.0:
            verdict = "Very Strong / Decisive evidence for Model 1"
        elif two_ln_bf > 6.0:
            verdict = "Strong evidence for Model 1"
        elif two_ln_bf > 2.0:
            verdict = "Positive evidence for Model 1"
        elif two_ln_bf >= -2.0:
            verdict = "Inconclusive / Weak evidence"
        elif two_ln_bf >= -6.0:
            verdict = "Positive evidence for Model 2"
        elif two_ln_bf >= -10.0:
            verdict = "Strong evidence for Model 2"
        else:
            verdict = "Very Strong / Decisive evidence for Model 2"
        return f"{verdict} ({method_note})" if method_note else verdict

    @staticmethod
    def compute_bayes_factor(
        evidence_1: ModelEvidence,
        evidence_2: ModelEvidence,
    ) -> tuple[float, str, float]:
        """Bayes factor B_12 = p(D|M1) / p(D|M2) from two marginal likelihoods.

        Returns ``(BF, Kass-Raftery verdict naming the method, log10 BF)``.
        Non-finite evidence raises: an unmeasurable quantity has no ratio.
        """
        for label, ev in (("model 1", evidence_1), ("model 2", evidence_2)):
            if not np.isfinite(ev.log_evidence):
                raise ValueError(
                    f"{label} log-evidence is not finite ({ev.log_evidence}); "
                    "a Bayes factor cannot be formed from an unmeasured marginal likelihood"
                )
        diff = float(evidence_1.log_evidence - evidence_2.log_evidence)
        bf = float(np.exp(np.clip(diff, -700.0, 700.0)))
        log_bf = float(diff / np.log(10.0))
        note = f"{evidence_1.method} vs {evidence_2.method}"
        return bf, BayesianRefinementEngine.kass_raftery_verdict(2.0 * diff, note), log_bf

    @staticmethod
    def compute_bayes_factor_from_log_evidence(
        log_evidence_1: float,
        log_evidence_2: float,
        method_1: str = "quadrature",
        method_2: str = "quadrature",
    ) -> tuple[float, str, float]:
        """Bayes factor from two *log marginal likelihoods*, underflow-safe.

        Both inputs must be ``log ∫ p(D|θ) p(θ) dθ``. Passing a maximised log
        likelihood here is the error this signature exists to prevent; use
        :meth:`compute_likelihood_ratio` if that is what you have.
        """
        return BayesianRefinementEngine.compute_bayes_factor(
            ModelEvidence(float(log_evidence_1), method_1, 0),
            ModelEvidence(float(log_evidence_2), method_2, 0),
        )

    @staticmethod
    def compute_likelihood_ratio(
        model_1_likelihoods: np.ndarray,
        model_2_likelihoods: np.ndarray,
        regularizer: float = 1e-15,
    ) -> tuple[float, str, float]:
        """Ratio of *summed pointwise likelihoods* -- **not** a Bayes factor.

        Kept because it is what a caller handing over likelihood arrays has
        actually computed, and it is a useful description of two evaluated
        models. It has no prior and no integral over parameters, so it cannot be
        read on the Kass-Raftery scale; the description reports direction only.
        Returns ``(ratio, description, log10 ratio)``.
        """
        ev1 = float(np.sum(model_1_likelihoods)) + regularizer
        ev2 = float(np.sum(model_2_likelihoods)) + regularizer
        if ev1 < 0.0 or ev2 < 0.0:
            raise ValueError(
                f"summed likelihoods must be non-negative, got {ev1:g} and {ev2:g}; "
                "supply likelihoods, not log-likelihoods"
            )
        ratio = ev1 / ev2 if ev2 > 0.0 else float("inf")
        log_ratio = float(np.log10(ratio)) if np.isfinite(ratio) else float("inf")
        caveat = "NOT a Bayes factor: no prior, no parameter integral"
        if ratio > 1.0 + 1e-9:
            desc = f"summed likelihood ratio favours Model 1 ({caveat})"
        elif ratio < 1.0 - 1e-9:
            desc = f"summed likelihood ratio favours Model 2 ({caveat})"
        else:
            desc = f"summed likelihood ratios are indistinguishable ({caveat})"
        return ratio, desc, log_ratio

    @staticmethod
    def compute_bic(log_likelihood: float, num_params: int, num_samples: int) -> float:
        """Bayesian Information Criterion: BIC = k*ln(n) - 2*ln(L). Lower is better."""
        return float(num_params * np.log(max(num_samples, 1)) - 2.0 * log_likelihood)

    @staticmethod
    def compute_aic(log_likelihood: float, num_params: int) -> float:
        """Akaike Information Criterion: AIC = 2k - 2*ln(L). Lower is better."""
        return float(2.0 * num_params - 2.0 * log_likelihood)

    @staticmethod
    def compute_aicc(log_likelihood: float, num_params: int, num_samples: int) -> float:
        """Corrected Akaike Information Criterion for small sample sizes."""
        aic = BayesianRefinementEngine.compute_aic(log_likelihood, num_params)
        denom = num_samples - num_params - 1
        if denom > 0:
            return float(aic + (2.0 * num_params * (num_params + 1)) / denom)
        return float(aic)

    @staticmethod
    def score_candidate(
        expression: sp.Expr,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_params: int | None = None,
    ) -> dict:
        """Score one fitted candidate with R² / log-likelihood / BIC / AIC.

        ``num_params`` defaults to the number of free symbols in
        ``expression`` (a conservative complexity proxy). Returns a dict
        with ``r_squared``, ``log_likelihood``, ``bic``, ``aic``,
        ``complexity`` and ``degenerate``/``degenerate_reason``.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if num_params is None:
            try:
                num_params = len(set(expression.free_symbols))
            except AttributeError:
                # Not a SymPy expression (a raw string or array was passed): fall
                # back to the minimal complexity proxy rather than failing.
                num_params = 1
        num_params = max(int(num_params), 1)
        n = int(y_true.size)
        # A shape disagreement makes every statistic below undefined, but
        # ``compute_r_squared`` raises only on *incompatible* shapes and silently
        # broadcasts compatible ones (e.g. (1,) against (n,)), returning a
        # meaningless figure. Check explicitly and report the pair as unscoreable
        # instead of letting it reach the arithmetic. ``is_degenerate`` already
        # names the mismatch and refuses such a candidate.
        if y_true.shape == y_pred.shape:
            r2 = BayesianRefinementEngine.compute_r_squared(y_true, y_pred)
        else:
            r2 = float("nan")
        degen, reason = BayesianRefinementEngine.is_degenerate(y_true, y_pred, expression)
        try:
            log_lik, _ = BayesianRefinementEngine.gaussian_log_likelihood(y_true, y_pred)
        except (TypeError, ValueError):
            # The likelihood function raises ValueError for shape mismatch and
            # empty data; np.asarray raises TypeError/ValueError for input that
            # cannot be cast to float. Both mean "no likelihood available".
            log_lik = float("-inf")
        try:
            complexity = len(list(sp.preorder_traversal(expression)))
        except (AttributeError, TypeError, ValueError):
            complexity = num_params
        if np.isfinite(log_lik):
            bic = BayesianRefinementEngine.compute_bic(log_lik, num_params, n)
            aic = BayesianRefinementEngine.compute_aic(log_lik, num_params)
        else:
            bic, aic = float("inf"), float("inf")
        return {
            "expression": expression,
            "r_squared": float(r2),
            "log_likelihood": float(log_lik),
            "bic": float(bic),
            "aic": float(aic),
            "complexity": int(complexity),
            "num_params": int(num_params),
            "degenerate": bool(degen),
            "degenerate_reason": reason,
        }


class MultiVariableSRE:
    """Dimensional-invariant guided evolutionary symbolic regression engine.

    Division of labor: ``MultiVariableSRE`` (this class) is the primary
    dimensionally-gated evolutionary engine used by the pipeline and API.
    ``apedp.symbolic_regression.SymbolicRegressor`` is a legacy standalone
    tournament/crossover engine kept for backwards compatibility and small
    dimensionless demos; prefer this class for all dimensional work.

    Optional extensions (on by default; pass False to restore the legacy
    minimal operator set):

    - ``allow_exp``: include ``exp(x)`` in generated candidates. The argument
      is required to be dimensionless (same rule the dimensional gatekeeper
      enforces), so generation retries with plain ``Mul``/``Pow`` when no
      dimensionless inner expression is found.
    - ``allow_trig``: include ``sin(x)``/``cos(x)`` with dimensionless
      arguments (wave phases, oscillations).
    - ``allow_log``: include ``log(x)`` with dimensionless arguments
      (entropy, Arrhenius linearizations).
    - ``allow_sqrt``: include explicit ``sqrt(x)`` nodes (in addition to
      ``Pow(x, 1/2)`` fractional exponents, which are always enabled).
      On by default; dimensionally halves the argument's dimensions.
    - ``extra_const_values``: mapping of ``name -> (value, unit_str)`` for
      physical constants absent from the global registry (e.g. Faraday's
      constant). Each entry is registered on the engine, added to the
      platform's constant values, and appended to the building blocks.
    - ``template_priors``: list of ``str``/``Expr`` physics-informed seed
      expressions (e.g. ``"A_freq*exp(-E_act/(R_gas*T_temp))"``). Entries
      that parse and pass the dimensional gatekeeper are prepended to the
      initial population; invalid entries are skipped with a warning.
    - ``inner_fit``: hill-climb the values of ephemeral random constants
      (``C0``, ``C1``, ...) that appear inside nonlinearities such as
      ``exp``. The legacy scheme fixes them at 1.0 and only fits outer
      linear coefficients, which cannot learn inner scales like the 19.5
      in ``exp(19.5*x)``. Uses only the standard library + numpy.
    - ``guided``: seed the population with dimensionally valid-by-construction
      candidates (unit-vector monomials of the target dimensionality, same-dim
      sums, dimensionless exp factors). Unguided random trees hit a specific
      cross-unit target roughly 0.03% of the time per candidate, so discovery
      outcomes were seed luck; guided seeding proposes structures such as
      ``m*v**2``, ``I*R`` or ``G*m1*m2/r**2`` directly.
    """

    def __init__(
        self,
        assurance_system: ClosedLoopAssuranceSystem,
        independent_vars: list[str] | None = None,
        target_unit_str: str = "dimensionless",
        extra_constants: list[str] | None = None,
        variables: list[str | sp.Symbol] | None = None,
        num_constants: int = 3,
        allow_exp: bool = True,
        allow_trig: bool = True,
        allow_log: bool = True,
        allow_sqrt: bool = True,
        extra_const_values: dict[str, tuple[float, str]] | None = None,
        template_priors: list[str | sp.Expr] | None = None,
        inner_fit: bool = True,
        inner_steps: int = 12,
        guided: bool = True,
        seed: int | None = 42,
        allow_power_mutation: bool = False,
        allow_rational_mutation: bool = False,
        allow_compose_mutation: bool = False,
        allow_logcomp_mutation: bool = False,
        allow_recursive_reduction: bool = False,
        allow_implicit_fitness: bool = False,
        implicit_var: str | None = None,
        nested_constraints: dict | None = None,
        allow_beam_seeding: bool = False,
    ):
        self.system = assurance_system
        vars_input = independent_vars if independent_vars is not None else (variables or [])
        self.var_names = [str(v) for v in vars_input]
        self.vars = [sp.Symbol(v) for v in self.var_names]
        self.target_unit_str = target_unit_str
        self.target_unit = self.system.engine.registry(target_unit_str)

        # Per-instance RNG to avoid global state corruption under concurrency
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        # Fitness is evaluated repeatedly for elites and Pareto candidates.
        # Cache within one search only; data/config changes create a new engine
        # or a new run and therefore cannot reuse stale scores.
        self._fitness_cache: dict[str, tuple[sp.Expr, float]] = {}

        # Constant symbols available for discovery
        default_const_names = ['G_grav', 'c_light', 'sigma_sb', 'h_bar', 'k_B', 'R_gas', 'g_earth']
        const_names = extra_constants if extra_constants is not None else default_const_names
        self.constants = [sp.Symbol(c) for c in const_names if c in self.system.constants_values]
        # Ephemeral random constants: dimensionless placeholders that become
        # constant features during coefficient fitting (so `C0 + x**2` can
        # discover `3 + 2*x**2`). Avoid naming variables `C0`, `C1`, ...
        self.erc_symbols = [sp.Symbol(f"C{i}") for i in range(max(0, num_constants))]
        for sym in self.erc_symbols:
            self.system.engine.register_variable(str(sym), "dimensionless")
        self.building_blocks = self.vars + self.constants + self.erc_symbols
        self.operators = [sp.Add, sp.Mul]
        self.allow_exp = bool(allow_exp)
        self.allow_trig = bool(allow_trig)
        self.allow_log = bool(allow_log)
        self.allow_sqrt = bool(allow_sqrt)
        self.inner_fit = bool(inner_fit)
        self.inner_steps = max(0, int(inner_steps))
        self.guided = bool(guided)
        # Optional mutation families + recursive reduction. All default OFF:
        # legacy RNG stream and behavior are exactly preserved unless opted in.
        self.allow_power_mutation = bool(allow_power_mutation)
        self.allow_rational_mutation = bool(allow_rational_mutation)
        self.allow_compose_mutation = bool(allow_compose_mutation)
        self.allow_logcomp_mutation = bool(allow_logcomp_mutation)
        self.allow_beam_seeding = bool(allow_beam_seeding)
        self.nested_constraints = nested_constraints
        self.allow_recursive_reduction = bool(allow_recursive_reduction)
        self.allow_implicit_fitness = bool(allow_implicit_fitness)
        self.implicit_var = implicit_var
        for name, spec in dict(extra_const_values or {}).items():
            try:
                value, unit_str = spec
            except (TypeError, ValueError) as e:
                raise ValueError(f"extra_const_values[{name!r}] must be a (value, unit_str) pair.") from e
            try:
                self.system.engine.register_variable(name, unit_str)
            except (ValueError, TypeError, AttributeError) as e:
                raise ValueError(f"Cannot register extra constant '{name}' with unit '{unit_str}'.") from e
            sym = sp.Symbol(name)
            self.system.constants_values[name] = float(value)
            if sym not in self.building_blocks:
                self.building_blocks.append(sym)
        self.template_exprs: list[sp.Expr] = []
        for prior in template_priors or []:
            try:
                if isinstance(prior, str):
                    from apedp.safety import safe_parse_expr

                    expr = safe_parse_expr(prior)
                else:
                    expr = sp.sympify(prior)
                unit = self.system.engine.check_expression_consistency(expr)
                if unit.dimensionality == self.target_unit.dimensionality:
                    self.template_exprs.append(expr)
                else:
                    logger.warning("Template prior %r is dimensionally inconsistent, skipping.", prior)
            except (ValueError, TypeError) as e:
                logger.warning("Template prior %r could not be parsed, skipping: %s", prior, e)
        # Populated by run_discovery; empty until a search has been run.
        self.stagnation_report: dict[str, Any] = {}

    def _generate_candidate_base(self, depth: int = 0, max_depth: int = 3) -> sp.Expr:
        # Seeded scientific search (seed=42 is a tested feature); secrets-style
        # non-determinism would break reproducibility, so stdlib random is correct here.
        if depth >= max_depth or random.random() < 0.3:  # nosec B311
            return random.choice(self.building_blocks)  # nosec B311

        choices: list = [*self.operators, sp.Pow]
        if self.allow_sqrt:
            choices.append(sp.sqrt)
        op = random.choice(choices)  # nosec B311
        if op == sp.Pow:
            base = self._generate_candidate(depth + 1, max_depth)
            # Half- and third-integer exponents so laws with fractional powers
            # (Hall-Petch b/sqrt(d), cube-root scalings) are representable.
            exponent = random.choice(  # nosec B311
                [-2, -1, 1, 2, 3, 4, sp.Rational(1, 2), sp.Rational(-1, 2),
                 sp.Rational(3, 2), sp.Rational(-3, 2), sp.Rational(1, 3),
                 sp.Rational(-1, 3), sp.Rational(2, 3), sp.Rational(-2, 3)]
            )
            return sp.Pow(base, exponent)
        if op == sp.sqrt:
            return sp.sqrt(self._generate_candidate(depth + 1, max_depth))
        else:
            left = self._generate_candidate(depth + 1, max_depth)
            right = self._generate_candidate(depth + 1, max_depth)
            return op(left, right)

    def _generate_candidate(self, depth: int = 0, max_depth: int = 3) -> sp.Expr:
        # Wrapper preserving the exact legacy RNG stream when all nonlinear
        # flags are off. Any enabled nonlinear changes the stream by design.
        if not (self.allow_exp or self.allow_trig or self.allow_log):
            return self._generate_candidate_base(depth, max_depth)
        if depth >= max_depth or random.random() < 0.25:  # nosec B311
            return random.choice(self.building_blocks)  # nosec B311
        nonlinear_picks: list[str] = []
        if self.allow_exp:
            nonlinear_picks.append("exp")
        if self.allow_trig:
            nonlinear_picks.extend(["sin", "cos"])
        if self.allow_log:
            nonlinear_picks.append("log")
        if nonlinear_picks and random.random() < 0.25:  # nosec B311
            inner = self._random_dimensionless_inner(depth, max_depth)
            if inner is not None:
                kind = random.choice(nonlinear_picks)  # nosec B311
                if kind == "exp":
                    return sp.exp(inner)
                if kind == "sin":
                    return sp.sin(inner)
                if kind == "cos":
                    return sp.cos(inner)
                if kind == "log":
                    return sp.log(inner)
        return self._generate_candidate_base(depth, max_depth)

    def _block_dim_dict(self) -> dict[sp.Symbol, dict[str, float]]:
        """Dimensionality of every building block, skipping unregistered ones.

        Cached per block; failures are retried on later calls so variables
        registered after construction are picked up.
        """
        cache: dict[sp.Symbol, dict[str, float]] = self.__dict__.setdefault("_block_dim_cache", {})
        for block in self.building_blocks:
            if block not in cache:
                try:
                    cache[block] = _dim_dict(self.system.engine.check_expression_consistency(block))
                except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                    logger.debug("Skipping dimension-less block %s (%s)", block, e)
        return dict(cache)

    def _monomial_for_dim(
        self,
        target: dict[str, float],
        block_dims: dict[sp.Symbol, dict[str, float]],
        tries: int = 100,
    ) -> sp.Expr | None:
        """Random unit-vector monomial with exactly ``target`` dimensionality.

        Searches exponent vectors over non-ERC blocks (``a**2 * b**-1``);
        every result is dimensionally valid by construction, divisions
        included. Returns None when the budget is exhausted.
        """
        erc_set = set(self.erc_symbols)
        items = [(b, d) for b, d in block_dims.items() if b not in erc_set]
        if not items:
            return None
        # Integer exponents are searched first: they cover the bulk of
        # dimensional targets, and widening the alphabet dilutes this
        # exact-match search (measured: 13.3% -> 6.1% hit rate on a joule
        # target). Halves keep Hall-Petch reachable; thirds keep cube-root
        # scalings reachable. All arithmetic here is exact Fraction so thirds
        # compare exactly (float 1/3 would not).
        from fractions import Fraction

        def _f(d: dict[str, float]) -> dict[str, Fraction]:
            # limit_denominator maps float 1/3 (0.333...) to exact 1/3 so
            # thirds compare exactly; ints and halves are unaffected.
            return {k: Fraction(v).limit_denominator(12) for k, v in d.items()}

        def _add(a: dict[str, Fraction], b: dict[str, Fraction]) -> dict[str, Fraction]:
            out = dict(a)
            for k, v in b.items():
                out[k] = out.get(k, Fraction(0)) + v
                if out[k] == 0:
                    del out[k]
            return out

        target_f = _f(target)
        items_f = [(b, _f(d)) for b, d in items]
        integer_exps: list[Fraction] = [Fraction(-2), Fraction(-1), Fraction(1), Fraction(1), Fraction(2)]
        extended_exps: list[Fraction] = [*integer_exps, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3), Fraction(2, 3), Fraction(-2, 3)]
        integer_budget = int(tries * 0.7)
        for attempt in range(tries):
            exps_pool = integer_exps if attempt < integer_budget else extended_exps
            picks = [random.choice(items_f) for _ in range(random.randint(1, 4))]  # nosec B311
            exps = [random.choice(exps_pool) for _ in picks]  # nosec B311
            acc: dict[str, Fraction] = {}
            for (_, dim), power in zip(picks, exps, strict=False):
                acc = _add(acc, {k: v * power for k, v in dim.items() if v * power != 0})
            if acc == target_f:
                expr: sp.Expr = sp.Integer(1)
                for (block, _), power in zip(picks, exps, strict=False):
                    expr = expr * sp.Pow(block, sp.Rational(power.numerator, power.denominator))
                return expr
        return None

    def _generate_guided_candidate(self, depth: int = 0, max_depth: int = 3) -> sp.Expr:
        """Dimensionally valid-by-construction candidate for the target unit.

        Strategies: exact target monomial (covers ``m*v**2``, ``I*R``,
        ``G*m1*m2/r**2`` shapes including divisions), same-dimension sums
        (``M + C*M``, keeping additive-offset reachability), sums of two
        independent target monomials (``h*omega - phi``, ``P + rho*v**2``),
        and — with ``allow_exp``/``allow_trig``/``allow_log`` — dimensionless
        nonlinear factors alone or multiplying a target monomial
        (``A*exp(-E/(R*T))`` shape). Falls back to the legacy generator when
        monomial search misses.
        """
        target = _dim_dict(self.target_unit)
        block_dims = self._block_dim_dict()
        scale = random.choice(self.erc_symbols) if self.erc_symbols else sp.Integer(1)  # nosec B311
        roll = random.random()  # nosec B311
        if roll < 0.45:
            mono = self._monomial_for_dim(target, block_dims)
            if mono is not None:
                return mono
        elif roll < 0.65:
            mono = self._monomial_for_dim(target, block_dims)
            if mono is not None:
                return mono + scale * mono
        elif roll < 0.80:
            first = self._monomial_for_dim(target, block_dims, tries=60)
            second = self._monomial_for_dim(target, block_dims, tries=60)
            if first is not None and second is not None:
                return first + scale * second
        elif self.allow_exp or self.allow_trig or self.allow_log:
            dimless = self._monomial_for_dim({}, block_dims)
            if dimless is not None and dimless != 1:
                inner = dimless
                if not isinstance(inner, sp.Expr):
                    inner = sp.sympify(inner)
                mono = self._monomial_for_dim(target, block_dims, tries=30)
                kinds: list[str] = []
                if self.allow_exp:
                    kinds.append("exp")
                if self.allow_trig:
                    kinds.extend(["sin", "cos"])
                if self.allow_log:
                    kinds.append("log")
                kind = random.choice(kinds) if kinds else "exp"  # nosec B311
                factor = (
                    sp.exp(inner)
                    if kind == "exp"
                    else sp.sin(inner)
                    if kind == "sin"
                    else sp.cos(inner)
                    if kind == "cos"
                    else sp.log(inner)
                )
                if target == {} or mono is None:
                    return factor
                return mono * factor
        return self._generate_candidate(depth, max_depth)

    def _random_dimensionless_inner(self, depth: int, max_depth: int) -> sp.Expr | None:
        """Builds a dimensionless sub-expression for exp()/sin()/cos()/log().

        Uses the nonlinearity-free base generator so transcendentals never
        nest, and verifies the dimensional gatekeeper rule (transcendental
        arguments must be dimensionless) before accepting.
        """
        dimensionless = self.system.engine.registry.dimensionless.dimensionality
        for _ in range(8):
            inner = self._generate_candidate_base(depth + 1, max_depth)
            try:
                if self.system.engine.check_expression_consistency(inner).dimensionality == dimensionless:
                    return inner
            except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                logger.debug("Discarding dimensioned exp() argument (%s)", e)
                continue
        return None

    @staticmethod
    def _ercs_inside_exp(candidate: sp.Expr, erc_symbols: list[sp.Symbol]) -> list[sp.Symbol]:
        """Ephemeral constants appearing inside exp() arguments.

        These are the inner scales (e.g. the 19.5 in ``exp(19.5*x)``) that
        the outer linear least-squares fit cannot learn, since it only fits
        one coefficient per additive term with ERCs fixed at 1.0.
        """
        erc_set = set(erc_symbols)
        found: set[sp.Symbol] = set()
        for node in candidate.atoms(sp.exp):
            found.update(s for s in node.args[0].free_symbols if s in erc_set)
        return sorted(found, key=str)

    def _lambdify_cached(self, syms: list[sp.Symbol], expr: sp.Expr):
        """Memoized ``sp.lambdify``: elites are re-scored every generation."""
        cache: dict = self.__dict__.setdefault("_lambda_cache", {})
        if len(cache) > 2000:
            cache.clear()
        key = (str(expr), tuple(str(s) for s in syms))
        fn = cache.get(key)
        if fn is None:
            fn = sp.lambdify(syms, expr, modules=["numpy"])
            cache[key] = fn
        return fn

    def _linear_outer_fit(
        self,
        candidate: sp.Expr,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        erc_values: dict[str, float],
    ) -> tuple[sp.Expr, float]:
        """Linear least-squares fit of outer coefficients for fixed ERC values.

        With all values at 1.0 this is exactly the legacy calibration path.
        """
        try:
            var_names = sorted(x_data.keys())
            syms = [sp.Symbol(k) for k in var_names]
            # Substitute physical constants (and ephemeral constants at their
            # current values) into candidate before lambdify
            subs = {sp.Symbol(k): v for k, v in self.system.constants_values.items()}
            subs.update({s: float(erc_values.get(str(s), 1.0)) for s in self.erc_symbols})
            args = [x_data[k] for k in var_names]

            def evaluate(expr: sp.Expr) -> np.ndarray:
                f = self._lambdify_cached(syms, expr.subs(subs))
                # Fitting the outer coefficients evaluates candidate parts on the
                # data, and an exp/inner-scale term routinely overflows. The
                # non-finite design matrix below rejects the candidate outright,
                # so the warning is expected rather than a signal.
                with quiet_float_warnings():
                    y_pred = f(*args)
                if not isinstance(y_pred, np.ndarray):
                    y_pred = np.full_like(y_data, float(y_pred))
                result: np.ndarray = y_pred
                return result

            base_const, term_tuple = candidate.as_coeff_add()
            parts: list[sp.Expr] = ([sp.sympify(base_const)] if base_const != 0 else []) + list(term_tuple)
            if not parts:
                return candidate, float('inf')
            design = np.column_stack([evaluate(part) for part in parts])
            if not np.all(np.isfinite(design)):
                return candidate, float('inf')

            coeffs, _, _, _ = np.linalg.lstsq(design, y_data, rcond=None)
            scale = float(np.sqrt(np.mean(y_data ** 2))) or 1.0
            tol = 1e-9 * (1.0 + scale)
            # Largest coefficients first so at least one term always survives pruning
            order = sorted(range(len(parts)), key=lambda i: abs(float(coeffs[i])), reverse=True)
            fitted: list[tuple[sp.Expr, sp.Expr, int]] = []
            for i in order:
                if abs(float(coeffs[i])) < tol and fitted:
                    continue
                fitted.append((_snap_coefficient(float(coeffs[i])), parts[i], i))

            rebuilt = [part if coeff == 1 else sp.Mul(coeff, part) for coeff, part, _ in fitted]
            calibrated_expr = sp.Add(*rebuilt)
            # Bake ephemeral constants back to their fitted values so the
            # returned expression is self-contained. Exactly-1.0 values keep
            # the legacy Integer(1) substitution.
            bake = {
                s: (sp.Integer(1) if erc_values.get(str(s), 1.0) == 1.0
                    else sp.Float(float(f"{erc_values[str(s)]:.6g}")))
                for s in self.erc_symbols
            }
            calibrated_expr = calibrated_expr.subs(bake)
            prediction = sum(float(coeff) * design[:, i] for coeff, _, i in fitted)
            mse = float(np.mean((prediction - y_data) ** 2))
            return calibrated_expr, mse
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as e:
            logger.warning("Parameter optimization failed, returning inf: %s", e)
            return candidate, float('inf')

    def _refine_inner_ercs(
        self,
        candidate: sp.Expr,
        inner_syms: list[sp.Symbol],
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        erc_values: dict[str, float],
        best_mse: float,
    ) -> tuple[dict[str, float], float]:
        """Coordinate-wise log-uniform hill-climb of inner ERC values.

        Each proposal re-runs the linear outer fit, so outer coefficients
        stay optimal for every tried inner scale. Standard library + numpy
        only; early-stops after 10 non-improving proposals.
        """
        theta = dict(erc_values)
        stalled = 0
        for _ in range(self.inner_steps):
            target = random.choice(inner_syms)  # nosec B311
            old = theta[str(target)]
            if old == 0.0 or not np.isfinite(old):
                continue
            # Two-scale proposals: wide jumps for range-finding, narrow
            # steps for fine-tuning near the optimum.
            width = random.choice([0.15, 0.7])  # nosec B311
            trial = dict(theta)
            trial[str(target)] = old * 10 ** random.uniform(-width, width)  # nosec B311
            _, mse = self._linear_outer_fit(candidate, x_data, y_data, trial)
            if np.isfinite(mse) and mse < best_mse * (1.0 - 1e-9):
                theta, best_mse, stalled = trial, mse, 0
            else:
                stalled += 1
                if stalled >= 10:
                    break
        return theta, best_mse

    def optimize_parameters(
        self,
        candidate: sp.Expr,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray
    ) -> tuple[sp.Expr, float]:
        """
        Calibrates continuous coefficients against empirical data.

        Decomposes the candidate into additive terms (ephemeral constants such
        as ``C0`` evaluate to 1.0 and become constant features) and solves a
        linear least-squares problem for one coefficient per term. Fitted
        values snap to exact physical fractions when close, and negligible
        terms are pruned.
        When ``inner_fit`` is enabled, ERCs appearing inside ``exp()`` are
        additionally hill-climbed (outer fit re-run per proposal) so inner
        scales like the 19.5 in ``exp(19.5*x)`` can be learned.
        """
        erc_values = {str(s): 1.0 for s in self.erc_symbols}
        calibrated_expr, mse = self._linear_outer_fit(candidate, x_data, y_data, erc_values)
        if not self.inner_fit or self.inner_steps <= 0 or not np.isfinite(mse):
            return calibrated_expr, mse
        inner_syms = self._ercs_inside_exp(candidate, self.erc_symbols)
        if not inner_syms:
            return calibrated_expr, mse
        theta, refined_mse = self._refine_inner_ercs(
            candidate, inner_syms, x_data, y_data, erc_values, mse
        )
        if np.isfinite(refined_mse) and refined_mse < mse:
            calibrated_expr, _ = self._linear_outer_fit(candidate, x_data, y_data, theta)
            return calibrated_expr, refined_mse
        return calibrated_expr, mse

    def implicit_fitness(self, candidate: sp.Expr, x_data: dict[str, np.ndarray],
                           y_data: np.ndarray) -> float:
        """Score a candidate as an implicit residual F(target)=0.

        Solves for the target variable per data row with
        eval_extended.evaluate_implicit and returns MSE vs observed values.
        Opt-in only (allow_implicit_fitness + implicit_var set); unsolved
        rows count against the candidate. Nascent: per-row solves are slow,
        so keep populations and datasets small when enabled.
        """
        try:
            from apedp.eval_extended import evaluate_implicit
        except ImportError:
            # evaluate_implicit lives behind the optional extended-evaluation
            # extra; without it implicit fitness is unavailable by design.
            logger.debug("evaluate_implicit unavailable; implicit fitness "
                         "declines the candidate")
            return float("inf")
        target = getattr(self, "implicit_var", None)
        if not target:
            return float("inf")
        try:
            names = list(x_data.keys())
            n = len(np.asarray(y_data, float).ravel())
            errs = []
            formula = str(candidate)
            obs_all = np.asarray(y_data, float).ravel()
            for i in range(n):
                params = {k: float(np.asarray(x_data[k], float).ravel()[i])
                          for k in names}
                obs = float(obs_all[i])
                try:
                    # Guess from observed value: tests whether the data
                    # satisfies the residual (root-consistency), converging
                    # to the nearest root otherwise.
                    pred = evaluate_implicit(formula, target, obs, params,
                                             timeout=0.5)
                    errs.append((pred - obs) ** 2)
                except (ValueError, TypeError, ArithmeticError,
                        NotImplementedError, TimeoutError) as exc:
                    # evaluate_implicit's contract: ValueError/TypeError for a
                    # malformed residual, ArithmeticError when nsolve fails,
                    # NotImplementedError for an unsolvable form, TimeoutError
                    # past the bounded-solver budget. The row is dropped and
                    # the candidate is scored on the rows that solved.
                    logger.debug("implicit solve skipped for %s: %s",
                                 target, exc)
                    continue
            if not errs:
                return float("inf")
            return float(sum(errs) / len(errs))
        except (TypeError, ValueError, ArithmeticError):
            return float("inf")

    def fitness(self, candidate: sp.Expr, x_data: dict[str, np.ndarray], y_data: np.ndarray) -> float:
        if bool(getattr(self, "allow_implicit_fitness", False)):
            return self.implicit_fitness(candidate, x_data, y_data)
        # 1. Strict Dimensional Gatekeeping
        try:
            unit = self.system.engine.check_expression_consistency(candidate)
            if unit.dimensionality != self.target_unit.dimensionality:
                return float('inf')
        except (TypeError, ValueError, NotImplementedError, ArithmeticError) as e:
            logger.warning("Dimensional gatekeeping failed, returning inf: %s", e)
            return float('inf')

        # 2. Continuous Parameter Optimization + Occam's Complexity Penalty
        cache_key = str(candidate)
        cached = self._fitness_cache.get(cache_key)
        try:
            if cached is None:
                cached = self.optimize_parameters(candidate, x_data, y_data)
                self._fitness_cache[cache_key] = cached
            calibrated_expr, mse = cached
            if np.isinf(mse) or np.isnan(mse):
                return float('inf')
            complexity = len(list(sp.preorder_traversal(calibrated_expr)))
            return mse + 1e-4 * complexity
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.warning("Parameter optimization failed, returning inf: %s", e)
            return float('inf')

    def _check_dimensional(self, expr: sp.Expr) -> bool:
        """Check expression against the target unit (for Pareto bookkeeping)."""
        try:
            unit = self.system.engine.check_expression_consistency(expr)
            return unit.dimensionality == self.target_unit.dimensionality
        except (TypeError, ValueError, NotImplementedError, ArithmeticError):
            return False

    def _power_mutate(self, parent: sp.Expr) -> sp.Expr:
        """Grow/shrink integer exponents (T**2 -> T**3/T**4) or wrap an atom
        in a small integer power. Dimensionally invalid results score inf in
        fitness and are selected out, so this is safe to propose broadly."""
        try:
            int_pows = []
            for node in sp.preorder_traversal(parent):
                if isinstance(node, sp.Pow):
                    try:
                        if node.exp.is_Integer and abs(int(node.exp)) <= 6:
                            int_pows.append(node)
                    except (TypeError, ValueError, AttributeError):
                        continue
            if int_pows and random.random() < 0.7:  # nosec B311
                target = random.choice(int_pows)  # nosec B311
                new_exp = int(target.exp) + random.choice([1, -1])  # nosec B311
                if new_exp == 0 or abs(new_exp) > 6:
                    return parent
                return parent.xreplace({target: sp.Pow(target.base, new_exp)})
            atoms = [n for n in sp.preorder_traversal(parent) if n.is_Symbol]
            base = random.choice(atoms) if atoms else parent  # nosec B311
            return parent.xreplace(
                {base: sp.Pow(base, random.choice([2, 3, 4]))})  # nosec B311
        except (TypeError, ValueError, AttributeError, ArithmeticError):
            return parent

    def _rational_mutate(self, parent: sp.Expr) -> sp.Expr:
        """Grow rational forms: divide by a subexpression, or wrap an atom in
        a fractional power (+-1/2, +-1/3, 2/3). Zero-divisions score inf in
        fitness and are selected out, so this is safe to propose broadly."""
        try:
            if random.random() < 0.5:  # nosec B311
                nodes = [n for n in sp.preorder_traversal(parent)
                         if isinstance(n, sp.Pow)]
                if nodes:
                    target = random.choice(nodes)  # nosec B311
                    new_exp = random.choice(  # nosec B311
                        [sp.Rational(1, 2), sp.Rational(-1, 2),
                         sp.Rational(1, 3), sp.Rational(-1, 3),
                         sp.Rational(2, 3)])
                    return parent.xreplace(
                        {target: sp.Pow(target.base, new_exp)})
            # Divisor pool is the full variable set, NOT the parent's symbols:
            # dividing by own symbols can only cancel (F*L**3/(F*L) -> L**2)
            # and can never introduce new denominator structure like E*I.
            pool = list(getattr(self, "vars", None) or [])
            if not pool:
                pool = [n for n in sp.preorder_traversal(parent)
                        if n.is_Symbol]
            if not pool:
                return parent
            divisor = random.choice(pool)  # nosec B311
            if random.random() < 0.5:  # nosec B311
                others = [s for s in pool if str(s) != str(divisor)]
                if others and random.random() < 0.8:  # nosec B311
                    divisor = divisor * random.choice(others)  # nosec B311
                return parent / divisor
            return parent.xreplace(
                {divisor: sp.Pow(divisor, random.choice(  # nosec B311
                    [sp.Rational(1, 2), sp.Rational(1, 3)]))})
        except (TypeError, ValueError, AttributeError, ArithmeticError):
            return parent

    def _compose_mutate(self, parent: sp.Expr) -> sp.Expr:
        """Nest a function application: wrap an atom in sin/exp/log/sqrt.
        Dimensionally invalid results score inf in fitness and are selected
        out, so this is safe to propose broadly. nested_constraints maps an
        enclosing function name to forbidden child names (e.g. no exp inside
        exp); None means legacy unconstrained behavior."""
        try:
            fns = []
            if self.allow_trig:
                fns.extend([sp.sin, sp.cos])
            if self.allow_exp:
                fns.append(sp.exp)
            if self.allow_log:
                fns.append(sp.log)
            fns.append(sp.sqrt)
            if not fns:
                return parent
            syms = [n for n in sp.preorder_traversal(parent) if n.is_Symbol]
            if not syms:
                return parent
            target = random.choice(syms)  # nosec B311
            fn = random.choice(fns)  # nosec B311
            rules = getattr(self, "nested_constraints", None) or {}
            if rules:
                ancestors: set = set()
                stack: list = [(parent, ())]
                while stack:
                    node, path = stack.pop()
                    if node is target:
                        ancestors.update(
                            type(a).__name__ for a in path
                            if isinstance(a, sp.Function))
                    for arg in getattr(node, "args", ()):
                        stack.append((arg, (*path, node)))
                banned = set()
                for anc in ancestors:
                    banned.update(rules.get(anc, ()))
                fname = ("sin" if fn is sp.sin else
                         "cos" if fn is sp.cos else
                         "exp" if fn is sp.exp else
                         "log" if fn is sp.log else "sqrt")
                if fname in banned:
                    return parent
            return parent.xreplace({target: fn(target)})
        except (TypeError, ValueError, AttributeError, ArithmeticError):
            return parent

    def _logcomp_mutate(self, parent: sp.Expr) -> sp.Expr:
        """Nest logarithmic compositions: log(log(x)), log(1+1/x),
        log(1+x). Targets friction-type log-log structure that single-log
        wrapping cannot reach. Invalid results score inf in fitness."""
        try:
            syms = [n for n in sp.preorder_traversal(parent) if n.is_Symbol]
            if not syms:
                return parent
            target = random.choice(syms)  # nosec B311
            choice = random.random()  # nosec B311
            if choice < 0.4:
                return parent.xreplace({target: sp.log(sp.log(target))})
            if choice < 0.7:
                return parent.xreplace({target: sp.log(1 + 1 / target)})
            return parent.xreplace({target: sp.log(1 + target)})
        except (TypeError, ValueError, AttributeError, ArithmeticError):
            return parent

    def mutate(self, parent: sp.Expr) -> sp.Expr:
        roll = random.random()  # nosec B311
        if roll < 0.5:
            # Dimension-preserving mutation: scale by dimensionless factors
            scale = random.choice([0.5, 2.0, sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(2, 3), 1.5])  # nosec B311
            return parent * scale
        if roll < 0.7 and self.erc_symbols:
            # Constant-shift mutation: append an ephemeral constant term so
            # additive offsets (e.g. the `+3` in `2*x**2 + 3`) are reachable
            return parent + random.choice(self.erc_symbols)  # nosec B311
        if self.allow_power_mutation and roll < 0.8:
            return self._power_mutate(parent)
        if self.allow_rational_mutation and roll < 0.9:
            return self._rational_mutate(parent)
        if self.allow_compose_mutation and roll < 0.95:
            return self._compose_mutate(parent)
        if self.allow_logcomp_mutation and roll < 0.97:
            return self._logcomp_mutate(parent)
        return self._generate_candidate(depth=1, max_depth=2)

    def _try_recursive_reduction(
        self,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
    ) -> tuple[sp.Expr | None, float, str] | None:
        """Attempt a direct separability solve. Returns None to fall through.

        Structure is built from Symbol objects directly: re-parsing the
        shape string would misread single-letter physics symbols (E -> Euler,
        I -> imaginary) via the shared safe parser.
        """
        try:
            det = detect_multiplicative_split(x_data, y_data)
            if det.get("kind") != "multiplicative":
                return None
            struct = sp.Integer(1)
            names = []
            for name, p in zip(det["variables"], det["exponents"], strict=False):
                sym = sp.Symbol(name)
                names.append(name)
                exp: sp.Expr = (sp.Integer(round(p))
                                if abs(p - round(p)) < 1e-9
                                else sp.sympify(p))
                struct = struct * sym ** exp
            cols = {k: np.asarray(x_data[k], float).ravel() for k in names}
            y = np.asarray(y_data, float).ravel()
            basis = np.array([float(struct.subs(
                {sp.Symbol(k): cols[k][i] for k in names}))
                for i in range(len(y))])
            denom = float(np.dot(basis, basis)) + 1e-300
            const = float(np.dot(basis, y)) / denom
            best = const * struct
            pred = np.array([float(best.subs(
                {sp.Symbol(k): cols[k][i] for k in names}))
                for i in range(len(y))])
            mse = float(np.mean((pred - y) ** 2))
            return best, mse, (
                "recursive-reduction r^2="
                f"{det['r_squared']} shape="
                f"{recombine_power_law(det['variables'], det['exponents'])}")
        except (TypeError, ValueError, ArithmeticError):
            return None

    def run_discovery(
        self,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        generations: int = 30,
        pop_size: int = 100,
        seed: int | None = 42
    ) -> tuple[sp.Expr | None, float, list[dict[str, Any]]]:
        """Runs the evolutionary search loop and returns (best_candidate, best_mse, history)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._fitness_cache.clear()

        if self.allow_recursive_reduction:
            reduced = self._try_recursive_reduction(x_data, y_data)
            if reduced is not None:
                best, mse, note = reduced
                return best, mse, [{"reduction": note}]

        population: list[sp.Expr] = []
        target_dim = self.target_unit.dimensionality
        attempts = 0

        # Seed population with physics-informed template priors first
        for prior_expr in self.template_exprs:
            if len(population) >= pop_size:
                break
            if all(prior_expr != existing for existing in population):
                population.append(prior_expr)

        # For wide, correlated feature sets (for example lagged sensor
        # histories), give evolution a compact affine alternative selected
        # only from the supplied training data.  This is a seed, not a
        # replacement for symbolic search, and BIC charges each retained term.
        selected_affine = bic_forward_select(x_data, y_data)
        selected_names = selected_affine.get("terms", [])
        if selected_names:
            sparse_terms = [sp.Symbol(name) for name in selected_names]
            if self.erc_symbols:
                sparse_terms.insert(0, self.erc_symbols[0])
            sparse_seed = sp.Add(*sparse_terms)
            try:
                if (
                    len(population) < pop_size
                    and self.system.engine.check_expression_consistency(
                        sparse_seed
                    ).dimensionality == target_dim
                    and all(sparse_seed != existing for existing in population)
                ):
                    population.append(sparse_seed)
            except (ValueError, TypeError, ArithmeticError, NotImplementedError):
                pass

        # A symbolic search must not lose an available affine baseline.  The
        # outer fitter calibrates one coefficient per additive term, so this
        # single seed covers an intercept plus every declared input.
        affine_seed = sp.Add(self.erc_symbols[0], *self.vars) if self.erc_symbols else sp.Add(*self.vars)
        try:
            if (
                len(population) < pop_size
                and self.system.engine.check_expression_consistency(
                    affine_seed
                ).dimensionality == target_dim
                and all(affine_seed != existing for existing in population)
            ):
                population.append(affine_seed)
        except (ValueError, TypeError, ArithmeticError, NotImplementedError):
            pass

        # Seed population with dimensionally consistent candidates
        seed_fn = self._generate_guided_candidate if self.guided else self._generate_candidate
        while len(population) < pop_size and attempts < pop_size * 20:
            cand = seed_fn(depth=0, max_depth=3)
            try:
                if self.system.engine.check_expression_consistency(cand).dimensionality == target_dim:
                    population.append(cand)
            except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                logger.warning("Population seeding failed (%s)", e)
            attempts += 1

        # Beam seeding (opt-in): breadth-capped construction forms enter
        # the population after dimensional filtering. Default off.
        if self.allow_beam_seeding:
            try:
                from apedp.safety import safe_parse_expr
                for form in beam_search_expressions(
                        self.var_names, depth=2, width=8, seed=seed):
                    if len(population) >= pop_size:
                        break
                    try:
                        cand = safe_parse_expr(form)
                    except (TypeError, ValueError):
                        continue
                    try:
                        if (self.system.engine.check_expression_consistency(
                                cand).dimensionality == target_dim
                                and all(cand != existing
                                        for existing in population)):
                            population.append(cand)
                    except (ValueError, TypeError, ArithmeticError,
                            NotImplementedError):
                        continue
            except _PROBE_ERRORS as exc:
                logger.debug("Diversity seeding failed; falling back to filler: %s", exc)

        # Fallback filler
        while len(population) < pop_size:
            population.append(self._generate_candidate(depth=0, max_depth=2))

        best_candidate: sp.Expr | None = None
        best_score = float('inf')
        history: list[dict[str, Any]] = []
        improvements = 0
        generations_without_improvement = 0
        pareto_frontier = ParetoFrontierTracker()

        for gen in range(generations):
            scored = []
            for p in population:
                score = self.fitness(p, x_data, y_data)
                scored.append((score, p))
            scored.sort(key=lambda item: item[0])

            gen_best_score, gen_best_expr = scored[0]
            improved = gen_best_score < best_score
            if improved:
                best_score = gen_best_score
                best_candidate, _ = self.optimize_parameters(gen_best_expr, x_data, y_data)
                improvements += 1
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            for score, expr in scored:
                complexity = len(list(sp.preorder_traversal(expr)))
                metric = ScientificImprovementMetric(
                    i_fit=1.0 / (1.0 + score),
                    i_stability=1.0 / (1.0 + score),
                    i_physicality=1.0 if self._check_dimensional(expr) else 0.0,
                    i_generality=0.0,
                    i_complexity=1.0 / (1.0 + complexity * 0.1),
                    i_falsifiability=1.0,
                )
                pareto_frontier.consider_candidate(expr, metric)

            history.append({
                'generation': gen,
                'best_score': best_score,
                'best_expr': best_candidate,
                'improved': improved,
                'generations_without_improvement': generations_without_improvement,
                'pareto_frontier_size': len(pareto_frontier.frontier),
            })

            if best_score < 1e-25:
                break

            # Selection: top 20% elites, plus Pareto-optimal candidates not
            # already present (diversity without changing legacy behavior
            # when the frontier adds nothing new).
            elites = [p for _, p in scored[:max(1, int(pop_size * 0.2))]]
            next_pop = list(elites)
            elite_exprs = {str(expr) for expr in elites}
            for expr in pareto_frontier.get_frontier():
                if str(expr) not in elite_exprs:
                    next_pop.append(expr)
                    elite_exprs.add(str(expr))
            while len(next_pop) < pop_size:
                parent = random.choice(elites)  # nosec B311
                next_pop.append(self.mutate(parent))
            population = next_pop[:pop_size]

        # Stagnation signal: the first generation normally improves on the
        # seeded population, so ``improvements <= 1`` means the search added
        # nothing beyond its own initialisation. That is only a problem when
        # the result is also a poor fit -- an immediate exact fit is
        # convergence, not stagnation -- so the verdict is gated on a
        # scale-free residual.
        y_arr = np.asarray(y_data, dtype=float)
        y_variance = float(np.var(y_arr))
        if y_variance > 0.0:
            normalized_score = float(best_score) / y_variance
        elif np.isfinite(best_score):
            normalized_score = 0.0  # constant target with a finite residual
        else:
            normalized_score = float('inf')
        added_nothing = improvements <= 1
        self.stagnation_report = {
            'generations_run': len(history),
            'generations_requested': generations,
            'improvements': improvements,
            'generations_without_improvement': generations_without_improvement,
            'added_nothing_over_initialization': added_nothing,
            'final_normalized_score': normalized_score,
            'stagnated': added_nothing and normalized_score > 1e-3,
            'converged_early': len(history) < generations,
        }

        return best_candidate, best_score, history


    def run_discovery_additive(self, x_data: dict[str, np.ndarray],
                               y_data: np.ndarray,
                               max_terms: int = 5,
                               generations: int = 30,
                               pop_size: int = 60,
                               seed: int | None = 42,
                               min_improvement: float = 0.01,
                               residual_threshold: float = 0.05) -> tuple:
        """Iterative residual fitting for additive multi-term models.

        Runs run_discovery repeatedly: fit best term -> subtract from target ->
        repeat on residuals -> combine terms with linear outer fit.
        Returns (combined_expression, mse, terms_list, history_list).
        """
        import numpy as np
        import sympy as sp

        y = np.asarray(y_data, float).ravel()
        y_mean = float(np.mean(y))
        y_std = float(np.std(y)) + 1e-300
        residuals = y - y_mean  # center
        terms = []
        histories = []

        for _term_idx in range(max_terms):
            if float(np.std(residuals)) < residual_threshold * y_std:
                break
            # Run GA on current residuals
            best, mse, history = self.run_discovery(
                x_data, residuals, generations=generations, pop_size=pop_size, seed=seed)
            if best is None:
                break
            # Evaluate best term on data
            try:
                syms = sorted(best.free_symbols, key=str)
                fn = sp.lambdify(syms, best, "numpy")
                cols = [np.asarray(x_data[str(s)]) for s in syms]
                term_pred = np.asarray(fn(*cols), dtype=float).ravel()
            except _PROBE_ERRORS as exc:
                logger.debug("term prediction failed; stopping residual "
                             "decomposition: %s", exc)
                break
            # Check if term explains significant variance
            r2 = float(np.sum((term_pred - np.mean(term_pred))**2)) / float(np.sum((residuals - np.mean(residuals))**2) + 1e-300)
            if r2 < min_improvement:
                break
            terms.append((best, term_pred))
            histories.append(history)
            # Subtract term from residuals
            residuals = residuals - term_pred
            if float(np.std(residuals)) < 1e-6:
                break

        # Final linear combination of all terms (with constant)
        if not terms:
            return sp.Integer(0), float('inf'), [], []
        try:
            A = np.column_stack([tp for _, tp in terms] + [np.ones_like(y)])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            combined = sum(c * t for c, (t, _) in zip(coef[:-1], terms, strict=False)) + coef[-1]
            final_pred = A @ coef
            mse = float(np.mean((final_pred - y) ** 2))
            return combined, mse, terms, histories
        except _PROBE_ERRORS as exc:
            # Fallback: simple sum
            logger.debug("outer linear refit failed; using simple sum: %s", exc)
            combined = sum(t for t, _ in terms) + y_mean
            try:
                syms = sorted(combined.free_symbols, key=str)
                fn = sp.lambdify(syms, combined, "numpy")
                cols = [np.asarray(x_data[str(s)]) for s in syms]
                final_pred = np.asarray(fn(*cols), dtype=float).ravel()
                mse = float(np.mean((final_pred - y) ** 2))
            except _PROBE_ERRORS as exc:
                logger.debug("fallback combination unscorable: %s", exc)
                mse = float('inf')
            return combined, mse, terms, histories


    def run_discovery_joint(self, x_data: dict[str, np.ndarray],
                            y_data: np.ndarray,
                            n_candidates: int = 200,
                            generations: int = 15,
                            pop_size: int = 60,
                            seed: int | None = 42,
                            lasso_alpha: float = 0.01,
                            max_terms: int = 10) -> tuple:
        """Joint sparse selection for additive multi-term models.

        1. Generate diverse candidate expressions via GA (short runs, different seeds)
        2. Evaluate all candidates on the data
        3. Use Lasso to select sparse combination
        4. Return combined expression with coefficients

        This avoids greedy convolution by selecting all terms simultaneously.
        Returns (combined_expression, mse, selected_terms, all_candidates).
        """
        import numpy as np
        import sympy as sp
        from sklearn.linear_model import LassoCV

        y = np.asarray(y_data, float).ravel()
        n = len(y)

        # Generate candidate expressions via GA
        candidates = []
        seen = set()
        n_seeds = max(1, n_candidates // 5)
        for i in range(n_seeds):
            s = (seed + i * 100) if seed is not None else None
            best, _, _ = self.run_discovery(
                x_data, y, generations=generations, pop_size=pop_size, seed=s)
            if best is not None:
                # Evaluate on data
                try:
                    syms = sorted(best.free_symbols, key=str)
                    fn = sp.lambdify(syms, best, "numpy")
                    cols = [np.asarray(x_data[str(s)]) for s in syms]
                    pred = np.asarray(fn(*cols), dtype=float).ravel()
                    key = str(best)
                    if key not in seen and len(pred) == n and np.all(np.isfinite(pred)):
                        seen.add(key)
                        candidates.append((best, pred))
                except _PROBE_ERRORS as exc:
                    logger.debug("Candidate prediction dropped from ensemble: %s", exc)
            if len(candidates) >= n_candidates:
                break

        if not candidates:
            return sp.Integer(0), float('inf'), [], []

        # Stack predictions: (n_samples, n_candidates)
        X = np.column_stack([p for _, p in candidates])
        # Lasso for sparse selection
        model = LassoCV(alphas=[lasso_alpha * (10 ** i) for i in range(-3, 2)],
                        cv=3, max_iter=5000, random_state=seed).fit(X, y)
        coef = model.coef_
        selected_idx = np.where(np.abs(coef) > 1e-6)[0]

        if len(selected_idx) == 0:
            return sp.Integer(0), float('inf'), [], candidates

        # Refit OLS on selected terms for unbiased coefficients
        X_sel = X[:, selected_idx]
        ols_coef, *_ = np.linalg.lstsq(np.column_stack([X_sel, np.ones(n)]), y, rcond=None)
        combined = sum(c * candidates[i][0] for c, i in zip(ols_coef[:-1], selected_idx, strict=False)) + ols_coef[-1]
        final_pred = X_sel @ ols_coef[:-1] + ols_coef[-1]
        mse = float(np.mean((final_pred - y) ** 2))
        selected_terms = [(candidates[i][0], float(ols_coef[j])) for j, i in enumerate(selected_idx)]

        return combined, mse, selected_terms, candidates


# =============================================================================
# Pure helper functions for separability detection and recursive solving
# =============================================================================


def detect_multiplicative_split(x_data: dict[str, np.ndarray],
                                y_data: np.ndarray,
                                tol: float = 1e-6) -> dict:
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
            return {"kind": "multiplicative",
                    "exponents": [round(float(v), 4) for v in coef[:-1]],
                    "variables": names,
                    "r_squared": round(float(ss), 6)}
        return {}
    except (TypeError, ValueError, ArithmeticError):
        return {}


def recursive_solve(x_data: dict[str, np.ndarray],
                    y_data: np.ndarray,
                    tol: float = 1e-6) -> dict:
    """If multiplicatively separable, read power laws off the log-linear fit
    and return the recombined expression. No evolution involved."""
    det = detect_multiplicative_split(x_data, y_data, tol)
    if det.get("kind") != "multiplicative":
        return {"split": False, "detail": det}
    return {"split": True,
            "expression": recombine_power_law(det["variables"], det["exponents"]),
            "r_squared": det["r_squared"],
            "note": "constant factor undetermined by log-fit; fit separately"}

# Alias for backwards compatibility
EvolutionaryDiscoveryEngine = MultiVariableSRE
# Mechanistic priors for APEDP discovery system

MECHANISTIC_PRIORS = {
    "protein_folding": [
        "time*charge_rate*dod",           
        "time*discharge_rate*dod",        
        "time*charge_rate*dod**2",        
        "exp(-charge_rate/discharge_rate)",
        "dod*exp(-charge_rate/dod)",      
    ],
    "battery_degradation": [
        "time*charge_rate*dod",           
        "time*discharge_rate*dod**2",     
        "exp(-dod/charge_rate)",          
        "time**0.5 * charge_rate",        
        "charge_rate**2 * dod",           
    ],
    "turbulence_closure": [
        "strain_rate**2",                  
        "vorticity**2",                    
        "strain_rate*vorticity",           
        "strain_rate**3",                  
        "strain_rate/dissipation",         
    ],
    "chemical_kinetics": [
        "conc_A*conc_B",                   
        "conc_A**2",                       
        "exp(-Ea/(R*T)) * conc_A",         
        "conc_A/(1 + K*conc_B)",           
    ],
    "epidemiology": [
        "S*I",                             
        "S*I/(S+I+R)",                     
        "beta*S*I - gamma*I",              
    ],
    "population_dynamics": [
        "r*N*(1-N/K)",                     
        "r*N*(1-N/K) - h*N",               
        "alpha*N1*N2 - beta*N1",           
    ],
}


def get_mechanistic_priors(domain: str) -> list:
    """Get physics-informed candidate expressions for a scientific domain."""
    return MECHANISTIC_PRIORS.get(domain, [])


def suggest_mechanistic_priors(variable_names: list[str], target_name: str) -> list:
    """Infer likely mechanistic priors from variable names."""
    names_lower = [v.lower() for v in variable_names]
    target_lower = target_name.lower()
    
    # Check for battery degradation
    if any('charge' in n or 'discharge' in n or 'dod' in n or 'capacity' in n for n in names_lower):
        return MECHANISTIC_PRIORS["battery_degradation"]
    
    # Check for protein folding (using same variables as battery but different target)
    if (any('charge' in n or 'discharge' in n or 'dod' in n for n in names_lower)
            and 'time' in names_lower
            and ('rate' in target_lower or 'capacity' in target_lower)):
        return MECHANISTIC_PRIORS["protein_folding"]
    
    # Check for turbulence
    if any('strain' in n or 'vorticity' in n or 'dissipation' in n for n in names_lower):
        return MECHANISTIC_PRIORS["turbulence_closure"]
    
    # Check for chemical kinetics
    if any('conc' in n or 'concentration' in n or 'rate' in target_lower for n in names_lower):
        return MECHANISTIC_PRIORS["chemical_kinetics"]
    
    # Check for epidemiology
    if (any('S' in n or 'I' in n or 'R' in n for n in names_lower)
            and ('infect' in target_lower or 'case' in target_lower)):
        return MECHANISTIC_PRIORS["epidemiology"]
    
    # Check for population dynamics
    if any('pop' in n or 'N' in n for n in names_lower):
        return MECHANISTIC_PRIORS["population_dynamics"]
    
    return []


# Update detect_composition_family to use mechanistic priors
def detect_composition_family_with_priors(x_data: dict[str, np.ndarray],
                                           y_data: np.ndarray,
                                           r2_threshold: float = 0.99,
                                           domain: str | None = None) -> dict:
    """Enhanced composition family detection with mechanistic priors."""
    from apedp.discovery import detect_composition_family
    result = detect_composition_family(x_data, y_data, r2_threshold)
    
    if not result and domain:
        priors = get_mechanistic_priors(domain)
        if priors:
            result = {"family": "mechanistic", "priors": priors, "note": f"Mechanistic priors from {domain}"}
    
    return result