"""Feature selection, forward stepwise search, and data adequacy gates for APEDP.

Extracts BIC-forward selection, SINDy-style sparse monomial selection with collinearity
and stability filtering, and preliminary identifiability audits into a focused component.
"""
from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np

from apedp.discovery_types import (
    BicSelectionResult,
    DataAdequacyResult,
    SparseSelectionResult,
)
from apedp.numerical_policy import DEFAULT_POLICY, DiscoveryNumericalPolicy

logger = logging.getLogger("apedp.feature_selection")

_PROBE_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    ArithmeticError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
)


def compute_structural_complexity(
    term: str,
    ln_n: float,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> float:
    """Computes the Minimum Description Length (MDL) structural penalty for a library term.

    Penalizes transcendental functions and non-integer powers relative to
    canonical dimensional monomials, preventing collinear overfitting on finite samples.
    """
    if not getattr(policy, "MDL_PENALTY_ENABLED", True):
        return 0.0
    weight = float(getattr(policy, "MDL_WEIGHT", 1.0))
    if weight <= 0.0:
        return 0.0

    if term.startswith(("exp(", "log(")):
        cost = getattr(policy, "MDL_COST_TRANSCENDENTAL", 2.5)
    elif "**(" in term:
        cost = getattr(policy, "MDL_COST_FRACTIONAL_POWER", 1.5)
    elif "**" in term:
        try:
            parts = term.split("**")
            p = int(parts[1])
            cost = getattr(policy, "MDL_COST_CANONICAL", 0.0) if p == 1 else getattr(policy, "MDL_COST_INTEGER_POWER", 0.5)
        except (ValueError, IndexError):
            cost = getattr(policy, "MDL_COST_INTEGER_POWER", 0.5)
    elif "*" in term:
        cost = getattr(policy, "MDL_COST_CROSS_INTERACTION", 1.0)

    else:
        cost = getattr(policy, "MDL_COST_CANONICAL", 0.0)

    return float(cost * ln_n * weight)


def boundary_holdout_mse(
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    terms: list[str],
    percentile: float = 12.0,
) -> float:
    """Computes boundary holdout prediction error to penalize divergent collinear artifacts."""
    n = len(y)
    if n < 15 or not terms:
        return 0.0
    first_col = next(iter(cols.values()))
    b_mask = (first_col <= np.percentile(first_col, percentile)) | (
        first_col >= np.percentile(first_col, 100.0 - percentile)
    )
    int_mask = ~b_mask
    if np.sum(int_mask) <= len(terms) + 1 or np.sum(b_mask) == 0:
        return 0.0
    try:
        X = np.column_stack([cols[c] for c in terms] + [np.ones(n)])
        coef, *_ = np.linalg.lstsq(X[int_mask], y[int_mask], rcond=None)
        pred_b = X[b_mask] @ coef
        return float(np.mean((pred_b - y[b_mask]) ** 2))
    except _PROBE_ERRORS:
        return float("inf")



def bic_forward_select(
    columns: dict[str, np.ndarray],
    y_data: np.ndarray,
    max_terms: int | None = None,
    *,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
    holdout_columns: dict[str, np.ndarray] | None = None,
    holdout_y: np.ndarray | None = None,
    holdout_max_ratio: float = 1.0,
) -> BicSelectionResult:
    """Greedy forward subset selection minimizing BIC with Kass-Raftery tie breaking and MDL.

    Two fail-open guards beyond BIC: (1) a scale-normalized multicollinearity
    screen (raw cond confounds units disparity with collinearity; the HP
    Track-C soup measured 2.5e16 raw but only ~2e2 scaled), refusing only
    genuinely singular designs; (2) an optional holdout refusal — when holdout
    data are supplied, a winner generalizing worse than intercept-only is
    refused rather than returned (small-n soup fits train by cancellation).
    Both default to no behavior change when holdout is absent.
    """
    try:
        names = sorted(columns)
        y = np.asarray(y_data, float).ravel()
        n = len(y)
        if n < policy.MIN_SAMPLE_SIZE_FORWARD_SELECT or not names:
            return BicSelectionResult(note="insufficient data")

        cols = {}
        for k in names:
            v = np.asarray(columns[k], float).ravel()
            if v.shape == (n,) and bool(np.all(np.isfinite(v))):
                cols[k] = v

        if not cols:
            return BicSelectionResult(note="no finite features")

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

        ssr0 = float(np.sum((y - y.mean()) ** 2)) + policy.VARIANCE_EPSILON
        ln_n = float(np.log(n))
        best_bic = float(n * np.log(ssr0 / n) + 1 * ln_n)
        best_overall_bic = best_bic
        best_overall_score = best_bic
        best_overall_terms: list[str] = []
        best_overall_b_mse = float("inf")

        beam_width = max(1, int(getattr(policy, "FORWARD_SELECT_BEAM_WIDTH", 6)))
        beam: list[list[str]] = [[]]

        for _depth in range(1, max_terms + 1):
            candidates: list[tuple[float, float, list[str], np.ndarray]] = []
            for current in beam:
                for k in names:
                    if k in current:
                        continue
                    trial = [*current, k]
                    X = np.column_stack([cols[c] for c in trial] + [np.ones(n)])
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    ssr = float(np.sum((y - X @ coef) ** 2)) + policy.VARIANCE_EPSILON
                    mdl_pen = sum(compute_structural_complexity(c, ln_n, policy) for c in trial)
                    pure_bic = float(n * np.log(ssr / n) + (len(trial) + 1) * ln_n)
                    score = float(pure_bic + mdl_pen)
                    candidates.append((score, pure_bic, trial, coef))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0])
            seen: set[tuple[str, ...]] = set()
            new_beam: list[list[str]] = []
            for _s, _bic, cand, _coef in candidates:
                canon = tuple(sorted(cand))
                if canon not in seen:
                    seen.add(canon)
                    new_beam.append(cand)
                if len(new_beam) >= beam_width:
                    break

            if not new_beam:
                break
            beam = new_beam

            top_score = candidates[0][0]
            # Kass-Raftery zone: contenders within DELTA_BIC points of best are evidentially tied
            tied = [item for item in candidates if item[0] - top_score < policy.KASS_RAFTERY_DELTA_BIC]
            percentile = getattr(policy, "BOUNDARY_HOLDOUT_PERCENTILE", 12.0)
            use_guard = getattr(policy, "BOUNDARY_HOLDOUT_GUARD", True)
            best_cand_tuple = min(
                tied,
                key=lambda item: (
                    boundary_holdout_mse(cols, y, item[2], percentile) if use_guard else 0.0,
                    sum(_simplicity(t)[0] for t in item[2]),
                    item[2],
                ),
            )
            cand_score, cand_bic, cand_terms, _cand_coef = best_cand_tuple
            cand_b_mse = boundary_holdout_mse(cols, y, cand_terms, percentile) if use_guard else 0.0


            # Stopping gate: adding terms must improve MDL score by at least KASS_RAFTERY_DELTA_BIC
            degradation_limit = float(getattr(policy, "BOUNDARY_HOLDOUT_DEGRADATION_LIMIT", 1.50))
            delta_improvement = best_overall_score - cand_score
            if delta_improvement >= policy.KASS_RAFTERY_DELTA_BIC:
                if use_guard and best_overall_terms:
                    # For marginal improvements (< 10 points), enforce degradation limit
                    # For decisive improvements (>= 10 points), only block on extreme divergence (> 3.0x)
                    limit = degradation_limit if delta_improvement < 10.0 else max(3.0, degradation_limit * 2.0)
                    if cand_b_mse > best_overall_b_mse * limit:
                        break
                best_overall_score = cand_score
                best_overall_bic = cand_bic
                best_overall_terms = cand_terms
                best_overall_b_mse = cand_b_mse
            else:
                break


        if not best_overall_terms:
            return BicSelectionResult(
                bic=round(best_bic, 4),
                note="no feature improves on intercept-only BIC",
            )

        X = np.column_stack([cols[c] for c in best_overall_terms] + [np.ones(n)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

        # Multicollinearity screen: raw cond confounds units disparity with
        # collinearity (HP soup: 2.5e16 raw but ~2e2 scale-normalized), so the
        # screen operates on scale-normalized columns and fires only on genuine
        # singularity. Refusal of generalizing-poor models is handled by the
        # holdout check below, not here.
        def _design_cond(terms: list[str]) -> float:
            try:
                Mens = []
                for c in terms:
                    col = np.asarray(cols[c], float)
                    Mens.append(col / max(float(np.max(np.abs(col))), 1e-300))
                Xm = np.column_stack([*Mens, np.ones(n)])
                cnd = float(np.linalg.cond(Xm))
                return cnd if np.isfinite(cnd) else float("inf")
            except (np.linalg.LinAlgError, ArithmeticError, ValueError):
                return float("inf")

        cond_limit = float(getattr(policy, "FORWARD_SELECT_MAX_COND", 1e10))
        if _design_cond(best_overall_terms) > cond_limit:
            return BicSelectionResult(
                bic=round(best_bic, 4),
                note="refused: selected model multicollinear "
                     f"(scaled cond>{cond_limit:.0e})",
            )

        # Holdout refusal (fail-open guard): when holdout data are supplied, a
        # winner generalizing worse than intercept-only is refused rather than
        # returned — small-n soup fits train by cancellation (HP Track-C soup:
        # holdout RMSE 5947 MPa vs 389 MPa intercept-only). Absent by default:
        # no behavior change for existing callers.
        if holdout_columns is not None and holdout_y is not None:
            try:
                yh = np.asarray(holdout_y, float).ravel()
                Xh = np.column_stack(
                    [np.asarray(holdout_columns[c], float).ravel() for c in best_overall_terms]
                    + [np.ones(yh.shape)]
                )
                if Xh.shape[0] == yh.shape[0] and Xh.shape[0] > 0:
                    pred_h = Xh @ np.asarray(coef)
                    mse_h = float(np.mean((yh - pred_h) ** 2))
                    mse_null = float(np.mean((yh - float(np.mean(y))) ** 2))
                    if mse_h > mse_null * float(holdout_max_ratio):
                        return BicSelectionResult(
                            bic=round(best_bic, 4),
                            note="refused: winner worse than intercept-only on holdout "
                                 f"(mse={mse_h:.3e} > {float(holdout_max_ratio):.2f}x null {mse_null:.3e})",
                        )
            except (ArithmeticError, TypeError, ValueError):
                logger.debug("Holdout refusal check skipped", exc_info=True)

        return BicSelectionResult(
            terms=best_overall_terms,
            coef=[round(float(v), 6) for v in coef[:-1]],
            intercept=round(float(coef[-1]), 6),
            bic=round(float(best_overall_bic), 4),
            mdl_score=round(float(best_overall_score), 4),
        )

    except _PROBE_ERRORS as e:
        logger.warning("bic_forward_select encountered recoverable error: %s", e, exc_info=True)
        return BicSelectionResult(note=f"bic-forward failed: {e}")


def check_data_adequacy(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    min_range_ratio: float = 4.0,
    *,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> DataAdequacyResult:
    """Data-adequacy gate for discovery runs."""
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        n = len(y)
        ratios: dict[str, float] = {}

        for k in names:
            v = np.asarray(x_data[k], float).ravel()
            if v.shape != (n,) or not bool(np.all(np.isfinite(v))):
                return DataAdequacyResult(
                    verdict="INSUFFICIENT",
                    n=n,
                    n_vars=len(names),
                    reasons=[f"input '{k}' is not finite length-{n}"],
                )
            if bool(np.all(v > 0)):
                ratios[k] = round(float(v.max() / (v.min() + policy.VARIANCE_EPSILON)), 4)

        reasons = []
        if n < policy.MIN_SAMPLE_SIZE_FORWARD_SELECT or float(np.sum((y - y.mean()) ** 2)) <= policy.VARIANCE_EPSILON:
            reasons.append("n < 5 or no y variance")
            return DataAdequacyResult(
                verdict="INSUFFICIENT",
                n=n,
                n_vars=len(names),
                range_ratios=ratios,
                reasons=reasons,
            )

        if n < policy.MIN_OBS_PER_PARAMETER * (len(names) + 1):
            reasons.append(
                f"n={n} < {policy.MIN_OBS_PER_PARAMETER}*(p+1)={policy.MIN_OBS_PER_PARAMETER * (len(names) + 1)} "
                f"for p={len(names)} inputs"
            )

        if ratios and all(r < min_range_ratio for r in ratios.values()):
            reasons.append(
                f"all positive inputs span max/min < {min_range_ratio} (power-law unidentifiability)"
            )

        return DataAdequacyResult(
            verdict="THIN" if reasons else "OK",
            n=n,
            n_vars=len(names),
            range_ratios=ratios,
            reasons=reasons,
        )
    except _PROBE_ERRORS as e:
        logger.warning("check_data_adequacy encountered recoverable error: %s", e, exc_info=True)
        return DataAdequacyResult(
            verdict="INSUFFICIENT",
            n=len(y_data) if hasattr(y_data, "__len__") else 0,
            n_vars=len(x_data),
            reasons=[f"adequacy check failed: {e}"],
        )


def sparse_select_terms(
    x_data: dict[str, np.ndarray],
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
    strict: bool = False,
    policy: DiscoveryNumericalPolicy = DEFAULT_POLICY,
) -> SparseSelectionResult:
    """SINDy-style sparse term selection over monomial library with collinearity/stability filtering."""
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        adequacy = check_data_adequacy(x_data, y_data, policy=policy)

        if strict and adequacy.verdict != "OK":
            raise ValueError(
                "data gate: "
                + adequacy.verdict
                + ": "
                + "; ".join(adequacy.reasons or ["unspecified"])
                + " (pass strict=False to override with an audit trail)"
            )

        if len(y) < policy.MIN_SAMPLE_SIZE_FORWARD_SELECT or float(np.sum((y - y.mean()) ** 2)) <= policy.VARIANCE_EPSILON:
            return SparseSelectionResult(note="insufficient variance", adequacy=adequacy)

        cols = {k: np.asarray(x_data[k], float).ravel() for k in names}
        feats: dict[str, np.ndarray] = {"1": np.ones_like(y)}

        for k in names:
            v = cols[k]
            for p in range(1, max_power + 1):
                # Integer powers use v**p directly: sign-preserving
                feats[f"{k}**{p}"] = v ** p
            if bool(np.all(v > 0)):
                for num, den in ((1, 2), (3, 2), (1, 3), (2, 3)):
                    feats[f"{k}**({num}/{den})"] = v ** (num / den)
            if include_reciprocals and bool(np.all(v != 0)):
                for q in range(1, max(1, int(max_reciprocal_power)) + 1):
                    feats[f"1/{k}" if q == 1 else f"1/{k}**{q}"] = 1.0 / (v ** q)
            if bool(np.all(v > 0)):
                v_mean = float(np.mean(v))
                v_std = float(np.std(v)) + policy.VARIANCE_EPSILON
                v_scaled = (v - v_mean) / v_std
                feats[f"exp({k})"] = np.exp(np.clip(v_scaled, -20, 20))
                feats[f"log({k})"] = np.log(v)

        for a, b in itertools.combinations(names, 2):
            feats[f"{a}*{b}"] = cols[a] * cols[b]

        keys = [k for k in feats if float(np.std(feats[k])) > policy.VARIANCE_EPSILON]
        if not keys:
            return SparseSelectionResult(note="no varying features", adequacy=adequacy)

        # Greedy collinearity filter: rank by |corr| with y, keep feature only if below threshold
        dropped: list[dict[str, Any]] = []
        if corr_threshold is not None:
            if not np.isfinite(corr_threshold) or not 0.0 < corr_threshold <= 1.0:
                return SparseSelectionResult(note="corr_threshold must be in (0, 1]", adequacy=adequacy)
            y_centered = y - y.mean()
            y_norm = float(np.sqrt(np.sum(y_centered ** 2))) + policy.VARIANCE_EPSILON
            scored = []
            for k in keys:
                v = feats[k] - feats[k].mean()
                denom = float(np.sqrt(np.sum(v ** 2))) * y_norm + policy.VARIANCE_EPSILON
                scored.append((abs(float(np.sum(v * y_centered) / denom)), k))
            scored.sort(key=lambda item: (-item[0], item[1]))

            kept: list[str] = []
            kept_cols: list[np.ndarray] = []
            for _, k in scored:
                v = feats[k]
                blocker = None
                for kk, col in zip(kept, kept_cols, strict=False):
                    w = col - col.mean()
                    denom = (
                        float(np.sqrt(np.sum((v - v.mean()) ** 2)))
                        * float(np.sqrt(np.sum(w ** 2)))
                        + policy.VARIANCE_EPSILON
                    )
                    c = abs(float(np.sum((v - v.mean()) * w) / denom))
                    if c >= corr_threshold:
                        blocker = (kk, round(c, 6))
                        break
                if blocker is None:
                    kept.append(k)
                    kept_cols.append(v)
                else:
                    dropped.append({"dropped": k, "kept": blocker[0], "corr": blocker[1]})
            keys = kept
            if not keys:
                return SparseSelectionResult(
                    note="all features filtered as collinear",
                    adequacy=adequacy,
                    dropped_collinear=dropped,
                    n_dropped=len(dropped),
                    n_library_raw=len(scored),
                )

        X = np.column_stack([feats[k] for k in keys])
        scales = np.std(X, axis=0) + policy.VARIANCE_EPSILON
        Xn = X / scales

        from sklearn.linear_model import Lasso, LassoCV

        alpha_for_stability: float | None = None
        selected_terms: list[str] = []
        fwd_intercept = 0.0
        fwd_bic = float("inf")
        note: str | None = None

        if selector == "bic_forward":
            fwd = bic_forward_select({k: feats[k] for k in keys}, y, policy=policy)
            selected_terms = fwd.terms
            fwd_intercept = fwd.intercept
            fwd_bic = fwd.bic
            note = fwd.note
        elif selector == "lasso":
            model = LassoCV(max_iter=5000).fit(Xn, y)
            selected_terms = [keys[i] for i, c in enumerate(model.coef_) if abs(c) > 1e-9]
            fwd_intercept = round(float(model.intercept_), 6)
            alpha_for_stability = float(model.alpha_)
        else:
            return SparseSelectionResult(note=f"unknown selector '{selector}'", adequacy=adequacy)

        stability_rate = 1.0
        if stability_reps is not None:
            if not isinstance(stability_reps, int) or stability_reps < 2 or not 0.0 < stability_threshold <= 1.0:
                return SparseSelectionResult(
                    note="stability_reps must be an int >= 2 and stability_threshold in (0, 1]",
                    adequacy=adequacy,
                )
            rng = np.random.default_rng(seed)
            n_samples = Xn.shape[0]
            freq = np.zeros(len(keys))
            key_to_idx = {k: i for i, k in enumerate(keys)}

            if selector == "bic_forward":
                for _ in range(stability_reps):
                    idx = rng.integers(0, n_samples, n_samples)
                    boot_cols = {k: feats[k][idx] for k in keys}
                    boot_y = y[idx]
                    boot_fwd = bic_forward_select(boot_cols, boot_y, policy=policy)
                    for t in boot_fwd.terms:
                        if t in key_to_idx:
                            freq[key_to_idx[t]] += 1.0
            else:
                if alpha_for_stability is None:
                    alpha_for_stability = float(LassoCV(max_iter=5000).fit(Xn, y).alpha_)
                for _ in range(stability_reps):
                    idx = rng.integers(0, n_samples, n_samples)
                    boot = Lasso(alpha=alpha_for_stability, max_iter=5000).fit(Xn[idx], y[idx])
                    freq += (np.abs(boot.coef_) > 1e-9).astype(float)

            freq /= stability_reps
            stable = [keys[i] for i in range(len(keys)) if freq[i] >= stability_threshold]
            stability_rate = float(np.mean(freq >= stability_threshold)) if len(freq) > 0 else 1.0
            selected_terms = stable

        ambiguity_groups: list[list[str]] = []
        if ambiguity_corr is not None:
            if not np.isfinite(ambiguity_corr) or not 0.0 < ambiguity_corr < 1.0:
                return SparseSelectionResult(note="ambiguity_corr must be in (0, 1)", adequacy=adequacy)
            centered = [feats[k] - feats[k].mean() for k in keys]
            norms = [float(np.sqrt(np.sum(c ** 2))) + policy.VARIANCE_EPSILON for c in centered]
            parent = list(range(len(keys)))

            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    c = abs(float(np.sum(centered[i] * centered[j]) / (norms[i] * norms[j])))
                    if c >= ambiguity_corr:
                        union(i, j)
            groups: dict[int, list[str]] = {}
            for i, k in enumerate(keys):
                groups.setdefault(find(i), []).append(k)
            ambiguity_groups = sorted(
                (sorted(members) for members in groups.values() if len(members) > 1),
                key=lambda members: (-len(members), members),
            )

        return SparseSelectionResult(
            terms=selected_terms,
            intercept=fwd_intercept,
            bic=fwd_bic,
            selector=selector,
            n_library=len(keys),
            n_dropped=len(dropped),
            dropped=[d["dropped"] for d in dropped],
            dropped_collinear=dropped,
            ambiguity_groups=ambiguity_groups,
            ambiguity_corr=ambiguity_corr,
            stability_rate=stability_rate,
            adequacy=adequacy,
            alpha=alpha_for_stability,
            note=note,
        )
    except _PROBE_ERRORS as e:
        logger.warning("sparse_select_terms encountered operational error: %s", e, exc_info=True)
        return SparseSelectionResult(note=f"error: {e}")


def beam_search_expressions(
    variables: list[str],
    depth: int = 2,
    width: int = 8,
    seed: int | None = 42,
) -> list[str]:
    """Breadth-capped construction search over basic algebraic operations."""
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


def control_variable_probe(
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """CVGP-style probe: drop-one linear R^2 gain per variable."""
    try:
        names = list(x_data.keys())
        y = np.asarray(y_data, float).ravel()
        if len(y) < 10:
            return {"relevant": [], "note": "insufficient data"}
        if float(np.sum((y - y.mean()) ** 2)) <= 1e-300:
            return {"relevant": [], "note": "constant target"}
        cols = {k: np.asarray(x_data[k], float).ravel() for k in names}

        def _r2(use: list[str]) -> float:
            if not use:
                return 0.0
            A = np.column_stack([cols[k] for k in use] + [np.ones_like(y)])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            denom = float(np.sum((y - y.mean()) ** 2)) + 1e-300
            return 1.0 - float(np.sum((y - pred) ** 2)) / denom

        full = _r2(names)
        gains = {}
        for k in names:
            without = [c for c in names if c != k]
            gains[k] = round(max(0.0, float(full - _r2(without))), 6)
        top = max(gains.values()) if gains else 0.0
        relevant = [k for k in names if gains[k] > tol and gains[k] >= 0.01 * top]
        return {
            "full_r2": round(full, 4),
            "drop_one_gains": gains,
            "relevant": relevant,
        }
    except _PROBE_ERRORS as e:
        logger.warning("control_variable_probe encountered recoverable error: %s", e, exc_info=True)
        return {"relevant": [], "note": f"probe failed: {e}"}
