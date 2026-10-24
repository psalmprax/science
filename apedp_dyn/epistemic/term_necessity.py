"""Term-necessity / near-tie guard for APEDP adjudication.

Standalone post-hoc filter (E3b contract, forensics doc SS66-67). It does NOT
modify the shared adjudicator: it consumes fitted term designs + holdout data
and reports, per additive term, whether removal materially degrades fit.

Contract: an expanded grammar must not by itself raise epistemic status.
A UNIQUE claim additionally requires every term RETAIN + a singleton
near-tie equivalence class (see `necessity_contract_check`).

Thresholds are provisional, calibrated on the E3b suite:
  REMOVE      rel delta < 1e-3   (E3b-B decorative: 3.6e-06)
  BORDERLINE  1e-3 .. 5e-2       (E3b-C tiny-real: 4.6e-02)
  RETAIN      rel delta > 5e-2   (E3b-A necessary: ~14x)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # deferred at runtime (see _split_additive): avoids a
    from apedp_dyn.discovery_l2.ast_generator import ASTNode  # circular import

REMOVE_REL = 1e-3
RETAIN_REL = 5e-2


class NecessityVerdict(str, Enum):
    RETAIN = "retain"
    REMOVE = "remove"
    BORDERLINE = "borderline"
    EQUIV = "equiv"  # member of a collinear/equivalent set; keep set, not term


@dataclass(frozen=True)
class TermAblation:
    term: str
    delta_rmse: float  # holdout RMSE(without term) - RMSE(full); ~0 => decorative
    rel_delta: float  # delta_rmse / RMSE(full)
    coef_full: float
    verdict: NecessityVerdict


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def ols_predict(
    X_terms: Mapping[str, np.ndarray],
    y: np.ndarray,
    drop: Sequence[str] = (),
) -> tuple[np.ndarray, dict[str, float]]:
    """OLS fit on train rows of y (NaN marks holdout), predict all rows.

    Returns (predictions, {term: coef}).
    """
    y = np.asarray(y, dtype=float)
    train = ~np.isnan(y)
    keep = [t for t in X_terms if t not in set(drop)]
    X = np.column_stack([np.ones(int(train.sum()))] + [np.asarray(X_terms[t])[train] for t in keep])
    w, *_ = np.linalg.lstsq(X, y[train], rcond=None)
    coefs = {"(intercept)": float(w[0])}
    coefs.update({t: float(c) for t, c in zip(keep, w[1:], strict=True)})
    Xa = np.column_stack([np.ones(len(y))] + [np.asarray(X_terms[t]) for t in keep])
    return Xa @ w, coefs


def ablate_terms(
    X_terms: Mapping[str, np.ndarray],
    y: np.ndarray,
    holdout_mask: np.ndarray,
    remove_rel: float = REMOVE_REL,
    retain_rel: float = RETAIN_REL,
) -> list[TermAblation]:
    """Drop-one-term OLS ablation; holdout-scored. NaN in y also marks holdout."""
    y = np.asarray(y, dtype=float)
    holdout_mask = np.asarray(holdout_mask, bool)
    yh = y[holdout_mask]
    full_pred, full_coefs = ols_predict(X_terms, np.where(holdout_mask, np.nan, y))
    base = _rmse(yh, full_pred[holdout_mask])
    out: list[TermAblation] = []
    for term in X_terms:
        pred, _ = ols_predict(X_terms, np.where(holdout_mask, np.nan, y), drop=[term])
        delta = _rmse(yh, pred[holdout_mask]) - base
        rel = delta / base if base > 0 else 0.0
        verdict = (
            NecessityVerdict.REMOVE
            if rel < remove_rel
            else (NecessityVerdict.RETAIN if rel > retain_rel else NecessityVerdict.BORDERLINE)
        )
        out.append(
            TermAblation(
                term=term,
                delta_rmse=delta,
                rel_delta=rel,
                coef_full=full_coefs.get(term, float("nan")),
                verdict=verdict,
            )
        )
    return out


def near_tie_class(rmses: Sequence[float], tol: float = 1e-9) -> list[list[int]]:
    """Group candidate indices whose RMSE matches the class best within tol."""
    rmses = [float(r) for r in rmses]
    order = sorted(range(len(rmses)), key=lambda i: rmses[i])
    classes: list[list[int]] = []
    for i in order:
        placed = False
        for cls in classes:
            if abs(rmses[i] - rmses[cls[0]]) <= tol:
                cls.append(i)
                placed = True
                break
        if not placed:
            classes.append([i])
    return classes


@dataclass(frozen=True)
class ContractCheck:
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def necessity_contract_check(
    ablations: Sequence[TermAblation],
    top_class_size: int,
) -> ContractCheck:
    """E3b contract for a UNIQUE claim: every term RETAIN + singleton top class."""
    reasons: list[str] = []
    for a in ablations:
        if a.verdict is not NecessityVerdict.RETAIN:
            reasons.append(f"term {a.term}: {a.verdict.value} (rel-delta {a.rel_delta:.2e})")
    if top_class_size != 1:
        reasons.append(f"top near-tie class size {top_class_size} != 1")
    return ContractCheck(passed=not reasons, reasons=tuple(reasons))


def ood_necessity_check(
    rmse_without_ood: float,
    rmse_with_ood: float,
    ratio_threshold: float = 2.0,
) -> NecessityVerdict:
    """OOD leg of evidence-relative necessity (E3b-E pattern).

    RETAIN if dropping the term degrades OOD fit by >= ratio_threshold;
    otherwise BORDERLINE (OOD evidence alone never demotes to REMOVE).
    """
    if rmse_with_ood <= 0:
        return NecessityVerdict.BORDERLINE
    if rmse_without_ood / rmse_with_ood >= ratio_threshold:
        return NecessityVerdict.RETAIN
    return NecessityVerdict.BORDERLINE


def apply_ood_upgrade(
    ablation: TermAblation,
    rmse_without_ood: float,
    rmse_with_ood: float,
    ratio_threshold: float = 2.0,
) -> TermAblation:
    """Upgrade a REMOVE/BORDERLINE term to RETAIN on OOD-critical evidence.

    One-way ratchet: small in-domain effect != unnecessary effect. Never
    downgrades an in-domain RETAIN.
    """
    if ablation.verdict is NecessityVerdict.RETAIN:
        return ablation
    if ood_necessity_check(rmse_without_ood, rmse_with_ood, ratio_threshold) is NecessityVerdict.RETAIN:
        return TermAblation(
            term=ablation.term,
            delta_rmse=ablation.delta_rmse,
            rel_delta=ablation.rel_delta,
            coef_full=ablation.coef_full,
            verdict=NecessityVerdict.RETAIN,
        )
    return ablation


def _split_additive(node: ASTNode, sign: float = 1.0) -> list[tuple[float, ASTNode]]:
    """Top-level additive terms with signs (mirrors pruning._additive_terms).

    Local copy (public-class only) so the guard stays additive: no private
    imports, no changes to the discovery pipeline. BinaryOpNode is imported
    here (deferred) because ast_generator transitively imports l2_engine,
    which imports this module at the Step-5b wire-in.
    """
    from apedp_dyn.discovery_l2.ast_generator import BinaryOpNode

    if isinstance(node, BinaryOpNode) and node.op in ("+", "-"):
        s_right = sign if node.op == "+" else -sign
        return _split_additive(node.left, sign) + _split_additive(node.right, s_right)
    return [(sign, node)]


@dataclass(frozen=True)
class GuardReport:
    """Audit record of one guard_selection() pass at the L2 selection boundary."""

    n_in: int
    n_guarded: int
    top_class_size: int
    dropped_ids: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    ablations: tuple[tuple[str, tuple[TermAblation, ...]], ...] = field(default_factory=tuple)


def _ablate_ast_candidate(
    candidate_id: str,
    node: ASTNode,
    X_tr: Mapping[str, np.ndarray],
    y_tr: np.ndarray,
    X_ho: Mapping[str, np.ndarray],
    y_ho: np.ndarray,
    min_overlap: int = 10,
) -> tuple[TermAblation, ...] | None:
    """Drop-one-term OLS ablation for one AST candidate; None if unablatable."""
    try:
        parts = _split_additive(node)
        cols_tr = [np.asarray(s * sub.evaluate(dict(X_tr)), dtype=float) for s, sub in parts]
        cols_ho = [np.asarray(s * sub.evaluate(dict(X_ho)), dtype=float) for s, sub in parts]
    except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError):
        return None
    y_tr = np.asarray(y_tr, dtype=float)
    y_ho = np.asarray(y_ho, dtype=float)
    m_tr = np.isfinite(y_tr)
    m_ho = np.isfinite(y_ho)
    for c in cols_tr:
        m_tr &= np.isfinite(c)
    for c in cols_ho:
        m_ho &= np.isfinite(c)
    if int(m_tr.sum()) < min_overlap or int(m_ho.sum()) < min_overlap:
        return None
    names = [sub.to_str() for _, sub in parts]
    if len(names) == 1:
        # Single-term expression: ablation is vacuous (nothing to remove
        # while leaving a viable model). Necessity holds by default; the
        # near-tie class still polices uniqueness downstream.
        return (TermAblation(term=names[0], delta_rmse=0.0, rel_delta=float("inf"),
                             coef_full=float("nan"), verdict=NecessityVerdict.RETAIN),)
    out: list[TermAblation] = []
    Xo = {t: c[m_ho] for t, c in zip(names, cols_ho, strict=True)}
    yh = y_ho[m_ho]
    Xt = {t: c[m_tr] for t, c in zip(names, cols_tr, strict=True)}
    yt = y_tr[m_tr]
    M = np.column_stack([Xt[t] for t in names])
    w, *_ = np.linalg.lstsq(M, yt, rcond=None)
    base = _rmse(yh, np.column_stack([Xo[t] for t in names]) @ w)
    coefs = dict(zip(names, [float(c) for c in w], strict=True))
    for j, t in enumerate(names):
        keep = [k for k in range(len(names)) if k != j]
        if not keep:
            delta, rel = 0.0, 0.0
        else:
            Mk = M[:, keep]
            wk, *_ = np.linalg.lstsq(Mk, yt, rcond=None)
            red = _rmse(yh, np.column_stack([Xo[names[k]] for k in keep]) @ wk)
            delta, rel = red - base, (red - base) / base if base > 0 else 0.0
        verdict = (
            NecessityVerdict.REMOVE
            if rel < REMOVE_REL
            else (NecessityVerdict.RETAIN if rel > RETAIN_REL else NecessityVerdict.BORDERLINE)
        )
        out.append(TermAblation(term=t, delta_rmse=delta, rel_delta=rel,
                                coef_full=coefs[t], verdict=verdict))
    return tuple(out)


def guard_selection(
    ranked: Sequence,
    X_tr: Mapping[str, np.ndarray],
    y_tr: np.ndarray,
    X_ho: Mapping[str, np.ndarray],
    y_ho: np.ndarray,
    tie_rel_tol: float = 1e-6,
    tie_abs_floor: float = 1e-12,
) -> tuple[list, GuardReport]:
    """Phase-1 wire-in: necessity/near-tie filter at the L2 selection boundary.

    Purely restrictive layer between ranking and adjudication (rules 1-8):
    may drop debris-carrying candidates, never manufactures evidence, never
    widens a verdict. Empty in -> empty out (adjudicate([]) -> NO_VALID).
    """
    finite = [s for s in ranked if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)]
    finite.sort(key=lambda s: float(s.holdout_rmse))
    if not finite:
        return [], GuardReport(n_in=len(ranked), n_guarded=0, top_class_size=0,
                               reasons=("no finite-holdout candidates; nothing to guard",))
    rmses = [float(s.holdout_rmse) for s in finite]
    tol = max(tie_abs_floor, tie_rel_tol * abs(rmses[0]))
    classes = near_tie_class(rmses, tol=tol)
    top_idx = classes[0]
    ablations: dict[str, tuple[TermAblation, ...]] = {}
    for i in top_idx:
        node = finite[i].candidate.metadata.get("ast")
        if node is None:
            continue
        cid = finite[i].candidate.candidate_id
        abl = _ablate_ast_candidate(cid, node, X_tr, y_tr, X_ho, y_ho)
        if abl is not None:
            ablations[cid] = abl
    reasons: list[str] = [f"top near-tie class size {len(top_idx)} (tol {tol:.2e})"]
    # Cascade: drop top-class members carrying REMOVE terms; BORDERLINE members
    # stay but block clean promotion by keeping rival context (rule 5).
    drop: set[int] = set()
    for i in top_idx:
        cid = finite[i].candidate.candidate_id
        abl = ablations.get(cid)
        if abl is None:
            reasons.append(f"{cid}: unablatable, kept with flag")
            continue
        bad = [a.term for a in abl if a.verdict is NecessityVerdict.REMOVE]
        border = [a.term for a in abl if a.verdict is NecessityVerdict.BORDERLINE]
        if bad:
            drop.add(i)
            reasons.append(f"{cid}: REMOVE term(s) {[t[:40] for t in bad]}, dropped from promotion set")
        elif border:
            reasons.append(f"{cid}: BORDERLINE term(s) {[t[:40] for t in border]}, rival context kept")
    kept = [s for k, s in enumerate(finite) if k not in drop]
    dropped_ids = tuple(finite[k].candidate.candidate_id for k in sorted(drop))
    if not kept:
        reasons.append("guard eliminated all candidates -> INSUFFICIENT_EVIDENCE path (no fallback)")
    return kept, GuardReport(
        n_in=len(ranked),
        n_guarded=len(kept),
        top_class_size=len(top_idx),
        dropped_ids=dropped_ids,
        reasons=tuple(reasons),
        ablations=tuple((k, v) for k, v in ablations.items()),
    )


def ood_necessity(
    fit_without: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
    fit_with: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_ood: np.ndarray,
    y_ood: np.ndarray,
) -> tuple[float, float]:
    """(rmse_without_term, rmse_with_term) on OOD data; E3b-E pattern."""
    return (
        _rmse(y_ood, fit_without(X_train, y_train)(X_ood)),
        _rmse(y_ood, fit_with(X_train, y_train)(X_ood)),
    )
