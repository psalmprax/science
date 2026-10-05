"""§83 autonomous finance-grammar adjudication (forensics doc).

Closed loop: frozen manifest -> autonomous L2 generation -> candidate
frontier -> canonicalization/domain/causality (engine gates) -> holdout
evaluation -> incremental-vs-H-P (QLIKE + month-block permutation, fixed
forms) -> representation-equivalence clustering (rho/D) -> L0-L5 promotion.
No grammar tuning after outcome; stability capped (single split).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.linalg as la

from apedp_dyn.epistemic.active_disambiguation import finance_manifest as FM
from apedp_dyn.sandbox.grammar_space import GateEvidence, promote

EQUIV_RHO = 0.98
NEAR_RHO_LO = 0.90
N_PERM = 200
PERM_SEED = 8383


@dataclass
class CandidateRecord:
    expr: str
    holdout_rmse: float
    d_qlike_vs_hp: float = float("nan")  # + means candidate better (QLIKE diff HP - cand)
    perm_p: float = float("nan")
    cluster: str = ""
    verdict: str = ""


@dataclass
class AdjudicationRecord:
    manifest_hash: str
    candidates: list[CandidateRecord] = field(default_factory=list)
    rho_matrix: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def representation_clusters(preds: dict[str, np.ndarray]) -> tuple[dict[str, str], list[list[float]], list[str]]:
    """Pairwise rho/D clustering: equivalent (>=0.98), near (0.90-0.98), distinct."""
    labels = sorted(preds)
    n = len(labels)
    rho = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = preds[labels[i]], preds[labels[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 else 0.0
            rho[i][j] = rho[j][i] = r
    assign: dict[str, str] = {}
    for i, lab in enumerate(labels):
        best = max((rho[i][j], labels[j]) for j in range(n) if j != i)[0] if n > 1 else 0.0
        assign[lab] = ("representation-equivalent" if best >= EQUIV_RHO
                       else ("near-equivalent" if best >= NEAR_RHO_LO else "distinct"))
    return assign, rho, labels


def _qvec(y: np.ndarray, p: np.ndarray, floor: float) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=float), 1e-8)
    p = np.maximum(np.asarray(p, dtype=float), floor)
    r = y / p
    return r - np.log(r) - 1.0


def incremental_vs_hp(y_ho: np.ndarray, pred_cand: np.ndarray, pred_hp: np.ndarray,
                      month_id: np.ndarray, floor: float) -> tuple[float, float]:
    """QLIKE diff (HP - candidate; + favors candidate) + month-block perm p."""
    ok = np.isfinite(y_ho) & np.isfinite(pred_cand) & np.isfinite(pred_hp)
    y, pc, ph = y_ho[ok], pred_cand[ok], pred_hp[ok]
    m = month_id[ok] if month_id is not None else np.zeros(len(y), dtype=int)
    obs = float(np.mean(_qvec(y, ph, floor) - _qvec(y, pc, floor)))
    rng = np.random.default_rng(PERM_SEED)
    months = np.unique(m)
    n_ge = 0
    for _ in range(N_PERM):
        perm = rng.permutation(months)
        mp = dict(zip(months, perm, strict=True))
        yp = y.copy()
        # Shuffle whole-month target blocks: rotate values across months.
        src = {mm: y[m == mm] for mm in months}
        parts = []
        for mm in months:
            take = src[mp[mm]][: int((m == mm).sum())]
            parts.append(take)
        yp = np.concatenate(parts)
        d = float(np.mean(_qvec(yp, ph, floor) - _qvec(yp, pc, floor)))
        if d >= obs:
            n_ge += 1
    return obs, (n_ge + 1) / (N_PERM + 1)


def fit_hp(X_tr: dict[str, np.ndarray], y_tr: np.ndarray) -> np.ndarray:
    s = np.abs(np.asarray(X_tr["r"], dtype=float))
    ok = np.isfinite(s) & np.isfinite(y_tr)
    X = np.column_stack([np.ones(ok.sum()), s[ok]])
    w, *_ = la.lstsq(X, np.asarray(y_tr)[ok], rcond=None)
    return w


def adjudicate_frontier(candidates: list[tuple[str, np.ndarray]],
                        y_ho: np.ndarray, hp_pred: np.ndarray,
                        month_id: np.ndarray, floor: float) -> AdjudicationRecord:
    """L0-L5 on autonomous candidates; stability capped (single split)."""
    FM.verify()
    preds = dict(candidates)
    assign, rho, labels = representation_clusters(preds)
    rec = AdjudicationRecord(FM.MANIFEST_HASH)
    rec.rho_matrix, rec.labels = rho, labels
    for expr, pred in candidates:
        d, p = incremental_vs_hp(y_ho, pred, hp_pred, month_id, floor)
        ev = GateEvidence(grammar_valid=True, causal=True,
                          predictive=d > 0, incremental=(d > 0 and p < 0.05),
                          stable=False, null_robust=False, interpretable=False)
        v = promote(f"auto:{expr[:40]}", ev)
        rec.candidates.append(CandidateRecord(expr, float("nan"), d, p, assign[expr],
                                              f"L{v.level} {v.verdict[:60]}"))
    return rec
