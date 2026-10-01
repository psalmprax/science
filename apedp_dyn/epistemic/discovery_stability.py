"""Discovery stability: recurrence, entropy, and variance decomposition.

Orthogonal to identification/transfer/promotion. Answers "what kind of
MULTI is this" by separating:
  - structural recurrence P(structure = H) over discovery seeds,
  - structural entropy H = -sum p log p (low = search finds the same thing),
  - parameter dispersion (CV per coefficient),
  - fit variance (Var of train RMSE across seeds),
  - transfer variance split: fit-seed vs scoring-seed components.

A structure can be recurrent yet unstable in parameters; a verdict can
be MULTI from observational equivalence or from search instability --
this module tells those apart with numbers instead of adjectives.
"""
from __future__ import annotations

import numpy as np


def structural_distribution(structures: list[str]) -> dict[str, float]:
    """Empirical P(structure) over discovery seeds."""
    n = len(structures)
    out: dict[str, float] = {}
    for s in structures:
        out[s] = out.get(s, 0.0) + 1.0 / n
    return out


def structural_entropy(structures: list[str]) -> float:
    """Shannon entropy of the discovery distribution (nats)."""
    dist = structural_distribution(structures)
    return float(-sum(p * np.log(p) for p in dist.values() if p > 0))


def coefficient_cv(values: list[float]) -> float | None:
    """Coefficient of variation |std/mean|; None when mean ~ 0."""
    a = np.asarray(values, dtype=float)
    m = float(np.mean(a))
    if abs(m) < 1e-12:
        return None
    return float(np.std(a) / abs(m))


def variance_split(transfer_matrix: np.ndarray) -> dict[str, float]:
    """Decompose transfer RMSE variance into fit-seed vs scoring-seed parts.

    transfer_matrix[i, j] = error of model fit on seed i, scored on seed j.
    Row means vary with fit quality; column means vary with scoring noise.
    """
    m = np.asarray(transfer_matrix, dtype=float)
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)
    total = float(np.var(m))
    return {"total": total,
            "fit_seed": float(np.var(row_means)),
            "scoring_seed": float(np.var(col_means)),
            "fit_share": float(np.var(row_means) / total) if total > 0 else 0.0}


def stability_report(structures: list[str],
                     param_samples: dict[str, list[float]] | None = None,
                     transfer_matrix=None) -> dict:
    """One diagnostic block for a multi-seed discovery experiment."""
    rep = {"n_seeds": len(structures),
           "distribution": structural_distribution(structures),
           "entropy_nats": structural_entropy(structures)}
    if param_samples:
        rep["parameter_cv"] = {k: coefficient_cv(v) for k, v in param_samples.items()}
    if transfer_matrix is not None:
        rep["transfer_variance"] = variance_split(np.asarray(transfer_matrix))
    return rep
