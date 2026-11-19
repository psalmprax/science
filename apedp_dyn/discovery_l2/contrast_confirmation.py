"""Contrast confirmation: decide between rival laws by measuring where they disagree.

Status: experimental

The referee compares rivals on a random holdout. Two laws that differ only by a
small term agree on almost every random row, so their holdout errors differ by
less than the resolution floor and the referee abstains (``MULTI_HYPOTHESIS``);
this is what PREREG-BLIND-L2-2026-11 measured on two-term laws. A random holdout
spends its rows where the rivals say the same thing.

Contrast confirmation spends them where the rivals say different things:

1. **Design from X only.** For the champion and each rival, rank the unlabelled
   pool rows by the squared disagreement of their predictions in noise units and
   take the top ``budget``. No target value is read to choose rows, and the
   design's SHA-256 is fixed before any is.
2. **Power before data.** With Gaussian noise the log-likelihood ratio on those
   rows has mean +-D^2/2 and standard deviation D, where D^2 is the summed
   squared disagreement. Deciding by its sign is wrong with probability
   Phi(-D/2). If D < 2 z(1 - alpha) the pair cannot be told apart at error rate
   alpha with this budget on this domain, and the two are reported as
   **equivalent on the domain** without reading a target. alpha is the canonical
   Diebold-Mariano threshold, Bonferroni-corrected over the rivals.
3. **Measure, eliminate.** Otherwise the targets of exactly those rows are read
   and the loser is eliminated.
4. **Refute the survivor.** The survivor's error on every row read is compared
   (F test, same alpha) with its own development error. The contrast rows are
   where the candidates differ, so a law missing from the candidate set shows up
   there: if the survivor fails, the verdict is **no survivor** -- evidence that
   the true law is outside the hypothesis space, which the holdout referee
   cannot distinguish from "rivals too close".

Verdicts: ``unique`` (the survivor eliminated at least one rival and nothing is
indistinguishable from it), ``equivalent_class`` (the survivor and everything
indistinguishable from it here), ``unchallenged`` (it passed the refutation test
but no rival was there to eliminate, which is weaker), ``no_survivor``.

Every prediction is charged for the uncertainty of its fitted coefficients
(sandwich variance from the development design), because contrast rows sit at
the edges of the domain where that uncertainty is largest.

What is and is not new: choosing measurements where models disagree is classical
(T-optimal design for model discrimination, query by committee), and
``active_intervention.py`` already proposes such a point. What this module adds
is the verdict: power computed before data, certified equivalence in place of a
bare abstention, and refutation of the survivor. Its error rates are a claim
until measured by a preregistered run.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from apedp_referee.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD

UNIQUE, EQUIVALENT_CLASS, NO_SURVIVOR, UNCHALLENGED = "unique", "equivalent_class", "no_survivor", "unchallenged"


@dataclass(frozen=True)
class FittedCandidate:
    """A law linear in its coefficients, fitted on development data.

    ``basis(X)`` returns one column per coefficient. Keeping the development
    design lets the procedure charge each prediction for the uncertainty of the
    fitted coefficients: contrast rows sit at the edges of the domain, where
    that uncertainty is largest, and ignoring it refutes true laws.
    """

    candidate_id: str
    basis: Callable[[np.ndarray], np.ndarray]  # (n, k) inputs -> (n, p) columns
    coef: np.ndarray
    dev_basis: np.ndarray
    dev_y: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.asarray(self.basis(X), dtype=float) @ self.coef
        return out

    @property
    def dev_pred(self) -> np.ndarray:
        out: np.ndarray = self.dev_basis @ self.coef
        return out

    @property
    def dev_resid(self) -> np.ndarray:
        out: np.ndarray = self.dev_y - self.dev_pred
        return out

    @property
    def n_params(self) -> int:
        return len(self.coef)


def fit_candidate(candidate_id: str, basis: Callable[[np.ndarray], np.ndarray],
                  dev_X: np.ndarray, dev_y: np.ndarray, noise: str = "relative") -> FittedCandidate:
    """Least squares weighted for the noise model (reweighted from the fit itself for relative noise).

    Unweighted least squares under relative noise lets the largest targets
    decide a small term's coefficient, which made the true two-term law fit
    worse, in noise units, than the same law without the term.
    """
    A = np.asarray(basis(np.asarray(dev_X, dtype=float)), dtype=float)
    y = np.asarray(dev_y, dtype=float)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(5 if noise == "relative" else 0):
        w = 1.0 / _weights(A @ coef, noise)
        coef = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    return FittedCandidate(candidate_id, basis, coef, A, y)


@dataclass
class ContrastResult:
    verdict: str
    survivor: str | None
    equivalent: list[str]
    eliminated: list[dict[str, Any]]
    rows_read: int
    design_sha256: str
    survivor_fit: dict[str, Any] = field(default_factory=dict)


def _weights(pred: np.ndarray, noise: str) -> np.ndarray:
    """Noise standard deviation up to a scale: 1 (absolute) or |prediction| (relative)."""
    if noise == "absolute":
        ones: np.ndarray = np.ones(np.shape(pred), dtype=float)
        return ones
    if noise == "relative":
        mag: np.ndarray = np.maximum(np.abs(pred), 1e-300)
        return mag
    raise ValueError(f"noise must be 'absolute' or 'relative', not {noise!r}")


def noise_scale(candidate: FittedCandidate, noise: str) -> float:
    """The candidate's development error in units of the noise model."""
    dof = max(1, len(candidate.dev_y) - candidate.n_params)
    return float(np.sqrt(np.sum((candidate.dev_resid / _weights(candidate.dev_pred, noise)) ** 2) / dof))


def coefficient_variance(candidate: FittedCandidate, X: np.ndarray, noise: str) -> np.ndarray:
    """Variance of the prediction at each row of ``X`` from coefficient error, per unit noise scale^2."""
    A = candidate.dev_basis / _weights(candidate.dev_pred, noise)[:, None]
    cov = np.linalg.pinv(A.T @ A)
    B = np.asarray(candidate.basis(X), dtype=float)
    out: np.ndarray = np.maximum(np.einsum("ij,jk,ik->i", B, cov, B), 0.0)
    return out


def contrast_confirm(
    candidates: Sequence[FittedCandidate],
    pool_X: np.ndarray,
    measure: Callable[[np.ndarray], np.ndarray],
    *,
    noise: str = "relative",
    budget: int = 30,
    alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    seed: int = 0,
) -> ContrastResult:
    """Run the procedure in the module docstring.

    ``candidates`` are in the caller's order of preference; the first is the
    initial champion. ``measure(rows)`` returns the targets of those pool rows
    and is called only for rows the design selected, each row once.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    pool_X = np.asarray(pool_X, dtype=float)
    preds, cvar = [], []
    for cand in candidates:
        with np.errstate(all="ignore"):
            preds.append(np.asarray(cand.predict(pool_X), dtype=float) * np.ones(len(pool_X)))
            cvar.append(coefficient_variance(cand, pool_X, noise))
    scales = [noise_scale(c, noise) for c in candidates]
    usable = [i for i, p in enumerate(preds) if np.all(np.isfinite(p)) and np.all(np.isfinite(cvar[i]))
              and np.isfinite(scales[i]) and scales[i] > 0]
    if not usable:
        raise ValueError("no candidate has finite predictions on the pool and a positive development error")
    n_rivals = max(1, len(usable) - 1)
    z = float(stats.norm.ppf(1.0 - alpha / n_rivals))
    read: dict[int, float] = {}
    design = hashlib.sha256()

    def targets(rows: np.ndarray) -> np.ndarray:
        new = np.array([r for r in rows if int(r) not in read], dtype=int)
        if len(new):
            design.update(new.tobytes())
            for r, value in zip(new, np.asarray(measure(new), dtype=float), strict=True):
                read[int(r)] = float(value)
        got: np.ndarray = np.array([read[int(r)] for r in rows], dtype=float)
        return got

    champion, queue = usable[0], list(usable[1:])
    equivalent: list[int] = []
    eliminated: list[dict[str, Any]] = []
    while queue:
        rival = queue.pop(0)
        s2 = min(scales[champion], scales[rival]) ** 2
        var = s2 * (_weights(0.5 * (preds[champion] + preds[rival]), noise) ** 2 + cvar[champion] + cvar[rival])
        d2 = (preds[champion] - preds[rival]) ** 2 / var
        rows = np.argsort(-d2, kind="stable")[:budget]
        power = float(np.sqrt(d2[rows].sum()))
        if power < 2.0 * z:
            equivalent.append(rival)
            continue
        y = targets(rows)
        llr = float(np.sum(((y - preds[rival][rows]) ** 2 - (y - preds[champion][rows]) ** 2) / (2.0 * var[rows])))
        winner, loser = (champion, rival) if llr > 0 else (rival, champion)
        eliminated.append({"candidate": candidates[loser].candidate_id, "by": candidates[winner].candidate_id,
                           "disagreement_sigma": power, "log_likelihood_ratio": abs(llr), "rows": len(rows)})
        if winner != champion:
            champion = rival
            queue = equivalent + queue  # what the old champion could not be told from must face the new one
            equivalent = []
    if not read:  # nothing was separable: still test the survivor on a seeded draw
        targets(np.random.default_rng(seed).choice(len(pool_X), size=min(budget, len(pool_X)), replace=False))
    rows_all = np.array(sorted(read), dtype=int)
    y_all = np.array([read[int(r)] for r in rows_all])
    cand = candidates[champion]
    var_c = scales[champion] ** 2 * (_weights(preds[champion][rows_all], noise) ** 2 + cvar[champion][rows_all])
    f_stat = float(np.mean((y_all - preds[champion][rows_all]) ** 2 / var_c))
    p_value = float(stats.f.sf(f_stat, len(rows_all), max(1, len(cand.dev_y) - cand.n_params)))
    fit = {"f": f_stat, "p_value": p_value, "rows": len(rows_all), "alpha": alpha}
    names = [candidates[i].candidate_id for i in equivalent]
    if p_value < alpha:
        return ContrastResult(NO_SURVIVOR, None, [], eliminated, len(read), design.hexdigest(), fit)
    verdict = EQUIVALENT_CLASS if names else (UNIQUE if eliminated else UNCHALLENGED)
    return ContrastResult(verdict, cand.candidate_id, names, eliminated, len(read), design.hexdigest(), fit)
