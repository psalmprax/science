"""Guard 3 of the referee: a promoted law must leave no structure in its holdout residuals over the input space.

Status: experimental

**Why.** ``adjudicate`` decides between the candidates it is given. Its verdict
``UNIQUE_SUPPORT`` is relative to them: the sole survivor, or the winner of a
pair by a margin. Neither says the winner fits. On SRBench the template engine's
promotions were 268, and 160 of them were not accurate on the test rows
(PREREG-SRBENCH-BASELINE-2026-11, audit M-10): on a law outside the grammar the
least bad candidate was certified. The existing whiteness guard reads the
residuals in row order, and rows of a table have no order.

**What.** ``promotion_structure`` asks whether a residual can be predicted from
the residuals of its nearest neighbours in the standardised input space. The
statistic is the leave-one-out R² of that prediction; its null distribution is
the same statistic on seeded permutations of the residuals, and the level is
the canonical significance level (``DIEBOLD_MARIANO_P_VALUE_THRESHOLD``), so a
true law is refused at about that rate. It fails closed on missing or
misaligned evidence.

**Where.** ``guard_promotion`` is applied to the verdict ``adjudicate``
returned, not inside it. So it covers every route to a promotion, including
ones added later, and ``adjudicate.py`` is unchanged. (A first version sat
inside ``adjudicate`` and guarded two of its three promotion exits; development
found three wrong promotions leaving through the third.)

**Rounding.** On noise-free data the right law leaves errors of a few units in
the last place. Those follow the size of the prediction, so they are structured
over the inputs. Residuals within ``ABSOLUTE_EPS`` of the holdout observation
RMS are an exact fit up to rounding and pass untested. (Development found this
too: without it 4 of 7 correct laws were refused at zero noise.)

It is off unless a caller asks for it (``L2DiscoveryEngine.discover(structure_rule=True)``)
until PREREG-REFEREE-STRUCTURE-2026-11 has been run.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.epistemic.adjudicate import (
    ABSOLUTE_EPS,
    Adjudication,
    EpistemicAttribution,
    EpistemicStatus,
)

NEIGHBOURS = 8
PERMUTATIONS = 999
MAX_ROWS = 1500
"""Rows the test reads at most (a seeded sample beyond that): the distance matrix is quadratic in it."""
FLAG = "STRUCTURED_RESIDUALS_IN_INPUT_SPACE"


def promotion_structure(resid: Any, inputs: Any, *, alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
                        k: int = NEIGHBOURS, permutations: int = PERMUTATIONS,
                        obs_rms: float | None = None) -> tuple[bool, str]:
    """``(passes, why)`` for one residual series and the inputs it was measured at, one row per residual.

    Deterministic. ``obs_rms`` is the RMS of the holdout observations; without
    it there is no allowance for rounding and only exact zeros pass untested.
    """
    if resid is None or inputs is None:
        return False, "no residuals or no inputs; structure in input space unassessable"
    r = np.asarray(resid, dtype=float).ravel()
    X = np.asarray(inputs, dtype=float)
    X = X.reshape(len(X), -1) if X.ndim != 2 else X
    if len(X) != r.size:
        return False, f"{r.size} residuals for {len(X)} input rows; structure in input space unassessable"
    if r.size < 4 * k:
        return False, f"too few holdout rows ({r.size}) to test structure with {k} neighbours"
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(X))):
        return False, "residuals or inputs contain non-finite values"
    if not np.any(r):
        return True, "residuals are exactly zero: exact fit, no error left to test"
    if obs_rms is not None and np.isfinite(obs_rms) and obs_rms > 0:
        rms = float(np.sqrt(np.mean(r * r)))
        if rms <= ABSOLUTE_EPS * float(obs_rms):
            return True, (f"residual RMS {rms:.3g} is within {ABSOLUTE_EPS:.0e} of the observation RMS "
                          f"{float(obs_rms):.3g}: exact fit up to rounding")
    rng = np.random.default_rng(0)
    if r.size > MAX_ROWS:
        rows = np.sort(rng.choice(r.size, MAX_ROWS, replace=False))
        r, X = r[rows], X[rows]
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    centred = r - r.mean()
    total = float(np.dot(centred, centred))
    if total <= 1e-24 * float(np.dot(r, r)):  # constant up to rounding
        return False, "residuals are constant and not zero: a fixed offset is left unexplained"

    def statistic(values: np.ndarray) -> float:
        return 1.0 - float(np.sum((values - values[nn].mean(axis=1)) ** 2)) / total

    observed = statistic(r)
    exceed = sum(statistic(rng.permutation(r)) >= observed for _ in range(permutations))
    p_value = (1 + exceed) / (1 + permutations)
    detail = f"neighbour R^2={observed:.3f}, permutation p={p_value:.3f} vs level {alpha:g}, k={k}, n={r.size}"
    return bool(p_value >= alpha), detail


def _gate4(scored: Any) -> dict[str, Any]:
    provenance = getattr(scored, "provenance", None)
    record = provenance.get("gate4_status", {}) if isinstance(provenance, dict) else {}
    return record if isinstance(record, dict) else {}


def guard_promotion(adj: Adjudication, holdout_inputs: Any, *, residuals: Any = None,
                    obs_rms: float | None = None) -> Adjudication:
    """``adj`` unchanged, unless it promotes a law that leaves structure in its holdout residuals: then a refusal.

    The residual series and the observation RMS are the ones gate 4 recorded
    for the promoted candidate, unless given. A verdict that promotes nothing
    is returned as it is. The refusal keeps every number of the original
    verdict and changes the status, the promoted law, the attribution, the
    flags and the reason.
    """
    if adj.status is not EpistemicStatus.UNIQUE_SUPPORT or adj.promoted is None:
        return adj
    record = _gate4(adj.promoted)
    series = residuals if residuals is not None else record.get("holdout_residuals")
    scale = obs_rms if obs_rms is not None else record.get("holdout_obs_rms")
    ok, why = promotion_structure(series, holdout_inputs, obs_rms=scale)
    if ok:
        return adj
    name = getattr(getattr(adj.promoted, "candidate", None), "candidate_id", "the promoted candidate")
    if adj.rival is None:
        reason = (f"sole survivor {name} leaves structure in its holdout residuals over the input space ({why}); "
                  f"being the only candidate left is not explanatory support")
    else:
        reason = (f"{name} beats its nearest rival and leaves structure in its holdout residuals over the input "
                  f"space ({why}); the better of two inadequate candidates is not a discovery")
    return replace(adj, status=EpistemicStatus.INSUFFICIENT_EVIDENCE, promoted=None,
                   epistemic_attribution=EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED.value,
                   promotion_confidence="", diagnostic_flags=(FLAG, *adj.diagnostic_flags),
                   reason=f"{reason}. Before this guard the verdict was: {adj.reason}")
