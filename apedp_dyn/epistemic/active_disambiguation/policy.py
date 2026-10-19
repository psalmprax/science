"""Active epistemic disambiguation (E4, forensics doc S71).

When adjudication yields MULTI_HYPOTHESIS, this layer selects the additional
measurement that maximally distinguishes the survivors — or abstains with
INSUFFICIENT_DISCRIMINATION when nothing clears the detectability threshold.

Additive layer only: adjudicate.py untouched. Frozen protocol E4-P1..P8:
  P1 hypothesis preservation (input genuinely MULTI), P2 no truth leakage
  (selector sees survivors + observation model only), P3 blindness, P4
  noise-normalized separation, P5 active advantage over random/worst,
  P6 post-measurement re-adjudication through adjudicate(), P7 abstention,
  P8 frozen seeds / hypotheses / measurements / ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


DEFAULT_DETECTABILITY_K = 3.0


@dataclass(frozen=True)
class Measurement:
    """One candidate observation: query point + measurement noise."""

    mid: str
    x_star: float
    sigma: float


@dataclass(frozen=True)
class DiscriminationRow:
    mid: str
    d_sigma: float  # min pairwise |pred_i - pred_j| / sigma (P4)


@dataclass(frozen=True)
class SelectionOutcome:
    selected: Measurement | None
    table: tuple[DiscriminationRow, ...]
    abstained: bool
    reason: str = ""


def predict_all(
    hypotheses: Mapping[str, Callable[[float], float]],
    measurements: list[Measurement],
) -> dict[str, dict[str, float]]:
    """Predicted response y_hat[m][h] (P2/P3: hypotheses + observation model only)."""
    return {m.mid: {h: float(fn(m.x_star)) for h, fn in hypotheses.items()} for m in measurements}


def discrimination_table(
    predictions: Mapping[str, Mapping[str, float]],
    sigmas: Mapping[str, float],
) -> tuple[DiscriminationRow, ...]:
    """Noise-normalized min-pairwise separation per measurement (P4)."""
    rows = []
    for mid, preds in predictions.items():
        vals = list(preds.values())
        sep = min(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1 :]) if len(vals) > 1 else 0.0
        rows.append(DiscriminationRow(mid=mid, d_sigma=sep / sigmas[mid] if sigmas[mid] > 0 else 0.0))
    return tuple(sorted(rows, key=lambda r: r.d_sigma, reverse=True))


def select_measurement(
    hypotheses: Mapping[str, Callable[[float], float]],
    measurements: list[Measurement],
    detectability_k: float = DEFAULT_DETECTABILITY_K,
) -> SelectionOutcome:
    """Max-D_sigma policy with abstention (P7: INSUFFICIENT_DISCRIMINATION)."""
    preds = predict_all(hypotheses, measurements)
    table = discrimination_table(preds, {m.mid: m.sigma for m in measurements})
    best = table[0] if table else None
    if best is None or best.d_sigma < detectability_k:
        got = f"{best.d_sigma:.2f}" if best is not None else "n/a"
        return SelectionOutcome(None, table, True, f"max D_sigma {got} < k={detectability_k}")
    by_id = {m.mid: m for m in measurements}
    return SelectionOutcome(by_id[best.mid], table, False, f"max D_sigma {best.d_sigma:.2f} >= k")


def _scored(hid: str, resid: np.ndarray, rmse: float) -> ScoredCandidate:
    c = CandidateDynamics(lhs="y", terms=(Term(hid, "a0"),), candidate_id=hid)
    return ScoredCandidate(
        c,
        {"gate4_status": {"passed": True, "holdout_residuals": list(resid),
                          "holdout_rmse": rmse, "holdout_obs_rms": 1.0}},
        train_loss=float(np.mean(np.asarray(resid) ** 2)),
        holdout_rmse=rmse,
    )


@dataclass(frozen=True)
class TrialResult:
    policy: str
    mid: str | None
    d_sigma: float
    status_post: str
    winner: str | None = None
    rounds: int = 0


def run_trial(
    policy: str,
    hypotheses: Mapping[str, Callable[[float], float]],
    measurements: list[Measurement],
    x_pre: np.ndarray,
    y_pre: np.ndarray,
    truth_fn: Callable[[float], float],
    rng: np.random.Generator,
    detectability_k: float = DEFAULT_DETECTABILITY_K,
    oracle_mid: str | None = None,
    max_rounds: int = 3,
) -> TrialResult:
    """One frozen-protocol trial: sequential select (P1-P5,P7) -> observe -> re-adjudicate (P6).

    Rounds continue while the verdict stays non-UNIQUE: a single observation
    rarely clears the sole-survivor margin (gamma=3.0), so the policy must
    re-select on updated evidence. Each round appends one (x*, y) observation.
    """
    by_id = {mm.mid: mm for mm in measurements}
    x_all = np.asarray(x_pre, dtype=float)
    y_all = np.asarray(y_pre, dtype=float)
    last_mid: str | None = None
    last_d = 0.0
    for round_ in range(1, max_rounds + 1):
        preds = predict_all(hypotheses, measurements)
        table = discrimination_table(preds, {m.mid: m.sigma for m in measurements})
        order = [r.mid for r in table]
        if policy == "active":
            sel = select_measurement(hypotheses, measurements, detectability_k)
            if sel.abstained or sel.selected is None:
                return TrialResult(policy, None, table[0].d_sigma if table else 0.0,
                                   "INSUFFICIENT_DISCRIMINATION", None, round_ - 1)
            m = sel.selected
        elif policy == "random":
            m = by_id[rng.choice(order)]
        elif policy == "worst":
            m = by_id[order[-1]]
        elif policy == "oracle":
            assert oracle_mid is not None
            m = by_id[oracle_mid]
        else:
            raise ValueError(f"unknown policy {policy}")
        last_mid = m.mid
        last_d = next(r.d_sigma for r in table if r.mid == m.mid)
        # Observe: truth at x_star + measurement noise (P2: truth used ONLY here).
        y_new = float(truth_fn(m.x_star)) + float(rng.normal(0, m.sigma))
        x_all = np.concatenate([x_all, [m.x_star]])
        y_all = np.concatenate([y_all, [y_new]])
        scored = []
        for hid, fn in hypotheses.items():
            resid = y_all - np.array([fn(x) for x in x_all])
            scored.append(_scored(hid, resid, float(np.sqrt(np.mean(resid**2)))))
        scored.sort(key=lambda s: float(s.holdout_rmse)
                    if s.holdout_rmse is not None else float("inf"))
        adj = adjudicate(list(scored), resid_best=np.asarray(scored[0].provenance["gate4_status"]["holdout_residuals"]),
                         resid_rival=np.asarray(scored[1].provenance["gate4_status"]["holdout_residuals"]),
                         bootstrap=False, data_origin="sim")
        status = str(getattr(adj, "status", adj))
        if "unique_support" in status.lower():
            return TrialResult(policy, last_mid, last_d, status, scored[0].candidate.candidate_id, round_)
    return TrialResult(policy, last_mid, last_d, status, scored[0].candidate.candidate_id, max_rounds)


@dataclass
class HarnessConfig:
    detectability_k: float = DEFAULT_DETECTABILITY_K
    n_trials: int = 40
    seed: int = 7


def compare_policies(
    hypotheses: Mapping[str, Callable[[float], float]],
    measurements: list[Measurement],
    x_pre: np.ndarray,
    y_pre_fn: Callable[[np.random.Generator], np.ndarray],
    truth_fn: Callable[[float], float],
    cfg: HarnessConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Active vs random vs worst vs oracle over frozen noise realizations (P5/P8)."""
    if cfg is None:
        cfg = HarnessConfig()
    preds = predict_all(hypotheses, measurements)
    table = discrimination_table(preds, {m.mid: m.sigma for m in measurements})
    oracle_mid = table[0].mid
    out: dict[str, dict[str, float]] = {}
    for policy in ("active", "random", "worst", "oracle"):
        uniq = 0
        for t in range(cfg.n_trials):
            rng = np.random.default_rng(cfg.seed * 1000 + t)
            r = run_trial(policy, hypotheses, measurements, x_pre, y_pre_fn(rng),
                          truth_fn, rng, cfg.detectability_k, oracle_mid)
            uniq += "unique_support" in r.status_post.lower()
        out[policy] = {"p_unique": uniq / cfg.n_trials, "n": float(cfg.n_trials)}
    return out
