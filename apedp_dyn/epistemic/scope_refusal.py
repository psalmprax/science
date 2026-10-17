"""Referee-side out-of-grammar scope refusal (v0.1.4 wiring, spec §16d).

Closes the gap the known-truth negative control exposed (§16c): on an
out-of-grammar world both the proposer and the baseline EMITTED spurious
fits that cleared every predictive gate. Scope refusal cannot come from the
search side for free -- it has to come from adjudication, which is exactly
where this module wires it.

**Anti-snooping discipline.** The referee never sees the planted truth, the
grammar's coverage declaration, or any dataset label. Its only inputs are the
(dev, holdout) frames, the emitted candidate's frozen predictor, and the
declared holdout noise scale of the measurement record. Refusal is decided by
the standard :func:`apedp_dyn.epistemic.adjudicate.adjudicate` contract --
nothing here reads a world flag.

**Mechanism.** The referee fits a fixed, preregistered set of
*family-distinct* rival probes (exponential family, log family) by least
squares from the data alone, then runs every probe through
``adjudicate([emitted, probe], ...)`` as a structural rival. The emitted
candidate is promoted only if it earns UNIQUE_SUPPORT against EVERY probe;
any MULTI / INSUFFICIENT / EXTRAPOLATION / probe-favored outcome declines
the emission. Attribution carries through adjudicate() unchanged
(``out_of_grammar_or_misspecified``, ``intrinsic_observational_nonidentifiability``,
``bootstrap_instability``).

Why exactly these probe families: they must be (a) family-distinct from the
proposer's declared static monomial algebra (so a probe cannot tie a true
monomial world and force wrongful refusals), and (b) different from the
emitted claim in *kind* (nonlinear reparametrizations of the axis), so an
out-of-grammar truth the emitted family mimics predictively is separated
structurally. They are declared before any run, not tuned afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

PROBE_IDS = ("probe_exp_family", "probe_log_family")
"""Preregistered referee probe families (spec §16d). Declared before any run."""

_EXP_C_GRID = np.logspace(0.0, np.log10(50.0), 33)
"""Exponential-family rate grid on the dev-normalized axis.

``y ~ a*exp(c*z) + d`` with ``z = (p - lo)/scale`` evaluated UNCLIPPED on the
holdout (an exponential must compound past the dev extent; clipping it is
precisely the extrapolation defect this repo recorded for polynomial OLS).
The rate cap of 50 keeps ``exp(c*z)`` finite for the reachable holdout
extents of the registered challenges (z beyond 1 by a modest factor); rates
above 50 on a unit-scaled axis would produce non-finite regressors before
any adjudication, so they are excluded from the declared probe range.
"""


def _mk_scored(cid: str,
               term_names: tuple[str, ...],
               sig_train: np.ndarray, y_train: np.ndarray,
               sig_hold: np.ndarray, y_hold: np.ndarray,
               obs_rms: float, noise_sigma: float | None,
               metadata: dict[str, Any] | None = None) -> ScoredCandidate:
    """Build a ScoredCandidate with the standard gate4 provenance record.

    Identical shape to the QEC threshold sandbox builders, so ``adjudicate``
    consumes it through the same paths (ranking, gates, bootstrap residual
    series) as every other candidate in the repo.
    """
    resid = y_hold - sig_hold
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    g4 = {"passed": True, "holdout_obs_rms": obs_rms,
          "holdout_noise_sigma": noise_sigma, "holdout_n_samples": len(y_hold),
          "holdout_residuals": resid.tolist()}
    cand = CandidateDynamics(
        lhs="target",
        terms=tuple(Term(n, f"b{i}") for i, n in enumerate(term_names)),
        candidate_id=cid,
        metadata=dict(metadata or {}))
    return ScoredCandidate(cand, {"gate4_status": g4},
                           train_loss=float(np.mean((y_train - sig_train) ** 2)),
                           holdout_rmse=rmse)


def build_family_probes(dev: pd.DataFrame, hold: pd.DataFrame,
                        feature_col: str = "p", target_col: str = "target",
                        noise_sigma: float | None = None) -> list[ScoredCandidate]:
    """Fit the preregistered family probes from the data alone.

    Both probes fit ordinary least squares on the DEV frame only and are then
    evaluated (and recorded) on the holdout. No planted truth, no grammar
    declaration, no world labels enter: the referee only ever sees frames.

    Returns ``[exp_probe, log_probe]``.
    """
    x_dev = dev[feature_col].to_numpy(dtype=float)
    y_dev = dev[target_col].to_numpy(dtype=float)
    x_ho = hold[feature_col].to_numpy(dtype=float)
    y_ho = hold[target_col].to_numpy(dtype=float)
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))

    ok = np.isfinite(x_dev) & np.isfinite(y_dev)
    x_dev, y_dev = x_dev[ok], y_dev[ok]
    lo, hi = float(x_dev.min()), float(x_dev.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("degenerate dev axis: probes cannot be built")
    scale = hi - lo

    # Probe 1: exponential family on the dev-normalized axis.
    # y ~ a*exp(c*z) + d with z = (p - lo)/scale; z is in [0, 1] on dev by
    # construction and extrapolated UNCLIPPED on the holdout (compounding),
    # with the rate grid capped so exp() stays finite there.
    z_dev = (x_dev - lo) / scale
    z_ho = (x_ho - lo) / scale
    best_mse, best_c, best_w = float("inf"), float("nan"), None
    with np.errstate(over="ignore", under="ignore"):
        for c in _EXP_C_GRID:
            e_dev = np.exp(c * z_dev)
            e_ho = np.exp(c * z_ho)
            if not (np.all(np.isfinite(e_dev)) and np.all(np.isfinite(e_ho))):
                continue
            A = np.column_stack([e_dev, np.ones_like(e_dev)])
            w, *_ = np.linalg.lstsq(A, y_dev, rcond=None)
            mse = float(np.mean((A @ w - y_dev) ** 2))
            if np.isfinite(mse) and mse < best_mse:
                best_mse, best_c, best_w = mse, float(c), w
    a_best, d_best = float(best_w[0]), float(best_w[1])
    sig_train = a_best * np.exp(best_c * z_dev) + d_best
    sig_hold = a_best * np.exp(best_c * z_ho) + d_best
    if not (np.all(np.isfinite(sig_train)) and np.all(np.isfinite(sig_hold))):
        # A selected rate must produce finite predictions on every frame the
        # referee scores; anything else is a probe build failure, not a
        # candidate property. Fail loudly rather than score inf/nan.
        raise FloatingPointError(
            f"exp-family probe (rate c={best_c:g}) produced non-finite "
            "predictions; the rate grid cap does not cover this axis extent")
    exp_probe = _mk_scored(
        "exp_family", ("one", "exp_family"),
        sig_train, y_dev, sig_hold, y_ho, obs_rms, noise_sigma,
        metadata={"probe_id": "probe_exp_family", "exp_rate_c": best_c})

    # Probe 2: log family on the raw axis. y ~ a*log(p) + b (p > 0 throughout).
    lx_dev = np.log(x_dev)
    lx_ho = np.log(x_ho)
    A = np.column_stack([lx_dev, np.ones_like(lx_dev)])
    (a_l, b_l), *_ = np.linalg.lstsq(A, y_dev, rcond=None)
    sig_train = a_l * lx_dev + b_l
    sig_hold = a_l * lx_ho + b_l
    log_probe = _mk_scored(
        "log_family", ("one", "log_family"),
        sig_train, y_dev, sig_hold, y_ho, obs_rms, noise_sigma,
        metadata={"probe_id": "probe_log_family"})
    return [exp_probe, log_probe]


def scored_from_predict(candidate_id: str,
                        predict: Callable[[pd.DataFrame], np.ndarray],
                        dev: pd.DataFrame, hold: pd.DataFrame,
                        target_col: str = "target",
                        noise_sigma: float | None = None,
                        term_name: str = "emitted") -> ScoredCandidate:
    """Wrap an emitted candidate's frozen predictor as a ScoredCandidate.

    The emitted proposal arrives from the proposer side as a read-only
    prediction closure (the challenge ``Frozen`` shape). This adapter scores
    it on the same (dev, holdout) frames with the same gate4 record that the
    probes carry, so the pair comparison is apples-to-apples inside
    adjudicate(). The adapter only calls the closure -- it never inspects
    the emitter's internals.
    """
    y_dev = dev[target_col].to_numpy(dtype=float)
    y_ho = hold[target_col].to_numpy(dtype=float)
    sig_train = np.asarray(predict(dev), dtype=float)
    sig_hold = np.asarray(predict(hold), dtype=float)
    if sig_train.shape != y_dev.shape or sig_hold.shape != y_ho.shape:
        raise ValueError(
            f"emitted predictor output shape {sig_train.shape} does not match "
            f"frames ({y_dev.shape} dev, {y_ho.shape} holdout)")
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))
    return _mk_scored(candidate_id, (term_name,), sig_train, y_dev,
                      sig_hold, y_ho, obs_rms, noise_sigma)


@dataclass
class ScopeRefusal:
    """Aggregated referee verdict on one emitted candidate.

    promoted  -- UNIQUE_SUPPORT against every probe (none probe-favored):
                 the emitted claim survived a family-distinct challenge.
    declined  -- at least one probe refused to concede (observational
                 indistinguishability, explicit favor for the probe, or an
                 INSUFFICIENT / EXTRAPOLATION outcome): no promotion.
    """

    verdict: str                       # "promoted" | "declined"
    emitted_id: str
    probes: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"scope_verdict": self.verdict,
                "emitted_id": self.emitted_id,
                "probes": self.probes,
                "reason": self.reason}


def _one_probe(emitted: ScoredCandidate, probe: ScoredCandidate,
               holdout_scale: dict[str, Any] | None,
               bootstrap: bool, bootstrap_n_boot: int, bootstrap_seed: int
               ) -> tuple[str, dict[str, Any]]:
    """adjudicate the emitted-vs-probe pair; classify from the emitted's seat.

    Returns (verdict, record). Role-aware, not naive: if the probe happens to
    rank first on train loss and wins UNIQUE_SUPPORT for itself, the EMITTED
    candidate still has not earned promotion -- it lost to a family-distinct
    rival.
    """
    emitted_id = emitted.candidate.candidate_id
    resid_best_series = emitted.provenance["gate4_status"]["holdout_residuals"]
    resid_rival_series = probe.provenance["gate4_status"]["holdout_residuals"]
    adj: Adjudication = adjudicate(
        [emitted, probe],
        holdout_scale=holdout_scale,
        resid_best=resid_best_series,
        resid_rival=resid_rival_series,
        bootstrap=bootstrap,
        bootstrap_n_boot=bootstrap_n_boot,
        bootstrap_seed=bootstrap_seed,
        data_origin="unknown", benchmark_level="unknown")
    status = adj.status
    promoted_is_emitted = (status is EpistemicStatus.UNIQUE_SUPPORT
                           and adj.promoted is not None
                           and adj.promoted.candidate.candidate_id == emitted_id)
    if promoted_is_emitted:
        verdict = "separated"
        reason = adj.reason
    else:
        verdict = "declined"
        if status is EpistemicStatus.UNIQUE_SUPPORT:
            reason = ("probe earned UNIQUE_SUPPORT; the emitted candidate "
                      "did not survive a family-distinct rival")
        else:
            reason = adj.reason
    record = {
        "probe": probe.candidate.metadata.get("probe_id",
                                              probe.candidate.candidate_id),
        "status": status.value,
        "verdict": verdict,
        "reason": reason,
        "tau_epi": None if adj.tau_epi is None else float(adj.tau_epi),
    }
    return verdict, record


def adjudicate_scope(emitted: ScoredCandidate, probes: list[ScoredCandidate],
                     holdout_scale: dict[str, Any] | None = None,
                     bootstrap: bool = True, bootstrap_n_boot: int = 1000,
                     bootstrap_seed: int = 7) -> ScopeRefusal:
    """Full referee pass: promoted only if EVERY probe concedes.

    Deterministic. ``bootstrap=True`` enables the paired block bootstrap
    tiebreaker inside each pairwise adjudication (an explicit opt-in that
    fails loudly when the tiebreaker is not compiled in, per the
    adjudicate() capability contract).
    """
    emitted_id = emitted.candidate.candidate_id
    records: list[dict[str, Any]] = []
    decline_reasons: list[str] = []
    for probe in probes:
        verdict, rec = _one_probe(emitted, probe, holdout_scale,
                                  bootstrap, bootstrap_n_boot, bootstrap_seed)
        records.append(rec)
        if verdict == "declined":
            decline_reasons.append(f"{rec['probe']}: {rec['reason']}")
    declined = bool(decline_reasons)
    return ScopeRefusal(
        verdict="declined" if declined else "promoted",
        emitted_id=emitted_id,
        probes=records,
        reason=("scope emission declined by referee; " + " | ".join(decline_reasons)
                if declined else
                "emitted candidate earned UNIQUE_SUPPORT against every "
                "preregistered family probe"))


__all__ = [
    "PROBE_IDS",
    "ScopeRefusal",
    "adjudicate_scope",
    "build_family_probes",
    "scored_from_predict",
]
