"""Pins: Hawkes standardized-intensity scorer couples ONLY to fit-time constants.

REG-MUT uncovered a gap (MUT-C1 survived): the existing dynamic-signal pin
scores an eval frame statistically IDENTICAL to the fit frame, so it cannot
distinguish fit-constant standardization from eval-frame standardization (the
eval-frame means coincide). These pins make the distinction load-bearing:

  H1  A DEGENERATE fit (ties at the plateau of the intensity series, so
      binary viability fails) must select score_mode
      'continuous_standardized_intensity' — i.e. the continuous fallback the
      battery's MUT-C1 mutates is actually exercised.
  H2  On an eval frame whose volume distribution is SHIFTED away from the fit
      distribution, the emitted score equals the FIT-constant standardized
      intensity, reconstructed independently from the proposer's own (mu,
      alpha, beta) metadata — and is NOT the eval-frame standardized series.
      This is the Finding 10 property in executable form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.proposers.hawkes_orderflow_proposer import HawkesOrderFlowProposer


def _fit_frame(n: int = 200) -> pd.DataFrame:
    # Constant volume, unit-step time: R recurses to a plateau with float-exact
    # ties at the top decile -> binary viability fails -> continuous branch.
    return pd.DataFrame({"timestamp": np.arange(n, dtype=float),
                         "volume": np.full(n, 1000.0)})


def _eval_frame_shifted(n: int = 200, spike_at: int = 100) -> pd.DataFrame:
    # Same shape as the fit frame, but one colossal volume event mid-frame —
    # the resulting eval-frame intensity distribution is skewed FAR away from
    # the fit constants.
    vol = np.full(n, 1000.0)
    vol[spike_at] = 5.0e8
    return pd.DataFrame({"timestamp": np.arange(n, dtype=float), "volume": vol})


def _reconstruct_intens(eval_df: pd.DataFrame, mu: float, alpha: float, beta: float,
                        fit_mean_v: float) -> np.ndarray:
    """Mirror the scorer's own intensity recursion: eval volumes normalized by
    the FIT-time volume mean constant (fit_mean_v), NOT by eval-frame stats."""
    rv = np.nan_to_num(eval_df["volume"].to_numpy(dtype=float), nan=0.0)
    weights = rv / fit_mean_v
    sig = np.zeros(len(rv), dtype=float)
    for idx in range(1, len(rv)):
        delta_t = max(float(eval_df["timestamp"].iloc[idx]) - float(eval_df["timestamp"].iloc[idx - 1]), 1e-6)
        sig[idx] = np.exp(-beta * delta_t) * (weights[idx - 1] + sig[idx - 1])
    return mu + alpha * sig


def test_degenerate_fit_selects_continuous_fallback():  # H1
    props = HawkesOrderFlowProposer().fit_and_propose(event_data=_fit_frame())
    assert len(props) > 0
    p = max(props, key=lambda c: float(c.parameters["branching_ratio"]))
    assert p.metadata["binary_surge_viable_on_fit"] is False
    assert p.metadata["score_mode"] == "continuous_standardized_intensity"


def test_score_couples_to_fit_constants_not_eval_frame():  # H2
    dart = _fit_frame()
    proposer = HawkesOrderFlowProposer()
    props = proposer.fit_and_propose(event_data=dart)
    p = max(props, key=lambda c: float(c.parameters["branching_ratio"]))
    score = p.compute_score(_eval_frame_shifted())
    assert float(np.std(score)) > 0.0

    mu, alpha, beta = (float(p.parameters["mu"]), float(p.parameters["alpha"]),
                       float(p.parameters["beta"]))
    fit_mean_v = float(np.mean(dart["volume"].to_numpy(dtype=float)))  # FIT-frame constant
    intens = _reconstruct_intens(_eval_frame_shifted(), mu, alpha, beta, fit_mean_v)
    # Fit-time constants measured on the FIT frame's intensity distribution
    # (the fit is degenerate-but-not-constant because early ramps diverge from
    # the plateau; the proposer stores tau from quantiles, so reconstruct the
    # same constants from a fit-frame replay).
    fit_intens = _reconstruct_intens(dart, mu, alpha, beta, fit_mean_v)
    fit_mean, fit_std = float(np.mean(fit_intens)), float(np.std(fit_intens))
    expected_fit_standardized = (intens - fit_mean) / fit_std

    # The Finding 10 property: equal (up to float noise) to FIT-constant
    # standardization, and demonstrably different from eval-frame
    # standardization — otherwise the pin cannot tell one from the other.
    assert np.allclose(score, expected_fit_standardized, atol=1e-9), (
        "score must standardize by FIT-time constants (Finding 10: no eval-frame leak)")
    eval_standardized = (intens - float(intens.mean())) / float(intens.std())
    assert not np.allclose(score, eval_standardized, atol=1e-6), (
        "pin is unsharp: eval-frame standardization coincides on this fixture")
