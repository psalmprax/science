"""Gate 4 — discrimination/intervention: the held-out experiment decides."""
from __future__ import annotations

import numpy as np

from apedp_dyn.gates.dimensional import GateResult
from apedp_dyn.simulation import simulate


def observation_scale(obs: np.ndarray) -> dict[str, float]:
    """Scale and noise floor of a holdout observation record.

    Measured over ALL channels, because gate 4's RMSE is all-channel: a
    tolerance derived from one channel bounds a statistic that mixes every
    state, so the gate's strictness drifted with how many states a problem had
    (2-DOF problems looked 2x looser than 1-DOF ones at the same setting).
    Accepts (n_samples,) or (n_samples, n_states); the two coincide for 1-D.

    Two different questions need two different scales, and conflating them was
    a design flaw:

      - gate 4 ACCEPTANCE asks "is this model good enough?" -> auto_tolerance
      - epistemic RESOLUTION asks "can two models be told apart?" -> the
        scale-free floor in apedp_dyn.epistemic, built from these metrics

    sigma-hat comes from a third difference (noise dominates the highest
    frequencies); it is a heuristic for smooth signals, recorded not proven.
    """
    obs = np.asarray(obs, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    n = int(obs.shape[0])
    if n >= 4:
        per_channel = (1.4826 * np.median(np.abs(np.diff(obs, n=3, axis=0)), axis=0)
                       / np.sqrt(6.0))
        sigma = float(np.sqrt(np.mean(per_channel ** 2)))
    else:
        # Too short for a third difference; report no noise estimate rather
        # than a number derived from an empty median.
        sigma = 0.0
    return {
        "obs_rms": float(np.sqrt(np.mean(obs ** 2))),
        "noise_sigma": sigma,
        "n_samples": n,
        "n_channels": int(obs.shape[1]),
    }


def auto_tolerance(obs: np.ndarray) -> float:
    """Noise-floor-aware holdout ACCEPTANCE tolerance (ported from apedp/dyn).

    max(10% signal RMS, 1.5x sigma-hat): comparing a simulation against noisy
    measurements cannot beat ~sigma, and a model at the floor is doing as well
    as achievable. The 1.5x factor covers estimator inflation on smooth
    signals without admitting wrong structures. Set problem
    holdout["tolerance"] = "auto" to use it.

    This is an accept/reject threshold, NOT a distinguishability margin; do
    not reuse it to resolve two candidates against each other.
    """
    s = observation_scale(obs)
    return max(0.10 * s["obs_rms"], 1.5 * s["noise_sigma"])


def intervention_holdout(model, problem) -> GateResult:
    """Gate 4: under the held-out intervention, simulation matches truth."""
    h = problem.holdout
    n_eval = len(h["truth_t"])
    # Physics precision, not fitter throughput. simulate() defaults to
    # rtol=1e-3 for speed, which is right for ranking thousands of candidates
    # and wrong for the gate that decides whether a model matches reality:
    # the truth model on PS-GRAV2 failed its own holdout with rmse 0.139
    # against a 0.117 tolerance, and every bit of that was integration error
    # against a DOP853 truth (rtol 1e-11) rather than model error. A gate that
    # can reject the exact truth is measuring the integrator.
    res = simulate(model, h["t_span"], h["u_of_t"],
                   np.asarray(h["x0"], dtype=float), n_eval=n_eval,
                   rtol=1e-9, atol=1e-11)
    if not res.success:
        return GateResult("intervention_holdout", False,
                          f"holdout simulation invalid: {res.detail}", res.metrics)
    obs = np.asarray(h["truth_y"])
    sq_err = (res.y.T - obs) ** 2
    rmse = float(np.sqrt(np.mean(sq_err)))
    tol = h["tolerance"]
    if isinstance(tol, str) and tol == "auto":
        tol = auto_tolerance(obs)
    tol = float(tol)
    # margin_ratio = rmse/tol, so <1 passes and the number says how much room
    # was left. Pass/fail saturates: a model passing at 0.95x and one at 0.05x
    # both read "passed", which is why the gate could not rank or compare
    # candidates. The ratio is the recorded, comparable quantity.
    margin_ratio = rmse / tol if tol > 0 else float("inf")
    # Observation scale travels with the evidence so adjudication can build a
    # scale-free resolution floor without re-deriving it (and without reading
    # the tolerance above).
    metrics = {"rmse": rmse, "tolerance": tol, "margin": tol - rmse,
               "margin_ratio": margin_ratio,
               # Per-timepoint squared error, meaned over channels. Carried
               # here because gate 4 already computes it and adjudication's
               # residual-whiteness guard needs it: evidence the gate has
               # should not have to be reconstructed by each caller to be
               # used. Until this existed, dyn_sweep hand-rolled it and every
               # other call site did not, which left the whiteness guard
               # unsatisfiable and the sole-survivor verdict structurally
               # unreachable outside that one script.
               #
               # tolist(), not the raw array: every other metric here is a
               # plain scalar, and provenance is serialized into artifacts
               # (dyn_tournament). A stray ndarray would make the whole
               # provenance dict unserializable, which is a far worse
               # failure than the one this key fixes.
               "sq_error": sq_err.mean(axis=1).tolist(),
               **observation_scale(obs), **res.metrics}
    # Signed residuals of the primary observed channel (channel 0). The
    # whiteness guard is defined on SIGNED residuals; squared error (above) is
    # a different statistic (it also fires on volatility clustering and on any
    # smooth error shape). Carried separately so callers can use the right one.
    # Signs cannot be meaned across channels, hence channel 0 only.
    resid0 = (res.y.T - obs)[:, 0]
    metrics["residuals"] = resid0.tolist()
    # R^2 against the holdout MEAN of the primary channel. Deliberately not
    # against zero: 1 - rmse^2/rms^2 credits any series with a non-zero level
    # for "explaining" its own mean (a constant model scored 0.64 that way on
    # the crypto holdout while explaining ~10% of the variance about the mean).
    y0 = np.asarray(obs, dtype=float)[:, 0]
    ss_tot0 = float(np.sum((y0 - y0.mean()) ** 2))
    metrics["r2"] = (float(1.0 - np.sum(resid0 ** 2) / ss_tot0)
                     if ss_tot0 > 1e-12 else None)
    verdict = (f"rmse {rmse:.3g} {'<=' if rmse <= tol else '>'} tol {tol:.3g} "
               f"(margin_ratio {margin_ratio:.3g})")
    return GateResult("intervention_holdout", rmse <= tol, verdict, metrics)
