"""Gate 4 — discrimination/intervention: the held-out experiment decides."""
from __future__ import annotations

import numpy as np

from apedp_dyn.gates.dimensional import GateResult
from apedp_dyn.simulation import simulate


def observation_scale(obs: np.ndarray) -> dict[str, float]:
    """Scale and noise floor of a holdout observation record.

    Two different questions need two different scales, and conflating them was
    a design flaw:

      - gate 4 ACCEPTANCE asks "is this model good enough?" -> auto_tolerance
      - epistemic RESOLUTION asks "can two models be told apart?" -> the
        scale-free floor in apedp_dyn.epistemic, built from these metrics

    sigma-hat comes from a third difference (noise dominates the highest
    frequencies); it is a heuristic for smooth signals, recorded not proven.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    return {
        "obs_rms": float(np.sqrt(np.mean(obs ** 2))),
        "noise_sigma": float(1.4826 * np.median(np.abs(np.diff(obs, n=3))) / np.sqrt(6.0)),
        "n_samples": int(obs.size),
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
    res = simulate(model, h["t_span"], h["u_of_t"],
                   np.asarray(h["x0"], dtype=float), n_eval=n_eval)
    if not res.success:
        return GateResult("intervention_holdout", False,
                          f"holdout simulation invalid: {res.detail}", res.metrics)
    obs = np.asarray(h["truth_y"])
    rmse = float(np.sqrt(np.mean((res.y.T - obs) ** 2)))
    tol = h["tolerance"]
    if isinstance(tol, str) and tol == "auto":
        tol = auto_tolerance(obs[:, 0])
    # Observation scale travels with the evidence so adjudication can build a
    # scale-free resolution floor without re-deriving it (and without reading
    # the tolerance above).
    metrics = {"rmse": rmse, "tolerance": float(tol),
               **observation_scale(obs[:, 0]), **res.metrics}
    if rmse <= float(tol):
        return GateResult("intervention_holdout", True,
                          f"rmse {rmse:.3g} <= tol {tol:.3g}", metrics)
    return GateResult("intervention_holdout", False,
                      f"rmse {rmse:.3g} > tol {tol:.3g}", metrics)
