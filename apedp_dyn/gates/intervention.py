"""Gate 4 — discrimination/intervention: the held-out experiment decides."""
from __future__ import annotations

import numpy as np

from apedp_dyn.gates.dimensional import GateResult
from apedp_dyn.simulation import simulate


def auto_tolerance(obs: np.ndarray) -> float:
    """Noise-floor-aware holdout tolerance (ported from apedp/dyn scoring).

    max(10% signal RMS, 1.5x Sigmanoise-hat): comparing a simulation against
    noisy measurements cannot beat ~sigma, and a model at the floor is doing
    as well as achievable. sigma-hat comes from the holdout itself via a
    third difference (noise dominates the highest frequencies); the 1.5x
    factor covers estimator inflation on smooth signals without admitting
    wrong structures. Set problem holdout["tolerance"] = "auto" to use it.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    rms_obs = float(np.sqrt(np.mean(obs ** 2)))
    sigma_hat = float(1.4826 * np.median(np.abs(np.diff(obs, n=3))) / np.sqrt(6.0))
    return max(0.10 * rms_obs, 1.5 * sigma_hat)


def intervention_holdout(model, problem) -> GateResult:
    """Gate 4: under the held-out intervention, simulation matches truth."""
    h = problem.holdout
    n_eval = len(h["truth_t"])
    res = simulate(model, h["t_span"], h["u_of_t"],
                   np.asarray(h["x0"], dtype=float), n_eval=n_eval)
    if not res.success:
        return GateResult("intervention_holdout", False,
                          f"holdout simulation invalid: {res.detail}", res.metrics)
    rmse = float(np.sqrt(np.mean((res.y.T - np.asarray(h["truth_y"])) ** 2)))
    tol = h["tolerance"]
    if isinstance(tol, str) and tol == "auto":
        tol = auto_tolerance(np.asarray(h["truth_y"])[:, 0])
    metrics = {"rmse": rmse, "tolerance": float(tol), **res.metrics}
    if rmse <= float(tol):
        return GateResult("intervention_holdout", True,
                          f"rmse {rmse:.3g} <= tol {tol:.3g}", metrics)
    return GateResult("intervention_holdout", False,
                      f"rmse {rmse:.3g} > tol {tol:.3g}", metrics)
