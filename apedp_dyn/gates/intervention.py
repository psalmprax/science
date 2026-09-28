"""Gate 4 — discrimination/intervention: the held-out experiment decides."""
from __future__ import annotations

import numpy as np

from apedp_dyn.gates.dimensional import GateResult
from apedp_dyn.simulation import simulate


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
    metrics = {"rmse": rmse, "tolerance": float(h["tolerance"]), **res.metrics}
    if rmse <= float(h["tolerance"]):
        return GateResult("intervention_holdout", True,
                          f"rmse {rmse:.3g} <= tol {h['tolerance']:.3g}", metrics)
    return GateResult("intervention_holdout", False,
                      f"rmse {rmse:.3g} > tol {h['tolerance']:.3g}", metrics)
