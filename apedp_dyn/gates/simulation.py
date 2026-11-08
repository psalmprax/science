"""Gate 3 — forward simulation: reject failures, NaNs, infinities, instability."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from apedp_dyn.gates.dimensional import GateResult
from apedp_dyn.simulation import simulate

if TYPE_CHECKING:
    from apedp_dyn.model.physical_model import PhysicalModel


def forward_simulation_validity(model: PhysicalModel, t_span: tuple[float, float],
                                u_of_t: Callable[[float], np.ndarray],
                                x0: np.ndarray | None = None) -> GateResult:
    """Gate 3: the model executes — integrates cleanly over the required span."""
    res = simulate(model, t_span, u_of_t,
                   model.initial_vector() if x0 is None else np.asarray(x0, dtype=float))
    if not res.success:
        return GateResult("forward_simulation_validity", False,
                          f"span {t_span}: {res.detail}", res.metrics)
    return GateResult("forward_simulation_validity", True,
                      f"span {t_span}: ok", res.metrics)
