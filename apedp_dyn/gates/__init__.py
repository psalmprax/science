"""D1 gates package. Fixed order, no short-circuit: full audit trail."""
import numpy as np

from apedp_dyn.gates.admissibility import (
    equation_admissibility,
    equation_admissibility_structure,
)
from apedp_dyn.gates.dimensional import (
    ALLOWED_FUNCTIONS,
    DIMENSIONLESS_FUNCS,
    GateResult,
    _dims_of,
    _unit_of,
    _ureg,
    dimensional_validity,
)
from apedp_dyn.gates.intervention import auto_tolerance, intervention_holdout
from apedp_dyn.gates.simulation import forward_simulation_validity


def run_all_gates(model, problem) -> list[GateResult]:
    """All four gates in fixed order. No short-circuit: full audit trail."""
    t = problem.train
    return [
        dimensional_validity(model),
        equation_admissibility(model),
        forward_simulation_validity(model, t["t_span"], t["u_of_t"],
                                    np.asarray(t["x0"], dtype=float)),
        intervention_holdout(model, problem),
    ]


__all__ = [
    "ALLOWED_FUNCTIONS",
    "DIMENSIONLESS_FUNCS",
    "GateResult",
    "_dims_of",
    "_unit_of",
    "_ureg",
    "auto_tolerance",
    "dimensional_validity",
    "equation_admissibility",
    "equation_admissibility_structure",
    "forward_simulation_validity",
    "intervention_holdout",
    "run_all_gates",
]
