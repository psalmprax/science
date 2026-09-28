"""APEDP-DYN (D1): executable dynamical laws.

Dependency direction (D1.8):

    grammar -> IR -> gates -> fitter -> simulator -> intervention

The optimizer never invents structure: the mechanical grammar emits a typed
CandidateDynamics IR; gates 1-2 prune before any integration; the fitter only
estimates numbers for frozen structures; gate 4 (held-out intervention) decides.
"""
from apedp_dyn.estimation import (
    WeakWindows,
    boundary_term,
    bump,
    bump_deriv,
    make_windows,
    spline_coefficients,
    weak_coefficients,
)
from apedp_dyn.fitting import FitResult, fit_candidate
from apedp_dyn.gates import (
    GateResult,
    dimensional_validity,
    equation_admissibility,
    forward_simulation_validity,
    intervention_holdout,
    run_all_gates,
)
from apedp_dyn.grammar import (
    GRAMMAR_VERSION,
    TERM_FAMILIES,
    CandidateDynamics,
    ScoredCandidate,
    Term,
    canonical_form,
    deduplicate,
    enumerate_candidates,
    full_topology_candidate,
    materialize,
    prune_admissible,
    select_by_train_loss,
)
from apedp_dyn.model import (
    TIME_SYMBOL,
    DynamicalEquation,
    InputVariable,
    Parameter,
    PhysicalModel,
    StateVariable,
)
from apedp_dyn.simulation import SimulateResult, simulate

__all__ = [
    "FitResult",
    "GateResult",
    "GRAMMAR_VERSION",
    "ScoredCandidate",
    "SimulateResult",
    "CandidateDynamics",
    "DynamicalEquation",
    "InputVariable",
    "Parameter",
    "PhysicalModel",
    "StateVariable",
    "TIME_SYMBOL",
    "TERM_FAMILIES",
    "Term",
    "WeakWindows",
    "boundary_term",
    "bump",
    "bump_deriv",
    "canonical_form",
    "deduplicate",
    "dimensional_validity",
    "enumerate_candidates",
    "equation_admissibility",
    "fit_candidate",
    "forward_simulation_validity",
    "full_topology_candidate",
    "intervention_holdout",
    "materialize",
    "prune_admissible",
    "run_all_gates",
    "select_by_train_loss",
    "simulate",
    "spline_coefficients",
    "weak_coefficients",
]
