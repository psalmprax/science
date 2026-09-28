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
    square_window_coefficients,
    weak_coefficients,
)
from apedp_dyn.epistemic import (
    REL_FLOOR,
    Adjudication,
    EpistemicStatus,
    adjudicate,
    resolution_floor,
)
from apedp_dyn.fitting import FitResult, fit_candidate
from apedp_dyn.gates import (
    GateResult,
    auto_tolerance,
    dimensional_validity,
    equation_admissibility,
    forward_simulation_validity,
    intervention_holdout,
    observation_scale,
    run_all_gates,
)
from apedp_dyn.grammar import (
    GRAMMAR_VERSION,
    TERM_FAMILIES,
    CandidateDynamics,
    ScoredCandidate,
    SystemCandidate,
    Term,
    canonical_form,
    canonical_system,
    deduplicate,
    enumerate_candidates,
    enumerate_coupled,
    full_topology_candidate,
    materialize,
    materialize_system,
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
from apedp_dyn.problems.base import make_signal
from apedp_dyn.simulation import SimulateResult, compile_rhs, simulate
from apedp_dyn.validation import (
    check_conservative_core,
    check_damped_decay,
    energy_drift,
    mechanical_energy,
    simulate_pair,
)

__all__ = [
    "Adjudication",
    "EpistemicStatus",
    "FitResult",
    "GateResult",
    "GRAMMAR_VERSION",
    "REL_FLOOR",
    "ScoredCandidate",
    "SimulateResult",
    "CandidateDynamics",
    "DynamicalEquation",
    "InputVariable",
    "Parameter",
    "PhysicalModel",
    "StateVariable",
    "SystemCandidate",
    "TIME_SYMBOL",
    "TERM_FAMILIES",
    "Term",
    "WeakWindows",
    "adjudicate",
    "auto_tolerance",
    "boundary_term",
    "bump",
    "bump_deriv",
    "canonical_form",
    "canonical_system",
    "check_conservative_core",
    "check_damped_decay",
    "compile_rhs",
    "deduplicate",
    "dimensional_validity",
    "energy_drift",
    "enumerate_candidates",
    "enumerate_coupled",
    "equation_admissibility",
    "fit_candidate",
    "forward_simulation_validity",
    "full_topology_candidate",
    "intervention_holdout",
    "make_signal",
    "materialize",
    "materialize_system",
    "mechanical_energy",
    "observation_scale",
    "prune_admissible",
    "resolution_floor",
    "run_all_gates",
    "select_by_train_loss",
    "simulate",
    "simulate_pair",
    "spline_coefficients",
    "square_window_coefficients",
    "weak_coefficients",
]
