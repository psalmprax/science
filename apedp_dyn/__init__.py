"""APEDP-DYN (D1): executable dynamical laws.

Dependency direction (D1.8):

    grammar -> IR -> gates -> fitter -> simulator -> intervention

The optimizer never invents structure: the mechanical grammar emits a typed
CandidateDynamics IR; gates 1-2 prune before any integration; the fitter only
estimates numbers for frozen structures; gate 4 (held-out intervention) decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apedp_dyn._lazy import lazy_exports

if TYPE_CHECKING:  # the names below, for type checkers and readers; nothing is imported at runtime
    from apedp_dyn import ode, pde
    from apedp_dyn.epistemic import (
        CAPABILITY_ABSENT,
        REL_FLOOR,
        Adjudication,
        EpistemicContract,
        EpistemicStatus,
        adjudicate,
        bootstrap_available,
        resolution_floor,
        scale_from_observations,
    )
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
        EnergyLaw,
        check_conservative,
        check_conservative_core,
        check_damped_decay,
        check_dissipative,
        derive_energy_law,
        energy_drift,
        energy_drift_from_law,
        mass_normalized_energy,
        mechanical_energy,
        simulate_pair,
        simulate_trajectory,
        total_energy,
    )

# Public name -> the module that defines it. Resolved on first access, so
# importing this package (which importing any of its submodules does) costs
# nothing and pulls in nothing. See apedp_dyn/_lazy.py.
_EXPORTS = {
    "Adjudication": "apedp_dyn.epistemic",
    "CAPABILITY_ABSENT": "apedp_dyn.epistemic",
    "EpistemicContract": "apedp_dyn.epistemic",
    "EpistemicStatus": "apedp_dyn.epistemic",
    "REL_FLOOR": "apedp_dyn.epistemic",
    "adjudicate": "apedp_dyn.epistemic",
    "bootstrap_available": "apedp_dyn.epistemic",
    "resolution_floor": "apedp_dyn.epistemic",
    "scale_from_observations": "apedp_dyn.epistemic",
    "WeakWindows": "apedp_dyn.estimation",
    "boundary_term": "apedp_dyn.estimation",
    "bump": "apedp_dyn.estimation",
    "bump_deriv": "apedp_dyn.estimation",
    "make_windows": "apedp_dyn.estimation",
    "spline_coefficients": "apedp_dyn.estimation",
    "square_window_coefficients": "apedp_dyn.estimation",
    "weak_coefficients": "apedp_dyn.estimation",
    "FitResult": "apedp_dyn.fitting",
    "fit_candidate": "apedp_dyn.fitting",
    "GateResult": "apedp_dyn.gates",
    "auto_tolerance": "apedp_dyn.gates",
    "dimensional_validity": "apedp_dyn.gates",
    "equation_admissibility": "apedp_dyn.gates",
    "forward_simulation_validity": "apedp_dyn.gates",
    "intervention_holdout": "apedp_dyn.gates",
    "observation_scale": "apedp_dyn.gates",
    "run_all_gates": "apedp_dyn.gates",
    "CandidateDynamics": "apedp_dyn.grammar",
    "GRAMMAR_VERSION": "apedp_dyn.grammar",
    "ScoredCandidate": "apedp_dyn.grammar",
    "SystemCandidate": "apedp_dyn.grammar",
    "TERM_FAMILIES": "apedp_dyn.grammar",
    "Term": "apedp_dyn.grammar",
    "canonical_form": "apedp_dyn.grammar",
    "canonical_system": "apedp_dyn.grammar",
    "deduplicate": "apedp_dyn.grammar",
    "enumerate_candidates": "apedp_dyn.grammar",
    "enumerate_coupled": "apedp_dyn.grammar",
    "full_topology_candidate": "apedp_dyn.grammar",
    "materialize": "apedp_dyn.grammar",
    "materialize_system": "apedp_dyn.grammar",
    "prune_admissible": "apedp_dyn.grammar",
    "select_by_train_loss": "apedp_dyn.grammar",
    "DynamicalEquation": "apedp_dyn.model",
    "InputVariable": "apedp_dyn.model",
    "Parameter": "apedp_dyn.model",
    "PhysicalModel": "apedp_dyn.model",
    "StateVariable": "apedp_dyn.model",
    "TIME_SYMBOL": "apedp_dyn.model",
    "make_signal": "apedp_dyn.problems.base",
    "SimulateResult": "apedp_dyn.simulation",
    "compile_rhs": "apedp_dyn.simulation",
    "simulate": "apedp_dyn.simulation",
    "EnergyLaw": "apedp_dyn.validation",
    "check_conservative": "apedp_dyn.validation",
    "check_conservative_core": "apedp_dyn.validation",
    "check_damped_decay": "apedp_dyn.validation",
    "check_dissipative": "apedp_dyn.validation",
    "derive_energy_law": "apedp_dyn.validation",
    "energy_drift": "apedp_dyn.validation",
    "energy_drift_from_law": "apedp_dyn.validation",
    "mass_normalized_energy": "apedp_dyn.validation",
    "mechanical_energy": "apedp_dyn.validation",
    "simulate_pair": "apedp_dyn.validation",
    "simulate_trajectory": "apedp_dyn.validation",
    "total_energy": "apedp_dyn.validation",
}

__all__ = [
    "CAPABILITY_ABSENT",
    "GRAMMAR_VERSION",
    "REL_FLOOR",
    "TERM_FAMILIES",
    "TIME_SYMBOL",
    "Adjudication",
    "CandidateDynamics",
    "DynamicalEquation",
    "EnergyLaw",
    "EpistemicContract",
    "EpistemicStatus",
    "FitResult",
    "GateResult",
    "InputVariable",
    "Parameter",
    "PhysicalModel",
    "ScoredCandidate",
    "SimulateResult",
    "StateVariable",
    "SystemCandidate",
    "Term",
    "WeakWindows",
    "adjudicate",
    "auto_tolerance",
    "bootstrap_available",
    "boundary_term",
    "bump",
    "bump_deriv",
    "canonical_form",
    "canonical_system",
    "check_conservative",
    "check_conservative_core",
    "check_damped_decay",
    "check_dissipative",
    "compile_rhs",
    "deduplicate",
    "derive_energy_law",
    "dimensional_validity",
    "energy_drift",
    "energy_drift_from_law",
    "enumerate_candidates",
    "enumerate_coupled",
    "equation_admissibility",
    "fit_candidate",
    "forward_simulation_validity",
    "full_topology_candidate",
    "intervention_holdout",
    "make_signal",
    "make_windows",
    "mass_normalized_energy",
    "materialize",
    "materialize_system",
    "mechanical_energy",
    "observation_scale",
    "ode",
    "pde",
    "prune_admissible",
    "resolution_floor",
    "run_all_gates",
    "scale_from_observations",
    "select_by_train_loss",
    "simulate",
    "simulate_pair",
    "simulate_trajectory",
    "spline_coefficients",
    "square_window_coefficients",
    "total_energy",
    "weak_coefficients",
]

# Subpackages that are public names in their own right (`from apedp_dyn import ode`).
_SUBMODULES = ("ode", "pde")

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, _SUBMODULES)
