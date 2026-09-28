"""APEDP-Dyn: Dynamical Systems Discovery Module.

Provides the type system, simulator, and validation framework for
physical model discovery from time-series observations.

Architecture:
    apedp-core → apedp.dyn.types → apedp.dyn.simulator → apedp.dyn.validation
"""
from __future__ import annotations

from apedp.dyn.estimation import (
    EstimationResult,
    HoldoutResult,
    ParameterEstimator,
)
from apedp.dyn.types import (
    DynamicalEquation,
    EpistemicStatus,
    InputVariable,
    InterventionTest,
    Observation,
    Parameter,
    PhysicalModel,
    SimulationResult,
    StateVariable,
    TrajectoryData,
    ValidationResult,
    VariableKind,
)

__all__ = [
    "DynamicalEquation",
    "EpistemicStatus",
    "EstimationResult",
    "HoldoutResult",
    "InputVariable",
    "InterventionTest",
    "Observation",
    "Parameter",
    "ParameterEstimator",
    "PhysicalModel",
    "SimulationResult",
    "StateVariable",
    "TrajectoryData",
    "ValidationResult",
    "VariableKind",
]
