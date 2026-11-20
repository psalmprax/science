"""Competing Proposers Package.

Provides diverse, independent hypothesis generators for the APEDP-X discovery pipeline:
  - SparseMicrostructureProposer: L1/ElasticNet regularized linear microstructure signals.
  - OrderBookResiliencyProposer: Event-driven exponential liquidity recovery signals.
  - DistributedLagInteractionProposer: Rich distributed-lag kernels and bilinear interactions.
  - RegimeSwitchingProposer: Piecewise threshold models adaptive to market volatility states.
  - HawkesOrderFlowProposer: Self-exciting point process modeling order-arrival clustering.
  - RimeProposer: Symbolic search over typed mechanism graphs (native RIME engine),
    emitting closed-form equations rather than coefficient vectors.
  - SINDyProposer: Sparse identification of nonlinear dynamics.
  - ReservoirProposer: Echo state network recurrent dynamics with linear readout.
  - BasisRegressionProposer: Polynomial and RBF basis expansions.
"""
from __future__ import annotations

from typing import Any

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)
from apedp_dyn.proposers.basis_regression_proposer import BasisRegressionProposer
from apedp_dyn.proposers.distributed_lag_interaction_proposer import (
    DistributedLagInteractionProposer,
)
from apedp_dyn.proposers.hawkes_orderflow_proposer import HawkesOrderFlowProposer
from apedp_dyn.proposers.ncm_structural_proposer import (
    MacroConceptArchetype,
    NCMStructuralProposer,
)
from apedp_dyn.proposers.regime_switching_proposer import RegimeSwitchingProposer
from apedp_dyn.proposers.reservoir_proposer import ReservoirProposer
from apedp_dyn.proposers.resiliency_proposer import OrderBookResiliencyProposer
from apedp_dyn.proposers.rime_proposer import RimeProposer, is_rime_available
from apedp_dyn.proposers.sindy_proposer import SINDyProposer
from apedp_dyn.proposers.sparse_microstructure_proposer import SparseMicrostructureProposer
from apedp_dyn.proposers.stem_proposers import (
    CompartmentalSirProposer,
    HamiltonianProposer,
    SolenoidalFluidProposer,
)

RIMEAdaptiveProposer: type[Any] | None = None
try:
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer
except ImportError:  # the RIME extension is optional; the name stays None
    pass
else:
    RIMEAdaptiveProposer = RimeAdProposer

__all__ = [
    "BasisRegressionProposer",
    "CandidateProposal",
    "CompartmentalSirProposer",
    "DistributedLagInteractionProposer",
    "HamiltonianProposer",
    "HawkesOrderFlowProposer",
    "MacroConceptArchetype",
    "NCMStructuralProposer",
    "OrderBookResiliencyProposer",
    "ProposerBase",
    "RIMEAdaptiveProposer",
    "RegimeSwitchingProposer",
    "ReservoirProposer",
    "RimeProposer",
    "SINDyProposer",
    "SolenoidalFluidProposer",
    "SparseMicrostructureProposer",
    "StrategyArchetype",
    "is_rime_available",
]
