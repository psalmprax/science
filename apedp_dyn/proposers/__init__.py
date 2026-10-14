"""Competing Proposers Package.

Provides diverse, independent hypothesis generators for the APEDP-X discovery pipeline:
  - SparseMicrostructureProposer: L1/ElasticNet regularized linear microstructure signals.
  - OrderBookResiliencyProposer: Event-driven exponential liquidity recovery signals.
  - DistributedLagInteractionProposer: Rich distributed-lag kernels and bilinear interactions.
  - RegimeSwitchingProposer: Piecewise threshold models adaptive to market volatility states.
  - HawkesOrderFlowProposer: Self-exciting point process modeling order-arrival clustering.
  - RimeProposer: Symbolic search over typed mechanism graphs (native RIME engine),
    emitting closed-form equations rather than coefficient vectors.
"""
from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)
from apedp_dyn.proposers.distributed_lag_interaction_proposer import (
    DistributedLagInteractionProposer,
)
from apedp_dyn.proposers.hawkes_orderflow_proposer import HawkesOrderFlowProposer
from apedp_dyn.proposers.regime_switching_proposer import RegimeSwitchingProposer
from apedp_dyn.proposers.resiliency_proposer import OrderBookResiliencyProposer
from apedp_dyn.proposers.rime_proposer import RimeProposer, is_rime_available
from apedp_dyn.proposers.sparse_microstructure_proposer import SparseMicrostructureProposer
from apedp_dyn.proposers.stem_proposers import (
    CompartmentalSirProposer,
    HamiltonianProposer,
    SolenoidalFluidProposer,
)

__all__ = [
    "CandidateProposal",
    "CompartmentalSirProposer",
    "DistributedLagInteractionProposer",
    "HamiltonianProposer",
    "HawkesOrderFlowProposer",
    "OrderBookResiliencyProposer",
    "ProposerBase",
    "RegimeSwitchingProposer",
    "RimeProposer",
    "SolenoidalFluidProposer",
    "SparseMicrostructureProposer",
    "StrategyArchetype",
    "is_rime_available",
]
