"""Common Multi-Proposer Interface (v2).

Defines the decoupled boundary between hypothesis generation and epistemic adjudication.
Independent proposers (symbolic search, sparse regression, event-queue models, etc.)
submit CandidateProposal instances conforming to a strict StrategyArchetype contract
without controlling their own acceptance or promotion rules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class StrategyArchetype(str, Enum):
    """Declared strategy archetype governing operational exposure and statistical power constraints."""
    TAIL_RISK = "tail_risk"                # Rare-event / crisis alpha: low active rate permitted, requires N_events >= 30
    CONTINUOUS_FACTOR = "continuous_factor"  # Broad cross-sectional factor: active >= 85%, constrained turnover
    TACTICAL = "tactical"                  # Regime-switching / swing: active in [5%, 90%]
    # STEM discovery archetypes: physical laws are not trading signals. Forcing
    # Hamiltonian, fluid, or quantum candidates through financial active-rate
    # and turnover screening misclassified conservation-of-energy style
    # invariants as ``continuous_factor`` and screened them out on gates that
    # have no meaning for a dynamical law.
    PHYSICAL_INVARIANT = "physical_invariant"    # Conservation / bound invariants: gated on declared invariants, not active fraction
    DYNAMICAL_SYSTEM = "dynamical_system"        # ODE/PDE laws: gated on rollout stability + holdout R^2, not turnover


@dataclass(frozen=True)
class CandidateProposal:
    """Formal candidate proposal emitted by an independent proposer.

    Note on ``declared_scale_invariant`` (declarative, non-enforced): as of the
    v1.4 frozen record, NO gate, referee, or ledger stage consumes this field.
    It records a proposer's *intent* that the emitted relation be invariant
    under a change of input units (e.g. via coefficient refit); it is never
    verified automatically and must not be cited as a property of the
    candidate. Any future enforcement requires a canonical preregistered spec
    change (AGENTS.md §1.3).
    """
    proposer_id: str
    candidate_id: str
    formula: str
    strategy_archetype: StrategyArchetype
    declared_scale_invariant: bool
    declared_nominal_dependencies: tuple[str, ...] = ()
    compute_score: Callable[[pd.DataFrame], np.ndarray] = field(default=lambda df: np.zeros(len(df)), repr=False)
    # Heterogeneous by contract: numeric fit parameters plus predictive-status
    # strings such as "identifiability_status" (readers cast explicitly).
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProposerBase(ABC):
    """Abstract base class for all competing hypothesis generators."""

    @property
    @abstractmethod
    def proposer_id(self) -> str:
        """Unique identifier of the proposer engine."""
        pass

    @abstractmethod
    def fit_and_propose(self, *args: Any, **kwargs: Any) -> Sequence[CandidateProposal]:
        """Generate candidate proposals from training data without self-adjudication.

        The abstract contract is intentionally call-shape-agnostic: concrete
        proposers consume different data layouts (a dict of frames, a single
        frame, a mapping of arrays) and own their own required knobs. Only the
        emission type is contractual (AGENTS.md §1.1); acceptance never lives
        here.
        """
        raise NotImplementedError
