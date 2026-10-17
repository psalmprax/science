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


@dataclass(frozen=True)
class CandidateProposal:
    """Formal candidate proposal emitted by an independent proposer."""
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
