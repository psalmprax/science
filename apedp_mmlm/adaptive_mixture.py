"""Track A with weights that adjust themselves: an online mixture of the model's own engines.

Status: experimental

``NonNeuralMMLM_A0`` combines its engines with fixed weights (0.44 automaton,
0.20 tensor train, 0.14 dynamics, 0.10 Koopman, 0.12 Bayesian). On real text
that combination costs more bits than the automaton alone
(``benchmarks/mmlm_text_dev_study.json``): the fixed weights give a say to
engines that predict worse.

``SelfWeightingA0`` wraps a fitted A0 and mixes the same engines with weights
that follow the evidence: after every token, each engine's weight is multiplied
by the probability it gave that token, the weights are renormalised, and a small
share is spread evenly so an engine can come back (the fixed-share rule; the
same update Track S uses per law). No engine is retrained and no weight is
chosen by hand. The weights are reset at the start of each line unless
``carry`` is set.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

ENGINES = ("automaton", "tensor_train", "dynamics", "koopman", "bayesian")


class SelfWeightingA0:
    def __init__(self, model: Any, share: float = 0.02, carry: bool = True) -> None:
        self.model = model
        self.share = float(share)
        self.carry = carry
        self.weights = np.full(len(ENGINES), 1.0 / len(ENGINES))

    def engine_probabilities(self, context_ids: Sequence[int]) -> np.ndarray:
        m = self.model
        rows = [m.automaton.predict_next_proba(context_ids), m.tensor_train.predict_next_proba(context_ids),
                m.dynamics.predict_next_proba(context_ids), m.koopman.predict_next_proba(context_ids, m.dynamics),
                m.bayesian.predictive_mean_and_uncertainty(context_ids)[0]]
        table = np.maximum(np.array(rows, dtype=float), 1e-12)
        out: np.ndarray = table / table.sum(axis=1, keepdims=True)
        return out

    def line_bits(self, ids: Sequence[int]) -> list[float]:
        """Bits for each token of a line after the first, adjusting the weights as it goes."""
        if not self.carry:
            self.weights = np.full(len(ENGINES), 1.0 / len(ENGINES))
        costs = []
        for i in range(1, len(ids)):
            column = self.engine_probabilities(ids[:i])[:, ids[i]]
            mixed = float(self.weights @ column)
            costs.append(-float(np.log2(mixed)))
            posterior = self.weights * column / mixed
            self.weights = (1.0 - self.share) * posterior + self.share / len(ENGINES)
        return costs

    def state(self) -> dict[str, float]:
        return {name: float(w) for name, w in zip(ENGINES, self.weights, strict=True)}
