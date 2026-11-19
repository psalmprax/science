"""A tensor-train engine for the equation language that is actually trained to predict.

Status: experimental

``TensorTrainMemoryEngine.fit_sequences`` adds each context state to the
readout column of the token that followed. That is a count, not a fit: the
logits grow with corpus size and the predictions become over-confident, which
is why the engine cost 375 bits per held-out equation where a uniform code
costs 52 (``benchmarks/equation_language_dev_study.json``).

Here the same tensor-train contraction supplies the context state, and the
readout is fitted for what it is used for: multinomial logistic regression of
the next token on the state, by gradient descent with weight decay. The cores
stay as initialised (a fixed random contraction, as in a reservoir); only the
readout learns.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from apedp_mmlm import equation_language as el
from apedp_mmlm.engines import TensorTrainMemoryEngine


class TensorTrainEquationModel:
    """Same interface as ``EquationLanguageModel``: ``fit(trees)`` and ``bits(tree)``."""

    def __init__(self, bond_dim: int = 16, steps: int = 300, learning_rate: float = 0.5,
                 weight_decay: float = 1e-3, seed: int = 42) -> None:
        self.engine = TensorTrainMemoryEngine(len(el.VOCAB), bond_dim=bond_dim, seed=seed)
        self.steps, self.learning_rate, self.weight_decay = int(steps), float(learning_rate), float(weight_decay)
        self.readout = np.zeros((bond_dim + 1, len(el.VOCAB)))

    @staticmethod
    def _sequence(tree: el.Tree) -> list[int]:
        return [el.ID["<s>"], *(el.ID[t] for t in el.tokens(tree)), el.ID["</s>"]]

    def _state(self, context: Sequence[int]) -> np.ndarray:
        out: np.ndarray = np.append(self.engine.contract_context(context), 1.0)  # bias term
        return out

    def fit(self, trees: Sequence[el.Tree]) -> TensorTrainEquationModel:
        states, targets = [], []
        for tree in trees:
            seq = self._sequence(tree)
            for i in range(1, len(seq)):
                states.append(self._state(seq[:i]))
                targets.append(seq[i])
        S = np.array(states)
        onehot = np.eye(len(el.VOCAB))[targets]
        W = np.zeros((S.shape[1], len(el.VOCAB)))
        for _ in range(self.steps):
            logits = S @ W
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            W -= self.learning_rate * (S.T @ (probs - onehot) / len(S) + self.weight_decay * W)
        self.readout = W
        return self

    def bits(self, tree: el.Tree) -> float:
        seq = self._sequence(tree)
        total = 0.0
        for i in range(1, len(seq)):
            logits = self._state(seq[:i]) @ self.readout
            logits -= logits.max()
            total -= float((logits[seq[i]] - np.log(np.exp(logits).sum())) / np.log(2.0))
        return total
