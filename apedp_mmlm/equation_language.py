"""MMLM Track S: a language model over equations, used as a prior for law discovery.

Status: experimental

The other tracks model sentences. This one applies the same engines to the
language science is written in: an expression is a short sentence over a small
vocabulary (operators, functions, variables), and a model of that language says
how plausible a candidate law is before any data is seen. That number has two
direct uses in the discovery pipeline: it orders a search (try plausible forms
first) and it replaces a flat complexity penalty with a learned code length.

Pieces:

- :func:`to_tree` / :func:`tokens`: an expression as a tree in the dialect the
  exhaustive search enumerates (``+ - * /``, ``sq sqrt inv exp log sin cos``),
  read in prefix order, with variables renamed ``v0, v1, ...`` by first
  appearance and numeric constants reduced to one token, so the model learns
  shapes and not the names or values in a particular law.
- :class:`EquationLanguageModel`: one of the package's engines
  (``automaton``: Kneser-Ney weighted automaton; ``bayesian``: hierarchical
  Dirichlet; ``tensor_train``) or one of two references defined here
  (``unigram``; ``tree``: each token given its parent operator and its position
  under it, the natural context in a tree). ``bits(tokens)`` is the code length
  of an expression.

What it is measured on, and by what, is in
``scripts/study_equation_language.py``: bits per held-out equation, and how
early a held-out law appears when the exhaustive search is ordered by the
model instead of by size.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

from apedp_mmlm.engines import (
    BayesianLanguageEngine,
    TensorTrainMemoryEngine,
    WeightedAutomatonEngine,
)

Tree = tuple[Any, ...]
BINARY = ("+", "-", "*", "/")
#: "pow" is the catch-all: a power outside the dialect, or a function with no token of its own.
UNARY = ("sq", "sqrt", "inv", "exp", "log", "sin", "cos", "tanh", "asin", "acos", "pow")
MAX_VARS = 10
VOCAB: tuple[str, ...] = ("<s>", "</s>", *BINARY, *UNARY, "C", *(f"v{i}" for i in range(MAX_VARS)))
ID = {token: i for i, token in enumerate(VOCAB)}
ENGINES = ("uniform", "unigram", "tree", "automaton", "bayesian", "tensor_train")
_FUNCTIONS = {sp.exp: "exp", sp.log: "log", sp.sin: "sin", sp.cos: "cos", sp.tanh: "tanh", sp.asin: "asin", sp.acos: "acos"}


def _fold(op: str, parts: list[Tree]) -> Tree:
    out = parts[0]
    for part in parts[1:]:
        out = (op, out, part)
    return out


def _power(base: Tree, exponent: sp.Expr) -> Tree:
    if exponent == 2:
        return ("sq", base)
    if exponent == sp.Rational(1, 2):
        return ("sqrt", base)
    if exponent == 3:
        return ("*", ("sq", base), base)
    if exponent == 4:
        return ("sq", ("sq", base))
    if exponent == sp.Rational(3, 2):
        return ("*", base, ("sqrt", base))
    return base if exponent == 1 else ("pow", base)


def to_tree(expr: sp.Expr) -> Tree:
    """A sympy expression in the search's dialect. Numeric factors and coefficients are dropped
    (the search fits scale and offset); a number that cannot be dropped becomes ``("C",)``."""
    if expr.is_Symbol:
        return ("x", str(expr))
    if expr.is_number:
        return ("C",)
    if expr.is_Add:
        plus: list[Tree] = []
        minus: list[Tree] = []
        for term in expr.args:
            coeff, rest = term.as_coeff_Mul()
            (minus if coeff < 0 else plus).append(to_tree(rest) if rest != 1 else ("C",))
        if not plus:
            plus, minus = minus, []
        tree = _fold("+", plus)
        for part in minus:
            tree = ("-", tree, part)
        return tree
    if expr.is_Mul:
        _coeff, rest = expr.as_coeff_Mul()
        top: list[Tree] = []
        bottom: list[Tree] = []
        for factor in sp.Mul.make_args(rest):
            base, exponent = factor.as_base_exp()
            if exponent.is_number and exponent < 0:
                bottom.append(_power(to_tree(base), -exponent))
            else:
                top.append(_power(to_tree(base), exponent) if exponent.is_number else ("pow", to_tree(base)))
        if not bottom:
            return _fold("*", top)
        denominator = _fold("*", bottom)
        return ("/", _fold("*", top), denominator) if top else ("inv", denominator)
    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if exponent.is_number and exponent < 0:
            return ("inv", _power(to_tree(base), -exponent))
        return _power(to_tree(base), exponent) if exponent.is_number else ("pow", to_tree(base))
    name = _FUNCTIONS.get(type(expr))
    if name is not None and len(expr.args) == 1:
        return (name, to_tree(expr.args[0]))
    if isinstance(expr, (sp.tan, sp.cot)):
        inner = to_tree(expr.args[0])
        ratio = (("sin", inner), ("cos", inner)) if isinstance(expr, sp.tan) else (("cos", inner), ("sin", inner))
        return ("/", *ratio)
    if len(expr.args) == 1:
        return ("pow", to_tree(expr.args[0]))  # any other one-argument function shares the catch-all token
    raise ValueError(f"no token for {type(expr).__name__}")


def from_search_node(node: Tree) -> Tree:
    """An ``exhaustive_search`` node (variables as column indices) as a tree here."""
    if node[0] == "x":
        return ("x", f"x{node[1]}")
    return (node[0], *(from_search_node(child) for child in node[1:]))


def tokens(tree: Tree) -> list[str]:
    """Prefix order, variables renamed by first appearance, without the start and end markers."""
    names: dict[str, str] = {}
    out: list[str] = []

    def walk(node: Tree) -> None:
        if node[0] == "x":
            out.append(names.setdefault(node[1], f"v{min(len(names), MAX_VARS - 1)}"))
        elif node[0] == "C":
            out.append("C")
        else:
            out.append(node[0])
            for child in node[1:]:
                walk(child)

    walk(tree)
    return out


def tree_contexts(tree: Tree) -> list[tuple[str, str]]:
    """(context, token) pairs where the context is the parent operator and the child's position."""
    flat = tokens(tree)
    pairs: list[tuple[str, str]] = []
    position = 0

    def walk(node: Tree, context: str) -> None:
        nonlocal position
        pairs.append((context, flat[position]))
        position += 1
        if node[0] not in ("x", "C"):
            for k, child in enumerate(node[1:]):
                walk(child, f"{node[0]}:{k}")

    walk(tree, "root")
    return pairs


class EquationLanguageModel:
    """Code lengths of expressions under one engine."""

    def __init__(self, engine: str = "automaton", smoothing: float = 0.5) -> None:
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        self.engine = engine
        self.smoothing = float(smoothing)
        self._model: Any = None
        self._counts: dict[str, np.ndarray] = {}
        self._cache: dict[tuple[int, ...], np.ndarray] = {}

    def fit(self, trees: Sequence[Tree]) -> EquationLanguageModel:
        sequences = [[ID["<s>"], *(ID[t] for t in tokens(tree)), ID["</s>"]] for tree in trees]
        self._cache.clear()
        if self.engine in ("unigram", "tree"):
            self._counts = {}
            for tree, seq in zip(trees, sequences, strict=True):
                pairs = tree_contexts(tree) if self.engine == "tree" else [("all", VOCAB[i]) for i in seq[1:-1]]
                for context, token in pairs:
                    self._counts.setdefault(context, np.zeros(len(VOCAB)))[ID[token]] += 1.0
        elif self.engine == "automaton":
            self._model = WeightedAutomatonEngine(len(VOCAB))
            self._model.fit_sequences(sequences)
        elif self.engine == "bayesian":
            self._model = BayesianLanguageEngine(len(VOCAB))
            self._model.fit_sequences(sequences)
        elif self.engine == "tensor_train":
            self._model = TensorTrainMemoryEngine(len(VOCAB), bond_dim=8)
            for _ in range(5):
                self._model.fit_sequences(sequences)
        return self

    def _table(self, context: str) -> np.ndarray:
        counts = self._counts.get(context)
        if counts is None:
            counts = sum(self._counts.values(), np.zeros(len(VOCAB)))  # back off to all contexts pooled
        smoothed = counts + self.smoothing
        out: np.ndarray = smoothed / smoothed.sum()
        return out

    def _next(self, context: tuple[int, ...]) -> np.ndarray:
        cached = self._cache.get(context)
        if cached is None:
            if self.engine == "bayesian":
                cached = np.asarray(self._model.predictive_mean_and_uncertainty(list(context))[0], dtype=float)
            else:
                cached = np.asarray(self._model.predict_next_proba(list(context)), dtype=float)
            cached = np.maximum(cached, 1e-12) / np.maximum(cached, 1e-12).sum()
            self._cache[context] = cached
        return cached

    def bits(self, tree: Tree) -> float:
        """Code length of the expression in bits (structure only; every engine is charged the same way).

        Sequence engines code the prefix string and its end marker. The tree
        and unigram references code each token; a tree's shape ends itself, so
        they need no end marker.
        """
        flat = tokens(tree)
        if self.engine == "uniform":
            return len(flat) * float(np.log2(len(VOCAB) - 2))
        if self.engine == "unigram":
            table = self._table("all")
            return float(-sum(np.log2(table[ID[t]]) for t in flat))
        if self.engine == "tree":
            return float(-sum(np.log2(self._table(context)[ID[token]]) for context, token in tree_contexts(tree)))
        seq = (ID["<s>"], *(ID[t] for t in flat), ID["</s>"])
        return float(-sum(np.log2(self._next(seq[:i])[seq[i]]) for i in range(1, len(seq))))


def search_prior(model: EquationLanguageModel) -> Any:
    """``structure_bits`` for ``apedp_dyn.discovery_l2.exhaustive_search``: the model's code length of a search node."""
    def structure_bits(node: Tree) -> float:
        return model.bits(from_search_node(node))

    return structure_bits


class SelfAdjustingEquationModel:
    """A mixture of engines that re-weights and re-trains itself on every confirmed law.

    For each law the caller reports as confirmed (:meth:`observe`):

    1. **Weight adjustment.** Each engine's weight is multiplied by the
       probability it gave that law, ``2 ** -bits``, and the weights are
       renormalised. This is Bayesian model averaging; an engine that keeps
       predicting confirmed laws badly loses its say. A small share
       (``share``) is then spread evenly, so an engine that was wrong early can
       recover if the laws change character (fixed-share, Herbster & Warmuth).
    2. **Learning.** The law joins the corpus and every engine is refitted.

    :meth:`retract` removes a law that was later refuted and refits (the weight
    history is replayed without it), so a mistake fed in can be taken out.

    Discipline: it learns only from what it is told was confirmed. Feeding it
    its own proposals would let it grow confident in whatever it already
    believed; the caller must pass referee- or contrast-confirmed laws only.
    Every update is deterministic, and :meth:`state` returns the weights, the
    corpus size and a SHA-256 of both for the audit trail.
    """

    def __init__(self, engines: Sequence[str] = ("unigram", "tree", "automaton", "bayesian"),
                 share: float = 0.05) -> None:
        if not engines or any(e not in ENGINES for e in engines):
            raise ValueError(f"engines must be drawn from {ENGINES}")
        if not 0.0 <= share < 1.0:
            raise ValueError("share must be in [0, 1)")
        self.engines = tuple(engines)
        self.share = float(share)
        self._seed: list[Tree] = []
        self._observed: list[Tree] = []
        self._models: dict[str, EquationLanguageModel] = {}
        self.weights = np.full(len(self.engines), 1.0 / len(self.engines))
        self._refit()

    def _refit(self) -> None:
        corpus = [*self._seed, *self._observed]
        self._models = {e: EquationLanguageModel(e).fit(corpus) for e in self.engines}

    def fit(self, trees: Sequence[Tree]) -> SelfAdjustingEquationModel:
        """Start from a corpus (no weight adjustment: these laws were not predicted, only read)."""
        self._seed = list(trees)
        self._observed = []
        self.weights = np.full(len(self.engines), 1.0 / len(self.engines))
        self._refit()
        return self

    def engine_bits(self, tree: Tree) -> np.ndarray:
        out: np.ndarray = np.array([self._models[e].bits(tree) for e in self.engines])
        return out

    def bits(self, tree: Tree) -> float:
        """Code length under the weighted mixture."""
        per_engine = self.engine_bits(tree)
        floor = float(per_engine.min())
        return floor - float(np.log2(np.sum(self.weights * 2.0 ** (floor - per_engine))))

    def _adjust(self, tree: Tree) -> None:
        per_engine = self.engine_bits(tree)
        posterior = self.weights * 2.0 ** (float(per_engine.min()) - per_engine)
        posterior /= posterior.sum()
        self.weights = (1.0 - self.share) * posterior + self.share / len(self.engines)

    def observe(self, tree: Tree) -> float:
        """Adjust the weights for a confirmed law, then learn it. Returns the bits it cost before learning."""
        cost = self.bits(tree)
        self._adjust(tree)
        self._observed.append(tree)
        self._refit()
        return cost

    def retract(self, tree: Tree) -> bool:
        """Unlearn a law that was observed and later refuted; replays the weight history without it."""
        if tree not in self._observed:
            return False
        remaining = list(self._observed)
        remaining.remove(tree)
        self._observed = []
        self.weights = np.full(len(self.engines), 1.0 / len(self.engines))
        self._refit()
        for kept in remaining:
            self.observe(kept)
        return True

    def state(self) -> dict[str, Any]:
        import hashlib

        weights = {e: float(w) for e, w in zip(self.engines, self.weights, strict=True)}
        payload = repr((sorted(weights.items()), [tokens(t) for t in (*self._seed, *self._observed)]))
        return {"weights": weights, "seed_laws": len(self._seed), "observed_laws": len(self._observed),
                "sha256": hashlib.sha256(payload.encode()).hexdigest()}
