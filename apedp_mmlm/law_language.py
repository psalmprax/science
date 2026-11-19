"""MMLM-L: the scientific side of MMLM, a language model over laws with a learned mixture of experts.

Status: experimental

:mod:`apedp_mmlm.equation_language` showed that a model of how laws are written
orders a search usefully. It also left three things open, and this module is the
answer to each:

- **The token order depended on variable names.** ``canonical`` sorts the terms
  of every sum and product by shape (size, then operators with variables
  anonymised), so a law reads the same whatever its variables are called.
- **The mixture was a weighted average of whole engines.** Here every expert
  predicts the next token from one view of the partial tree (the parent
  operator, the grandparent, the sibling already written, how many variables
  have appeared, the depth, the previous tokens), and the experts' log
  probabilities are added with weights fitted on held-out laws
  (``p(t) ~ exp(sum_e w_e log p_e(t))``, the same geometric mixing the text
  model :mod:`apedp_mmlm.context_mixing` uses). Weights are kept per position
  class (root, under a function, left or right of an operator), in two sets:
  one fitted with whole domains held out, used for a field with no laws yet,
  and one fitted within domains, used once a field has laws of its own.
- **A new field started from nothing.** Four of the views exist twice: pooled
  over all laws and restricted to the law's domain. A domain expert shrinks its
  counts towards the pooled expert of the same view, so in a domain with no laws
  yet it is the pooled expert, and it departs from it as the domain's laws
  arrive.

It learns the way the earlier model did: ``observe`` takes a law the referee
confirmed, takes one gradient step on the mixing weights against that law (the
self-weight adjustment), then adds the law to the counts. ``retract`` removes a
law that was later refuted and replays the others. Feed it confirmed laws only.

Tokens that cannot occur are never given probability: variables are numbered by
first appearance, so after k distinct variables the only variable tokens allowed
are ``v0`` .. ``vk``.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from itertools import islice, permutations, product
from typing import Any

import numpy as np
from scipy.optimize import minimize

from apedp_mmlm.equation_language import BINARY, MAX_VARS, UNARY, Tree, from_search_node

VOCAB: tuple[str, ...] = (*BINARY, *UNARY, "C", *(f"v{i}" for i in range(MAX_VARS)))
ID = {token: i for i, token in enumerate(VOCAB)}
V = len(VOCAB)
FIRST_VAR = ID["v0"]
VIEWS = ("uni", "seq1", "seq2", "par", "gpar", "sib", "vars", "depth")
EXPERTS = (*VIEWS, *(f"{view}@domain" for view in ("uni", "par", "sib", "vars")))
GROUPS = 4  # root, under a function, left of an operator, right of an operator
COMMUTATIVE = ("+", "*")


# -- canonical form ---------------------------------------------------------------------------
def _size(tree: Tree) -> int:
    return 1 if tree[0] in ("x", "C") else 1 + sum(_size(child) for child in tree[1:])


def _shape(tree: Tree) -> tuple[str, ...]:
    if tree[0] == "x":
        return ("v",)
    if tree[0] == "C":
        return ("C",)
    out: tuple[str, ...] = (tree[0],)
    for child in tree[1:]:
        out += _shape(child)
    return out


def _variables(tree: Tree, into: list[str]) -> list[str]:
    if tree[0] == "x":
        into.append(tree[1])
    elif tree[0] != "C":
        for child in tree[1:]:
            _variables(child, into)
    return into


def _plain_tokens(tree: Tree) -> tuple[int, ...]:
    names: dict[str, int] = {}
    out: list[int] = []

    def walk(node: Tree) -> None:
        if node[0] == "x":
            out.append(FIRST_VAR + names.setdefault(node[1], min(len(names), MAX_VARS - 1)))
        elif node[0] == "C":
            out.append(ID["C"])
        else:
            out.append(ID[node[0]])
            for child in node[1:]:
                walk(child)

    walk(tree)
    return tuple(out)


_CANONICAL: dict[Tree, Tree] = {}


def canonical(tree: Tree, limit: int = 64) -> Tree:
    """The same law with the terms of every sum and product in an order that ignores variable names.

    Terms are sorted by size, then by shape with variables anonymised, then by how
    often their variables occur in the whole law. Terms that are still tied are
    tried in every order (up to ``limit`` candidates in all) and the reading whose
    token sequence is smallest is kept, so the choice does not depend on what the
    variables are called.
    """
    cached = _CANONICAL.get(tree)
    if cached is not None:
        return cached
    occurrences: dict[str, int] = defaultdict(int)
    for name in _variables(tree, []):
        occurrences[name] += 1

    def key(node: Tree) -> tuple[Any, ...]:
        return (_size(node), _shape(node), tuple(occurrences[name] for name in _variables(node, [])))

    def flatten(node: Tree, op: str, into: list[Tree]) -> None:
        if node[0] == op:
            for child in node[1:]:
                flatten(child, op, into)
        else:
            into.append(node)

    def variants(node: Tree) -> list[Tree]:
        if node[0] in ("x", "C"):
            return [node]
        if node[0] not in COMMUTATIVE:
            return [(node[0], *children) for children in islice(product(*(variants(c) for c in node[1:])), limit)]
        parts: list[Tree] = []
        flatten(node, node[0], parts)
        parts.sort(key=key)
        runs: list[list[Tree]] = []
        for part in parts:
            if runs and key(runs[-1][0]) == key(part):
                runs[-1].append(part)
            else:
                runs.append([part])
        out: list[Tree] = []
        for arrangement in islice(product(*(permutations(run) for run in runs)), limit):
            ordered = [part for run in arrangement for part in run]
            for chosen in islice(product(*(variants(part) for part in ordered)), limit):
                folded = chosen[0]
                for part in chosen[1:]:
                    folded = (node[0], folded, part)
                out.append(folded)
                if len(out) >= limit:
                    return out
        return out

    best = min(variants(tree), key=_plain_tokens)
    if len(_CANONICAL) < 500_000:
        _CANONICAL[tree] = best
    return best


def law_tokens(tree: Tree) -> list[str]:
    """Prefix tokens of the canonical form, variables numbered by first appearance."""
    return [token for token, _context in _walk(canonical(tree))]


def _walk(tree: Tree) -> list[tuple[str, dict[str, tuple[Any, ...]]]]:
    """(token, the context each view sees before the token is written), in prefix order."""
    names: dict[str, str] = {}
    out: list[tuple[str, dict[str, tuple[Any, ...]]]] = []
    written: list[str] = []

    def head(node: Tree) -> str:
        return "v" if node[0] == "x" else node[0]

    def visit(node: Tree, parent: str, index: int, grand: str, grand_index: int, sibling: str, depth: int) -> None:
        group = 0 if parent == "root" else 1 if parent in UNARY else 2 + index
        context = {
            "uni": (),
            "seq1": tuple(written[-1:]),
            "seq2": tuple(written[-2:]),
            "par": (parent, index),
            "gpar": (grand, grand_index, parent, index),
            "sib": (parent, index, sibling),
            "vars": (len(names), parent, index),
            "depth": (min(depth, 6), parent),
            "_group": (group,),
            "_nvars": (len(names),),
        }
        if node[0] == "x":
            symbol = names.setdefault(node[1], f"v{min(len(names), MAX_VARS - 1)}")
        else:
            symbol = node[0]
        out.append((symbol, context))
        written.append(symbol)
        if node[0] not in ("x", "C"):
            for k, child in enumerate(node[1:]):
                visit(child, node[0], k, parent, index, head(node[1]) if k else "-", depth + 1)

    visit(tree, "root", 0, "root", 0, "-", 0)
    return out


# -- counting experts -------------------------------------------------------------------------
class _Counts:
    """Next-token counts per (view, context); a context's estimate is shrunk towards the unigram."""

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha, self.beta = alpha, beta
        self.table: dict[tuple[Any, ...], np.ndarray] = {}
        self.unigram = np.zeros(V)

    def add(self, walked: list[tuple[str, dict[str, tuple[Any, ...]]]], domain: str | None, sign: float = 1.0) -> None:
        for written, context in walked:
            t = ID[written]
            self.unigram[t] += sign
            for view in VIEWS:
                self._bump((view, None, context[view]), t, sign)
                if domain is not None and f"{view}@domain" in EXPERTS:
                    self._bump((view, domain, context[view]), t, sign)

    def _bump(self, key: tuple[Any, ...], t: int, sign: float) -> None:
        row = self.table.get(key)
        if row is None:
            row = self.table[key] = np.zeros(V)
        row[t] += sign

    def base(self) -> np.ndarray:
        out: np.ndarray = (self.unigram + 0.5) / (self.unigram.sum() + 0.5 * V)
        return out

    def log_probabilities(self, context: dict[str, tuple[Any, ...]], domain: str | None) -> np.ndarray:
        """``[len(EXPERTS), V]`` log probabilities.

        A pooled expert shrinks its context's counts towards the unigram. A domain
        expert shrinks the domain's counts towards the pooled expert of the same
        view, so with no laws from the domain it is the pooled expert, and it moves
        away from it only as the domain supplies evidence.
        """
        base = self.base()
        pooled: dict[str, np.ndarray] = {}
        out = np.zeros((len(EXPERTS), V))
        for e, expert in enumerate(EXPERTS):
            view, _, scoped = expert.partition("@")
            if not scoped:
                row = self.table.get((view, None, context[view]))
                pooled[view] = base if row is None else (row + self.alpha * base) / (row.sum() + self.alpha)
                out[e] = np.log(pooled[view])
            else:
                row = self.table.get((view, domain, context[view])) if domain is not None else None
                out[e] = np.log(pooled[view] if row is None else (row + self.beta * pooled[view]) / (row.sum() + self.beta))
        return out


def _allowed(n_vars: int) -> np.ndarray:
    mask: np.ndarray = np.ones(V, dtype=bool)
    mask[FIRST_VAR + min(n_vars, MAX_VARS - 1) + 1 :] = False
    return mask


def _mix(weights: np.ndarray, logp: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Log distribution over the vocabulary from one weight row and the experts' log probabilities."""
    z = weights @ logp
    z = np.where(allowed, z, -np.inf)
    out: np.ndarray = z - np.logaddexp.reduce(z[allowed])
    return out


class LawLanguageModel:
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, folds: int = 10, ridge: float = 1e-3, step: float = 0.02,
                 settle: float = 8.0) -> None:
        self.alpha, self.beta, self.folds, self.ridge, self.step, self.settle = alpha, beta, folds, ridge, step, settle
        self.counts = _Counts(alpha, beta)
        # weights[0]: for a domain with no laws yet; weights[1]: for a domain with many
        self.weights = np.full((2, GROUPS, len(EXPERTS)), 1.0 / len(VIEWS))
        self._fitted_weights = self.weights.copy()
        self._domain_laws: dict[str, int] = defaultdict(int)
        self._seed: list[tuple[Tree, str | None]] = []
        self._observed: list[tuple[Tree, str | None]] = []
        self._cache: dict[tuple[Any, ...], np.ndarray] = {}

    # -- fitting ----------------------------------------------------------------------------
    def fit(self, trees: Sequence[Tree], domains: Sequence[str | None] | None = None) -> LawLanguageModel:
        """Count the corpus and fit two sets of mixing weights on held-out laws.

        The set for an established domain holds laws out fold by fold with their
        domain known. The set for a new domain holds out one whole domain at a
        time and scores it as unseen (when the corpus has fewer than three
        domains, fold by fold with the domain hidden). Scoring blends the two by
        how many laws the domain already has (``_blend``).
        """
        labels: list[str | None] = list(domains) if domains is not None else [None] * len(trees)
        walked = [_walk(canonical(tree)) for tree in trees]
        folds = min(self.folds, len(trees))
        by_index = [i % folds for i in range(len(trees))]
        names = sorted({label for label in labels if label is not None})
        by_domain = [names.index(label) if label is not None else len(names) for label in labels]
        whole_domains = len(set(by_domain)) >= 3
        known = self._held_out_rows(walked, labels, by_index, hide_domain=False)
        new = self._held_out_rows(walked, labels, by_domain if whole_domains else by_index, hide_domain=True)
        if folds >= 2 and known and new:
            self.weights = np.stack([self._fit_weights(new), self._fit_weights(known)])
        self._fitted_weights = self.weights.copy()
        self.counts = _Counts(self.alpha, self.beta)
        self._domain_laws = defaultdict(int)
        for w, label in zip(walked, labels, strict=True):
            self.counts.add(w, label)
            if label is not None:
                self._domain_laws[label] += 1
        self._seed = list(zip(trees, labels, strict=True))
        self._observed = []
        self._cache.clear()
        return self

    def _held_out_rows(self, walked: list[Any], labels: list[str | None], fold_of: list[int],
                       hide_domain: bool) -> list[tuple[np.ndarray, int, int, int]]:
        rows: list[tuple[np.ndarray, int, int, int]] = []
        for fold in sorted(set(fold_of)):
            part = _Counts(self.alpha, self.beta)
            for i, w in enumerate(walked):
                if fold_of[i] != fold:
                    part.add(w, labels[i])
            for i, w in enumerate(walked):
                if fold_of[i] == fold:
                    label = None if hide_domain else labels[i]
                    for written, context in w:
                        rows.append((part.log_probabilities(context, label), context["_group"][0], context["_nvars"][0], ID[written]))
        return rows

    def _blend(self, domain: str | None) -> np.ndarray:
        """Mixing weights for a domain: the new-domain set, moving to the established set as its laws accumulate."""
        n = self._domain_laws.get(domain, 0) if domain is not None else 0
        share = n / (n + self.settle)
        out: np.ndarray = (1.0 - share) * self.weights[0] + share * self.weights[1]
        return out

    def _fit_weights(self, rows: list[tuple[np.ndarray, int, int, int]]) -> np.ndarray:
        logp = np.stack([r[0] for r in rows])
        group = np.array([r[1] for r in rows])
        allowed = np.stack([_allowed(r[2]) for r in rows])
        target = np.array([r[3] for r in rows])
        start = np.full((GROUPS, len(EXPERTS)), 1.0 / len(VIEWS))

        def loss(flat: np.ndarray) -> tuple[float, np.ndarray]:
            w = flat.reshape(start.shape)
            z = np.einsum("te,tev->tv", w[group], logp)
            z = np.where(allowed, z, -np.inf)
            z -= np.logaddexp.reduce(z, axis=1, keepdims=True)
            p = np.exp(z)
            value = -z[np.arange(len(rows)), target].sum() + self.ridge * len(rows) * ((w - start) ** 2).sum()
            expected = np.einsum("tv,tev->te", p, logp)
            per_row = expected - logp[np.arange(len(rows)), :, target]
            grad = np.zeros_like(w)
            np.add.at(grad, group, per_row)
            grad += 2 * self.ridge * len(rows) * (w - start)
            return float(value), grad.ravel()

        result = minimize(loss, start.ravel(), jac=True, method="L-BFGS-B", bounds=[(0.0, 4.0)] * start.size)
        out: np.ndarray = result.x.reshape(start.shape)
        return out

    # -- scoring ----------------------------------------------------------------------------
    def token_bits(self, tree: Tree, domain: str | None = None) -> list[float]:
        out = []
        weights = self._blend(domain)
        for token, context in _walk(canonical(tree)):
            key = (domain, *(context[view] for view in VIEWS))
            dist = self._cache.get(key)
            if dist is None:
                dist = _mix(weights[context["_group"][0]], self.counts.log_probabilities(context, domain), _allowed(context["_nvars"][0]))
                self._cache[key] = dist
            out.append(float(-dist[ID[token]] / np.log(2.0)))
        return out

    def bits(self, tree: Tree, domain: str | None = None) -> float:
        """Code length of a law in bits."""
        return float(sum(self.token_bits(tree, domain)))

    # -- learning from confirmed laws -------------------------------------------------------
    def _adjust(self, tree: Tree, domain: str | None) -> None:
        n = self._domain_laws.get(domain, 0) if domain is not None else 0
        share = n / (n + self.settle)
        for written, context in _walk(canonical(tree)):
            g = context["_group"][0]
            logp = self.counts.log_probabilities(context, domain)
            dist = np.exp(_mix(self._blend(domain)[g], logp, _allowed(context["_nvars"][0])))
            gradient = logp @ dist - logp[:, ID[written]]
            self.weights[0, g] = np.clip(self.weights[0, g] - self.step * (1.0 - share) * gradient, 0.0, 4.0)
            self.weights[1, g] = np.clip(self.weights[1, g] - self.step * share * gradient, 0.0, 4.0)

    def observe(self, tree: Tree, domain: str | None = None) -> float:
        """Adjust the weights for a confirmed law, then count it. Returns the bits it cost before learning."""
        cost = self.bits(tree, domain)
        self._adjust(tree, domain)
        self.counts.add(_walk(canonical(tree)), domain)
        if domain is not None:
            self._domain_laws[domain] += 1
        self._observed.append((tree, domain))
        self._cache.clear()
        return cost

    def retract(self, tree: Tree, domain: str | None = None) -> bool:
        """Unlearn a law that was observed and later refuted; replays the others from the fitted weights."""
        if (tree, domain) not in self._observed:
            return False
        remaining = list(self._observed)
        remaining.remove((tree, domain))
        self.counts = _Counts(self.alpha, self.beta)
        self._domain_laws = defaultdict(int)
        for seeded, label in self._seed:
            self.counts.add(_walk(canonical(seeded)), label)
            if label is not None:
                self._domain_laws[label] += 1
        self.weights = self._fitted_weights.copy()
        self._observed = []
        self._cache.clear()
        for kept, label in remaining:
            self.observe(kept, label)
        return True

    def state(self) -> dict[str, Any]:
        """The weights and corpus sizes, with a SHA-256 of both for the audit trail."""
        body = {"weights": {which: {f"group{g}": {e: round(float(self.weights[k, g, i]), 6) for i, e in enumerate(EXPERTS)}
                                    for g in range(GROUPS)} for k, which in enumerate(("new_domain", "established_domain"))},
                "seed_laws": len(self._seed), "observed_laws": len(self._observed)}
        return {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}


def search_prior(model: LawLanguageModel, domain: str | None = None) -> Any:
    """``structure_bits`` for the exhaustive and guided searches: the model's code length of a search node."""
    def structure_bits(node: Tree) -> float:
        return model.bits(from_search_node(node), domain)
    return structure_bits
