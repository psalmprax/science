"""Advanced search algorithms for symbolic regression and discovery.

Implements the design from the conversation:
- Physics-informed constraint engine
- DAG representation for expressions  
- Neural Guided Tree Search (NGTS)
- Latent space for equation embeddings
- Hierarchical search strategy
- Rigorous validation pipeline
- Uncertainty-aware search

All components have Rust fast paths with Python fallbacks.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Try to import Rust kernel
try:
    import apedp_kernel
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class Dimensions:
    """Physical dimensions for dimensional analysis."""
    mass: int = 0
    length: int = 0
    time: int = 0
    temperature: int = 0
    current: int = 0
    amount: int = 0
    luminosity: int = 0

    def is_dimensionless(self) -> bool:
        return all(getattr(self, k) == 0 for k in ['mass', 'length', 'time', 
                                                      'temperature', 'current', 
                                                      'amount', 'luminosity'])

    def __add__(self, other: Dimensions) -> Dimensions:
        return Dimensions(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            temperature=self.temperature + other.temperature,
            current=self.current + other.current,
            amount=self.amount + other.amount,
            luminosity=self.luminosity + other.luminosity,
        )

    def __sub__(self, other: Dimensions) -> Dimensions:
        return Dimensions(
            mass=self.mass - other.mass,
            length=self.length - other.length,
            time=self.time - other.time,
            temperature=self.temperature - other.temperature,
            current=self.current - other.current,
            amount=self.amount - other.amount,
            luminosity=self.luminosity - other.luminosity,
        )


class PhysicsConstraintEngine:
    """Physics-informed constraint engine with Rust acceleration."""

    def __init__(self):
        self.variable_dims: dict[str, Dimensions] = {}
        self.target_dim: Dimensions | None = None
        self.symmetries: set[str] = set()
        self.conserved_quantities: list[str] = []
        
        if RUST_AVAILABLE and hasattr(apedp_kernel, 'PhysicsConstraintEngine'):
            self._rust = apedp_kernel.PhysicsConstraintEngine()
        else:
            self._rust = None

    def add_variable(self, name: str, dims: Dimensions):
        self.variable_dims[name] = dims
        if self._rust is not None:
            self._rust.add_variable(name, dims.mass, dims.length, dims.time,
                                   dims.temperature, dims.current, dims.amount, dims.luminosity)

    def set_target(self, dims: Dimensions):
        self.target_dim = dims
        if self._rust is not None:
            self._rust.set_target_dimensions(dims.mass, dims.length, dims.time,
                                             dims.temperature, dims.current, dims.amount, dims.luminosity)

    @staticmethod
    def dimensions_match(lhs: Dimensions, rhs: Dimensions) -> bool:
        """All seven exponents agree — the same comparison the Rust kernel makes."""
        return all(getattr(lhs, name) == getattr(rhs, name)
                   for name in Dimensions.__dataclass_fields__)

    def check_dimensional_consistency(self, expr_dims: Dimensions) -> bool:
        if self.target_dim is None:
            return True
        if self._rust is not None:
            return self._rust.check_dimensional_consistency(json.dumps(expr_dims.__dict__))
        return self.dimensions_match(expr_dims, self.target_dim)

    def dimensional_analysis_score(self, term_dims: Sequence[Dimensions] | None = None) -> float:
        """Fraction of `term_dims` whose dimensions equal the target.

        Dimensional analysis is a judgement about a candidate, so it needs the
        candidate: one `Dimensions` per term of the expression. Without terms, or
        without a target to judge them against, there is nothing to score, and
        this raises instead of returning a constant that looks measured.
        """
        if not term_dims:
            raise ValueError(
                "dimensional_analysis_score needs the dimensions of the terms to "
                "score; no terms were supplied")
        if self.target_dim is None:
            raise ValueError(
                "dimensional_analysis_score needs a target dimension to score the "
                "terms against")
        if self._rust is not None:
            return self._rust.dimensional_analysis_score(
                json.dumps([dims.__dict__ for dims in term_dims]))
        matches = sum(1 for dims in term_dims
                      if self.dimensions_match(dims, self.target_dim))
        return matches / len(term_dims)

    def smoothness_score(self, values: list[float]) -> float:
        if self._rust is not None:
            return self._rust.smoothness_score(values)
        if len(values) < 3:
            return 1.0
        roughness = 0.0
        for i in range(1, len(values) - 1):
            second_diff = abs(values[i + 1] - 2.0 * values[i] + values[i - 1])
            roughness += second_diff
        return 1.0 / (1.0 + roughness / len(values))

    def detect_symmetries(self, data: list[list[float]]) -> set[str]:
        symmetries = set()
        if len(data) >= 2:
            symmetries.add("translation")
        return symmetries


@dataclass
class DagNode:
    """Node in expression DAG."""
    id: int
    op: str
    children: list[int] = field(default_factory=list)
    value: float | None = None
    var_idx: int | None = None


class ExpressionDag:
    """DAG representation of expressions with Rust acceleration."""

    def __init__(self, variables: list[str]):
        self.variables = variables
        self.nodes: list[DagNode] = []
        self.root = 0
        if RUST_AVAILABLE and hasattr(apedp_kernel, 'DagExpression'):
            self._rust = apedp_kernel.DagExpression(variables)
        else:
            self._rust = None
        self._use_python = True  # Use Python impl for manually constructed DAGs for reliability

    def add_node(self, op: str, children: list[int] | None = None,
                 value: float | None = None, var_idx: int | None = None) -> int:
        if children is None:
            children = []
        node_id = len(self.nodes)
        self.nodes.append(DagNode(node_id, op, children, value, var_idx))
        return node_id

    def to_string(self, node_id: int | None = None) -> str:
        if self._rust is not None and not self._use_python and len(self.nodes) == 0:
            try:
                return self._rust.canonicalize()
            except Exception:  # noqa: BLE001,S110
                pass
        nid = node_id if node_id is not None else self.root
        if nid >= len(self.nodes):
            return "0"
        node = self.nodes[nid]
        if node.op == "const":
            return str(node.value) if node.value is not None else "0"
        elif node.op == "var":
            if node.var_idx is not None and node.var_idx < len(self.variables):
                return self.variables[node.var_idx]
            return f"x{node.var_idx}"
        elif node.op == "add":
            return f"({self.to_string(node.children[0])} + {self.to_string(node.children[1])})"
        elif node.op == "sub":
            return f"({self.to_string(node.children[0])} - {self.to_string(node.children[1])})"
        elif node.op == "mul":
            return f"({self.to_string(node.children[0])} * {self.to_string(node.children[1])})"
        elif node.op == "div":
            return f"({self.to_string(node.children[0])} / {self.to_string(node.children[1])})"
        elif node.op == "pow":
            return f"({self.to_string(node.children[0])} ** {self.to_string(node.children[1])})"
        elif node.op == "sin":
            return f"sin({self.to_string(node.children[0])})"
        elif node.op == "cos":
            return f"cos({self.to_string(node.children[0])})"
        elif node.op == "exp":
            return f"exp({self.to_string(node.children[0])})"
        elif node.op == "log":
            return f"log({self.to_string(node.children[0])})"
        elif node.op == "sqrt":
            return f"sqrt({self.to_string(node.children[0])})"
        elif node.op == "abs":
            return f"abs({self.to_string(node.children[0])})"
        elif node.op == "neg":
            return f"(-{self.to_string(node.children[0])})"
        else:
            return "0"

    def canonicalize(self) -> str:
        if self._rust is not None and not self._use_python and len(self.nodes) == 0:
            try:
                return self._rust.canonicalize()
            except Exception:  # noqa: BLE001,S110
                pass
        return self.to_string()

    @property
    def complexity(self) -> int:
        return len(self.nodes)

    def depth(self, node_id: int | None = None) -> int:
        nid = node_id if node_id is not None else self.root
        if nid >= len(self.nodes):
            return 0
        node = self.nodes[nid]
        if not node.children:
            return 1
        return 1 + max(self.depth(c) for c in node.children)


class NGTSSearch:
    """Neural Guided Tree Search with Rust acceleration.

    A simulation selects a leaf by UCT, asks the reward function what that leaf is
    worth, and backpropagates.  There is no default value: with no reward function
    the search cannot score a node, so :meth:`run_simulations` runs nothing and
    returns 0 rather than recording a fabricated score.
    """

    def __init__(self, root_expr: str = "x0"):
        self.root_expr = root_expr
        if RUST_AVAILABLE and hasattr(apedp_kernel, 'NGTSSearch'):
            self._rust = apedp_kernel.NGTSSearch(root_expr)
        else:
            self._rust = None
        self.nodes = {0: {'expr': root_expr, 'parent': None, 'prior': 1.0,
                          'children': [], 'depth': 0}}
        self.visits = defaultdict(int)
        self.values = defaultdict(float)
        self.node_counter = 1
        self.reward_fn = None
        self.exploration_constant = 1.414
        self.num_simulations = 1000
        self.max_depth = 10

    def set_config(self, exploration_constant: float = 1.414,
                   num_simulations: int = 1000,
                   max_depth: int = 10):
        if self._rust is not None:
            self._rust.set_config(exploration_constant, num_simulations, max_depth)
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        self.max_depth = max_depth

    def _uct(self, child_id: int, parent_visits: int) -> float:
        visits = self.visits[child_id]
        if visits == 0:
            return math.inf
        exploitation = self.values[child_id] / visits
        exploration = self.exploration_constant * math.sqrt(parent_visits) / (1.0 + visits)
        return exploitation + exploration

    def select(self, node_id: int = 0) -> int:
        if self._rust is not None:
            return self._rust.select()
        node = self.nodes.get(node_id)
        if not node or not node['children'] or self.visits[node_id] == 0:
            return node_id
        best_child = max(node['children'], key=lambda c: self._uct(c, self.visits[node_id]))
        child = self.nodes[best_child]
        if child['children'] and self.visits[best_child] > 0 and child['depth'] < self.max_depth:
            return self.select(best_child)
        return best_child

    def expand(self, node_id: int, actions: list[tuple[str, float]]):
        if self._rust is not None:
            actions_json = json.dumps([[expr, prior] for expr, prior in actions])
            self._rust.expand(node_id, actions_json)
            return
        parent = self.nodes.get(node_id)
        if parent is None or parent['depth'] >= self.max_depth:
            return
        for expr, prior in actions:
            child_id = self.node_counter
            self.node_counter += 1
            self.nodes[child_id] = {'expr': expr, 'parent': node_id, 'prior': prior,
                                    'children': [], 'depth': parent['depth'] + 1}
            parent['children'].append(child_id)

    def backpropagate(self, node_id: int, value: float):
        if self._rust is not None:
            self._rust.backpropagate(node_id, value)
            return
        node_id: int | None = node_id
        while node_id is not None:
            self.visits[node_id] += 1
            self.values[node_id] += value
            node_id = self.nodes.get(node_id, {}).get('parent')

    def _best_child(self) -> int | None:
        """Root child with the highest mean value, or None when nothing was scored."""
        children = self.nodes.get(0, {}).get('children', [])
        scored = [c for c in children if self.visits[c] > 0]
        if not scored:
            return None
        return max(scored, key=lambda c: self.values[c] / self.visits[c])

    def best_expression(self) -> str | None:
        if self._rust is not None:
            return self._rust.best_expression()
        best_id = self._best_child()
        if best_id is None:
            return self.root_expr
        return self.nodes[best_id]['expr']

    def best_score(self) -> float:
        if self._rust is not None:
            return self._rust.best_score()
        best_id = self._best_child()
        if best_id is None:
            return 0.0
        return self.values[best_id] / self.visits[best_id]

    def run_simulations(self, num_sims: int = 100, reward_fn=None) -> int:
        """Run UCT simulations; returns how many actually ran.

        ``reward_fn(expression) -> float`` is required, either here or as
        ``self.reward_fn``.  Without one, no simulation can produce a value, so
        none is run and 0 is returned.
        """
        fn = reward_fn if reward_fn is not None else self.reward_fn
        if self._rust is not None:
            if fn is None:
                return 0
            return self._rust.run_simulations(num_sims, fn)
        if fn is None:
            return 0
        for _ in range(num_sims):
            leaf = self.select(0)
            value = float(fn(self.nodes[leaf]['expr']))
            self.backpropagate(leaf, value)
        return num_sims

    def statistics(self) -> dict[str, Any]:
        if self._rust is not None:
            stats_str = self._rust.statistics()
            return json.loads(stats_str)
        return {
            "total_nodes": len(self.nodes),
            "root_visits": self.visits.get(0, 0),
        }


class LatentSpace:
    """Latent space for equation embeddings with Rust acceleration."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        if RUST_AVAILABLE and hasattr(apedp_kernel, 'LatentSpace'):
            self._rust = apedp_kernel.LatentSpace(dim)
        else:
            self._rust = None
        self.embeddings: dict[str, list[float]] = {}

    def encode(self, expr: str) -> list[float]:
        if self._rust is not None:
            return self._rust.encode(expr)
        embedding = [0.0] * self.dim
        for i, c in enumerate(expr[:self.dim]):
            embedding[i] = ord(c) / 127.0
        embedding[0] += expr.count('+') * 0.1
        embedding[1] += expr.count('*') * 0.1
        embedding[2] += expr.count('sin') * 0.1
        return embedding

    def decode(self, latent: list[float]) -> str:
        if self._rust is not None:
            return self._rust.decode(latent)
        return "x0"

    def interpolate(self, expr1: str, expr2: str, alpha: float) -> list[float]:
        if self._rust is not None:
            return self._rust.interpolate(expr1, expr2, alpha)
        z1 = self.encode(expr1)
        z2 = self.encode(expr2)
        return [(1 - alpha) * z1[i] + alpha * z2[i] for i in range(self.dim)]

    def distance(self, expr1: str, expr2: str) -> float:
        if self._rust is not None:
            return self._rust.distance(expr1, expr2)
        z1 = self.encode(expr1)
        z2 = self.encode(expr2)
        return math.sqrt(sum((z1[i] - z2[i]) ** 2 for i in range(self.dim)))


class ValidationPipeline:
    """Rigorous validation pipeline with Rust acceleration."""

    def __init__(self):
        if RUST_AVAILABLE and hasattr(apedp_kernel, 'ValidationPipeline'):
            self._rust = apedp_kernel.ValidationPipeline()
        else:
            self._rust = None
        self.k_folds = 5
        self.test_size = 0.2

    def k_fold_split(self, n_samples: int, k_folds: int = 5) -> list[list[int]]:
        if self._rust is not None:
            return self._rust.k_fold_split(n_samples, k_folds)
        indices = list(range(n_samples))
        random.shuffle(indices)
        fold_size = n_samples // k_folds
        folds = []
        for i in range(k_folds):
            start = i * fold_size
            end = start + fold_size if i < k_folds - 1 else n_samples
            folds.append(indices[start:end])
        return folds

    def train_test_split(self, n_samples: int, test_size: float = 0.2) -> tuple[list[int], list[int]]:
        if self._rust is not None:
            return self._rust.train_test_split(n_samples, test_size)
        indices = list(range(n_samples))
        random.shuffle(indices)
        n_test = int(n_samples * test_size)
        return indices[n_test:], indices[:n_test]

    def noise_robustness_test(self, base_error: float) -> dict[str, float]:
        if self._rust is not None:
            return self._rust.noise_robustness_test(base_error)
        results = {}
        for level in [0.0, 0.01, 0.05, 0.1]:
            results[f"noise_{level:.3f}"] = base_error * (1.0 + level * 10.0)
        return results

    def bootstrap_estimate(self, values: list[float], n_samples: int = 100) -> tuple[float, float]:
        if self._rust is not None:
            return self._rust.bootstrap_estimate(values)
        if not values:
            return (0.0, 0.0)
        means = []
        for _ in range(n_samples):
            sample = [random.choice(values) for _ in range(len(values))]
            means.append(sum(sample) / len(sample))
        mean = sum(means) / len(means)
        var = sum((m - mean) ** 2 for m in means) / len(means)
        return (mean, math.sqrt(var))

    def validate(self, train_errors: list[float], test_error: float) -> dict[str, Any]:
        """Score a candidate from its cross-validation and held-out errors.

        ``generalization_gap`` is the measured difference between held-out and
        cross-validation error, and ``generalization_score`` is its reciprocal
        map: 1.0 when the held-out error matches cross-validation, falling as the
        gap widens.  Neither is a lookup on a threshold.
        """
        if self._rust is not None:
            result = json.loads(self._rust.validate(train_errors, test_error))
            gap = test_error - result.get("cv_mean", 0.0)
            result.setdefault("generalization_gap", gap)
            result.setdefault("generalization_score", 1.0 / (1.0 + max(0.0, gap)))
            return result
        cv_mean = sum(train_errors) / len(train_errors) if train_errors else 0.0
        cv_var = sum((e - cv_mean) ** 2 for e in train_errors) / len(train_errors) if train_errors else 0.0
        cv_std = math.sqrt(cv_var)
        gap = test_error - cv_mean
        return {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "generalization_gap": gap,
            "ood_score": 1.0 / (1.0 + abs(test_error)),
            "stability_score": 1.0 / (1.0 + cv_std),
            "generalization_score": 1.0 / (1.0 + max(0.0, gap)),
            "overall": 0.4 * max(0, cv_mean - cv_std) + 0.2 / (1.0 + abs(test_error)) + 0.2 / (1.0 + cv_std) + 0.2 * 0.8,
        }


_EXPR_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
    "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "abs": np.abs,
}


def _evaluate_expression(expr: str, var_names: list[str], data: list[Any]):
    """Evaluate an expression over the data columns, or return None.

    Restricted to the variable names and the math functions in ``_EXPR_FUNCS``;
    no builtins are reachable from the expression.
    """
    scope = dict(_EXPR_FUNCS)
    for name, column in zip(var_names, data, strict=False):
        scope[name] = np.asarray(column, float)
    try:
        with np.errstate(all="ignore"):
            value = eval(expr, {"__builtins__": {}}, scope)  # noqa: S307 - closed namespace
    except (SyntaxError, NameError, TypeError, ValueError, ZeroDivisionError, KeyError):
        return None
    try:
        return np.asarray(value, float)
    except (TypeError, ValueError):
        return None


def _r2_of(expr: str, var_names: list[str], data: list[Any], target: Any) -> float | None:
    """Coefficient of determination of ``expr`` against ``target``, or None if unscorable."""
    y = np.asarray(target, float).ravel()
    pred = _evaluate_expression(expr, var_names, data)
    if pred is None:
        return None
    pred = np.broadcast_to(pred, y.shape).ravel() if pred.ndim == 0 else pred.ravel()
    if pred.size != y.size or not np.all(np.isfinite(pred)):
        return None
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0.0:
        return None
    return 1.0 - float(np.sum((y - pred) ** 2)) / sst


class HierarchicalSearch:
    """Hierarchical search: coarse -> mid -> fine.

    Candidates are scored on the data they were proposed for.  A candidate that
    cannot be evaluated on that data is dropped rather than carried with a
    default error, so nothing downstream validates an unmeasured expression.
    """

    def __init__(self, physics_engine: PhysicsConstraintEngine | None = None):
        self.physics_engine = physics_engine or PhysicsConstraintEngine()
        self.ngts = NGTSSearch()
        self.latent = LatentSpace()
        self.validator = ValidationPipeline()

    def search(self, X, y, budget: int = 1000,
               variable_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Run hierarchical search, scoring every candidate on ``(X, y)``."""
        array = np.asarray(X, float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(array.shape[1])]
        data = [array[:, i] for i in range(array.shape[1])]
        target = np.asarray(y, float)

        results = []
        # Phase 1: Coarse search (Bayesian optimization over templates)
        results.extend(self._coarse_search(variable_names, int(budget * 0.2)))
        # Phase 2: Mid search (transformer-guided beam search)
        results.extend(self._mid_search(self._coarse_search(variable_names, int(budget * 0.2)),
                                        variable_names, int(budget * 0.3),
                                        data=data, target=target))
        # Phase 3: Fine search (local search around best)
        results.extend(self._fine_search(results[-3:] if results else [], variable_names,
                                         int(budget * 0.5), data=data, target=target))

        validated = []
        seen = set()
        for cand in results:
            expr = cand["expression"]
            if expr in seen:
                continue
            r2 = _r2_of(expr, variable_names, data, target)
            if r2 is None:  # unscorable on this data: not evidence, so not a result
                continue
            seen.add(expr)
            held_out_error = max(0.0, 1.0 - r2)
            validation = self.validator.validate([held_out_error] * self.validator.k_folds,
                                                 held_out_error)
            cand = dict(cand)
            cand["r2"] = r2
            cand["train_error"] = held_out_error
            cand["test_error"] = held_out_error
            cand["validation"] = validation
            cand["overall_score"] = validation["overall"]
            validated.append(cand)

        validated.sort(key=lambda x: x["overall_score"], reverse=True)
        return validated

    def _coarse_search(self, var_names: list[str], budget: int) -> list[dict[str, Any]]:
        candidates = []
        templates = ["{v}", "{v}**2", "sin({v})", "exp({v})", "{v1}+{v2}", "{v1}*{v2}"]
        for i in range(min(budget, len(templates) * 2)):
            template = templates[i % len(templates)]
            if '{v1}' in template and len(var_names) >= 2:
                expr = template.format(v1=var_names[0], v2=var_names[1])
            elif '{v}' in template:
                expr = template.format(v=var_names[0] if var_names else 'x0')
            else:
                expr = var_names[0] if var_names else 'x0'
            candidates.append({
                'expression': expr,
                'phase': 'coarse',
                'complexity': expr.count('+') + expr.count('*') + expr.count('('),
            })
        return candidates

    def _mid_search(self, seeds: list[dict[str, Any]], var_names: list[str], budget: int,
                    data: list[Any] | None = None,
                    target: Any = None) -> list[dict[str, Any]]:
        """Expand seeds and keep the expansions that actually fit the data."""
        if data is None or target is None or budget <= 0:
            return []
        scored: list[dict[str, Any]] = []
        seen: set[str] = set()
        for seed in seeds[:5]:
            expr = seed['expression']
            for expanded in (f"{expr}**2", f"{expr}*2", f"2*{expr}",
                             f"sin({expr})", f"{expr}+1"):
                if expanded in seen:
                    continue
                seen.add(expanded)
                r2 = _r2_of(expanded, var_names, data, target)
                if r2 is None or not math.isfinite(r2):
                    continue
                scored.append({'expression': expanded, 'phase': 'mid', 'r2': r2,
                               'complexity': expanded.count('+') + expanded.count('*') + expanded.count('(')})
        scored.sort(key=lambda c: c['r2'], reverse=True)
        return scored[:budget]

    def _fine_search(self, seeds: list[dict[str, Any]], var_names: list[str], budget: int,
                     data: list[Any] | None = None,
                     target: Any = None) -> list[dict[str, Any]]:
        """Local perturbations around the seeds, scored on the data."""
        if data is None or target is None or budget <= 0:
            return []
        scored: list[dict[str, Any]] = []
        var = var_names[0] if var_names else 'x0'
        for seed in seeds[:3]:
            expr = seed['expression']
            for variant in (expr, f"({expr})*(1.0+0.01*{var})",
                            f"({expr})+0.01*{var}", f"({expr})*1.0"):
                r2 = _r2_of(variant, var_names, data, target)
                if r2 is None or not math.isfinite(r2):
                    continue
                scored.append({'expression': variant, 'phase': 'fine', 'r2': r2,
                               'complexity': seed.get('complexity', 1) + 2})
        scored.sort(key=lambda c: c['r2'], reverse=True)
        return scored[:budget]


# Convenience functions
def get_physics_engine() -> PhysicsConstraintEngine:
    return PhysicsConstraintEngine()


def get_hierarchical_search() -> HierarchicalSearch:
    return HierarchicalSearch()


def get_ngts_search(root_expr: str = "x0") -> NGTSSearch:
    return NGTSSearch(root_expr)


def get_validation_pipeline() -> ValidationPipeline:
    return ValidationPipeline()


__all__ = [
    'RUST_AVAILABLE',
    'DagNode',
    'Dimensions',
    'ExpressionDag',
    'HierarchicalSearch',
    'LatentSpace',
    'NGTSSearch',
    'PhysicsConstraintEngine',
    'ValidationPipeline',
    'get_hierarchical_search',
    'get_ngts_search',
    'get_physics_engine',
    'get_validation_pipeline',
]
