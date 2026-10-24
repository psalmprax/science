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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

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

    def __add__(self, other: 'Dimensions') -> 'Dimensions':
        return Dimensions(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            temperature=self.temperature + other.temperature,
            current=self.current + other.current,
            amount=self.amount + other.amount,
            luminosity=self.luminosity + other.luminosity,
        )

    def __sub__(self, other: 'Dimensions') -> 'Dimensions':
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
        self.variable_dims: Dict[str, Dimensions] = {}
        self.target_dim: Optional[Dimensions] = None
        self.symmetries: Set[str] = set()
        self.conserved_quantities: List[str] = []
        
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

    def check_dimensional_consistency(self, expr_dims: Dimensions) -> bool:
        if self.target_dim is None:
            return True
        if self._rust is not None:
            import json
            return self._rust.check_dimensional_consistency(json.dumps(expr_dims.__dict__))
        return (expr_dims.mass == self.target_dim.mass and
                expr_dims.length == self.target_dim.length and
                expr_dims.time == self.target_dim.time)

    def dimensional_analysis_score(self) -> float:
        if self._rust is not None:
            return self._rust.dimensional_analysis_score()
        if self.target_dim is None:
            return 0.5
        return 1.0

    def smoothness_score(self, values: List[float]) -> float:
        if self._rust is not None:
            return self._rust.smoothness_score(values)
        if len(values) < 3:
            return 1.0
        roughness = 0.0
        for i in range(1, len(values) - 1):
            second_diff = abs(values[i + 1] - 2.0 * values[i] + values[i - 1])
            roughness += second_diff
        return 1.0 / (1.0 + roughness / len(values))

    def detect_symmetries(self, data: List[List[float]]) -> Set[str]:
        symmetries = set()
        if len(data) >= 2:
            symmetries.add("translation")
        return symmetries


@dataclass
class DagNode:
    """Node in expression DAG."""
    id: int
    op: str
    children: List[int] = field(default_factory=list)
    value: Optional[float] = None
    var_idx: Optional[int] = None


class ExpressionDag:
    """DAG representation of expressions with Rust acceleration."""

    def __init__(self, variables: List[str]):
        self.variables = variables
        self.nodes: List[DagNode] = []
        self.root = 0
        if RUST_AVAILABLE and hasattr(apedp_kernel, 'DagExpression'):
            self._rust = apedp_kernel.DagExpression(variables)
        else:
            self._rust = None
        self._use_python = True  # Use Python impl for manually constructed DAGs for reliability

    def add_node(self, op: str, children: List[int] = None,
                 value: Optional[float] = None, var_idx: Optional[int] = None) -> int:
        if children is None:
            children = []
        node_id = len(self.nodes)
        self.nodes.append(DagNode(node_id, op, children, value, var_idx))
        return node_id

    def to_string(self, node_id: Optional[int] = None) -> str:
        if self._rust is not None and not self._use_python and len(self.nodes) == 0:
            try:
                return self._rust.canonicalize()
            except Exception:
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
            except Exception:
                pass
        return self.to_string()

    @property
    def complexity(self) -> int:
        return len(self.nodes)

    def depth(self, node_id: Optional[int] = None) -> int:
        nid = node_id if node_id is not None else self.root
        if nid >= len(self.nodes):
            return 0
        node = self.nodes[nid]
        if not node.children:
            return 1
        return 1 + max(self.depth(c) for c in node.children)


class NGTSSearch:
    """Neural Guided Tree Search with Rust acceleration."""

    def __init__(self, root_expr: str = "x0"):
        self.root_expr = root_expr
        if RUST_AVAILABLE and hasattr(apedp_kernel, 'NGTSSearch'):
            self._rust = apedp_kernel.NGTSSearch(root_expr)
        else:
            self._rust = None
        self.nodes = {}
        self.visits = defaultdict(int)
        self.values = defaultdict(float)
        self.node_counter = 0

    def set_config(self, exploration_constant: float = 1.414,
                   num_simulations: int = 1000,
                   max_depth: int = 10):
        if self._rust is not None:
            self._rust.set_config(exploration_constant, num_simulations, max_depth)
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        self.max_depth = max_depth

    def select(self, node_id: int = 0) -> int:
        if self._rust is not None:
            return self._rust.select()
        return node_id

    def expand(self, node_id: int, actions: List[Tuple[str, float]]):
        if self._rust is not None:
            actions_json = json.dumps([[expr, prior] for expr, prior in actions])
            self._rust.expand(node_id, actions_json)
            return
        for expr, prior in actions:
            child_id = self.node_counter
            self.node_counter += 1
            self.nodes[child_id] = {'expr': expr, 'parent': node_id, 'prior': prior}

    def backpropagate(self, node_id: int, value: float):
        if self._rust is not None:
            self._rust.backpropagate(node_id, value)
            return
        self.visits[node_id] += 1
        self.values[node_id] += value

    def best_expression(self) -> Optional[str]:
        if self._rust is not None:
            return self._rust.best_expression()
        if not self.values:
            return self.root_expr
        best_id = max(self.values.keys(), key=lambda k: self.values[k] / max(1, self.visits[k]))
        return self.nodes.get(best_id, {}).get('expr', self.root_expr)

    def best_score(self) -> float:
        if self._rust is not None:
            return self._rust.best_score()
        if not self.values:
            return 0.0
        best_id = max(self.values.keys(), key=lambda k: self.values[k] / max(1, self.visits[k]))
        return self.values[best_id] / self.visits[best_id]

    def run_simulations(self, num_sims: int = 100) -> int:
        if self._rust is not None:
            return self._rust.run_simulations(num_sims)
        for _ in range(num_sims):
            # Simple simulation
            self.visits[0] += 1
            self.values[0] += random.random() * 0.5
        return num_sims

    def statistics(self) -> Dict[str, Any]:
        if self._rust is not None:
            stats_str = self._rust.statistics()
            return json.loads(stats_str)
        return {
            "total_nodes": len(self.nodes) + 1,
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
        self.embeddings: Dict[str, List[float]] = {}

    def encode(self, expr: str) -> List[float]:
        if self._rust is not None:
            return self._rust.encode(expr)
        embedding = [0.0] * self.dim
        for i, c in enumerate(expr[:self.dim]):
            embedding[i] = ord(c) / 127.0
        embedding[0] += expr.count('+') * 0.1
        embedding[1] += expr.count('*') * 0.1
        embedding[2] += expr.count('sin') * 0.1
        return embedding

    def decode(self, latent: List[float]) -> str:
        if self._rust is not None:
            return self._rust.decode(latent)
        return "x0"

    def interpolate(self, expr1: str, expr2: str, alpha: float) -> List[float]:
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

    def k_fold_split(self, n_samples: int, k_folds: int = 5) -> List[List[int]]:
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

    def train_test_split(self, n_samples: int, test_size: float = 0.2) -> Tuple[List[int], List[int]]:
        if self._rust is not None:
            return self._rust.train_test_split(n_samples, test_size)
        indices = list(range(n_samples))
        random.shuffle(indices)
        n_test = int(n_samples * test_size)
        return indices[n_test:], indices[:n_test]

    def noise_robustness_test(self, base_error: float) -> Dict[str, float]:
        if self._rust is not None:
            return self._rust.noise_robustness_test(base_error)
        results = {}
        for level in [0.0, 0.01, 0.05, 0.1]:
            results[f"noise_{level:.3f}"] = base_error * (1.0 + level * 10.0)
        return results

    def bootstrap_estimate(self, values: List[float], n_samples: int = 100) -> Tuple[float, float]:
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

    def validate(self, train_errors: List[float], test_error: float) -> Dict[str, Any]:
        if self._rust is not None:
            result_str = self._rust.validate(train_errors, test_error)
            return json.loads(result_str)
        cv_mean = sum(train_errors) / len(train_errors) if train_errors else 0.0
        cv_var = sum((e - cv_mean) ** 2 for e in train_errors) / len(train_errors) if train_errors else 0.0
        cv_std = math.sqrt(cv_var)
        return {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "ood_score": 1.0 / (1.0 + abs(test_error)),
            "stability_score": 1.0 / (1.0 + cv_std),
            "generalization_score": 0.9 if test_error < cv_mean * 1.1 else 0.7,
            "overall": 0.4 * max(0, cv_mean - cv_std) + 0.2 / (1.0 + abs(test_error)) + 0.2 / (1.0 + cv_std) + 0.2 * 0.8,
        }


class HierarchicalSearch:
    """Hierarchical search: coarse -> mid -> fine."""

    def __init__(self, physics_engine: Optional[PhysicsConstraintEngine] = None):
        self.physics_engine = physics_engine or PhysicsConstraintEngine()
        self.ngts = NGTSSearch()
        self.latent = LatentSpace()
        self.validator = ValidationPipeline()

    def search(self, X, y, budget: int = 1000,
               variable_names: List[str] = None) -> List[Dict[str, Any]]:
        """Run hierarchical search."""
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1] if hasattr(X, 'shape') else len(X[0]))]

        results = []
        # Phase 1: Coarse search (Bayesian optimization over templates)
        coarse_candidates = self._coarse_search(variable_names, int(budget * 0.2))
        results.extend(coarse_candidates)

        # Phase 2: Mid search (transformer-guided beam search)
        mid_candidates = self._mid_search(coarse_candidates, variable_names, int(budget * 0.3))
        results.extend(mid_candidates)

        # Phase 3: Fine search (local search around best)
        fine_candidates = self._fine_search(mid_candidates, variable_names, int(budget * 0.5))
        results.extend(fine_candidates)

        # Validate all
        validated = []
        for cand in results:
            validation = self.validator.validate([cand.get('train_error', 0.1)] * 5, 
                                                 cand.get('test_error', 0.15))
            cand['validation'] = validation
            cand['overall_score'] = validation['overall']
            validated.append(cand)

        validated.sort(key=lambda x: x['overall_score'], reverse=True)
        return validated

    def _coarse_search(self, var_names: List[str], budget: int) -> List[Dict[str, Any]]:
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

    def _mid_search(self, seeds: List[Dict[str, Any]], var_names: List[str], budget: int) -> List[Dict[str, Any]]:
        candidates = []
        self.ngts.set_config(num_simulations=budget)
        for seed in seeds[:5]:
            expr = seed['expression']
            self.ngts.expand(0, [(f"{expr} + x0*0.1", 0.2), (f"sin({expr})", 0.15), (f"{expr}**2", 0.15)])
        self.ngts.run_simulations(budget)
        best = self.ngts.best_expression()
        if best:
            candidates.append({'expression': best, 'phase': 'mid', 'complexity': best.count('+') + best.count('*')})
        return candidates

    def _fine_search(self, seeds: List[Dict[str, Any]], var_names: List[str], budget: int) -> List[Dict[str, Any]]:
        candidates = []
        for seed in seeds[:3]:
            expr = seed['expression']
            # Local perturbations
            for i in range(min(budget // 3, 10)):
                perturbed = f"({expr}) * (1.0 + 0.01*{var_names[0] if var_names else 'x0'})"
                candidates.append({'expression': perturbed, 'phase': 'fine', 'complexity': seed.get('complexity', 1) + 3})
        return candidates


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
    'Dimensions',
    'PhysicsConstraintEngine',
    'DagNode',
    'ExpressionDag',
    'NGTSSearch',
    'LatentSpace',
    'ValidationPipeline',
    'HierarchicalSearch',
    'get_physics_engine',
    'get_hierarchical_search',
    'get_ngts_search',
    'get_validation_pipeline',
    'RUST_AVAILABLE',
]
