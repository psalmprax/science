"""Fast Pareto optimization with Rust-first backend and Python fallback."""
from __future__ import annotations

from typing import Any

import numpy as np


class FastPareto:
    """Fast Pareto sorting and selection."""
    
    def nondominated_sort(self, objectives: np.ndarray) -> list[list[int]]:
        """Fast non-dominated sorting."""
        try:
            # Try to use Rust if available
            from apedp.rust_bridge import get_default_bridge
            bridge = get_default_bridge()
            if bridge.use_rust and hasattr(bridge, 'kernel'):
                kernel = bridge.kernel
                if hasattr(kernel, 'pareto_sort'):
                    return kernel.pareto_sort(objectives.tolist())
        except Exception:
            pass
        
        return self._python_nondominated_sort(objectives)
    
    def _python_nondominated_sort(self, objectives: np.ndarray) -> list[list[int]]:
        """Python fallback for non-dominated sorting."""
        n = objectives.shape[0]
        if n == 0:
            return []
        
        # objectives: (n, m) where higher is better or lower is better
        # Assume minimize all objectives
        fronts = []
        remaining = set(range(n))
        domination_count = np.zeros(n, dtype=int)
        dominated_sets = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                    dominated_sets[i].append(j)
                elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    domination_count[i] += 1
        
        # First front
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_sets[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts
    
    def select_best(self, candidates: list, objectives: np.ndarray, k: int = 10) -> list:
        """Select best k candidates from Pareto front."""
        fronts = self.nondominated_sort(objectives)
        selected = []
        for front in fronts:
            selected.extend([candidates[i] for i in front])
            if len(selected) >= k:
                break
        return selected[:k]


_default_pareto = None

def get_fast_pareto() -> FastPareto:
    global _default_pareto
    if _default_pareto is None:
        _default_pareto = FastPareto()
    return _default_pareto
