"""Fast cross-validation leveraging fast fitness evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np

from apedp.fast_fitness import get_fast_evaluator


class FastCrossValidator:
    """Fast cross-validation using accelerated fitness evaluation."""
    
    def __init__(self):
        self.evaluator = get_fast_evaluator()
    
    def cross_validate(self, expr_str: str, variable_names: list[str],
                      X: np.ndarray, y: np.ndarray, k: int = 5) -> dict[str, Any]:
        """K-fold cross-validation."""
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        scores = []
        for fold_idx in range(k):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_idx])
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # For simplicity, evaluate on validation set using original expression
            # (In practice, would need to refit parameters)
            metrics = self.evaluator.evaluate(expr_str, variable_names, X_val, y_val)
            if not metrics.get('has_nan_or_inf'):
                scores.append(metrics.get('r_squared', float('-inf')))
        
        return {
            "mean_r2": np.mean(scores) if scores else float('-inf'),
            "std_r2": np.std(scores) if scores else 0.0,
            "scores": scores,
            "k": k
        }


_default_cv = None

def get_fast_cv() -> FastCrossValidator:
    global _default_cv
    if _default_cv is None:
        _default_cv = FastCrossValidator()
    return _default_cv
