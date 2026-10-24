"""Fast fitness evaluation with Rust-first backend and Python fallback.

Provides drop-in replacements for expensive fitness evaluations.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from apedp.rust_bridge import RustKernelBridge, default_kernel


class FastFitnessEvaluator:
    """Rust-first fitness evaluator with automatic fallback."""
    
    def __init__(self, bridge: RustKernelBridge | None = None):
        self.bridge = bridge or default_kernel
    
    def evaluate(self, expr_str: str, variable_names: list[str], 
                 X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Evaluate expression fitness with Rust acceleration."""
        try:
            # Try Rust path via bridge
            if hasattr(self.bridge, 'evaluate_expression'):
                y_pred = self.bridge.evaluate_expression(expr_str, variable_names, 
                                                       [X[:, i] for i in range(X.shape[1])])
                metrics = self.bridge.compute_metrics(y, y_pred)
                return metrics
        except Exception:
            pass
        
        # Fallback to Python
        return self._python_fallback(expr_str, variable_names, X, y)
    
    def batch_evaluate(self, expressions: list[str], variable_names: list[str],
                      X: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
        """Batch evaluate with parallel Rust path."""
        try:
            if hasattr(self.bridge, 'score_candidates_batch'):
                data = [X[:, i] for i in range(X.shape[1])]
                return self.bridge.score_candidates_batch(expressions, variable_names, data, y)
        except Exception:
            pass
        
        # Fallback
        results = []
        for expr in expressions:
            results.append(self.evaluate(expr, variable_names, X, y))
        return results
    
    def _python_fallback(self, expr_str: str, variable_names: list[str],
                        X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Pure Python fallback."""
        import sympy as sp
        from apedp.safety import safe_parse_expr
        
        try:
            expr = safe_parse_expr(expr_str)
            var_syms = [sp.Symbol(v) for v in variable_names]
            f = sp.lambdify(var_syms, expr, modules=['numpy'])
            X_list = [X[:, i] for i in range(X.shape[1])]
            y_pred = f(*X_list)
            y_pred = np.asarray(y_pred, dtype=float).flatten()
            
            residuals = y - y_pred
            mse = float(np.mean(residuals**2))
            mae = float(np.mean(np.abs(residuals)))
            var_y = float(np.var(y))
            r2 = 1.0 - (mse / var_y) if var_y > 1e-15 else (1.0 if mse < 1e-15 else 0.0)
            
            return {
                "mse": mse,
                "mae": mae,
                "r_squared": r2,
                "max_residual": float(np.max(np.abs(residuals))),
                "has_nan_or_inf": bool(np.any(~np.isfinite(residuals))),
                "kernel_path": "fallback"
            }
        except Exception as e:
            return {
                "mse": float('nan'),
                "mae": float('nan'),
                "r_squared": float('-inf'),
                "max_residual": float('nan'),
                "has_nan_or_inf": True,
                "kernel_path": "fallback",
                "error": str(e)
            }


# Global instance
_default_evaluator = None

def get_fast_evaluator() -> FastFitnessEvaluator:
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = FastFitnessEvaluator()
    return _default_evaluator


def fast_evaluate(expr_str: str, variable_names: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Convenience function for fast evaluation."""
    return get_fast_evaluator().evaluate(expr_str, variable_names, X, y)


def fast_batch_evaluate(expressions: list[str], variable_names: list[str], 
                       X: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Convenience function for fast batch evaluation."""
    return get_fast_evaluator().batch_evaluate(expressions, variable_names, X, y)
