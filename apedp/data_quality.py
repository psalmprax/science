"""Data quality validation for equation discovery.

Catches garbage-in-garbage-out issues before expensive searches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DataQualityReport:
    """Report on data quality issues."""
    is_acceptable: bool
    warnings: list[str]
    errors: list[str]
    recommendations: list[str]
    score: float  # 0-1


class DataQualityValidator:
    """Validates data before running expensive searches."""
    
    def validate(self, X: np.ndarray, y: np.ndarray = None) -> DataQualityReport:
        """Validate input data quality."""
        warnings = []
        errors = []
        recommendations = []
        score = 1.0
        
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        
        # Check shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Sample size
        if n_samples < 20:
            errors.append(f"Too few samples ({n_samples}). Need at least 20-50 for reliable discovery.")
            score -= 0.3
        elif n_samples < 100:
            warnings.append(f"Small sample size ({n_samples}). Results may be less reliable.")
            score -= 0.1
            recommendations.append("Consider collecting more data or using simpler models.")
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(X)):
            errors.append("Data contains NaN or Inf values.")
            score -= 0.4
            recommendations.append("Clean or interpolate missing/invalid values.")
        
        if y is not None and np.any(~np.isfinite(y)):
            errors.append("Target contains NaN or Inf values.")
            score -= 0.4
        
        # Check for constant features
        for i in range(n_features):
            if np.std(X[:, i]) < 1e-12:
                warnings.append(f"Feature {i} is constant or nearly constant.")
                score -= 0.05
        
        # Check correlation (multicollinearity warning)
        if n_features > 1 and n_samples > n_features:
            try:
                corr = np.corrcoef(X.T)
                high_corr = np.sum(np.abs(corr) > 0.95) - n_features  # subtract diagonal
                if high_corr > 0:
                    warnings.append(f"High multicollinearity detected ({high_corr//2} pairs).")
                    score -= 0.1
                    recommendations.append("Consider removing redundant features.")
            except Exception:
                pass
        
        # Noise estimation
        if y is not None and n_samples > 10:
            try:
                # Simple noise estimate via roughness
                sorted_idx = np.argsort(X[:, 0]) if n_features > 0 else np.arange(n_samples)
                y_sorted = y[sorted_idx]
                diffs = np.diff(y_sorted)
                noise_est = np.std(diffs) / (np.std(y) + 1e-12)
                if noise_est > 0.5:
                    warnings.append(f"High noise level estimated ({noise_est:.2f}). Discovery may be difficult.")
                    score -= 0.15
                    recommendations.append("Try smoothing data or using more robust methods.")
            except Exception:
                pass
        
        is_acceptable = len(errors) == 0 and score >= 0.5
        score = max(0.0, score)
        
        if is_acceptable and not warnings:
            recommendations.append("Data quality looks good. Ready for discovery.")
        
        return DataQualityReport(
            is_acceptable=is_acceptable,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            score=score
        )
