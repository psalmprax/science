"""Uncertainty quantification for equation discovery.

Helps distinguish real laws from curve fits by assessing confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UncertaintyReport:
    """Uncertainty assessment for a discovered equation."""
    confidence_score: float  # 0-1, overall confidence
    overfit_risk: str  # low/medium/high
    extrapolation_reliability: str  # low/medium/high
    interpretability: str  # low/medium/high
    warnings: list[str]
    recommendations: list[str]


class UncertaintyQuantifier:
    """Quantify uncertainty in discovered equations."""
    
    def assess(self, equation: str, train_score: float, test_score: float = None, 
               complexity: int = 1) -> UncertaintyReport:
        """Assess uncertainty of a discovered equation."""
        warnings = []
        recommendations = []
        
        # Check for overfitting
        if test_score is not None:
            gap = train_score - test_score
            if gap > 0.1:
                overfit_risk = "high"
                warnings.append(f"Large train-test gap ({gap:.3f}) suggests overfitting.")
                recommendations.append("Simplify the equation or get more data.")
            elif gap > 0.05:
                overfit_risk = "medium"
                warnings.append("Moderate train-test gap. Use with caution.")
            else:
                overfit_risk = "low"
        else:
            overfit_risk = "medium"
            warnings.append("No test set evaluation available. Overfit risk unknown.")
            recommendations.append("Always test on held-out data.")
        
        # Complexity penalty
        if complexity > 6:
            warnings.append(f"High complexity ({complexity} terms). May be overfitting.")
            if overfit_risk == "low":
                overfit_risk = "medium"
        elif complexity > 8:
            if overfit_risk in ["low", "medium"]:
                overfit_risk = "high"
        
        # Extrapolation
        if test_score is not None and test_score > 0.9 and overfit_risk == "low":
            extrapolation_reliability = "medium"
            warnings.append("Good interpolation, but extrapolation reliability is uncertain.")
            recommendations.append("Test beyond training data range if possible.")
        elif test_score and test_score > 0.8:
            extrapolation_reliability = "low"
        else:
            extrapolation_reliability = "low"
            warnings.append("Poor or unknown extrapolation reliability.")
        
        # Interpretability
        if complexity <= 3 and self._looks_physical(equation):
            interpretability = "high"
        elif complexity <= 5:
            interpretability = "medium"
        else:
            interpretability = "low"
            warnings.append("Complex equation may be hard to interpret physically.")
        
        # Overall confidence
        confidence = train_score * 0.3
        if test_score:
            confidence += test_score * 0.4
        if overfit_risk == "low":
            confidence += 0.2
        elif overfit_risk == "medium":
            confidence += 0.1
        if interpretability == "high":
            confidence += 0.1
        
        confidence = min(1.0, max(0.0, confidence))
        
        if confidence < 0.6:
            recommendations.append("Low confidence. Verify with domain expertise and additional tests.")
        elif confidence >= 0.8:
            recommendations.append("Good confidence, but still verify causality experimentally if possible.")
        
        return UncertaintyReport(
            confidence_score=confidence,
            overfit_risk=overfit_risk,
            extrapolation_reliability=extrapolation_reliability,
            interpretability=interpretability,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _looks_physical(self, equation: str) -> bool:
        """Heuristic check if equation looks physically reasonable."""
        # Check for common physical patterns
        physical_patterns = ['sin', 'cos', 'exp', 'sqrt', '/', '*']
        # Very rough heuristic
        return any(p in equation for p in physical_patterns) and len(equation) < 100
