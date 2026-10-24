"""Automatic proposer selection based on problem characteristics.

Reduces the learning curve by recommending the best proposer for each problem type.
"""
from __future__ import annotations

from typing import Any


class ProposerRouter:
    """Routes problems to the most suitable proposer."""
    
    def __init__(self):
        self.proposer_map = {
            "mechanical": ["rime_ad", "grammar", "library"],
            "trading": ["rime_ad", "library", "grammar"],
            "chemical": ["rime_ad", "grammar"],
            "oscillatory": ["rime_ad", "grammar"],
            "feynman": ["rime_ad", "grammar"],
            "default": ["rime_ad", "grammar"],
        }
    
    def select_proposer(self, problem: Any, budget: int = 100) -> tuple[str, dict[str, Any]]:
        """Select best proposer and recommended settings for a problem."""
        problem_type = self._detect_problem_type(problem)
        category = getattr(problem, "category", None) or problem_type
        
        if category in self.proposer_map:
            recommended = self.proposer_map[category][0]
        else:
            recommended = self.proposer_map["default"][0]
        
        settings = self._get_settings_for_proposer(recommended, budget, problem_type)
        return recommended, settings
    
    def _detect_problem_type(self, problem: Any) -> str:
        """Detect problem type from problem attributes."""
        name = str(getattr(problem, "name", "")).lower()
        pid = str(getattr(problem, "problem_id", "")).lower()
        
        if any(x in name or x in pid for x in ["trading", "finance", "volatility"]):
            return "trading"
        elif any(x in name or x in pid for x in ["pendulum", "oscill", "harmonic", "mechanic"]):
            return "mechanical"
        elif any(x in name or x in pid for x in ["chem", "reaction", "brusselator", "oregon"]):
            return "chemical"
        elif any(x in name or x in pid for x in ["rlc", "circuit", "vdp", "chua"]):
            return "oscillatory"
        elif any(x in name or x in pid for x in ["feynman", "relativ", "quantum"]):
            return "feynman"
        return "default"
    
    def _get_settings_for_proposer(self, proposer: str, budget: int, problem_type: str) -> dict[str, Any]:
        """Get optimized settings for each proposer."""
        base_settings = {
            "budget": budget,
            "parallel": True,
        }
        
        if proposer == "rime_ad":
            return {
                **base_settings,
                "description": "RIME-AD: Best for complex/saturating forms (Langmuir, Hill equations)",
                "recommended_for": "Problems with rational/saturating behavior",
            }
        elif proposer == "grammar":
            return {
                **base_settings,
                "max_terms": 4,
                "description": "Grammar enumeration: Good for structured physical laws",
                "recommended_for": "Well-known physical forms",
            }
        elif proposer == "library":
            return {
                **base_settings,
                "description": "SINDy-style sparse regression: Fast, good for discovering linear combinations",
                "recommended_for": "Linear/additive relationships",
            }
        return base_settings
    
    def explain_choice(self, proposer: str, problem: Any) -> str:
        """Explain why a proposer was chosen."""
        _, settings = self.select_proposer(problem)
        return f"Auto-selected {proposer}: {settings.get('description', '')}. {settings.get('recommended_for', '')}"


def auto_select_proposer(problem: Any, budget: int = 100) -> tuple[str, dict[str, Any]]:
    """Convenience function for auto-selecting proposer."""
    router = ProposerRouter()
    return router.select_proposer(problem, budget)
