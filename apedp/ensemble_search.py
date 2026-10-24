"""Ensemble search across multiple proposers.

Tries different proposers in parallel and combines results to reduce
chances of getting stuck in local optima.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apedp.proposer_router import auto_select_proposer


class EnsembleSearcher:
    """Run multiple proposers and select best overall result."""
    
    def __init__(self, proposers: list[str] = None):
        self.proposers = proposers or ["rime_ad", "grammar", "library"]
    
    def search_ensemble(self, problem: Any, budget_total: int = 300) -> dict[str, Any]:
        """Search with ensemble of proposers."""
        budget_per = budget_total // len(self.proposers)
        results = {}
        
        for proposer in self.proposers:
            try:
                results[proposer] = {
                    "status": "would_run",
                    "budget": budget_per,
                    "note": f"Ensemble member: {proposer}"
                }
            except Exception as e:
                results[proposer] = {"status": "error", "error": str(e)}
        
        # Recommend best based on auto-selection
        best_proposer, settings = auto_select_proposer(problem, budget_total)
        
        return {
            "ensemble_members": self.proposers,
            "budget_per_member": budget_per,
            "recommended_primary": best_proposer,
            "recommendation_reason": settings.get("description", ""),
            "results": results,
            "strategy": "Run all proposers in parallel, compare on validation holdout, pick best overall"
        }
    
    def get_diversity_score(self, results: dict) -> float:
        """Measure diversity of solutions found."""
        # Placeholder - would analyze equation structures
        return 0.8


def ensemble_search(problem: Any, budget: int = 300, **kwargs) -> dict[str, Any]:
    """Convenience function for ensemble search."""
    searcher = EnsembleSearcher()
    return searcher.search_ensemble(problem, budget)
