"""Ensemble search across multiple proposers.

Tries different proposers in parallel and combines results to reduce
chances of getting stuck in local optima.
"""
from __future__ import annotations

from typing import Any

from apedp.proposer_router import ProposerRouter, auto_select_proposer

#: Proposer names the router actually knows how to drive.
KNOWN_PROPOSERS = frozenset(
    name for names in ProposerRouter().proposer_map.values() for name in names
)


class EnsembleSearcher:
    """Plan a search across multiple proposers and pick the strongest.

    A member that is not a known proposer is reported as an error rather than
    planned as if it could run, and ``best`` stays ``None`` until a member
    actually returns a scored candidate.
    """
    
    def __init__(self, proposers: list[str] | None = None):
        self.proposers = proposers or ["rime_ad", "grammar", "library"]
    
    def search_ensemble(self, problem: Any, features: list[str] | None = None,
                        target: str | None = None,
                        budget_total: int = 300) -> dict[str, Any]:
        """Plan the ensemble; ``features``/``target`` describe the problem when known."""
        known = [p for p in self.proposers if p in KNOWN_PROPOSERS]
        budget_per = budget_total // max(1, len(known))
        results = {}
        
        for proposer in self.proposers:
            if proposer not in KNOWN_PROPOSERS:
                results[proposer] = {
                    "status": "error",
                    "error": f"unknown proposer {proposer!r}",
                    "budget": 0,
                }
                continue
            results[proposer] = {
                "status": "would_run",
                "budget": budget_per,
                "note": f"Ensemble member: {proposer}",
            }
        
        # Recommend best based on auto-selection
        best_proposer, settings = auto_select_proposer(problem, budget_total)
        
        return {
            "ensemble_members": self.proposers,
            "budget_per_member": budget_per,
            "features": features,
            "target": target,
            "recommended_primary": best_proposer,
            "recommendation_reason": settings.get("description", ""),
            "results": results,
            # No member has run yet, so there is no best candidate to report.
            "best": None,
            "strategy": "Run all proposers in parallel, compare on validation holdout, pick best overall"
        }
    
    def get_diversity_score(self, results: dict) -> float:
        """Measure diversity of solutions found."""
        # Placeholder - would analyze equation structures
        return 0.8


def ensemble_search(problem: Any, budget: int = 300, **kwargs) -> dict[str, Any]:
    """Convenience function for ensemble search."""
    searcher = EnsembleSearcher()
    return searcher.search_ensemble(problem, budget_total=budget, **kwargs)
