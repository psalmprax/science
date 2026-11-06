"""Ensemble search across multiple proposers.

Tries different proposers in parallel and combines results to reduce
chances of getting stuck in local optima.
"""
from __future__ import annotations

from typing import Any

from apedp.proposer_router import ProposerRouter, auto_select_proposer

#: Proposer names the router actually knows how to drive.
KNOWN_PROPOSERS = frozenset(
    name for names in ProposerRouter().proposer_map.values() for name in names)

#: Keys under which a member's result may carry the solutions it found.
SOLUTION_KEYS = ("best", "expression", "expressions", "equations", "solutions")


def _solutions_in(value: Any) -> list[str]:
    """Collect the expression strings one result entry carries, depth-first."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [solution for key in SOLUTION_KEYS if key in value
                for solution in _solutions_in(value[key])]
    if isinstance(value, (list, tuple)):
        return [solution for item in value for solution in _solutions_in(item)]
    return []


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
        """How far the reported solutions are from all being the same.

        Diversity is a property of what was found, so this needs the solutions:
        each member's entry is searched for ``best`` / ``expression`` /
        ``expressions`` / ``equations`` / ``solutions``. The score is where the
        distinct count sits in its possible range, ``(distinct - 1) / (total - 1)``:
        0.0 when every member returned the same solution, 1.0 when no two agree.
        A plan with fewer than two solutions in it — including this class's own
        ``search_ensemble`` output before anything has run — has no diversity to
        measure, and this raises rather than reporting a constant.
        """
        solutions = [solution for entry in results.values()
                     for solution in _solutions_in(entry)]
        if len(solutions) < 2:
            raise ValueError(
                "get_diversity_score needs at least two reported solutions to "
                f"measure; results carries {len(solutions)}")
        return (len(set(solutions)) - 1) / (len(solutions) - 1)


def ensemble_search(problem: Any, budget: int = 300, **kwargs) -> dict[str, Any]:
    """Convenience function for ensemble search."""
    searcher = EnsembleSearcher()
    return searcher.search_ensemble(problem, budget_total=budget, **kwargs)
