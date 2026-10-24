# Databricks notebook source
"""Offline benchmark utilities for adaptive decision controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apedp.decision import LocalSearchController, SearchDecision


@dataclass(frozen=True)
class ControllerBenchmark:
    total: int
    action_agreement: float
    fallback_count: int
    action_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "action_agreement": self.action_agreement, "fallback_count": self.fallback_count, "action_counts": dict(self.action_counts)}


def benchmark_controller(states: list[dict[str, Any]], candidate, baseline: LocalSearchController | None = None) -> ControllerBenchmark:
    """Compare a candidate controller against the local deterministic baseline."""
    baseline = baseline or LocalSearchController()
    agreements = 0
    action_counts: dict[str, int] = {}
    for state in states:
        expected = baseline.decide(state)
        actual = candidate.decide(state)
        if not isinstance(actual, SearchDecision):
            raise TypeError("controller must return SearchDecision")
        action_counts[actual.action.value] = action_counts.get(actual.action.value, 0) + 1
        agreements += actual.action == expected.action
    total = len(states)
    return ControllerBenchmark(total, agreements / total if total else 1.0, int(getattr(candidate, "failures", 0)), action_counts)


__all__ = ["ControllerBenchmark", "benchmark_controller"]