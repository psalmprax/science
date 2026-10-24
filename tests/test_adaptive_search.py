# Databricks notebook source
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp.adaptive_search import run_adaptive_discovery
from apedp.decision import LocalSearchController, SearchAction


class FakeSearch:
    allow_power_mutation = False
    allow_rational_mutation = False
    allow_compose_mutation = False
    allow_logcomp_mutation = False

    def __init__(self):
        self.calls = 0
        self.stagnation_report = {}

    def run_discovery(self, x_data, y_data, generations, pop_size, seed):
        self.calls += 1
        score = 1.0 / self.calls
        self.stagnation_report = {
            "generations_without_improvement": 0,
            "improvements": 1,
        }
        return sp.Symbol("x"), score, [
            {"generation": index, "best_score": score, "pareto_frontier_size": 1}
            for index in range(generations)
        ]


def test_local_controller_expands_after_stagnation():
    controller = LocalSearchController(stagnation_limit=3)
    decision = controller.decide({
        "generation": 4,
        "generations_without_improvement": 3,
        "final_normalized_score": 0.5,
        "pareto_frontier_size": 2,
    })
    assert decision.action == SearchAction.EXPAND_GRAMMAR
    assert decision.parameters["allow_power_mutation"] is True


def test_adaptive_runner_applies_controller_chunks():
    engine = FakeSearch()
    result = run_adaptive_discovery(
        engine,
        {"x": np.arange(10.0)},
        np.arange(10.0),
        generations=6,
        chunk_size=2,
        pop_size=4,
    )
    assert engine.calls == 3
    assert result.best_candidate == sp.Symbol("x")
    assert len(result.actions) == 3
    assert result.best_score == 1.0 / 3.0
