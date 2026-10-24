# Databricks notebook source
from __future__ import annotations

import numpy as np

from apedp.decision import JevCompatibleBackend, LocalSearchController, SearchAction
from apedp.decision_benchmark import benchmark_controller
from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.neural_discovery.integration import ModelDiscoveryIntegration
from apedp.neural_discovery.symbolic import ExpressionTokenizer
from apedp.neural_discovery.validation import run_full_benchmark


class FakeJev:
    def decide(self, state, choices):
        assert "expand_grammar" in choices
        return {"action": "run_validation", "reason": "frontier review", "parameters": {}}


class BrokenJev:
    def decide(self, state):
        raise RuntimeError("offline")


class LinearModel:
    def apply(self, variables, x):
        return np.asarray(x[:, :1]) * 2.0


def test_jev_compatible_backend_validates_typed_response():
    result = JevCompatibleBackend(FakeJev()).decide({"generation": 1})
    assert result.action == SearchAction.RUN_VALIDATION
    assert result.source == "jev"
    assert result.fallback_used is False


def test_jev_compatible_backend_falls_back_locally():
    backend = JevCompatibleBackend(BrokenJev(), LocalSearchController(stagnation_limit=1))
    result = backend.decide({"generations_without_improvement": 1, "final_normalized_score": 1.0})
    assert result.action == SearchAction.EXPAND_GRAMMAR
    assert result.source == "local_fallback"
    assert result.fallback_used is True
    assert backend.failures == 1


def test_controller_benchmark_is_offline_and_measurable():
    states = [
        {"generation": 0, "generations_without_improvement": 0, "final_normalized_score": 1.0, "pareto_frontier_size": 0},
        {"generation": 3, "generations_without_improvement": 3, "final_normalized_score": 1.0, "pareto_frontier_size": 0},
    ]
    controller = LocalSearchController()
    report = benchmark_controller(states, controller, baseline=LocalSearchController())
    assert report.total == 2
    assert report.action_agreement == 1.0
    assert report.fallback_count == 0


def test_experiment_suggestion_separates_labeled_data_from_pool():
    integration = ModelDiscoveryIntegration(
        NeuralDiscoveryConfig(hidden_dims=(4,), num_layers=1, max_epochs=1, ensemble_size=2)
    )
    labeled = {"x": np.arange(6.0)}
    pool = {"x": np.arange(6.0, 10.0)}
    result = integration.suggest_next_experiment(labeled, np.arange(6.0), pool, n_select=2)
    assert len(result["indices"]) == 2
    assert result["candidate_count"] == 4


def test_symbolic_tokenizer_parses_prefix_structure():
    tokenizer = ExpressionTokenizer(n_variables=2, max_len=16)
    tokens = tokenizer.tokenize("x0 + sqrt(x1)")
    assert tokens[1] == tokenizer.vocab_to_id["ADD"]
    assert tokenizer.vocab_to_id["VAR_0"] in tokens
    assert tokenizer.vocab_to_id["SQRT"] in tokens


def test_benchmark_report_contains_measured_predictive_metrics():
    report = run_full_benchmark(
        NeuralDiscoveryConfig(),
        LinearModel(),
        {},
        {"linear": (np.arange(5.0).reshape(-1, 1), 2.0 * np.arange(5.0))},
    )
    assert report.summary["benchmark_status"] == "MEASURED"
    assert report.summary["mean_r2"] > 0.99
    assert report.results[0].metrics["mse"] < 1e-10
