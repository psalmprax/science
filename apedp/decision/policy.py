# Databricks notebook source
"""Decision backends for evidence review and adaptive search control."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DecisionBackend(ABC):
    @abstractmethod
    def decide(self, state: dict) -> dict:
        """Return ``{'decision': str, 'probabilities': {decision: prob}}``."""
        raise NotImplementedError


class SearchAction(str, Enum):
    CONTINUE = "continue"
    INCREASE_EXPLORATION = "increase_exploration"
    INCREASE_EXPLOITATION = "increase_exploitation"
    EXPAND_GRAMMAR = "expand_grammar"
    RUN_VALIDATION = "run_validation"
    STOP_STAGNATION = "stop_stagnation"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class SearchDecision:
    action: SearchAction
    reason: str
    parameters: dict[str, float | bool]
    source: str = "local"
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {"action": self.action.value, "reason": self.reason, "parameters": dict(self.parameters), "source": self.source, "fallback_used": self.fallback_used}


class RuleBasedBackend(DecisionBackend):
    """Deterministic evidence policy and measurable controller baseline."""

    def decide(self, state: dict) -> dict:
        def one(decision: str) -> dict:
            return {"decision": decision, "probabilities": {decision: 1.0}}
        if state.get("mathematical_passed") is False:
            return one("request_derivation")
        if state.get("numerical_passed") is False:
            return one("run_more_tests")
        if int(state.get("independent_datasets", 0)) < 2:
            return one("seek_empirical_validation")
        if state.get("empirical_passed"):
            return one("accept_for_benchmark")
        return one("human_review")


class LocalSearchController:
    """Offline adaptive policy and baseline for a Jev-compatible controller.

    The policy is an escalation ladder, evaluated in this order:

    1. no finite incumbent score -> HUMAN_REVIEW;
    2. stagnated at the limit while still above the residual bar ->
       EXPAND_GRAMMAR, because widening the search space is the right first
       response to a stall;
    3. the same, but the stall has now run to ``stagnation_limit *
       ESCALATION_FACTOR`` -> STOP_STAGNATION;
    4. large Pareto frontier -> RUN_VALIDATION;
    5. mild stagnation -> INCREASE_EXPLORATION;
    6. otherwise -> CONTINUE.

    Rungs 2 and 3 both exist because "expand the grammar" is only useful
    advice once. Handing out the same rung-2 advice on every chunk, forever,
    is how a search that has plainly stalled goes on spending generations
    while its ``stopped_reason`` stays ``None`` -- the caller has no way to
    tell that from a search that is genuinely still making progress.

    Rung 1 sits above the rest on purpose. Every other rung is conditioned on
    a usable score: "above the residual bar", "large enough frontier", "not
    improved". With a missing or NaN score none of those comparisons mean
    anything, and ``nan > 1e-3`` is ``False``, so without this guard a NaN
    falls through to CONTINUE and a search with no incumbent at all looks
    healthy.
    """

    #: How far past ``stagnation_limit`` a stall must run before the
    #: controller gives up on widening the grammar and stops the search.
    ESCALATION_FACTOR = 2

    def __init__(self, stagnation_limit: int = 5, validation_frontier_limit: int = 25):
        if stagnation_limit < 1 or validation_frontier_limit < 1:
            raise ValueError("controller limits must be positive")
        self.stagnation_limit = stagnation_limit
        self.validation_frontier_limit = validation_frontier_limit

    def decide(self, state: dict) -> SearchDecision:
        stagnation = int(state.get("generations_without_improvement", 0))
        normalized = float(state.get("final_normalized_score", float("inf")))
        frontier = int(state.get("pareto_frontier_size", 0))
        generation = int(state.get("generation", 0))
        if not math.isfinite(normalized):
            return SearchDecision(SearchAction.HUMAN_REVIEW, "no finite incumbent score: there is nothing for a policy to steer, and widening the grammar cannot invent one", {"requires_finite_incumbent": True})
        if stagnation >= self.stagnation_limit and normalized > 1e-3:
            if stagnation >= self.stagnation_limit * self.ESCALATION_FACTOR:
                return SearchDecision(SearchAction.STOP_STAGNATION, f"no improvement for {stagnation} generations, including after a wider grammar was requested; the incumbent is still above the residual bar", {"generations_without_improvement": float(stagnation)})
            return SearchDecision(SearchAction.EXPAND_GRAMMAR, "search stagnated above the residual bar", {"allow_power_mutation": True, "allow_rational_mutation": True})
        if frontier >= self.validation_frontier_limit:
            return SearchDecision(SearchAction.RUN_VALIDATION, "Pareto frontier is large enough for an independent check", {"frontier_size": float(frontier)})
        if generation > 0 and stagnation >= 2:
            return SearchDecision(SearchAction.INCREASE_EXPLORATION, "recent generations have not improved the incumbent", {"allow_compose_mutation": True, "allow_logcomp_mutation": True})
        return SearchDecision(SearchAction.CONTINUE, "search telemetry is within baseline bounds", {})


class JevCompatibleBackend:
    """Adapter for a typed local or remote Jev-style decision client."""

    def __init__(self, client: Any, fallback: LocalSearchController | None = None):
        if client is None or not callable(getattr(client, "decide", None)):
            raise TypeError("client must provide a callable decide method")
        self.client = client
        self.fallback = fallback or LocalSearchController()
        self.failures = 0

    def decide(self, state: dict) -> SearchDecision:
        choices = [action.value for action in SearchAction]
        try:
            try:
                raw = self.client.decide(state, choices)
            except TypeError:
                raw = self.client.decide(state)
            if isinstance(raw, SearchDecision):
                return SearchDecision(raw.action, raw.reason, raw.parameters, source="jev", fallback_used=False)
            if not isinstance(raw, dict):
                raise TypeError("client response must be a mapping or SearchDecision")
            action = SearchAction(str(raw.get("action", "")))
            parameters = raw.get("parameters", {})
            if not isinstance(parameters, dict):
                raise TypeError("client parameters must be a mapping")
            return SearchDecision(action, str(raw.get("reason", "jev decision")), parameters, source="jev", fallback_used=False)
        # Adapter boundary: ``self.client`` is injected and typed ``Any``, so
        # its failure modes cannot be enumerated. Any client failure must
        # degrade to the local policy and is counted in ``self.failures``;
        # BLE001 is deliberate here.
        except Exception:  # noqa: BLE001 - untyped injected client
            self.failures += 1
            local = self.fallback.decide(state)
            return SearchDecision(local.action, f"Jev fallback: {local.reason}", local.parameters, source="local_fallback", fallback_used=True)


class JevBackend(DecisionBackend):
    """Optional evidence-decision adapter for a typed Jev-compatible client."""

    def __init__(self, client=None):
        self.client = client

    def decide(self, state: dict) -> dict:
        if self.client is None:
            raise NotImplementedError("JevBackend has no client: supply a typed-decision client implementing decide(input, choices) first.")
        result = self.client.decide(state)
        if not isinstance(result, dict) or "decision" not in result:
            raise TypeError("Jev client must return a decision dictionary")
        return result
