"""L2 Open Discovery Engine with Physically Typed AST and Epistemic Adjudication.

Fulfills the L2 Open Discovery contract:
  1. Core Contract:
     - Receives ONLY raw data D = {X, t, Y}, variable names, SI dimensions, and primitive operators.
     - Never receives expected equation families, expected exponents, or hand-written candidate templates.
  2. Dimensionally Typed AST Generation:
     - Enforces dimensional homogeneity during AST construction.
  3. Three-Stage Pruning:
     - Gate A: Algebraic validity.
     - Gate B: Physical validity against target LHS dimension.
     - Gate C: Parameter estimation, holdout scoring, residual whiteness, Pareto complexity selection.
  4. Adjudication:
     - Hands surviving Pareto candidates to epistemic adjudication at BenchmarkLevel.L2_OPEN_DISCOVERY.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from apedp_dyn.discovery_l2.ast_generator import (
    ASTNode,
    generate_dimensionally_typed_ast,
)
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.pruning import (
    fit_and_score_candidate,
    gate_a_algebraic_validity,
    gate_b_physical_validity,
    select_pareto_candidates,
)
from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    BenchmarkLevel,
    DataOrigin,
    adjudicate,
)
from apedp_dyn.epistemic.term_necessity import GuardReport, guard_selection
from apedp_dyn.grammar.candidate_ir import ScoredCandidate


@dataclass
class L2DiscoveryResult:
    """Complete audit record of an L2 open discovery execution."""

    adjudication: Adjudication
    candidates: list[ScoredCandidate] = field(default_factory=list)
    pareto_candidates: list[ScoredCandidate] = field(default_factory=list)
    generated_nodes: list[ASTNode] = field(default_factory=list)
    gate_a_passed: list[ASTNode] = field(default_factory=list)
    gate_b_passed: list[ASTNode] = field(default_factory=list)
    guard_report: GuardReport | None = None


class L2DiscoveryEngine:
    """Autonomous Level 2 Open Equation Discovery Engine."""

    def __init__(
        self,
        operators: Sequence[str] = ("+", "-", "*", "/", "pow", "exp", "log", "sin", "cos", "abs"),
        max_depth: int = 2,
        max_complexity: int = 6,
    ) -> None:
        self.operators = tuple(operators)
        self.max_depth = max_depth
        self.max_complexity = max_complexity

    def discover(
        self,
        X: Mapping[str, np.ndarray],
        Y: np.ndarray,
        dimensions: Mapping[str, Dimension],
        target_dimension: Dimension,
        t: np.ndarray | None = None,
        target_name: str = "Y",
        noise_sigma: float | None = None,
        train_ratio: float = 0.5,
        bootstrap: bool = False,
        data_origin: str = DataOrigin.SIMULATION,
        fit_intercept: bool = False,
        seed: int = 42,
        X_holdout: Mapping[str, np.ndarray] | None = None,
        Y_holdout: np.ndarray | None = None,
    ) -> L2DiscoveryResult:
        """Execute unconstrained L2 discovery pipeline from raw data and dimensions."""
        y_arr = np.asarray(Y, dtype=float)
        n_samples = len(y_arr)

        data_vars: dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in X.items()}
        if t is not None and "t" not in data_vars:
            data_vars["t"] = np.asarray(t, dtype=float)

        all_dimensions: dict[str, Dimension] = dict(dimensions)
        if "t" in data_vars and "t" not in all_dimensions:
            all_dimensions["t"] = Dimension(t=1)

        for k, v in data_vars.items():
            if len(v) != n_samples:
                raise ValueError(
                    f"Sample count mismatch for variable '{k}': expected {n_samples}, got {len(v)}"
                )

        if n_samples < 4:
            adj = adjudicate(
                [],
                data_origin=data_origin,
                benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY,
            )
            return L2DiscoveryResult(adjudication=adj)

        if X_holdout is not None and Y_holdout is not None:
            X_tr = data_vars
            y_tr = y_arr
            X_ho = {k: np.asarray(v, dtype=float).ravel() for k, v in X_holdout.items()}
            y_ho = np.asarray(Y_holdout, dtype=float).ravel()
        else:
            n_train = max(2, int(n_samples * train_ratio))
            X_tr = {k: v[:n_train] for k, v in data_vars.items()}
            X_ho = {k: v[n_train:] for k, v in data_vars.items()}
            y_tr = y_arr[:n_train]
            y_ho = y_arr[n_train:]
        y_std = float(np.std(y_arr))

        # Step 1: Physically Typed AST Construction
        # Builds dimensionally valid trees without template hinting
        generated_nodes = generate_dimensionally_typed_ast(
            variables=all_dimensions,
            operators=self.operators,
            max_depth=self.max_depth,
            max_complexity=self.max_complexity,
        )

        # Step 2: Gate A (Algebraic Validity across train and holdout domains)
        gate_a_passed: list[ASTNode] = []
        for node in generated_nodes:
            ok_tr, _why_tr, _ = gate_a_algebraic_validity(node, X_tr, y_std=y_std)
            if not ok_tr:
                continue
            ok_ho, _why_ho, _ = gate_a_algebraic_validity(node, X_ho, allow_constant=True)
            if ok_ho:
                gate_a_passed.append(node)

        # Step 3: Gate B (Physical Validity against Target LHS Dimension)
        gate_b_passed: list[ASTNode] = []
        for node in gate_a_passed:
            ok, _why = gate_b_physical_validity(node, target_dimension)
            if ok:
                gate_b_passed.append(node)

        # If zero candidates match physical dimension, epistemic layer returns NO_VALID_CANDIDATE
        if not gate_b_passed:
            adj = adjudicate(
                [],
                data_origin=data_origin,
                benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY,
            )
            return L2DiscoveryResult(
                adjudication=adj,
                generated_nodes=generated_nodes,
                gate_a_passed=gate_a_passed,
                gate_b_passed=gate_b_passed,
            )

        # Step 4: Gate C (Evidential Validity & Parameter Fitting)
        scored_candidates: list[ScoredCandidate] = []
        for node in gate_b_passed:
            sc = fit_and_score_candidate(
                node=node,
                X_tr=X_tr,
                y_tr=y_tr,
                X_ho=X_ho,
                y_ho=y_ho,
                target_name=target_name,
                noise_sigma=noise_sigma,
                data_origin=data_origin,
                fit_intercept=fit_intercept,
            )
            scored_candidates.append(sc)

        # Step 5: Pareto Complexity Selection
        # Selects the non-dominated Pareto frontier and competitive candidates
        pareto_candidates = select_pareto_candidates(scored_candidates)

        # Step 5b: Term-necessity / near-tie guard (Phase-1 wire-in, §68-§69).
        # Purely restrictive additive layer: drops debris-carrying candidates,
        # records the top near-tie class, never manufactures evidence.
        # Empty in -> empty out -> adjudicate([]) -> NO_VALID path (rule 8).
        guarded, guard_report = guard_selection(
            pareto_candidates, X_tr, y_tr, X_ho, y_ho,
        )

        # Step 6: Epistemic Adjudication
        ranked = sorted(
            [
                s
                for s in guarded
                if s.train_loss is not None and np.isfinite(s.train_loss)
            ],
            key=lambda s: s.train_loss,
        )

        r_best = None
        r_rival = None
        if len(ranked) >= 1:
            r_best = ranked[0].provenance.get("gate4_status", {}).get("holdout_residuals")
        if len(ranked) >= 2:
            r_rival = ranked[1].provenance.get("gate4_status", {}).get("holdout_residuals")

        adj = adjudicate(
            guarded,
            resid_best=r_best,
            resid_rival=r_rival,
            bootstrap=bootstrap,
            bootstrap_n_boot=1000,
            bootstrap_seed=seed,
            sole_survivor_rule=True,
            data_origin=data_origin,
            benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY,
        )

        return L2DiscoveryResult(
            adjudication=adj,
            candidates=scored_candidates,
            pareto_candidates=pareto_candidates,
            generated_nodes=generated_nodes,
            gate_a_passed=gate_a_passed,
            gate_b_passed=gate_b_passed,
            guard_report=guard_report,
        )
