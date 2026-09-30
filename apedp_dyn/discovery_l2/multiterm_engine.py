"""Multi-Term Superposition Discovery Engine for L2 Dynamics.

Extends L2 open discovery from single-term expressions to multi-term physical balances:
    Y = \\sum_{k=1}^K c_k f_k(X)

Key Architectural Gates:
  1. Autonomous Basis Term Generation: Terms f_k(X) are generated via physically typed AST
     rather than human-authored candidate dictionaries.
  2. Fisher Information Matrix (FIM) / Collinearity Gate:
     For any candidate combination of terms F = [f_1(X), ..., f_K(X)], the Gram matrix
     G = F^T F must satisfy condition number \\kappa(G) <= 10^3. Combinations with internal
     collinearity are pruned before parameter fitting.
  3. Multi-Variable OLS & Residual Whiteness:
     Fits coefficients c_k on the train split, evaluates holdout RMSE and residual whiteness.
  4. Pareto Frontier Complexity Selection:
     Complexity is defined as \\sum complexity(f_k) + (K - 1). A multi-term model is admitted
     to the Pareto frontier IF AND ONLY IF it strictly improves holdout RMSE over all simpler
     models by at least min_improvement.
  5. Sovereign Epistemic Adjudication:
     Surviving Pareto candidates are submitted to adjudicate() with L2_OPEN_DISCOVERY.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2.ast_generator import (
    ASTNode,
    generate_dimensionally_typed_ast,
)
from apedp_dyn.discovery_l2.dimensional_types import BaseDimension, Dimension
from apedp_dyn.discovery_l2.pruning import (
    check_residual_whiteness,
    gate_a_algebraic_validity,
    gate_b_physical_validity,
    select_pareto_candidates,
)
from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    adjudicate,
)
from apedp_dyn.fitting.parameter_fit import FitResult
from apedp_dyn.gates.intervention import observation_scale
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


@dataclass(frozen=True)
class MultiTermDiscoveryResult:
    """Result of multi-term L2 open discovery."""

    adjudication: Any
    promoted_equation: str
    pareto_candidates: list[ScoredCandidate]
    all_evaluated_count: int
    collinear_pruned_count: int


class MultiTermL2Engine:
    """Discovers multi-term governing equations Y = sum c_k f_k(X) from raw data."""

    def __init__(
        self,
        max_depth: int = 2,
        max_term_complexity: int = 4,
        max_terms: int = 3,
        max_condition_number: float = 1000.0,
        operators: Sequence[str] = ("+", "-", "*", "/", "pow", "exp", "log", "sin", "cos"),
    ) -> None:
        self.max_depth = max_depth
        self.max_term_complexity = max_term_complexity
        self.max_terms = max_terms
        self.max_condition_number = max_condition_number
        self.operators = operators

    def discover_multiterm(
        self,
        X: Mapping[str, np.ndarray],
        Y: np.ndarray,
        dimensions: Mapping[str, BaseDimension],
        target_dimension: BaseDimension,
        target_name: str = "y",
        train_ratio: float = 0.6,
        data_origin: str = DataOrigin.SIMULATION,
        bootstrap: bool = False,
        seed: int = 42,
    ) -> MultiTermDiscoveryResult:
        """Execute autonomous multi-term L2 discovery."""
        y_arr = np.asarray(Y, dtype=float).ravel()
        n_samples = len(y_arr)
        data_vars = {k: np.asarray(v, dtype=float).ravel() for k, v in X.items()}

        n_train = max(4, int(n_samples * train_ratio))
        X_tr = {k: v[:n_train] for k, v in data_vars.items()}
        X_ho = {k: v[n_train:] for k, v in data_vars.items()}
        y_tr = y_arr[:n_train]
        y_ho = y_arr[n_train:]

        # Step 1: Autonomous Basis Term Generation via Physically Typed AST
        dim_dict = {k: (v if isinstance(v, Dimension) else Dimension()) for k, v in dimensions.items()}
        target_dim = target_dimension if isinstance(target_dimension, Dimension) else Dimension()

        raw_nodes = generate_dimensionally_typed_ast(
            variables=dim_dict,
            operators=self.operators,
            max_depth=self.max_depth,
            max_complexity=self.max_term_complexity,
        )

        # Filter valid terms through Gate A and Gate B
        y_std = float(np.std(y_arr))
        valid_basis_terms: list[ASTNode] = []
        term_vectors_tr: list[np.ndarray] = []
        term_vectors_ho: list[np.ndarray] = []

        for node in raw_nodes:
            ok_a_tr, _, _ = gate_a_algebraic_validity(node, X_tr, y_std=y_std)
            ok_a_ho, _, _ = gate_a_algebraic_validity(node, X_ho, y_std=y_std)
            if not (ok_a_tr and ok_a_ho):
                continue
            ok_b, _ = gate_b_physical_validity(node, target_dim)
            if not ok_b:
                continue

            # Evaluate basis term
            try:
                v_tr = node.evaluate(X_tr)
                v_ho = node.evaluate(X_ho)
                if np.all(np.isfinite(v_tr)) and np.all(np.isfinite(v_ho)) and not np.all(v_tr == 0):
                    valid_basis_terms.append(node)
                    term_vectors_tr.append(v_tr)
                    term_vectors_ho.append(v_ho)
            except (ArithmeticError, ValueError, TypeError):
                continue

        if not valid_basis_terms:
            adj = adjudicate([], data_origin=data_origin, benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY)
            return MultiTermDiscoveryResult(
                adjudication=adj,
                promoted_equation="",
                pareto_candidates=[],
                all_evaluated_count=0,
                collinear_pruned_count=0,
            )

        # Step 2: Form Combinations of 1 to max_terms Basis Functions
        scored_candidates: list[ScoredCandidate] = []
        collinear_pruned = 0
        obs_scale = observation_scale(y_ho)
        obs_rms = float(obs_scale.get("obs_rms", np.sqrt(np.mean(y_ho**2))))
        noise_sigma = float(obs_scale.get("noise_sigma", 0.01))
        if noise_sigma <= 0.0:
            noise_sigma = 0.01

        m_terms = len(valid_basis_terms)
        for k in range(1, min(self.max_terms + 1, m_terms + 1)):
            for idx_tuple in itertools.combinations(range(m_terms), k):
                # Construct feature matrices
                M_tr = np.column_stack([term_vectors_tr[i] for i in idx_tuple])
                M_ho = np.column_stack([term_vectors_ho[i] for i in idx_tuple])

                # FIM / Collinearity Gate: condition number of Gram matrix
                if k > 1:
                    gram = M_tr.T @ M_tr
                    s_vals = np.linalg.svd(gram, compute_uv=False)
                    cond = float(s_vals[0] / s_vals[-1]) if s_vals[-1] > 1e-15 else float("inf")
                    if cond > self.max_condition_number:
                        collinear_pruned += 1
                        continue

                # OLS Parameter Fitting
                try:
                    coefs, *_ = np.linalg.lstsq(M_tr, y_tr, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                pred_tr = M_tr @ coefs
                pred_ho = M_ho @ coefs
                loss_tr = float(np.mean((y_tr - pred_tr) ** 2))
                resid_ho = y_ho - pred_ho
                rmse_ho = float(np.sqrt(np.mean(resid_ho**2)))

                # Residual Whiteness
                is_white, rho1 = check_residual_whiteness(resid_ho, threshold=0.5)

                # Gate 4 evidential pass criteria
                ss_tot = float(np.sum((y_ho - np.mean(y_ho)) ** 2))
                ss_res = float(np.sum(resid_ho**2))
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-14 else (1.0 if rmse_ho < 1e-6 else 0.0)

                passed_gate4 = bool(
                    np.isfinite(rmse_ho)
                    and (r2 >= 0.20 or rmse_ho <= 1e-5)
                    and (rmse_ho <= 1.5 * obs_rms if obs_rms > 0 else True)
                )

                # Assemble Candidate Dynamics IR
                chosen_nodes = [valid_basis_terms[i] for i in idx_tuple]
                term_objects = tuple(
                    Term(node.to_str(), f"theta_{j+1}") for j, node in enumerate(chosen_nodes)
                )
                cid = " + ".join(f"theta_{j+1}*({node.to_str()})" for j, node in enumerate(chosen_nodes))
                total_complexity = sum(node.complexity for node in chosen_nodes) + (k - 1)

                param_vals = {f"theta_{j+1}": float(coefs[j]) for j in range(k)}
                fit_obj = FitResult(param_values=param_vals, train_loss=loss_tr, nfev=1, success=True, message="Multi-OLS")

                cand_dyn = CandidateDynamics(
                    lhs=target_name,
                    terms=term_objects,
                    candidate_id=cid,
                    metadata={
                        "complexity": total_complexity,
                        "n_terms": k,
                        "canonical_expr": " + ".join(node.canonical_str() for node in chosen_nodes),
                        "param_values": param_vals,
                    },
                )

                g4_dict = {
                    "passed": passed_gate4,
                    "holdout_obs_rms": obs_rms,
                    "holdout_noise_sigma": noise_sigma,
                    "holdout_n_samples": len(y_ho),
                    "rmse": rmse_ho,
                    "r2": r2,
                    "rho1": rho1,
                    "is_white": is_white,
                    "holdout_residuals": resid_ho.tolist(),
                    "holdout_sq_error": (resid_ho**2).tolist(),
                }

                sc = ScoredCandidate(
                    cand_dyn,
                    fit=fit_obj,
                    train_loss=loss_tr,
                    holdout_rmse=rmse_ho,
                    provenance={
                        "benchmark_level": BenchmarkLevel.L2_OPEN_DISCOVERY,
                        "data_origin": data_origin,
                        "gate4_status": g4_dict,
                    },
                )
                scored_candidates.append(sc)

        # Step 3: Pareto Frontier Selection
        pareto = select_pareto_candidates(scored_candidates)

        # Step 4: Epistemic Adjudication
        r_best = pareto[0].provenance.get("gate4_status", {}).get("holdout_residuals") if pareto else None
        r_rival = pareto[1].provenance.get("gate4_status", {}).get("holdout_residuals") if len(pareto) > 1 else None

        adj = adjudicate(
            pareto,
            resid_best=r_best,
            resid_rival=r_rival,
            bootstrap=bootstrap,
            bootstrap_n_boot=1000,
            bootstrap_seed=seed,
            sole_survivor_rule=True,
            data_origin=data_origin,
            benchmark_level=BenchmarkLevel.L2_OPEN_DISCOVERY,
        )

        inferred_eq = "[NO PROMOTED EQUATION]"
        if adj.promoted is not None:
            best = adj.promoted
            terms_str = []
            for t_obj in best.candidate.terms:
                val = best.fit.param_values.get(t_obj.parameter, 1.0) if best.fit else 1.0
                terms_str.append(f"{val:+.4g} * ({t_obj.name})")
            inferred_eq = f"{target_name} = " + " ".join(terms_str)

        return MultiTermDiscoveryResult(
            adjudication=adj,
            promoted_equation=inferred_eq,
            pareto_candidates=pareto,
            all_evaluated_count=len(scored_candidates),
            collinear_pruned_count=collinear_pruned,
        )
