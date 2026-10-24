"""Multi-Seed Discovery Stability Harness & Identifiability Matrix Generator.

Orchestrates multi-seed discovery runs to rigorously separate:
  - Discovery Stability (structural recurrence, Shannon entropy, parameter dispersion)
  - Structural Identification (separation vs. observational equivalence within tau_epi)
  - Transfer Generalization (cross-regime leadership survival vs. failure)
  - Epistemic Promotion (promotion rights vs. diagnostic abstention)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import BaseDimension
from apedp_dyn.epistemic.discovery_stability import (
    compose_identifiability_stability_matrix,
    stability_report,
)


@dataclass(frozen=True)
class StabilityAuditResult:
    """Complete diagnostic audit of discovery stability and identifiability."""

    matrix: dict[str, Any]
    n_seeds: int
    structures: list[str]
    parameter_samples: dict[str, list[float]]
    stability_metrics: dict[str, Any]
    baseline_result: Any


def run_discovery_stability_audit(
    engine: Any,
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    dimensions: Mapping[str, BaseDimension],
    target_dimension: BaseDimension,
    seeds: Sequence[int] = tuple(range(10)),
    train_ratio: float = 0.6,
    transfer_scope: Any | None = None,
    target_name: str = "Y",
) -> StabilityAuditResult:
    """Execute multi-seed discovery stability audit and compose the stability matrix."""
    structures: list[str] = []
    param_samples: dict[str, list[float]] = {}
    baseline_res = None

    for i, s in enumerate(seeds):
        if hasattr(engine, "discover_multiterm"):
            res = engine.discover_multiterm(
                X=X,
                Y=Y,
                dimensions=dimensions,
                target_dimension=target_dimension,
                target_name=target_name,
                train_ratio=train_ratio,
                bootstrap=False,
                seed=s,
            )
            adj = res.adjudication
        else:
            res = engine.discover(
                X=X,
                Y=Y,
                dimensions=dimensions,
                target_dimension=target_dimension,
                target_name=target_name,
                train_ratio=train_ratio,
                bootstrap=False,
                seed=s,
            )
            adj = res.adjudication

        if i == 0:
            baseline_res = res

        # Structural extraction
        if adj.promoted is not None:
            p_cand = adj.promoted
            s_id = (
                p_cand.provenance.get("canonical_expr")
                or p_cand.candidate.metadata.get("canonical_expr")
                or p_cand.candidate.candidate_id
            )
            structures.append(s_id)
            if p_cand.fit and p_cand.fit.param_values:
                for k, v in p_cand.fit.param_values.items():
                    param_samples.setdefault(k, []).append(float(v))
        elif hasattr(res, "pareto_candidates") and res.pareto_candidates:
            top = res.pareto_candidates[0]
            s_id = (
                top.provenance.get("canonical_expr")
                or top.candidate.metadata.get("canonical_expr")
                or top.candidate.candidate_id
            )
            structures.append(f"unpromoted:{s_id}")
            if top.fit and top.fit.param_values:
                for k, v in top.fit.param_values.items():
                    param_samples.setdefault(k, []).append(float(v))
        else:
            structures.append("NO_VALID_CANDIDATE")

    rep = stability_report(structures, param_samples=param_samples)

    search_scope = {
        "type": "L2_OPEN" if not hasattr(engine, "max_terms") else "L2_MULTITERM",
        "max_depth": getattr(engine, "max_depth", 2),
        "operators": list(getattr(engine, "operators", ())),
    }

    matrix = compose_identifiability_stability_matrix(
        search_scope=search_scope,
        stability_rep=rep,
        adjudication=baseline_res.adjudication if baseline_res else None,
        transfer_scope=transfer_scope,
    )

    return StabilityAuditResult(
        matrix=matrix,
        n_seeds=len(seeds),
        structures=structures,
        parameter_samples=param_samples,
        stability_metrics=rep,
        baseline_result=baseline_res,
    )
