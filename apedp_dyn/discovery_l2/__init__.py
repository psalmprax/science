"""Level 2 Open Equation Discovery Module.

Autonomous equation discovery over raw data and SI dimensions without predefined
candidate vocabularies or equation families.
"""

from __future__ import annotations

from apedp_dyn.discovery_l2.active_intervention import (
    ProposedIntervention,
    compute_fisher_information_matrix,
    synthesize_discriminating_intervention,
)
from apedp_dyn.discovery_l2.ast_generator import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    PowNode,
    UnaryOpNode,
    VariableNode,
    generate_dimensionally_typed_ast,
)
from apedp_dyn.discovery_l2.dimensional_types import (
    DIMENSIONLESS,
    BaseDimension,
    Dimension,
    DimensionalError,
)
from apedp_dyn.discovery_l2.discovery_stability_harness import (
    StabilityAuditResult,
    run_discovery_stability_audit,
)
from apedp_dyn.discovery_l2.l2_engine import (
    L2DiscoveryEngine,
    L2DiscoveryResult,
)
from apedp_dyn.discovery_l2.multiterm_engine import (
    MultiTermDiscoveryResult,
    MultiTermL2Engine,
)
from apedp_dyn.discovery_l2.ode_engine import (
    L2ODEDiscoveryEngine,
    ODEDiscoveryResult,
    estimate_time_derivative,
)
from apedp_dyn.discovery_l2.pruning import (
    fit_and_score_candidate,
    gate_a_algebraic_validity,
    gate_b_physical_validity,
    select_pareto_candidates,
)

__all__ = [
    "DIMENSIONLESS",
    "ASTNode",
    "BaseDimension",
    "BinaryOpNode",
    "ConstantNode",
    "Dimension",
    "DimensionalError",
    "L2DiscoveryEngine",
    "L2DiscoveryResult",
    "L2ODEDiscoveryEngine",
    "MultiTermDiscoveryResult",
    "MultiTermL2Engine",
    "ODEDiscoveryResult",
    "PowNode",
    "ProposedIntervention",
    "StabilityAuditResult",
    "UnaryOpNode",
    "VariableNode",
    "compute_fisher_information_matrix",
    "estimate_time_derivative",
    "fit_and_score_candidate",
    "gate_a_algebraic_validity",
    "gate_b_physical_validity",
    "generate_dimensionally_typed_ast",
    "run_discovery_stability_audit",
    "select_pareto_candidates",
    "synthesize_discriminating_intervention",
]
