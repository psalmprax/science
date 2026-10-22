"""APEDP-X differential-equation discovery arm (separate arm, shared infra).

Boundary contract (external review, RIME-APEDP-X DE-support thread):
  - This arm OWNS differential-equation discovery logic (typed static /
    ODE-system / PDE-system semantics, Level-1 dependency-graph screening
    with an escape hatch, and its own candidate libraries).
  - It REUSES shared infrastructure instead of duplicating it: the RIME
    expression/arena machinery stays untouched; the coupled-ODE kernel path
    (``discover_coupled_system``) remains the dense-library engine; the
    existing finite-difference stencils remain the derivative source for
    MEASURED data (data is not an AD-differentiable computation).
  - The AD service is SHARED infrastructure, not a discovery arm.
  - Every produced object is referee-facing HANDOFF material. The arm never
    certifies its own candidates: no acceptance, promotion, or verdict
    field exists anywhere in this package; ``adjudicate()`` and the frozen
    protocol remain the sole authority.

Status: experimental (DE-discovery arm phase 1: typed semantics + AD
service + screened ODE discovery wired into the dispatch surface).
"""
from apedp_dyn.de_arm.ad_service import (
    Add,
    Const,
    Cos,
    Div,
    Exp,
    Expr,
    Log,
    Mul,
    Pow,
    Sin,
    Sub,
    Tanh,
    Var,
    differentiate,
    evaluate,
    numerical_check,
)
from apedp_dyn.de_arm.dependency_screen import (
    OdeArmOutcome,
    ScreenedStateFit,
    StateParentRanking,
    discover_ode_system,
    granger_screen,
)
from apedp_dyn.de_arm.typed_model import (
    ModelKind,
    OdeMathModel,
    PdeMathModel,
    StaticExpressionModel,
    TypedMathModel,
)

__all__ = [
    "Add",
    "Const",
    "Cos",
    "Div",
    "Exp",
    "Expr",
    "Log",
    "ModelKind",
    "Mul",
    "OdeArmOutcome",
    "OdeMathModel",
    "PdeMathModel",
    "Pow",
    "ScreenedStateFit",
    "Sin",
    "StateParentRanking",
    "StaticExpressionModel",
    "Sub",
    "Tanh",
    "TypedMathModel",
    "Var",
    "differentiate",
    "discover_ode_system",
    "evaluate",
    "granger_screen",
    "numerical_check",
]
