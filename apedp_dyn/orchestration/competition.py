"""Competition across all discovery engines on one dataset.

Every engine that applies to the data runs on the same training split. Each
engine's result is returned as its own record, side by side. This module never
picks a winner: promotion stays with the referee (``Adjudication``), and an
engine that does not apply to the data is recorded as ``NOT_APPLICABLE`` with
the reason, never silently dropped.

Engines that need a declared axis the upload does not have (for example ODE
needs a time axis) are marked ``NOT_APPLICABLE`` instead of being run on
invented axes, matching the dispatcher's no-invented-axis rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine

NOT_APPLICABLE = "NOT_APPLICABLE"
COMPETED = "COMPETED"
FAILED = "FAILED"

# Dynamic families run through the dispatch layer's own runners, and only when
# the modality sieve admits them for the data. L2 engines always run.
DYNAMIC_IDS: tuple[str, ...] = ("rime", "ode", "pde", "hawkes")

# Array proposers fit on the same (X, Y) training split as the L2 engines.
PROPOSER_IDS: tuple[str, ...] = (
    "sindy",
    "basis_regression",
    "reservoir",
    "distributed_lag",
    "regime_switching",
    "ncm_structural",
    "rime_ad",
    "invariant_transfer",
    "sparse_microstructure",
    "order_book_resiliency",
    "hamiltonian",
    "solenoidal_fluid",
    "compartmental_sir",
)

ENGINE_IDS: tuple[str, ...] = (
    "l2_single_term",
    "l2_multiterm",
    *PROPOSER_IDS,
    *DYNAMIC_IDS,
)


@dataclass(frozen=True)
class EngineResult:
    """One engine's outcome in a competition. Not an acceptance decision."""

    engine: str
    status: str  # COMPETED | FAILED | NOT_APPLICABLE
    reason: str = ""
    adjudication_status: str = ""
    promoted_equation: str | None = None
    best_rmse: float | None = None
    holdout_error: float | None = None
    candidate_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "status": self.status,
            "reason": self.reason,
            "adjudication_status": self.adjudication_status,
            "promoted_equation": self.promoted_equation,
            "best_rmse": self.best_rmse,
            "holdout_error": self.holdout_error,
            "candidate_count": self.candidate_count,
            "extra": self.extra,
        }


def _summarise_adjudication(adj: Any) -> dict[str, Any]:
    promoted = getattr(adj, "promoted", None)
    promoted_expr = None
    if promoted is not None:
        meta = getattr(getattr(promoted, "candidate", None), "metadata", {}) or {}
        promoted_expr = meta.get(
            "canonical_expr", getattr(promoted.candidate, "candidate_id", None)
        )
    status = getattr(adj.status, "name", str(adj.status))
    return {
        "adjudication_status": status,
        "promoted_equation": promoted_expr,
        "best_rmse": adj.best_rmse,
    }


def _l2_single(
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    train_ratio: float,
) -> EngineResult:
    eng = L2DiscoveryEngine(
        operators=("+", "-", "*", "/", "pow", "sin", "cos"),
        max_depth=2,
        max_complexity=4,
    )
    dims = {k: Dimension() for k in X}
    res = eng.discover(
        X=dict(X),
        Y=Y,
        dimensions=dims,
        target_dimension=Dimension(),
        train_ratio=train_ratio,
        bootstrap=False,
        seed=42,
    )
    summary = _summarise_adjudication(res.adjudication)
    return EngineResult(
        engine="l2_single_term",
        status=COMPETED,
        candidate_count=len(res.candidates),
        extra={"pareto_size": len(res.pareto_candidates)},
        **summary,
    )


def _l2_multiterm(
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    train_ratio: float,
) -> EngineResult:
    eng = MultiTermL2Engine(
        operators=("+", "-", "*", "/", "pow", "sin", "cos", "sat", "inv_quad"),
        max_depth=2,
        max_terms=2,
    )
    dims = {k: Dimension() for k in X}
    res = eng.discover_multiterm(
        X=dict(X),
        Y=Y,
        dimensions=dims,
        target_dimension=Dimension(),
        train_ratio=train_ratio,
        bootstrap=True,
        seed=42,
    )
    summary = _summarise_adjudication(res.adjudication)
    return EngineResult(
        engine="l2_multiterm",
        status=COMPETED,
        candidate_count=len(res.pareto_candidates),
        **summary,
    )


def _proposer_result(engine: str, proposals: Any) -> EngineResult:
    """Summarise a proposer's candidates. Proposals are NOT adjudicated here."""
    proposals = list(proposals)
    return EngineResult(
        engine=engine, status=COMPETED, candidate_count=len(proposals),
        adjudication_status="NOT_ADJUDICATED",
        extra={"formulas": [p.formula for p in proposals][:20]},
    )


def _train_matrix(X: Mapping[str, np.ndarray], Y: np.ndarray, train_ratio: float):
    names = list(X)
    n = len(Y)
    cut = max(1, int(n * train_ratio))
    M = np.column_stack([np.asarray(X[k], dtype=float)[:cut] for k in names])
    y = np.asarray(Y, dtype=float)[:cut]
    return names, M, y


def _sindy(X, Y, train_ratio) -> EngineResult:
    from apedp_dyn.proposers.sindy_proposer import SINDyProposer

    names, M, y = _train_matrix(X, Y, train_ratio)
    props = SINDyProposer().fit_and_propose(
        M, dX=y.reshape(-1, 1), variable_names=names)
    return _proposer_result("sindy", props)


def _basis(X, Y, train_ratio) -> EngineResult:
    from apedp_dyn.proposers.basis_regression_proposer import (
        BasisRegressionProposer,
    )

    names, M, y = _train_matrix(X, Y, train_ratio)
    props = BasisRegressionProposer().fit_and_propose(
        M, y=y, variable_names=names)
    return _proposer_result("basis_regression", props)


def _reservoir(X, Y, train_ratio) -> EngineResult:
    from apedp_dyn.proposers.reservoir_proposer import ReservoirProposer

    names, M, y = _train_matrix(X, Y, train_ratio)
    props = ReservoirProposer().fit_and_propose(
        M, dX=y.reshape(-1, 1), variable_names=names)
    return _proposer_result("reservoir", props)


def _frame_proposer(
    engine: str,
    proposer_factory: Callable[[], Any],
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    train_ratio: float,
    **call_kwargs: Any,
) -> EngineResult:
    """Run a DataFrame-input proposer on the training split of (X, Y)."""
    names, M, y = _train_matrix(X, Y, train_ratio)
    df = pd.DataFrame(M, columns=names)
    df["target"] = y
    props = proposer_factory().fit_and_propose(
        df, feature_cols=names, target_col="target", **call_kwargs)
    return _proposer_result(engine, props)


def _distributed_lag(X, Y, train_ratio):
    from apedp_dyn.proposers.distributed_lag_interaction_proposer import (
        DistributedLagInteractionProposer,
    )
    return _frame_proposer("distributed_lag",
                           DistributedLagInteractionProposer, X, Y, train_ratio)


def _regime_switching(X, Y, train_ratio):
    from apedp_dyn.proposers.regime_switching_proposer import (
        RegimeSwitchingProposer,
    )
    return _frame_proposer("regime_switching",
                           RegimeSwitchingProposer, X, Y, train_ratio)


def _ncm(X, Y, train_ratio):
    from apedp_dyn.proposers.ncm_structural_proposer import (
        NCMStructuralProposer,
    )
    return _frame_proposer("ncm_structural",
                           NCMStructuralProposer, X, Y, train_ratio)


def _rime_ad(X, Y, train_ratio):
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer
    names, M, y = _train_matrix(X, Y, train_ratio)
    df = pd.DataFrame(M, columns=names)
    df["target"] = y
    props = RimeAdProposer(target_col="target").fit_and_propose(df)
    return _proposer_result("rime_ad", props)


def _invariant_transfer(X, Y, train_ratio):
    from apedp_dyn.proposers.invariant_transfer_proposer import (
        InvariantTransferProposer,
    )
    return _frame_proposer("invariant_transfer",
                           InvariantTransferProposer, X, Y, train_ratio)


def _not_applicable_columns(engine: str, needs: tuple[str, ...]) -> EngineResult:
    return EngineResult(
        engine=engine, status=NOT_APPLICABLE,
        reason=f"needs named columns {list(needs)} that an array dataset does not have")


_RUNNERS: dict[str, Callable[..., EngineResult]] = {
    "l2_single_term": _l2_single,
    "l2_multiterm": _l2_multiterm,
    "sindy": _sindy,
    "basis_regression": _basis,
    "reservoir": _reservoir,
    "distributed_lag": _distributed_lag,
    "regime_switching": _regime_switching,
    "ncm_structural": _ncm,
    "rime_ad": _rime_ad,
    "invariant_transfer": _invariant_transfer,
}

# Proposers whose input is fixed-named event, price or state columns. They
# cannot run on a generic (X, Y) array; the frame path or a named-column
# upload must supply those columns.
_NAMED_COLUMN_PROPOSERS: dict[str, tuple[str, ...]] = {
    "sparse_microstructure": ("price", "V", "taker_buy"),
    "order_book_resiliency": ("price",),
    "hamiltonian": ("q", "p", "p_dot"),
    "solenoidal_fluid": ("x", "y", "u", "v"),
    "compartmental_sir": ("S", "I", "R", "dI"),
}


def competition_records(
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    *,
    train_ratio: float = 0.6,
    not_applicable: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Competition records for an array dataset (physics and market routes).

    Dynamic families are not run here: they need a frame, so they stay as
    ``NOT_APPLICABLE`` placeholders with their reason.
    """
    results = run_competition(X, Y, train_ratio=train_ratio,
                              not_applicable=not_applicable)
    return [r.to_dict() for r in results]


def competition_for_upload(
    csv_text: str,
    target_col: str | None = None,
    train_ratio: float = 0.6,
) -> list[dict[str, Any]]:
    """Run the competition on an uploaded CSV, returning plain dict records.

    The CSV is parsed with the same parser the upload sandbox uses, so the
    competition sees the same target and feature columns as the sandbox run.
    """
    from apedp_dyn.sandbox.custom_upload import parse_csv_data

    parsed = parse_csv_data(csv_text, target_col=target_col)
    target = parsed["target_col"]
    feats = list(parsed["feature_cols"])
    cols = parsed["numeric_cols"]
    if not feats:
        # 1D series: the L2 engines need a regressor, so use the time axis.
        if parsed["time_col"]:
            feats = [parsed["time_col"]]
        else:
            feats = []
    frame = pd.DataFrame({c: cols[c] for c in [*feats, target]})
    dyn = dynamic_results(frame, value_cols=None,
                          time_col=parsed["time_col"])
    skip = {"l2_single_term": "no regressor column: a 1D series has no feature axis",
            "l2_multiterm": "no regressor column: a 1D series has no feature axis"}
    l2 = run_competition({c: cols[c] for c in feats}, cols[target],
                         train_ratio=train_ratio,
                         not_applicable=skip if not feats else None)
    dyn_ids = {r.engine for r in dyn}
    merged = [r for r in l2 if r.engine not in dyn_ids] + dyn
    order = {e: i for i, e in enumerate(ENGINE_IDS)}
    merged.sort(key=lambda r: order[r.engine])
    return [r.to_dict() for r in merged]


def _dynamic_result(family: str, product: Any, error: str | None) -> EngineResult:
    """Summarise one dispatch-layer product. Proposals are NOT adjudicated here."""
    if error is not None:
        return EngineResult(engine=family, status=FAILED, reason=error)
    if isinstance(product, list):
        n = len(product)
        return EngineResult(engine=family, status=COMPETED, candidate_count=n,
                            adjudication_status="NOT_ADJUDICATED",
                            extra={"proposals": n})
    formula = getattr(product, "formula", None)
    return EngineResult(
        engine=family, status=COMPETED,
        candidate_count=1 if formula else 0,
        adjudication_status="NOT_ADJUDICATED",
        promoted_equation=formula,
        best_rmse=getattr(product, "residual_rmse", None),
        holdout_error=getattr(product, "holdout_r2", None),
        extra={"holdout_passed": bool(getattr(product, "holdout_passed", False))},
    )


def dynamic_results(
    frame: pd.DataFrame,
    *,
    value_cols: list[str] | None = None,
    time_col: str | None = None,
    coord_col: str | None = None,
) -> list[EngineResult]:
    """Run the sieve-admitted dynamic families on the frame via dispatch.

    Families the sieve masks are recorded as ``NOT_APPLICABLE`` with the
    sieve's reason. Families the sieve admits run through dispatch's own
    runners, which refuse to invent axes and report their own errors.
    """
    from apedp_dyn.orchestration.dispatch import dispatch_frame

    outcome = dispatch_frame(
        frame, value_cols=value_cols, time_col=time_col, coord_col=coord_col)
    masked = set(outcome.decision.decision.masked_engines)
    reason = outcome.decision.verdict.reason
    results: list[EngineResult] = []
    for family in DYNAMIC_IDS:
        if family in outcome.products or family in outcome.errors:
            results.append(_dynamic_result(
                family, outcome.products.get(family),
                outcome.errors.get(family)))
        elif family in masked:
            results.append(EngineResult(
                engine=family, status=NOT_APPLICABLE,
                reason=f"sieve masks '{family}': {reason}"))
        else:
            results.append(EngineResult(
                engine=family, status=NOT_APPLICABLE,
                reason=f"sieve did not select '{family}' for this data"))
    return results


def run_competition(
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    *,
    train_ratio: float = 0.6,
    not_applicable: Mapping[str, str] | None = None,
) -> list[EngineResult]:
    """Run every engine on the same data and return each result, unranked.

    ``not_applicable`` maps an engine id to the reason it cannot run on this
    data. Those engines are recorded as ``NOT_APPLICABLE`` and not executed.
    Per-engine exceptions are recorded as ``FAILED`` and never abort the others.
    """
    skips = dict(not_applicable or {})
    results: list[EngineResult] = []
    for engine in ENGINE_IDS:
        if engine in skips:
            results.append(EngineResult(engine=engine, status=NOT_APPLICABLE,
                                        reason=skips[engine]))
            continue
        if engine in _NAMED_COLUMN_PROPOSERS:
            results.append(_not_applicable_columns(
                engine, _NAMED_COLUMN_PROPOSERS[engine]))
            continue
        if engine in DYNAMIC_IDS:
            # Placeholder: upload path replaces it with the sieve-routed result.
            results.append(EngineResult(
                engine=engine, status=NOT_APPLICABLE,
                reason="dynamic family: routed by dynamic_results(frame)"))
            continue
        try:
            results.append(_RUNNERS[engine](X, Y, train_ratio))
        except Exception as exc:  # noqa: BLE001 - per-engine isolation by design
            results.append(EngineResult(
                engine=engine, status=FAILED,
                reason=f"{type(exc).__name__}: {exc}"))
    return results
