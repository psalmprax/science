"""Discovery dispatch: sieve verdict + probabilistic selector route proposals.

Wires the Level-5 modality sieve (`orchestration/modality_sieve`) and the
P(M|D) model-class selector (`orchestration/model_class_selector`) into the
discovery dispatch surface so a raw, unannotated dataset is converted into a
routed problem specification and the appropriate proposer/discovery engines
run AUTOMATICALLY — no human choosing which family to invoke.

Routing discipline (this layer never decides acceptance):
  1. The sieve's structural verdict ADMITS/MASKS engine families
     (ill-posedness masking: a static isotherm never reaches ODE/PDE
     derivative machinery; a static-manifold verdict keeps only `rime`).
  2. The probabilistic selector CONFIRMS the class from the data; a decisive
     selector outcome for a family the sieve MASKED is a CONFLICT, not an
     override — the run abstains with both verdicts recorded.
  3. Selector ambiguity keeps EVERY sieve-admitted family live
     (`MULTIPLE_ROUTES`), so ambiguity resolves only on caller declarations
     or new evidence — never on a hidden default.
  4. UNRESOLVED structural evidence dispatches nothing: the caller is told
     to declare axes/kind and the routing record carries the open questions.
Direction does not flow the other way: this layer returns proposals /
discovery results / residual HANDOFF objects ONLY. It never fabricates an
accepted verdict — `adjudicate()` and the referee gates remain the sole
acceptance authority.

Status: experimental (dispatch wiring of Pillar 1 into the pipeline surface).
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.orchestration.modality_sieve import (
    Modality,
    ModalityVerdict,
    classify_modality,
    route_engines,
)
from apedp_dyn.orchestration.model_class_selector import (
    ModelClassSelection,
    select_model_class,
)

__all__ = [
    "DEFAULT_RUNNER_FAMILIES",
    "DispatchDecision",
    "DispatchOutcome",
    "RouteDecision",
    "dispatch_frame",
]

# Selector outcome -> dispatch family name (the sieve's family vocabulary).
_SELECTOR_FAMILY = {
    "STATIC_RELATION_SUPPORTED": "rime",
    "ODE_PLAUSIBLE": "ode",
    "PDE_PLAUSIBLE": "pde",
    "HAWKES_PLAUSIBLE": "hawkes",
    "POISSON_RENEWAL_PLAUSIBLE": "hawkes",
}
_SIEVE_FAMILY = {
    Modality.STATIC_MANIFOLD: "rime",
    Modality.ODE_KINETIC: "ode",
    Modality.PDE_FIELD: "pde",
    Modality.HAWKES_POINT_PROCESS: "hawkes",
}

_DEFAULT_RUNNER_FAMILIES = ("rime", "ode", "pde", "hawkes")
DEFAULT_RUNNER_FAMILIES = _DEFAULT_RUNNER_FAMILIES

#: The working frame for the (lazy, context-scoped) runner closures. Set
#: inside dispatch_frame for the duration of the dispatch; never a global
#: conquering of user state — the runner seam stays injectable via ``runner``.
_FRAME_CTX: contextvars.ContextVar[pd.DataFrame | None] = contextvars.ContextVar(
    "apedp_dispatch_frame", default=None)


@dataclass(frozen=True)
class RouteDecision:
    """The routing evidence behind one dispatch (complete audit record)."""

    sieve_modality: str
    sieve_reason: str
    selector_outcome: str
    selector_posteriors: dict[str, float]
    selector_bayes_factor: float
    chosen_engines: tuple[str, ...]
    masked_engines: tuple[str, ...]
    status: str  # ROUTED | MULTIPLE_ROUTES | CONFLICT | UNROUTABLE
    reason: str


@dataclass(frozen=True)
class DispatchDecision:
    decision: RouteDecision
    verdict: ModalityVerdict
    selection: ModelClassSelection


@dataclass
class DispatchOutcome:
    """Routed family products + the full routing record, caller-facing.

    ``products`` maps engine family -> whatever the proposer/discovery layer
    returned (proposals, discovery results, residual handoffs): INPUT to the
    referee, never a verdict. ``errors`` maps family -> failure text; a
    failed engine is RECORDED, never laundered into a fake success.
    """

    decision: DispatchDecision
    products: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def dispatch_frame(
    frame: pd.DataFrame,
    *,
    value_cols: list[str] | None = None,
    time_col: str | None = None,
    coord_col: str | None = None,
    data_kind: str = "frames",
    stamp_col: str | None = None,
    runner: dict[str, Callable[[], Any]] | None = None,
) -> DispatchOutcome:
    """Route one raw frame through sieve + selector, then run the engines.

    ``runner`` optionally maps family -> callable() replacing the default
    runner for that family (the injection seam used by the pin tests); the
    defaults dispatch to the repository's real engines. Per-family exceptions
    are captured in ``errors`` and never abort the other families.
    """
    verdict = classify_modality(
        frame, value_cols=value_cols, time_col=time_col, coord_col=coord_col,
        data_kind=data_kind)
    selection = select_model_class(
        frame, value_cols=value_cols, time_col=time_col, coord_col=coord_col,
        data_kind=data_kind)
    decision = _decide(verdict, selection)
    products: dict[str, Any] = {}
    errors: dict[str, str] = {}
    table = _default_runners(
        value_cols=value_cols, time_col=time_col, coord_col=coord_col,
        stamp_col=stamp_col)
    if runner:
        table.update(runner)
    frame_token = _FRAME_CTX.set(frame)
    try:
        for family in decision.chosen_engines:
            fn = table.get(family)
            if fn is None:
                errors[family] = "no runner registered for the routed family"
                continue
            try:
                products[family] = fn()
            except Exception as exc:  # noqa: BLE001 - per-family isolation by design
                errors[family] = f"{type(exc).__name__}: {exc}"
    finally:
        _FRAME_CTX.reset(frame_token)
    return DispatchOutcome(decision=DispatchDecision(decision, verdict, selection),
                           products=products, errors=errors)


def _decide(verdict: ModalityVerdict, selection: ModelClassSelection) -> RouteDecision:
    """Sieve-admitted families ∩ selector-confirmed class ⇒ the auto route."""
    sieve_family = _SIEVE_FAMILY.get(verdict.modality)
    if sieve_family is None:
        return RouteDecision(
            sieve_modality=verdict.modality.value, sieve_reason=verdict.reason,
            selector_outcome=selection.outcome,
            selector_posteriors=dict(selection.posteriors),
            selector_bayes_factor=selection.bayes_factor,
            chosen_engines=(), masked_engines=tuple(verdict.masked_engine_families),
            status="UNROUTABLE",
            reason=("sieve could not resolve a structural modality — declare "
                    "axes/kind (nothing was dispatched)"),
        )
    selector_family = _SELECTOR_FAMILY.get(selection.outcome)
    if selector_family is None:
        return RouteDecision(
            sieve_modality=verdict.modality.value, sieve_reason=verdict.reason,
            selector_outcome=selection.outcome,
            selector_posteriors=dict(selection.posteriors),
            selector_bayes_factor=selection.bayes_factor,
            chosen_engines=tuple(route_engines(verdict)),
            masked_engines=tuple(verdict.masked_engine_families),
            status=("MULTIPLE_ROUTES" if verdict.admissible_engine_families
                    else "UNROUTABLE"),
            reason=(f"selector kept ambiguity explicit ({selection.outcome}, "
                    f"bf {selection.bayes_factor:.2f}): every sieve-admitted family "
                    "stays live — resolution is caller declarations or new evidence, "
                    "never a hidden default"),
        )
    if selector_family not in verdict.admissible_engine_families:
        return RouteDecision(
            sieve_modality=verdict.modality.value, sieve_reason=verdict.reason,
            selector_outcome=selection.outcome,
            selector_posteriors=dict(selection.posteriors),
            selector_bayes_factor=selection.bayes_factor,
            chosen_engines=(), masked_engines=tuple(verdict.masked_engine_families),
            status="CONFLICT",
            reason=(f"selector decisively proposes '{selector_family}' but the sieve "
                    f"masks it ({verdict.reason}): structural disagreement is an "
                    "ABSTENTION, never an override"),
        )
    return RouteDecision(
        sieve_modality=verdict.modality.value, sieve_reason=verdict.reason,
        selector_outcome=selection.outcome,
        selector_posteriors=dict(selection.posteriors),
        selector_bayes_factor=selection.bayes_factor,
        chosen_engines=(selector_family,),
        masked_engines=tuple(verdict.masked_engine_families),
        status="ROUTED",
        reason=(f"sieve ({verdict.modality.value}) and selector "
                f"({selection.outcome}, bf {selection.bayes_factor:.2f}) agree on "
                f"'{selector_family}' — dispatched to the routed engine"),
    )


# ---------------------------------------------------------------------------
# Default runner table: the repository's real engines, one per family. Each
# runner returns proposer / discovery results WITHOUT any acceptance decision
# (referee-facing material only) and reads its working frame from _FRAME_CTX.
# ---------------------------------------------------------------------------


def _default_runners(
    *,
    value_cols: list[str] | None,
    time_col: str | None,
    coord_col: str | None,
    stamp_col: str | None,
) -> dict[str, Callable[[], Any]]:
    return {
        "rime": _run_static(value_cols, coord_col),
        "ode": _run_ode(value_cols, time_col),
        "pde": _run_pde(value_cols, time_col, coord_col),
        "hawkes": _run_hawkes(stamp_col, time_col),
    }


def _frame_or_raise() -> pd.DataFrame:
    frame = _FRAME_CTX.get()
    if frame is None:
        raise RuntimeError(
            "no frame context: use dispatch_frame(frame=...) — raw runner "
            "closures have no ambient frame to work on")
    return frame


def _declared_numeric(
    frame: pd.DataFrame,
    *,
    value_cols: list[str] | None,
    exclude: tuple[str, ...],
) -> list[str]:
    """Declared (or auto-detected) numeric value columns, declared order."""
    detected = {
        c: np.isfinite(pd.to_numeric(frame[c], errors="coerce")).mean() > 0.9
        for c in frame.columns if c not in exclude
    }
    cols = [c for c, ok in detected.items() if ok]
    if value_cols:
        missing = [c for c in value_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"declared value_cols not in frame: {missing}")
        low_quality = [c for c in value_cols if not detected.get(c, False)]
        if low_quality:
            raise ValueError(
                "declared value_cols are >10% non-finite, refusing to fit: "
                f"{low_quality}")
        order = {c: i for i, c in enumerate(frame.columns)}
        return sorted(value_cols, key=lambda c: order[c])
    return cols


def _finite_rows(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    # pd.to_numeric is single-column only; a DataFrame would raise
    # TypeError("arg must be a list, tuple, 1-d array, or Series").
    values = np.column_stack([
        pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        for c in cols
    ])
    return np.isfinite(values).all(axis=1) if values.size else np.zeros(len(frame), bool)


def _run_static(value_cols, coord_col):
    def run():
        from apedp_dyn.discovery_l2.dimensional_types import Dimension
        from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine

        frame = _frame_or_raise()
        numeric = _declared_numeric(frame, value_cols=value_cols,
                                    exclude=(coord_col,) if coord_col else ())
        if len(numeric) < 2:
            raise ValueError(
                "static open discovery needs ≥ 2 numeric columns "
                "(features + response)")
        response = numeric[-1]   # isotherm convention: LAST value column = Y
        features = numeric[:-1]
        use = _finite_rows(frame, [*features, response])
        if int(use.sum()) < 8:
            raise ValueError(
                f"only {int(use.sum())} complete (features, response) rows; "
                "static open discovery refuses to fit on fewer")
        X = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)[use]
             for c in features}
        Y = pd.to_numeric(frame[response], errors="coerce").to_numpy(dtype=float)[use]
        # Honest default: UNTYPED dimensions for every variable — all-zero
        # dimensions are homogeneous so the dimensional-AST gate stays
        # satisfied; a caller who KNOWS the units should declare typed facts
        # through their own L2 call rather than dispatch guesswork.
        dims = {c: Dimension() for c in features}
        return L2DiscoveryEngine().discover(
            X=X, Y=Y, dimensions=dims, target_dimension=Dimension(),
            target_name=response,
        )
    return run


def _run_ode(value_cols, time_col):
    def run():
        from apedp_dyn.de_arm.dependency_screen import discover_ode_system
        from apedp_dyn.ode.dispatch import discover_ode

        frame = _frame_or_raise()
        if time_col is None:
            raise ValueError(
                "ode route needs a declared time axis; isotherm-like data "
                "should be reclassified — the dispatcher never invents a "
                "derivative axis")
        numeric = _declared_numeric(frame, value_cols=value_cols, exclude=(time_col,))
        if not numeric:
            raise ValueError("ode route found no numeric value column")
        use = _finite_rows(frame, [time_col, *numeric])
        t = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)[use]
        if len(t) < 12:
            raise ValueError(
                f"only {len(t)} finite kinetic rows; refuse ODE discovery")
        diffs = np.diff(t)
        dt = float(np.median(diffs))
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(
                f"non-positive median time step ({dt}): the axis is not a real "
                "time axis — the dispatcher refuses to fit an ODE on it")
        if len(numeric) == 1:
            series = pd.to_numeric(frame[numeric[0]], errors="coerce").to_numpy(
                dtype=float)[use]
            return discover_ode(t.tolist(), series.tolist())
        table = {
            c: pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)[use]
            for c in numeric
        }
        # the COUPLED route owns the DE arm: dense kernel result + Level-1
        # dependency screening + screened mirror fits as ONE referee-facing
        # evidence artifact (no acceptance decision inside the arm)
        return discover_ode_system(table, dt=dt)
    return run


def _run_pde(value_cols, time_col, coord_col):
    def run():
        from apedp_dyn.epistemic.pde_discovery import discover_pde

        frame = _frame_or_raise()
        if time_col is None or coord_col is None:
            raise ValueError(
                "pde route needs declared (time, coord) axes; reclassify or "
                "declare the grid")
        numeric = _declared_numeric(frame, value_cols=value_cols,
                                    exclude=(time_col, coord_col))
        if not numeric:
            raise ValueError("pde route found no numeric field column")
        field_name = numeric[0]
        use = _finite_rows(frame, [time_col, coord_col])
        t = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)[use]
        xs = pd.to_numeric(frame[coord_col], errors="coerce").to_numpy(dtype=float)[use]
        u = pd.to_numeric(frame[field_name], errors="coerce").to_numpy(dtype=float)[use]
        ts, ts_inv = np.unique(np.round(t, 12), return_inverse=True)
        xs_u, xs_inv = np.unique(np.round(xs, 12), return_inverse=True)
        if ts.size * xs_u.size != len(u):
            raise ValueError(
                "rows are not a complete (time, coord) grid; the sieve's grid "
                "verdict does not certify a broken mesh — reclassify or fix the data")
        grid = np.full((int(ts.size), int(xs_u.size)), np.nan)
        grid[ts_inv, xs_inv] = u
        if not np.isfinite(grid).all():
            raise ValueError("grid has holes; refuse PDE discovery on broken fields")
        dt = float(np.median(np.diff(ts))) if ts.size > 2 else 1.0
        dx = float(np.median(np.diff(xs_u))) if xs_u.size > 2 else 1.0
        return discover_pde(grid, dt, dx)
    return run


def _run_hawkes(stamp_col, time_col):
    def run():
        from apedp_dyn.proposers.hawkes_orderflow_proposer import (
            HawkesOrderFlowProposer,
        )

        frame = _frame_or_raise()
        ann_col = stamp_col if stamp_col is not None else time_col
        if ann_col is None or ann_col not in frame.columns:
            raise ValueError(
                "event route needs a declared stamps column (stamp_col or time_col)")
        stamps = pd.to_numeric(frame[ann_col], errors="coerce").dropna().to_numpy(
            dtype=float)
        stamps = np.sort(stamps)
        if len(stamps) < 50:
            raise ValueError(
                f"only {len(stamps)} stamps: the Hawkes fit needs ≥ 50 events to "
                "identify (mu, alpha, beta); fewer would fabricate burstiness")
        return HawkesOrderFlowProposer().fit_and_propose(event_timestamps=stamps)
    return run
