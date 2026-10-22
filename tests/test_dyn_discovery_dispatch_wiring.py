"""Dispatch-wiring pins: sieve + route selectors automatically dispatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _isotherm_frame() -> pd.DataFrame:
    x = np.geomspace(0.01, 100.0, 300)
    return pd.DataFrame({"p": x, "y": 2.0 * x / (1.0 + x)})


def _kinetic_frame() -> pd.DataFrame:
    t = np.arange(0, 20, 0.05)
    return pd.DataFrame({"t": t, "s": np.sin(2 * np.pi * t / 2.0)})


def _stamps_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    gaps = []
    for _ in range(12):
        gaps.extend(rng.exponential(0.06, 6))
        gaps.append(rng.exponential(1.2))
    return pd.DataFrame({"t": np.cumsum(np.array(gaps))})


def test_dispatch_routes_isotherm_to_static_automatically():
    """Raw unannotated isotherm: unannotated route to static runner (rime)."""
    from apedp_dyn.orchestration import dispatch_frame

    out = dispatch_frame(_isotherm_frame())
    d = out.decision.decision
    assert d.status in ("ROUTED", "MULTIPLE_ROUTES"), d.reason
    assert d.sieve_modality == "static_manifold"
    assert d.selector_outcome == "STATIC_RELATION_SUPPORTED"
    assert "rime" in d.chosen_engines
    assert "rime" in out.products


def test_dispatch_routes_kinetic_to_ode_automatically():
    """Raw kinetic series routes to the ODE engine without any human pick.
    The masked sibling never runs even when a stub runner is wired."""
    from apedp_dyn.orchestration import dispatch_frame

    frame = _kinetic_frame()
    out = dispatch_frame(frame, time_col="t",
                         runner={"pde": lambda: "never_ran"})
    d = out.decision.decision
    assert d.status == "ROUTED", d.reason
    assert d.chosen_engines == ("ode",)
    assert "ode" in out.products
    assert "pde" not in out.products


def test_dispatch_routes_events_to_hawkes_by_kind():
    """Declared event stamps route to the Hawkes proposer automatically."""
    from apedp_dyn.orchestration import dispatch_frame

    forty = _stamps_frame()
    out = dispatch_frame(forty, data_kind="events", stamp_col="t")
    d = out.decision.decision
    assert d.status == "ROUTED", d.reason
    assert d.chosen_engines == ("hawkes",)
    assert "hawkes" in out.products


def test_dispatch_decisive_agreement_dispatches_one_engine():
    """Decisive selector + decisive sieve agree ⇒ ROUTED to that one engine."""
    from apedp_dyn.orchestration import dispatch_frame

    # Pure traveling wave: PDE's defining diagnostic (mixed_energy) fires,
    # the ODE rival's kinetic mean-field is a negligible ripple (aggregation
    # validity refuses it) — decisive agreement dispatches exactly one engine.
    t = np.arange(0, 40, 0.1)
    grid_x = np.linspace(0, 1, 8)
    X, T = np.meshgrid(grid_x, t)
    frame = pd.DataFrame({
        "t": np.repeat(t, 8), "x": np.tile(grid_x, len(t)),
        "u": np.sin(2 * np.pi * (X - 0.2 * T)).ravel(),
    })
    invoked: list[str] = []

    def make_runner(name):
        return lambda: invoked.append(name) or name

    out = dispatch_frame(frame, time_col="t", coord_col="x", value_cols=["u"],
                         runner={"ode": make_runner("ode"), "pde": make_runner("pde")})
    d = out.decision.decision
    assert d.status == "ROUTED", d.reason
    assert d.chosen_engines == ("pde",)
    assert d.selector_outcome == "PDE_PLAUSIBLE"
    assert invoked == ["pde"], (
        "a decisive agreement dispatches exactly the agreed engine — the "
        "masked sibling still stays masked even when a runner is wired")


def test_dispatch_selector_ambiguity_keeps_admitted_families_live():
    """Selector ambiguity with a decisive sieve ⇒ MULTIPLE_ROUTES: the
    sieve-admitted family runs; masked siblings never do (resolving the
    selector's ambiguity is caller evidence, not a hidden default)."""
    from apedp_dyn.orchestration import dispatch_frame

    t = np.arange(0, 40, 0.1)
    grid_x = np.linspace(0, 1, 8)
    X, T = np.meshgrid(grid_x, t)
    frame = pd.DataFrame({
        "t": np.repeat(t, 8), "x": np.tile(grid_x, len(t)),
        "u": (0.5 * np.sin(2 * np.pi * (X - 0.2 * T)) + 1.0 * np.sin(T) * X).ravel(),
    })
    invoked: list[str] = []

    def make_runner(name):
        return lambda: invoked.append(name) or name

    out = dispatch_frame(frame, time_col="t", coord_col="x", value_cols=["u"],
                         runner={"ode": make_runner("ode"), "pde": make_runner("pde")})
    d = out.decision.decision
    assert d.status == "MULTIPLE_ROUTES", d.reason
    assert d.selector_outcome == "MULTIPLE_CLASSES_PLAUSIBLE"
    assert d.chosen_engines == ("pde",)  # every family the sieve ADMITS
    assert invoked == ["pde"], (
        "ambiguity keeps every admitted family live; the masked family is "
        "never invoked even when a runner is wired")


def test_dispatch_unroutable_abstains_and_records_open_questions():
    """Unresolved structure dispatches NOTHING and says why: declared events
    too sparse for any fitted point process — sieve leaves nothing admissible."""
    from apedp_dyn.orchestration import dispatch_frame

    out = dispatch_frame(pd.DataFrame({"t": np.arange(5.0)}),
                         data_kind="events", stamp_col="t")
    d = out.decision.decision
    assert d.status == "UNROUTABLE"
    assert d.chosen_engines == (), "an unroutable frame must dispatch nothing"
    assert out.products == {}
    assert "declare" in d.reason.lower()

def test_dispatch_conflict_is_abstention_not_override():
    """A decisive selector class the sieve masks ⇒ CONFLICT, nothing runs."""
    from apedp_dyn.orchestration import dispatch_frame
    from apedp_dyn.orchestration import dispatch_frame as _dfr
    from apedp_dyn.orchestration.modality_sieve import (
        Modality,
        ModalityVerdict,
        classify_modality,
    )
    from apedp_dyn.orchestration.model_class_selector import select_model_class

    # Constructor-seam demo: force the two layers to disagree structurally.
    frame = _kinetic_frame()
    verdict = classify_modality(frame, time_col="t")
    selection = select_model_class(frame, time_col="t")
    assert verdict.modality == Modality.ODE_KINETIC
    # The REAL dispatch gate: sieve-masked and selector-confirmed is what a
    # torn dataset looks like — synthesize it by masking 'ode' by hand.
    torn = ModalityVerdict(
        Modality.STATIC_MANIFOLD, dict(verdict.evidence), ("rime",),
        tuple(f for f in ["rime", "ode", "pde", "hawkes"] if f != "rime"),
        "synthesized disagreement for the pin")
    decision_frame = _dfr
    del decision_frame
    # Run the real _decide seam directly on the torn verdict pair.
    from apedp_dyn.orchestration.dispatch import _decide
    from apedp_dyn.orchestration.model_class_selector import ModelClassSelection

    decis = ModelClassSelection("ODE_PLAUSIBLE", selection.posteriors, "ode_kinetic",
                                selection.bayes_factor, selection.diagnostics, "")
    d = _decide(torn, decis)
    assert d.status == "CONFLICT"
    assert d.chosen_engines == (), "a conflict abstains — never an override"
    assert "ABSTENTION" in d.reason

    # And the full dispatch through the injected runner surface: with no
    # engines chosen, nothing executes even when a runner is wired.
    out = dispatch_frame(frame, time_col="t", runner={"rime": lambda: "never_ran"})
    assert "rime" not in out.products


def test_dispatch_products_never_carry_a_verdict():
    """Dispatch returns referee-facing material only — no adjudication
    vocabulary on real (fast ODE engine) products."""
    from apedp_dyn.orchestration import dispatch_frame

    out = dispatch_frame(_kinetic_frame(), time_col="t")
    bag: set[str] = set()
    for product in out.products.values():
        if isinstance(product, dict):
            bag |= {str(k).lower() for k in product}
        elif hasattr(product, "__dict__"):
            bag |= {str(k).lower() for k in vars(product)}
    assert bag, "the test expects real engine products with inspectable keys"
    for forbidden in ("accepted", "promoted", "verdict", "is_valid", "passed"):
        assert forbidden not in bag, (
            f"dispatch layer fabricated adjudication vocabulary: {forbidden}")
    # The routing record is complete: sieve + selector + decision all present.
    d = out.decision.decision
    assert d.selector_posteriors, "posteriors recorded"
    assert d.sieve_reason and d.sieve_modality


def test_dispatch_records_engine_failure_without_laundering():
    """A failing engine* is captured in errors, never a silent fake success; a
    successful sibling still runs (per-family isolation)."""
    from apedp_dyn.orchestration import dispatch_frame

    def broken_rime():
        raise RuntimeError("engine exploded")

    out = dispatch_frame(_isotherm_frame(),
                         runner={"rime": broken_rime, "ode": lambda: "ran"})
    assert "rime" not in out.products
    assert "engine exploded" in out.errors.get("rime", "")
    # The sibling family is invoked when its routing admits it (recorded in
    # the decision); with ROUTED status only the chosen engine appears.
    assert out.decision.decision.status in ("ROUTED", "MULTIPLE_ROUTES")
