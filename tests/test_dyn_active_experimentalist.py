"""Level-5 Pillar 3 pins: active experiment orders and referee handoff.

An experimentalist that decided promotions would grade its own homework —
the pins here assert the layer STOPS at structured orders and packages
residuals for adjudicate(); promotion authority is never exercised here.
Deterministic: frozen divergence grid, no seeds in the order designer.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.active_disambiguation.policy import DEFAULT_DETECTABILITY_K
from apedp_dyn.epistemic.active_experimentalist import (
    DataAcquisitionOrder,
    closed_loop_order,
    design_experiment_orders,
    record_order_results,
    referee_handoff,
)


def _langmuir(p: float) -> float:
    return 4.0 * p / (1.0 + p)


def _temkin(p: float) -> float:
    return 2.0 * np.log(1.0 + 3.0 * p)


def test_orders_maximize_divergence_and_clamp_to_domain():
    survivors = {"langmuir": _langmuir, "temkin": _temkin}
    domain = (0.01, 100.0)
    orders = design_experiment_orders(survivors, domain, sigma_noise=0.05)
    assert len(orders) >= 1
    # Temkin keeps growing while the rational saturates: the top of the
    # domain carries the maximal pairwise divergence.
    primary = orders[0]
    assert isinstance(primary, DataAcquisitionOrder)
    assert primary.variable == "x"
    r_lo, r_hi = primary.target_ranges[0]
    assert domain[0] <= r_lo < r_hi <= domain[1] + 1e-9
    assert r_hi >= domain[1] - 2.0  # the divergence sits at the high end
    assert primary.pole_margin > 0.0
    assert str(DEFAULT_DETECTABILITY_K) in primary.rationale


def test_identical_hypotheses_emit_no_order_insufficient_disambiguation():
    survivors = {"a": lambda x: np.sin(x), "b": lambda x: np.sin(x), "c": lambda x: np.sin(x)}
    assert design_experiment_orders(survivors, (0.0, 1.0), sigma_noise=0.05) == ()


def test_record_order_results_rejects_out_of_band_points_fail_closed():
    orders = design_experiment_orders({"langmuir": _langmuir, "temkin": _temkin},
                                      (0.01, 100.0), sigma_noise=0.05)
    primary = orders[0]
    bands = primary.target_ranges
    in_lo = min(b[0] for b in bands)
    in_hi = max(b[1] for b in bands)
    x_new = np.array([in_hi + 5.0, (in_lo + in_hi) / 2.0, in_lo + 0.01, np.nan])
    y_new = np.array([1.0, 2.0, 3.0, 4.0])
    x_ok, y_ok, n_ok, n_rej = record_order_results(primary, x_new, y_new)
    # The primary band is wide for this survivor pair, so exactly the two
    # in-range points pass: the out-of-band value AND the NaN are rejected
    # (never clamped, never laundered into referee evidence).
    assert n_ok == 2 and n_rej == 2
    assert len(x_ok) == len(y_ok) == 2 and np.all(np.isfinite(x_ok)) and np.all(np.isfinite(y_ok))
    with pytest.raises(ValueError):
        record_order_results(primary, np.array([1.0]), np.array([1.0, 2.0]))


def test_referee_handoff_packages_residuals_but_never_decides():
    x_obs = np.linspace(0.5, 60.0, 12)
    y_obs = np.array([_langmuir(v) for v in x_obs]) + 0.01
    handoff = referee_handoff({"langmuir": _langmuir, "temkin": _temkin}, x_obs, y_obs)
    assert "observations" in handoff and "langmuir" in handoff and "temkin" in handoff
    assert np.allclose(handoff["langmuir"]["residuals"],
                       y_obs - np.array([_langmuir(v) for v in x_obs]))
    # Handoff carries no verdict field: nothing here claims the referee's role.
    for key in ("langmuir", "temkin"):
        assert "residuals" in handoff[key] and "predictions" in handoff[key]


def test_closed_loop_ingests_acquisition_and_counts_rejections():
    survivors = {"langmuir": _langmuir, "temkin": _temkin}
    rng = np.random.default_rng(5)
    calls: list[DataAcquisitionOrder] = []

    def acquisition(order: DataAcquisitionOrder):
        calls.append(order)
        r_lo, r_hi = order.target_ranges[0]
        x = r_lo + (r_hi - r_lo) * rng.random(25)
        y = np.array([_langmuir(v) for v in x]) + 0.01 * rng.standard_normal(25)
        # Inject two deliberately out-of-band points: instruments lie, the
        # layer must not launder them into referee evidence.
        x = np.concatenate([x, [1e6, -1e6]])
        y = np.concatenate([y, [0.0, 0.0]])
        return x, y

    result = closed_loop_order(survivors, (0.01, 100.0), sigma_noise=0.05,
                               acquisition=acquisition, rounds=2)
    assert len(calls) == len(result.orders) == result.rounds == 2
    assert result.accepted_points >= 2 * 20
    assert result.rejected_points == 4  # two injected liars per round
    assert set(result.referee_handoff) >= {"observations", "langmuir", "temkin"}
    assert len(result.referee_handoff["observations"]["x_eval"]) == result.accepted_points
