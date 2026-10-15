"""Closed-loop prior-guided search: the loop is shut iff a pipeline call suffices.

Acceptance structure (all preregistered, fixed seeds):
  1. Unsolicited Landauer recovery on sealed splits: the ONLY human input is
     calling `propose_unsolicited` on dev rows. The test asserts the TABLE
     selected the path (not a human), the parameters came from data, and the
     sealed holdout — never seen by the adapter — scores R2 == 1.0.
  2. No-human-primitives guard: the adapter module contains no quoted operator
     tokens usable for per-equation selection; routing is by family only.
  3. Table truth: staircase -> rule path, singular/power -> transmutation path,
     smooth line -> abstain with a reason. No per-case input anywhere.
  4. Noise rung 1: 2%-noise staircase through the same loop, sealed holdout.
  5. Noise rung 2 (semi-synthetic, labeled): known staircase injected into real
     BTC 5m returns at stated SNR; recovery within tolerance AND abstention on
     the clean stream. The clean-stream control is what makes rung 2 evidence
     rather than curve-fitting: the same code must decline the un-injected data.
  6. Single-threshold edge: <2 change-points -> dev-exact basis only, with an
     explicit no-holdout-claim rationale (the adapter says what it cannot do).

Boundaries recorded, not tuned: rung-2 holdout is asserted against the ORACLE
(truth scored on holdout = the noise ceiling), not against 1.0; the offset is
asserted modulo the fitted width (phase ambiguity is intrinsic to floor forms).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.prior_guided_search import (
    PATH_ABSTAIN,
    PATH_EXTRAPOLATABLE_RULE,
    PATH_TRANSMUTED_SINGULAR,
    propose_unsolicited,
    score_on_holdout,
    select_search_path,
)

SHAPE_KW = dict(min_segment_len=5, max_segments=32, min_pc_r2=0.05, min_rows=40)

_BANNED_SELECTION_LITERALS = re.compile(r"""['"](floor|ceil|arrh|bose|planck|lor_norm|inv_sq)['"]""")


def test_no_human_primitive_literals():
    """The adapter routes by family; it cannot name an operator to select one.

    A quoted operator token in this file would be a human smuggling a primitive
    past the table. Formula strings emit math notation (floor() inside a larger
    expression); the regex only fires on standalone quoted tokens.
    """
    src = (Path(__file__).resolve().parent.parent
           / "apedp_dyn" / "epistemic" / "prior_guided_search.py").read_text(encoding="utf-8")
    hits = _BANNED_SELECTION_LITERALS.findall(src)
    assert not hits, f"human-selectable primitive literals present: {sorted(set(hits))}"


def _landauer_splits():
    xd = np.linspace(0.1, 4.9, 100)
    return xd, np.floor(xd), np.linspace(5.0, 9.8, 100), np.floor(np.linspace(5.0, 9.8, 100))


def test_unsolicited_landauer_closed_loop():
    xd, yd, xh, yh = _landauer_splits()
    g = propose_unsolicited(xd, yd, "Vg", shape_kwargs=SHAPE_KW)
    print("\n[loop]", g.to_dict())
    assert not g.abstained, g.abstain_reason
    # The TABLE selected this path: family verdict in, rule form out.
    assert g.path == PATH_EXTRAPOLATABLE_RULE, g.plan_rationale
    assert g.family == "piecewise_constant"
    assert len(g.thresholds) == 4, g.thresholds  # steps at 1,2,3,4 inside [0.1,4.9]
    # Parameters from data, within tight tolerance of truth (1.0, 1.0, 0.0).
    assert g.parameters["quantum_step"] == pytest.approx(1.0, abs=1e-3)
    assert g.parameters["step_width"] == pytest.approx(1.0, abs=2e-2)
    # Dev-exact supporting basis proves exactness where thresholds exist...
    dev = score_on_holdout(g.supporting[0], xd, yd, "Vg")
    assert dev["r2"] == pytest.approx(1.0, abs=1e-9), dev
    # ...and the rule form generalizes to rows the adapter never saw.
    hold = score_on_holdout(g.primary, xh, yh, "Vg")
    assert hold["r2"] == pytest.approx(1.0, abs=1e-9), hold
    assert hold["rmse"] == pytest.approx(0.0, abs=1e-6), hold
    assert len(g.candidates()) == 2


def test_table_routes_singular_power_and_declines_smooth():
    rng = np.random.RandomState(9)
    # Singular -> transmutation path, no human routing.
    E = np.linspace(0.15, 1.2, 50)
    gs = propose_unsolicited(E, 10.0 * E**2 * np.exp(-1.0 / E), "E", shape_kwargs=SHAPE_KW)
    print("\n[singular]", gs.to_dict())
    assert gs.path == PATH_TRANSMUTED_SINGULAR, gs.plan_rationale
    assert not gs.abstained
    assert gs.dev_r2 > 0.999999, gs.dev_r2
    # Power law -> transmutation path.
    R = np.linspace(1.5, 8.0, 50)
    gp = propose_unsolicited(R, 4.5 / R**2, "R", shape_kwargs=SHAPE_KW)
    print("[power]", gp.to_dict())
    assert gp.path == PATH_TRANSMUTED_SINGULAR, gp.plan_rationale
    assert not gp.abstained
    # Smooth line -> abstain with a reason, never a step family.
    xl = rng.uniform(0.1, 6.0, 200)
    gl = propose_unsolicited(xl, 2.5 * xl + 10.0, "x", shape_kwargs=SHAPE_KW)
    print("[line]", gl.to_dict())
    assert gl.path == PATH_ABSTAIN and gl.abstained
    assert gl.primary is None and gl.candidates() == ()


def test_noise_rung1_2pct_through_closed_loop():
    rng = np.random.RandomState(42)
    xd = np.linspace(0.1, 4.9, 100)
    yd = np.floor(xd) + 0.02 * np.floor(xd).std() * rng.randn(100)
    xh = np.linspace(5.0, 9.8, 100)
    yh = np.floor(xh)
    g = propose_unsolicited(xd, yd, "Vg", shape_kwargs=SHAPE_KW)
    print("\n[rung1]", g.to_dict())
    assert g.path == PATH_EXTRAPOLATABLE_RULE, g.plan_rationale
    assert not g.abstained
    hold = score_on_holdout(g.primary, xh, yh, "Vg")
    assert hold["r2"] >= 0.999, hold
    assert abs(g.parameters["quantum_step"] - 1.0) < 0.05, g.parameters


def _btc_5m_returns(n=1000):
    path = Path(__file__).resolve().parent.parent / "data" / "market" / "intraday" / "intraday_crypto_bars.csv"
    if not path.exists():
        pytest.skip(f"intraday market data absent: {path}")
    df = pd.read_csv(path)
    b = df[(df.symbol == "BTC") & (df.interval == "5m")].sort_values("open_time_ms").reset_index(drop=True)
    px = b["close"].to_numpy(float)
    r = np.diff(np.log(px))
    t = np.arange(r.size, dtype=float)
    return t[:n], r[:n]


def test_rung2_semisynthetic_btc_injection_with_abstention_control():
    """Labeled semi-synthetic rung 2: stated SNR, oracle-scored, controlled.

    Injects a KNOWN staircase (W=125 bars, OFF=40, height SNR=5 x dev-window
    return noise) into real BTC 5m returns. Asserts recovery against the truth
    AND abstention on the clean stream by the same code. Holdout is asserted
    against the ORACLE (truth scored on holdout = the noise ceiling), because
    1.0 is unreachable under observation noise and asserting it would be
    dishonest. Measured boundary, pinned here: at local SNR <= 4 the loop
    honestly abstains (family=smooth); SNR=5 is the working point.
    """
    t, r = _btc_5m_returns()
    sd = float(np.std(r[:500]))
    h_true, W_true, OFF_true, SNR = 5.0 * sd, 125.0, 40.0, 5.0
    print(f"\n[rung2] N={t.size} return_std={sd:.6f} injected h={h_true:.6f} (SNR={SNR})")
    phase = np.floor((t - OFF_true) / W_true)
    inj = r + h_true * (phase - phase[:500].mean())
    n_dev = 500

    g = propose_unsolicited(t[:n_dev], inj[:n_dev], "t", shape_kwargs=SHAPE_KW)
    print("[rung2-injected]", g.to_dict())
    assert g.path == PATH_EXTRAPOLATABLE_RULE, g.plan_rationale
    assert not g.abstained
    p = g.parameters
    assert abs(p["quantum_step"] - h_true) / h_true < 0.10, p
    assert abs(p["step_width"] - W_true) / W_true < 0.05, p
    # Offset is intrinsic phase: asserted modulo the fitted width.
    assert min(abs((p["offset"] + k * p["step_width"]) - OFF_true)
               for k in range(-2, 3)) < 0.05 * p["step_width"], p
    hold = score_on_holdout(g.primary, t[n_dev:], inj[n_dev:], "t")
    yh = inj[n_dev:]
    oracle = 1.0 - float(np.sum((yh - h_true * (np.floor((t[n_dev:] - OFF_true) / W_true)
                                           - phase[:500].mean())) ** 2)) / float(np.sum((yh - yh.mean()) ** 2))
    print(f"[rung2] holdout R2={hold['r2']:.4f} vs oracle={oracle:.4f}")
    assert hold["r2"] >= oracle - 0.01, (hold, oracle)

    # The control that makes this evidence: the same code declines clean data.
    gc = propose_unsolicited(t[:n_dev], r[:n_dev], "t", shape_kwargs=SHAPE_KW)
    print("[rung2-clean]", gc.to_dict())
    assert gc.abstained and gc.primary is None, gc.to_dict()


def test_rung2_detection_boundary_low_snr_abstains():
    """The measured limit, pinned rather than hidden: at local SNR=3 the injected
    steps are buried in heavy-tailed return noise and the loop declines
    (family=smooth). A detector that claimed steps here would be fabricating."""
    t, r = _btc_5m_returns()
    sd = float(np.std(r[:500]))
    h, W, OFF = 3.0 * sd, 125.0, 40.0
    phase = np.floor((t - OFF) / W)
    inj = r + h * (phase - phase[:500].mean())
    g = propose_unsolicited(t[:500], inj[:500], "t", shape_kwargs=SHAPE_KW)
    print("\n[boundary-snr3]", g.to_dict())
    assert g.abstained, g.to_dict()


def test_single_threshold_yields_basis_only_no_holdout_claim():
    """<2 change-points: the adapter states its limit instead of extrapolating."""
    rng = np.random.RandomState(5)
    x = np.sort(rng.uniform(0.0, 6.0, 120))
    y = (x >= 3.0).astype(float)
    g = propose_unsolicited(x, y, "x", shape_kwargs=SHAPE_KW)
    print("\n[edge]", g.to_dict())
    assert not g.abstained
    assert "no holdout claim" in g.plan_rationale, g.plan_rationale
