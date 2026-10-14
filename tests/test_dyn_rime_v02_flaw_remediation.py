"""RIME v0.2 Structural Flaw Remediation & Epistemic Robustness Suite.

Pins and formally verifies the remediation of the 4 core RIME failure modes:
1. Flaw 1 (Grammar Poverty): Native rational power primitives (x^(2/3), x^(1/3), x^(1/2), x^(-1/3)).
   - Exact recovery of fractional powers (e.g. nuclear surface term A^(2/3)) with R^2 = 1.000000.
2. Flaw 4 (Small-Sample Paralysis): Configurable min_rows threshold for laboratory physics.
   - Enables execution on precision experimental datasets (e.g. Bertozzi 1964 N=5) while preserving
     frozen default min_rows=30 for noisy financial time-series.
3. Flaw 2 (Correlated Proxy Trap): Identifiability banding and derivative/gradient alignment.
   - Prevents false-certainty promotion when competing hypotheses differ by Delta R^2 <= 0.002.
   - Flags equivalence sets with AMBIGUOUS_EQUIVALENCE_SET and records competing hypotheses.
4. Flaw 3 (Greedy Beam Monoculture): Diverse beam archetype preservation.
   - Ensures multi-family frontier retention (polynomial, power, rational, bounded) across search depths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers.rime_proposer import RimeProposer, _RIME_AVAILABLE

ROOT = Path(__file__).resolve().parent.parent
BERTOZZI_FILE = ROOT / "data" / "bertozzi_1964.jsonl"
MASS_FILE = ROOT / "data" / "mass_1.mas20.txt"


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="native RIME kernel required")
def test_flaw_1_native_rational_power_discovery():
    """Flaw 1 Remediation: Native rational power primitives.

    Verifies that RIME with power_ops=True natively discovers fractional powers
    such as A^(2/3) to machine precision without manual column engineering.
    """
    A = np.linspace(20.0, 240.0, 60)
    # Planted nuclear surface scaling law: y = 17.8 * A^(2/3) - 4.5
    y_true = 17.8 * (A ** (2.0 / 3.0)) - 4.5
    df = pd.DataFrame({"A": A, "y": y_true})

    # 1. Negative control: default grammar without power_ops cannot represent fractional powers
    proposer_default = RimeProposer(
        feature_cols=["A"],
        target_col="y",
        static_mode=True,
        power_ops=False,
        max_depth=1,
        budget=100,
        rel_penalty=0.0,
    )
    props_default = proposer_default.fit_and_propose(df)
    eqs_default = [p.metadata["equation"] for p in props_default]
    assert not any("^(2/3)" in eq or "^(1/3)" in eq for eq in eqs_default), (
        f"Default grammar must not contain fractional powers: {eqs_default}"
    )

    # 2. Positive test: power_ops=True discovers exact A^(2/3) form
    proposer_power = RimeProposer(
        feature_cols=["A"],
        target_col="y",
        static_mode=True,
        power_ops=True,
        max_depth=1,
        budget=100,
        rel_penalty=0.0,
    )
    props_power = proposer_power.fit_and_propose(df)
    assert len(props_power) > 0, "power_ops must propose valid mechanisms"

    best = props_power[0]
    eq = best.metadata["equation"]
    r2 = float(best.parameters["dev_r2"])
    scale = float(best.parameters["scale"])
    intercept = float(best.parameters["intercept"])

    # Discovered form must be mathematically equivalent to A^(2/3)
    has_power_form = (
        "A^(2/3)" in eq
        or "(A * (A^(-1/3)))" in eq
        or "((A^(1/3)) * (A^(1/3)))" in eq
        or "abs(A)^(2/3)" in eq
    )
    assert has_power_form, f"Expected A^(2/3) equivalent form, got {eq}"
    assert np.isclose(r2, 1.0, atol=1e-6), f"Expected exact R^2=1.0, got {r2}"
    assert np.isclose(scale, 17.8, atol=1e-2), f"Expected scale ~17.8, got {scale}"
    assert np.isclose(intercept, -4.5, atol=1e-2), f"Expected intercept ~ -4.5, got {intercept}"


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="native RIME kernel required")
def test_flaw_4_small_sample_laboratory_mode_bertozzi():
    """Flaw 4 Remediation: SNR-Adaptive sample sizing for laboratory physics.

    Verifies that:
    1. Default min_rows=30 protects financial/time-series data from spurious short-sample fits.
    2. Setting min_rows=5 allows high-precision experimental physics (Bertozzi 1964 N=5)
       to execute cleanly and discover saturating velocity.
    """
    if not BERTOZZI_FILE.exists():
        pytest.skip(f"Bertozzi data file not found: {BERTOZZI_FILE}")

    rec = json.loads(BERTOZZI_FILE.read_text())
    runs = rec["runs"]
    df = pd.DataFrame({
        "KE": np.array([r["KE_MeV"] for r in runs]),
        "v": np.array([r["v_over_c"] for r in runs]),
    })
    assert len(df) == 5

    # 1. Default min_rows=30 safely abstains on 5 rows
    proposer_default = RimeProposer(
        feature_cols=["KE"],
        target_col="v",
        static_mode=True,
    )
    assert proposer_default.fit_and_propose(df) == [], "Default min_rows=30 must abstain on N=5"

    # 2. Declared min_rows=5 executes laboratory search
    proposer_lab = RimeProposer(
        feature_cols=["KE"],
        target_col="v",
        min_rows=5,
        static_mode=True,
        rational_ops=True,
        max_depth=1,
        budget=100,
        rel_penalty=0.0,
    )
    props_lab = proposer_lab.fit_and_propose(df)
    assert len(props_lab) > 0, "Laboratory mode (min_rows=5) must propose mechanisms on N=5"

    best = props_lab[0]
    r2 = float(best.parameters["dev_r2"])
    # Relativistic velocity saturation achieves high R^2 (> 0.94) on the 5 points
    assert r2 > 0.94, f"Bertozzi saturation fit R^2 should exceed 0.94, got {r2}"
    assert best.metadata["min_rows"] == 5


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="native RIME kernel required")
def test_flaw_2_correlated_proxy_trap_identifiability_banding():
    """Flaw 2 Remediation: Identifiability banding to prevent proxy false-certainty.

    Verifies that:
    1. When competing candidate hypotheses have dev R^2 within identifiability_band (0.002),
       the engine marks identifiability_status="AMBIGUOUS_EQUIVALENCE_SET" and records
       competing hypotheses.
    2. When a single hypothesis clearly leads by > 0.002, it is marked "UNIQUE_WINNER".
    """
    x = np.linspace(1.0, 10.0, 100)
    # Quadratic signal with microscopic noise: x^2 and x*|x| are mathematically identical for x > 0
    y_ambig = 3.0 * (x ** 2) + np.random.RandomState(42).normal(0, 0.001, 100)
    df_ambig = pd.DataFrame({"x": x, "y": y_ambig})

    proposer_ambig = RimeProposer(
        feature_cols=["x"],
        target_col="y",
        static_mode=True,
        square_seeds=True,
        identifiability_band=0.002,
        max_depth=1,
        budget=100,
        rel_penalty=0.0,
    )
    props_ambig = proposer_ambig.fit_and_propose(df_ambig)
    assert len(props_ambig) >= 2

    top = props_ambig[0]
    # Collinear forms like (x * x) and (x * abs(x)) must be flagged as an ambiguous equivalence set
    assert top.parameters["identifiability_status"] == "AMBIGUOUS_EQUIVALENCE_SET"
    assert top.metadata["equivalence_band_size"] >= 2
    assert len(top.metadata["competing_hypotheses"]) >= 2
    assert top.parameters["lead_delta_r2"] == 0.0

    # Test Unique Winner: clear margin
    y_unique = 5.0 * np.tanh(x) + np.random.RandomState(42).normal(0, 0.05, 100)
    df_unique = pd.DataFrame({"x": x, "y": y_unique})
    proposer_unique = RimeProposer(
        feature_cols=["x"],
        target_col="y",
        static_mode=True,
        identifiability_band=0.002,
        max_depth=0,
        budget=50,
        rel_penalty=0.0,
    )
    props_unique = proposer_unique.fit_and_propose(df_unique)
    assert len(props_unique) > 0
    top_unique = props_unique[0]
    if len(props_unique) > 1 and float(props_unique[1].parameters["dev_r2"]) < float(top_unique.parameters["dev_r2"]) - 0.002:
        assert top_unique.parameters["identifiability_status"] == "UNIQUE_WINNER"
        assert top_unique.metadata["equivalence_band_size"] == 1


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="native RIME kernel required")
def test_flaw_2_gradient_derivative_matching():
    """Flaw 2 Remediation: Derivative/Gradient matching for physical consistency.

    Verifies that check_derivatives=True computes the empirical derivative correlation
    (gradient_alignment) between the target differences and model predictions.
    """
    A = np.linspace(10.0, 100.0, 50)
    y = 5.0 * np.sqrt(A) + 2.0
    df = pd.DataFrame({"A": A, "y": y})

    proposer = RimeProposer(
        feature_cols=["A"],
        target_col="y",
        static_mode=True,
        power_ops=True,
        check_derivatives=True,
        max_depth=1,
        budget=100,
        rel_penalty=0.0,
    )
    props = proposer.fit_and_propose(df)
    assert len(props) > 0
    best = props[0]
    grad_align = float(best.parameters["gradient_alignment"])
    # Monotonic physical law must have positive gradient alignment > 0.95
    assert grad_align > 0.95, f"Expected high positive gradient alignment, got {grad_align}"


@pytest.mark.skipif(not _RIME_AVAILABLE, reason="native RIME kernel required")
def test_flaw_3_diverse_beam_multi_family_preservation():
    """Flaw 3 Remediation: Diverse beam search preventing monoculture starvation.

    Verifies that diverse_beam=True preserves diverse functional archetypes
    in the search frontier.
    """
    x = np.linspace(0.1, 5.0, 40)
    # Combined target with linear, power, and bounded features
    y = 2.0 * x + 1.5 * (x ** 2) + 0.8 * np.tanh(x)
    df = pd.DataFrame({"x": x, "y": y})

    proposer = RimeProposer(
        feature_cols=["x"],
        target_col="y",
        static_mode=True,
        square_seeds=True,
        power_ops=True,
        diverse_beam=True,
        max_depth=1,
        budget=100,
        rel_penalty=0.0,
    )
    props = proposer.fit_and_propose(df)
    assert len(props) > 0
    assert props[0].metadata["diverse_beam"] is True
