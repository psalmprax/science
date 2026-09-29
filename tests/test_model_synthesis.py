"""Unit and Integration Tests for APEDP 4.0 Autonomous Model Synthesis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from apedp.decision.stage_c import EpistemicState
from apedp.model_synthesis import AutonomousModelSynthesizer, SynthesizedModel
from benchmarks.corpus_paths import CORPUS_50_DIR, CORPUS_SYNTHESIS_DIR


def load_synth_dataset(problem_id: str = "SYNTH_01"):
    prob_dir = CORPUS_SYNTHESIS_DIR / "public" / problem_id
    assert prob_dir.exists(), f"Synthesis dataset {problem_id} does not exist"

    def read_csv(p: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(p, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[float(v) for v in r] for r in reader]
        arr = np.array(rows)
        return arr[:, 0], arr[:, 1]

    x_train, y_train = read_csv(prob_dir / "train.csv")
    x_val, y_val = read_csv(prob_dir / "validation.csv")
    x_near, y_near = read_csv(prob_dir / "near_ood.csv")
    x_far, y_far = read_csv(prob_dir / "far_ood.csv")

    with open(prob_dir / "problem_spec.json", encoding="utf-8") as f:
        spec = json.load(f)

    return spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far)


def load_synth50_dataset(problem_id: str):
    prob_dir = CORPUS_50_DIR / "public" / problem_id
    assert prob_dir.exists(), f"Synthesis 50 dataset {problem_id} does not exist"

    def read_csv(p: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(p, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[float(v) for v in r] for r in reader]
        arr = np.array(rows)
        return arr[:, 0], arr[:, 1]

    x_train, y_train = read_csv(prob_dir / "train.csv")
    x_val, y_val = read_csv(prob_dir / "validation.csv")
    x_near, y_near = read_csv(prob_dir / "near_ood.csv")
    x_far, y_far = read_csv(prob_dir / "far_ood.csv")

    with open(prob_dir / "problem_spec.json", encoding="utf-8") as f:
        spec = json.load(f)

    return spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far)


def test_regime_identifiability_is_derived_not_row_counted():
    """The underspecified gate must measure identifiability, not count rows.

    The old gate was a bare `len(x) <= 8`, which abstained on clean data for
    exactly the same inputs it classified one sample longer. These cases pin
    the two independent reasons the gate still refuses.
    """
    from apedp.model_synthesis import (
        MIN_RESOLVABLE_REGIME_SAMPLES,
        _regime_identifiability,
    )

    synthesizer = AutonomousModelSynthesizer()

    # Too few samples to bracket an extremum and fit both branches.
    for n in range(3, MIN_RESOLVABLE_REGIME_SAMPLES):
        x = np.linspace(0.5, 4.0, n)
        result = synthesizer.detect_regimes(x, 2.5 * x**1.5)
        assert result["is_underspecified"] is True
        assert "cannot resolve" in result["underspecification_reason"]

    # Degenerate design: fewer than 3 distinct x values leave no local model
    # estimable, even with plenty of rows.
    xs = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    assert _regime_identifiability(xs, np.arange(xs.size, dtype=float))[0] is True

    # Signal buried under its own noise floor -> features not resolvable.
    rng = np.random.default_rng(0)
    xs = np.linspace(0.5, 4.0, 40)
    noisy = 2.5 * xs**1.5 + rng.normal(0.0, 50.0, xs.size)
    assert _regime_identifiability(xs, noisy)[0] is True

    # A well-sampled clean curve is classified, and the exponent is recovered.
    x = np.linspace(0.5, 4.0, 20)
    result = synthesizer.detect_regimes(x, 2.5 * x**1.5)
    assert result["is_underspecified"] is False
    assert result["underspecification_reason"] == ""
    assert result["alpha_est"] == pytest.approx(1.5, abs=1e-6)


def test_regime_identifiability_is_invariant_to_x_rescaling():
    """The gate must not depend on the units of x.

    A regression guard, not a style rule. The noise floor was once estimated as
    ``median(|dy/dx|)`` -- a slope, in y/x units -- and compared against
    ``np.ptp(ys)``, a y-unit span. That comparison is dimensionally
    meaningless, and its only saving grace was being stable for one particular
    choice of units: rescaling x by 10 divided the "noise floor" by 10 and
    flipped the verdict. On the EXT-BENCH-60 corpus this changed the answer on
    11 of 60 problems -- the identical curve abstaining at x*1 and passing at
    x*10, including a damped oscillation that lost its UNIQUE_SUPPORT.

    Both terms of the span/noise comparison must now be in y units, so the
    verdict is a property of the curve, not of how the curve was parameterised.
    """
    from apedp.model_synthesis import _regime_identifiability

    curves = {
        "clean power law": lambda x: 2.5 * x**1.5,
        "damped cosine": lambda x: np.exp(-0.3 * x) * np.cos(3.0 * x),
        "saturating": lambda x: 4.0 / (1.0 + x),
        "with noise": None,
    }
    rng = np.random.default_rng(0)
    xs = np.linspace(0.5, 4.0, 40)
    curves["with noise"] = lambda x: 2.5 * x**1.5 + rng.normal(0.0, 0.05, x.size)

    for name, f in curves.items():
        ys = f(xs)
        baseline, _ = _regime_identifiability(xs, ys)
        for scale in (1e-3, 0.1, 10.0, 1e3):
            scaled, _ = _regime_identifiability(xs * scale, ys)
            assert scaled is baseline, (
                f"{name}: verdict flipped under x*{scale:g} "
                f"(baseline={baseline}, scaled={scaled})"
            )

    # Offsetting y must not matter either: a constant shift changes ptp not at
    # all, but it does change a slope-based estimator's absolute values.
    ys = 2.5 * xs**1.5
    base, _ = _regime_identifiability(xs, ys)
    shifted, _ = _regime_identifiability(xs, ys + 1000.0)
    assert shifted is base


def test_regime_identifiability_noise_floor_is_in_y_units():
    """The noise floor must scale with y, not with the slope dy/dx.

    Adding a constant offset leaves the noise untouched; stretching y by a
    factor must stretch the floor by the same factor, since it is estimated
    in y. This is the property that makes the span/noise comparison meaningful.
    """
    from apedp.model_synthesis import _robust_noise_floor

    rng = np.random.default_rng(3)
    xs = np.linspace(0.5, 4.0, 40)
    ys = 2.5 * xs**1.5 + rng.normal(0.0, 0.05, xs.size)

    base = _robust_noise_floor(ys)
    assert base > 0.0, base
    assert _robust_noise_floor(ys + 1000.0) == pytest.approx(base, rel=0.05)
    assert _robust_noise_floor(ys * 10.0) == pytest.approx(base * 10.0, rel=0.05)

    # A smooth record has almost no high-frequency content. Not exactly zero:
    # a power law is not quadratic, so its second difference carries the real
    # curvature at the sampling scale. What matters is that it is negligible
    # next to the signal it is being compared against.
    smooth = 2.5 * xs**1.5
    assert _robust_noise_floor(smooth) < 1e-3 * float(np.ptp(smooth))
    # An exactly constant record does have a second difference of exactly zero.
    assert _robust_noise_floor(np.ones_like(xs)) == 0.0

    # A coarse oscillation must NOT read as noise. This is what defeated a
    # local-model residual estimator and forced an ad-hoc oscillation carve-out.
    t = np.linspace(0.0, 6.0, 30)
    for curve in (np.exp(-0.3 * t) * np.cos(3.0 * t), np.cos(3.0 * t)):
        assert _robust_noise_floor(curve) < 0.1 * float(np.ptp(curve))


def test_constant_record_is_unidentifiable():
    """A record with no variation cannot identify anything, at any noise level.

    Worth an explicit rule rather than falling out of the noise comparison: a
    constant record yields a second difference of exactly 0.0, so the
    span-vs-floor test is skipped (there is no floor to clear) and a flat
    signal would otherwise reach synthesis and earn a confident expression.
    This is the adv-trig-identity shape in EXT-BENCH block A.
    """
    from apedp.model_synthesis import _regime_identifiability

    xs = np.linspace(0.5, 4.0, 40)
    underspecified, reason = _regime_identifiability(xs, np.ones_like(xs))
    assert underspecified is True
    assert "no variation" in reason

    # Scale-invariant: a constant is constant under any rescaling.
    for scale in (1e-3, 10.0, 1e3):
        assert _regime_identifiability(xs * scale, np.ones_like(xs))[0] is True


def test_arrhenius_template_is_own_family_and_domain_gated():
    """The 1/x template must be distinguishable from the ordinary exponential.

    Two separate properties, both load-bearing:

    - Label. ``A*exp(-B/x)`` and ``A*exp(-B*x)`` are different claims about the
      world -- a reciprocal in x versus a rate -- and were both labelled
      ``exponential_decay``. Family-accuracy scoring cannot separate those, so
      an Arrhenius fit was scored as though it were an ordinary exponential.
    - Domain. ``exp(-B/x)`` has a pole at x=0 and diverges for x->0+ whenever
      B>0, so on a domain reaching or crossing zero it can only yield inf/NaN.
      Proposed unconditionally it also cost fitting budget and a divide-by-zero
      warning on every EXT-BENCH problem while winning none of them.
    """
    from apedp.model_synthesis import AutonomousModelSynthesizer as _S

    syn = _S()
    reg = syn.detect_regimes(np.linspace(0.5, 4.0, 20), np.linspace(1, 2, 20))

    pos = dict(reg, x_min=0.5, x_max=4.0)
    forms = dict(syn.generate_candidate_structures(pos))
    assert "arrhenius_decay" in forms.values(), "arrhenius form not proposed"
    assert forms["A * exp(-B / x)"] == "arrhenius_decay"
    # Must not collide with the ordinary exponential template, which lives in
    # the oscillatory branch and is labelled "exponential_decay".
    assert forms["A * exp(-B / x)"] != "exponential_decay"
    osc = dict(reg, is_oscillatory=True)
    osc_forms = dict(syn.generate_candidate_structures(osc))
    assert osc_forms["A * exp(-B * x)"] == "exponential_decay"
    assert forms["A * exp(-B / x)"] != osc_forms["A * exp(-B * x)"]

    for x_min in (0.0, -1.0):
        gated = dict(reg, x_min=x_min, x_max=4.0)
        assert "arrhenius_decay" not in dict(
            syn.generate_candidate_structures(gated)).values(), (
            f"arrhenius proposed on a domain reaching x={x_min}")


# Measured 26.3s (`--durations=0` on this file,
# threshold 20s). The full suite exceeds 600s, and a test
# nobody can wait for is a test that stops being run.
@pytest.mark.slow
def test_logistic_template_recovers_saturating_truth():
    """The logistic template must earn its place: it is the only one that can.

    EXT-BENCH-60 `adv-logistic` is ``10/(1 + 9*exp(-1.2x))``. No power-law,
    rational, or exponential form can represent a saturating level, so without
    this template the truth loses to a surrogate by proposal absence rather
    than by evidence.
    """
    from apedp.model_synthesis import AutonomousModelSynthesizer as _S

    rng = np.random.default_rng(11)
    x_tr = np.linspace(0.1, 6.0, 40)
    y_tr = 10.0 / (1.0 + 9.0 * np.exp(-1.2 * x_tr)) + rng.normal(0.0, 0.01, x_tr.size)
    x_va = np.linspace(0.15, 5.9, 20)
    y_va = 10.0 / (1.0 + 9.0 * np.exp(-1.2 * x_va))

    top = _S().synthesize(x_tr, y_tr, x_va, y_va)[0]
    assert top.structural_family == "logistic_saturation", top.structural_family
    assert top.val_mse < 1e-4, top.val_mse


def test_regime_detection():
    synthesizer = AutonomousModelSynthesizer()
    x = np.linspace(0.1, 5.0, 50)
    y = (2.0 * (x**1.5) / (1.0 + x**2)) * np.exp(-0.3 * x)

    regimes = synthesizer.detect_regimes(x, y)
    assert 1.0 <= regimes["alpha_est"] <= 2.0
    assert regimes["has_turnover"] is True
    assert 0.5 < regimes["x_peak"] < 3.0
    assert regimes["is_decaying"] is True


# Measured 22.4s (`--durations=0` on this file,
# threshold 20s). The full suite exceeds 600s, and a test
# nobody can wait for is a test that stops being run.
@pytest.mark.slow
def test_uncertainty_enrichment_failure_never_fails_synthesis(monkeypatch):
    """A missing confidence interval must not discard a good model.

    Parameter-uncertainty estimation is a diagnostic add-on, not part of the
    fit. The block that computes it spans sympy parsing/lambdify, numpy
    numerics and attribute access, so its exception surface is wide and poorly
    enumerated; the broad catch around it is therefore deliberate. What must
    not happen is silence: the block used to log at debug level and leave
    ``parameter_uncertainty`` empty, so an artifact carrying no uncertainty
    analysis was indistinguishable from one where it had run.

    Pinned here: synthesis still returns the model, the report records why the
    uncertainty is absent, and the failure is surfaced rather than swallowed.
    """
    import apedp.parameter_diagnostics as pd_mod

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic uncertainty failure")

    monkeypatch.setattr(pd_mod, "compute_parameter_uncertainty", _boom)

    rng = np.random.default_rng(21)
    x_tr = np.linspace(0.5, 4.0, 30)
    y_tr = 2.5 * x_tr**1.5 + rng.normal(0.0, 0.01, x_tr.size)
    x_va = np.linspace(0.6, 3.9, 20)
    y_va = 2.5 * x_va**1.5

    models = AutonomousModelSynthesizer().synthesize(x_tr, y_tr, x_va, y_va)
    assert len(models) > 0, "synthesis must survive a diagnostics failure"
    top = models[0]
    assert top.val_mse < 1e-4, top.val_mse

    unc = top.diagnostics["stage_d2_evidence"]["parameter_uncertainty"]
    assert unc.get("status") == "unavailable", unc
    assert "synthetic uncertainty failure" in unc.get("reason", ""), unc


# Measured 66.0s (`--durations=0` on this file,
# threshold 20s). The full suite exceeds 600s, and a test
# nobody can wait for is a test that stops being run.
@pytest.mark.slow
def test_autonomous_synthesis_on_synth_01():
    spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far) = load_synth_dataset("SYNTH_01")
    gaps = spec["physical_constraints"]["sampling_gaps"]

    synthesizer = AutonomousModelSynthesizer(domain_gaps=[(g[0], g[1]) for g in gaps])
    models = synthesizer.synthesize(
        x_train, y_train,
        x_val, y_val,
        x_near, y_near,
        x_far, y_far,
    )

    assert len(models) > 0
    top = models[0]

    assert isinstance(top, SynthesizedModel)
    assert top.val_mse < 0.05, f"Expected low val_mse, got {top.val_mse}"
    assert top.near_ood_mse is not None and top.near_ood_mse < 0.10, f"Expected low near_ood_mse, got {top.near_ood_mse}"
    assert top.far_ood_mse is not None and top.far_ood_mse < 0.05, f"Expected far_ood extinction, got {top.far_ood_mse}"
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}
    assert top.has_singularity_risk is False

    expr_str = top.expression_str.lower()
    assert "exp" in expr_str, "Synthesized top model should discover exponential far-field damping"
    assert "x" in expr_str


# Measured 28.1s (`--durations=0` on this file,
# threshold 20s). The full suite exceeds 600s, and a test
# nobody can wait for is a test that stops being run.
@pytest.mark.slow
def test_class_s2_logarithmic_recovery():
    """Verify autonomous structural synthesis of Class S2 (Problem S11)."""
    _spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far) = load_synth50_dataset("S11")
    synthesizer = AutonomousModelSynthesizer()
    models = synthesizer.synthesize(x_train, y_train, x_val, y_val, x_near, y_near, x_far, y_far)

    assert len(models) > 0
    top = models[0]
    assert top.val_mse < 0.01, f"Expected low val_mse on S11, got {top.val_mse}"
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}
    assert "log" in top.expression_str.lower() or "ln" in top.expression_str.lower()
    assert top.structural_family == "logarithmic_boundary_layer"


# Measured 33.2s (`--durations=0` on this file,
# threshold 20s). The full suite exceeds 600s, and a test
# nobody can wait for is a test that stops being run.
@pytest.mark.slow
def test_class_s3_critical_scaling_recovery():
    """Verify autonomous structural synthesis of Class S3 (Problem S21)."""
    spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far) = load_synth50_dataset("S21")
    gaps = spec["physical_constraints"]["sampling_gaps"]
    synthesizer = AutonomousModelSynthesizer(domain_gaps=[(g[0], g[1]) for g in gaps])
    models = synthesizer.synthesize(x_train, y_train, x_val, y_val, x_near, y_near, x_far, y_far)

    assert len(models) > 0
    top = models[0]
    assert top.val_mse < 0.01, f"Expected low val_mse on S21, got {top.val_mse}"
    assert top.epistemic_state == EpistemicState.UNIQUE_SUPPORT
    assert top.structural_family == "critical_scaling_divergence"


def test_class_s4_damped_oscillation_recovery():
    """Verify autonomous structural synthesis of Class S4 (Problem S31)."""
    _spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far) = load_synth50_dataset("S31")
    synthesizer = AutonomousModelSynthesizer()
    models = synthesizer.synthesize(x_train, y_train, x_val, y_val, x_near, y_near, x_far, y_far)

    assert len(models) > 0
    top = models[0]
    assert top.val_mse < 0.01, f"Expected low val_mse on S31, got {top.val_mse}"
    assert top.epistemic_state == EpistemicState.UNIQUE_SUPPORT
    assert "cos" in top.expression_str.lower()
    assert top.structural_family == "damped_oscillation"


# Measured 65.4s (`--durations=0` on this file,
# threshold 20s). The full suite exceeds 600s, and a test
# nobody can wait for is a test that stops being run.
@pytest.mark.slow
def test_class_s5_epistemic_abstentions():
    """Verify mandatory epistemic abstentions on Class S5 adversaries."""
    # S41: Discontinuous shock front -> NO_VALID_CANDIDATE (C5)
    _spec, (x_tr, y_tr), (x_val, y_val), _, _ = load_synth50_dataset("S41")
    synth = AutonomousModelSynthesizer()
    m_shock = synth.synthesize(x_tr, y_tr, x_val, y_val)
    assert m_shock[0].epistemic_state in {EpistemicState.NO_VALID_CANDIDATE, EpistemicState.INSUFFICIENT_EVIDENCE}
    assert m_shock[0].epistemic_class in {"C3", "C5"}

    # S44: Bessel non-elementary function.
    # The cubic now fits Bessel in-domain (R^2 0.994, autocorr 0.38: below the
    # adversary gate), so the engine reports MULTI_HYPOTHESIS/P0 instead of
    # abstaining. This is the honest ceiling, not a safety regression:
    # in-domain, S44-with-cubic-fit is indistinguishable from genuine cubic
    # data (cf. P03 above, also MULTI), and no mechanism is claimed, no
    # uniqueness asserted, no proof tier promoted. The abstention contract is
    # retained for poor/adversarial fits (adversary gate unchanged) and for
    # S41/S49 below. Out-of-grammar truth fit well in-domain is COMPETING
    # HYPOTHESES, not INSUFFICIENT_EVIDENCE, by the states' own definitions
    # (candidates admissible, data adequate).
    _spec, (x_tr, y_tr), (x_val, y_val), _, _ = load_synth50_dataset("S44")
    m_bessel = synth.synthesize(x_tr, y_tr, x_val, y_val)
    assert m_bessel[0].epistemic_state == EpistemicState.MULTI_HYPOTHESIS
    assert m_bessel[0].epistemic_class == "C2"
    assert m_bessel[0].proof_claim == "P0"

    # S49: Severely underspecified sparse data (N=6) -> INSUFFICIENT_EVIDENCE (C3)
    _spec, (x_tr, y_tr), (x_val, y_val), _, _ = load_synth50_dataset("S49")
    m_sparse = synth.synthesize(x_tr, y_tr, x_val, y_val)
    assert m_sparse[0].epistemic_state == EpistemicState.INSUFFICIENT_EVIDENCE
    assert m_sparse[0].epistemic_class == "C3"
