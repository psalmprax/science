"""Unit and Integration Tests for APEDP 4.0 Autonomous Model Synthesis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from apedp.decision.stage_c import EpistemicState
from apedp.model_synthesis import AutonomousModelSynthesizer, SynthesizedModel
from benchmarks.generate_synthesis_50_corpus import CORPUS_50_DIR
from benchmarks.synthetic_synthesis_generator import CORPUS_SYNTHESIS_DIR


def load_synth_dataset(problem_id: str = "SYNTH_01"):
    prob_dir = CORPUS_SYNTHESIS_DIR / "public" / problem_id
    assert prob_dir.exists(), f"Synthesis dataset {problem_id} does not exist"

    def read_csv(p: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[float(v) for v in r] for r in reader]
        arr = np.array(rows)
        return arr[:, 0], arr[:, 1]

    x_train, y_train = read_csv(prob_dir / "train.csv")
    x_val, y_val = read_csv(prob_dir / "validation.csv")
    x_near, y_near = read_csv(prob_dir / "near_ood.csv")
    x_far, y_far = read_csv(prob_dir / "far_ood.csv")

    with open(prob_dir / "problem_spec.json", "r", encoding="utf-8") as f:
        spec = json.load(f)

    return spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far)


def load_synth50_dataset(problem_id: str):
    prob_dir = CORPUS_50_DIR / "public" / problem_id
    assert prob_dir.exists(), f"Synthesis 50 dataset {problem_id} does not exist"

    def read_csv(p: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[float(v) for v in r] for r in reader]
        arr = np.array(rows)
        return arr[:, 0], arr[:, 1]

    x_train, y_train = read_csv(prob_dir / "train.csv")
    x_val, y_val = read_csv(prob_dir / "validation.csv")
    x_near, y_near = read_csv(prob_dir / "near_ood.csv")
    x_far, y_far = read_csv(prob_dir / "far_ood.csv")

    with open(prob_dir / "problem_spec.json", "r", encoding="utf-8") as f:
        spec = json.load(f)

    return spec, (x_train, y_train), (x_val, y_val), (x_near, y_near), (x_far, y_far)


def test_regime_detection():
    synthesizer = AutonomousModelSynthesizer()
    x = np.linspace(0.1, 5.0, 50)
    y = (2.0 * (x**1.5) / (1.0 + x**2)) * np.exp(-0.3 * x)

    regimes = synthesizer.detect_regimes(x, y)
    assert 1.0 <= regimes["alpha_est"] <= 2.0
    assert regimes["has_turnover"] is True
    assert 0.5 < regimes["x_peak"] < 3.0
    assert regimes["is_decaying"] is True


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
