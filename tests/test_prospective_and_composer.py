# Databricks notebook source
"""Tests for the prospective/composer roadmap (A1-A4, B)."""
from __future__ import annotations

import csv

import sympy as sp

from apedp.composer import THEORETICAL_MATURITY, HypothesisComposer
from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.discovery import MultiVariableSRE
from apedp.epistemic import DataPartitionVault, PriorArtAuditor
from apedp.safety import safe_parse_expr


def test_a1_rename_isomorphism_matches_ohm():
    r = PriorArtAuditor.audit_novelty("I*R")
    assert r.status.value == "KNOWN_MODEL"
    assert len(PriorArtAuditor.STANDARD_PHYSICS_CORPUS) >= 20
    assert PriorArtAuditor.corpus_hash() == PriorArtAuditor.corpus_hash()


def test_a1_trivial_parabola_stays_novel():
    r = PriorArtAuditor.audit_novelty("x**2")
    assert r.status.value == "APPARENTLY_NOVEL_WITHIN_CORPUS"
    assert "does NOT constitute" in r.epistemic_disclaimer


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in rows:
            w.writerow([x, y])


def test_a2_prospective_persists_on_matching_holdout(tmp_path):
    from apedp.pipeline import DiscoveryPipeline

    tr = tmp_path / "train.csv"
    ho = tmp_path / "hold.csv"
    _write_csv(str(tr), [(float(x), float(x**2)) for x in range(1, 11)])
    _write_csv(str(ho), [(float(x), float(x**2)) for x in range(11, 16)])
    pipe = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    r = pipe.run_prospective(str(tr), str(ho), "y", variable_units={"x": "dimensionless"})
    assert r["status"] == "PERSISTED"
    assert r["holdout_r_squared"] > 0.99
    assert r["evidence_level"] == "locked_holdout_prospective"


def test_a2_prospective_refuses_shifted_holdout(tmp_path):
    from apedp.pipeline import DiscoveryPipeline

    tr = tmp_path / "train.csv"
    ho = tmp_path / "hold.csv"
    _write_csv(str(tr), [(float(x), float(x**2)) for x in range(1, 11)])
    _write_csv(str(ho), [(float(x), float(x**3)) for x in range(11, 16)])
    pipe = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    r = pipe.run_prospective(str(tr), str(ho), "y", variable_units={"x": "dimensionless"})
    assert r["status"] == "REFUSED"
    assert pipe.store.list_models() == []


def test_a2_vault_blocks_development_touching_holdout():
    v = DataPartitionVault("train", "train", "hold", "hold")
    assert v.verify_isolation("train", "development") is True
    try:
        v.verify_isolation("hold", "development")
    except PermissionError:
        return
    raise AssertionError("vault should have raised")


def test_a3_erf_parses_and_gates_dimensionless():
    expr = safe_parse_expr("erf(x)")
    eng = DimensionalAssuranceEngine()
    eng.register_variable("x", "dimensionless")
    assert eng.check_expression_consistency(expr).dimensionality == eng.registry("dimensionless").dimensionality


def test_a3_thirds_monomial_reachable():
    import random
    from fractions import Fraction

    random.seed(12345)
    eng = DimensionalAssuranceEngine()
    plat = ClosedLoopAssuranceSystem(eng)
    sre = MultiVariableSRE(assurance_system=plat, independent_vars=["a"], target_unit_str="dimensionless")
    eng.register_variable("a", "meter")

    block_dims = {sp.Symbol("a"): {"meter": 1.0}}
    got = sre._monomial_for_dim(
        {"meter": float(Fraction(1, 3))}, block_dims, tries=2000
    )
    assert got is not None
    assert got.has(sp.Symbol("a"))


def test_a4_metrology_passthrough_and_lambda_cache(tmp_path):
    from apedp.pipeline import DiscoveryPipeline

    p = tmp_path / "data.csv"
    _write_csv(str(p), [(float(x), float(2 * x + 1)) for x in range(1, 11)])
    pipe = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    r = pipe.run(str(p), "y", variable_units={"x": "dimensionless"}, param_uncertainties={"x": 0.01})
    assert r["status"] == "PERSISTED"
    assert r["metrology"] is not None
    assert r["metrology"]["method"] in ("gum", "monte_carlo")

    eng = DimensionalAssuranceEngine()
    plat = ClosedLoopAssuranceSystem(eng)
    sre = MultiVariableSRE(assurance_system=plat, independent_vars=["x"])
    syms = [sp.Symbol("x")]
    e = sp.Symbol("x") ** 2
    assert sre._lambdify_cached(syms, e) is sre._lambdify_cached(syms, e)


def test_b_composer_returns_ranked_speculation():
    c = HypothesisComposer()
    res = c.compose(["m*v**2/2", "G*m1*m2/r**2"], max_candidates=5)
    assert res
    assert all(r.maturity == THEORETICAL_MATURITY for r in res)
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)
    # Handoff: templates must parse in the fitter without crashing seeding.
    eng = DimensionalAssuranceEngine()
    plat = ClosedLoopAssuranceSystem(eng)
    priors = [r.as_template_prior() for r in res[:2]]
    sre = MultiVariableSRE(
        assurance_system=plat, independent_vars=["m", "v"],
        target_unit_str="joule", template_priors=priors,
    )
    assert isinstance(sre.template_exprs, list)


def test_corpus_covers_broad_law_classes():
    from apedp.epistemic import PriorArtAuditor

    assert len(PriorArtAuditor.STANDARD_PHYSICS_CORPUS) >= 150
    for expr, ref in [
        ("m*v**2/2", "KINETIC_ENERGY"),
        ("2*pi*sqrt(L/g)", "PENDULUM_PERIOD"),
        ("A*exp(-Ea/(R*T))", "ARRHENIUS_RATE"),
    ]:
        r = PriorArtAuditor.audit_novelty(expr)
        assert r.status.value == "KNOWN_MODEL", (expr, r.status)
        assert r.closest_reference_id == ref


def test_pde_templates_match_canonical_shapes():
    from apedp import recognize_pde
    from apedp.safety import safe_parse_pde

    assert "DIFFUSION_1D" in recognize_pde(safe_parse_pde("Derivative(T(x,t),t) - D*Derivative(T(x,t),x,2)"))
    assert "WAVE_1D" in recognize_pde(safe_parse_pde("Derivative(u(x,t),t,2) - c**2*Derivative(u(x,t),x,2)"))
    assert "LAPLACE_2D" in recognize_pde(safe_parse_pde("Derivative(u(x,t),x,2) + Derivative(u(x,t),y,2)"))


def test_pde_audit_and_out_of_scope():
    from apedp.epistemic import PriorArtAuditor

    r = PriorArtAuditor.audit_novelty("Derivative(T(x,t),t) - D*Derivative(T(x,t),x,2)")
    assert r.status.value == "KNOWN_MODEL"
    assert (r.closest_reference_id or "").startswith("PDE_TEMPLATE_")
    r2 = PriorArtAuditor.audit_novelty("not (( a formula")
    assert r2.status.value == "OUT_OF_SCOPE"


# COMMAND ----------

import numpy as np

from apedp.discovery_admission import admit_candidate, numeric_conditioning_report


def test_admission_persists_valid_locked_holdout_candidate():
    x = np.arange(1.0, 6.0)
    decision = admit_candidate(
        sp.Symbol("x") ** 2,
        x_holdout={"x": x},
        y_holdout=x ** 2,
        train_r_squared=1.0,
        min_holdout_r_squared=0.99,
        max_r2_drop=0.01,
        dimensionally_consistent=True,
        repair_status="ACCEPTED",
        conditioning=numeric_conditioning_report(sp.Symbol("x") ** 2, {"x": x}),
    )
    assert decision.status.value == "PERSISTED"
    assert decision.evidence["holdout_r_squared"] == 1.0


def test_admission_refuses_failed_repair_even_with_perfect_holdout():
    x = np.arange(1.0, 6.0)
    decision = admit_candidate(
        sp.Symbol("x") ** 2,
        x_holdout={"x": x},
        y_holdout=x ** 2,
        train_r_squared=1.0,
        min_holdout_r_squared=0.99,
        max_r2_drop=0.01,
        repair_status="REJECTED",
    )
    assert decision.status.value == "REFUSED"
    assert "repair status is not admissible: REJECTED" in decision.reasons


def test_admission_refuses_unconditioned_cancellation_and_missing_baseline():
    x = np.array([1.0, 2.0, 3.0])
    symbol = sp.Symbol("x")
    expression = sp.Add(1.0e16 * symbol, -1.0e16 * symbol, symbol, evaluate=False)
    conditioning = numeric_conditioning_report(expression, {"x": x}, cancellation_ratio_limit=1.0e6)
    decision = admit_candidate(
        expression,
        x_holdout={"x": x},
        y_holdout=x,
        train_r_squared=1.0,
        min_holdout_r_squared=0.99,
        max_r2_drop=0.01,
        require_baseline_improvement=True,
        conditioning=conditioning,
    )
    assert conditioning["cancellation_within_limit"] is False
    assert decision.status.value == "REFUSED"
    assert "destructive cancellation exceeds declared limit" in decision.reasons
    assert "baseline improvement is required but no baseline was supplied" in decision.reasons

# COMMAND ----------

from apedp.cross_domain_benchmark import (
    DEFAULT_CASES,
    BenchmarkOutcome,
    EvidenceTrack,
    dataset_hash,
    run_protocol,
    summarize,
)


def test_cross_domain_protocol_is_complete_and_hashes_data():
    rows = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}]
    assert dataset_hash(rows) == dataset_hash(rows)
    outcomes = run_protocol(
        DEFAULT_CASES,
        lambda case, track, seed: BenchmarkOutcome(
            case_id=case.case_id,
            track=track,
            seed=seed,
            dataset_hash=dataset_hash(rows),
            discovered=not case.expected_refusal,
            admitted=not case.expected_refusal,
            refused=case.expected_refusal,
            baseline_improved=True if not case.expected_refusal else None,
            stable_across_seeds=True if not case.expected_refusal else None,
            expression="x**2" if not case.expected_refusal else None,
            refusal_reason="locked holdout failure" if case.expected_refusal else None,
        ),
    )
    assert len(outcomes) == len(DEFAULT_CASES) * len(EvidenceTrack) * 3
    summary = summarize(outcomes)
    assert summary.total == len(outcomes)
    assert summary.false_admission_rate == 0.0
    assert summary.by_track[EvidenceTrack.UNINFORMED.value]["n"] == float(len(DEFAULT_CASES) * 3)


def test_cross_domain_summary_does_not_hide_unmeasured_metrics():
    case = DEFAULT_CASES[0]
    outcome = BenchmarkOutcome(
        case_id=case.case_id,
        track=EvidenceTrack.UNINFORMED,
        seed=7,
        dataset_hash="locked",
        discovered=True,
        admitted=True,
        refused=False,
        baseline_improved=None,
        stable_across_seeds=None,
    )
    summary = summarize([outcome], cases=DEFAULT_CASES)
    assert summary.baseline_improvement_rate is None
    assert summary.seed_stability_rate is None

# COMMAND ----------

from apedp.cross_domain_benchmark import (
    DiscoveryStatus,
    SyntheticDiscoveryAdapter,
    run_adapter_protocol,
)


def test_synthetic_adapter_runs_real_discovery_to_outcome():
    case = DEFAULT_CASES[0]
    outcomes = run_adapter_protocol(
        SyntheticDiscoveryAdapter(),
        [case],
        tracks=[EvidenceTrack.UNINFORMED],
        seeds=[7],
    )
    assert len(outcomes) == 1
    assert outcomes[0].dataset_hash
    assert outcomes[0].evidence["provenance"]["case_id"] == case.case_id
    assert outcomes[0].evidence["holdout_r_squared"] is not None
    assert outcomes[0].evidence["track_configuration"]["prior_disclosure"] == "none"
    assert outcomes[0].status == DiscoveryStatus.INCONCLUSIVE
    assert outcomes[0].evidence["information_firewall"]["hidden"]


def test_synthetic_adapter_declares_distinct_prior_tracks():
    case = DEFAULT_CASES[0]
    outcomes = run_adapter_protocol(
        SyntheticDiscoveryAdapter(),
        [case],
        tracks=[EvidenceTrack.UNINFORMED, EvidenceTrack.WEAK_PRIOR, EvidenceTrack.LITERATURE_INFORMED],
        seeds=[7],
    )
    configurations = [outcome.evidence["track_configuration"] for outcome in outcomes]
    assert configurations[0]["template_priors"] == []
    assert configurations[1]["prior_disclosure"] == "generic_alternatives"
    assert configurations[2]["prior_disclosure"] == "declared_reference_form"


def test_discovery_status_refuses_conflicting_firewall_declarations():
    from apedp.cross_domain_benchmark import InformationFirewall

    try:
        InformationFirewall(("units",), ("units",), (), ())
    except ValueError:
        return
    raise AssertionError("firewall should reject known/hidden overlap")

# COMMAND ----------



def test_domain_limited_status_requires_explicit_domain_evidence():
    outcome = BenchmarkOutcome(
        case_id="materials_hall_petch",
        track=EvidenceTrack.UNINFORMED,
        seed=7,
        dataset_hash="locked",
        discovered=True,
        admitted=True,
        refused=False,
        baseline_improved=True,
        stable_across_seeds=True,
        evidence={"domain_limited": True, "validated_domain": "M >= 3"},
    )
    assert outcome.status == DiscoveryStatus.DOMAIN_LIMITED
    assert outcome.to_dict()["status"] == "DOMAIN_LIMITED"

# COMMAND ----------



def test_conditional_status_records_supported_structure_with_prediction_limit():
    outcome = BenchmarkOutcome(
        case_id="aftershock_omori",
        track=EvidenceTrack.UNINFORMED,
        seed=7,
        dataset_hash="locked",
        discovered=True,
        admitted=True,
        refused=False,
        baseline_improved=True,
        stable_across_seeds=True,
        evidence={"conditional": True, "holdout_prediction_interval": "missed"},
    )
    assert outcome.status == DiscoveryStatus.CONDITIONAL


def test_predictive_only_status_separates_prediction_from_mechanism():
    outcome = BenchmarkOutcome(
        case_id="climate_correlation",
        track=EvidenceTrack.UNINFORMED,
        seed=7,
        dataset_hash="locked",
        discovered=True,
        admitted=True,
        refused=False,
        baseline_improved=True,
        stable_across_seeds=True,
        evidence={"predictive_only": True, "causal_identification": False},
    )
    assert outcome.status == DiscoveryStatus.PREDICTIVE_ONLY


def test_hard_refusal_precedes_qualified_status():
    outcome = BenchmarkOutcome(
        case_id="hall_petch",
        track=EvidenceTrack.UNINFORMED,
        seed=7,
        dataset_hash="locked",
        discovered=True,
        admitted=False,
        refused=True,
        baseline_improved=True,
        stable_across_seeds=True,
        refusal_reason="physical admissibility failed",
        evidence={"conditional": True},
    )
    assert outcome.status == DiscoveryStatus.REFUSE

# COMMAND ----------

from apedp.cross_domain_benchmark import SensitivityVariant, run_sensitivity_analysis


def test_sensitivity_analysis_preserves_primary_and_marks_contamination_sensitivity():
    primary = {"structure": "shifted_power", "holdout_error": 18.0, "p": 0.989}
    analysis = run_sensitivity_analysis(
        primary,
        [SensitivityVariant(
            variant_id="remove_nias_mainshock",
            rationale="independently identified secondary-mainshock contamination",
            rule="remove documented Nias event window before refitting",
            outcome={"structure": "shifted_power", "holdout_error": 4.0, "p": 1.148},
        )],
        tolerance=0.1,
    )
    assert analysis.primary == primary
    assert analysis.robustness == "SENSITIVE"
    assert analysis.variants[0].rationale.startswith("independently")


def test_sensitivity_analysis_keeps_missing_measurements_unresolved():
    analysis = run_sensitivity_analysis(
        {"structure": "power", "holdout_error": 1.0},
        [SensitivityVariant("robust_fit", "documented estimator check", "Huber fit", {"structure": "power"})],
    )
    assert analysis.robustness == "UNRESOLVED"
