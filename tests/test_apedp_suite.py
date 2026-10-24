"""Comprehensive verification test suite for APEDP (Autonomous Physics & Engineering Discovery Platform)."""
import math

import numpy as np
import pytest
import sympy as sp

from apedp.applied_defects import (
    AppliedChemistryDefectAuditor,
    AppliedMaterialScienceDefectAuditor,
    AppliedNanotechnologyDefectAuditor,
    AppliedPhysicsDefectAuditor,
    UniversalAppliedDefectEngine,
)
from apedp.core import (
    PHYSICAL_CONSTANTS,
    BuckinghamPiEngine,
    ClosedLoopAssuranceSystem,
    ConservationInvariantRegistry,
    DimensionalAssuranceEngine,
    VectorizedDimensionalEngine,
    ureg,
)
from apedp.discovery import (
    BayesianRefinementEngine,
    MultiVariableSRE,
)
from apedp.epistemic import (
    AdversarialDeliberationTournament,
    EpistemicGraph,
    LifecycleManager,
    RequirementsTraceabilityMatrix,
)
from apedp.metrology import (
    AnalyticalUncertaintyPropagator,
    CalibrationTraceabilityChain,
    MetrologySensorStream,
)
from apedp.qec import (
    ConcatenatedThresholdRecursion,
    KnillLaflammeAuditor,
    QECEquationDefectAuditor,
    QECMetrologyPropagator,
    QuantumCodeValidator,
    SurfaceCodeThresholdEngine,
)
from apedp.repair import (
    AgnosticClosedLoopAssuranceSystem,
    PhysicalAnomalyRepairEngine,
)
from apedp.scenarios import APEDPScenarioSuite


def test_fundamental_constants_registry():
    c = PHYSICAL_CONSTANTS.get_constant('c_light')
    assert c.value == 299792458.0
    assert str(c.unit.dimensionality) == '[length] / [time]'

    G = PHYSICAL_CONSTANTS.get_constant('G_grav')
    assert G.value == 6.67430e-11

    val_dict = PHYSICAL_CONSTANTS.values_dict()
    assert 'h_bar' in val_dict
    assert 'k_B' in val_dict
    assert 'sigma_sb' in val_dict


def test_dimensional_assurance_engine():
    engine = DimensionalAssuranceEngine()
    m = engine.register_variable('m', 'kilogram')
    a = engine.register_variable('a', 'meter / second**2')

    # F = m * a
    F_expr = m * a
    unit = engine.check_expression_consistency(F_expr)
    assert unit.dimensionality == engine.registry('newton').dimensionality

    # Illegal addition: Force + Mass
    with pytest.raises(TypeError):
        engine.check_expression_consistency(F_expr + m)

    # Transcendental argument must be dimensionless
    v = engine.register_variable('v', 'meter / second')
    with pytest.raises(TypeError):
        engine.check_expression_consistency(sp.exp(v))

    # Dimensionless exponent in power
    dimless = engine.register_variable('dimless', 'dimensionless')
    dimless_pow = (dimless + 1) ** dimless
    assert engine.check_expression_consistency(dimless_pow).dimensionality == engine.registry.dimensionless.dimensionality
    pow_expr = m ** 2
    assert engine.check_expression_consistency(pow_expr).dimensionality == engine.registry('kilogram**2').dimensionality


def test_closed_loop_model_evaluation():
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)

    m = engine.register_variable('m_phys', 'kilogram')
    v = engine.register_variable('v_phys', 'meter / second')

    # Kinetic energy E = 0.5 * m * v^2
    ke_expr = sp.Rational(1, 2) * m * (v**2)
    platform.register_model_spec("Kinetic_Energy", ke_expr, "joule")

    # Evaluate
    val = platform.evaluate_model("Kinetic_Energy", {'m_phys': 2.0, 'v_phys': 3.0})
    assert np.isclose(val, 9.0)

    # Test with Pint quantity
    val_qty = platform.evaluate_model("Kinetic_Energy", {
        'm_phys': 2000.0 * ureg.gram,
        'v_phys': 3.0 * ureg.meter / ureg.second
    })
    assert np.isclose(val_qty, 9.0)


def test_conservation_invariants():
    # First Law
    conserved, res = ConservationInvariantRegistry.audit_energy_conservation(100.0, 40.0, 60.0)
    assert conserved and res == 0.0

    not_conserved, res2 = ConservationInvariantRegistry.audit_energy_conservation(100.0, 40.0, 50.0)
    assert not not_conserved and res2 == 10.0

    # Bernoulli
    b_conserved, b_res = ConservationInvariantRegistry.audit_bernoulli_invariant(
        P1=101325.0, v1=1.0, h1=0.0,
        P2=101325.0 + 0.5 * 1000.0 * (1.0**2 - 2.0**2), v2=2.0, h2=0.0,
        rho=1000.0
    )
    assert b_conserved and b_res < 1e-6


def test_buckingham_pi_engine():
    # Simple pendulum: T (time), l (length), g (gravity), m (mass)
    pendulum = {
        'T': {'T': 1},
        'l': {'L': 1},
        'g': {'L': 1, 'T': -2},
        'm': {'M': 1}
    }
    pi_groups = BuckinghamPiEngine.compute_pi_groups(pendulum)
    assert len(pi_groups) == 1
    assert 'Pi_1' in pi_groups[0]

    # Empty inputs
    assert BuckinghamPiEngine.compute_pi_groups({}) == []


def test_vectorized_dimensional_engine():
    num_items = 50000
    force_dims = np.tile(VectorizedDimensionalEngine.DIM_MAP['newton'], (num_items, 1))
    dist_dims = np.tile(VectorizedDimensionalEngine.DIM_MAP['meter'], (num_items, 1))

    # Force * Dist = Joule
    work_dims = VectorizedDimensionalEngine.batch_validate_multiplication(force_dims, dist_dims)
    expected_joule = np.tile(VectorizedDimensionalEngine.DIM_MAP['joule'], (num_items, 1))
    assert np.all(VectorizedDimensionalEngine.batch_check_homogeneity(work_dims, expected_joule))

    # Invalid addition raises TypeError
    with pytest.raises(TypeError):
        VectorizedDimensionalEngine.batch_validate_addition(force_dims, dist_dims)


def test_gum_analytical_uncertainty_propagation():
    m, v = sp.symbols('m v')
    ke = 0.5 * m * (v**2)

    nominals = {'m': 2.0, 'v': 10.0}
    uncertainties = {'m': 0.05, 'v': 0.2}

    report = AnalyticalUncertaintyPropagator.propagate_uncertainty(ke, nominals, uncertainties)
    assert np.isclose(report.evaluated_value, 100.0)
    # dy/dm = 0.5 * v^2 = 50. term = (50 * 0.05)^2 = 2.5^2 = 6.25
    # dy/dv = m * v = 20. term = (20 * 0.2)^2 = 4^2 = 16
    # sum = 22.25 => sqrt(22.25) ~ 4.71699
    expected_u = np.sqrt(6.25 + 16.0)
    assert np.isclose(report.combined_uncertainty, expected_u, rtol=1e-4)
    assert np.isclose(report.expanded_uncertainty_95, 2.0 * expected_u, rtol=1e-4)
    assert 'm' in report.percentage_budget
    assert 'v' in report.percentage_budget


def test_metrology_sensor_stream_and_anscombe():
    stream = MetrologySensorStream(seed=123)
    gravity_data = stream.stream_gravity_experiment_data(num_samples=30)
    assert len(gravity_data['m_phys']) == 30
    assert np.all(gravity_data['force_primary'] > 0)
    assert np.all(gravity_data['force_orthogonal'] > 0)

    # Anscombe
    anscombe = stream.stream_anscombe_metrology_experiment()
    assert len(anscombe) == 4
    for key in ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']:
        assert key in anscombe
        assert len(anscombe[key]['x']) == 11


def test_calibration_traceability():
    chain = CalibrationTraceabilityChain()
    rec_id = chain.log_calibration(
        sensor_id="SENSOR-PRESSURE-001",
        standard_id="NIST-PTB-BAROMETER-STD",
        procedure_name="ISO-17025-STATIC-CAL",
        drift_compensation_factor=1.0002,
        uncertainty_budget=0.05,
        timestamp="2026-09-16T12:00:00Z"
    )
    assert rec_id.startswith("CAL-SENSOR-PRESSURE-001")
    assert chain.verify_traceability("SENSOR-PRESSURE-001")
    assert not chain.verify_traceability("UNKNOWN-SENSOR")


def test_bayesian_refinement_engine():
    """A Bayes factor comes from marginal likelihoods; a likelihood ratio is labelled as one."""
    m1_likes = np.array([0.9, 0.95, 0.92])
    m2_likes = np.array([0.1, 0.12, 0.08])

    ratio, description, _log_ratio = BayesianRefinementEngine.compute_likelihood_ratio(m1_likes, m2_likes)
    assert ratio > 3.0
    assert "Model 1" in description
    assert "NOT a Bayes factor" in description

    bf, verdict, _log_bf = BayesianRefinementEngine.compute_bayes_factor_from_log_evidence(-12.0, -20.0)
    assert bf > 3.0
    assert "Model 1" in verdict

    bic = BayesianRefinementEngine.compute_bic(log_likelihood=-50.0, num_params=3, num_samples=100)
    aic = BayesianRefinementEngine.compute_aic(log_likelihood=-50.0, num_params=3)
    assert bic > aic


def test_multivariable_sre():
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('m_phys', 'kilogram')
    engine.register_variable('v_phys', 'meter / second')

    sre = MultiVariableSRE(
        assurance_system=platform,
        independent_vars=['m_phys', 'v_phys'],
        target_unit_str='joule'
    )

    # Valid candidate check
    cand = sp.Symbol('m_phys') * (sp.Symbol('v_phys')**2)
    x_data = {'m_phys': np.array([1.0, 2.0, 3.0]), 'v_phys': np.array([2.0, 3.0, 4.0])}
    y_data = np.array([4.0, 18.0, 48.0])
    fitness = sre.fitness(cand, x_data, y_data)
    assert fitness < 1.0  # Perfect fit (MSE = 0 + small complexity penalty)

    # Dimensionally invalid candidate gives inf
    bad_cand = sp.Symbol('m_phys') + sp.Symbol('v_phys')
    assert sre.fitness(bad_cand, x_data, y_data) == float('inf')


def test_anomaly_repair_and_agnostic():
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    repair = PhysicalAnomalyRepairEngine(platform)

    # Test gravity repair
    engine.register_variable('m_phys', 'kilogram')
    engine.register_variable('m_target', 'kilogram')
    engine.register_variable('r_dist', 'meter')
    model_name = repair.audit_and_repair_gravity('r_dist')
    val = platform.evaluate_model(model_name, {'m_phys': 5.0, 'm_target': 100.0, 'r_dist': 0.0, 'epsilon_soft': 1e-5})
    assert val > 0 and not np.isinf(val)

    # Test Agnostic Pipeline
    agnostic = AgnosticClosedLoopAssuranceSystem()
    res = agnostic.autonomous_stress_test_and_repair("G * m_phys * m_target / (r_dist**2)")
    assert res['status'] == 'REPAIRED'
    assert len(res['anomalies']) > 0

    stable_res = agnostic.autonomous_stress_test_and_repair("0.5 * m_phys * (v_phys**2)")
    assert stable_res['status'] == 'APPROVED'


def test_epistemic_graph_and_lifecycle():
    graph = EpistemicGraph()
    graph.add_node("Hyp_1", "Hypothesis")
    graph.add_node("Axiom_Euclid", "Hypothesis")
    graph.add_node("Claim_1", "Claim")

    graph.add_edge("Axiom_Euclid", "Hyp_1", "ASSUMED_BY")
    graph.add_edge("Hyp_1", "Claim_1", "DERIVED_FROM")

    roots = graph.get_root_assumptions("Claim_1")
    assert "Axiom_Euclid" in roots
    assert graph.validate_graph_acyclicity()

    # Contradiction edge
    graph.add_node("Evidence_Disproof", "Evidence")
    graph.add_edge("Evidence_Disproof", "Hyp_1", "CONTRADICTED_BY")
    contradictions = graph.find_contradictions()
    assert len(contradictions) == 1
    assert contradictions[0][0] == "Evidence_Disproof"
    assert contradictions[0][1] == "Hyp_1"

    # Lifecycle manager
    mgr = LifecycleManager()
    mgr.register_model("Model_A", "Draft")
    mgr.promote("Model_A", "Candidate", checks_passed=True)
    mgr.promote("Model_A", "Validated", checks_passed=True)
    mgr.promote("Model_A", "Operational", checks_passed=True)

    # Illegal transition: Candidate to Operational directly
    mgr.register_model("Model_B", "Candidate")
    with pytest.raises(ValueError):
        mgr.promote("Model_B", "Operational", checks_passed=True)

    # Failed check blocks promotion
    with pytest.raises(ValueError):
        mgr.promote("Model_B", "Validated", checks_passed=False)


def test_requirements_traceability_matrix():
    rtm = RequirementsTraceabilityMatrix()
    rtm.register_requirement(
        req_id="REQ-DIM-001",
        goal="Enforce dimensional consistency at compile time",
        owner="Physics Assurance Lead",
        design_adr="ADR-001-Pint-SymPy-Ontology",
        impl_artifact="apedp/core.py:DimensionalAssuranceEngine",
        acceptance_test="tests/test_apedp_suite.py:test_dimensional_assurance_engine",
        validation_evidence="EV-TEST-RUN-SUCCESS"
    )
    is_complete, missing = rtm.verify_completeness()
    assert is_complete
    assert len(missing) == 0


def test_all_22_scenarios_suite():
    suite = APEDPScenarioSuite()
    results = suite.run_all_scenarios()
    assert len(results) == 22, f"Expected 22 scenarios, got {len(results)}"
    for name, res in results.items():
        assert res['status'] == 'PASS', f"Scenario failed: {name} with details: {res.get('error')}"


def test_legacy_modules_syntax_and_imports():
    """Confirms that the previously broken prototype files now run and import cleanly."""
    from apedp.ingestion import list_targets
    from tests.audit_suite import build_demo_platform

    # Run demo platform build
    engine, platform, _expr, _unit = build_demo_platform()
    assert engine is not None
    assert platform is not None

    # Check academic targets
    targets = list_targets()
    assert 'arXiv' in targets
    assert len(targets) >= 20


def test_gum_covariance_matrix_and_correlations():
    """Tests GUM Section 5.2 full covariance matrix propagation and correlated cancellation."""
    m = sp.Symbol('m')
    v = sp.Symbol('v')
    # Kinetic energy E = 0.5 * m * v^2
    ke_expr = sp.Rational(1, 2) * m * (v**2)

    param_values = {'m': 2.0, 'v': 10.0}
    # c_m = dE/dm = 0.5 * 100 = 50.0
    # c_v = dE/dv = 2 * 10 = 20.0
    # Base variances: u_m^2 = 0.05^2 = 0.0025 -> (50^2)*0.0025 = 6.25
    #                 u_v^2 = 0.2^2 = 0.04 -> (20^2)*0.04 = 16.0
    # Uncorrelated combined variance = 22.25 -> u_c = sqrt(22.25) ~ 4.71699

    # 1. Uncorrelated baseline via covariance matrix diagonal
    cov_uncorrelated = np.array([
        [0.0025, 0.0],
        [0.0,    0.04]
    ])
    rep_uncorr = AnalyticalUncertaintyPropagator.propagate_uncertainty(
        ke_expr,
        param_values=param_values,
        covariance_matrix=cov_uncorrelated,
        variable_order=['m', 'v']
    )
    assert np.isclose(rep_uncorr.combined_uncertainty, np.sqrt(22.25))

    # 2. Positive covariance: +0.005 -> 2 * c_m * c_v * 0.005 = 2 * 50 * 20 * 0.005 = 10.0
    # Total variance = 22.25 + 10.0 = 32.25 -> u_c = sqrt(32.25) ~ 5.6789
    cov_positive = np.array([
        [0.0025, 0.005],
        [0.005,  0.04]
    ])
    rep_pos = AnalyticalUncertaintyPropagator.propagate_uncertainty(
        ke_expr,
        param_values=param_values,
        covariance_matrix=cov_positive,
        variable_order=['m', 'v']
    )
    assert np.isclose(rep_pos.combined_uncertainty, np.sqrt(32.25))
    assert ('m', 'v') in rep_pos.covariance_budget
    assert np.isclose(rep_pos.covariance_budget[('m', 'v')], 10.0)

    # 3. Negative covariance (variance reduction)
    cov_negative = np.array([
        [0.0025, -0.005],
        [-0.005,  0.04]
    ])
    rep_neg = AnalyticalUncertaintyPropagator.propagate_uncertainty(
        ke_expr,
        param_values=param_values,
        covariance_matrix=cov_negative,
        variable_order=['m', 'v']
    )
    assert np.isclose(rep_neg.combined_uncertainty, np.sqrt(12.25))


def test_multivariable_sre_continuous_parameter_optimization():
    """Tests that SRE calibrates continuous parameters and snaps to rational fractions."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    m = engine.register_variable('m', 'kilogram')
    v = engine.register_variable('v', 'meter / second')

    sre = MultiVariableSRE(
        assurance_system=platform,
        variables=[m, v],
        target_unit_str='joule'
    )

    # Synthetic data generated from y = 0.5 * m * v^2
    np.random.seed(42)
    m_data = np.random.uniform(1.0, 5.0, 20)
    v_data = np.random.uniform(5.0, 20.0, 20)
    y_data = 0.5 * m_data * (v_data ** 2)

    x_data = {'m': m_data, 'v': v_data}
    candidate_unscaled = m * (v ** 2)

    calibrated_expr, mse = sre.optimize_parameters(candidate_unscaled, x_data, y_data)
    assert mse < 1e-12, f"Expected near-zero MSE after calibration, got {mse}"

    # Check rational snapping: coeff should be exact 1/2
    assert sp.Rational(1, 2) in calibrated_expr.args or calibrated_expr == sp.Rational(1, 2) * candidate_unscaled


def test_sre_ephemeral_constants_term_vector_fit():
    """Term-vector fit recovers additive offsets: C0 + x**2 on y = 2x^2 + 3."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('x', 'dimensionless')
    sre = MultiVariableSRE(platform, independent_vars=['x'])

    x = np.linspace(1.0, 5.0, 25)
    y = 2.0 * x ** 2 + 3.0
    calibrated, mse = sre.optimize_parameters(sp.Symbol('C0') + sp.Symbol('x') ** 2, {'x': x}, y)
    assert mse == 0.0, f"Expected exact fit, got mse={mse}"
    assert calibrated == 2 * sp.Symbol('x') ** 2 + 3, f"Unexpected expression: {calibrated}"

    # Bare ephemeral constant discovers a pure-constant law
    calibrated_const, mse_const = sre.optimize_parameters(sp.Symbol('C0'), {'x': x}, np.full_like(x, 5.0))
    assert mse_const < 1e-24, f"Expected exact constant fit, got mse={mse_const}"
    assert abs(float(calibrated_const) - 5.0) < 1e-9, f"Unexpected expression: {calibrated_const}"


def test_sre_discovers_additive_offset_end_to_end():
    """Seeded end-to-end search recovers 2*x**2 + 3 essentially exactly.

    Score is penalized MSE (exact recovery scores ~7e-4, the complexity
    penalty alone; best linear fit scores ~6.6), so the threshold is robust
    to cosmetic expression differences.
    """
    engine = DimensionalAssuranceEngine()
    engine.register_variable('x', 'dimensionless')
    platform = ClosedLoopAssuranceSystem(engine)
    sre = MultiVariableSRE(platform, independent_vars=['x'])

    x = np.linspace(1.0, 5.0, 25)
    y = 2.0 * x ** 2 + 3.0
    expr, score, _ = sre.run_discovery({'x': x}, y, generations=15, pop_size=30, seed=3)
    assert score < 0.01, f"Expected exact recovery, got {expr} (score={score})"


def test_tokenized_and_configurable_agnostic_unit_inference():
    """Tests token-based multi-domain variable unit inference and dynamic registration."""
    agnostic = AgnosticClosedLoopAssuranceSystem()

    # Compound names with prefixes/modifiers
    expr = sp.parse_expr("total_mass_kg * avg_velocity_v + memory_byte_count * price_usd")
    inferred = agnostic.infer_units_for_expression(expr)

    assert inferred[sp.Symbol('total_mass_kg')].dimensionality == agnostic.ureg('kilogram').dimensionality
    assert inferred[sp.Symbol('avg_velocity_v')].dimensionality == agnostic.ureg('meter / second').dimensionality
    assert inferred[sp.Symbol('memory_byte_count')].dimensionality == agnostic.ureg('byte').dimensionality
    assert inferred[sp.Symbol('price_usd')].dimensionality == agnostic.ureg('dollar').dimensionality

    # Dynamic domain registration
    agnostic.register_unit_mapping('qubit', 'dimensionless')
    q_expr = sp.parse_expr("n_qubit_count * 2")
    q_inferred = agnostic.infer_units_for_expression(q_expr)
    assert q_inferred[sp.Symbol('n_qubit_count')].dimensionality == agnostic.ureg.dimensionless.dimensionality


def test_extended_anomaly_repair_logs_and_roots():
    """Tests detection and regularization of logarithmic and square-root boundary singularities."""
    agnostic = AgnosticClosedLoopAssuranceSystem()

    # 1. Logarithmic boundary anomaly
    res_log = agnostic.autonomous_stress_test_and_repair("log(x_dist)", variable_units={"x_dist": "dimensionless"})
    assert res_log['status'] == 'REPAIRED'
    assert any("Logarithmic" in err for _, err in res_log['anomalies'])
    # Check that repaired formula has no log(0) blowup
    repaired_log = res_log['repaired_expression']
    subbed = float(repaired_log.subs({sp.Symbol('x_dist'): 0.0, sp.Symbol('eps_x_dist'): 1e-4}))
    assert not np.isnan(subbed) and not np.isinf(subbed)

    # 2. Square-root branch-cut anomaly
    res_sqrt = agnostic.autonomous_stress_test_and_repair("sqrt(h_height)", variable_units={"h_height": "dimensionless"})
    assert res_sqrt['status'] == 'REPAIRED'
    assert any("Square-root" in err for _, err in res_sqrt['anomalies'])
    repaired_sqrt = res_sqrt['repaired_expression']
    # Evaluate at negative input; should be real and non-negative
    subbed_sqrt = float(repaired_sqrt.subs({sp.Symbol('h_height'): -10.0, sp.Symbol('eps_h_height'): 1e-4}))
    assert subbed_sqrt >= 0.0


def test_configurable_dimension_basis_vectorized_engine():
    """Tests arbitrary basis dimensions across Economics and Computer Science domains."""
    # 1. Economics Engine
    econ = VectorizedDimensionalEngine.create_economics_engine()
    dollar_vec = econ.get_unit_vector('dollar')
    second_vec = econ.get_unit_vector('second')
    item_vec = econ.get_unit_vector('item')
    price_vec = econ.get_unit_vector('price')
    rev_rate_vec = econ.get_unit_vector('revenue_rate')

    # price = dollar / item
    assert np.array_equal(dollar_vec - item_vec, price_vec)
    # revenue_rate = dollar / second
    assert np.array_equal(dollar_vec - second_vec, rev_rate_vec)

    # 2. Computer Science Engine
    cs = VectorizedDimensionalEngine.create_computer_science_engine()
    bit_vec = cs.get_unit_vector('bit')
    sec_vec = cs.get_unit_vector('second')
    bps_vec = cs.get_unit_vector('bps')
    # bps = bit / second
    assert np.array_equal(bit_vec - sec_vec, bps_vec)

    # 3. Buckingham-Pi with custom Economics dimensions
    econ_vars = {
        'revenue_rate': {'Currency': 1, 'Time': -1},
        'price':        {'Currency': 1, 'Quantity': -1},
        'sales_rate':   {'Quantity': 1, 'Time': -1}
    }
    pi_groups = BuckinghamPiEngine.compute_pi_groups(econ_vars)
    assert len(pi_groups) == 1
    # revenue_rate / (price * sales_rate) is dimensionless
    assert "revenue_rate" in pi_groups[0]


def test_dimensional_assurance_extended_functions():
    """Tests Abs, Min, Max, sign, Heaviside, atan2, and transcendental checks in DimensionalAssuranceEngine."""
    engine = DimensionalAssuranceEngine()
    x = engine.register_variable('x', 'meter')
    y = engine.register_variable('y', 'meter')
    t = engine.register_variable('t', 'second')

    # Abs preserves unit
    assert engine.check_expression_consistency(sp.Abs(x)).dimensionality == engine.registry('meter').dimensionality

    # Min/Max preserve unit when matching
    assert engine.check_expression_consistency(sp.Min(x, y)).dimensionality == engine.registry('meter').dimensionality
    assert engine.check_expression_consistency(sp.Max(x, y)).dimensionality == engine.registry('meter').dimensionality

    # Min with mismatch raises TypeError
    with pytest.raises(TypeError):
        engine.check_expression_consistency(sp.Min(x, t))

    # sign and Heaviside accept dimensioned inputs and return dimensionless
    assert engine.check_expression_consistency(sp.sign(x)).dimensionality == engine.registry.dimensionless.dimensionality
    assert engine.check_expression_consistency(sp.Heaviside(t)).dimensionality == engine.registry.dimensionless.dimensionality

    # atan2 requires matching units
    assert engine.check_expression_consistency(sp.atan2(y, x)).dimensionality == engine.registry.dimensionless.dimensionality
    with pytest.raises(TypeError):
        engine.check_expression_consistency(sp.atan2(y, t))

    # Transcendental functions reject dimensioned arguments
    with pytest.raises(TypeError):
        engine.check_expression_consistency(sp.exp(x))


def test_buckingham_pi_integer_powers():
    """Tests Buckingham-Pi integer power reduction (coprime powers, denominator clearing)."""
    # Simple pendulum: T (time), L (length), g (accel: length / time^2)
    variables = {
        'T': {'T': 1},
        'L': {'L': 1},
        'g': {'L': 1, 'T': -2}
    }
    pi_groups = BuckinghamPiEngine.compute_pi_groups(variables)
    assert len(pi_groups) == 1
    group_str = pi_groups[0]
    # Should be integer powers: (T^2 * g) / L or (L) / (T^2 * g)
    assert "^" in group_str or "/" in group_str
    # Must NOT have fractional powers like 1/2 or 0.5 in string
    assert "1/2" not in group_str
    assert "0.5" not in group_str


def test_vectorized_engine_1d_and_batch_extensions():
    """Tests 1D arrays, shape validation, subtraction, and power in VectorizedDimensionalEngine."""
    engine = VectorizedDimensionalEngine.create_si_engine()

    # Shape validation
    with pytest.raises(ValueError):
        engine.register_unit('bad_unit', np.array([1, 0, 0]))  # Requires 7 elements for SI

    # 1D array operations
    m1 = engine.get_unit_vector('meter')
    m2 = engine.get_unit_vector('meter')
    s1 = engine.get_unit_vector('second')

    # 1D addition / subtraction
    res_add = engine.batch_validate_addition(m1, m2)
    assert np.array_equal(res_add, m1)
    res_sub = engine.batch_validate_subtraction(m1, m2)
    assert np.array_equal(res_sub, m1)

    with pytest.raises(TypeError):
        engine.batch_validate_addition(m1, s1)

    # Power
    res_pow = engine.batch_validate_power(m1, 3)
    assert np.array_equal(res_pow, np.array([0, 3, 0, 0, 0, 0, 0]))

    # 1D homogeneity check
    assert engine.batch_check_homogeneity(m1, m2)
    assert not engine.batch_check_homogeneity(m1, s1)


def test_closed_loop_missing_variable_and_singularity_handling():
    """Tests missing free variable validation and arithmetic singularity in ClosedLoopAssuranceSystem."""
    engine = DimensionalAssuranceEngine()
    system = ClosedLoopAssuranceSystem(engine)
    m1 = engine.register_variable('m1', 'kilogram')
    m2 = engine.register_variable('m2', 'kilogram')
    r = engine.register_variable('r', 'meter')

    system.register_model_spec("Gravity", sp.Symbol('G_grav') * m1 * m2 / (r**2), "newton")

    # Missing variable raises ValueError
    with pytest.raises(ValueError, match="Missing required variable"):
        system.evaluate_model("Gravity", {'m1': 5.0, 'm2': 10.0})  # Missing 'r'

    # Division by zero raises ArithmeticError
    with pytest.raises(ArithmeticError):
        system.evaluate_model("Gravity", {'m1': 5.0, 'm2': 10.0, 'r': 0.0})

    # Accepts np.number
    res = system.evaluate_model("Gravity", {'m1': np.float64(5.0), 'm2': 10.0, 'r': 2.0})
    assert res > 0.0


def test_epistemic_adversarial_tournament_retirement():
    """Tests that contradicted hypothesis is formally retired in tournament to avoid zombie science."""
    graph = EpistemicGraph()
    lifecycle = LifecycleManager()
    tournament = AdversarialDeliberationTournament(graph, lifecycle)

    result = tournament.run_tournament(
        hypothesis_id="Newton_Gravity_Hyp",
        formula_str="G_grav * m1 * m2 / r**2",
        adversarial_probe_name="r_zero_limit",
        boundary_flaw_detected=True,
        repaired_formula_str="G_grav * m1 * m2 / (r**2 + eps**2)",
        variable_units={"m1": "kilogram", "m2": "kilogram", "r": "meter", "eps": "meter", "G_grav": "meter**3 / (kilogram * second**2)"},
        strict_units=True,
    )

    # The tournament now returns REPAIRED_CANDIDATE - promotion requires explicit validation
    assert result['outcome'] == 'REPAIRED_CANDIDATE'
    # The original hypothesis MUST be Retired
    assert result['hypothesis_state'] == 'Retired'
    assert lifecycle.get_state("Newton_Gravity_Hyp") == 'Retired'
    # The repaired model is a Candidate (not auto-promoted to Validated)
    assert result['repaired_model_state'] == 'Candidate'


def test_requirements_traceability_matrix_empty_guard():
    """Tests that empty traceability matrix fails completeness check."""
    rtm = RequirementsTraceabilityMatrix()
    complete, missing = rtm.verify_completeness()
    assert not complete
    assert "No requirements registered" in missing[0]


def test_multivariable_sre_physical_constants_substitution():
    """Tests that SRE candidate evaluation resolves physical constants like G_grav without NameError."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    m = engine.register_variable('m_phys', 'kilogram')
    r = engine.register_variable('r_dist', 'meter')

    sre = MultiVariableSRE(
        assurance_system=platform,
        variables=[m, r],
        target_unit_str='meter / second**2'
    )

    # Candidate with G_grav
    cand = sp.Symbol('G_grav') * m / (r**2)
    x_data = {
        'm_phys': np.array([1e24, 2e24]),
        'r_dist': np.array([6e6, 7e6])
    }
    y_data = (6.67430e-11 * x_data['m_phys']) / (x_data['r_dist'] ** 2)

    # Should evaluate and calibrate without NameError
    _calibrated_expr, mse = sre.optimize_parameters(cand, x_data, y_data)
    assert mse < 1e-12


def test_calibration_revocation_and_retrieval():
    """Tests CalibrationTraceabilityChain revocation and record retrieval."""
    chain = CalibrationTraceabilityChain()
    rec_id = chain.log_calibration(
        sensor_id="TEMP-PROBE-01",
        standard_id="ITS-90 Standard Platinum Thermometer",
        procedure_name="ISO-17025-TEMP-CAL",
        drift_compensation_factor=1.0001,
        uncertainty_budget=0.005,
        timestamp="2026-09-17T00:00:00Z"
    )

    # Retrieve record
    retrieved = chain.get_calibration_record(rec_id)
    assert retrieved is not None
    assert retrieved['sensor_id'] == "TEMP-PROBE-01"

    # Revoke record
    success = chain.revoke_calibration(rec_id, "Secondary probe drift detected")
    assert success
    rec_after = chain.get_calibration_record(rec_id)
    assert rec_after is not None
    assert rec_after['status'] == "REVOKED"
    assert "Secondary probe drift" in rec_after['revocation_reason']

    # Check that verification now reflects revocation
    assert not chain.verify_traceability("TEMP-PROBE-01")


def test_qec_quantum_coding_bounds():
    """Tests Quantum Singleton and Hamming bounds on [[5,1,3]], [[7,1,3]], and invalid [[4,1,3]] codes."""
    # 1. [[5, 1, 3]] Perfect code (saturates both Singleton and Hamming bounds)
    res_513 = QuantumCodeValidator.validate_code_parameters("Five_Qubit_Code", 5, 1, 3)
    assert res_513['physically_admissible']
    assert res_513['singleton_bound']['is_qmds_code']
    assert res_513['hamming_bound']['is_perfect_code']

    # 2. [[7, 1, 3]] Steane CSS code
    res_713 = QuantumCodeValidator.validate_code_parameters("Steane_Code", 7, 1, 3)
    assert res_713['physically_admissible']
    assert res_713['singleton_bound']['satisfied']
    assert res_713['hamming_bound']['satisfied']

    # 3. [[4, 1, 3]] Unphysical code (violates Quantum Singleton bound: 4 - 1 = 3 < 2*(3 - 1) = 4)
    res_413 = QuantumCodeValidator.validate_code_parameters("Invalid_Code", 4, 1, 3)
    assert not res_413['singleton_bound']['satisfied']
    assert not res_413['physically_admissible']


def test_qec_knill_laflamme_conditions():
    """Tests Knill-Laflamme algebraic condition verification on 3-qubit bit-flip code."""
    # 3-qubit bit-flip codespace: |000> and |111> in 8D Hilbert space
    P = np.zeros((8, 8), dtype=complex)
    P[0, 0] = 1.0  # |000><000|
    P[7, 7] = 1.0  # |111><111|

    # Pauli matrices
    I_2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Correctable error set: {I, X_1, X_2, X_3}
    E_I = np.eye(8, dtype=complex)
    E_X1 = np.kron(X, np.kron(I_2, I_2))
    E_X2 = np.kron(I_2, np.kron(X, I_2))
    E_X3 = np.kron(I_2, np.kron(I_2, X))

    correctable_errors = [E_I, E_X1, E_X2, E_X3]
    res_kl = KnillLaflammeAuditor.verify_knill_laflamme(P, correctable_errors)
    assert res_kl['is_correctable']
    assert res_kl['is_hermitian_alpha']
    assert res_kl['codespace_dimension'] == 2

    # Uncorrectable error set: {I, Z_1} (bit-flip code cannot distinguish phase flip Z_1)
    E_Z1 = np.kron(Z, np.kron(I_2, I_2))
    uncorrectable_errors = [E_I, E_Z1]
    res_uncorr = KnillLaflammeAuditor.verify_knill_laflamme(P, uncorrectable_errors)
    assert not res_uncorr['is_correctable']


def test_qec_surface_code_threshold_and_boundary_regularization():
    """Tests raw threshold divergence audit and physical depolarizing regularizer in surface code."""
    engine = SurfaceCodeThresholdEngine()
    audit_res = engine.register_and_audit_surface_code(d_val=3, C_prefactor=0.1, p_th_nominal=0.01)

    # 1. Raw model flaw: at p = 5 * p_th, raw formula gives p_L = 2.5 > 0.5 (unphysical)
    assert audit_res['raw_flaw_detected']
    assert audit_res['raw_model_evaluation'] > 1.0

    # 2. Regularized physical model: guaranteed <= 0.5 ceiling
    assert audit_res['regularized_within_physical_bounds']
    assert 0.0 <= audit_res['regularized_evaluation'] <= 0.5

    # 3. Near-zero threshold stress p_th -> 0: softened model evaluates without division by zero
    assert not np.isnan(audit_res['zero_threshold_evaluation'])
    assert 0.0 <= audit_res['zero_threshold_evaluation'] <= 0.5


def test_qec_cross_talk_metrology_propagation():
    """Tests GUM Section 5.2 propagation of correlated gate cross-talk to logical error budget."""
    gate_values = {
        'p_1q': 1e-4,
        'p_2q': 1e-3,
        'p_meas': 3e-3,
        'p_idle': 5e-5
    }
    gate_uncertainties = {
        'p_1q': 1e-5,
        'p_2q': 1e-4,
        'p_meas': 3e-4,
        'p_idle': 1e-5
    }

    # 1. Uncorrelated baseline
    res_uncorr = QECMetrologyPropagator.propagate_qec_error_budget(
        gate_values,
        gate_uncertainties,
        code_distance=3
    )
    u_uncorr = res_uncorr['combined_uncertainty']
    assert u_uncorr > 0.0

    # 2. Correlated cross-talk between two-qubit gates and measurements
    cross_talk = {
        ('p_2q', 'p_meas'): 2e-8  # Positive cross-talk covariance
    }
    res_corr = QECMetrologyPropagator.propagate_qec_error_budget(
        gate_values,
        gate_uncertainties,
        cross_talk_covariances=cross_talk,
        code_distance=3
    )
    u_corr = res_corr['combined_uncertainty']
    # Correlated cross-talk increases combined metrological uncertainty
    assert u_corr > u_uncorr
    assert ('p_2q', 'p_meas') in res_corr['cross_talk_budget']


def test_qec_concatenated_threshold_recursion():
    """Tests AGP concatenated code pseudo-threshold recursion convergence and divergence."""
    # c = 100 -> pseudo_threshold = 1/c = 0.01
    c_factor = 100.0

    # Sub-threshold: p_0 = 0.005 < 0.01 -> error suppression to 0
    res_sub = ConcatenatedThresholdRecursion.analyze_threshold_convergence(c_factor, p_initial=0.005, max_levels=4)
    assert res_sub['below_pseudo_threshold']
    assert res_sub['error_suppression_achieved']
    assert res_sub['trajectory'][-1] < 1e-6

    # Super-threshold: p_0 = 0.02 > 0.01 -> error amplification towards ceiling
    res_super = ConcatenatedThresholdRecursion.analyze_threshold_convergence(c_factor, p_initial=0.02, max_levels=4)
    assert not res_super['below_pseudo_threshold']
    assert not res_super['error_suppression_achieved']
    assert res_super['trajectory'][-1] >= 0.02


def test_qec_equation_defect_auditor():
    """Tests systematic detection of QEC scaling equation defects: probability ceiling, decoding backlog, thermal quench, cross-talk blindspots, and magic state distillation overhead."""
    # 1. Unphysical Probability Defect
    p_phys = np.array([0.005, 0.01, 0.02, 0.05])
    # Raw formula predicts 0.1 * (p / 0.01)^2 -> at 0.05 gives 2.5
    raw_p_L = 0.1 * ((p_phys / 0.01) ** 2)
    audit_prob = QECEquationDefectAuditor.audit_unphysical_probability_defect(p_phys, raw_p_L)
    assert audit_prob['defect_detected']
    assert any("UNPHYSICAL_PROBABILITY" in d for d in audit_prob['defects'])

    # 2. Decoding Backlog Instability Defect (syndrome rate 1 MHz, decoding throughput 500 kHz)
    audit_backlog = QECEquationDefectAuditor.audit_decoding_backlog_instability(
        syndrome_generation_rate_hz=1e6,
        decoding_throughput_hz=5e5,
        run_duration_seconds=0.1
    )
    assert audit_backlog['defect_detected']
    assert audit_backlog['accumulated_syndromes_after_duration'] == 5e4

    # 3. Cryogenic Thermal Quench Defect (10,000 qubits at 10 nW = 100 uW > 20 uW base capacity)
    audit_thermal = QECEquationDefectAuditor.audit_cryogenic_thermal_quench(
        num_physical_qubits=10000,
        heat_dissipation_per_channel_watts=1e-8,
        fridge_cooling_power_at_base_watts=20e-6
    )
    assert audit_thermal['defect_detected']
    assert audit_thermal['overload_ratio'] == 5.0

    # 4. Correlated Noise Blindspot Defect
    audit_noise = QECEquationDefectAuditor.audit_correlated_noise_blindspot(
        uncorrelated_uncertainty=1e-4,
        cross_talk_covariance=2e-8,
        sensitivity_1=2.0,
        sensitivity_2=3.0
    )
    assert audit_noise['defect_detected']
    assert audit_noise['uncertainty_inflation_factor'] > 1.5

    # 5. Magic State Distillation Overhead Defect
    audit_magic = QECEquationDefectAuditor.audit_magic_state_distillation_overhead(
        target_logical_qubits=1000,
        target_t_gate_count=10000000
    )
    assert audit_magic['factory_overhead_percentage'] == 90.0
    assert audit_magic['total_physical_qubits_required'] == 14580000


def test_applied_physics_defect_auditor():
    """Tests Prandtl-Glauert transonic compressibility blowup and Knudsen continuum breakdown."""
    # 1. Transonic Singularity
    res_aero = AppliedPhysicsDefectAuditor.audit_prandtl_glauert_compressibility(mach_number=1.0)
    assert res_aero['defect_detected']
    assert any("TRANSONIC_SINGULARITY" in d for d in res_aero['defects'])
    # Regularized Cp must be finite and bounded
    assert not math.isnan(res_aero['regularized_cp']) and not math.isinf(res_aero['regularized_cp'])

    # 2. Knudsen Continuum Breakdown (Kn = 68nm / 50nm = 1.36 > 0.1)
    res_knudsen = AppliedPhysicsDefectAuditor.audit_knudsen_continuum_breakdown(
        mean_free_path_m=68e-9, characteristic_length_m=50e-9
    )
    assert res_knudsen['defect_detected']
    assert res_knudsen['knudsen_number'] > 0.1
    assert any("CONTINUUM_BREAKDOWN" in d for d in res_knudsen['defects'])


def test_applied_chemistry_defect_auditor():
    """Tests Butler-Volmer mass-transport diffusion saturation and Semenov thermal runaway."""
    # 1. Butler-Volmer Mass Transport Defect (high overpotential eta = 0.8 V)
    res_bv = AppliedChemistryDefectAuditor.audit_butler_volmer_electrochemical_saturation(
        overpotential_volts=0.8,
        exchange_current_density_a_m2=10.0,
        diffusion_limiting_current_a_m2=500.0
    )
    assert res_bv['defect_detected']
    assert any("MASS_TRANSPORT_VIOLATION" in d for d in res_bv['defects'])
    # Regularized current must be strictly capped below diffusion limit
    assert res_bv['regularized_current_density_a_m2'] <= 500.0

    # 2. Semenov Thermal Runaway Defect
    res_runaway = AppliedChemistryDefectAuditor.audit_semenov_thermal_runaway(
        heat_of_reaction_joules=-100000.0,
        activation_energy_joules_per_mole=80000.0,
        reactor_temp_kelvin=380.0,
        coolant_temp_kelvin=300.0,
        heat_transfer_coefficient_ua=10.0,
        reaction_rate_mol_per_sec=0.1
    )
    assert res_runaway['defect_detected']
    assert any("SEMENOV_THERMAL_RUNAWAY" in d for d in res_runaway['defects'])


def test_applied_material_science_defect_auditor():
    """Tests Inverse Hall-Petch nanoscale softening breakdown and Griffith crack tip singularity."""
    # 1. Inverse Hall-Petch Breakdown (d = 5 nm < 10 nm cutoff)
    res_hp = AppliedMaterialScienceDefectAuditor.audit_hall_petch_nanoscale_breakdown(
        grain_size_meters=5e-9,
        theoretical_shear_strength_pa=3e9
    )
    assert res_hp['defect_detected']
    assert any("INVERSE_HALL_PETCH_BREAKDOWN" in d for d in res_hp['defects'])
    # Yield stress must not exceed theoretical lattice limit
    assert res_hp['regularized_yield_stress_pa'] <= 3e9

    # 2. Griffith Crack Tip Singularity (r = 100 nm from crack tip)
    res_crack = AppliedMaterialScienceDefectAuditor.audit_griffith_crack_tip_singularity(
        distance_from_crack_tip_meters=1e-7,
        stress_intensity_factor_k1=30e6,
        yield_strength_pa=500e6
    )
    assert res_crack['defect_detected']
    assert any("CRACK_TIP_SINGULARITY" in d for d in res_crack['defects'])
    # Regularized cohesive stress must be capped at yield strength
    assert res_crack['regularized_stress_pa'] == 500e6


def test_applied_nanotechnology_defect_auditor():
    """Tests Gibbs-Thomson negative melting temperature violation and Casimir stiction divergence."""
    # 1. Gibbs-Thomson Negative Temperature Violation (r = 0.5 nm nanoparticle)
    res_gt = AppliedNanotechnologyDefectAuditor.audit_gibbs_thomson_melting_depression(
        nanoparticle_radius_meters=0.1e-9,
        bulk_melting_temp_kelvin=1337.0
    )
    assert res_gt['defect_detected']
    assert any("UNPHYSICAL_NEGATIVE_TEMPERATURE" in d for d in res_gt['defects'])
    # Regularized temperature must be strictly positive (Third Law of Thermodynamics)
    assert res_gt['regularized_melting_temp_k'] > 0.0

    # 2. Casimir NEMS Stiction Divergence (z = 0.1 nm sub-nanometer gap)
    res_casimir = AppliedNanotechnologyDefectAuditor.audit_casimir_nems_stiction_divergence(
        separation_distance_meters=1e-10
    )
    assert res_casimir['defect_detected']
    assert any("CASIMIR_STICTION_DIVERGENCE" in d for d in res_casimir['defects'])
    assert not math.isinf(res_casimir['regularized_force_n'])


def test_sre_exp_dimensionless_guard():
    """exp() rejects dimensioned arguments but accepts dimensionless ratios."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('T_temp', 'kelvin')
    engine.register_variable('E_act', 'joule / mole')
    sre = MultiVariableSRE(
        platform, independent_vars=['T_temp'], target_unit_str='dimensionless', allow_exp=True
    )

    x = np.linspace(0.1, 1.0, 10)
    y = np.exp(x)
    # Raw temperature inside exp() is dimensionally illegal -> inf fitness
    assert sre.fitness(sp.exp(sp.Symbol('T_temp')), {'T_temp': x}, y) == float('inf')

    # The Arrhenius group E/(R*T) is dimensionless and passes the gatekeeper
    ratio = sp.Symbol('E_act') / (sp.Symbol('R_gas') * sp.Symbol('T_temp'))
    assert engine.check_expression_consistency(ratio).dimensionality == engine.registry.dimensionless.dimensionality


def test_sre_inner_fit_learns_inner_scale():
    """Hill-climbing learns the inner 2.0 in 3*exp(2*x) from a C0=1.0 start."""
    import random as _random

    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('x', 'dimensionless')
    sre = MultiVariableSRE(platform, independent_vars=['x'], inner_fit=True, inner_steps=60)

    rng = np.random.RandomState(0)
    x = rng.uniform(0.1, 2.0, 40)
    y = 3.0 * np.exp(2.0 * x)

    _random.seed(0)
    np.random.seed(0)
    calibrated, mse = sre.optimize_parameters(sp.exp(sp.Symbol('C0') * sp.Symbol('x')), {'x': x}, y)

    assert mse / np.mean(y ** 2) < 1e-3
    exp_nodes = list(calibrated.atoms(sp.exp))
    assert len(exp_nodes) == 1
    scale = float(sp.expand(exp_nodes[0].args[0]).coeff(sp.Symbol('x')))
    assert abs(scale - 2.0) / 2.0 < 0.05


def test_sre_exp_template_prior_exact_recovery():
    """Physics-informed priors let discovery recover exp(x)-exp(-x) exactly."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('eta_norm', 'dimensionless')
    sre = MultiVariableSRE(
        platform,
        independent_vars=['eta_norm'],
        allow_exp=True,
        template_priors=['exp(eta_norm)-exp(-eta_norm)', 'not valid python ((('],
    )

    rng = np.random.RandomState(0)
    x = rng.uniform(0.1, 1.0, 40)
    y = np.exp(x) - np.exp(-x)
    best, score, _ = sre.run_discovery({'eta_norm': x}, y, generations=10, pop_size=40, seed=42)

    assert best is not None and np.isfinite(score)
    consts = {sp.Symbol(k): v for k, v in platform.constants_values.items()}
    f = sp.lambdify(sp.Symbol('eta_norm'), best.subs(consts), modules=['numpy'])
    xh = np.array([0.2, 0.5, 0.8])
    yh = np.asarray(f(xh), dtype=float)
    ye = np.exp(xh) - np.exp(-xh)
    r2 = 1.0 - float(np.sum((yh - ye) ** 2) / np.sum((ye - ye.mean()) ** 2))
    assert r2 > 0.999999


def test_sre_extra_const_values_faraday():
    """Custom constants (e.g. Faraday) register with units and join the search."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('eta_over', 'volt')
    engine.register_variable('T_temp', 'kelvin')
    sre = MultiVariableSRE(
        platform,
        independent_vars=['eta_over', 'T_temp'],
        extra_const_values={'F_far': (96485.33212, 'coulomb / mole')},
    )

    assert sp.Symbol('F_far') in sre.building_blocks
    assert platform.constants_values['F_far'] == 96485.33212
    # F*eta/(R*T) is the dimensionless Butler-Volmer group
    group = sp.Symbol('F_far') * sp.Symbol('eta_over') / (sp.Symbol('R_gas') * sp.Symbol('T_temp'))
    assert engine.check_expression_consistency(group).dimensionality == engine.registry.dimensionless.dimensionality

    # Unparseable units raise a clear error
    with pytest.raises(ValueError):
        MultiVariableSRE(
            platform,
            independent_vars=['eta_over'],
            extra_const_values={'BAD': (1.0, 'not_a_unit_xyz')},
        )
def test_agnostic_repair_declared_units():
    """Caller-supplied units keep real laws out of REJECTED status."""
    agnostic = AgnosticClosedLoopAssuranceSystem()

    # Coulomb with true units: singularity found and repaired, never rejected
    res = agnostic.autonomous_stress_test_and_repair(
        "q1*q2/(4*pi*eps_0*r_dist**2)",
        {"q1": "coulomb", "q2": "coulomb", "r_dist": "meter", "eps_0": "farad / meter"},
    )
    assert res["status"] == "REPAIRED"
    assert any("Division" in err for _, err in res["anomalies"])
    # Repaired form stays finite at r = 0 with a small epsilon
    rep = res["repaired_expression"]
    subs = {sp.Symbol("q1"): 1e-6, sp.Symbol("q2"): 1e-6, sp.Symbol("r_dist"): 0.0}
    for s in rep.free_symbols:
        if str(s).startswith("eps_") and s not in subs:
            subs[s] = 1e-6
    assert np.isfinite(float(complex(rep.subs(subs)).real))

    # First-law rate with true units: no anomaly, approved as-is
    res2 = agnostic.autonomous_stress_test_and_repair("Q_dot-W_dot", {"Q_dot": "watt", "W_dot": "watt"})
    assert res2["status"] == "APPROVED"


def test_sre_guided_finds_product_law():
    """Guided seeding recovers V=I*R exactly (unguided search misses it)."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('I_cur', 'ampere')
    engine.register_variable('R_res', 'ohm')
    sre = MultiVariableSRE(platform, independent_vars=['I_cur', 'R_res'], target_unit_str='volt')

    rng = np.random.RandomState(0)
    xtr = {'I_cur': rng.uniform(0.1, 5, 40), 'R_res': rng.uniform(1, 100, 40)}
    ytr = xtr['I_cur'] * xtr['R_res']
    best, score, _ = sre.run_discovery(xtr, ytr, generations=15, pop_size=50, seed=42)

    assert best is not None and np.isfinite(score)
    assert "I_cur" in str(best) and "R_res" in str(best)
    f = sp.lambdify([sp.Symbol('I_cur'), sp.Symbol('R_res')], best, modules=['numpy'])
    xh = {'I_cur': np.array([0.5, 2.0]), 'R_res': np.array([10.0, 47.0])}
    yh = np.asarray(f(xh['I_cur'], xh['R_res']), dtype=float)
    ye = xh['I_cur'] * xh['R_res']
    r2 = 1.0 - float(np.sum((yh - ye) ** 2) / np.sum((ye - ye.mean()) ** 2))
    assert r2 > 0.999


def test_sre_legacy_unguided_path_intact():
    """guided=False preserves a working legacy search path."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    engine.register_variable('x', 'dimensionless')
    sre = MultiVariableSRE(platform, independent_vars=['x'], guided=False)

    x = np.linspace(1.0, 5.0, 20)
    best, score, _ = sre.run_discovery({'x': x}, x ** 2, generations=10, pop_size=30, seed=42)
    assert best is not None and np.isfinite(score)


def test_universal_applied_defect_engine_cross_domain():
    """Tests UniversalAppliedDefectEngine cross-domain routing across physics, chemistry, materials, and nano."""
    engine = UniversalAppliedDefectEngine()

    # Physics query
    phys_res = engine.audit_applied_equation("Aerodynamics", "Prandtl_Glauert", {'mach_number': 1.0})
    assert phys_res['defect_detected']

    # Chemistry query
    chem_res = engine.audit_applied_equation("Electrochemistry", "Butler_Volmer", {'overpotential_volts': 1.0})
    assert chem_res['defect_detected']

    # Material Science query
    mat_res = engine.audit_applied_equation("Materials", "Hall_Petch", {'grain_size_meters': 3e-9})
    assert mat_res['defect_detected']

    # Nano query
    nano_res = engine.audit_applied_equation("Nanotechnology", "Casimir_Stiction", {'separation_distance_meters': 5e-11})
    nano_res_detected = nano_res['defect_detected']
    assert nano_res_detected


def test_metrology_missing_symbol_and_covariance_shape():
    """Verify AnalyticalUncertaintyPropagator validates missing symbols and covariance dimensions."""
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    expr = x + y
    # Missing symbol y raises ValueError
    try:
        AnalyticalUncertaintyPropagator.propagate_uncertainty(expr, {'x': 1.0}, {'x': 0.1})
        assert False, "Should raise ValueError for missing symbol"
    except ValueError as e:
        assert "Missing parameter values" in str(e)

    # 1D covariance array raises ValueError
    try:
        AnalyticalUncertaintyPropagator.propagate_uncertainty(
            expr, {'x': 1.0, 'y': 2.0}, covariance_matrix=np.array([0.1, 0.2])
        )
        assert False, "Should raise ValueError for 1D covariance matrix"
    except ValueError as e:
        assert "Covariance matrix must be 2D square" in str(e)


def test_repair_linear_denominator_pole_removal():
    """Verify denominator regularization removes poles for 1/r everywhere on R."""
    agnostic = AgnosticClosedLoopAssuranceSystem()
    res = agnostic.autonomous_stress_test_and_repair("1 / r")
    assert res['status'] == 'REPAIRED'
    repaired = res['repaired_expression']
    # For any real r (including negative r), denominator sqrt(r^2 + eps^2) is never 0
    val_at_0 = float(repaired.subs({sp.Symbol('r'): 0.0, sp.Symbol('eps_r'): 1e-3}))
    assert np.isfinite(val_at_0) and val_at_0 > 0
    val_at_neg = float(repaired.subs({sp.Symbol('r'): -1e-3, sp.Symbol('eps_r'): 1e-3}))
    assert np.isfinite(val_at_neg) and val_at_neg > 0


def test_repair_logarithm_dimensioned_coordinate():
    """Verify audit_and_repair_logarithm registers reference scale when coordinate is dimensioned."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    repair = PhysicalAnomalyRepairEngine(platform)
    model_name = repair.audit_and_repair_logarithm('r_coord', 'meter')
    spec = platform.registered_models[model_name]
    assert spec['expected_unit'].dimensionality == engine.registry('dimensionless').dimensionality
    assert 'r_coord_0' in [str(s) for s in spec['expression'].free_symbols]


def test_symbolic_regression_broadcast_and_cap():
    """Verify SymbolicRegressor evaluates scalar outputs properly and fits within attempt caps."""
    import warnings

    from apedp.symbolic_regression import RegressionConfig, SymbolicRegressor
    engine = DimensionalAssuranceEngine()
    engine.register_variable('x', 'dimensionless')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reg = SymbolicRegressor(engine, ['x'], 'dimensionless', config=RegressionConfig(population_size=10, generations=2))
    X = {'x': np.linspace(0, 1, 10)}
    y = np.ones(10)
    best_expr, hist = reg.fit(X, y)
    assert best_expr is not None
    assert len(hist) == 2


def test_serializer_numpy_bool_handling():
    """Verify serializer converts np.bool_ to JSON boolean true/false, not string."""
    from apedp.persistence import to_json
    data = {"flag_true": np.bool_(True), "flag_false": np.bool_(False)}
    json_str = to_json(data)
    assert '"flag_true": true' in json_str
    assert '"flag_false": false' in json_str





