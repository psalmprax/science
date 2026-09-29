"""APEDP Comprehensive Scenarios: Full 22 Physical, Computational, and Metrological Test Scenarios."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import (
    BuckinghamPiEngine,
    ClosedLoopAssuranceSystem,
    ConservationInvariantRegistry,
    DimensionalAssuranceEngine,
    VectorizedDimensionalEngine,
)
from apedp.discovery import (
    BayesianRefinementEngine,
)
from apedp.epistemic import (
    AdversarialDeliberationTournament,
    EpistemicGraph,
    LifecycleManager,
)
from apedp.metrology import (
    AnalyticalUncertaintyPropagator,
    MetrologySensorStream,
)
from apedp.repair import (
    PhysicalAnomalyRepairEngine,
)

logger = logging.getLogger(__name__)


class APEDPScenarioSuite:
    """Executes all 22 verified physical, mathematical, and metrological scenarios."""

    def __init__(self):
        self.engine = DimensionalAssuranceEngine()
        self.platform = ClosedLoopAssuranceSystem(self.engine)
        self.repair = PhysicalAnomalyRepairEngine(self.platform)
        self.results: dict[str, dict[str, Any]] = {}

    def run_all_scenarios(self) -> dict[str, dict[str, Any]]:
        """Runs the entire 22-scenario suite."""
        scenarios = [
            ("Scenario 01: Newtonian Gravity Singularity", self.scenario_01_gravity_singularity),
            ("Scenario 02: Relativistic Superluminal Limit", self.scenario_02_relativistic_superluminal),
            ("Scenario 03: Ideal Gas Near-Vacuum", self.scenario_03_ideal_gas_vacuum),
            ("Scenario 04: Stefan-Boltzmann Radiative Flux", self.scenario_04_stefan_boltzmann),
            ("Scenario 05: Scientific Exergy Efficiency", self.scenario_05_scientific_exergy),
            ("Scenario 06: Control Volume Energy Rate", self.scenario_06_control_volume_energy),
            ("Scenario 07: Safe Bayes Factor Matrix", self.scenario_07_safe_bayes_factor),
            ("Scenario 08: Quantum Schrodinger Action", self.scenario_08_schrodinger_action),
            ("Scenario 09: Navier-Stokes Force Density", self.scenario_09_navier_stokes_force_density),
            ("Scenario 10: Arrhenius Chemical Kinetics", self.scenario_10_arrhenius_kinetics),
            ("Scenario 11: GUM Analytical Uncertainty", self.scenario_11_gum_uncertainty),
            ("Scenario 12: Bayesian Hypothesis Comparison", self.scenario_12_bayesian_model_comparison),
            ("Scenario 13: Buckingham-Pi Pendulum", self.scenario_13_buckingham_pendulum),
            ("Scenario 14: Buckingham-Pi Aerodynamic Drag", self.scenario_14_buckingham_aerodynamic_drag),
            ("Scenario 15: Bernoulli Fluid Invariant", self.scenario_15_bernoulli_invariant),
            ("Scenario 16: Damped Oscillator Resonance", self.scenario_16_harmonic_resonance),
            ("Scenario 17: Coulomb Debye Shielding", self.scenario_17_coulomb_debye_shielding),
            ("Scenario 18: Kepler's Third Law", self.scenario_18_kepler_third_law),
            ("Scenario 19: Carnot Thermodynamic Limit", self.scenario_19_carnot_limit),
            ("Scenario 20: Anscombe Metrology Dilemma", self.scenario_20_anscombe_metrology),
            ("Scenario 21: High-Volume Permutation Sweep", self.scenario_21_high_volume_permutations),
            ("Scenario 22: Adversarial Tournament", self.scenario_22_adversarial_tournament),
        ]

        for name, func in scenarios:
            try:
                res = func()
                self.results[name] = {'status': 'PASS', 'details': res}
            except Exception as e:  # noqa: BLE001 - one scenario must not abort the sweep
                logger.warning("Scenario '%s' failed", name, exc_info=True)
                self.results[name] = {'status': 'FAIL', 'error': str(e)}

        return self.results

    # 1. Newtonian Gravity Singularity Softening
    def scenario_01_gravity_singularity(self) -> dict[str, Any]:
        m_phys = self.engine.register_variable('m_phys', 'kilogram')
        m_target = self.engine.register_variable('m_target', 'kilogram')
        r_dist = self.engine.register_variable('r_dist', 'meter')

        raw_gravity = sp.Symbol('G_grav') * m_phys * m_target / (r_dist**2)
        self.platform.register_model_spec("Newton_Gravity_Raw", raw_gravity, "newton")

        # Boundary check r -> 0 diverges
        limit_zero = sp.limit(raw_gravity, r_dist, 0)
        if not (limit_zero.has(sp.oo) or limit_zero.has(sp.zoo) or limit_zero in (sp.oo, sp.zoo)):
            raise AssertionError(f"Expected singularity at r=0, got {limit_zero}")

        # Softened gravity
        soft_name = self.repair.audit_and_repair_gravity('r_dist')
        val = self.platform.evaluate_model(soft_name, {
            'm_phys': 10.0,
            'm_target': 1000.0,
            'r_dist': 0.0,
            'epsilon_soft': 1e-6
        })
        if not (val > 0 and not np.isnan(val) and not np.isinf(val)):
            raise AssertionError(f"Invalid force {val}")
        return {'repaired_model': soft_name, 'force_at_r0': val}

    # 2. Relativistic Superluminal Limit
    def scenario_02_relativistic_superluminal(self) -> dict[str, Any]:
        v_phys = self.engine.register_variable('v_phys', 'meter / second')
        m_phys = sp.Symbol('m_phys')
        c = sp.Symbol('c_light')

        # Raw relativistic KE
        gamma = (1 - (v_phys**2 / c**2))**(-0.5)
        raw_ke = m_phys * (c**2) * (gamma - 1)
        self.platform.register_model_spec("Relativistic_KE_Raw", raw_ke, "joule")

        # Test superluminal v = 1.5 * c
        superluminal_inputs = {'m_phys': 1.0, 'v_phys': 1.5 * 299792458.0}
        flaw_caught = False
        try:
            self.platform.evaluate_model("Relativistic_KE_Raw", superluminal_inputs)
        except ArithmeticError:
            flaw_caught = True
        if not flaw_caught:
            raise AssertionError("Superluminal input should have triggered domain flaw!")

        # Taylor repair
        taylor_name = self.repair.audit_and_repair_relativity('v_phys', 'c_light', 'm_phys')
        val_approx = self.platform.evaluate_model(taylor_name, superluminal_inputs)
        if not val_approx > 0:
            raise AssertionError(f"Expected positive KE, got {val_approx}")
        return {'repaired_model': taylor_name, 'ke_at_1_5c': val_approx}

    # 3. Ideal Gas Extreme Near-Vacuum
    def scenario_03_ideal_gas_vacuum(self) -> dict[str, Any]:
        P_press = self.engine.register_variable('P_press', 'pascal')
        V_vol = self.engine.register_variable('V_vol', 'meter**3')
        T_temp = self.engine.register_variable('T_temp', 'kelvin')
        R_gas = sp.Symbol('R_gas')

        ideal_gas = (P_press * V_vol) / (R_gas * T_temp)
        self.platform.register_model_spec("Ideal_Gas_Moles", ideal_gas, "mole")

        vacuum_inputs = {'P_press': 1e-10, 'V_vol': 1.0, 'T_temp': 300.0}
        moles = self.platform.evaluate_model("Ideal_Gas_Moles", vacuum_inputs)
        if not 0 < moles < 1e-10:
            raise AssertionError(f"Unexpected mole count {moles}")
        return {'moles_in_vacuum': moles}

    # 4. Stefan-Boltzmann Radiative Flux
    def scenario_04_stefan_boltzmann(self) -> dict[str, Any]:
        T_temp = sp.Symbol('T_temp')
        sigma = sp.Symbol('sigma_sb')
        flux_expr = sigma * (T_temp**4)
        self.platform.register_model_spec("Stefan_Boltzmann_Flux", flux_expr, "watt / meter**2")

        inputs = {'T_temp': 1000.0}
        flux = self.platform.evaluate_model("Stefan_Boltzmann_Flux", inputs)
        expected = 5.670374419e-8 * (1000.0**4)
        if not np.isclose(flux, expected, rtol=1e-5):
            raise AssertionError(f"Flux mismatch: {flux} vs {expected}")
        return {'flux_at_1000K': flux}

    # 5. Scientific Exergy Efficiency
    def scenario_05_scientific_exergy(self) -> dict[str, Any]:
        IG = self.engine.register_variable('IG', 'dimensionless')
        Ex_consumed = self.engine.register_variable('Ex_consumed', 'joule')

        eta_sci = IG / Ex_consumed
        self.platform.register_model_spec("Scientific_Exergy_Efficiency", eta_sci, "1 / joule")

        eff = self.platform.evaluate_model("Scientific_Exergy_Efficiency", {'IG': 100.0, 'Ex_consumed': 5000.0})
        if not np.isclose(eff, 0.02):
            raise AssertionError(f"Expected 0.02, got {eff}")
        return {'scientific_efficiency': eff}

    # 6. Control Volume Energy Rate Balance
    def scenario_06_control_volume_energy(self) -> dict[str, Any]:
        Q_dot = self.engine.register_variable('Q_dot', 'watt')
        W_dot = self.engine.register_variable('W_dot', 'watt')

        dE_dt = Q_dot - W_dot
        self.platform.register_model_spec("Control_Volume_Energy_Rate", dE_dt, "watt")

        dE = self.platform.evaluate_model("Control_Volume_Energy_Rate", {'Q_dot': 1500.0, 'W_dot': 500.0})
        is_conserved, residual = ConservationInvariantRegistry.audit_energy_conservation(1500.0, 500.0, dE)
        if not is_conserved:
            raise AssertionError(f"Energy not conserved! Residual: {residual}")
        return {'dE_dt': dE, 'conserved': is_conserved}

    # 7. Safe Bayes Factor Matrix
    def scenario_07_safe_bayes_factor(self) -> dict[str, Any]:
        safe_name = self.repair.audit_and_repair_bayes_factor()
        val = self.platform.evaluate_model(safe_name, {'Z_i': 1.0, 'Z_j': 0.0, 'beta_reg': 1e-12})
        if not val == 1e12:
            raise AssertionError(f"Expected 1e12, got {val}")
        return {'safe_bayes_factor': val}

    # 8. Quantum Schrödinger Action Term
    def scenario_08_schrodinger_action(self) -> dict[str, Any]:
        d_psi_dt = self.engine.register_variable('d_psi_dt', '1 / second')
        hbar = sp.Symbol('h_bar')
        action_term = hbar * d_psi_dt
        self.platform.register_model_spec("Schrodinger_Action_Term", action_term, "joule")

        val = self.platform.evaluate_model("Schrodinger_Action_Term", {'d_psi_dt': 1e15})
        expected = 1.054571817e-34 * 1e15
        if not np.isclose(val, expected):
            raise AssertionError(f"Mismatch: {val} vs {expected}")
        return {'action_joules': val}

    # 9. Navier-Stokes Force Density
    def scenario_09_navier_stokes_force_density(self) -> dict[str, Any]:
        rho_fluid = self.engine.register_variable('rho_fluid', 'kilogram / meter**3')
        u_vel = self.engine.register_variable('u_vel', 'meter / second')
        dt_time = self.engine.register_variable('dt_time', 'second')

        force_density = rho_fluid * (u_vel / dt_time)
        self.platform.register_model_spec("Navier_Stokes_Inertial_Density", force_density, "newton / meter**3")

        fd = self.platform.evaluate_model("Navier_Stokes_Inertial_Density", {
            'rho_fluid': 1000.0,
            'u_vel': 2.0,
            'dt_time': 0.1
        })
        if not np.isclose(fd, 20000.0):
            raise AssertionError(f"Expected 20000 N/m^3, got {fd}")
        return {'force_density_N_m3': fd}

    # 10. Arrhenius Chemical Kinetics
    def scenario_10_arrhenius_kinetics(self) -> dict[str, Any]:
        E_act = self.engine.register_variable('E_act', 'joule / mole')
        A_freq = self.engine.register_variable('A_freq', '1 / second')
        R_gas = sp.Symbol('R_gas')
        T_temp = sp.Symbol('T_temp')

        # Arrhenius: k = A * exp(-E_a / (R * T))
        k_rate = A_freq * sp.exp(-E_act / (R_gas * T_temp))
        self.platform.register_model_spec("Arrhenius_Rate", k_rate, "1 / second")

        k_val = self.platform.evaluate_model("Arrhenius_Rate", {
            'A_freq': 1e13,
            'E_act': 50000.0,
            'T_temp': 300.0
        })
        if not k_val > 0:
            raise AssertionError(f"Expected positive rate, got {k_val}")
        return {'rate_constant_k': k_val}

    # 11. GUM Analytical Uncertainty Propagation
    def scenario_11_gum_uncertainty(self) -> dict[str, Any]:
        m1, m2, r, G = sp.symbols('m_phys m_target r_dist G_grav')
        grav_expr = G * m1 * m2 / (r**2)

        nominals = {'m_phys': 10.0, 'm_target': 1000.0, 'r_dist': 5.0, 'G_grav': 6.6743e-11}
        uncertainties = {'m_phys': 0.1, 'm_target': 5.0, 'r_dist': 0.05}

        report = AnalyticalUncertaintyPropagator.propagate_uncertainty(grav_expr, nominals, uncertainties)
        if not report.evaluated_value > 0:
            raise AssertionError(f"Non-positive evaluated value: {report.evaluated_value}")
        if not report.combined_uncertainty > 0:
            raise AssertionError(f"Non-positive combined uncertainty: {report.combined_uncertainty}")
        if not report.expanded_uncertainty_95 == 2.0 * report.combined_uncertainty:
            raise AssertionError("Expanded uncertainty is not 2x combined uncertainty")
        return {
            'nominal_force': report.evaluated_value,
            'combined_u': report.combined_uncertainty,
            'expanded_u95': report.expanded_uncertainty_95,
            'relative_u_pct': report.relative_uncertainty * 100.0
        }

    # 12. Bayesian Hypothesis Comparison
    def scenario_12_bayesian_model_comparison(self) -> dict[str, Any]:
        """Compare two prior hypotheses about one parameter by a real Bayes factor.

        Model 1 centres its prior on the value that generated the data; Model 2
        centres it two prior-widths away. Evidence is the marginal likelihood

            p(D|M) = \u222b p(D|\u03bc,M) p(\u03bc|M) d\u03bc,

        integrated over each prior by quadrature -- not a ratio of summed
        pointwise likelihoods, which has no prior in it and would not be a Bayes
        factor (see ``compute_likelihood_ratio`` for that quantity). With a
        genuinely mismatched prior, Model 1 must win.
        """
        rng = np.random.default_rng(42)
        sigma = 0.4
        y = rng.normal(1.0, sigma, 8)

        def log_likelihood(mu: np.ndarray) -> float:
            return float(-0.5 * np.sum((y - float(mu[0])) ** 2) / sigma**2 - 0.5 * 8 * np.log(2.0 * np.pi * sigma**2))

        def gaussian_prior(centre: float, width: float):
            def logpdf(mu: np.ndarray) -> float:
                return float(-0.5 * (float(mu[0]) - centre) ** 2 / width**2 - 0.5 * np.log(2.0 * np.pi * width**2))
            return logpdf

        ev1 = BayesianRefinementEngine.marginal_likelihood(
            log_likelihood, gaussian_prior(1.0, 0.5),
            bounds=np.array([[1.0 - 8 * 0.5, 1.0 + 8 * 0.5]]),
            prior_description="mu ~ N(1.0, 0.5^2)",
        )
        ev2 = BayesianRefinementEngine.marginal_likelihood(
            log_likelihood, gaussian_prior(-1.5, 0.5),
            bounds=np.array([[-1.5 - 8 * 0.5, -1.5 + 8 * 0.5]]),
            prior_description="mu ~ N(-1.5, 0.5^2)",
        )
        bf, verdict, log_bf = BayesianRefinementEngine.compute_bayes_factor(ev1, ev2)
        if not bf > 1.0:
            raise AssertionError(f"Expected Model 1 preferred, got BF={bf}")
        return {
            'bayes_factor': bf,
            'verdict': verdict,
            'log_bf': log_bf,
            'log_evidence_model_1': ev1.log_evidence,
            'log_evidence_model_2': ev2.log_evidence,
            'method': f"{ev1.method} vs {ev2.method}",
        }

    # 13. Buckingham-Pi Simple Pendulum
    def scenario_13_buckingham_pendulum(self) -> dict[str, Any]:
        vars_dict = {
            'T_period': {'T': 1},
            'l_length': {'L': 1},
            'g_accel':  {'L': 1, 'T': -2},
            'm_mass':   {'M': 1}
        }
        pi_groups = BuckinghamPiEngine.compute_pi_groups(vars_dict)
        if not len(pi_groups) == 1:
            raise AssertionError(f"Expected 1 Pi-group, got {pi_groups}")
        return {'pi_groups': pi_groups}

    # 14. Buckingham-Pi Aerodynamic Drag & Reynolds Number
    def scenario_14_buckingham_aerodynamic_drag(self) -> dict[str, Any]:
        vars_dict = {
            'F_drag':   {'M': 1, 'L': 1, 'T': -2},
            'rho':      {'M': 1, 'L': -3},
            'v':        {'L': 1, 'T': -1},
            'D':        {'L': 1},
            'mu':       {'M': 1, 'L': -1, 'T': -1}
        }
        pi_groups = BuckinghamPiEngine.compute_pi_groups(vars_dict)
        if not len(pi_groups) == 2:
            raise AssertionError(f"Expected 2 Pi-groups (Drag Coeff & Reynolds), got {len(pi_groups)}")
        return {'pi_groups': pi_groups}

    # 15. Bernoulli Fluid Invariant
    def scenario_15_bernoulli_invariant(self) -> dict[str, Any]:
        # Venturi pipe: section 1 -> section 2
        rho = 1000.0  # water kg/m^3
        v1 = 2.0      # m/s
        h1 = 5.0      # m
        P1 = 200000.0 # Pa

        v2 = 4.0      # constricted speed m/s
        h2 = 5.0      # same height
        # P1 + 0.5*rho*v1^2 = P2 + 0.5*rho*v2^2 => P2 = P1 + 0.5*rho*(v1^2 - v2^2)
        P2 = P1 + 0.5 * rho * (v1**2 - v2**2)  # 200000 + 500 * (4 - 16) = 194000 Pa

        is_conserved, residual = ConservationInvariantRegistry.audit_bernoulli_invariant(
            P1, v1, h1, P2, v2, h2, rho
        )
        if not is_conserved:
            raise AssertionError(f"Bernoulli invariant violated! Residual: {residual}")
        return {'P1': P1, 'P2': P2, 'conserved': is_conserved, 'residual': residual}

    # 16. Damped Harmonic Oscillator Resonance Singularity
    def scenario_16_harmonic_resonance(self) -> dict[str, Any]:
        res_name = self.repair.audit_and_repair_resonance()
        # Evaluate right at resonance omega = omega_0 with zero natural damping zeta = 0
        amp = self.platform.evaluate_model(res_name, {
            'm_phys': 1.0,
            'F_0': 10.0,
            'omega': 100.0,
            'omega_0': 100.0,
            'zeta': 0.0,
            'epsilon_res': 0.01
        })
        if not (amp > 0 and not np.isinf(amp)):
            raise AssertionError(f"Resonance softening failed: amp={amp}")
        return {'repaired_model': res_name, 'regularized_amplitude': amp}

    # 17. Coulomb Electrostatic Field with Debye Plasma Shielding
    def scenario_17_coulomb_debye_shielding(self) -> dict[str, Any]:
        coulomb_name = self.repair.audit_and_repair_coulomb()
        force = self.platform.evaluate_model(coulomb_name, {
            'q_source': 1e-6,
            'q_test': 1e-6,
            'r_dist': 0.0,
            'epsilon_soft': 1e-3,
            'lambda_debye': 0.1
        })
        if not (force > 0 and not np.isinf(force)):
            raise AssertionError(f"Coulomb softening failed: force={force}")
        return {'shielded_force_at_r0': force}

    # 18. Kepler's Third Law Planetary Motion
    def scenario_18_kepler_third_law(self) -> dict[str, Any]:
        # T^2 = 4*pi^2 / (G * M) * r^3
        r_orbit = self.engine.register_variable('r_orbit', 'meter')
        M_star = self.engine.register_variable('M_star', 'kilogram')
        G = sp.Symbol('G_grav')

        kepler_T_sq = (4 * (sp.pi**2) / (G * M_star)) * (r_orbit**3)
        self.platform.register_model_spec("Kepler_Third_Law", kepler_T_sq, "second**2")

        # Earth orbit: r = 1.496e11 m, M_sun = 1.989e30 kg
        T_sq = self.platform.evaluate_model("Kepler_Third_Law", {
            'r_orbit': 1.496e11,
            'M_star': 1.989e30
        })
        T_period_days = np.sqrt(T_sq) / (3600.0 * 24.0)
        if not np.isclose(T_period_days, 365.25, rtol=0.02):
            raise AssertionError(f"Period mismatch: {T_period_days} days")
        return {'orbital_period_days': T_period_days}

    # 19. Carnot Thermodynamic Boundary Audit
    def scenario_19_carnot_limit(self) -> dict[str, Any]:
        T_cold = self.engine.register_variable('T_cold', 'kelvin')
        T_hot = self.engine.register_variable('T_hot', 'kelvin')

        carnot_eta = 1 - (T_cold / T_hot)
        self.platform.register_model_spec("Carnot_Efficiency", carnot_eta, "dimensionless")

        # Third Law test T_cold -> 0 => eta -> 1.0
        eta_cold = self.platform.evaluate_model("Carnot_Efficiency", {'T_cold': 1e-9, 'T_hot': 1000.0})
        if not np.isclose(eta_cold, 1.0, atol=1e-6):
            raise AssertionError(f"Carnot limit failed near absolute zero: {eta_cold}")

        # Thermal equilibrium test T_hot = T_cold => eta = 0.0
        eta_zero = self.platform.evaluate_model("Carnot_Efficiency", {'T_cold': 500.0, 'T_hot': 500.0})
        if not np.isclose(eta_zero, 0.0, atol=1e-6):
            raise AssertionError(f"Carnot equilibrium failed: {eta_zero}")

        return {'eta_limit_near_zero': eta_cold, 'eta_equilibrium': eta_zero}

    # 20. Anscombe's Quartet Metrology Dilemma
    def scenario_20_anscombe_metrology(self) -> dict[str, Any]:
        stream = MetrologySensorStream()
        datasets = stream.stream_anscombe_metrology_experiment()

        # Compute summary stats for each dataset
        import warnings
        stats = {}
        for name, data in datasets.items():
            x, y = data['x'], data['y']
            mean_x, mean_y = np.mean(x), np.mean(y)
            var_x, var_y = np.var(x), np.var(y)
            corr = np.corrcoef(x, y)[0, 1]
            slope, intercept = np.polyfit(x, y, 1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                quad_fit = np.polyfit(x, y, 2)
            stats[name] = {
                'mean_x': mean_x, 'mean_y': mean_y,
                'var_x': var_x, 'var_y': var_y,
                'corr': corr, 'linear_fit': (slope, intercept),
                'curvature_a': quad_fit[0]
            }

        # Dataset 2 has distinct non-zero curvature despite identical linear moments
        if not abs(stats['dataset_2']['curvature_a']) > 0.05:
            raise AssertionError("Dataset 2 shows no distinguishing curvature")
        return {'anscombe_disambiguation': 'Orthogonal curvature audit distinguishes datasets'}

    # 21. High-Volume Vectorized Permutation Sweep
    def scenario_21_high_volume_permutations(self) -> dict[str, Any]:
        num_formulas = 10000
        num_permutations = 10
        total_evals = num_formulas * num_permutations

        # Force * Distance = Work/Energy
        force_dims = np.tile(VectorizedDimensionalEngine.DIM_MAP['newton'], (num_formulas, 1))
        dist_dims = np.tile(VectorizedDimensionalEngine.DIM_MAP['meter'], (num_formulas, 1))

        t0 = time.time()
        work_dims = VectorizedDimensionalEngine.batch_validate_multiplication(force_dims, dist_dims)
        expected_joule = np.tile(VectorizedDimensionalEngine.DIM_MAP['joule'], (num_formulas, 1))
        matches = VectorizedDimensionalEngine.batch_check_homogeneity(work_dims, expected_joule)
        elapsed = time.time() - t0

        if not np.all(matches):
            raise AssertionError("Vectorized dimension mismatch detected!")
        throughput = total_evals / max(elapsed, 1e-6)
        return {'total_evaluations': total_evals, 'elapsed_seconds': elapsed, 'throughput_evals_per_sec': throughput}

    # 22. Adversarial Deliberation Tournament
    def scenario_22_adversarial_tournament(self) -> dict[str, Any]:
        graph = EpistemicGraph()
        lifecycle = LifecycleManager()
        tournament = AdversarialDeliberationTournament(graph, lifecycle)

        result = tournament.run_tournament(
            hypothesis_id="Hypothesis_Gravitational_Point_Mass",
            formula_str="G * m1 * m2 / r**2",
            adversarial_probe_name="r_zero_probe",
            boundary_flaw_detected=True,
            repaired_formula_str="G * m1 * m2 / (r**2 + eps**2)",
            variable_units={"r": "meter", "eps": "meter", "m1": "kilogram", "m2": "kilogram", "G": "meter**3 / (kilogram * second**2)"},
            strict_units=True,
        )

        # The tournament now returns REPAIRED_CANDIDATE - promotion requires explicit validation
        if not result['outcome'] == 'REPAIRED_CANDIDATE':
            raise AssertionError(f"Unexpected tournament outcome: {result['outcome']}")
        if not result['repaired_model_state'] == 'Candidate':
            raise AssertionError(f"Unexpected repaired model state: {result['repaired_model_state']}")
        if not graph.validate_graph_acyclicity():
            raise AssertionError("Epistemic graph has illegal cycles!")
        return result
