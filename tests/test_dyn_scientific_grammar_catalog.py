"""Unit tests for APEDP Scientific Grammar Space (Physics, Chemistry, Biology).

Verifies:
  1. Coverage and dimensional integrity of APEDP-PHYSICS, APEDP-CHEMISTRY, APEDP-BIOLOGY.
  2. Physics validity gates: dimensional homogeneity and total energy conservation.
  3. Chemistry validity gates: concentration non-negativity and stoichiometric mass balance.
  4. Biology validity gates: population bounds and mandatory biological context tags.
  5. Scientific operators: Michaelis-Menten kinetics, Hill activation/repression, derivatives/integrals, Lotka-Volterra.
  6. Machine-readable schema serialization and deterministic cryptographic fingerprinting.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from apedp_dyn.grammar.scientific_grammar_catalog import (
    BiologyFamily,
    BiologyValidityGate,
    ChemistryFamily,
    ChemistryValidityGate,
    PhysicsFamily,
    PhysicsValidityGate,
    ScientificDomain,
    ScientificGrammarCatalog,
    ScientificVariableKind,
    ScientificVariableSpec,
    ValueType,
    MASTER_SCIENTIFIC_CATALOG,
    op_delay,
    op_hill_activation,
    op_hill_repression,
    op_lotka_volterra_step,
    op_michaelis_menten,
    op_temporal_derivative,
    op_temporal_integral,
    op_threshold_activation,
)


def test_scientific_domain_packages_coverage():
    """Verify APEDP-PHYSICS, APEDP-CHEMISTRY, and APEDP-BIOLOGY packages are fully populated."""
    cat = ScientificGrammarCatalog()
    assert cat.total_variables >= 75

    phys_vars = cat.get_package_variables("APEDP-PHYSICS")
    chem_vars = cat.get_package_variables("APEDP-CHEMISTRY")
    bio_vars = cat.get_package_variables("APEDP-BIOLOGY")

    assert len(phys_vars) >= 35
    assert len(chem_vars) >= 20
    assert len(bio_vars) >= 20

    astro_vars = cat.get_package_variables("APEDP-ASTROPHYSICS")
    math_vars = cat.get_package_variables("APEDP-MATHEMATICS")
    assert len(astro_vars) >= 5
    assert len(math_vars) >= 3

    # Check key physics variables
    assert "VELOCITY" in phys_vars
    assert "FORCE" in phys_vars
    assert "KINETIC_ENERGY" in phys_vars
    assert "TEMPERATURE" in phys_vars
    assert "REYNOLDS_NUMBER" in phys_vars
    assert "INDUCTOR_CURRENT" in phys_vars
    assert "PLANCK_CONSTANT" in phys_vars
    assert "VORTICITY" in phys_vars

    # Check key chemistry variables
    assert "SPECIES_CONCENTRATION" in chem_vars
    assert "REACTION_RATE" in chem_vars
    assert "RATE_CONSTANT" in chem_vars
    assert "EQUILIBRIUM_CONSTANT" in chem_vars
    assert "PH" in chem_vars
    assert "ARRHENIUS_PRE_EXPONENTIAL" in chem_vars
    assert "HALF_LIFE" in chem_vars
    assert "ACID_DISSOCIATION_CONSTANT" in chem_vars

    # Check key biology variables
    assert "GENE_EXPRESSION" in bio_vars
    assert "MICHAELIS_CONSTANT" in bio_vars
    assert "CELL_COUNT" in bio_vars
    assert "POPULATION_SIZE" in bio_vars
    assert "CARRYING_CAPACITY" in bio_vars
    assert "SUSCEPTIBLE_POPULATION" in bio_vars
    assert "HILL_COEFFICIENT" in bio_vars
    assert "ATP_CONCENTRATION" in bio_vars

    # Check astrophysics & mathematics
    assert "GRAVITATIONAL_POTENTIAL" in astro_vars
    assert "HUBBLE_PARAMETER" in astro_vars
    assert "LYAPUNOV_EXPONENT" in math_vars
    assert "BIFURCATION_PARAMETER" in math_vars


def test_physics_si_dimensions_and_validity_gate():
    """Verify SI base dimensions on physics primitives and test dimensional homogeneity gate."""
    cat = ScientificGrammarCatalog()
    vel = cat.get_variable("VELOCITY")
    force = cat.get_variable("FORCE")
    ke = cat.get_variable("KINETIC_ENERGY")

    # SI dims: (M, L, T, Theta, N, I, J)
    # Velocity: L T^-1
    assert vel.dimensions == (0, 1, -1, 0, 0, 0, 0)
    # Force: M L T^-2
    assert force.dimensions == (1, 1, -2, 0, 0, 0, 0)
    # Kinetic energy: M L^2 T^-2
    assert ke.dimensions == (1, 2, -2, 0, 0, 0, 0)

    # 1. Dimensional Homogeneity Gate
    # Matching dimensions pass
    v1 = PhysicsValidityGate.check_dimensional_homogeneity(force.dimensions, (1, 1, -2, 0, 0, 0, 0))
    assert v1.passed is True

    # Mismatched dimensions fail (e.g. Force vs Energy)
    v2 = PhysicsValidityGate.check_dimensional_homogeneity(force.dimensions, ke.dimensions)
    assert v2.passed is False
    assert "Dimensional mismatch" in v2.reasons[0]

    # 2. Energy Conservation Gate
    t = np.linspace(0, 10, 100)
    ke_arr = np.sin(t) ** 2
    pe_arr = np.cos(t) ** 2  # Total energy = 1.0 (conserved)
    v_cons = PhysicsValidityGate.check_energy_conservation(ke_arr, pe_arr, tol=0.01)
    assert v_cons.passed is True

    # Dissipating/diverging energy fails
    pe_diverge = np.cos(t) ** 2 + 0.2 * t
    v_drift = PhysicsValidityGate.check_energy_conservation(ke_arr, pe_diverge, tol=0.05)
    assert v_drift.passed is False


def test_chemistry_validity_gates():
    """Verify concentration positivity and stoichiometric mass balance gates."""
    # 1. Positivity
    valid_c = np.array([0.0, 1.5, 3.2, 0.05])
    v_pos = ChemistryValidityGate.check_concentration_positivity(valid_c)
    assert v_pos.passed is True

    bad_c = np.array([1.0, -0.05, 2.0])
    v_neg = ChemistryValidityGate.check_concentration_positivity(bad_c)
    assert v_neg.passed is False
    assert "positivity constraint" in v_neg.reasons[0]

    # 2. Mass Balance: A + B -> C
    # Stoichiometry S = [-1, -1, +1]
    S = np.array([[-1.0], [-1.0], [1.0]])
    r = np.array([0.5])  # Reaction rate 0.5 mol/(m^3*s)
    expected_species_rates = np.array([-0.5, -0.5, 0.5])

    v_mb = ChemistryValidityGate.check_mass_balance(S, r, expected_species_rates)
    assert v_mb.passed is True

    # Unbalanced rates fail
    bad_rates = np.array([-0.5, -0.5, 0.2])  # Violates mass conservation
    v_bad_mb = ChemistryValidityGate.check_mass_balance(S, r, bad_rates)
    assert v_bad_mb.passed is False


def test_biology_validity_gates():
    """Verify biological context requirements and population bounds."""
    cat = ScientificGrammarCatalog()
    gene_var = cat.get_variable("GENE_EXPRESSION")

    # 1. Biological Context Gate
    valid_ctx = {"organism": "Homo sapiens", "cell_type": "T-cell", "assay": "RNA-seq"}
    v_ctx_ok = BiologyValidityGate.check_biological_context(gene_var, valid_ctx)
    assert v_ctx_ok.passed is True

    missing_ctx = {"organism": "Homo sapiens"}  # Missing cell_type and assay
    v_ctx_fail = BiologyValidityGate.check_biological_context(gene_var, missing_ctx)
    assert v_ctx_fail.passed is False
    assert "missing required biological context" in v_ctx_fail.reasons[0]

    # 2. Population Bounds Gate
    pop_normal = np.array([10.0, 50.0, 100.0, 95.0])
    v_pop = BiologyValidityGate.check_population_bounds(pop_normal, carrying_capacity=100.0)
    assert v_pop.passed is True

    pop_negative = np.array([10.0, -2.0, 50.0])
    v_pop_neg = BiologyValidityGate.check_population_bounds(pop_negative)
    assert v_pop_neg.passed is False

    pop_exploded = np.array([10.0, 50.0, 300.0])  # > 2x capacity 100
    v_pop_exp = BiologyValidityGate.check_population_bounds(pop_exploded, carrying_capacity=100.0)
    assert v_pop_exp.passed is False


def test_scientific_operator_library():
    """Verify Michaelis-Menten, Hill kinetics, calculus operators, and discrete updates."""
    # 1. Michaelis-Menten: v(Km) = Vmax / 2
    v_half = op_michaelis_menten(s=2.0, vmax=10.0, km=2.0)
    assert np.isclose(v_half, 5.0)

    # 2. Hill Activation: f(K) = 0.5
    h_act = op_hill_activation(x=3.0, k=3.0, n=2.0)
    assert np.isclose(h_act, 0.5)

    # Hill Repression: f(0) = 1.0, f(K) = 0.5
    h_rep_0 = op_hill_repression(x=0.0, k=3.0, n=2.0)
    h_rep_k = op_hill_repression(x=3.0, k=3.0, n=2.0)
    assert np.isclose(h_rep_0, 1.0)
    assert np.isclose(h_rep_k, 0.5)

    # 3. Numerical Derivative and Integral
    dt = 0.01
    t = np.arange(0, 1.0, dt)
    x = t ** 2  # dx/dt = 2*t
    dx = op_temporal_derivative(x, dt)
    assert np.isnan(dx[0])
    # At t=0.5 (index 50), dx/dt should be approx 2*0.5 = 1.0 (backward diff: (0.5^2 - 0.49^2)/0.01 = 0.99)
    assert np.isclose(dx[50], 0.99, atol=0.02)

    # Integral of constant 2 should be 2*t
    const_arr = np.full(len(t), 2.0)
    integral = op_temporal_integral(const_arr, dt)
    assert np.isclose(integral[-1], 2.0 * t[-1], atol=0.05)

    # 4. Lotka-Volterra step
    prey_1, pred_1 = op_lotka_volterra_step(prey=10.0, pred=5.0, alpha=0.1, beta=0.01, delta=0.01, gamma=0.1, dt=0.1)
    assert prey_1 > 0.0 and pred_1 > 0.0


def test_schema_json_serialization_and_fingerprint():
    """Verify JSON export and deterministic SHA-256 fingerprinting for scientific catalog."""
    cat = ScientificGrammarCatalog()
    json_str = cat.to_json()
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert len(parsed) == cat.total_variables

    fp1 = cat.compute_catalog_fingerprint()
    fp2 = cat.compute_catalog_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64
