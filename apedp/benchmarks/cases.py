"""The 25-case and 100-case pathological benchmark case tables.

Fixtures only: the class in :mod:`apedp.benchmarks.pathological` holds the
behaviour (suite selection, running, scoring); this module holds the case
literals, so a reader sees the whole suite without scrolling past the harness.
The tables are byte-identical to the ones the class used to build inline.
"""
from __future__ import annotations

from .results import (
    PathologicalCase,
)
from .types import (
    BenchmarkCategory,
    NoveltyTier,
)


def _get_25_case_suite() -> list[PathologicalCase]:
    return [
    # =========================================================================
    # PART 1: 13 SOLVABLE / PHYSICALLY REGULARIZABLE CASES
    # =========================================================================
    PathologicalCase(
    case_id="AERO-01",
    name="Prandtl-Glauert Compressibility Singularity",
    domain="Aerodynamics",
    formula_str="-0.5 / sqrt(1 - M**2)",
    independent_vars=["M"],
    variable_domains={"M": (0.0, 1.05)},
    critical_boundaries={"M": [1.0]},
    expected_repairable=True,
    failure_mode_description="Transonic singularity as M -> 1.",
    regularized_candidate_factory=lambda eps: f"-0.5 / (((1 - M**2)**2 + {eps}**2)**0.25)"
    ),
    PathologicalCase(
    case_id="MAT-01",
    name="Hall-Petch Nanoscale Dislocation Breakdown",
    domain="Materials Science",
    formula_str="70e6 + 0.74e6 / sqrt(d)",
    independent_vars=["d"],
    variable_domains={"d": (1e-10, 1e-4)},
    critical_boundaries={"d": [0.0]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Infinite yield stress as grain size d -> 0.",
    regularized_candidate_factory=lambda dc: f"70e6 + 0.74e6 * sqrt(d) * d / (d**2 + ({dc} * 1e-7)**2)"
    ),
    PathologicalCase(
    case_id="NANO-01",
    name="Casimir NEMS Stiction Divergence",
    domain="Nanotechnology",
    formula_str="1e-19 * 1e-8 / (6 * 3.1415926535 * z**3)",
    independent_vars=["z"],
    variable_domains={"z": (1e-11, 1e-6)},
    critical_boundaries={"z": [0.0]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Continuum 1/z^3 attraction diverges to infinity at contact.",
    regularized_candidate_factory=lambda z0: f"(1e-19 * 1e-8 / (6 * 3.1415926535 * (z + {z0})**3)) * (1 - ({z0} / (z + {z0}))**6)"
    ),
    PathologicalCase(
    case_id="QEC-01",
    name="Surface Code Super-Threshold Logical Error Blowup",
    domain="Quantum Computing",
    formula_str="0.1 * (p / p_th)**2",
    independent_vars=["p", "p_th"],
    variable_domains={"p": (0.001, 0.08), "p_th": (0.001, 0.02)},
    critical_boundaries={"p_th": [0.0]},
    invariants=["probability_bound", "positivity"],
    expected_repairable=True,
    failure_mode_description="Exceeds absolute probability 1.0 and depolarizing ceiling 0.5 when p > p_th.",
    regularized_candidate_factory=lambda eps: f"0.5 * (1 - exp(-0.2 * (p / (p_th + {eps}))**2))"
    ),
    PathologicalCase(
    case_id="EM-01",
    name="Coulomb Point Charge Potential Singularity",
    domain="Electromagnetism",
    formula_str="1.6e-19 / (1.11e-10 * r)",
    independent_vars=["r"],
    variable_domains={"r": (1e-15, 1e-8)},
    critical_boundaries={"r": [0.0]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Potential diverges to infinity as r -> 0 for point charges.",
    regularized_candidate_factory=lambda r0: f"1.6e-19 / (1.11e-10 * sqrt(r**2 + {r0}**2))"
    ),
    PathologicalCase(
    case_id="GRAV-01",
    name="Newtonian Gravitational Force Point Singularity",
    domain="Astrophysics",
    formula_str="6.674e-11 * 5.97e24 * 1.0 / (r**2)",
    independent_vars=["r"],
    variable_domains={"r": (1e-3, 1e7)},
    critical_boundaries={"r": [0.0]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Inverse-square gravitational force blows up at r=0.",
    regularized_candidate_factory=lambda r0: f"6.674e-11 * 5.97e24 * 1.0 / (r**2 + {r0}**2)"
    ),
    PathologicalCase(
    case_id="REL-01",
    name="Relativistic Kinetic Energy Velocity Singular Limit",
    domain="Relativistic Mechanics",
    formula_str="9.1e-31 * (3e8)**2 * (1 / sqrt(1 - (v / 3e8)**2) - 1)",
    independent_vars=["v"],
    variable_domains={"v": (0.0, 3e8)},
    critical_boundaries={"v": [3e8]},
    invariants=["lorentz_causality", "positivity"],
    expected_repairable=True,
    failure_mode_description="Diverges to infinity as velocity approaches c.",
    regularized_candidate_factory=lambda eps: f"9.1e-31 * (3e8)**2 * (1 / (((1 - (v / 3e8)**2)**2 + {eps}**2)**0.25) - 1)"
    ),
    PathologicalCase(
    case_id="CHEM-01",
    name="Butler-Volmer Exponential Current Density Blowup",
    domain="Electrochemistry",
    formula_str="10.0 * (exp(19.4 * eta) - exp(-19.4 * eta))",
    independent_vars=["eta"],
    variable_domains={"eta": (0.0, 1.2)},
    invariants=["current_density_ceiling", "positivity"],
    expected_repairable=True,
    failure_mode_description="Ignores Fickian mass-transport diffusion limiting current.",
    regularized_candidate_factory=lambda jlim: "(10.0 * (exp(19.4 * eta) - exp(-19.4 * eta)) * 1000.0) / (1000.0 + abs(10.0 * (exp(19.4 * eta) - exp(-19.4 * eta))))"
    ),
    PathologicalCase(
    case_id="THERM-01",
    name="Gibbs-Thomson Nanoparticle Negative Melting Point",
    domain="Thermodynamics",
    formula_str="1337.0 * (1.0 - 4.37e-10 / r)",
    independent_vars=["r"],
    variable_domains={"r": (1e-10, 1e-7)},
    critical_boundaries={"r": [0.0, 4.37e-10]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Predicts unphysical negative absolute melting point below 0.44 nm.",
    regularized_candidate_factory=lambda r0: f"1337.0 * exp(-4.37e-10 / (r + {r0}))"
    ),
    PathologicalCase(
    case_id="FLUID-01",
    name="Hagen-Poiseuille Continuum Breakdown in Nanofluidics",
    domain="Microfluidics",
    formula_str="3.14159 * 1e5 * R**4 / (8 * 1e-3 * 1e-4)",
    independent_vars=["R"],
    variable_domains={"R": (1e-9, 1e-5)},
    critical_boundaries={"R": [0.0]},
    invariants=["knudsen_continuum", "positivity"],
    expected_repairable=True,
    failure_mode_description="No-slip Hagen-Poiseuille formula fails when Kn > 0.01.",
    regularized_candidate_factory=lambda c: "(3.14159 * 1e5 / (8 * 1e-3 * 1e-4)) * (R**4 + 6.0 * 68e-9 * R**3)"
    ),
    PathologicalCase(
    case_id="FRACT-01",
    name="Griffith Linear Elastic Crack Tip Singularity",
    domain="Fracture Mechanics",
    formula_str="30e6 / sqrt(2 * 3.14159 * r)",
    independent_vars=["r"],
    variable_domains={"r": (1e-9, 1e-2)},
    critical_boundaries={"r": [0.0]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Linear elastic stress intensity blows up as r -> 0.",
    regularized_candidate_factory=lambda r0: f"30e6 / sqrt(2 * 3.14159 * (r + {r0}))"
    ),
    PathologicalCase(
    case_id="KINET-01",
    name="Arrhenius Reaction Rate at Absolute Zero",
    domain="Chemical Kinetics",
    formula_str="1e13 * exp(-80000.0 / (8.314 * T))",
    independent_vars=["T"],
    variable_domains={"T": (0.0, 500.0)},
    critical_boundaries={"T": [0.0]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Exponential zero singularity at T=0.",
    regularized_candidate_factory=lambda t0: f"1e13 * exp(-80000.0 / (8.314 * (T + {t0})))"
    ),
    PathologicalCase(
    case_id="BIO-01",
    name="Michaelis-Menten Substrate Depletion Limit",
    domain="Biochemistry",
    formula_str="(10.0 / 100.0) / s_substrate + 1.0 / 100.0",
    independent_vars=["s_substrate"],
    variable_domains={"s_substrate": (1e-5, 100.0)},
    critical_boundaries={"s_substrate": [0.0]},
    invariants=["positivity"],
    expected_repairable=True,
    failure_mode_description="Lineweaver-Burk double reciprocal form diverges to infinity as [S] -> 0.",
    regularized_candidate_factory=lambda km: f"100.0 * s_substrate / ({km} + s_substrate)"
    ),

    # =========================================================================
    # PART 2: 12 INTENTIONALLY UNREPAIRABLE / PATHOLOGICAL CASES
    # (System MUST detect flaw and issue REFUSED_CANNOT_REPAIR)
    # =========================================================================
    PathologicalCase(
    case_id="PATHO-01",
    name="Negative Kelvin Ideal Gas Equation",
    domain="Thermodynamics",
    formula_str="8.314 * T / 0.0224",
    independent_vars=["T"],
    variable_domains={"T": (-100.0, -1.0)},
    invariants=["third_law_thermodynamics", "positivity"],
    expected_repairable=False,
    failure_mode_description="UNPHYSICAL_NEGATIVE_TEMPERATURE: Negative Kelvin temperature violates 3rd Law of Thermodynamics; must be refused."
    ),
    PathologicalCase(
    case_id="PATHO-02",
    name="Superluminal Tachyon Acausal Mass Equation",
    domain="Relativistic Physics",
    formula_str="9.1e-31 / sqrt(1 - (v / 3e8)**2)",
    independent_vars=["v"],
    variable_domains={"v": (3.5e8, 1e9)},
    invariants=["lorentz_causality"],
    expected_repairable=False,
    failure_mode_description="SUPERLUMINAL_LORENTZ_VIOLATION_ACAUSAL: Velocity exceeds speed of light, yielding imaginary mass; must be refused."
    ),
    PathologicalCase(
    case_id="PATHO-03",
    name="Imaginary Speed of Sound in Spinodal Instability",
    domain="Fluid Dynamics",
    formula_str="sqrt(-500.0 * rho)",
    independent_vars=["rho"],
    variable_domains={"rho": (1.0, 1000.0)},
    invariants=["positivity"],
    expected_repairable=False,
    failure_mode_description="IMAGINARY_SOUND_SPEED: Negative bulk modulus causes elliptic instability and imaginary sound speed; must be refused."
    ),
    PathologicalCase(
    case_id="PATHO-04",
    name="Negative Entropy Production Thermodynamic Process",
    domain="Non-Equilibrium Thermodynamics",
    formula_str="-5.0 * sigma_entropy",
    independent_vars=["sigma_entropy"],
    variable_domains={"sigma_entropy": (1.0, 100.0)},
    invariants=["second_law_thermodynamics", "positivity"],
    expected_repairable=False,
    failure_mode_description="NEGATIVE_ENTROPY_PRODUCTION_2ND_LAW_VIOLATION: Violates Clausius-Duhem 2nd law inequality; must be refused."
    ),
    PathologicalCase(
    case_id="PATHO-05",
    name="Naked Singularity Curvature Inversion",
    domain="General Relativity",
    formula_str="48.0 * m_mass**2 / (r**6)",
    independent_vars=["r", "m_mass"],
    variable_domains={"r": (1e-5, 10.0), "m_mass": (-5.0, -1.0)},
    invariants=["cosmic_censorship", "positivity"],
    expected_repairable=False,
    failure_mode_description="NAKED_SINGULARITY_INVERSION: Negative mass Schwarzschild curvature violates Cosmic Censorship; must be refused."
    ),
    PathologicalCase(
    case_id="PATHO-06",
    name="Unbounded Exponential Growth in Closed Finite Space",
    domain="Ecology",
    formula_str="100.0 * exp(5.0 * t)",
    independent_vars=["t"],
    variable_domains={"t": (0.0, 1000.0)},
    invariants=["positivity"],
    expected_repairable=False,
    failure_mode_description="Exponential divergence violates finite matter/carrying capacity conservation; cannot soften without external carrying capacity model."
    ),
    PathologicalCase(
    case_id="PATHO-07",
    name="Zero-Viscosity Multi-Valued Shock Profile",
    domain="Gas Dynamics",
    formula_str="1.0 / (x - t)",
    independent_vars=["x", "t"],
    variable_domains={"x": (-5.0, 5.0), "t": (-5.0, 5.0)},
    critical_boundaries={"x": [0.0]},
    expected_repairable=False,
    failure_mode_description="Singular shock formation without viscous Rankine-Hugoniot dissipation condition; cannot be repaired as continuous field without jump condition."
    ),
    PathologicalCase(
    case_id="PATHO-08",
    name="Negative Capacitance Electrostatic Collapse",
    domain="Electrodynamics",
    formula_str="0.5 * q_charge**2 / (-1e-9)",
    independent_vars=["q_charge"],
    variable_domains={"q_charge": (1e-6, 1e-3)},
    invariants=["positivity"],
    expected_repairable=False,
    failure_mode_description="UNPHYSICAL_NEGATIVE energy density in electrostatic state without domain stabilizing terms."
    ),
    PathologicalCase(
    case_id="PATHO-09",
    name="Non-Renormalizable High-Energy Unitarity Violation",
    domain="High Energy Physics",
    formula_str="1.16e-5 * s",
    independent_vars=["s"],
    variable_domains={"s": (1e8, 1e16)},
    invariants=["unitarity_bound"],
    expected_repairable=False,
    failure_mode_description="UNITARITY_VIOLATION: Linear growth in s-channel cross section violates partial wave unitarity bound."
    ),
    PathologicalCase(
    case_id="PATHO-10",
    name="Negative Mass Runaway Self-Acceleration",
    domain="Classical Mechanics",
    formula_str="10.0 / m_mass",
    independent_vars=["m_mass"],
    variable_domains={"m_mass": (-10.0, -0.1)},
    invariants=["positivity"],
    expected_repairable=False,
    failure_mode_description="UNPHYSICAL_NEGATIVE mass produces runaway self-acceleration violating momentum conservation."
    ),
    PathologicalCase(
    case_id="PATHO-11",
    name="Infrared Logarithmic Gauge Field Divergence",
    domain="Quantum Electrodynamics",
    formula_str="log(omega)",
    independent_vars=["omega"],
    variable_domains={"omega": (1e-15, 1.0)},
    critical_boundaries={"omega": [0.0]},
    expected_repairable=False,
    failure_mode_description="Logarithmic infrared divergence as photon energy omega -> 0 cannot be resolved without detector finite resolution."
    ),
    PathologicalCase(
    case_id="PATHO-12",
    name="Agnostic Negative Coordinate Logarithm",
    domain="Complex Analysis",
    formula_str="log(x)",
    independent_vars=["x"],
    variable_domains={"x": (-10.0, -0.1)},
    expected_repairable=False,
    failure_mode_description="Strictly negative real coordinate in logarithm produces imaginary output without real analytic continuation."
    )
    ]


def _get_100_case_suite() -> list[PathologicalCase]:
    """Frozen 100-case cross-disciplinary benchmark suite spanning 10 distinct pathology categories."""
    cases: list[PathologicalCase] = []
    
    # -------------------------------------------------------------
    # 1. KNOWN MATHEMATICAL SINGULARITIES (15 cases)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "SING-01", "Cauchy Pole", "Analysis", "1.0 / x", ["x"], {"x": (1e-5, 10.0)},
        critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"x / (x**2 + {eps}**2)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-02", "Quadratic Pole", "Analysis", "1.0 / (x**2)", ["x"], {"x": (1e-5, 10.0)},
        critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / (x**2 + {eps}**2)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-03", "Shifted Finite Pole", "Analysis", "1.0 / (x - 2.5)", ["x"], {"x": (0.0, 5.0)},
        critical_boundaries={"x": [2.5]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"(x - 2.5) / ((x - 2.5)**2 + {eps}**2)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-04", "Bipolar Dipole Singularity", "Analysis", "1.0 / (x**2 - 1.0)", ["x"], {"x": (0.0, 2.0)},
        critical_boundaries={"x": [1.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"(x**2 - 1.0) / ((x**2 - 1.0)**2 + {eps}**2)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-05", "Cubic Pole", "Analysis", "1.0 / (x**3)", ["x"], {"x": (1e-5, 10.0)},
        critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"x / (x**4 + {eps}**4)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-06", "Quartic Pole", "Analysis", "1.0 / (x**4)", ["x"], {"x": (1e-5, 10.0)},
        critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / (x**4 + {eps}**4)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-07", "Inverse Square Root Singularity", "Analysis", "1.0 / sqrt(x)", ["x"], {"x": (1e-6, 10.0)},
        critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / sqrt(x + {eps})",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-08", "Square Root Branch Cut Failure", "Complex Analysis", "sqrt(x)", ["x"], {"x": (-5.0, -0.1)},
        expected_repairable=False, failure_mode_description="Negative radicand produces unphysical imaginary outputs.",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-09", "Fourth Root Branch Cut Failure", "Complex Analysis", "x**0.25", ["x"], {"x": (-5.0, -0.1)},
        expected_repairable=False, failure_mode_description="Fourth root on negative domain produces imaginary outputs.",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-10", "Negative Logarithmic Pole", "Complex Analysis", "log(x)", ["x"], {"x": (-5.0, -0.1)},
        expected_repairable=False, failure_mode_description="Logarithm of strictly negative coordinate.",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-11", "Denominator Polynomial Real Root", "Algebra", "1.0 / (x**3 - 8.0)", ["x"], {"x": (0.0, 4.0)},
        critical_boundaries={"x": [2.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"(x**3 - 8.0) / ((x**3 - 8.0)**2 + {eps}**2)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-12", "Essential Exponent Singularity", "Analysis", "exp(-1.0 / x)", ["x"], {"x": (0.0, 5.0)},
        critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"exp(-1.0 / (x + {eps}))",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-13", "Biquadratic Branch Pole", "Analysis", "1.0 / sqrt(1.0 - x**2)", ["x"], {"x": (0.0, 1.0)},
        critical_boundaries={"x": [1.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / (((1.0 - x**2)**2 + {eps}**2)**0.25)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-14", "Double Root Pole", "Algebra", "1.0 / (x**2 - 4.0*x + 4.0)", ["x"], {"x": (0.0, 4.0)},
        critical_boundaries={"x": [2.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / ((x - 2.0)**2 + {eps}**2)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "SING-15", "Removable Sinc Singularity", "Analysis", "sin(x) / x", ["x"], {"x": (-5.0, 5.0)},
        critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"sin(x) * x / (x**2 + {eps}**2)",
        category=BenchmarkCategory.KNOWN_MATHEMATICAL_SINGULARITIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 2. PHYSICAL DIVERGENCES (15 cases)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "DIV-01", "Coulomb Potential Point Singularity", "Electromagnetism", "1.6e-19 / (1.11e-10 * r)", ["r"],
        {"r": (1e-15, 1e-8)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.6e-19 / (1.11e-10 * sqrt(r**2 + {eps}**2))",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-02", "Newtonian Gravity Point Singularity", "Astrophysics", "6.674e-11 * 5.97e24 * 1.0 / (r**2)", ["r"],
        {"r": (1e-3, 1e7)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"6.674e-11 * 5.97e24 * 1.0 / (r**2 + {eps}**2)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-03", "Casimir NEMS Stiction Divergence", "Nanotechnology", "1e-19 * 1e-8 / (6 * 3.1415926535 * z**3)", ["z"],
        {"z": (1e-11, 1e-6)}, critical_boundaries={"z": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda z0: f"(1e-19 * 1e-8 / (6 * 3.1415926535 * (z + {z0})**3)) * (1 - ({z0} / (z + {z0}))**6)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-04", "Magnetic Dipole On-Axis Field Divergence", "Electromagnetism", "1e-7 * 2.0 * 1.0 / (r**3)", ["r"],
        {"r": (1e-5, 1.0)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1e-7 * 2.0 * 1.0 / ((r**2 + {eps}**2)**1.5)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-05", "Electric Quadrupole Singularity", "Electromagnetism", "1.0 / (1.11e-10 * r**4)", ["r"],
        {"r": (1e-5, 1.0)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / (1.11e-10 * (r**2 + {eps}**2)**2)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-06", "London Dispersion 1/r^6 Energy Divergence", "Molecular Physics", "1e-78 / (r**6)", ["r"],
        {"r": (1e-10, 1e-7)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1e-78 / ((r**2 + {eps}**2)**3)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-07", "Lennard-Jones Steep 1/r^12 Divergence", "Chemical Physics", "1e-134 / (r**12)", ["r"],
        {"r": (1e-10, 1e-7)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1e-134 / ((r**2 + {eps}**2)**6)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-08", "Biot-Savart Wire Field Divergence", "Magnetostatics", "4e-7 * 3.14159 * 10.0 / (2 * 3.14159 * r)", ["r"],
        {"r": (1e-5, 1.0)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"4e-7 * 3.14159 * 10.0 / (2 * 3.14159 * sqrt(r**2 + {eps}**2))",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-09", "Young-Laplace Capillary Divergence", "Fluid Mechanics", "2.0 * 0.0728 / r", ["r"],
        {"r": (1e-8, 1e-3)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"2.0 * 0.0728 / (r + {eps})",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-10", "Hertzian Contact Stress Singularity", "Contact Mechanics", "100.0 / sqrt(delta_c)", ["delta_c"],
        {"delta_c": (1e-6, 1e-2)}, critical_boundaries={"delta_c": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"100.0 / sqrt(delta_c + {eps})",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-11", "Classical Electron Self-Energy Divergence", "Electrodynamics", "2.3e-28 / r_e", ["r_e"],
        {"r_e": (1e-18, 1e-14)}, critical_boundaries={"r_e": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"2.3e-28 / (r_e + {eps})",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-12", "Fourier Point Heat Source Singularity", "Heat Transfer", "1000.0 / sqrt(t)", ["t"],
        {"t": (1e-5, 10.0)}, critical_boundaries={"t": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1000.0 / sqrt(t + {eps})",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-13", "Vortex Filament Circulation Singularity", "Aerodynamics", "5.0 / (2 * 3.14159 * r)", ["r"],
        {"r": (1e-4, 5.0)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"5.0 * r / (2 * 3.14159 * (r**2 + {eps}**2))",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-14", "Acoustic Nozzle Throat Divergence", "Acoustics", "1.0 / (1.0 - M**2)", ["M"],
        {"M": (0.0, 1.05)}, critical_boundaries={"M": [1.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"(1.0 - M**2) / ((1.0 - M**2)**2 + {eps}**2)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DIV-15", "Plasma Sheath Field Divergence", "Plasma Physics", "1000.0 / x", ["x"],
        {"x": (1e-5, 1.0)}, critical_boundaries={"x": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1000.0 * x / (x**2 + {eps}**2)",
        category=BenchmarkCategory.PHYSICAL_DIVERGENCES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 3. DOMAIN OF VALIDITY FAILURES (15 cases)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "DOM-01", "Hagen-Poiseuille Continuum Breakdown", "Microfluidics", "3.14159 * 1e5 * R**4 / (8 * 1e-3 * 1e-4)", ["R"],
        {"R": (1e-9, 1e-5)}, critical_boundaries={"R": [0.0]}, invariants=["knudsen_continuum", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "(3.14159 * 1e5 / (8 * 1e-3 * 1e-4)) * (R**4 + 6.0 * 68e-9 * R**3)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-02", "Hall-Petch Nanoscale Dislocation Breakdown", "Materials Science", "70e6 + 0.74e6 / sqrt(d)", ["d"],
        {"d": (1e-10, 1e-4)}, critical_boundaries={"d": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda dc: f"70e6 + 0.74e6 * sqrt(d) * d / (d**2 + ({dc} * 1e-7)**2)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-03", "Arrhenius Reaction Rate at Absolute Zero", "Chemical Kinetics", "1e13 * exp(-80000.0 / (8.314 * T))", ["T"],
        {"T": (0.0, 500.0)}, critical_boundaries={"T": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda t0: f"1e13 * exp(-80000.0 / (8.314 * (T + {t0})))",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-04", "Relativistic Kinetic Energy Velocity Limit", "Relativistic Mechanics", "9.1e-31 * (3e8)**2 * (1 / sqrt(1 - (v / 3e8)**2) - 1)", ["v"],
        {"v": (0.0, 3e8)}, critical_boundaries={"v": [3e8]}, invariants=["lorentz_causality", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"9.1e-31 * (3e8)**2 * (1 / (((1 - (v / 3e8)**2)**2 + {eps}**2)**0.25) - 1)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-05", "Prandtl-Glauert Compressibility Singularity", "Aerodynamics", "-0.5 / sqrt(1 - M**2)", ["M"],
        {"M": (0.0, 1.05)}, critical_boundaries={"M": [1.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"-0.5 / (((1 - M**2)**2 + {eps}**2)**0.25)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-06", "High-Field Drift Velocity Saturation", "Semiconductors", "0.14 * E_field", ["E_field"],
        {"E_field": (1e3, 1e7)}, invariants=["velocity_saturation_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "0.14 * E_field / sqrt(1.0 + (0.14 * E_field / 1e5)**2)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-07", "Butler-Volmer Current Density Blowup", "Electrochemistry", "10.0 * (exp(19.4 * eta) - exp(-19.4 * eta))", ["eta"],
        {"eta": (0.0, 1.2)}, invariants=["current_density_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda jlim: "(10.0 * (exp(19.4 * eta) - exp(-19.4 * eta)) * 1000.0) / (1000.0 + abs(10.0 * (exp(19.4 * eta) - exp(-19.4 * eta))))",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-08", "Gibbs-Thomson Nanoparticle Negative Melting Point", "Thermodynamics", "1337.0 * (1.0 - 4.37e-10 / r)", ["r"],
        {"r": (1e-10, 1e-7)}, critical_boundaries={"r": [0.0, 4.37e-10]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda r0: f"1337.0 * exp(-4.37e-10 / (r + {r0}))",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-09", "Ideal Gas Law Finite Volume Failure", "Thermodynamics", "8.314 * 300.0 / V_vol", ["V_vol"],
        {"V_vol": (1e-6, 1.0)}, critical_boundaries={"V_vol": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda b: f"8.314 * 300.0 / (V_vol + {b})",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-10", "Linear Hooke Law Plasticity Breakdown", "Solid Mechanics", "200e9 * strain", ["strain"],
        {"strain": (0.0, 0.1)}, invariants=["stress_saturation_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "400e6 * (1.0 - exp(-200e9 * strain / 400e6))",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-11", "Diode Series Resistance High Injection", "Electronics", "1e-12 * exp(38.9 * V_d)", ["V_d"],
        {"V_d": (0.0, 1.0)}, invariants=["current_density_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "(1e-12 * exp(38.9 * V_d) * 10.0) / (10.0 + 1e-12 * exp(38.9 * V_d))",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-12", "Classical Doppler Frequency Breakdown", "Relativistic Optics", "5e14 / (1.0 - v / 3e8)", ["v"],
        {"v": (0.0, 2.99e8)}, critical_boundaries={"v": [3e8]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"5e14 * (1.0 - v / 3e8) / ((1.0 - v / 3e8)**2 + {eps}**2)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-13", "Stokes Drag High Reynolds Breakdown", "Fluid Dynamics", "6 * 3.14159 * 1e-3 * 0.01 * vel", ["vel"],
        {"vel": (0.1, 50.0)}, invariants=["reynolds_laminar_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "6 * 3.14159 * 1e-3 * 0.01 * vel / (1.0 + 1000.0 * vel * 0.02 / 1e-3)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-14", "Thin Film Fourier Phonon Breakdown", "Nanoscale Heat Transfer", "150.0 * 10.0 / thickness", ["thickness"],
        {"thickness": (1e-9, 1e-6)}, critical_boundaries={"thickness": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "150.0 * 10.0 / (thickness + 40e-9)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "DOM-15", "Dulong-Petit Low Temperature Failure", "Solid State Physics", "24.942 * (1.0 + 0.0001 * T_temp)", ["T_temp"],
        {"T_temp": (1.0, 300.0)}, critical_boundaries={"T_temp": [0.0]}, invariants=["third_law_heat_capacity", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "24.942 * T_temp**3 / (T_temp**3 + 50.0**3)",
        category=BenchmarkCategory.DOMAIN_OF_VALIDITY_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 4. CONSERVATION & INVARIANT VIOLATIONS (10 cases - PATHOLOGICAL)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "INV-01", "Negative Kelvin Ideal Gas Equation", "Thermodynamics", "8.314 * T / 0.0224", ["T"],
        {"T": (-100.0, -1.0)}, invariants=["third_law_thermodynamics", "positivity"], expected_repairable=False,
        failure_mode_description="UNPHYSICAL_NEGATIVE temperature violates 3rd Law of Thermodynamics.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-02", "Negative Entropy Production Process", "Non-Equilibrium Thermodynamics", "-5.0 * sigma_entropy", ["sigma_entropy"],
        {"sigma_entropy": (1.0, 100.0)}, invariants=["second_law_thermodynamics", "positivity"], expected_repairable=False,
        failure_mode_description="NEGATIVE_ENTROPY_PRODUCTION violates 2nd Law of Thermodynamics.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-03", "Superluminal Tachyon Acausal Mass Equation", "Relativistic Physics", "9.1e-31 / sqrt(1 - (v / 3e8)**2)", ["v"],
        {"v": (3.5e8, 1e9)}, invariants=["lorentz_causality"], expected_repairable=False,
        failure_mode_description="SUPERLUMINAL_LORENTZ_VIOLATION_ACAUSAL: Velocity exceeds speed of light.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-04", "Naked Singularity Curvature Inversion", "General Relativity", "2.0 * 6.674e-11 * mass_curv / (3e8)**2", ["mass_curv"],
        {"mass_curv": (-10.0, -0.1)}, invariants=["cosmic_censorship", "positivity"], expected_repairable=False,
        failure_mode_description="Negative mass naked singularity curvature inversion violates Cosmic Censorship.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-05", "High-Energy Froissart Unitarity Violation", "High Energy Physics", "1.0 * (s / 1e10)**2", ["s"],
        {"s": (1e10, 1e16)}, invariants=["unitarity_bound"], expected_repairable=False,
        failure_mode_description="Cross section grows as s^2, violating Froissart unitarity bound.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-06", "Negative Quantum Probability Density", "Quantum Mechanics", "-0.5 * psi_sq", ["psi_sq"],
        {"psi_sq": (0.1, 1.0)}, invariants=["positivity"], expected_repairable=False,
        failure_mode_description="UNPHYSICAL_NEGATIVE quantum probability violates Born rule.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-07", "Surface Code Super-Threshold Logical Error Blowup", "Quantum Computing", "0.1 * (p / p_th)**2", ["p", "p_th"],
        {"p": (0.001, 0.08), "p_th": (0.001, 0.02)}, critical_boundaries={"p_th": [0.0]}, invariants=["probability_bound", "positivity"],
        expected_repairable=True, regularized_candidate_factory=lambda eps: f"0.5 * (1 - exp(-0.2 * (p / (p_th + {eps}))**2))",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-08", "Negative Mass Runaway Self-Acceleration", "Classical Mechanics", "10.0 / m_mass", ["m_mass"],
        {"m_mass": (-10.0, -0.1)}, invariants=["positivity"], expected_repairable=False,
        failure_mode_description="Negative mass produces runaway self-acceleration violating momentum conservation.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-09", "Negative Capacitance Electrostatic Explosion", "Electrodynamics", "-1e-6 * V_pot**2", ["V_pot"],
        {"V_pot": (1.0, 100.0)}, invariants=["positivity"], expected_repairable=False,
        failure_mode_description="Negative capacitance electrostatic energy violates thermodynamic stability.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INV-10", "Unbounded Exponential Growth in Closed Basin", "Ecology", "100.0 * exp(5.0 * t)", ["t"],
        {"t": (1.0, 200.0)}, expected_repairable=False,
        failure_mode_description="Exponential growth without carrying capacity violates finite basin conservation.",
        category=BenchmarkCategory.CONSERVATION_INVARIANT_VIOLATIONS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 5. BOUNDARY CONDITION FAILURES (10 cases)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "BND-01", "Griffith Linear Elastic Crack Tip Singularity", "Fracture Mechanics", "30e6 / sqrt(2 * 3.14159 * r)", ["r"],
        {"r": (1e-9, 1e-2)}, critical_boundaries={"r": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda r0: f"30e6 / sqrt(2 * 3.14159 * (r + {r0}))",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-02", "Huh-Scriven Moving Contact Line Singularity", "Fluid Mechanics", "1.0 / x_dist", ["x_dist"],
        {"x_dist": (1e-6, 1e-2)}, critical_boundaries={"x_dist": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"x_dist / (x_dist**2 + {eps}**2)",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-03", "Moffatt Corner Eddy Shear Singularity", "Fluid Mechanics", "10.0 / (r_corner**0.5)", ["r_corner"],
        {"r_corner": (1e-5, 1.0)}, critical_boundaries={"r_corner": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"10.0 / sqrt(r_corner + {eps})",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-04", "Blasius Boundary Layer Leading Edge Singularity", "Aerodynamics", "0.332 / sqrt(x_plate)", ["x_plate"],
        {"x_plate": (1e-5, 1.0)}, critical_boundaries={"x_plate": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"0.332 / sqrt(x_plate + {eps})",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-05", "Re-entrant Corner Electrostatic Singularity", "Electrostatics", "100.0 / (r_wedge**0.333)", ["r_wedge"],
        {"r_wedge": (1e-5, 1.0)}, critical_boundaries={"r_wedge": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"100.0 / ((r_wedge + {eps})**0.333)",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-06", "Cavitating Suction Peak Singularity", "Hydrodynamics", "50.0 / sqrt(s_foil)", ["s_foil"],
        {"s_foil": (1e-5, 1.0)}, critical_boundaries={"s_foil": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"50.0 / sqrt(s_foil + {eps})",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-07", "Kelvin Wave Wake Cusp Divergence", "Wave Mechanics", "1.0 / (theta_angle**0.5)", ["theta_angle"],
        {"theta_angle": (1e-4, 1.0)}, critical_boundaries={"theta_angle": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / sqrt(theta_angle + {eps})",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-08", "Acoustic Point Monopole Radiation", "Acoustics", "1.0 / (4 * 3.14159 * r_dist)", ["r_dist"],
        {"r_dist": (1e-5, 10.0)}, critical_boundaries={"r_dist": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"1.0 / (4 * 3.14159 * sqrt(r_dist**2 + {eps}**2))",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-09", "Zero-Viscosity Multi-Valued Shock Profile", "Gas Dynamics", "100.0 / (x_shock**2)", ["x_shock"],
        {"x_shock": (-1.0, 1.0)}, critical_boundaries={"x_shock": [0.0]}, expected_repairable=False,
        failure_mode_description="Inviscid shock profile forms multi-valued unphysical Riemann catastrophe.",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "BND-10", "Taylor-Couette Zero Gap Torque Divergence", "Fluid Dynamics", "10.0 / gap_width", ["gap_width"],
        {"gap_width": (1e-6, 1e-2)}, critical_boundaries={"gap_width": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"10.0 / (gap_width + {eps})",
        category=BenchmarkCategory.BOUNDARY_CONDITION_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 6. NUMERICAL PATHOLOGIES (10 cases)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "NUM-01", "Exponential Floating Point Overflow", "Numerical Analysis", "exp(1000.0 * x)", ["x"],
        {"x": (1.0, 10.0)}, expected_repairable=False, failure_mode_description="Floating-point IEEE 754 overflow (+inf).",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-02", "Imaginary Sound Speed Spinodal Instability", "Fluid Dynamics", "sqrt(-500.0 * rho)", ["rho"],
        {"rho": (1.0, 1000.0)}, invariants=["positivity"], expected_repairable=False,
        failure_mode_description="Negative bulk modulus causes elliptic instability and imaginary sound speed.",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-03", "Extreme Stiff Sigmoid Gradient Explosion", "Numerical Methods", "1.0 / (1.0 + exp(-1e7 * x))", ["x"],
        {"x": (-1.0, 1.0)}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: "1.0 / (1.0 + exp(-100.0 * x))",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-04", "Infrared Logarithmic Gauge Field Divergence", "Quantum Electrodynamics", "log(omega)", ["omega"],
        {"omega": (1e-15, 1.0)}, critical_boundaries={"omega": [0.0]}, expected_repairable=False,
        failure_mode_description="Logarithmic infrared divergence as photon energy omega -> 0.",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-05", "Subnormal Floating Point Underflow", "Numerical Methods", "1.0 / (x * exp(-1000.0 * x))", ["x"],
        {"x": (1.0, 10.0)}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: "exp(50.0 * x) / (x + 0.01)",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-06", "Catastrophic Cancellation Difference", "Floating Point Arithmetic", "1.0 / (sqrt(x + 1e-16) - sqrt(x))", ["x"],
        {"x": (1.0, 100.0)}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: "2.0 * sqrt(x)",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-07", "Geometric Series Boundary Singularity", "Analysis", "1.0 / (1.0 - x)", ["x"],
        {"x": (0.0, 1.0)}, critical_boundaries={"x": [1.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"(1.0 - x) / ((1.0 - x)**2 + {eps}**2)",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-08", "Rational Function High Degree Oscillation", "Approximation Theory", "1.0 / (1.0 - 25.0 * x**2)", ["x"],
        {"x": (-1.0, 1.0)}, critical_boundaries={"x": [0.2, -0.2]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"(1.0 - 25.0 * x**2) / ((1.0 - 25.0 * x**2)**2 + {eps}**2)",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-09", "Unnormalized Cotangent Zero Pole", "Trigonometry", "1.0 / tan(x)", ["x"],
        {"x": (1e-5, 1.5)}, critical_boundaries={"x": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"tan(x) / (tan(x)**2 + {eps}**2)",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "NUM-10", "Zero Reciprocal Division by Parameter", "Arithmetic", "1.0 / param_zero", ["param_zero"],
        {"param_zero": (1e-6, 1.0)}, critical_boundaries={"param_zero": [0.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"param_zero / (param_zero**2 + {eps}**2)",
        category=BenchmarkCategory.NUMERICAL_PATHOLOGIES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 7. REPRESENTATION & MODEL-FORM FAILURES (10 cases)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "REP-01", "Lineweaver-Burk Double Reciprocal Enzyme Kinetics", "Biochemistry", "(10.0 / 100.0) / s_substrate + 1.0 / 100.0", ["s_substrate"],
        {"s_substrate": (1e-5, 100.0)}, critical_boundaries={"s_substrate": [0.0]}, invariants=["positivity"], expected_repairable=True,
        failure_mode_description="Lineweaver-Burk double reciprocal form diverges to infinity as [S] -> 0.",
        regularized_candidate_factory=lambda km: f"100.0 * s_substrate / ({km} + s_substrate)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-02", "Scatchard Plot Reciprocal Ligand Distortion", "Pharmacology", "10.0 / b_bound - 0.5", ["b_bound"],
        {"b_bound": (1e-5, 20.0)}, critical_boundaries={"b_bound": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "b_bound / (10.0 + 0.5 * b_bound)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-03", "Parallel Resistance Reciprocal Sum", "Circuit Theory", "1.0 / R_1 + 1.0 / 10.0", ["R_1"],
        {"R_1": (1e-5, 100.0)}, critical_boundaries={"R_1": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "10.0 * R_1 / (R_1 + 10.0)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-04", "Logit Log-Odds Representation Blowup", "Information Theory", "log(prob / (1.0 - prob))", ["prob"],
        {"prob": (1e-6, 0.999999)}, critical_boundaries={"prob": [0.0, 1.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: "log((prob + 0.01) / (1.0 - prob + 0.01))",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-05", "LC Resonance Reciprocal Impedance Blowup", "Electrical Engineering", "1.0 / (omega_rad * 1e-3 - 1.0 / (omega_rad * 1e-3))", ["omega_rad"],
        {"omega_rad": (100.0, 2000.0)}, critical_boundaries={"omega_rad": [1000.0]}, expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"(omega_rad * 1e-3 - 1.0 / (omega_rad * 1e-3)) / ((omega_rad * 1e-3 - 1.0 / (omega_rad * 1e-3))**2 + {eps}**2)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-06", "Reciprocal Carrier Mobility Resistivity", "Solid State Electronics", "1.0 / (1.6e-19 * n_density * 0.14)", ["n_density"],
        {"n_density": (1e12, 1e20)}, critical_boundaries={"n_density": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "1.0 / (1.6e-19 * (n_density + 1e12) * 0.14)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-07", "Reciprocal Stiffness Natural Frequency", "Mechanical Vibration", "1.0 / (k_stiff / 2.0)", ["k_stiff"],
        {"k_stiff": (1e-3, 1e5)}, critical_boundaries={"k_stiff": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "1.0 / ((k_stiff + 0.04) / 2.0)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-08", "Reciprocal Compliance Interface Model", "Continuum Mechanics", "1.0 / c_compliance", ["c_compliance"],
        {"c_compliance": (1e-6, 1.0)}, critical_boundaries={"c_compliance": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"c_compliance / (c_compliance**2 + {eps}**2)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-09", "Linearized Langmuir Adsorption Reciprocal", "Surface Chemistry", "1.0 / (0.5 * p_gas) + 1.0", ["p_gas"],
        {"p_gas": (1e-4, 10.0)}, critical_boundaries={"p_gas": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "0.5 * p_gas / (1.0 + 0.5 * p_gas)",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "REP-10", "Acoustic Vacuum Specific Impedance Reciprocal", "Acoustics", "1.0 / (rho_medium * 343.0)", ["rho_medium"],
        {"rho_medium": (1e-5, 1.2)}, critical_boundaries={"rho_medium": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"rho_medium / (343.0 * (rho_medium**2 + {eps}**2))",
        category=BenchmarkCategory.REPRESENTATION_MODEL_FORM_FAILURES, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 8. DELIBERATELY INCOMPATIBLE CONSTRAINTS (5 cases - PATHOLOGICAL)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "INCOMP-01", "Incompatible Contact Zero vs Inverse Square Attraction", "Applied Mechanics", "1.0 / (x**2)", ["x"],
        {"x": (1e-5, 10.0)}, expected_repairable=False,
        failure_mode_description="Mutually contradictory requirements: contact zero f(0)=0 and lower bound f(x)>=1/x^2.",
        constraints=["contact_zero: f(0) = 0", "lower_bound: f(x) >= 1/x^2"],
        category=BenchmarkCategory.DELIBERATELY_INCOMPATIBLE_CONSTRAINTS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INCOMP-02", "Incompatible Closed Conservation vs Perpetual Generation", "Thermodynamics", "100.0 * x", ["x"],
        {"x": (0.0, 10.0)}, expected_repairable=False,
        failure_mode_description="Mutually contradictory: isolated conservation dE/dt = 0 and positive power generation dE/dt >= P_0 > 0.",
        constraints=["conservation: dE/dt = 0", "positive_power_generation: dE/dt >= P_0 > 0"],
        category=BenchmarkCategory.DELIBERATELY_INCOMPATIBLE_CONSTRAINTS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INCOMP-03", "Incompatible Subluminal Causality vs Superluminal Tachyon", "Relativistic Mechanics", "9.1e-31 * v", ["v"],
        {"v": (1e8, 5e8)}, expected_repairable=False,
        failure_mode_description="Mutually contradictory: relativistic causality v <= c and tachyon prescription v >= 1.5c.",
        constraints=["lorentz_causality: v <= c", "tachyon_specification: v >= 1.5c"],
        category=BenchmarkCategory.DELIBERATELY_INCOMPATIBLE_CONSTRAINTS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INCOMP-04", "Incompatible Real Output vs Strictly Negative Radicand", "Complex Analysis", "sqrt(-1.0 - x**2)", ["x"],
        {"x": (-10.0, 10.0)}, expected_repairable=False,
        failure_mode_description="Mutually contradictory: real output requirement Im(f)=0 and strictly negative radicand -1 - x^2 < 0.",
        constraints=["real_output: Im(f) = 0", "negative_radicand: sqrt(-1 - x^2)"],
        category=BenchmarkCategory.DELIBERATELY_INCOMPATIBLE_CONSTRAINTS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))
    cases.append(PathologicalCase(
        "INCOMP-05", "Incompatible Monotonic Decay vs Bounded Harmonic Integral", "Functional Analysis", "1.0 / x", ["x"],
        {"x": (1.0, 1000.0)}, expected_repairable=False,
        failure_mode_description="Mutually contradictory: harmonic power law f(x)=1/x and finite total integral on [1, inf).",
        constraints=["harmonic_form: 1/x", "bounded_integral: int_1_inf_f_finite"],
        category=BenchmarkCategory.DELIBERATELY_INCOMPATIBLE_CONSTRAINTS, novelty_tier=NoveltyTier.TIER_A_KNOWN_REPAIR
    ))

    # -------------------------------------------------------------
    # 9. HIDDEN KNOWN REPAIRS (5 cases - TIER B)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "HID-01", "Hidden Maxwell Slip Boundary in Microchannel Flow", "Fluid Dynamics", "1000.0 * R**4", ["R"],
        {"R": (1e-9, 1e-5)}, critical_boundaries={"R": [0.0]}, invariants=["knudsen_continuum", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "1000.0 * (R**4 + 6.0 * 68e-9 * R**3)",
        category=BenchmarkCategory.HIDDEN_KNOWN_REPAIRS, novelty_tier=NoveltyTier.TIER_B_HIDDEN_REPAIR
    ))
    cases.append(PathologicalCase(
        "HID-02", "Hidden Caughey-Thomas Carrier Velocity Saturation", "Semiconductors", "1400.0 * E_f", ["E_f"],
        {"E_f": (1e2, 1e6)}, invariants=["velocity_saturation_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "1400.0 * E_f / sqrt(1.0 + (1400.0 * E_f / 1e5)**2)",
        category=BenchmarkCategory.HIDDEN_KNOWN_REPAIRS, novelty_tier=NoveltyTier.TIER_B_HIDDEN_REPAIR
    ))
    cases.append(PathologicalCase(
        "HID-03", "Hidden Barenblatt Cohesive Zone Tip Stress", "Fracture Mechanics", "50e6 / sqrt(2 * 3.14159 * r_t)", ["r_t"],
        {"r_t": (1e-9, 1e-2)}, critical_boundaries={"r_t": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda r0: f"50e6 / sqrt(2 * 3.14159 * (r_t + {r0}))",
        category=BenchmarkCategory.HIDDEN_KNOWN_REPAIRS, novelty_tier=NoveltyTier.TIER_B_HIDDEN_REPAIR
    ))
    cases.append(PathologicalCase(
        "HID-04", "Hidden Michaelis-Menten Rational Reformulation", "Enzymology", "(0.05 / 50.0) / s_sub + 1.0 / 50.0", ["s_sub"],
        {"s_sub": (1e-5, 100.0)}, critical_boundaries={"s_sub": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda km: f"50.0 * s_sub / ({km} + s_sub)",
        category=BenchmarkCategory.HIDDEN_KNOWN_REPAIRS, novelty_tier=NoveltyTier.TIER_B_HIDDEN_REPAIR
    ))
    cases.append(PathologicalCase(
        "HID-05", "Hidden Plummer Softening Gravitational Potential", "Astrophysics", "100.0 / r_dist", ["r_dist"],
        {"r_dist": (1e-5, 100.0)}, critical_boundaries={"r_dist": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: f"100.0 / sqrt(r_dist**2 + {eps}**2)",
        category=BenchmarkCategory.HIDDEN_KNOWN_REPAIRS, novelty_tier=NoveltyTier.TIER_B_HIDDEN_REPAIR
    ))

    # -------------------------------------------------------------
    # 10. NOVEL / OPEN SYNTHESIS PROBLEMS (5 cases - TIER C)
    # -------------------------------------------------------------
    cases.append(PathologicalCase(
        "NOV-01", "Coupled Nanoscale Ballistic-Diffusive Thermoelectric Transition", "Quantum Transport", "50.0 * (1.0 / l_mfp)", ["l_mfp"],
        {"l_mfp": (1e-9, 1e-4)}, critical_boundaries={"l_mfp": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "50.0 / (l_mfp + 10e-9)",
        category=BenchmarkCategory.NOVEL_OPEN_REPAIRS, novelty_tier=NoveltyTier.TIER_C_NOVEL_OPEN
    ))
    cases.append(PathologicalCase(
        "NOV-02", "Turbulent Boundary Dissipation Rate with Bounded Floor", "Turbulence Physics", "0.09 * (k_turb**1.5) / (y_wall)", ["k_turb", "y_wall"],
        {"k_turb": (0.01, 10.0), "y_wall": (1e-6, 1.0)}, critical_boundaries={"y_wall": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "0.09 * (k_turb**1.5) / sqrt(y_wall**2 + 0.04**2)",
        category=BenchmarkCategory.NOVEL_OPEN_REPAIRS, novelty_tier=NoveltyTier.TIER_C_NOVEL_OPEN
    ))
    cases.append(PathologicalCase(
        "NOV-03", "Non-Newtonian Shear-Thinning Bounded Viscosity", "Rheology", "10.0 * (shear_rate**(-0.5))", ["shear_rate"],
        {"shear_rate": (1e-4, 1e4)}, critical_boundaries={"shear_rate": [0.0]}, invariants=["positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "10.0 / (shear_rate**0.5 + 0.1) + 0.01",
        category=BenchmarkCategory.NOVEL_OPEN_REPAIRS, novelty_tier=NoveltyTier.TIER_C_NOVEL_OPEN
    ))
    cases.append(PathologicalCase(
        "NOV-04", "2D Material Optical Phonon Carrier Saturation", "Condensed Matter", "2000.0 * e_field_2d", ["e_field_2d"],
        {"e_field_2d": (1e2, 1e7)}, invariants=["velocity_saturation_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "2000.0 * e_field_2d / sqrt(1.0 + (2000.0 * e_field_2d / 1e5)**2)",
        category=BenchmarkCategory.NOVEL_OPEN_REPAIRS, novelty_tier=NoveltyTier.TIER_C_NOVEL_OPEN
    ))
    cases.append(PathologicalCase(
        "NOV-05", "Quantum Dot Auger Non-Radiative Quenching Saturation", "Quantum Optics", "1e8 * (n_exciton**3)", ["n_exciton"],
        {"n_exciton": (0.0, 10.0)}, invariants=["saturation_ceiling", "positivity"], expected_repairable=True,
        regularized_candidate_factory=lambda eps: "(1e8 * n_exciton**3) / (1.0 + 0.1 * n_exciton**3)",
        category=BenchmarkCategory.NOVEL_OPEN_REPAIRS, novelty_tier=NoveltyTier.TIER_C_NOVEL_OPEN
    ))

    return cases
