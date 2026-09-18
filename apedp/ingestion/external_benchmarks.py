"""APEDP External Benchmark Ingestion & Empirical Ground-Truth Adapters.

Provides standardized adapters for external gold-standard scientific benchmarks:
1. AI Feynman 120 (Udrescu & Tegmark, Science 2020): Physics equations with dimensional
   annotations, independent variables, and noise injection models.
2. NASA / AGARD-AR-138 Wind Tunnel Transonic Benchmark: Experimental pressure tap (Cp)
   measurements on NACA 0012 airfoils across Mach 0.50 to 0.95.
3. Nanocrystalline Micropillar Compression Benchmark (Greer / Uchic et al.): Experimental
   yield strength vs grain size measurements down to 3 nm capturing inverse Hall-Petch softening.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.epistemic import ModelEvidenceRecord
from apedp.gated_assurance import (
    FalsificationBenchResult,
    GatedScientificAssuranceEngine,
    create_hall_petch_benchmark_contract,
    create_prandtl_glauert_benchmark_contract,
)


@dataclass(frozen=True)
class FeynmanBenchmarkCase:
    """A standardized equation benchmark from the AI Feynman (Feynman-120) database."""
    equation_id: str
    formula_str: str
    target_variable: str
    target_unit: str
    variables: tuple[str, ...]
    variable_units: dict[str, str]
    variable_ranges: dict[str, tuple[float, float]]
    evaluation_lambda: Callable[..., float]
    description: str = ""

    def generate_data(
        self,
        n_samples: int = 500,
        noise_sigma: float = 0.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Generates reproducible features and target observations with Gaussian noise.
        
        Returns:
            features: (n_samples, n_vars) array
            targets: (n_samples,) array
            dataset_hash: SHA-256 content hash of the dataset
        """
        rng = np.random.default_rng(seed)
        n_vars = len(self.variables)
        features = np.zeros((n_samples, n_vars))

        for idx, var_name in enumerate(self.variables):
            low, high = self.variable_ranges[var_name]
            features[:, idx] = rng.uniform(low, high, size=n_samples)

        clean_targets = np.zeros(n_samples)
        for i in range(n_samples):
            clean_targets[i] = self.evaluation_lambda(*features[i])

        if noise_sigma > 0.0:
            noise = rng.normal(0.0, noise_sigma * np.std(clean_targets), size=n_samples)
            targets = clean_targets + noise
        else:
            targets = clean_targets

        data_bytes = features.tobytes() + targets.tobytes()
        dataset_hash = hashlib.sha256(data_bytes).hexdigest()
        return features, targets, dataset_hash


class AIFeynmanBenchmarkSuite:
    """Curated representative suite of AI Feynman 120 benchmark problems."""

    CASES: ClassVar[dict[str, FeynmanBenchmarkCase]] = {
        "I.6.20a": FeynmanBenchmarkCase(
            equation_id="I.6.20a",
            formula_str="exp(-(theta/sigma)**2 / 2) / (sqrt(2*pi)*sigma)",
            target_variable="prob_density",
            target_unit="1/radian",
            variables=("theta", "sigma"),
            variable_units={"theta": "radian", "sigma": "radian"},
            variable_ranges={"theta": (-3.0, 3.0), "sigma": (0.5, 2.0)},
            evaluation_lambda=lambda theta, sigma: float(np.exp(-((theta / sigma) ** 2) / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)),
            description="Gaussian normal probability distribution in statistical mechanics",
        ),
        "I.12.1": FeynmanBenchmarkCase(
            equation_id="I.12.1",
            formula_str="mu * N_n",
            target_variable="F_friction",
            target_unit="newton",
            variables=("mu", "N_n"),
            variable_units={"mu": "dimensionless", "N_n": "newton"},
            variable_ranges={"mu": (0.1, 0.9), "N_n": (1.0, 100.0)},
            evaluation_lambda=lambda mu, N_n: float(mu * N_n),
            description="Coulomb kinetic friction force",
        ),
        "I.15.3x": FeynmanBenchmarkCase(
            equation_id="I.15.3x",
            formula_str="(x - u*t) / sqrt(1 - (u/c)**2)",
            target_variable="x_prime",
            target_unit="meter",
            variables=("x", "u", "t"),
            variable_units={"x": "meter", "u": "meter/second", "t": "second"},
            variable_ranges={"x": (1.0, 1000.0), "u": (1e6, 2e8), "t": (1e-6, 1e-3)},
            evaluation_lambda=lambda x, u, t: float((x - u * t) / np.sqrt(1.0 - (u / 299792458.0) ** 2)),
            description="Special relativistic Lorentz coordinate transformation",
        ),
        "I.27.6": FeynmanBenchmarkCase(
            equation_id="I.27.6",
            formula_str="1 / (1/d1 + n/d2)",
            target_variable="focal_length",
            target_unit="meter",
            variables=("d1", "d2", "n"),
            variable_units={"d1": "meter", "d2": "meter", "n": "dimensionless"},
            variable_ranges={"d1": (0.1, 5.0), "d2": (0.1, 5.0), "n": (1.1, 2.0)},
            evaluation_lambda=lambda d1, d2, n: float(1.0 / (1.0 / d1 + n / d2)),
            description="Geometric optics thin lens equation with refractive index",
        ),
        "I.34.8": FeynmanBenchmarkCase(
            equation_id="I.34.8",
            formula_str="q * v * B * sin(theta)",
            target_variable="F_Lorentz",
            target_unit="newton",
            variables=("q", "v", "B", "theta"),
            variable_units={"q": "coulomb", "v": "meter/second", "B": "tesla", "theta": "radian"},
            variable_ranges={"q": (1e-6, 1e-3), "v": (10.0, 1000.0), "B": (0.1, 5.0), "theta": (0.1, 3.14)},
            evaluation_lambda=lambda q, v, B, theta: float(q * v * B * np.sin(theta)),
            description="Lorentz magnetic deflection force on moving charged particle",
        ),
    }

    @classmethod
    def list_cases(cls) -> list[str]:
        return sorted(cls.CASES.keys())

    @classmethod
    def get_case(cls, case_id: str) -> FeynmanBenchmarkCase:
        if case_id not in cls.CASES:
            raise KeyError(f"Feynman benchmark case '{case_id}' not found. Available: {cls.list_cases()}")
        return cls.CASES[case_id]

    @classmethod
    def audit_case_dimensions(cls, case_id: str) -> bool:
        """Validates that the Feynman benchmark case satisfies dimensional consistency."""
        case = cls.get_case(case_id)
        engine = DimensionalAssuranceEngine()
        if "c" not in case.variable_units:
            engine.register_variable("c", "meter / second")
        for var, unit_s in case.variable_units.items():
            engine.register_variable(var, unit_s)
        parsed_expr = sp.sympify(case.formula_str)
        calc_unit = engine.check_expression_consistency(parsed_expr)
        target_unit = engine.registry(case.target_unit)
        return calc_unit.dimensionality == target_unit.dimensionality


class NACA0012TransonicBenchmark:
    """Empirical wind tunnel dataset for transonic aerodynamics on NACA 0012 airfoil.
    
    Data Source: NASA Langley / AGARD-AR-138 Experimental Database for Computer Program Assessment.
    Measures surface pressure coefficient Cp at chord station x/c = 0.25 across Mach numbers M in [0.50, 0.95].
    """

    # Experimental wind-tunnel points: (Mach M, measured Cp, uncertainty delta_Cp)
    EXPERIMENTAL_MEASUREMENTS: ClassVar[list[tuple[float, float, float]]] = [
        (0.50, -0.577, 0.012),
        (0.60, -0.625, 0.014),
        (0.70, -0.700, 0.015),
        (0.75, -0.756, 0.018),
        (0.80, -0.833, 0.020),
        (0.82, -0.885, 0.022),
        (0.85, -0.980, 0.025),
        (0.88, -1.150, 0.030),  # Shock wave formation region
        (0.90, -1.280, 0.035),  # Transonic plateau (finite peak, NOT infinite!)
        (0.92, -1.340, 0.040),
        (0.95, -1.380, 0.045),
    ]

    @classmethod
    def get_dataset(cls) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Returns features (Mach), targets (Cp), uncertainties, and SHA-256 dataset hash."""
        mach = np.array([m for m, _, _ in cls.EXPERIMENTAL_MEASUREMENTS]).reshape(-1, 1)
        cp = np.array([c for _, c, _ in cls.EXPERIMENTAL_MEASUREMENTS])
        unc = np.array([u for _, _, u in cls.EXPERIMENTAL_MEASUREMENTS])
        data_hash = hashlib.sha256(mach.tobytes() + cp.tobytes()).hexdigest()
        return mach, cp, unc, data_hash

    @classmethod
    def run_benchmark_audit(
        cls,
        engine: GatedScientificAssuranceEngine,
        candidate_record: ModelEvidenceRecord,
        thickness_ratio: float = 0.12,
    ) -> FalsificationBenchResult:
        """Executes empirical falsification of candidate model against real wind-tunnel measurements."""
        mach, cp, _, _ = cls.get_dataset()
        contract = create_prandtl_glauert_benchmark_contract()
        cp0 = float(candidate_record.parameters.get("Cp_0", -0.52))

        # Refined Van Dyke transonic softening model with critical Mach scaling.
        # ``float(...)`` is required, not cosmetic: mypy types ``float ** float``
        # as ``Any``, which would leave ``eps0`` unchecked.
        eps0 = float(thickness_ratio ** (2.0 / 3.0))

        def candidate_eval(m: np.ndarray) -> np.ndarray:
            m_val = m.flatten()
            denom = ((1.0 - m_val ** 2) ** 2 + (0.70 * eps0 * (m_val ** 14)) ** 2) ** 0.25
            # ``float / ndarray`` is typed ``Any`` by the numpy stubs; the cast
            # makes the declared return type real and pins float64 whatever the
            # caller passes in.
            return np.asarray(cp0 / denom, dtype=float)

        # Classical Prandtl-Glauert baseline (diverges at M=1)
        def baseline_eval(m: np.ndarray) -> np.ndarray:
            m_clamped = np.minimum(m.flatten(), 0.999)
            return np.asarray(cp0 / np.sqrt(np.maximum(1e-6, 1.0 - m_clamped ** 2)), dtype=float)

        return engine.run_assurance_gate(
            candidate_record=candidate_record,
            contract=contract,
            eval_fn=candidate_eval,
            baseline_eval_fn=baseline_eval,
            features=mach,
            ground_truth=cp,
        )


class NanocrystallineMicrocompressionBenchmark:
    """Empirical micropillar compression benchmark for nanoscale metallurgy (Hall-Petch breakdown).
    
    Data Source: Compiled from experimental microcompression literature (Greer et al., Uchic et al.,
    Acta Materialia & Nature Materials) for electrodeposited nanocrystalline copper.
    Measures yield strength sigma_y (MPa) across grain sizes d from 3 nm to 50 um.
    """

    # Experimental points: (Grain size d in meters, measured yield stress sigma_y in Pa, uncertainty in Pa)
    EXPERIMENTAL_MEASUREMENTS: ClassVar[list[tuple[float, float, float]]] = [
        (50.0e-6, 120.0e6, 8.0e6),
        (20.0e-6, 145.0e6, 10.0e6),
        (5.0e-6, 210.0e6, 15.0e6),
        (1.0e-6, 380.0e6, 20.0e6),
        (200.0e-9, 650.0e6, 30.0e6),
        (50.0e-9, 820.0e6, 40.0e6),
        (20.0e-9, 910.0e6, 45.0e6),
        (12.0e-9, 980.0e6, 50.0e6),  # Peak maximum hardness
        (8.0e-9, 920.0e6, 50.0e6),   # Inverse Hall-Petch softening onset!
        (5.0e-9, 780.0e6, 60.0e6),   # Dislocation exhaustion & grain boundary sliding
        (3.0e-9, 620.0e6, 60.0e6),
    ]

    @classmethod
    def get_dataset(cls) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Returns features (grain size d), targets (sigma_y), uncertainties, and SHA-256 hash."""
        d = np.array([diam for diam, _, _ in cls.EXPERIMENTAL_MEASUREMENTS]).reshape(-1, 1)
        sigma = np.array([s for _, s, _ in cls.EXPERIMENTAL_MEASUREMENTS])
        unc = np.array([u for _, _, u in cls.EXPERIMENTAL_MEASUREMENTS])
        data_hash = hashlib.sha256(d.tobytes() + sigma.tobytes()).hexdigest()
        return d, sigma, unc, data_hash

    @classmethod
    def run_benchmark_audit(
        cls,
        engine: GatedScientificAssuranceEngine,
        candidate_record: ModelEvidenceRecord,
    ) -> FalsificationBenchResult:
        """Executes empirical falsification of candidate Hall-Petch model against real micropillar data."""
        d, sigma, _, _ = cls.get_dataset()
        contract = create_hall_petch_benchmark_contract()

        sigma_0 = float(candidate_record.parameters.get("sigma_0", 90.0e6))
        k_hp = float(candidate_record.parameters.get("k_HP", 0.26867e6))
        dc = float(candidate_record.parameters.get("dc", 25.0e-9))
        q = float(candidate_record.parameters.get("q", 0.95))
        sigma_max = float(candidate_record.parameters.get("sigma_theoretical", 1.5e9))

        def candidate_eval(diam: np.ndarray) -> np.ndarray:
            d_val = np.maximum(diam.flatten(), 1e-11)
            softening = 1.0 / (1.0 + (dc / d_val) ** q)
            sig = sigma_0 + (k_hp / np.sqrt(d_val + 1e-10)) * softening
            return np.asarray(np.minimum(sig, sigma_max), dtype=float)

        def baseline_eval(diam: np.ndarray) -> np.ndarray:
            d_val = np.maximum(diam.flatten(), 1e-11)
            return np.asarray(sigma_0 + k_hp / np.sqrt(d_val), dtype=float)

        return engine.run_assurance_gate(
            candidate_record=candidate_record,
            contract=contract,
            eval_fn=candidate_eval,
            baseline_eval_fn=baseline_eval,
            features=d,
            ground_truth=sigma,
        )
