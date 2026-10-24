"""Deterministic Benchmark Corpus Generator for Phase 4 External Tournament.

Generates the 30 balanced benchmark problems (6 problems x 5 classes) with
strict physical separation between public observation data and hidden ground truth.
Enforces seed=42 and generates SHA-256 cryptographic checksums.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

BASE_SEED = 42


def save_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([f"{v:.8g}" for v in r])


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BenchmarkCorpusGenerator:
    """Generates the 30-problem balanced scientific discovery benchmark suite."""

    def __init__(self, root_dir: Path | str, seed: int = BASE_SEED):
        self.root_dir = Path(root_dir)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.public_dir = self.root_dir / "public"
        self.hidden_dir = self.root_dir / "hidden"

    def generate_all(self) -> dict[str, Any]:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.public_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dir.mkdir(parents=True, exist_ok=True)

        with open(self.root_dir / "generation_seed.txt", "w", encoding="utf-8") as f:
            f.write(f"BASE_SEED={self.seed}\nGENERATOR_VERSION=1.0.0\n")

        manifest: dict[str, Any] = {
            "version": "1.0.0",
            "seed": self.seed,
            "num_classes": 5,
            "problems_per_class": 6,
            "total_problems": 30,
            "problems": {},
        }

        generators = self._get_all_problem_generators()
        assert len(generators) == 30, f"Expected 30 generators, got {len(generators)}"

        for prob_id, gen_fn in generators.items():
            prob_info = gen_fn(prob_id)
            manifest["problems"][prob_id] = {
                "problem_id": prob_id,
                "name": prob_info["name"],
                "class_id": prob_info["class_id"],
                "class_name": prob_info["class_name"],
                "oracle_epistemic_class": prob_info["oracle_epistemic_class"],
                "target_variable": prob_info["target_variable"],
                "input_variables": prob_info["input_variables"],
            }

        # Save manifest
        manifest_path = self.root_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Compute checksums
        checksums: dict[str, str] = {}
        for p in sorted(self.root_dir.rglob("*")):
            if p.is_file() and p.name != "checksums.sha256":
                rel = str(p.relative_to(self.root_dir))
                checksums[rel] = compute_sha256(p)

        with open(self.root_dir / "checksums.sha256", "w", encoding="utf-8") as f:
            for rel, h in sorted(checksums.items()):
                f.write(f"{h}  {rel}\n")

        return manifest

    def _save_problem(
        self,
        prob_id: str,
        name: str,
        class_id: int,
        class_name: str,
        oracle_epistemic_class: str,
        expected_state: str,
        inputs: list[str],
        target: str,
        units: dict[str, str],
        train_data: np.ndarray,
        val_data: np.ndarray,
        near_ood_data: np.ndarray,
        far_ood_data: np.ndarray,
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        pub_p = self.public_dir / prob_id
        hid_p = self.hidden_dir / prob_id
        pub_p.mkdir(parents=True, exist_ok=True)
        hid_p.mkdir(parents=True, exist_ok=True)

        header = inputs + [target]
        save_csv(pub_p / "train.csv", header, train_data.tolist())
        save_csv(pub_p / "validation.csv", header, val_data.tolist())
        save_csv(pub_p / "near_ood.csv", header, near_ood_data.tolist())
        save_csv(pub_p / "far_ood.csv", header, far_ood_data.tolist())

        pub_spec = {
            "problem_id": prob_id,
            "problem_name": name,
            "class_id": class_id,
            "class_name": class_name,
            "input_variables": inputs,
            "target_variable": target,
            "variable_units": units,
            "dataset_sizes": {
                "train": len(train_data),
                "validation": len(val_data),
                "near_ood": len(near_ood_data),
                "far_ood": len(far_ood_data),
            },
        }
        with open(pub_p / "problem_spec.json", "w", encoding="utf-8") as f:
            json.dump(pub_spec, f, indent=2)

        hid_gt = {
            "problem_id": prob_id,
            "problem_name": name,
            "class_id": class_id,
            "class_name": class_name,
            "oracle_epistemic_class": oracle_epistemic_class,
            "expected_state": expected_state,
            "ground_truth_formula": ground_truth.get("formula", ""),
            "canonical_structural_family": ground_truth.get("family", ""),
            "true_parameters": ground_truth.get("parameters", {}),
            "competing_hypotheses": ground_truth.get("competing_hypotheses", []),
            "pathology_details": ground_truth.get("pathology_details", {}),
        }
        with open(hid_p / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(hid_gt, f, indent=2)

        return {
            "name": name,
            "class_id": class_id,
            "class_name": class_name,
            "oracle_epistemic_class": oracle_epistemic_class,
            "input_variables": inputs,
            "target_variable": target,
        }

    # --------------------------------------------------------------------------
    # 30 Problem Definitions
    # --------------------------------------------------------------------------
    def _get_all_problem_generators(self) -> dict[str, Callable[[str], dict[str, Any]]]:
        return {
            # Class 1: Canonical Physical Laws (P01 - P06)
            "P01": self._gen_p01_kepler,
            "P02": self._gen_p02_newton_gravity,
            "P03": self._gen_p03_stefan_boltzmann,
            "P04": self._gen_p04_ideal_gas,
            "P05": self._gen_p05_relativistic_energy,
            "P06": self._gen_p06_harmonic_oscillator,
            # Class 2: Observationally Equivalent & Narrow Regimes (P07 - P12)
            "P07": self._gen_p07_heat_capacity_shomate_quad,
            "P08": self._gen_p08_elasticity_hooke_neohooke,
            "P09": self._gen_p09_michaelis_menten_saturation,
            "P10": self._gen_p10_arrhenius_biexponential,
            "P11": self._gen_p11_barometric_linear_exp,
            "P12": self._gen_p12_damped_wave_nearfield,
            # Class 3: Asymptotic & Extrapolation Regimes (P13 - P18)
            "P13": self._gen_p13_relativistic_velocity_addition,
            "P14": self._gen_p14_yukawa_potential,
            "P15": self._gen_p15_nonlinear_oscillator_decay,
            "P16": self._gen_p16_blackbody_rayleigh_jeans,
            "P17": self._gen_p17_fermi_dirac_tail,
            "P18": self._gen_p18_drag_reynolds_plateau,
            # Class 4: Singular & Pathological Regimes (P19 - P24)
            "P19": self._gen_p19_resonance_pole_trap,
            "P20": self._gen_p20_critical_opalescence_pole,
            "P21": self._gen_p21_boundary_layer_singularity,
            "P22": self._gen_p22_runge_polynomial_explosion,
            "P23": self._gen_p23_dimensional_incompatibility_trap,
            "P24": self._gen_p24_collinear_singularity,
            # Class 5: Multivariable Non-Separable & Hybrid Regimes (P25 - P30)
            "P25": self._gen_p25_aerodynamic_mach_compressibility,
            "P26": self._gen_p26_turbulent_dissipation,
            "P27": self._gen_p27_pharmacokinetic_two_compartment,
            "P28": self._gen_p28_quantum_wavepacket_dispersion,
            "P29": self._gen_p29_hall_mhd_drift_wave,
            "P30": self._gen_p30_carreau_yasuda_fluid,
        }

    # --- Class 1: Canonical Physical Laws ---
    def _gen_p01_kepler(self, pid: str) -> dict[str, Any]:
        # T^2 = a^3 -> T = a^1.5
        a_in = np.linspace(0.8, 3.5, 45)
        self.rng.shuffle(a_in)
        a_train, a_val = a_in[:30], a_in[30:]
        a_near = np.linspace(3.6, 5.0, 15)
        a_far = np.linspace(5.1, 8.0, 15)

        noise_train = self.rng.normal(0, 0.01, size=len(a_train))
        noise_val = self.rng.normal(0, 0.01, size=len(a_val))

        d_train = np.column_stack([a_train, a_train**1.5 + noise_train])
        d_val = np.column_stack([a_val, a_val**1.5 + noise_val])
        d_near = np.column_stack([a_near, a_near**1.5])
        d_far = np.column_stack([a_far, a_far**1.5])

        return self._save_problem(
            pid, "Kepler's Third Law", 1, "Canonical Physical Laws", "C1", "UNIQUE_SUPPORT",
            ["a"], "T", {"a": "astronomical_unit", "T": "year"},
            d_train, d_val, d_near, d_far,
            {"formula": "a**1.5", "family": "power_law", "parameters": {"power": 1.5}},
        )

    def _gen_p02_newton_gravity(self, pid: str) -> dict[str, Any]:
        # F = 10.0 / r^2
        r_in = np.linspace(1.5, 5.0, 45)
        self.rng.shuffle(r_in)
        r_train, r_val = r_in[:30], r_in[30:]
        r_near = np.linspace(5.1, 7.5, 15)
        r_far = np.linspace(7.6, 12.0, 15)

        d_train = np.column_stack([r_train, 10.0 / (r_train**2) + self.rng.normal(0, 0.005, len(r_train))])
        d_val = np.column_stack([r_val, 10.0 / (r_val**2) + self.rng.normal(0, 0.005, len(r_val))])
        d_near = np.column_stack([r_near, 10.0 / (r_near**2)])
        d_far = np.column_stack([r_far, 10.0 / (r_far**2)])

        return self._save_problem(
            pid, "Newton's Law of Gravitation", 1, "Canonical Physical Laws", "C1", "UNIQUE_SUPPORT",
            ["r"], "F", {"r": "meter", "F": "newton"},
            d_train, d_val, d_near, d_far,
            {"formula": "10.0 / r**2", "family": "inverse_square", "parameters": {"G_M1_M2": 10.0}},
        )

    def _gen_p03_stefan_boltzmann(self, pid: str) -> dict[str, Any]:
        # P = 2.5e-4 * T^4 (normalized T in [1.0, 3.0])
        t_in = np.linspace(1.0, 3.0, 45)
        self.rng.shuffle(t_in)
        t_train, t_val = t_in[:30], t_in[30:]
        t_near = np.linspace(3.1, 4.2, 15)
        t_far = np.linspace(4.3, 6.0, 15)

        d_train = np.column_stack([t_train, 2.5e-4 * (t_train**4) + self.rng.normal(0, 1e-4, len(t_train))])
        d_val = np.column_stack([t_val, 2.5e-4 * (t_val**4) + self.rng.normal(0, 1e-4, len(t_val))])
        d_near = np.column_stack([t_near, 2.5e-4 * (t_near**4)])
        d_far = np.column_stack([t_far, 2.5e-4 * (t_far**4)])

        return self._save_problem(
            pid, "Stefan-Boltzmann Radiation Law", 1, "Canonical Physical Laws", "C1", "UNIQUE_SUPPORT",
            ["T"], "P", {"T": "kelvin", "P": "watt"},
            d_train, d_val, d_near, d_far,
            {"formula": "2.5e-4 * T**4", "family": "power_law", "parameters": {"sigma_eff": 2.5e-4}},
        )

    def _gen_p04_ideal_gas(self, pid: str) -> dict[str, Any]:
        # P = 8.314 * T / V
        t_grid = np.linspace(280.0, 380.0, 8)
        v_grid = np.linspace(0.02, 0.08, 6)
        tt, vv = np.meshgrid(t_grid, v_grid)
        pts = np.column_stack([tt.flatten(), vv.flatten()])
        self.rng.shuffle(pts)

        pts_train, pts_val = pts[:36], pts[36:]
        # Near OOD: higher T, lower V
        t_near = np.linspace(390.0, 450.0, 4)
        v_near = np.linspace(0.012, 0.018, 4)
        tt_n, vv_n = np.meshgrid(t_near, v_near)
        pts_near = np.column_stack([tt_n.flatten(), vv_n.flatten()])

        # Far OOD: high compression
        t_far = np.linspace(460.0, 550.0, 4)
        v_far = np.linspace(0.005, 0.010, 4)
        tt_f, vv_f = np.meshgrid(t_far, v_far)
        pts_far = np.column_stack([tt_f.flatten(), vv_f.flatten()])

        f = lambda x: 8.314 * x[:, 0] / x[:, 1]
        d_train = np.column_stack([pts_train, f(pts_train) + self.rng.normal(0, 5.0, len(pts_train))])
        d_val = np.column_stack([pts_val, f(pts_val) + self.rng.normal(0, 5.0, len(pts_val))])
        d_near = np.column_stack([pts_near, f(pts_near)])
        d_far = np.column_stack([pts_far, f(pts_far)])

        return self._save_problem(
            pid, "Ideal Gas Law", 1, "Canonical Physical Laws", "C1", "UNIQUE_SUPPORT",
            ["T", "V"], "P", {"T": "kelvin", "V": "meter**3", "P": "pascal"},
            d_train, d_val, d_near, d_far,
            {"formula": "8.314 * T / V", "family": "rational_multivariable", "parameters": {"nR": 8.314}},
        )

    def _gen_p05_relativistic_energy(self, pid: str) -> dict[str, Any]:
        # E = sqrt(p^2 + 1.0)
        p_in = np.linspace(0.1, 3.0, 45)
        self.rng.shuffle(p_in)
        p_train, p_val = p_in[:30], p_in[30:]
        p_near = np.linspace(3.1, 5.0, 15)
        p_far = np.linspace(5.1, 9.0, 15)

        f = lambda p: np.sqrt(p**2 + 1.0)
        d_train = np.column_stack([p_train, f(p_train) + self.rng.normal(0, 0.005, len(p_train))])
        d_val = np.column_stack([p_val, f(p_val) + self.rng.normal(0, 0.005, len(p_val))])
        d_near = np.column_stack([p_near, f(p_near)])
        d_far = np.column_stack([p_far, f(p_far)])

        return self._save_problem(
            pid, "Relativistic Energy-Momentum Relation", 1, "Canonical Physical Laws", "C1", "UNIQUE_SUPPORT",
            ["p"], "E", {"p": "joule_second_per_meter", "E": "joule"},
            d_train, d_val, d_near, d_far,
            {"formula": "sqrt(p**2 + 1.0)", "family": "algebraic_root", "parameters": {"m0_c2": 1.0, "c": 1.0}},
        )

    def _gen_p06_harmonic_oscillator(self, pid: str) -> dict[str, Any]:
        # T = 1.0 * sqrt(m)
        m_in = np.linspace(0.2, 4.0, 45)
        self.rng.shuffle(m_in)
        m_train, m_val = m_in[:30], m_in[30:]
        m_near = np.linspace(4.1, 7.0, 15)
        m_far = np.linspace(7.1, 12.0, 15)

        f = lambda m: np.sqrt(m)
        d_train = np.column_stack([m_train, f(m_train) + self.rng.normal(0, 0.005, len(m_train))])
        d_val = np.column_stack([m_val, f(m_val) + self.rng.normal(0, 0.005, len(m_val))])
        d_near = np.column_stack([m_near, f(m_near)])
        d_far = np.column_stack([m_far, f(m_far)])

        return self._save_problem(
            pid, "Simple Harmonic Oscillator Period", 1, "Canonical Physical Laws", "C1", "UNIQUE_SUPPORT",
            ["m"], "T", {"m": "kilogram", "T": "second"},
            d_train, d_val, d_near, d_far,
            {"formula": "sqrt(m)", "family": "power_law", "parameters": {"coeff": 1.0, "power": 0.5}},
        )

    # --- Class 2: Observationally Equivalent & Narrow Regimes ---
    def _gen_p07_heat_capacity_shomate_quad(self, pid: str) -> dict[str, Any]:
        # T in [0.25, 0.55] normalized
        t_in = np.linspace(0.25, 0.55, 45)
        self.rng.shuffle(t_in)
        t_train, t_val = t_in[:30], t_in[30:]
        t_near = np.linspace(0.60, 1.00, 15)
        t_far = np.linspace(1.10, 1.80, 15)

        # Midpoint of Shomate and quadratic
        y_shom = lambda t: 30.0 + 15.0*t - 5.0*(t**2) + 0.5*(t**3)
        y_quad = lambda t: 30.0 + 14.8*t - 4.5*(t**2)
        f_mid = lambda t: 0.5 * (y_shom(t) + y_quad(t))

        d_train = np.column_stack([t_train, f_mid(t_train) + self.rng.normal(0, 0.01, len(t_train))])
        d_val = np.column_stack([t_val, f_mid(t_val) + self.rng.normal(0, 0.01, len(t_val))])
        d_near = np.column_stack([t_near, y_shom(t_near)])  # True underlying law diverges OOD
        d_far = np.column_stack([t_far, y_shom(t_far)])

        return self._save_problem(
            pid, "Isobaric Heat Capacity (Shomate vs Quadratic)", 2, "Observationally Equivalent Regimes", "C2", "MULTI_HYPOTHESIS",
            ["T"], "Cp", {"T": "kelvin", "Cp": "joule_per_mole_kelvin"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "30.0 + 15.0*T - 5.0*T**2 + 0.5*T**3",
                "family": "shomate_polynomial",
                "competing_hypotheses": [
                    {"id": "shomate", "formula": "30.0 + 15.0*T - 5.0*T**2 + 0.5*T**3"},
                    {"id": "quadratic", "formula": "30.0 + 14.8*T - 4.5*T**2"},
                ],
            },
        )

    def _gen_p08_elasticity_hooke_neohooke(self, pid: str) -> dict[str, Any]:
        # eps in [0.01, 0.05], Hooke: F = 3.0*eps, Neo-Hookean: F = 1.0 * ((1+eps) - (1+eps)**(-2))
        eps_in = np.linspace(0.01, 0.05, 45)
        self.rng.shuffle(eps_in)
        e_train, e_val = eps_in[:30], eps_in[30:]
        e_near = np.linspace(0.10, 0.40, 15)  # Divergence zone
        e_far = np.linspace(0.50, 1.20, 15)

        f_neo = lambda e: (1.0 + e) - (1.0 + e)**(-2)
        f_lin = lambda e: 3.0 * e
        f_mid = lambda e: 0.5 * (f_neo(e) + f_lin(e))

        d_train = np.column_stack([e_train, f_mid(e_train) + self.rng.normal(0, 0.0005, len(e_train))])
        d_val = np.column_stack([e_val, f_mid(e_val) + self.rng.normal(0, 0.0005, len(e_val))])
        d_near = np.column_stack([e_near, f_neo(e_near)])
        d_far = np.column_stack([e_far, f_neo(e_far)])

        return self._save_problem(
            pid, "Linear Hookean vs Nonlinear Neo-Hookean Elasticity", 2, "Observationally Equivalent Regimes", "C2", "MULTI_HYPOTHESIS",
            ["strain"], "stress", {"strain": "dimensionless", "stress": "pascal"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "(1+strain) - (1+strain)**(-2)",
                "family": "neo_hookean",
                "competing_hypotheses": [
                    {"id": "hookean_linear", "formula": "3.0*strain"},
                    {"id": "neo_hookean", "formula": "(1+strain) - (1+strain)**(-2)"},
                ],
            },
        )

    def _gen_p09_michaelis_menten_saturation(self, pid: str) -> dict[str, Any]:
        # High substrate regime S in [20, 60] mM where v = Vmax*S/(Km + S) with Vmax=10.0, Km=1.0
        # In this saturation zone, v ~ 10.0 - 10.0/S vs affine v ~ 9.5 + 0.01*S
        s_in = np.linspace(20.0, 60.0, 45)
        self.rng.shuffle(s_in)
        s_train, s_val = s_in[:30], s_in[30:]
        s_near = np.linspace(1.0, 15.0, 15)  # Low substrate where models strongly separate
        s_far = np.linspace(0.1, 0.9, 15)

        f_true = lambda s: 10.0 * s / (1.0 + s)
        f_affine = lambda s: 9.6 + 0.007 * s
        f_mid = lambda s: 0.5 * (f_true(s) + f_affine(s))

        d_train = np.column_stack([s_train, f_mid(s_train) + self.rng.normal(0, 0.01, len(s_train))])
        d_val = np.column_stack([s_val, f_mid(s_val) + self.rng.normal(0, 0.01, len(s_val))])
        d_near = np.column_stack([s_near, f_true(s_near)])
        d_far = np.column_stack([s_far, f_true(s_far)])

        return self._save_problem(
            pid, "Michaelis-Menten Kinetics in Substrate Saturation", 2, "Observationally Equivalent Regimes", "C2", "MULTI_HYPOTHESIS",
            ["S"], "v", {"S": "mole_per_meter**3", "v": "mole_per_meter**3_second"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "10.0 * S / (1.0 + S)",
                "family": "rational_langmuir",
                "competing_hypotheses": [
                    {"id": "michaelis_menten", "formula": "10.0*S / (1.0 + S)"},
                    {"id": "saturation_linear", "formula": "9.6 + 0.007*S"},
                ],
            },
        )

    def _gen_p10_arrhenius_biexponential(self, pid: str) -> dict[str, Any]:
        # Narrow temp window T in [295, 310] K.
        # Arrhenius: k = A*exp(-Ea/RT). Over delta T = 15K, linear & exponential fit equally.
        t_in = np.linspace(295.0, 310.0, 45)
        self.rng.shuffle(t_in)
        t_train, t_val = t_in[:30], t_in[30:]
        t_near = np.linspace(315.0, 350.0, 15)
        t_far = np.linspace(360.0, 450.0, 15)

        # Normalized: k = exp(-2.0 / (T/300))
        f_arr = lambda t: np.exp(-2.0 / (t / 300.0))
        # Tangent line at 302.5K
        t0 = 302.5
        k0 = f_arr(t0)
        dk = k0 * (2.0 * 300.0 / (t0**2))
        f_lin = lambda t: k0 + dk * (t - t0)
        f_mid = lambda t: 0.5 * (f_arr(t) + f_lin(t))

        d_train = np.column_stack([t_train, f_mid(t_train) + self.rng.normal(0, 0.001, len(t_train))])
        d_val = np.column_stack([t_val, f_mid(t_val) + self.rng.normal(0, 0.001, len(t_val))])
        d_near = np.column_stack([t_near, f_arr(t_near)])
        d_far = np.column_stack([t_far, f_arr(t_far)])

        return self._save_problem(
            pid, "Arrhenius Reaction Rate in Narrow Temperature Band", 2, "Observationally Equivalent Regimes", "C2", "MULTI_HYPOTHESIS",
            ["T"], "k", {"T": "kelvin", "k": "per_second"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "exp(-600.0 / T)",
                "family": "arrhenius_exponential",
                "competing_hypotheses": [
                    {"id": "arrhenius", "formula": "exp(-600.0 / T)"},
                    {"id": "linear_approximation", "formula": f"{k0:.5g} + {dk:.5g}*(T - 302.5)"},
                ],
            },
        )

    def _gen_p11_barometric_linear_exp(self, pid: str) -> dict[str, Any]:
        # Planetary altitude z in [0, 400] m with scale height H = 8000 m.
        # rho = rho0 * exp(-z/8000) vs linearized rho0 * (1 - z/8000).
        z_in = np.linspace(0.0, 400.0, 45)
        self.rng.shuffle(z_in)
        z_train, z_val = z_in[:30], z_in[30:]
        z_near = np.linspace(1000.0, 4000.0, 15)  # Separation
        z_far = np.linspace(5000.0, 15000.0, 15)

        f_exp = lambda z: 1.225 * np.exp(-z / 8000.0)
        f_lin = lambda z: 1.225 * (1.0 - z / 8000.0)
        f_mid = lambda z: 0.5 * (f_exp(z) + f_lin(z))

        d_train = np.column_stack([z_train, f_mid(z_train) + self.rng.normal(0, 0.0002, len(z_train))])
        d_val = np.column_stack([z_val, f_mid(z_val) + self.rng.normal(0, 0.0002, len(z_val))])
        d_near = np.column_stack([z_near, f_exp(z_near)])
        d_far = np.column_stack([z_far, f_exp(z_far)])

        return self._save_problem(
            pid, "Atmospheric Barometric Density", 2, "Observationally Equivalent Regimes", "C2", "MULTI_HYPOTHESIS",
            ["z"], "rho", {"z": "meter", "rho": "kilogram_per_meter**3"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "1.225 * exp(-z / 8000.0)",
                "family": "exponential_decay",
                "competing_hypotheses": [
                    {"id": "barometric_exponential", "formula": "1.225 * exp(-z / 8000.0)"},
                    {"id": "linear_truncation", "formula": "1.225 - 0.000153125 * z"},
                ],
            },
        )

    def _gen_p12_damped_wave_nearfield(self, pid: str) -> dict[str, Any]:
        # y = exp(-0.5*x)*cos(1.2*x) on x in [0.02, 0.20]. Quadratic Taylor series matches.
        x_in = np.linspace(0.02, 0.20, 45)
        self.rng.shuffle(x_in)
        x_train, x_val = x_in[:30], x_in[30:]
        x_near = np.linspace(0.5, 2.0, 15)
        x_far = np.linspace(2.5, 6.0, 15)

        f_true = lambda x: np.exp(-0.5 * x) * np.cos(1.2 * x)
        # Taylor: 1 - 0.5*x - 0.595*x**2
        f_tay = lambda x: 1.0 - 0.5 * x - 0.595 * (x**2)
        f_mid = lambda x: 0.5 * (f_true(x) + f_tay(x))

        d_train = np.column_stack([x_train, f_mid(x_train) + self.rng.normal(0, 0.0005, len(x_train))])
        d_val = np.column_stack([x_val, f_mid(x_val) + self.rng.normal(0, 0.0005, len(x_val))])
        d_near = np.column_stack([x_near, f_true(x_near)])
        d_far = np.column_stack([x_far, f_true(x_far)])

        return self._save_problem(
            pid, "Damped Wave Near-Field Amplitude", 2, "Observationally Equivalent Regimes", "C2", "MULTI_HYPOTHESIS",
            ["x"], "y", {"x": "meter", "y": "meter"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "exp(-0.5*x) * cos(1.2*x)",
                "family": "damped_oscillation",
                "competing_hypotheses": [
                    {"id": "true_damped_wave", "formula": "exp(-0.5*x) * cos(1.2*x)"},
                    {"id": "quadratic_taylor", "formula": "1.0 - 0.5*x - 0.595*x**2"},
                ],
            },
        )

    # --- Class 3: Asymptotic & Extrapolation Regimes ---
    def _gen_p13_relativistic_velocity_addition(self, pid: str) -> dict[str, Any]:
        # v = (u + w)/(1 + u*w) with c=1. Sampled in [0.05, 0.45], tested as u, w -> 0.95
        u = np.linspace(0.05, 0.45, 7)
        w = np.linspace(0.05, 0.45, 7)
        uu, ww = np.meshgrid(u, w)
        pts = np.column_stack([uu.flatten(), ww.flatten()])
        self.rng.shuffle(pts)
        pts_train, pts_val = pts[:35], pts[35:]

        u_near = np.linspace(0.50, 0.75, 4)
        w_near = np.linspace(0.50, 0.75, 4)
        un, wn = np.meshgrid(u_near, w_near)
        pts_near = np.column_stack([un.flatten(), wn.flatten()])

        u_far = np.linspace(0.80, 0.95, 4)
        w_far = np.linspace(0.80, 0.95, 4)
        uf, wf = np.meshgrid(u_far, w_far)
        pts_far = np.column_stack([uf.flatten(), wf.flatten()])

        f = lambda p: (p[:, 0] + p[:, 1]) / (1.0 + p[:, 0] * p[:, 1])
        d_train = np.column_stack([pts_train, f(pts_train) + self.rng.normal(0, 0.002, len(pts_train))])
        d_val = np.column_stack([pts_val, f(pts_val) + self.rng.normal(0, 0.002, len(pts_val))])
        d_near = np.column_stack([pts_near, f(pts_near)])
        d_far = np.column_stack([pts_far, f(pts_far)])

        return self._save_problem(
            pid, "Relativistic Velocity Addition", 3, "Asymptotic & Extrapolation Regimes", "C1", "UNIQUE_SUPPORT",
            ["u", "w"], "v", {"u": "meter_per_second", "w": "meter_per_second", "v": "meter_per_second"},
            d_train, d_val, d_near, d_far,
            {"formula": "(u + w) / (1.0 + u*w)", "family": "rational_asymptotic", "parameters": {"c": 1.0}},
        )

    def _gen_p14_yukawa_potential(self, pid: str) -> dict[str, Any]:
        # V(r) = 5.0 * exp(-0.8*r) / r
        r_in = np.linspace(0.5, 2.5, 45)
        self.rng.shuffle(r_in)
        r_train, r_val = r_in[:30], r_in[30:]
        r_near = np.linspace(2.6, 5.0, 15)
        r_far = np.linspace(5.1, 10.0, 15)

        f = lambda r: 5.0 * np.exp(-0.8 * r) / r
        d_train = np.column_stack([r_train, f(r_train) + self.rng.normal(0, 0.005, len(r_train))])
        d_val = np.column_stack([r_val, f(r_val) + self.rng.normal(0, 0.005, len(r_val))])
        d_near = np.column_stack([r_near, f(r_near)])
        d_far = np.column_stack([r_far, f(r_far)])

        return self._save_problem(
            pid, "Yukawa Screened Potential", 3, "Asymptotic & Extrapolation Regimes", "C1", "UNIQUE_SUPPORT",
            ["r"], "V", {"r": "meter", "V": "joule"},
            d_train, d_val, d_near, d_far,
            {"formula": "5.0 * exp(-0.8*r) / r", "family": "screened_coulomb", "parameters": {"g2": 5.0, "mu": 0.8}},
        )

    def _gen_p15_nonlinear_oscillator_decay(self, pid: str) -> dict[str, Any]:
        # x(t) = 4.0 / (1.0 + 0.5*t)
        t_in = np.linspace(0.0, 3.0, 45)
        self.rng.shuffle(t_in)
        t_train, t_val = t_in[:30], t_in[30:]
        t_near = np.linspace(3.5, 8.0, 15)
        t_far = np.linspace(9.0, 25.0, 15)

        f = lambda t: 4.0 / (1.0 + 0.5 * t)
        d_train = np.column_stack([t_train, f(t_train) + self.rng.normal(0, 0.01, len(t_train))])
        d_val = np.column_stack([t_val, f(t_val) + self.rng.normal(0, 0.01, len(t_val))])
        d_near = np.column_stack([t_near, f(t_near)])
        d_far = np.column_stack([t_far, f(t_far)])

        return self._save_problem(
            pid, "Nonlinear Oscillator Algebraic Decay", 3, "Asymptotic & Extrapolation Regimes", "C1", "UNIQUE_SUPPORT",
            ["t"], "x", {"t": "second", "x": "meter"},
            d_train, d_val, d_near, d_far,
            {"formula": "4.0 / (1.0 + 0.5*t)", "family": "rational_decay", "parameters": {"A": 4.0, "lambda": 0.5}},
        )

    def _gen_p16_blackbody_rayleigh_jeans(self, pid: str) -> dict[str, Any]:
        # Planck spectral density u(nu) = 2.0 * nu**3 / (exp(0.5*nu) - 1.0)
        nu_in = np.linspace(0.4, 3.0, 45)
        self.rng.shuffle(nu_in)
        nu_train, nu_val = nu_in[:30], nu_in[30:]
        nu_near = np.linspace(3.1, 6.0, 15)  # Wien peak
        nu_far = np.linspace(6.5, 12.0, 15)  # Exponential Wien tail

        f = lambda nu: 2.0 * (nu**3) / (np.exp(0.5 * nu) - 1.0)
        d_train = np.column_stack([nu_train, f(nu_train) + self.rng.normal(0, 0.005, len(nu_train))])
        d_val = np.column_stack([nu_val, f(nu_val) + self.rng.normal(0, 0.005, len(nu_val))])
        d_near = np.column_stack([nu_near, f(nu_near)])
        d_far = np.column_stack([nu_far, f(nu_far)])

        return self._save_problem(
            pid, "Planck Blackbody Radiation Spectrum", 3, "Asymptotic & Extrapolation Regimes", "C1", "UNIQUE_SUPPORT",
            ["nu"], "u", {"nu": "per_second", "u": "joule_second_per_meter**3"},
            d_train, d_val, d_near, d_far,
            {"formula": "2.0 * nu**3 / (exp(0.5*nu) - 1.0)", "family": "planck_distribution", "parameters": {"A": 2.0, "beta": 0.5}},
        )

    def _gen_p17_fermi_dirac_tail(self, pid: str) -> dict[str, Any]:
        # f(E) = 1.0 / (exp(E - 2.0) + 1.0)
        e_in = np.linspace(0.5, 3.5, 45)
        self.rng.shuffle(e_in)
        e_train, e_val = e_in[:30], e_in[30:]
        e_near = np.linspace(3.6, 6.0, 15)  # Maxwell-Boltzmann transition
        e_far = np.linspace(6.1, 10.0, 15)  # Asymptotic zero tail

        f = lambda e: 1.0 / (np.exp(e - 2.0) + 1.0)
        d_train = np.column_stack([e_train, f(e_train) + self.rng.normal(0, 0.002, len(e_train))])
        d_val = np.column_stack([e_val, f(e_val) + self.rng.normal(0, 0.002, len(e_val))])
        d_near = np.column_stack([e_near, f(e_near)])
        d_far = np.column_stack([e_far, f(e_far)])

        return self._save_problem(
            pid, "Fermi-Dirac Distribution High-Energy Tail", 3, "Asymptotic & Extrapolation Regimes", "C1", "UNIQUE_SUPPORT",
            ["E"], "f", {"E": "electron_volt", "f": "dimensionless"},
            d_train, d_val, d_near, d_far,
            {"formula": "1.0 / (exp(E - 2.0) + 1.0)", "family": "fermi_dirac", "parameters": {"Ef": 2.0}},
        )

    def _gen_p18_drag_reynolds_plateau(self, pid: str) -> dict[str, Any]:
        # Cd = 0.44 + 24.0 / Re (Re in [5.0, 50.0]) -> asymptotes to 0.44
        re_in = np.linspace(5.0, 50.0, 45)
        self.rng.shuffle(re_in)
        re_train, re_val = re_in[:30], re_in[30:]
        re_near = np.linspace(60.0, 200.0, 15)
        re_far = np.linspace(250.0, 1000.0, 15)

        f = lambda r: 0.44 + 24.0 / r
        d_train = np.column_stack([re_train, f(re_train) + self.rng.normal(0, 0.01, len(re_train))])
        d_val = np.column_stack([re_val, f(re_val) + self.rng.normal(0, 0.01, len(re_val))])
        d_near = np.column_stack([re_near, f(re_near)])
        d_far = np.column_stack([re_far, f(re_far)])

        return self._save_problem(
            pid, "High-Reynolds Number Drag Coefficient Plateau", 3, "Asymptotic & Extrapolation Regimes", "C1", "UNIQUE_SUPPORT",
            ["Re"], "Cd", {"Re": "dimensionless", "Cd": "dimensionless"},
            d_train, d_val, d_near, d_far,
            {"formula": "0.44 + 24.0 / Re", "family": "drag_curve", "parameters": {"Cd_inf": 0.44, "stokes": 24.0}},
        )

    # --- Class 4: Singular & Pathological Regimes ---
    def _gen_p19_resonance_pole_trap(self, pid: str) -> dict[str, Any]:
        # Pole at x=2.5001 strictly inside evaluation domain [1.0, 5.0], but avoided on sample grid
        # Sample points chosen deterministically away from pole
        x_train = np.array([1.0, 1.4, 1.8, 2.1, 2.9, 3.2, 3.6, 4.0, 4.5, 5.0])
        x_val = np.array([1.2, 1.6, 2.0, 3.0, 3.4, 3.8, 4.2, 4.8])
        x_near = np.array([5.2, 5.5, 5.8, 6.2, 6.5])
        x_far = np.array([7.0, 8.0, 9.0, 10.0])

        # Target generated from a smooth function y = 2.0*x + 1.0, but pathological candidate 2.0*x + 1.0/(x - 2.5001) has tiny error on grid
        f = lambda x: 2.0 * x + 1.0
        d_train = np.column_stack([x_train, f(x_train)])
        d_val = np.column_stack([x_val, f(x_val)])
        d_near = np.column_stack([x_near, f(x_near)])
        d_far = np.column_stack([x_far, f(x_far)])

        return self._save_problem(
            pid, "Resonance Pole Avoidance Trap", 4, "Singular & Pathological Regimes", "C3", "INSUFFICIENT_EVIDENCE",
            ["x"], "y", {"x": "meter", "y": "meter"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "2.0*x + 1.0",
                "family": "linear",
                "pathology_details": {
                    "trap_type": "POLE_INSIDE_DOMAIN",
                    "pathological_candidate": "2.0*x + 1.0 / (x - 2.5001)",
                    "pole_location": 2.5001,
                    "domain": [1.0, 5.0],
                },
            },
        )

    def _gen_p20_critical_opalescence_pole(self, pid: str) -> dict[str, Any]:
        # Divergence chi = 1.5 / (T - 300.0)**1.2 as T -> 300.0 from above
        t_in = np.linspace(301.0, 350.0, 45)
        self.rng.shuffle(t_in)
        t_train, t_val = t_in[:30], t_in[30:]
        t_near = np.linspace(355.0, 400.0, 15)
        t_far = np.linspace(410.0, 550.0, 15)

        f = lambda t: 1.5 / ((t - 300.0)**1.2)
        d_train = np.column_stack([t_train, f(t_train) + self.rng.normal(0, 0.002, len(t_train))])
        d_val = np.column_stack([t_val, f(t_val) + self.rng.normal(0, 0.002, len(t_val))])
        d_near = np.column_stack([t_near, f(t_near)])
        d_far = np.column_stack([t_far, f(t_far)])

        return self._save_problem(
            pid, "Critical Opalescence Susceptibility Divergence", 4, "Singular & Pathological Regimes", "C3", "INSUFFICIENT_EVIDENCE",
            ["T"], "chi", {"T": "kelvin", "chi": "meter**3_per_joule"},
            d_train, d_val, d_near, d_far,
            {"formula": "1.5 / ((T - 300.0)**1.2)", "family": "critical_power_law", "pathology_details": {"critical_temperature": 300.0}},
        )

    def _gen_p21_boundary_layer_singularity(self, pid: str) -> dict[str, Any]:
        # y = 1.0 / sqrt(x) singular at leading edge x=0
        x_in = np.linspace(0.05, 2.0, 45)
        self.rng.shuffle(x_in)
        x_train, x_val = x_in[:30], x_in[30:]
        x_near = np.linspace(2.1, 4.0, 15)
        x_far = np.linspace(4.5, 8.0, 15)

        f = lambda x: 1.0 / np.sqrt(x)
        d_train = np.column_stack([x_train, f(x_train) + self.rng.normal(0, 0.005, len(x_train))])
        d_val = np.column_stack([x_val, f(x_val) + self.rng.normal(0, 0.005, len(x_val))])
        d_near = np.column_stack([x_near, f(x_near)])
        d_far = np.column_stack([x_far, f(x_far)])

        return self._save_problem(
            pid, "Viscous Boundary Layer Leading Edge Singularity", 4, "Singular & Pathological Regimes", "C3", "INSUFFICIENT_EVIDENCE",
            ["x"], "tau", {"x": "meter", "tau": "pascal"},
            d_train, d_val, d_near, d_far,
            {"formula": "1.0 / sqrt(x)", "family": "inverse_root", "pathology_details": {"singularity_at": 0.0}},
        )

    def _gen_p22_runge_polynomial_explosion(self, pid: str) -> dict[str, Any]:
        # Runge function y = 1 / (1 + 25*x^2) on [-1, 1]. High-degree polynomial explodes near ends.
        x_in = np.linspace(-0.85, 0.85, 45)
        self.rng.shuffle(x_in)
        x_train, x_val = x_in[:30], x_in[30:]
        x_near = np.linspace(0.86, 0.99, 15)
        x_far = np.linspace(1.05, 1.50, 15)

        f = lambda x: 1.0 / (1.0 + 25.0 * (x**2))
        d_train = np.column_stack([x_train, f(x_train) + self.rng.normal(0, 0.005, len(x_train))])
        d_val = np.column_stack([x_val, f(x_val) + self.rng.normal(0, 0.005, len(x_val))])
        d_near = np.column_stack([x_near, f(x_near)])
        d_far = np.column_stack([x_far, f(x_far)])

        return self._save_problem(
            pid, "Runge Polynomial Boundary Oscillation", 4, "Singular & Pathological Regimes", "C1", "UNIQUE_SUPPORT",
            ["x"], "y", {"x": "dimensionless", "y": "dimensionless"},
            d_train, d_val, d_near, d_far,
            {"formula": "1.0 / (1.0 + 25.0 * x**2)", "family": "rational_cauchy", "pathology_details": {"runge_susceptible": True}},
        )

    def _gen_p23_dimensional_incompatibility_trap(self, pid: str) -> dict[str, Any]:
        # T (time [s]) and x (length [m]). Target y is velocity [m/s].
        # Incompatible sum candidate: y = T + x (invalid dimensions) fits numerical grid by accident.
        t_in = np.linspace(1.0, 5.0, 45)
        x_in = 2.0 * t_in + self.rng.normal(0, 0.05, 45)
        # Ground truth law is v = x / T = 2.0
        v_true = 2.0
        d_train = np.column_stack([t_in[:30], x_in[:30], np.full(30, v_true) + self.rng.normal(0, 0.01, 30)])
        d_val = np.column_stack([t_in[30:], x_in[30:], np.full(15, v_true) + self.rng.normal(0, 0.01, 15)])
        d_near = np.column_stack([np.linspace(5.5, 8.0, 15), 2.0*np.linspace(5.5, 8.0, 15), np.full(15, v_true)])
        d_far = np.column_stack([np.linspace(9.0, 15.0, 15), 2.0*np.linspace(9.0, 15.0, 15), np.full(15, v_true)])

        return self._save_problem(
            pid, "Dimensional Incompatibility Trap", 4, "Singular & Pathological Regimes", "C3", "INSUFFICIENT_EVIDENCE",
            ["T", "x"], "v", {"T": "second", "x": "meter", "v": "meter_per_second"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "2.0",
                "family": "constant_ratio",
                "pathology_details": {
                    "trap_candidate": "T + x",
                    "incompatibility": "[second] + [meter]",
                    "expected_stage_a": "INVALID",
                },
            },
        )

    def _gen_p24_collinear_singularity(self, pid: str) -> dict[str, Any]:
        # y = exp(x). Collinear candidate c1*exp(x) + c2*exp(x + 1e-5) has cond(J) > 1e8
        x_in = np.linspace(0.1, 2.0, 45)
        self.rng.shuffle(x_in)
        x_train, x_val = x_in[:30], x_in[30:]
        x_near = np.linspace(2.1, 3.2, 15)
        x_far = np.linspace(3.5, 5.0, 15)

        f = lambda x: np.exp(x)
        d_train = np.column_stack([x_train, f(x_train) + self.rng.normal(0, 0.01, len(x_train))])
        d_val = np.column_stack([x_val, f(x_val) + self.rng.normal(0, 0.01, len(x_val))])
        d_near = np.column_stack([x_near, f(x_near)])
        d_far = np.column_stack([x_far, f(x_far)])

        return self._save_problem(
            pid, "Multi-Parameter Collinear Singularity Trap", 4, "Singular & Pathological Regimes", "C3", "INSUFFICIENT_EVIDENCE",
            ["x"], "y", {"x": "dimensionless", "y": "dimensionless"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "exp(x)",
                "family": "exponential",
                "pathology_details": {
                    "collinear_candidate": "c1*exp(x) + c2*exp(x + 1e-5)",
                    "expected_cond_number": 1e8,
                    "expected_stage_b": "REJECT_UNIDENTIFIABLE",
                },
            },
        )

    # --- Class 5: Multivariable Non-Separable & Hybrid Regimes ---
    def _gen_p25_aerodynamic_mach_compressibility(self, pid: str) -> dict[str, Any]:
        # Cd(M, alpha) = 0.02 + 0.15*alpha**2 + transonic_nn_rise(M)
        # Transonic rise is non-symbolic smooth tanh bump: 0.08 / (1 + exp(-20*(M - 0.95)))
        m_grid = np.linspace(0.3, 0.9, 7)
        a_grid = np.linspace(0.0, 0.2, 7)
        mm, aa = np.meshgrid(m_grid, a_grid)
        pts = np.column_stack([mm.flatten(), aa.flatten()])
        self.rng.shuffle(pts)
        pts_train, pts_val = pts[:35], pts[35:]

        m_near = np.linspace(0.92, 1.05, 4)
        a_near = np.linspace(0.0, 0.2, 4)
        mn, an = np.meshgrid(m_near, a_near)
        pts_near = np.column_stack([mn.flatten(), an.flatten()])

        m_far = np.linspace(1.10, 1.40, 4)
        a_far = np.linspace(0.0, 0.2, 4)
        mf, af = np.meshgrid(m_far, a_far)
        pts_far = np.column_stack([mf.flatten(), af.flatten()])

        f = lambda p: 0.02 + 0.15 * (p[:, 1]**2) + 0.08 / (1.0 + np.exp(-20.0 * (p[:, 0] - 0.95)))
        d_train = np.column_stack([pts_train, f(pts_train) + self.rng.normal(0, 0.001, len(pts_train))])
        d_val = np.column_stack([pts_val, f(pts_val) + self.rng.normal(0, 0.001, len(pts_val))])
        d_near = np.column_stack([pts_near, f(pts_near)])
        d_far = np.column_stack([pts_far, f(pts_far)])

        return self._save_problem(
            pid, "Aerodynamic Drag with Transonic Mach Rise", 5, "Multivariable Non-Separable & Hybrid Regimes", "C5", "MISSING_STRUCTURE",
            ["M", "alpha"], "Cd", {"M": "dimensionless", "alpha": "radian", "Cd": "dimensionless"},
            d_train, d_val, d_near, d_far,
            {
                "formula": "0.02 + 0.15*alpha**2 + Delta_NN(M)",
                "family": "hybrid_physics_residual",
                "base_physics": "0.02 + 0.15*alpha**2",
                "missing_component": "Transonic compressibility drag rise",
            },
        )

    def _gen_p26_turbulent_dissipation(self, pid: str) -> dict[str, Any]:
        # epsilon = 0.09 * (k**1.5) / L
        k_grid = np.linspace(0.5, 3.0, 7)
        l_grid = np.linspace(0.2, 1.5, 7)
        kk, ll = np.meshgrid(k_grid, l_grid)
        pts = np.column_stack([kk.flatten(), ll.flatten()])
        self.rng.shuffle(pts)
        pts_train, pts_val = pts[:35], pts[35:]

        k_near = np.linspace(3.2, 5.0, 4)
        l_near = np.linspace(0.1, 0.18, 4)
        kn, ln = np.meshgrid(k_near, l_near)
        pts_near = np.column_stack([kn.flatten(), ln.flatten()])

        k_far = np.linspace(5.5, 8.0, 4)
        l_far = np.linspace(0.05, 0.09, 4)
        kf, lf = np.meshgrid(k_far, l_far)
        pts_far = np.column_stack([kf.flatten(), lf.flatten()])

        f = lambda p: 0.09 * (p[:, 0]**1.5) / p[:, 1]
        d_train = np.column_stack([pts_train, f(pts_train) + self.rng.normal(0, 0.01, len(pts_train))])
        d_val = np.column_stack([pts_val, f(pts_val) + self.rng.normal(0, 0.01, len(pts_val))])
        d_near = np.column_stack([pts_near, f(pts_near)])
        d_far = np.column_stack([pts_far, f(pts_far)])

        return self._save_problem(
            pid, "Turbulent Kinetic Energy Dissipation Rate", 5, "Multivariable Non-Separable & Hybrid Regimes", "C1", "UNIQUE_SUPPORT",
            ["k", "L"], "eps", {"k": "meter**2_per_second**2", "L": "meter", "eps": "meter**2_per_second**3"},
            d_train, d_val, d_near, d_far,
            {"formula": "0.09 * (k**1.5) / L", "family": "turbulence_closure", "parameters": {"Cmu": 0.09}},
        )

    def _gen_p27_pharmacokinetic_two_compartment(self, pid: str) -> dict[str, Any]:
        # C(t) = 5.0 * exp(-1.2*t) + 2.0 * exp(-0.15*t)
        t_in = np.linspace(0.1, 6.0, 45)
        self.rng.shuffle(t_in)
        t_train, t_val = t_in[:30], t_in[30:]
        t_near = np.linspace(6.5, 12.0, 15)
        t_far = np.linspace(13.0, 24.0, 15)

        f = lambda t: 5.0 * np.exp(-1.2 * t) + 2.0 * np.exp(-0.15 * t)
        d_train = np.column_stack([t_train, f(t_train) + self.rng.normal(0, 0.01, len(t_train))])
        d_val = np.column_stack([t_val, f(t_val) + self.rng.normal(0, 0.01, len(t_val))])
        d_near = np.column_stack([t_near, f(t_near)])
        d_far = np.column_stack([t_far, f(t_far)])

        return self._save_problem(
            pid, "Two-Compartment Pharmacokinetic Clearance", 5, "Multivariable Non-Separable & Hybrid Regimes", "C1", "UNIQUE_SUPPORT",
            ["t"], "C", {"t": "hour", "C": "milligram_per_liter"},
            d_train, d_val, d_near, d_far,
            {"formula": "5.0*exp(-1.2*t) + 2.0*exp(-0.15*t)", "family": "biexponential", "parameters": {"A": 5.0, "alpha": 1.2, "B": 2.0, "beta": 0.15}},
        )

    def _gen_p28_quantum_wavepacket_dispersion(self, pid: str) -> dict[str, Any]:
        # psi(x, t) = exp(-0.2*t) * exp(-(x - 1.5*t)**2 / 2.0)
        x_grid = np.linspace(0.0, 5.0, 7)
        t_grid = np.linspace(0.0, 2.5, 7)
        xx, tt = np.meshgrid(x_grid, t_grid)
        pts = np.column_stack([xx.flatten(), tt.flatten()])
        self.rng.shuffle(pts)
        pts_train, pts_val = pts[:35], pts[35:]

        x_near = np.linspace(5.2, 8.0, 4)
        t_near = np.linspace(2.6, 4.0, 4)
        xn, tn = np.meshgrid(x_near, t_near)
        pts_near = np.column_stack([xn.flatten(), tn.flatten()])

        x_far = np.linspace(8.5, 12.0, 4)
        t_far = np.linspace(4.5, 6.0, 4)
        xf, tf = np.meshgrid(x_far, t_far)
        pts_far = np.column_stack([xf.flatten(), tf.flatten()])

        f = lambda p: np.exp(-0.2 * p[:, 1]) * np.exp(-((p[:, 0] - 1.5 * p[:, 1])**2) / 2.0)
        d_train = np.column_stack([pts_train, f(pts_train) + self.rng.normal(0, 0.005, len(pts_train))])
        d_val = np.column_stack([pts_val, f(pts_val) + self.rng.normal(0, 0.005, len(pts_val))])
        d_near = np.column_stack([pts_near, f(pts_near)])
        d_far = np.column_stack([pts_far, f(pts_far)])

        return self._save_problem(
            pid, "Non-Hermitian Quantum Damped Wavepacket Dispersion", 5, "Multivariable Non-Separable & Hybrid Regimes", "C5", "MISSING_STRUCTURE",
            ["x", "t"], "psi", {"x": "meter", "t": "second", "psi": "dimensionless"},
            d_train, d_val, d_near, d_far,
            {"formula": "exp(-0.2*t) * exp(-(x - 1.5*t)**2 / 2.0)", "family": "gaussian_soliton_decay", "parameters": {"gamma": 0.2, "v": 1.5, "sigma2": 2.0}},
        )

    def _gen_p29_hall_mhd_drift_wave(self, pid: str) -> dict[str, Any]:
        # omega(k_perp, k_par) = 2.0*k_par / (1.0 + 0.8*k_perp**2)
        kp_grid = np.linspace(0.2, 2.0, 7)
        kz_grid = np.linspace(0.5, 3.0, 7)
        kpp, kzz = np.meshgrid(kp_grid, kz_grid)
        pts = np.column_stack([kpp.flatten(), kzz.flatten()])
        self.rng.shuffle(pts)
        pts_train, pts_val = pts[:35], pts[35:]

        kp_near = np.linspace(2.2, 4.0, 4)
        kz_near = np.linspace(3.2, 5.0, 4)
        kpn, kzn = np.meshgrid(kp_near, kz_near)
        pts_near = np.column_stack([kpn.flatten(), kzn.flatten()])

        kp_far = np.linspace(4.5, 8.0, 4)
        kz_far = np.linspace(5.5, 9.0, 4)
        kpf, kzf = np.meshgrid(kp_far, kz_far)
        pts_far = np.column_stack([kpf.flatten(), kzf.flatten()])

        f = lambda p: 2.0 * p[:, 1] / (1.0 + 0.8 * (p[:, 0]**2))
        d_train = np.column_stack([pts_train, f(pts_train) + self.rng.normal(0, 0.01, len(pts_train))])
        d_val = np.column_stack([pts_val, f(pts_val) + self.rng.normal(0, 0.01, len(pts_val))])
        d_near = np.column_stack([pts_near, f(pts_near)])
        d_far = np.column_stack([pts_far, f(pts_far)])

        return self._save_problem(
            pid, "Plasma Hall-MHD Coupled Drift Wave Velocity", 5, "Multivariable Non-Separable & Hybrid Regimes", "C1", "UNIQUE_SUPPORT",
            ["k_perp", "k_par"], "omega", {"k_perp": "per_meter", "k_par": "per_meter", "omega": "radian_per_second"},
            d_train, d_val, d_near, d_far,
            {"formula": "2.0 * k_par / (1.0 + 0.8 * k_perp**2)", "family": "drift_wave_dispersion", "parameters": {"v_alfven": 2.0, "rho_s2": 0.8}},
        )

    def _gen_p30_carreau_yasuda_fluid(self, pid: str) -> dict[str, Any]:
        # eta(gamma_dot) = 0.05 + (1.0 - 0.05) / (1.0 + (0.5 * gamma_dot)**2)**0.3
        # Non-Newtonian shear-thinning polymer viscosity
        g_in = np.linspace(0.1, 4.0, 45)
        self.rng.shuffle(g_in)
        g_train, g_val = g_in[:30], g_in[30:]
        g_near = np.linspace(4.5, 10.0, 15)
        g_far = np.linspace(12.0, 30.0, 15)

        f = lambda g: 0.05 + 0.95 / ((1.0 + (0.5 * g)**2)**0.3)
        d_train = np.column_stack([g_train, f(g_train) + self.rng.normal(0, 0.005, len(g_train))])
        d_val = np.column_stack([g_val, f(g_val) + self.rng.normal(0, 0.005, len(g_val))])
        d_near = np.column_stack([g_near, f(g_near)])
        d_far = np.column_stack([g_far, f(g_far)])

        return self._save_problem(
            pid, "Carreau-Yasuda Non-Newtonian Shear-Thinning Viscosity", 5, "Multivariable Non-Separable & Hybrid Regimes", "C1", "UNIQUE_SUPPORT",
            ["gamma_dot"], "eta", {"gamma_dot": "per_second", "eta": "pascal_second"},
            d_train, d_val, d_near, d_far,
            {"formula": "0.05 + 0.95 / (1.0 + 0.25*gamma_dot**2)**0.3", "family": "carreau_yasuda", "parameters": {"eta_inf": 0.05, "eta_0": 1.0, "lambda": 0.5, "n": 0.4}},
        )


if __name__ == "__main__":
    generator = BenchmarkCorpusGenerator(Path(__file__).parent.parent / "benchmark_corpus")
    manifest = generator.generate_all()
    print(f"Generated {len(manifest['problems'])} benchmark problems in benchmark_corpus/")
