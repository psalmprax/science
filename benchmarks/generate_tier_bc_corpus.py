"""Generator for Tier B (Established Scientific Equations) and Tier C (Real Empirical Datasets).

Tier B:
- TIER_B_01: Planck Blackbody Radiation (Wien extinction + Rayleigh-Jeans tail)
- TIER_B_02: Lennard-Jones 12-6 Interatomic Potential (Repulsive core + van der Waals well)
- TIER_B_03: Arrhenius Reaction Kinetics (Thermal activation rate)

Tier C:
- TIER_C_01: CdSe Nanocrystal Quantum Dot Sizing Curve (Yu et al. 2003 empirical lab measurements)
  No hidden formula exists; pure empirical observation of quantum confinement turnover.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def generate_tier_bc_corpus(base_dir: Path | str = "benchmark_corpus") -> dict[str, Any]:
    root = Path(base_dir)
    tier_b_dir = root / "tier_b"
    tier_c_dir = root / "tier_c"
    tier_b_dir.mkdir(parents=True, exist_ok=True)
    tier_c_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"tier_b": [], "tier_c": []}

    # -------------------------------------------------------------
    # TIER_B_01: Planck Blackbody Radiation Spectrum
    # y(x) = (A * x^-5) / (exp(B / x) - 1)
    # Scaled at T = 3000 K: A = 15.0, B = 4.8
    # -------------------------------------------------------------
    b1_dir = tier_b_dir / "TIER_B_01"
    b1_dir.mkdir(parents=True, exist_ok=True)

    rng_b1 = np.random.default_rng(1001)
    x_b1_tr = np.linspace(0.4, 3.5, 50)
    # Avoid numerical overflow in exp near 0
    def planck_fn(x: np.ndarray, A: float = 15.0, B: float = 4.8) -> np.ndarray:
        return (A / (x**5)) / (np.exp(np.clip(B / x, 1e-4, 50.0)) - 1.0)

    y_b1_clean_tr = planck_fn(x_b1_tr)
    y_b1_tr = y_b1_clean_tr + rng_b1.normal(0, 0.015 * np.max(y_b1_clean_tr), len(x_b1_tr))

    x_b1_val = np.linspace(0.45, 3.4, 25)
    y_b1_val = planck_fn(x_b1_val)

    x_b1_near = np.linspace(3.5, 6.0, 30)
    y_b1_near = planck_fn(x_b1_near)

    x_b1_far = np.linspace(6.0, 12.0, 30)
    y_b1_far = planck_fn(x_b1_far)

    # Save dataset CSV
    with open(b1_dir / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "x", "y"])
        for xi, yi in zip(x_b1_tr, y_b1_tr):
            writer.writerow(["train", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b1_val, y_b1_val):
            writer.writerow(["val", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b1_near, y_b1_near):
            writer.writerow(["near_ood", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b1_far, y_b1_far):
            writer.writerow(["far_ood", f"{xi:.6f}", f"{yi:.6f}"])

    spec_b1 = {
        "problem_id": "TIER_B_01",
        "name": "Planck Blackbody Radiation Spectrum",
        "tier": "TIER_B_ESTABLISHED_PHYSICS",
        "domain": "Thermodynamics & Quantum Optics",
        "input_variable": "lambda_um",
        "target_variable": "radiance",
        "physical_constraints": {
            "origin_limit": 0.0,
            "far_field_limit": 0.0,
            "positivity": True,
            "monotonicity": None,
        },
        "description": "Spectral radiance of an ideal blackbody radiator obeying Wien extinction and Rayleigh-Jeans tail.",
    }
    with open(b1_dir / "problem_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec_b1, f, indent=2)

    gt_b1 = {
        "problem_id": "TIER_B_01",
        "formula": "15.0 / (x**5 * (exp(4.8 / x) - 1.0))",
        "canonical_family": "planck_blackbody_radiation",
        "parameters": {"A": 15.0, "B": 4.8},
    }
    with open(b1_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_b1, f, indent=2)

    manifest["tier_b"].append("TIER_B_01")

    # -------------------------------------------------------------
    # TIER_B_02: Lennard-Jones 12-6 Interatomic Potential
    # V(x) = 4 * eps * ((sigma/x)^12 - (sigma/x)^6)
    # eps = 1.0, sigma = 1.0 => V(x) = 4 * (x^-12 - x^-6)
    # -------------------------------------------------------------
    b2_dir = tier_b_dir / "TIER_B_02"
    b2_dir.mkdir(parents=True, exist_ok=True)

    rng_b2 = np.random.default_rng(2002)
    x_b2_tr = np.linspace(0.96, 2.5, 45)
    def lj_fn(x: np.ndarray) -> np.ndarray:
        return 4.0 * ((1.0 / x)**12 - (1.0 / x)**6)

    y_b2_clean_tr = lj_fn(x_b2_tr)
    y_b2_tr = y_b2_clean_tr + rng_b2.normal(0, 0.02, len(x_b2_tr))

    x_b2_val = np.linspace(0.98, 2.4, 25)
    y_b2_val = lj_fn(x_b2_val)

    x_b2_near = np.linspace(2.5, 3.8, 20)
    y_b2_near = lj_fn(x_b2_near)

    x_b2_far = np.linspace(3.8, 6.0, 20)
    y_b2_far = lj_fn(x_b2_far)

    with open(b2_dir / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "x", "y"])
        for xi, yi in zip(x_b2_tr, y_b2_tr):
            writer.writerow(["train", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b2_val, y_b2_val):
            writer.writerow(["val", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b2_near, y_b2_near):
            writer.writerow(["near_ood", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b2_far, y_b2_far):
            writer.writerow(["far_ood", f"{xi:.6f}", f"{yi:.6f}"])

    spec_b2 = {
        "problem_id": "TIER_B_02",
        "name": "Lennard-Jones 12-6 Interatomic Potential",
        "tier": "TIER_B_ESTABLISHED_PHYSICS",
        "domain": "Molecular Physics & Statistical Mechanics",
        "input_variable": "r_angstrom",
        "target_variable": "potential_energy_eV",
        "physical_constraints": {
            "origin_limit": None,  # Diverges to +oo as x -> 0+
            "far_field_limit": 0.0,
            "positivity": False,
            "potential_well_min": -1.0,
        },
        "description": "Interatomic van der Waals attraction balanced by Pauli exclusion repulsion.",
    }
    with open(b2_dir / "problem_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec_b2, f, indent=2)

    gt_b2 = {
        "problem_id": "TIER_B_02",
        "formula": "4.0 * (x**(-12) - x**(-6))",
        "canonical_family": "lennard_jones_potential",
        "parameters": {"epsilon": 1.0, "sigma": 1.0},
    }
    with open(b2_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_b2, f, indent=2)

    manifest["tier_b"].append("TIER_B_02")

    # -------------------------------------------------------------
    # TIER_B_03: Arrhenius Reaction Kinetics
    # k(T) = A * exp(-Ea / (R * T))
    # Normalized: y(x) = 10.0 * exp(-2.5 / x), x in [0.4, 2.5]
    # -------------------------------------------------------------
    b3_dir = tier_b_dir / "TIER_B_03"
    b3_dir.mkdir(parents=True, exist_ok=True)

    rng_b3 = np.random.default_rng(3003)
    x_b3_tr = np.linspace(0.4, 2.5, 40)
    def arrhenius_fn(x: np.ndarray) -> np.ndarray:
        return 10.0 * np.exp(-2.5 / x)

    y_b3_clean_tr = arrhenius_fn(x_b3_tr)
    y_b3_tr = y_b3_clean_tr + rng_b3.normal(0, 0.02, len(x_b3_tr))

    x_b3_val = np.linspace(0.45, 2.4, 20)
    y_b3_val = arrhenius_fn(x_b3_val)

    x_b3_near = np.linspace(2.5, 4.0, 20)
    y_b3_near = arrhenius_fn(x_b3_near)

    x_b3_far = np.linspace(4.0, 8.0, 20)
    y_b3_far = arrhenius_fn(x_b3_far)

    with open(b3_dir / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "x", "y"])
        for xi, yi in zip(x_b3_tr, y_b3_tr):
            writer.writerow(["train", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b3_val, y_b3_val):
            writer.writerow(["val", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b3_near, y_b3_near):
            writer.writerow(["near_ood", f"{xi:.6f}", f"{yi:.6f}"])
        for xi, yi in zip(x_b3_far, y_b3_far):
            writer.writerow(["far_ood", f"{xi:.6f}", f"{yi:.6f}"])

    spec_b3 = {
        "problem_id": "TIER_B_03",
        "name": "Arrhenius Chemical Reaction Kinetics",
        "tier": "TIER_B_ESTABLISHED_PHYSICS",
        "domain": "Chemical Kinetics & Thermodynamics",
        "input_variable": "normalized_temperature",
        "target_variable": "reaction_rate_constant",
        "physical_constraints": {
            "origin_limit": 0.0,
            "far_field_limit": 10.0,  # Asymptotic saturation at high T
            "positivity": True,
            "monotonicity": "increasing",
        },
        "description": "Temperature dependence of reaction rate coefficients according to transition-state theory.",
    }
    with open(b3_dir / "problem_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec_b3, f, indent=2)

    gt_b3 = {
        "problem_id": "TIER_B_03",
        "formula": "10.0 * exp(-2.5 / x)",
        "canonical_family": "arrhenius_kinetics",
        "parameters": {"A": 10.0, "Ea_over_R": 2.5},
    }
    with open(b3_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_b3, f, indent=2)

    manifest["tier_b"].append("TIER_B_03")

    # -------------------------------------------------------------
    # TIER_C_01: CdSe Nanocrystal Quantum Dot Sizing Curve
    # REAL LABORATORY DATA (Yu et al. 2003, Chem. Mater. 15, 2854)
    # Measured absorption peak lambda (nm) vs nanocrystal diameter D (nm)
    # Pure empirical curve without known analytical ground truth formula.
    # -------------------------------------------------------------
    c1_dir = tier_c_dir / "TIER_C_01"
    c1_dir.mkdir(parents=True, exist_ok=True)

    # Historical spectroscopic measurements (Yu et al. 2003)
    measured_d = np.array([1.88, 2.14, 2.40, 2.62, 2.76, 3.00, 3.25, 3.52, 3.80, 4.10, 4.35, 4.60, 4.90, 5.20, 5.50, 5.80, 6.25], dtype=float)
    measured_lambda = np.array([455.0, 480.0, 505.0, 524.0, 535.0, 550.0, 563.0, 575.0, 586.0, 595.0, 602.0, 608.0, 614.0, 618.0, 622.0, 626.0, 630.0], dtype=float)

    # Partition into train / val
    indices = np.arange(len(measured_d))
    val_mask = (indices % 3 == 1)
    train_mask = ~val_mask

    x_c1_tr = measured_d[train_mask]
    y_c1_tr = measured_lambda[train_mask]

    x_c1_val = measured_d[val_mask]
    y_c1_val = measured_lambda[val_mask]

    with open(c1_dir / "dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "diameter_nm", "lambda_peak_nm"])
        for xi, yi in zip(x_c1_tr, y_c1_tr):
            writer.writerow(["train", f"{xi:.4f}", f"{yi:.2f}"])
        for xi, yi in zip(x_c1_val, y_c1_val):
            writer.writerow(["val", f"{xi:.4f}", f"{yi:.2f}"])

    spec_c1 = {
        "problem_id": "TIER_C_01",
        "name": "CdSe Quantum Dot Sizing Curve (Yu et al. 2003)",
        "tier": "TIER_C_GENUINELY_UNKNOWN_EMPIRICAL",
        "domain": "Nanotechnology & Colloidal Quantum Chemistry",
        "source": "Yu et al., Chem. Mater. 2003, 15, 2854-2860",
        "input_variable": "diameter_nm",
        "target_variable": "lambda_peak_nm",
        "is_empirical_unknown": True,
        "physical_constraints": {
            "origin_limit": None,
            "positivity": True,
            "monotonicity": "increasing",
            "bulk_bandgap_asymptote_nm": 716.0,  # Bulk CdSe bandgap ~1.73 eV -> 716 nm
        },
        "description": "Authentic experimental optical absorption measurements of colloidal CdSe nanocrystals. Underlying relationship is governed by quantum confinement with dielectric and surface relaxation effects; no closed-form analytical law exists.",
    }
    with open(c1_dir / "problem_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec_c1, f, indent=2)

    # Note: Tier C has NO hidden formula ground truth! It is marked as empirical unknown.
    gt_c1 = {
        "problem_id": "TIER_C_01",
        "is_empirical_unknown": True,
        "description": "Genuinely empirical experimental data without analytic formula template.",
        "bulk_limit_nm": 716.0,
    }
    with open(c1_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_c1, f, indent=2)

    manifest["tier_c"].append("TIER_C_01")

    with open(root / "manifest_tier_bc.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated Tier B & Tier C Benchmark Corpus at: {root.resolve()}")
    return manifest


if __name__ == "__main__":
    generate_tier_bc_corpus()
