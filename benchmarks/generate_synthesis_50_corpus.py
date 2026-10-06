"""Synthetic 50-Problem Synthesis Corpus Generator for APEDP 4.0.

Generates benchmark_corpus_synthesis_50/ across 5 distinct physical classes:
- Class S1: Multi-Regime Rational-Exponential Envelopes (S01–S10)
- Class S2: Logarithmic & Boundary-Layer Corrections (S11–S20)
- Class S3: Critical Exponents & Phase Transitions (S21–S30)
- Class S4: Modulated Dissipative & Oscillatory Transport (S31–S40)
- Class S5: Epistemic Abstention & Out-of-Grammar Adversaries (S41–S50)
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.special as sp_special

CORPUS_50_DIR = Path(__file__).resolve().parent.parent / "benchmark_corpus_synthesis_50"


def generate_class_s1_problems(rng: np.random.Generator) -> list[dict]:
    """Generates Class S1: Multi-Regime Rational-Exponential Envelopes."""
    problems = []
    configs = [
        (1.5, 2.0, 1.0, 2.45, 0.85, 0.30),
        (0.5, 1.0, 1.0, 1.80, 0.50, 0.20),
        (1.0, 2.0, 1.5, 3.20, 1.20, 0.40),
        (2.0, 2.0, 0.5, 1.50, 0.70, 0.25),
        (1.25, 1.75, 1.0, 2.10, 0.90, 0.35),
        (0.75, 1.5, 1.2, 2.80, 0.60, 0.15),
        (1.5, 2.5, 0.8, 1.95, 1.10, 0.50),
        (2.0, 3.0, 1.0, 4.00, 1.50, 0.30),
        (1.0, 1.0, 2.0, 2.20, 0.40, 0.10),
        (1.5, 2.0, 2.0, 3.50, 0.80, 0.25),
    ]

    for i, (alpha, beta, gamma, A, B, C) in enumerate(configs, 1):
        pid = f"S{i:02d}"
        def gt(x, A=A, alpha=alpha, B=B, beta=beta, C=C, gamma=gamma):
            return (A * (x ** alpha) / (1.0 + B * (x ** beta))) * np.exp(-C * (x ** gamma))

        # Train: [0.1, 5.0], gap [2.2, 3.0]
        pts_left = np.linspace(0.1, 2.2, 45)
        pts_right = np.linspace(3.0, 5.0, 35)
        x_tr = np.sort(np.concatenate([pts_left, pts_right]))
        y_tr_clean = gt(x_tr)
        noise = rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_tr))
        y_tr = np.maximum(y_tr_clean + noise, 0.0)

        # Val: [0.15, 4.95]
        v_left = np.linspace(0.15, 2.15, 25)
        v_right = np.linspace(3.05, 4.95, 25)
        x_v = np.sort(np.concatenate([v_left, v_right]))
        y_v = np.maximum(gt(x_v) + rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_v)), 0.0)

        # OOD
        x_near = np.linspace(5.0, 8.0, 50)
        y_near = gt(x_near)
        x_far = np.linspace(8.0, 15.0, 50)
        y_far = gt(x_far)

        spec = {
            "problem_id": pid,
            "problem_name": f"Rational-Exponential Transition S{i:02d}",
            "class_id": 1,
            "class_name": "Multi-Regime Envelopes",
            "input_variables": ["x"],
            "target_variable": "y",
            "variable_units": {"x": "meter", "y": "pascal"},
            "physical_constraints": {"origin_limit": 0.0, "far_field_limit": 0.0, "positivity": True, "sampling_gaps": [[2.2, 3.0]]},
            "dataset_sizes": {"train": len(x_tr), "validation": len(x_v), "near_ood": len(x_near), "far_ood": len(x_far)},
        }
        gt_doc = {
            "problem_id": pid,
            "class_id": 1,
            "oracle_epistemic_class": "C1",
            "expected_state": "UNIQUE_SUPPORT",
            "ground_truth_formula": f"({A} * x**{alpha} / (1.0 + {B} * x**{beta})) * exp(-{C} * x**{gamma})",
            "canonical_structural_family": "rational_exponential_envelope",
            "true_parameters": {"A": A, "alpha": alpha, "B": B, "beta": beta, "C": C, "gamma": gamma},
        }
        problems.append({"spec": spec, "gt": gt_doc, "train": (x_tr, y_tr), "val": (x_v, y_v), "near": (x_near, y_near), "far": (x_far, y_far)})

    return problems


def generate_class_s2_problems(rng: np.random.Generator) -> list[dict]:
    """Generates Class S2: Logarithmic & Boundary-Layer Corrections."""
    problems = []
    configs = [
        (0.5, 1.0, 1.5, 0.8),
        (1.0, 1.0, 2.0, 0.5),
        (0.75, 1.5, 1.2, 1.0),
        (1.2, 0.8, 2.5, 0.4),
        (0.5, 2.0, 1.0, 0.6),
        (1.5, 1.0, 1.8, 0.3),
        (0.8, 1.2, 2.2, 0.7),
        (1.0, 2.0, 1.4, 0.5),
        (0.6, 0.5, 1.7, 0.9),
        (1.1, 1.5, 2.1, 0.4),
    ]

    for i, (alpha, beta, A, B) in enumerate(configs, 11):
        pid = f"S{i:02d}"
        def gt(x, A=A, alpha=alpha, B=B, beta=beta):
            return A * (x ** alpha) * np.log(1.0 + B * (x ** beta))

        x_tr = np.linspace(0.1, 5.0, 70)
        y_tr_clean = gt(x_tr)
        y_tr = y_tr_clean + rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_tr))

        x_v = np.linspace(0.15, 4.95, 40)
        y_v = gt(x_v) + rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_v))

        x_near = np.linspace(5.0, 8.0, 50)
        y_near = gt(x_near)
        x_far = np.linspace(8.0, 15.0, 50)
        y_far = gt(x_far)

        spec = {
            "problem_id": pid,
            "problem_name": f"Logarithmic Boundary Layer S{i:02d}",
            "class_id": 2,
            "class_name": "Logarithmic & Boundary Layers",
            "input_variables": ["x"],
            "target_variable": "y",
            "variable_units": {"x": "meter", "y": "meter / second"},
            "physical_constraints": {"origin_limit": 0.0, "positivity": True},
            "dataset_sizes": {"train": len(x_tr), "validation": len(x_v), "near_ood": len(x_near), "far_ood": len(x_far)},
        }
        gt_doc = {
            "problem_id": pid,
            "class_id": 2,
            "oracle_epistemic_class": "C1",
            "expected_state": "UNIQUE_SUPPORT",
            "ground_truth_formula": f"{A} * x**{alpha} * ln(1.0 + {B} * x**{beta})",
            "canonical_structural_family": "logarithmic_boundary_layer",
            "true_parameters": {"A": A, "alpha": alpha, "B": B, "beta": beta},
        }
        problems.append({"spec": spec, "gt": gt_doc, "train": (x_tr, y_tr), "val": (x_v, y_v), "near": (x_near, y_near), "far": (x_far, y_far)})

    return problems


def generate_class_s3_problems(rng: np.random.Generator) -> list[dict]:
    """Generates Class S3: Critical Exponents & Phase Transitions."""
    problems = []
    configs = [
        (2.5, 0.5, 1.2, 0.8),
        (3.0, 0.75, 1.5, 0.5),
        (2.0, 0.33, 2.0, 0.4),
        (3.5, 0.67, 1.1, 0.9),
        (2.2, 0.5, 1.4, 0.6),
        (2.8, 0.4, 1.8, 0.7),
        (3.2, 0.8, 1.0, 0.5),
        (2.6, 0.6, 1.3, 0.8),
        (3.1, 0.5, 1.6, 0.4),
        (2.4, 0.7, 1.7, 0.6),
    ]

    for i, (xc, gamma, A, B) in enumerate(configs, 21):
        pid = f"S{i:02d}"
        def gt(x, xc=xc, gamma=gamma, A=A, B=B):
            dx = np.abs(x - xc)
            return A * (dx ** (-gamma)) + B * x

        # Gap right around xc: [xc - 0.2, xc + 0.2]
        pts_left = np.linspace(0.5, xc - 0.2, 35)
        pts_right = np.linspace(xc + 0.2, 6.0, 35)
        x_tr = np.sort(np.concatenate([pts_left, pts_right]))
        y_tr_clean = gt(x_tr)
        y_tr = y_tr_clean + rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_tr))

        v_left = np.linspace(0.6, xc - 0.25, 20)
        v_right = np.linspace(xc + 0.25, 5.8, 20)
        x_v = np.sort(np.concatenate([v_left, v_right]))
        y_v = gt(x_v) + rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_v))

        x_near = np.linspace(6.0, 9.0, 50)
        y_near = gt(x_near)
        x_far = np.linspace(9.0, 15.0, 50)
        y_far = gt(x_far)

        spec = {
            "problem_id": pid,
            "problem_name": f"Critical Scaling S{i:02d}",
            "class_id": 3,
            "class_name": "Critical Exponents & Scaling",
            "input_variables": ["x"],
            "target_variable": "y",
            "variable_units": {"x": "kelvin", "y": "joule / kelvin"},
            "physical_constraints": {"sampling_gaps": [[round(xc - 0.2, 2), round(xc + 0.2, 2)]]},
            "dataset_sizes": {"train": len(x_tr), "validation": len(x_v), "near_ood": len(x_near), "far_ood": len(x_far)},
        }
        gt_doc = {
            "problem_id": pid,
            "class_id": 3,
            "oracle_epistemic_class": "C1",
            "expected_state": "UNIQUE_SUPPORT",
            "ground_truth_formula": f"{A} * abs(x - {xc})**(-{gamma}) + {B} * x",
            "canonical_structural_family": "critical_scaling_divergence",
            "true_parameters": {"xc": xc, "gamma": gamma, "A": A, "B": B},
        }
        problems.append({"spec": spec, "gt": gt_doc, "train": (x_tr, y_tr), "val": (x_v, y_v), "near": (x_near, y_near), "far": (x_far, y_far)})

    return problems


def generate_class_s4_problems(rng: np.random.Generator) -> list[dict]:
    """Generates Class S4: Modulated Dissipative & Oscillatory Transport."""
    problems = []
    configs = [
        (2.0, 0.4, 2.5),
        (1.5, 0.3, 3.0),
        (2.5, 0.5, 2.0),
        (1.8, 0.25, 4.0),
        (3.0, 0.6, 1.5),
        (2.2, 0.35, 3.5),
        (1.6, 0.45, 2.8),
        (2.4, 0.55, 1.8),
        (1.9, 0.3, 3.2),
        (2.1, 0.4, 2.2),
    ]

    for i, (A, beta, omega) in enumerate(configs, 31):
        pid = f"S{i:02d}"
        def gt(x, A=A, beta=beta, omega=omega):
            return A * np.exp(-beta * x) * np.cos(omega * x)

        x_tr = np.linspace(0.0, 6.0, 80)
        y_tr_clean = gt(x_tr)
        y_tr = y_tr_clean + rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_tr))

        x_v = np.linspace(0.05, 5.95, 40)
        y_v = gt(x_v) + rng.normal(0, 0.02 * np.std(y_tr_clean), len(x_v))

        x_near = np.linspace(6.0, 10.0, 50)
        y_near = gt(x_near)
        x_far = np.linspace(10.0, 16.0, 50)
        y_far = gt(x_far)

        spec = {
            "problem_id": pid,
            "problem_name": f"Damped Ringdown S{i:02d}",
            "class_id": 4,
            "class_name": "Modulated Transport",
            "input_variables": ["x"],
            "target_variable": "y",
            "variable_units": {"x": "second", "y": "volt"},
            "physical_constraints": {"far_field_limit": 0.0},
            "dataset_sizes": {"train": len(x_tr), "validation": len(x_v), "near_ood": len(x_near), "far_ood": len(x_far)},
        }
        gt_doc = {
            "problem_id": pid,
            "class_id": 4,
            "oracle_epistemic_class": "C1",
            "expected_state": "UNIQUE_SUPPORT",
            "ground_truth_formula": f"{A} * exp(-{beta} * x) * cos({omega} * x)",
            "canonical_structural_family": "damped_oscillation",
            "true_parameters": {"A": A, "beta": beta, "omega": omega},
        }
        problems.append({"spec": spec, "gt": gt_doc, "train": (x_tr, y_tr), "val": (x_v, y_v), "near": (x_near, y_near), "far": (x_far, y_far)})

    return problems


def generate_class_s5_problems(rng: np.random.Generator) -> list[dict]:
    """Generates Class S5: Epistemic Abstention & Out-of-Grammar Adversaries."""
    problems = []

    # 10 Adversarial cases requiring formal epistemic abstention (C3 or C5)
    for i in range(41, 51):
        pid = f"S{i:02d}"

        if i <= 43:
            # Piecewise Discontinuous Shock Front
            x_shock = 2.5
            def gt(x, x_shock=x_shock):
                return np.where(x < x_shock, 1.5 * x, 4.0 - 0.5 * x)
            name = f"Discontinuous Shock Front S{i:02d}"
            fam = "piecewise_shock_discontinuity"
        elif i <= 46:
            # Non-elementary Bessel function J0(x)
            def gt(x):
                return float(sp_special.j0(float(x))) if np.isscalar(x) else sp_special.j0(x)
            name = f"Bessel Special Function S{i:02d}"
            fam = "bessel_j0_non_elementary"
        elif i <= 48:
            # Chaotic logistic map trajectory / high-entropy oscillation
            def gt(x):
                return np.sin(x ** 3) * np.cos(np.exp(0.5 * x))
            name = f"High-Entropy Chaotic Trajectory S{i:02d}"
            fam = "high_entropy_oscillation"
        else:
            # Severely Underspecified Sparse Data (N=6 points over 4 orders of magnitude)
            def gt(x):
                return 2.5 * (x ** 1.8) * np.exp(-0.4 * x)
            name = f"Severely Underspecified Sparse System S{i:02d}"
            fam = "underspecified_data_regime"

        if i <= 48:
            x_tr = np.linspace(0.1, 5.0, 60)
            y_tr_clean = gt(x_tr)
            y_tr = y_tr_clean + rng.normal(0, 0.05 * np.std(y_tr_clean), len(x_tr))
            x_v = np.linspace(0.15, 4.95, 30)
            y_v = gt(x_v) + rng.normal(0, 0.05 * np.std(y_tr_clean), len(x_v))
        else:
            # Sparse 6 points only!
            x_tr = np.array([0.2, 0.8, 1.5, 2.5, 3.8, 4.8])
            y_tr = gt(x_tr) + rng.normal(0, 0.02, 6)
            x_v = np.array([0.4, 2.0, 4.2])
            y_v = gt(x_v)

        x_near = np.linspace(5.0, 8.0, 40)
        y_near = gt(x_near)
        x_far = np.linspace(8.0, 14.0, 40)
        y_far = gt(x_far)

        spec = {
            "problem_id": pid,
            "problem_name": name,
            "class_id": 5,
            "class_name": "Epistemic Abstention Adversaries",
            "input_variables": ["x"],
            "target_variable": "y",
            "variable_units": {"x": "dimensionless", "y": "dimensionless"},
            "physical_constraints": {},
            "dataset_sizes": {"train": len(x_tr), "validation": len(x_v), "near_ood": len(x_near), "far_ood": len(x_far)},
        }
        gt_doc = {
            "problem_id": pid,
            "class_id": 5,
            "oracle_epistemic_class": "C3" if i >= 49 else "C5",
            "expected_state": "INSUFFICIENT_EVIDENCE" if i >= 49 else "MISSING_STRUCTURE",
            "ground_truth_formula": "UNREPRESENTABLE_IN_ELEMENTARY_GRAMMAR",
            "canonical_structural_family": fam,
            "true_parameters": {},
            "mandatory_action": "FORMAL_ABSTENTION_REQUIRED",
        }
        problems.append({"spec": spec, "gt": gt_doc, "train": (x_tr, y_tr), "val": (x_v, y_v), "near": (x_near, y_near), "far": (x_far, y_far)})

    return problems


def build_and_save_corpus_50(output_dir: Path | None = None) -> Path:
    out = output_dir or CORPUS_50_DIR
    rng = np.random.default_rng(20260925)

    all_problems = []
    all_problems.extend(generate_class_s1_problems(rng))
    all_problems.extend(generate_class_s2_problems(rng))
    all_problems.extend(generate_class_s3_problems(rng))
    all_problems.extend(generate_class_s4_problems(rng))
    all_problems.extend(generate_class_s5_problems(rng))

    assert len(all_problems) == 50, f"Expected 50 problems, got {len(all_problems)}"

    for p in all_problems:
        pid = p["spec"]["problem_id"]
        pub_p = out / "public" / pid
        hid_p = out / "hidden" / pid
        pub_p.mkdir(parents=True, exist_ok=True)
        hid_p.mkdir(parents=True, exist_ok=True)

        with open(pub_p / "problem_spec.json", "w", encoding="utf-8") as f:
            json.dump(p["spec"], f, indent=2)

        with open(hid_p / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(p["gt"], f, indent=2)

        def write_csv(fpath, x_arr, y_arr):
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["x", "y"])
                for xi, yi in zip(x_arr, y_arr):
                    w.writerow([f"{xi:.8f}", f"{yi:.8f}"])

        write_csv(pub_p / "train.csv", *p["train"])
        write_csv(pub_p / "validation.csv", *p["val"])
        write_csv(pub_p / "near_ood.csv", *p["near"])
        write_csv(pub_p / "far_ood.csv", *p["far"])

    # Build manifest
    manifest = {
        "benchmark_name": "APEDP 4.0 50-Problem Autonomous Model Synthesis Tournament",
        "total_problems": 50,
        "num_classes": 5,
        "problems_per_class": 10,
        "problems": {
            p["spec"]["problem_id"]: {
                "class_id": p["spec"]["class_id"],
                "class_name": p["spec"]["class_name"],
                "problem_name": p["spec"]["problem_name"],
            }
            for p in all_problems
        },
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Checksums
    checksum_lines = []
    for fpath in sorted(out.rglob("*")):
        if fpath.is_file() and fpath.name != "checksums.sha256":
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            rel = fpath.relative_to(out)
            checksum_lines.append(f"{h.hexdigest()}  {rel}\n")

    with open(out / "checksums.sha256", "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)

    print(f"Successfully generated APEDP 4.0 50-Problem Synthesis Corpus at: {out}")
    return out


if __name__ == "__main__":
    build_and_save_corpus_50()
