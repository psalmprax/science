"""Synthetic Synthesis Problem Generator for APEDP 4.0.

Generates the benchmark_corpus_synthesis/ benchmark containing blind equation
synthesis challenges, beginning with SYNTH_01 (multi-regime rational-exponential envelope):
    y*(x) = (A * x**alpha / (1 + B * x**beta)) * exp(-C * x**gamma)
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CORPUS_SYNTHESIS_DIR = Path(__file__).resolve().parent.parent / "benchmark_corpus_synthesis"


def generate_synth_01_dataset(seed: int = 42) -> dict[str, Any]:
    """Generates the SYNTH_01 multi-regime physical transition challenge."""
    rng = np.random.default_rng(seed)

    A_true = 2.45
    alpha_true = 1.50
    B_true = 0.85
    beta_true = 2.00
    C_true = 0.30
    gamma_true = 1.00

    def ground_truth(x: np.ndarray) -> np.ndarray:
        num = A_true * (x ** alpha_true)
        denom = 1.0 + B_true * (x ** beta_true)
        env = np.exp(-C_true * (x ** gamma_true))
        return (num / denom) * env

    # Domain specifications
    # Train/Val: x in [0.1, 5.0], with avoided gap in [2.2, 3.0]
    n_train_pts = 100
    n_val_pts = 50

    # Region 1: [0.1, 2.2]
    # Region 2: [3.0, 5.0]
    pts_left = np.linspace(0.1, 2.2, int(n_train_pts * 0.55))
    pts_right = np.linspace(3.0, 5.0, int(n_train_pts * 0.45))
    x_train = np.sort(np.concatenate([pts_left, pts_right]))

    # Noise on train
    y_train_clean = ground_truth(x_train)
    noise_sigma = 0.02 * float(np.std(y_train_clean))
    y_train = y_train_clean + rng.normal(0, noise_sigma, len(x_train))
    y_train = np.maximum(y_train, 0.0)  # Positivity constraint

    # Validation
    v_left = np.linspace(0.15, 2.15, int(n_val_pts * 0.55))
    v_right = np.linspace(3.05, 4.95, int(n_val_pts * 0.45))
    x_val = np.sort(np.concatenate([v_left, v_right]))
    y_val = ground_truth(x_val) + rng.normal(0, noise_sigma, len(x_val))
    y_val = np.maximum(y_val, 0.0)

    # Near-OOD: [5.0, 8.0]
    x_near = np.linspace(5.0, 8.0, 100)
    y_near = ground_truth(x_near)

    # Far-OOD: [8.0, 15.0]
    x_far = np.linspace(8.0, 15.0, 100)
    y_far = ground_truth(x_far)

    # Problem spec (public)
    public_spec = {
        "problem_id": "SYNTH_01",
        "problem_name": "Multi-Regime Rational-Exponential Transition Law",
        "class_id": 1,
        "class_name": "Autonomous Model Synthesis",
        "input_variables": ["x"],
        "target_variable": "y",
        "variable_units": {
            "x": "meter",
            "y": "pascal",
        },
        "physical_constraints": {
            "origin_limit": 0.0,
            "far_field_limit": 0.0,
            "positivity": True,
            "sampling_gaps": [[2.2, 3.0]],
        },
        "dataset_sizes": {
            "train": len(x_train),
            "validation": len(x_val),
            "near_ood": len(x_near),
            "far_ood": len(x_far),
        },
    }

    # Ground truth (hidden)
    hidden_ground_truth = {
        "problem_id": "SYNTH_01",
        "problem_name": "Multi-Regime Rational-Exponential Transition Law",
        "class_id": 1,
        "class_name": "Autonomous Model Synthesis",
        "oracle_epistemic_class": "C1",
        "expected_state": "UNIQUE_SUPPORT",
        "ground_truth_formula": f"({A_true} * x**{alpha_true} / (1.0 + {B_true} * x**{beta_true})) * exp(-{C_true} * x**{gamma_true})",
        "canonical_structural_family": "rational_exponential_envelope",
        "true_parameters": {
            "A": A_true,
            "alpha": alpha_true,
            "B": B_true,
            "beta": beta_true,
            "C": C_true,
            "gamma": gamma_true,
        },
        "competing_hypotheses": [
            "2.45 * x**1.5 * exp(-0.3 * x)",
            "2.45 * x**1.5 / (1.0 + 0.85 * x**2)",
            "1.85 * x * exp(-0.25 * x)",
        ],
        "pathology_details": {
            "has_sampling_gap": True,
            "gap_interval": [2.2, 3.0],
            "risk_type": "AVOIDED_POLE_IN_GAP",
        },
    }

    return {
        "public_spec": public_spec,
        "hidden_ground_truth": hidden_ground_truth,
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "near_ood": (x_near, y_near),
        "far_ood": (x_far, y_far),
    }


def write_synthesis_corpus(output_dir: Path | None = None) -> Path:
    out = output_dir or CORPUS_SYNTHESIS_DIR
    public_dir = out / "public" / "SYNTH_01"
    hidden_dir = out / "hidden" / "SYNTH_01"
    public_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    data = generate_synth_01_dataset()

    # Write public spec
    with open(public_dir / "problem_spec.json", "w", encoding="utf-8") as f:
        json.dump(data["public_spec"], f, indent=2)

    # Write hidden ground truth
    with open(hidden_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(data["hidden_ground_truth"], f, indent=2)

    def write_csv(p: Path, x_arr: np.ndarray, y_arr: np.ndarray):
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for xi, yi in zip(x_arr, y_arr):
                writer.writerow([f"{xi:.8f}", f"{yi:.8f}"])

    write_csv(public_dir / "train.csv", *data["train"])
    write_csv(public_dir / "validation.csv", *data["val"])
    write_csv(public_dir / "near_ood.csv", *data["near_ood"])
    write_csv(public_dir / "far_ood.csv", *data["far_ood"])

    # Build manifest
    manifest = {
        "benchmark_name": "APEDP 4.0 Autonomous Model Synthesis Benchmark",
        "total_problems": 1,
        "num_classes": 1,
        "problems": {
            "SYNTH_01": {
                "class_id": 1,
                "problem_name": data["public_spec"]["problem_name"],
            }
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

    print(f"Generated APEDP 4.0 Synthesis Corpus at: {out}")
    return out


if __name__ == "__main__":
    write_synthesis_corpus()
