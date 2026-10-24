"""Empirical Noise Sensitivity Study for Off-Grid Continuous Rational Synthesis.

Evaluates recovery probability P(rational | sigma, N) and continuous parameter
error degradation (epsilon_p, epsilon_q, epsilon_theta) across noise levels:
    sigma in {0.0, 0.005, 0.010, 0.015, 0.030, 0.050}
for the rational transition law:
    y(x) = (2.0 * x^1.3) / (1.0 + 0.7 * x^2.1)

Disentangles prediction MSE from parameter identification and outputs a
structured JSON artifact documenting the empirical identifiability curve.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from apedp.model_synthesis import AutonomousModelSynthesizer
from apedp.parameter_diagnostics import (
    RecoveryCategory,
    compute_parameter_recovery,
)


def run_noise_sensitivity_study(
    noise_levels: list[float] | None = None,
    trials_per_level: int = 10,
    output_path: Path | str = "tournament_results/rational_noise_sensitivity.json",
) -> dict[str, Any]:
    if noise_levels is None:
        noise_levels = [0.0, 0.005, 0.010, 0.015, 0.030, 0.050]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    target_family = "rational_transition"
    target_params = {"A": 2.0, "p": 1.3, "B": 0.7, "q": 2.1}

    synthesizer = AutonomousModelSynthesizer()

    study_results: dict[str, Any] = {
        "metadata": {
            "law": "y = (2.0 * x^1.3) / (1 + 0.7 * x^2.1)",
            "target_family": target_family,
            "target_params": target_params,
            "n_train": 40,
            "n_val": 20,
            "x_domain_train": [0.5, 4.5],
            "x_domain_val": [0.6, 4.3],
            "noise_levels": noise_levels,
            "trials_per_level": trials_per_level,
        },
        "level_summaries": {},
        "trials": [],
    }

    base_seed = 8000

    for sigma in noise_levels:
        level_trials = []
        rational_count = 0
        cat_a_count = 0
        cat_b_count = 0
        cat_c_count = 0

        p_errors: list[float] = []
        q_errors: list[float] = []
        max_errors: list[float] = []
        val_mses: list[float] = []

        print(f"\n--- Evaluating Noise Level sigma = {sigma:.4f} ({trials_per_level} trials) ---", flush=True)

        for trial_idx in range(trials_per_level):
            trial_seed = base_seed + int(sigma * 10000) * 100 + trial_idx
            rng = np.random.default_rng(trial_seed)

            x_tr = np.linspace(0.5, 4.5, 40)
            y_clean_tr = (2.0 * x_tr**1.3) / (1.0 + 0.7 * x_tr**2.1)
            y_tr = y_clean_tr + rng.normal(0, sigma, len(x_tr)) if sigma > 0 else y_clean_tr.copy()

            x_val = np.linspace(0.6, 4.3, 20)
            y_val = (2.0 * x_val**1.3) / (1.0 + 0.7 * x_val**2.1)

            models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
            top = models[0] if models else None

            if top is None:
                continue

            # Compute fine-grained diagnostic report
            rec_family = top.structural_family
            rep = compute_parameter_recovery(
                recovered_family=rec_family,
                target_family=target_family,
                recovered_params=top.parameters,
                target_params=target_params,
                val_mse=top.val_mse,
                ood_mse=top.near_ood_mse,
                proof_tier=top.proof_claim,
                val_mse_threshold=max(1e-3, 5 * (sigma**2 if sigma > 0 else 1e-4)),
                param_rel_tolerance=0.08,
            )

            is_rational = (rec_family in {"rational_transition", "rational_saturation"})
            if is_rational:
                rational_count += 1

            if rep.category == RecoveryCategory.CATEGORY_A:
                cat_a_count += 1
            elif rep.category == RecoveryCategory.CATEGORY_B:
                cat_b_count += 1
            else:
                cat_c_count += 1

            print(
                f"  Trial {trial_idx+1}/{trials_per_level}: family={rec_family}, "
                f"cat={rep.category.value}, val_mse={top.val_mse:.2e}",
                flush=True,
            )

            p_err = rep.param_errors["p"].relative_error if "p" in rep.param_errors else None
            q_err = rep.param_errors["q"].relative_error if "q" in rep.param_errors else None

            if p_err is not None:
                p_errors.append(p_err)
            if q_err is not None:
                q_errors.append(q_err)
            if rep.max_relative_param_error < float("inf"):
                max_errors.append(rep.max_relative_param_error)

            val_mses.append(top.val_mse)

            trial_record = {
                "sigma": sigma,
                "trial_idx": trial_idx,
                "seed": trial_seed,
                "recovered_expression": top.expression_str,
                "recovered_family": rec_family,
                "is_rational": is_rational,
                "category": rep.category.value,
                "val_mse": top.val_mse,
                "near_ood_mse": top.near_ood_mse,
                "proof_tier": top.proof_claim,
                "p_error_rel": p_err,
                "q_error_rel": q_err,
                "max_param_error_rel": rep.max_relative_param_error if rep.max_relative_param_error < float("inf") else None,
                "parameters": top.parameters,
            }
            level_trials.append(trial_record)
            study_results["trials"].append(trial_record)

        p_rational = rational_count / trials_per_level
        p_cat_a = cat_a_count / trials_per_level
        p_cat_b = cat_b_count / trials_per_level
        p_cat_c = cat_c_count / trials_per_level

        med_p_err = float(np.median(p_errors)) if p_errors else None
        med_q_err = float(np.median(q_errors)) if q_errors else None
        med_val_mse = float(np.median(val_mses)) if val_mses else None

        summary = {
            "sigma": sigma,
            "trials": trials_per_level,
            "p_rational_selection": p_rational,
            "category_a_rate": p_cat_a,
            "category_b_rate": p_cat_b,
            "category_c_rate": p_cat_c,
            "median_val_mse": med_val_mse,
            "median_p_error": med_p_err,
            "median_q_error": med_q_err,
        }
        study_results["level_summaries"][str(sigma)] = summary
        err_str = f"{med_p_err:.2%}" if med_p_err is not None else "N/A"
        print(f"Summary for sigma={sigma:.4f}: P(rational)={p_rational:.1%}, Cat A={p_cat_a:.1%}, "
              f"Cat B={p_cat_b:.1%}, Cat C={p_cat_c:.1%}, Med MSE={med_val_mse:.2e}, "
              f"Med eps_p={err_str}", flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(study_results, f, indent=2)

    print(f"\nSaved empirical sensitivity report to: {output_file}", flush=True)
    return study_results


if __name__ == "__main__":
    run_noise_sensitivity_study(
        noise_levels=[0.0, 0.005, 0.015, 0.030],
        trials_per_level=3,
    )
