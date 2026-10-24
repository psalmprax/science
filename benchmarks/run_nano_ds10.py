"""NANO-DS10: 10-Seed Nanoscale Discovery Stability & Identification Benchmark.

Evaluates the formal distinction:
    Recovery != Identification != Stability != Transfer != Promotion

On the M4 nanoscale Hall-Petch crossover (dc = 15 nm):
  1. Discovery Stability: The system repeatedly and stably recovers the piecewise
     structure across 10 independent noise seeds (winner_recurrence = 1.0, entropy = 0.0 nats).
  2. Parameter Dispersion: Parameter estimation exhibits well-characterized CV
     (global k_hp CV = 0.045, piecewise k_hp CV ~ 0.34).
  3. Identification Failure: Despite 100% structural recovery and low entropy,
     passive observational data cannot separate piecewise from constant_baseline within tau_epi,
     correctly returning MULTI_HYPOTHESIS with INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY
     until L3 active intervention is performed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.epistemic.discovery_stability import (
    coefficient_cv,
    compose_identifiability_stability_matrix,
    structural_distribution,
    structural_entropy,
    variance_split,
)
from apedp_dyn.sandbox.nano_crossover_m4 import (
    M4_DC_TRUE_NM,
    fit_m4_nano_candidates,
    generate_m4_data,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class NanoDS10Summary:
    """Summary record of the NANO-DS10 discovery stability benchmark."""

    benchmark: str
    full_name: str
    fit_seeds: int
    winner_recurrence: dict[str, float]
    entropy_nats: float
    k_hp_cv: float
    adjudication_status: str
    variance_split: dict[str, float]
    matrix: dict[str, Any]
    passed_contract: bool
    question: str


def run_nano_ds10(
    n_seeds: int = 10,
    n_points: int = 60,
    noise_sigma: float = 0.02,
    output_path: Path | None = None,
) -> tuple[NanoDS10Summary, dict[str, Any]]:
    """Execute 10-seed discovery stability audit on M4 nanoscale crossover data."""
    winners: list[str] = []
    k_global_vals: list[float] = []
    k_piecewise_vals: list[float] = []
    baseline_adj = None

    # Track fits for 10x10 scoring matrix
    fitted_piecewise_weights: list[np.ndarray] = []

    for s in range(n_seeds):
        d, sigma_obs = generate_m4_data(n_points=n_points, noise_sigma=noise_sigma, seed=s)
        cands, info = fit_m4_nano_candidates(
            d, sigma_obs, n_train=30, noise_sigma=noise_sigma, data_origin=DataOrigin.SIMULATION.value
        )

        # Determine winner by best training loss among candidate families
        valid_cands = [c for c in cands if c.train_loss is not None and np.isfinite(c.train_loss)]
        best_cand = min(valid_cands,
                        key=lambda c: c.train_loss if c.train_loss is not None else float("inf"))
        winners.append(best_cand.candidate.candidate_id)

        # Adjudicate baseline seed 0
        if s == 0:
            baseline_adj = adjudicate(
                cands,
                sole_survivor_rule=True,
                bootstrap=True,
                bootstrap_n_boot=200,
                bootstrap_seed=42,
                data_origin=DataOrigin.SIMULATION.value,
                benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY.value,
            )

        # Parameter extraction: global k_hp vs piecewise k
        d_tr = d[:30]
        y_tr = sigma_obs[:30]
        w_global, *_ = np.linalg.lstsq(
            np.column_stack([np.ones_like(d_tr), d_tr**-0.5]), y_tr, rcond=None
        )
        k_global_vals.append(float(w_global[1]))

        dc_hat = info["dc_hat_nm"]
        m_tr = d_tr >= dc_hat
        Xp_tr = np.zeros((len(d_tr), 4))
        Xp_tr[m_tr, 0], Xp_tr[m_tr, 1] = 1.0, d_tr[m_tr] ** -0.5
        Xp_tr[~m_tr, 2], Xp_tr[~m_tr, 3] = 1.0, d_tr[~m_tr]
        w_pw, *_ = np.linalg.lstsq(Xp_tr, y_tr, rcond=None)
        k_piecewise_vals.append(float(w_pw[1]))
        fitted_piecewise_weights.append(w_pw)

    # 10x10 scoring matrix across evaluation seeds
    scoring_matrix = np.zeros((n_seeds, n_seeds))
    for i in range(n_seeds):
        w_i = fitted_piecewise_weights[i]
        for j in range(n_seeds):
            d_eval, y_eval = generate_m4_data(n_points=30, noise_sigma=noise_sigma, seed=200 + j)
            m_eval = d_eval >= M4_DC_TRUE_NM
            Xp_eval = np.zeros((len(d_eval), 4))
            Xp_eval[m_eval, 0], Xp_eval[m_eval, 1] = 1.0, d_eval[m_eval] ** -0.5
            Xp_eval[~m_eval, 2], Xp_eval[~m_eval, 3] = 1.0, d_eval[~m_eval]
            rmse = float(np.sqrt(np.mean((y_eval - Xp_eval @ w_i) ** 2)))
            scoring_matrix[i, j] = rmse

    v_split = variance_split(scoring_matrix)
    dist = structural_distribution(winners)
    entropy = structural_entropy(winners)
    cv_global = coefficient_cv(k_global_vals)
    k_hp_cv = round(cv_global, 3) if cv_global is not None else float("nan")

    stability_rep = {
        "structures": winners,
        "structural_distribution": dist,
        "structural_entropy": entropy,
        "structural_recurrence": dist.get("piecewise_hp_inv", 0.0),
        "parameter_cv": {
            "k_hp_global": coefficient_cv(k_global_vals),
            "k_hp_piecewise": coefficient_cv(k_piecewise_vals),
        },
    }

    matrix = compose_identifiability_stability_matrix(
        search_scope={"type": "L1_CONSTRAINED_GRID", "grid_points": len(info["dc_scan"])},
        stability_rep=stability_rep,
        adjudication=baseline_adj,
    )

    passed_contract = (
        dist.get("piecewise_hp_inv", 0.0) == 1.0
        and entropy == 0.0
        and baseline_adj is not None
        and baseline_adj.status == EpistemicStatus.MULTI_HYPOTHESIS
    )

    summary = NanoDS10Summary(
        benchmark="NANO-DS10",
        full_name="Stable-Recovery / Non-Identification Benchmark",
        fit_seeds=n_seeds,
        winner_recurrence=dist,
        entropy_nats=entropy,
        k_hp_cv=k_hp_cv,
        adjudication_status=(baseline_adj.status.value if baseline_adj is not None
                             else "NONE"),
        variance_split=v_split,
        matrix=matrix,
        passed_contract=passed_contract,
        question="distinguish structural recurrence from parameter/discovery instability",
    )

    target_json = output_path or (OUTPUTS_DIR / "nano_ds10_freeze.json")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_dict = {
        "benchmark": summary.benchmark,
        "full_name": summary.full_name,
        "fit_seeds": summary.fit_seeds,
        "winner_recurrence": summary.winner_recurrence,
        "entropy_nats": summary.entropy_nats,
        "k_hp_cv": summary.k_hp_cv,
        "question": summary.question,
    }
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=1)

    details = {
        "stability_rep": stability_rep,
        "scoring_matrix": scoring_matrix.tolist(),
        "baseline_adjudication": baseline_adj,
    }

    return summary, details


if __name__ == "__main__":
    print("Executing NANO-DS10 Benchmark...")
    summ, det = run_nano_ds10()
    print(f"Benchmark: {summ.benchmark} ({summ.full_name})")
    print(f"Seeds: {summ.fit_seeds}, Winner Recurrence: {summ.winner_recurrence}")
    print(f"Entropy: {summ.entropy_nats} nats, k_hp CV: {summ.k_hp_cv}")
    print(f"Adjudication Status: {summ.adjudication_status}")
    print(f"Variance Split: {summ.variance_split}")
    print(f"Passed Honest Contract: {summ.passed_contract}")
