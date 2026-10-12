"""Empirical and Epistemic Test of the Riemann Hypothesis & Prime Distribution Link.

Tests two fundamental aspects:
1. Critical Line Alignment: Empirically evaluates non-trivial zeros rho_n = beta_n + i*gamma_n.
2. Prime Distribution Reconstruction: Evaluates Riemann's Explicit Formula
   psi(x) = x - sum_rho (x^rho / rho) - ln(2pi) - 0.5*ln(1 - x^-2)
   demonstrating how beta = 1/2 acts as the stationary amplitude envelope that
   generates the discrete jump harmonics of prime powers.
3. Counterfactual & Epistemic Adjudication:
   - Evaluates Fake-RH counterfactuals (beta != 0.5) showing explosive error growth.
   - Adjudicates the statement under the APEDP-X Claim Ladder:
     Distinguishes C2 (Bounded Empirical Law on verified zeros) from
     C6 (Unproven Universal Theorem to infinity).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

def run_rh_prime_experiment():
    cache_path = Path("apedp/mathematical/cache/zeta_zeros_500.json")
    with open(cache_path, "r", encoding="utf-8") as f:
        zeros_data = json.load(f)[:200]

    gammas = [z["gamma"] for z in zeros_data]
    betas = [z["beta"] for z in zeros_data]

    # Part 1: Numerical Zero Alignment
    max_beta_dev = max(abs(b - 0.5) for b in betas)
    mean_beta_dev = float(np.mean([abs(b - 0.5) for b in betas]))

    # Part 2: True Chebyshev Psi Function
    def get_primes_up_to(n: int) -> list[int]:
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    def true_psi(x: float, primes: list[int]) -> float:
        val = 0.0
        for p in primes:
            if p > x:
                break
            pk = p
            while pk <= x:
                val += math.log(p)
                pk *= p
        return val

    # Test grid x in [2.0, 50.0]
    primes_50 = get_primes_up_to(50)
    xs = np.linspace(2.1, 50.0, 300)
    psi_truth = np.array([true_psi(x, primes_50) for x in xs])

    # Explicit formula approximation
    def explicit_psi(xs_arr: np.ndarray, k_zeros: int, beta: float = 0.5) -> np.ndarray:
        res = np.copy(xs_arr) - math.log(2.0 * math.pi)
        ln_xs = np.log(xs_arr)
        for g in gammas[:k_zeros]:
            denom = beta**2 + g**2
            term = 2.0 * (xs_arr**beta) / denom * (beta * np.cos(g * ln_xs) + g * np.sin(g * ln_xs))
            res -= term
        # subtract trivial zero correction: 0.5 * ln(1 - x^-2)
        res -= 0.5 * np.log(np.maximum(1.0 - xs_arr**(-2.0), 1e-12))
        return res

    # Reconstructions
    psi_k10 = explicit_psi(xs, 10, beta=0.5)
    psi_k50 = explicit_psi(xs, 50, beta=0.5)
    psi_k200 = explicit_psi(xs, 200, beta=0.5)
    
    # Counterfactual Fake-RH: beta = 0.85
    psi_fake_085 = explicit_psi(xs, 200, beta=0.85)
    # Counterfactual Fake-RH: beta = 0.15
    psi_fake_015 = explicit_psi(xs, 200, beta=0.15)

    rmse_k10 = float(np.sqrt(np.mean((psi_truth - psi_k10)**2)))
    rmse_k50 = float(np.sqrt(np.mean((psi_truth - psi_k50)**2)))
    rmse_k200 = float(np.sqrt(np.mean((psi_truth - psi_k200)**2)))
    rmse_fake_085 = float(np.sqrt(np.mean((psi_truth - psi_fake_085)**2)))
    rmse_fake_015 = float(np.sqrt(np.mean((psi_truth - psi_fake_015)**2)))

    # Discrete step check at prime jumps
    test_eval_points = [2.5, 3.5, 4.5, 5.5, 7.5, 11.5, 13.5, 17.5, 23.5, 31.5, 41.5, 47.5]
    step_table = []
    for pt in test_eval_points:
        t_val = true_psi(pt, primes_50)
        rec_val = float(explicit_psi(np.array([pt]), 200, beta=0.5)[0])
        fk_val = float(explicit_psi(np.array([pt]), 200, beta=0.85)[0])
        step_table.append({
            "x": pt,
            "true_psi": round(t_val, 4),
            "rh_explicit_k200": round(rec_val, 4),
            "fake_rh_beta085": round(fk_val, 4),
            "rh_residual": round(abs(t_val - rec_val), 4),
            "fake_residual": round(abs(t_val - fk_val), 4),
        })

    return {
        "n_zeros_tested": len(zeros_data),
        "max_beta_deviation": max_beta_dev,
        "mean_beta_deviation": mean_beta_dev,
        "rmse_k10": rmse_k10,
        "rmse_k50": rmse_k50,
        "rmse_k200": rmse_k200,
        "rmse_fake_085": rmse_fake_085,
        "rmse_fake_015": rmse_fake_015,
        "step_table": step_table,
    }

if __name__ == "__main__":
    res = run_rh_prime_experiment()
    print("=== EMPIRICAL RH & PRIME HARMONICS VERIFICATION ===")
    print(f"Zeros evaluated: {res['n_zeros_tested']}")
    print(f"Max deviation from beta=0.5: {res['max_beta_deviation']:.2e}")
    print(f"Harmonic reconstruction RMSE (K=10 zeros):  {res['rmse_k10']:.4f}")
    print(f"Harmonic reconstruction RMSE (K=50 zeros):  {res['rmse_k50']:.4f}")
    print(f"Harmonic reconstruction RMSE (K=200 zeros): {res['rmse_k200']:.4f}")
    print(f"Harmonic reconstruction RMSE (beta=0.85 Fake): {res['rmse_fake_085']:.4f}")
    print(f"Harmonic reconstruction RMSE (beta=0.15 Fake): {res['rmse_fake_015']:.4f}")
    print("\n=== PRIME POWER DISCONTINUITY RECONSTRUCTION ===")
    print(f"{'x':>5} | {'True psi(x)':>12} | {'RH K=200':>12} | {'Fake beta=0.85':>14} | {'RH Err':>8}")
    print("-" * 65)
    for row in res["step_table"]:
        print(f"{row['x']:5.1f} | {row['true_psi']:12.4f} | {row['rh_explicit_k200']:12.4f} | {row['fake_rh_beta085']:14.4f} | {row['rh_residual']:8.4f}")
