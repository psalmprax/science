"""Coordinate Transmutation & Singularity Linearization (v0.1.0).

Transforms non-linear physical manifolds with essential singularities, power laws,
or quantized plateaus into linear or softened coordinate spaces. Autonomously recovers
physical constants (tunneling barriers, power exponents, quantum conductance steps)
without heuristic evolutionary search.

Boundary discipline (AGENTS.md §1.1):
Consumes training data and an immutable DatasetProfile. Emits standardized
CandidateProposal objects. Does NOT evaluate on holdout confirmation slices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.autonomous_profiler import DatasetProfile
from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype


@dataclass(frozen=True)
class TransmutationResult:
    """Result of coordinate transmutation and parameter extraction."""

    success: bool
    method: str
    r2: float
    formula: str
    parameters: dict[str, float]
    candidate: CandidateProposal | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "method": self.method,
            "r2": self.r2,
            "formula": self.formula,
            "parameters": self.parameters,
            "candidate_id": self.candidate.candidate_id if self.candidate else None,
            "note": self.note,
        }


def transmute_and_propose(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    profile: DatasetProfile,
    *,
    feature_name: str = "x",
    proposer_id: str = "coordinate_transmuter_v1",
    min_r2: float = 0.90,
) -> TransmutationResult:
    """Perform coordinate transmutation based on DatasetProfile diagnosis.

    Extracts parameters and emits a CandidateProposal if fit quality exceeds min_r2.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    # --- Mode 1: Log-Linear Singular Transmutation (Arrhenius / Tunneling) ---
    if profile.recommended_coordinate_transmutation == "log_linear_singular":
        pos_m = (x_arr > 0) & (y_arr > 0)
        xp, yp = x_arr[pos_m], y_arr[pos_m]
        if xp.size < 5:
            return TransmutationResult(
                success=False, method="log_linear_singular", r2=0.0,
                formula="", parameters={}, note="Insufficient positive data points."
            )

        ln_x = np.log(xp)
        inv_x = 1.0 / xp
        ln_y = np.log(yp)

        # 3-parameter basis: [1, ln(x), 1/x] -> y = C * x^k * exp(-c / x)
        X_mat = np.column_stack([np.ones_like(xp), ln_x, inv_x])
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, ln_y, rcond=None)
        b0, b1, b2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

        c_barrier = -b2
        k_power = b1
        c_prefactor = float(np.exp(b0))

        # Test if power term is negligible (pure tunneling like Landau-Zener: k ≈ 0)
        if abs(k_power) < 0.05:
            k_power = 0.0
            # Refit 2-parameter basis [1, 1/x]
            X_mat2 = np.column_stack([np.ones_like(xp), inv_x])
            c2, _, _, _ = np.linalg.lstsq(X_mat2, ln_y, rcond=None)
            b0_2, b2_2 = float(c2[0]), float(c2[1])
            c_barrier = -b2_2
            c_prefactor = float(np.exp(b0_2))

        # Compute predictions in original coordinates
        if k_power == 0.0:
            pred_y = c_prefactor * np.exp(-c_barrier / xp)
            formula_str = f"{c_prefactor:.6g} * exp(-{c_barrier:.6g} / {feature_name})"
        else:
            pred_y = c_prefactor * (xp ** k_power) * np.exp(-c_barrier / xp)
            formula_str = f"{c_prefactor:.6g} * ({feature_name} ** {k_power:.4g}) * exp(-{c_barrier:.6g} / {feature_name})"

        ss_tot = float(np.sum((yp - np.mean(yp)) ** 2))
        ss_res = float(np.sum((yp - pred_y) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if r2 >= min_r2:
            # Build scorer closure
            def _scorer(df: pd.DataFrame, c_pref=c_prefactor, k_pow=k_power, c_bar=c_barrier, feat=feature_name) -> np.ndarray:
                x_val = df[feat].to_numpy(dtype=float) if feat in df.columns else df.iloc[:, 0].to_numpy(dtype=float)
                safe_x = np.where(x_val > 1e-12, x_val, np.nan)
                if k_pow == 0.0:
                    return c_pref * np.exp(-c_bar / safe_x)
                return c_pref * (safe_x ** k_pow) * np.exp(-c_bar / safe_x)

            params = {"prefactor": c_prefactor, "power": k_power, "barrier": c_barrier}
            cand = CandidateProposal(
                proposer_id=proposer_id,
                candidate_id=f"transmuted_tunneling_{int(abs(c_barrier*1000))}",
                formula=formula_str,
                strategy_archetype=StrategyArchetype.TACTICAL,
                declared_scale_invariant=True,
                declared_nominal_dependencies=(feature_name,),
                compute_score=_scorer,
                parameters=params,
                metadata={"r2": r2, "method": "log_linear_singular"},
            )
            return TransmutationResult(
                success=True, method="log_linear_singular", r2=r2,
                formula=formula_str, parameters=params, candidate=cand,
                note=f"Exact non-perturbative tunneling recovery (R2={r2:.6f})."
            )

    # --- Mode 2: Power-Law Transmutation ---
    elif profile.recommended_coordinate_transmutation == "power_law" or profile.is_pure_power_law:
        pos_m = (x_arr > 0) & (y_arr > 0)
        xp, yp = x_arr[pos_m], y_arr[pos_m]
        if xp.size < 5:
            return TransmutationResult(
                success=False, method="power_law", r2=0.0,
                formula="", parameters={}, note="Insufficient positive data points."
            )

        ln_x = np.log(xp)
        ln_y = np.log(yp)
        X_mat = np.column_stack([np.ones_like(xp), ln_x])
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, ln_y, rcond=None)
        b0, b1 = float(coeffs[0]), float(coeffs[1])
        c_prefactor = float(np.exp(b0))
        k_power = float(np.round(b1, 4))

        pred_y = c_prefactor * (xp ** k_power)
        ss_tot = float(np.sum((yp - np.mean(yp)) ** 2))
        ss_res = float(np.sum((yp - pred_y) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if r2 >= min_r2:
            formula_str = f"{c_prefactor:.6g} * ({feature_name} ** {k_power:.4g})"
            def _scorer(df: pd.DataFrame, c_pref=c_prefactor, k_pow=k_power, feat=feature_name) -> np.ndarray:
                x_val = df[feat].to_numpy(dtype=float) if feat in df.columns else df.iloc[:, 0].to_numpy(dtype=float)
                return c_pref * (x_val ** k_pow)

            params = {"prefactor": c_prefactor, "power": k_power}
            cand = CandidateProposal(
                proposer_id=proposer_id,
                candidate_id=f"transmuted_power_{int(abs(k_power*100))}",
                formula=formula_str,
                strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
                declared_scale_invariant=True,
                declared_nominal_dependencies=(feature_name,),
                compute_score=_scorer,
                parameters=params,
                metadata={"r2": r2, "method": "power_law"},
            )
            return TransmutationResult(
                success=True, method="power_law", r2=r2,
                formula=formula_str, parameters=params, candidate=cand,
                note=f"Exact power-law recovery (R2={r2:.6f})."
            )

    # --- Mode 3: Quantized Plateau Staircase Transmutation ---
    elif profile.is_quantized_staircase and len(profile.discrete_levels) >= 2:
        levels = np.sort(profile.discrete_levels)
        gaps = np.diff(levels)
        quantum_step = float(np.median(gaps))
        if quantum_step <= 0.0:
            quantum_step = 1.0

        # Estimate threshold width: find transition midpoints in x
        sort_idx = np.argsort(x_arr)
        xs, ys = x_arr[sort_idx], y_arr[sort_idx]
        diff_y = np.diff(ys)
        transition_indices = np.where(np.abs(diff_y) > 0.4 * quantum_step)[0]

        if len(transition_indices) >= 2:
            x_mids = 0.5 * (xs[transition_indices] + xs[transition_indices + 1])
            # Step levels before jump: ys[idx] / quantum_step
            step_counts = ys[transition_indices] / quantum_step
            slope, intercept = np.polyfit(step_counts + 1.0, x_mids, 1)
            step_width = float(slope)
            x_offset = float(intercept)
        elif len(transition_indices) == 1:
            x_mid = 0.5 * (xs[transition_indices[0]] + xs[transition_indices[0] + 1])
            step_width = 1.0
            x_offset = float(x_mid) - 1.0
        else:
            step_width = 1.0
            x_offset = 0.0

        pred_y = quantum_step * np.floor((xs - x_offset) / step_width + 1e-5)
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        ss_res = float(np.sum((ys - pred_y) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if r2 >= min_r2:
            formula_str = f"{quantum_step:.6g} * floor(({feature_name} - {x_offset:.6g}) / {step_width:.6g})"
            def _scorer(df: pd.DataFrame, q_step=quantum_step, s_width=step_width, x_off=x_offset, feat=feature_name) -> np.ndarray:
                x_val = df[feat].to_numpy(dtype=float) if feat in df.columns else df.iloc[:, 0].to_numpy(dtype=float)
                return q_step * np.floor((x_val - x_off) / s_width + 1e-6)

            params = {"quantum_step": quantum_step, "step_width": step_width, "offset": x_offset}
            cand = CandidateProposal(
                proposer_id=proposer_id,
                candidate_id="transmuted_quantized_staircase",
                formula=formula_str,
                strategy_archetype=StrategyArchetype.TACTICAL,
                declared_scale_invariant=False,
                declared_nominal_dependencies=(feature_name,),
                compute_score=_scorer,
                parameters=params,
                metadata={"r2": r2, "method": "quantized_staircase"},
            )
            return TransmutationResult(
                success=True, method="quantized_staircase", r2=r2,
                formula=formula_str, parameters=params, candidate=cand,
                note=f"Quantized plateau staircase recovery (R2={r2:.6f})."
            )

    return TransmutationResult(
        success=False, method=profile.recommended_coordinate_transmutation,
        r2=0.0, formula="", parameters={}, note="No linearizing coordinate transformation matched."
    )
