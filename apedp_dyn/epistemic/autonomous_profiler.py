"""Autonomous Epistemic Profiler (v0.1.0).

Automated diagnostic pre-scan and regime classification for physical datasets.
Detects scale invariance, essential singularities, non-perturbative tunneling,
quantized plateau staircases, and dual-asymptotic boundary crossovers.

Boundary discipline (AGENTS.md §1.1):
This module is a diagnostic observer. It consumes dataset arrays, measures
topological and statistical invariants, and emits an immutable DatasetProfile.
It does NOT decide candidate acceptance and does NOT inspect holdout confirmation slices.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """Immutable diagnostic profile of a physical target dataset."""

    n_rows: int
    dynamic_range: float
    recommended_loss: str
    is_arrhenius_tunneling: bool
    arrhenius_scale: float
    arrhenius_cv: float
    is_pure_power_law: bool
    power_law_exponent: float
    power_law_cv: float
    is_quantized_staircase: bool
    discrete_levels: tuple[float, ...]
    flat_derivative_fraction: float
    is_dual_asymptotic: bool
    low_x_exponent: float
    high_x_exponent: float
    recommended_grammar: tuple[str, ...]
    recommended_coordinate_transmutation: str
    profile_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "dynamic_range": self.dynamic_range,
            "recommended_loss": self.recommended_loss,
            "is_arrhenius_tunneling": self.is_arrhenius_tunneling,
            "arrhenius_scale": self.arrhenius_scale,
            "arrhenius_cv": self.arrhenius_cv,
            "is_pure_power_law": self.is_pure_power_law,
            "power_law_exponent": self.power_law_exponent,
            "power_law_cv": self.power_law_cv,
            "is_quantized_staircase": self.is_quantized_staircase,
            "discrete_levels": list(self.discrete_levels),
            "flat_derivative_fraction": self.flat_derivative_fraction,
            "is_dual_asymptotic": self.is_dual_asymptotic,
            "low_x_exponent": self.low_x_exponent,
            "high_x_exponent": self.high_x_exponent,
            "recommended_grammar": list(self.recommended_grammar),
            "recommended_coordinate_transmutation": self.recommended_coordinate_transmutation,
            "profile_hash": self.profile_hash,
        }


def profile_dataset(
    x: np.ndarray | pd.Series | Sequence[float],
    y: np.ndarray | pd.Series | Sequence[float],
    *,
    dynamic_range_threshold: float = 100.0,
    arrhenius_cv_cut: float = 0.35,
    power_law_cv_cut: float = 0.15,
    flat_slope_cut: float = 1e-4,
    flat_fraction_cut: float = 0.35,
    max_discrete_levels: int = 20,
    dual_asymptotic_margin: float = 1.2,
) -> DatasetProfile:
    """Analyze (x, y) arrays and autonomously extract structural invariants.

    All diagnostic cutoffs are caller-supplied arguments (preserving AGENTS.md §1.3).
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[finite_mask], y_arr[finite_mask]
    n_rows = int(x_arr.size)

    if n_rows < 5:
        # Insufficient data for reliable topological diagnosis
        h = hashlib.sha256(b"insufficient_data").hexdigest()
        return DatasetProfile(
            n_rows=n_rows,
            dynamic_range=1.0,
            recommended_loss="standard_mse",
            is_arrhenius_tunneling=False,
            arrhenius_scale=0.0,
            arrhenius_cv=float("nan"),
            is_pure_power_law=False,
            power_law_exponent=0.0,
            power_law_cv=float("nan"),
            is_quantized_staircase=False,
            discrete_levels=(),
            flat_derivative_fraction=0.0,
            is_dual_asymptotic=False,
            low_x_exponent=0.0,
            high_x_exponent=0.0,
            recommended_grammar=(),
            recommended_coordinate_transmutation="none",
            profile_hash=h,
        )

    # Sort strictly by x
    sort_idx = np.argsort(x_arr)
    x_s, y_s = x_arr[sort_idx], y_arr[sort_idx]

    # 1. Dynamic range test
    pos_mask = (y_s > 0) & (x_s > 0)
    xp, yp = x_s[pos_mask], y_s[pos_mask]

    if yp.size > 0:
        dyn_range = float(np.max(yp) / np.min(yp))
    else:
        dyn_range = float(np.ptp(y_s)) if np.ptp(y_s) > 0 else 1.0

    recommended_loss = "log_mse" if (dyn_range >= dynamic_range_threshold and yp.size == y_s.size) else "standard_mse"

    # 2. Arrhenius / Essential Singularity Tunneling test: d(ln y) / d(1/x)
    is_arrh = False
    arrh_scale = 0.0
    arrh_cv = float("nan")

    if xp.size >= 8:
        inv_x = 1.0 / xp
        d_inv_x = np.diff(inv_x)
        d_ln_y = np.diff(np.log(yp))
        valid_diff = np.abs(d_inv_x) > 1e-12
        if np.sum(valid_diff) >= 5:
            slopes = d_ln_y[valid_diff] / d_inv_x[valid_diff]
            arrh_mean = float(np.mean(slopes))
            arrh_std = float(np.std(slopes))
            arrh_cv = abs(arrh_std / arrh_mean) if abs(arrh_mean) > 1e-6 else 999.0
            arrh_scale = -arrh_mean
            if arrh_cv < arrhenius_cv_cut and arrh_mean < 0 and dyn_range > 10.0:
                is_arrh = True

    # 3. Pure Power-Law Scaling test: d(ln y) / d(ln x)
    is_power = False
    power_exp = 0.0
    power_cv = float("nan")

    if xp.size >= 8:
        d_ln_x = np.diff(np.log(xp))
        d_ln_y = np.diff(np.log(yp))
        valid_ln_diff = np.abs(d_ln_x) > 1e-12
        if np.sum(valid_ln_diff) >= 5:
            p_slopes = d_ln_y[valid_ln_diff] / d_ln_x[valid_ln_diff]
            p_mean = float(np.mean(p_slopes))
            p_std = float(np.std(p_slopes))
            power_cv = abs(p_std / p_mean) if abs(p_mean) > 1e-6 else 999.0
            power_exp = float(np.round(p_mean, 4))
            if power_cv < power_law_cv_cut and abs(power_exp) > 1e-3:
                is_power = True

    # 4. Quantized Plateau / Staircase Topology test
    dx = np.diff(x_s)
    dy = np.diff(y_s)
    valid_dx = np.abs(dx) > 1e-12
    flat_fraction = 0.0
    is_staircase = False
    discrete_levels: list[float] = []

    if np.sum(valid_dx) >= 5:
        numerical_slopes = np.abs(dy[valid_dx] / dx[valid_dx])
        flat_fraction = float(np.mean(numerical_slopes < flat_slope_cut))
        # Cluster unique target levels
        rounded_y = np.round(y_s, 3)
        unique_vals, _counts = np.unique(rounded_y, return_counts=True)
        if len(unique_vals) <= max_discrete_levels and flat_fraction >= flat_fraction_cut:
            is_staircase = True
            discrete_levels = [float(v) for v in unique_vals]

    # 5. Dual-Asymptotic Regime test (Low-x vs High-x scaling crossover)
    is_dual = False
    low_x_exp = 0.0
    high_x_exp = 0.0

    if xp.size >= 20 and not is_power and not is_staircase:
        n_third = xp.size // 3
        x_low, y_low = xp[:n_third], yp[:n_third]
        x_high, y_high = xp[-n_third:], yp[-n_third:]
        if np.std(np.log(x_low)) > 1e-6 and np.std(np.log(x_high)) > 1e-6:
            p_low = float(np.polyfit(np.log(x_low), np.log(y_low), 1)[0])
            p_high = float(np.polyfit(np.log(x_high), np.log(y_high), 1)[0])
            low_x_exp = float(np.round(p_low, 3))
            high_x_exp = float(np.round(p_high, 3))
            if abs(p_low - p_high) > dual_asymptotic_margin:
                is_dual = True

    # 6. Synthesize Recommended Grammar & Coordinate Transmutation
    grammar_recs: list[str] = []
    transmutation = "none"

    if is_arrh:
        grammar_recs.append("arrh")
        transmutation = "log_linear_singular"
    elif is_power:
        transmutation = "power_law"
        if power_exp < 0:
            grammar_recs.extend(["inv", "inv_sq"])
    elif is_staircase:
        grammar_recs.extend(["floor", "ceil"])
    elif is_dual:
        grammar_recs.extend(["bose", "planck", "lor_norm"])

    # Compute deterministic profile hash
    blob = json.dumps(
        {
            "n_rows": n_rows,
            "dyn_range": round(dyn_range, 4),
            "is_arrh": is_arrh,
            "arrh_scale": round(arrh_scale, 4),
            "is_power": is_power,
            "power_exp": round(power_exp, 4),
            "is_staircase": is_staircase,
            "discrete_levels": [round(lv, 3) for lv in discrete_levels],
            "is_dual": is_dual,
            "low_x_exp": round(low_x_exp, 3),
            "high_x_exp": round(high_x_exp, 3),
            "transmutation": transmutation,
        },
        sort_keys=True,
    )
    profile_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    return DatasetProfile(
        n_rows=n_rows,
        dynamic_range=dyn_range,
        recommended_loss=recommended_loss,
        is_arrhenius_tunneling=is_arrh,
        arrhenius_scale=arrh_scale,
        arrhenius_cv=arrh_cv,
        is_pure_power_law=is_power,
        power_law_exponent=power_exp,
        power_law_cv=power_cv,
        is_quantized_staircase=is_staircase,
        discrete_levels=tuple(discrete_levels),
        flat_derivative_fraction=flat_fraction,
        is_dual_asymptotic=is_dual,
        low_x_exponent=low_x_exp,
        high_x_exponent=high_x_exp,
        recommended_grammar=tuple(grammar_recs),
        recommended_coordinate_transmutation=transmutation,
        profile_hash=profile_hash,
    )
