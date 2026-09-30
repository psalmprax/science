"""Active Singularity & Pole Stress Testing Engine for APEDP.

Provides analytical inspection and active domain probing for mathematical singularities:
- Discovers zero-crossings and poles in denominators 1/f(x), f(x)^(-k)
- Inspects domain restrictions for logarithms log(f(x)) and roots sqrt(f(x))
- Detects sampling gaps that artificially avoid singular poles (avoided pole traps)
- Detects unobserved asymptotic divergence regimes (critical exponents without near-critical data)
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize as opt
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger("apedp.singularity_probing")


@dataclass(frozen=True)
class SingularityProbeReport:
    """Report detailing singularity stress testing and data gap analysis."""

    has_singularity_risk: bool
    risk_type: str  # "NONE", "AVOIDED_POLE_IN_GAP", "UNOBSERVED_CRITICAL_REGIME", "POLE_IN_DOMAIN"
    pole_locations: list[dict[str, Any]] = field(default_factory=list)
    gap_details: dict[str, Any] | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_singularity_risk": self.has_singularity_risk,
            "risk_type": self.risk_type,
            "pole_locations": self.pole_locations,
            "gap_details": self.gap_details,
            "rationale": self.rationale,
        }


class ActiveSingularityProbeEngine:
    """Performs analytic and numerical singularity stress testing on candidate models."""

    def __init__(
        self,
        domain_buffer_ratio: float = 0.25,
        gap_factor_threshold: float = 1.3,
        blowup_threshold_factor: float = 15.0,
    ):
        self.domain_buffer_ratio = domain_buffer_ratio
        self.gap_factor_threshold = gap_factor_threshold
        self.blowup_threshold_factor = blowup_threshold_factor

    def extract_singular_components(self, expr: sp.Expr | str) -> list[tuple[sp.Expr, str]]:
        """Extracts subexpressions that can induce singular or undefined behavior."""
        if isinstance(expr, str):
            try:
                parsed = safe_parse_expr(expr)
            except Exception:  # noqa: BLE001 - best-effort probe must not crash
                return []
        else:
            parsed = expr

        singular_components: list[tuple[sp.Expr, str]] = []

        # 1. Denominators from together / fraction
        try:
            together_expr = sp.together(parsed)
            _num, denom = sp.fraction(together_expr)
            if denom != 1 and denom != -1 and not denom.is_number:
                singular_components.append((denom, "DENOMINATOR"))
        except Exception:
            logger.debug("Denominator extraction failed", exc_info=True)

        # 2. Walk AST to find localized denominators, powers, logs, and roots
        for node in sp.preorder_traversal(parsed):
            if isinstance(node, sp.Pow):
                base, exp = node.args
                if exp.is_number and exp < 0:
                    singular_components.append((base, "POW_NEGATIVE"))
                elif exp.is_number and 0 < exp < 1:
                    singular_components.append((base, "BRANCH_POINT"))
            elif isinstance(node, sp.log):
                arg = node.args[0]
                singular_components.append((arg, "LOG_ARG"))

        return singular_components

    def find_singular_roots(
        self,
        component: sp.Expr,
        var_name: str,
        domain: tuple[float, float],
    ) -> list[float]:
        """Finds real roots of component(var) = 0 within buffered domain."""
        vmin, vmax = domain
        span = vmax - vmin if vmax > vmin else 1.0
        buf_min = vmin - self.domain_buffer_ratio * span
        buf_max = vmax + self.domain_buffer_ratio * span

        roots: list[float] = []
        sym = sp.Symbol(var_name)
        if sym not in component.free_symbols:
            return roots

        # 1. Analytic solver
        try:
            analytic_sol = sp.solve(component, sym)
            for s in analytic_sol:
                try:
                    if s.is_real or (hasattr(s, "evalf") and abs(sp.im(s.evalf())) < 1e-9):
                        val = float(sp.re(s.evalf()))
                        if buf_min <= val <= buf_max and not any(abs(val - r) < 1e-4 for r in roots):
                            roots.append(val)
                except Exception:
                    logger.debug("Analytic root conversion failed", exc_info=True)
        except Exception:
            logger.debug("Analytic solve failed for %s", component, exc_info=True)

        if roots:
            return roots

        # 2. Numerical sign-change grid scan
        grid = np.linspace(buf_min, buf_max, 300)
        try:
            fn = sp.lambdify(sym, component, modules=["numpy"])
            vals = fn(grid)
            if not isinstance(vals, np.ndarray):
                vals = np.full_like(grid, float(vals))
            vals = np.asarray(vals, dtype=float)

            for i in range(len(grid) - 1):
                if not (math.isfinite(vals[i]) and math.isfinite(vals[i + 1])):
                    continue
                if vals[i] == 0.0:
                    if not any(abs(grid[i] - r) < 1e-4 for r in roots):
                        roots.append(float(grid[i]))
                elif vals[i] * vals[i + 1] < 0:
                    try:
                        r_opt = opt.brentq(fn, grid[i], grid[i + 1])
                        if not any(abs(r_opt - r) < 1e-4 for r in roots):
                            roots.append(float(r_opt))
                    except Exception:
                        logger.debug("brentq refinement failed", exc_info=True)
        except Exception:
            logger.debug("Grid scan failed for %s", component, exc_info=True)

        return roots

    def probe_candidate_singularity(
        self,
        candidate_expr: str,
        competing_expressions: Sequence[str],
        observed_data: Mapping[str, np.ndarray],
    ) -> SingularityProbeReport:
        """Audits candidate and ensemble for avoided poles and unobserved singularity risks."""
        all_exprs = [candidate_expr, *list(competing_expressions)]
        pole_locations: list[dict[str, Any]] = []

        target_vals = None
        if "y" in observed_data:
            target_vals = np.asarray(observed_data["y"], dtype=float)
        elif "chi" in observed_data:
            target_vals = np.asarray(observed_data["chi"], dtype=float)
        y_max = float(np.max(np.abs(target_vals))) if target_vals is not None and len(target_vals) > 0 else 1.0
        y_max = max(y_max, 1e-6)

        # 1. Identify all singular roots across candidate pool
        for expr_str in all_exprs:
            if not expr_str:
                continue
            singular_comps = self.extract_singular_components(expr_str)
            for comp, op_type in singular_comps:
                for var in comp.free_symbols:
                    var_name = str(var)
                    if var_name not in observed_data:
                        continue
                    obs = np.asarray(observed_data[var_name], dtype=float)
                    if len(obs) < 2:
                        continue
                    vmin, vmax = float(np.min(obs)), float(np.max(obs))
                    r_list = self.find_singular_roots(comp, var_name, (vmin, vmax))
                    for r in r_list:
                        if not any(
                            p["variable"] == var_name and abs(p["root"] - r) < 1e-4
                            for p in pole_locations
                        ):
                            pole_locations.append(
                                {
                                    "variable": var_name,
                                    "root": r,
                                    "operator_type": op_type,
                                    "expression_source": expr_str,
                                }
                            )

        if not pole_locations:
            return SingularityProbeReport(
                has_singularity_risk=False,
                risk_type="NONE",
                pole_locations=[],
                rationale="Analytic probing detected no real poles or singularities in domain.",
            )

        # 2. Check each pole against empirical observations and active probe evaluations
        for pole in pole_locations:
            var_name = pole["variable"]
            r_val = pole["root"]
            src_expr = pole["expression_source"]
            obs = np.sort(np.asarray(observed_data[var_name], dtype=float))
            vmin, vmax = float(obs[0]), float(obs[-1])
            avg_spacing = (vmax - vmin) / max(len(obs) - 1, 1)

            # Check for Active Probe Blowup near r_val
            # Evaluate src_expr near r_val
            try:
                sym_var = sp.Symbol(var_name)
                parsed_src = safe_parse_expr(src_expr)
                fn_src = sp.lambdify(sym_var, parsed_src, modules=["numpy"])
                # Probe offsets towards observation domain
                direction = 1.0 if r_val <= vmin else (-1.0 if r_val >= vmax else 1.0)
                probe_offsets = np.array([1e-5, 1e-4, 1e-3, 1e-2]) * direction
                probe_pts = r_val + probe_offsets
                probe_vals = np.abs(np.asarray([float(fn_src(pt)) for pt in probe_pts]))
                max_probe_val = float(np.max(probe_vals))
            except Exception:  # noqa: BLE001 - best-effort probe must not crash
                max_probe_val = float("inf")

            is_divergent = max_probe_val > self.blowup_threshold_factor * y_max or not math.isfinite(max_probe_val)

            # Check A: Pole inside domain range [vmin, vmax] with an avoided sampling gap (P19)
            if vmin < r_val < vmax:
                left_pts = obs[obs < r_val]
                right_pts = obs[obs > r_val]
                x_left = float(left_pts[-1]) if len(left_pts) > 0 else vmin
                x_right = float(right_pts[0]) if len(right_pts) > 0 else vmax
                gap_width = x_right - x_left

                if gap_width >= self.gap_factor_threshold * avg_spacing and is_divergent:
                    return SingularityProbeReport(
                        has_singularity_risk=True,
                        risk_type="AVOIDED_POLE_IN_GAP",
                        pole_locations=pole_locations,
                        gap_details={
                            "variable": var_name,
                            "pole_location": r_val,
                            "gap_left": x_left,
                            "gap_right": x_right,
                            "gap_width": gap_width,
                            "average_spacing": avg_spacing,
                            "max_probe_val": max_probe_val,
                        },
                        rationale=(
                            f"Avoided Pole Trap detected: potential mathematical pole at {var_name} = {r_val:.4f} "
                            f"lies inside an unsampled data gap [{x_left:.3f}, {x_right:.3f}] (width {gap_width:.3f} >= "
                            f"{self.gap_factor_threshold:.1f}x mean spacing {avg_spacing:.3f}), where candidate diverges to "
                            f"{max_probe_val:.2e} (> {self.blowup_threshold_factor}x max observed). "
                            "Observations cannot confirm whether the system is smooth or singular in this gap."
                        ),
                    )

            # Check B: Asymptotic pole near domain boundary with unobserved critical regime (P20, P21)
            min_dist_to_obs = float(np.min(np.abs(obs - r_val)))
            min_dist_to_obs / (vmax - vmin) if vmax > vmin else 0.0

            # If pole is near boundary and expression diverges to > threshold, but observations never saw the divergence:
            if is_divergent and (abs(r_val - vmin) <= 0.25 * (vmax - vmin) or abs(r_val - vmax) <= 0.25 * (vmax - vmin)):
                # Distinguish from canonical central force laws: must compete against a smooth alternative (regular polynomial)
                has_smooth_alternative = any(len(self.extract_singular_components(e)) == 0 for e in competing_expressions)
                if min_dist_to_obs > 0.005 and has_smooth_alternative:
                    return SingularityProbeReport(
                        has_singularity_risk=True,
                        risk_type="UNOBSERVED_CRITICAL_REGIME",
                        pole_locations=pole_locations,
                        gap_details={
                            "variable": var_name,
                            "pole_location": r_val,
                            "min_distance": min_dist_to_obs,
                            "average_spacing": avg_spacing,
                            "max_probe_val": max_probe_val,
                        },
                        rationale=(
                            f"Unobserved Critical Regime: expression has mathematical singularity at {var_name} = {r_val:.4f} "
                            f"with divergent probe behavior ({max_probe_val:.2e} >> {y_max:.2e}), but observations never "
                            f"entered the critical asymptotic divergence regime (min distance {min_dist_to_obs:.3f}). "
                            "Claiming unique discovery of critical singularity over smooth alternatives is epistemically unevidenced."
                        ),
                    )

        return SingularityProbeReport(
            has_singularity_risk=False,
            risk_type="NONE",
            pole_locations=pole_locations,
            rationale="Poles detected but adequate observational coverage exists across neighborhoods.",
        )


__all__ = [
    "ActiveSingularityProbeEngine",
    "SingularityProbeReport",
]
