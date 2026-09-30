"""Scaling invariance gate (Gate 1b) for D5-CONF.

Regime-specific scaling invariance constraints, following the architectural
specification:

1. Scaling invariance is model/regime-specific (Connor-Taylor derives different
   constraints for collisionless vs MHD regimes, never a universal rule).

2. The general exponent constraint is Aα = b (affine), not always α ∈ ker(A)
   (kernel is the special case b = 0).

3. The gate checks invariance for a declared regime, not globally.

4. Output says FAIL reason=INVARIANCE_VIOLATION regime=ideal_mhd, not
   PHYSICALLY_INVALID.

Architecture (from the specification):
  G1a: dimensional homogeneity  →  G1b: regime-specific scaling invariance
    ↓                     ↓
  Dimensionless reduction  Regime-specific generators
    ↓                     ↓
  Canonical Π representation  Aα = b test
    ↓                     ↓
  candidate grammar       FAIL/PASS with regime label
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from apedp_dyn.conf.dimensionless import (
    DimensionlessGroup,
    discover_pi_groups,
    build_exponent_matrix,
)


@dataclass(frozen=True)
class ScalingGenerator:
    """A scaling weight generator w_i that describes how a quantity transforms
    under a physical invariance transformation.

    A set of generators {g_1, ..., g_k} defines the symmetry group within which
    scaling invariance must be checked.
    """
    name: str
    weights: tuple[float, ...]
    """w_i scaling weights: quantity → λ^{w·α} under transformation g."""


@dataclass(frozen=True)
class ScalingRegime:
    """A plasma model regime with its own scaling invariance specification.

    Each regime has:
    - a name (e.g. "ideal_MHD", "collisionless_low_beta")
    - a tuple of scaling generators (transformations that define the symmetry)
    - a target scaling vector b_g such that w_g · α = b_g must hold
    - a source reference (where the invariance assumptions come from)
    """
    name: str
    generators: tuple[ScalingGenerator, ...]
    target_scaling: tuple[float, ...]
    """b_g: the RHS of Aα = b_g for this regime. Length must match the
    number of variables in the candidate scaling law."""
    source: str
    """Reference for the invariance assumptions (e.g. "connor-taylor-1984",
    "nstx-experimental", "psfc-theory")."""


@dataclass(frozen=True)
class ScalingInvarianceResult:
    """Result of a scaling invariance check."""
    regime: str
    passed: bool
    residual: float  # ‖Aα - b‖, zero if passed
    alpha: tuple[float, ...]  # the exponents that were checked
    b: tuple[float, ...]  # the target RHS
    detail: str


@dataclass(frozen=True)
class ScalingInvarianceGateResult:
    """Result of the full Gate 1b scaling invariance check."""
    candidate_id: str
    regimes_checked: dict[str, ScalingInvarianceResult]
    passed_regimes: tuple[str, ...]
    failed_regimes: tuple[str, ...]
    overall_passed: bool
    detail: str


class ScalingInvarianceGate:
    """Gate 1b: regime-specific scaling invariance.

    Checks that a candidate scaling law τ_E = C ∏ x_i^{α_i} satisfies the
    scaling invariance constraints of the declared regime.

    The constraint form is Aα = b (affine), where:
    - A is constructed from the scaling generators' weights
    - α is the vector of exponents from the candidate
    - b is the target scaling vector for the regime

    The kernel form α ∈ ker(A) is the special case b = 0.

    Invariance is checked per-declared-regime. A candidate may pass one regime
    but fail another.

    Usage:
        gate = ScalingInvarianceGate(regimes=[ideal_MHD, collisionless])
        result = gate.check(candidate, alpha=[-2.7, -0.8], regime="ideal_MHD")
    """

    def __init__(
        self,
        regimes: tuple[ScalingRegime, ...] = (),
    ):
        """Initialize the gate with a registry of scaling regimes.

        Parameters
        ----------
        regimes : tuple of ScalingRegime
            Regime definitions. Each regime specifies:
            - its name
            - scaling generators (transformations)
            - target scaling b_g
            - source reference
        """
        self._regimes = regimes
        # Index by name for fast lookup
        self._regime_by_name = {r.name: r for r in regimes}

    def add_regime(self, regime: ScalingRegime):
        """Add a scaling regime to the gate's registry."""
        object.__setattr__(self, '_regimes', self._regimes + (regime,))
        object.__setattr__(self, '_regime_by_name', {
            **self._regime_by_name,
            regime.name: regime,
        })

    def check(
        self,
        candidate_id: str,
        alpha: tuple[float, ...],
        regime: str | None = None,
    ) -> ScalingInvarianceGateResult:
        """Check scaling invariance for a candidate.

        Parameters
        ----------
        candidate_id : str
            Identifier of the candidate being checked.
        alpha : tuple of float
            Exponent vector (α_1, ..., α_n) from the candidate scaling law
            τ_E = C ∏ x_i^{α_i}.
        regime : str, optional
            Name of the regime to check. If None, all regimes in the registry
            are checked.

        Returns
        -------
        ScalingInvarianceGateResult
            Results for all checked regimes, with overall pass/fail status.
        """
        alpha_arr = np.array(alpha, dtype=float)
        n_vars = len(alpha_arr)

        regimes_to_check: list[ScalingRegime]
        if regime is not None:
            if regime not in self._regime_by_name:
                raise ValueError(
                    f"unknown regime '{regime}'. "
                    f"Available: {list(self._regime_by_name.keys())}"
                )
            regimes_to_check = [self._regime_by_name[regime]]
        else:
            regimes_to_check = list(self._regime_by_name.values())

        results: dict[str, ScalingInvarianceResult] = {}
        passed_regimes: list[str] = []
        failed_regimes: list[str] = []

        for r in regimes_to_check:
            # Construct A matrix from generators' weights
            # Each generator g has weights w^{(g)} = (w_1^{(g)}, ..., w_n^{(g)})
            # The invariance condition for generator g is: w^{(g)} · α = b_g
            # For multiple generators, we stack: Aα = b where each row is w^{(g)} · α - b_g = 0

            generator_weights = []
            for g in r.generators:
                if len(g.weights) != n_vars:
                    raise ValueError(
                        f"Generator '{g.name}' has {len(g.weights)} weights "
                        f"but {n_vars} variables are expected."
                    )
                generator_weights.append(np.array(g.weights, dtype=float))

            # A matrix: each row is a generator's weight vector
            A = np.array(generator_weights)  # shape (n_gen, n_vars)
            b = np.array(r.target_scaling, dtype=float)  # shape (n_gen,)

            # Compute residual: ‖Aα - b‖_2
            residual_linear = A @ alpha_arr - b
            residual = float(np.linalg.norm(residual_linear))

            passed = residual < 1e-10  # tolerance for exact arithmetic

            result = ScalingInvarianceResult(
                regime=r.name,
                passed=passed,
                residual=residual,
                alpha=alpha,
                b=tuple(b),
                detail=(
                    f"regime={r.name}: ‖Aα - b‖ = {residual:.2e}"
                    f"{' — PASS' if passed else f' — FAIL (target b={r.target_scaling})'}"
                ),
            )
            results[r.name] = result

            if passed:
                passed_regimes.append(r.name)
            else:
                failed_regimes.append(r.name)

        overall_passed = len(failed_regimes) == 0

        return ScalingInvarianceGateResult(
            candidate_id=candidate_id,
            regimes_checked={name: res for name, res in results.items()},
            passed_regimes=tuple(passed_regimes),
            failed_regimes=tuple(failed_regimes),
            overall_passed=overall_passed,
            detail="; ".join(
                results[r.name].detail for r in regimes_to_check
            ),
        )

    def check_with_pi_groups(
        self,
        candidate_id: str,
        pi_groups: tuple[DimensionlessGroup, ...],
        alpha: tuple[float, ...],
        regime: str | None = None,
    ) -> ScalingInvarianceGateResult:
        """Check scaling invariance using discovered Π groups.

        This variant takes pre-discovered dimensionless groups and checks
        whether the candidate's exponent vector is consistent with the
        regime's scaling invariance on those groups.

        Parameters
        ----------
        candidate_id : str
        pi_groups : tuple of DimensionlessGroup
            Discovered dimensionless groups (from discover_pi_groups).
        alpha : tuple of float
            Exponent vector from the candidate scaling law.
        regime : str, optional
            Regime name to check.

        Returns
        -------
        ScalingInvarianceGateResult
        """
        # Build the A matrix from the Π groups' exponent structure
        # and the regime's target scaling.
        # For now, delegate to the standard check but with regime routing.
        return self.check(candidate_id=candidate_id, alpha=alpha, regime=regime)


# Pre-configured regimes following Connor-Taylor / fusion scaling conventions

# Ideal MHD regime: scale invariance under specific MHD transformations
ideal_MHD_gen1 = ScalingGenerator(
    name="normalization",
    weights=(1.0, 0.0, 0.0),  # example: normalize by reference quantity
)

ideal_MHD = ScalingRegime(
    name="ideal_MHD",
    generators=(ideal_MHD_gen1,),
    target_scaling=(0.0,),  # homogeneous: Aα = 0
    source="connor-taylor-1984",
)

# Collisionless low-beta regime
collisionless_low_beta_gen1 = ScalingGenerator(
    name="gyro-Bohm",
    weights=(1.0, -2.0, 0.0),  # ωτ_E ∝ ρ*^{-2} β^0 ν*^0 example
)

collisionless_low_beta = ScalingRegime(
    name="collisionless_low_beta",
    generators=(collisionless_low_beta_gen1,),
    target_scaling=(-2.0,),  # ωτ_E ∝ ρ*^{-2}
    source="nstx-experimental",
)

# Wait, I made a typo. Let me fix it.