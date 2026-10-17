"""Modular Multi-Engine STEM Proposers (Option 2).

Specialized domain discovery engines inheriting ProposerBase:
1. HamiltonianProposer: Discovers energy-conserving mechanical systems (dH/dt = 0).
2. SolenoidalFluidProposer: Discovers divergence-free fluid velocity fields (div u = 0).
3. CompartmentalSirProposer: Discovers epidemiological compartmental dynamics (S+I+R = N).

Boundary discipline (AGENTS.md §1.1):
Proposers only generate hypotheses and CandidateProposal objects.
Adjudication and acceptance decisions belong strictly to the Epistemic Referee.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class HamiltonianProposer(ProposerBase):
    """Discovers energy-conserving Hamiltonian mechanical systems.

    Hamilton's equations:
        dq/dt = dH/dp = p / m
        dp/dt = -dH/dq = -k*q - alpha*q^3

    Invariant:
        dH/dt = 0 (exact energy conservation on closed orbits)
    """

    def __init__(
        self,
        dt: float = 0.01,
        include_nonlinear: bool = True,
        max_energy_drift: float = 0.05,
    ) -> None:
        self.dt = float(dt)
        self.include_nonlinear = bool(include_nonlinear)
        self.max_energy_drift = float(max_energy_drift)

    @property
    def proposer_id(self) -> str:
        return "hamiltonian_mechanical_proposer_v1"

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame] | pd.DataFrame,
        target_col: str = "p_dot",
        q_col: str = "q",
        p_col: str = "p",
    ) -> Sequence[CandidateProposal]:
        """Fit conservative dynamics and propose candidate Hamiltonian laws."""
        df = train_data if isinstance(train_data, pd.DataFrame) else next(iter(train_data.values()))
        if q_col not in df.columns or p_col not in df.columns or len(df) < 20:
            return []

        q = df[q_col].to_numpy(dtype=float)
        p = df[p_col].to_numpy(dtype=float)

        # Estimate derivatives
        q_dot = np.gradient(q, self.dt)
        p_dot = np.gradient(p, self.dt)

        # Fit mass m: dq/dt = p / m => m = mean(p / dq/dt) via least squares: p = m * q_dot
        denom_q = float(np.sum(q_dot ** 2))
        if denom_q < 1e-12:
            return []
        m = float(np.sum(p * q_dot) / denom_q)
        if m <= 0.0 or not np.isfinite(m):
            m = 1.0

        # Fit potential force: dp/dt = -k*q (- alpha*q^3)
        features = [q]
        names = ["q"]
        if self.include_nonlinear:
            features.append(q ** 3)
            names.append("q^3")

        X = np.column_stack(features)
        y = -p_dot  # positive force -dp/dt = dV/dq

        coeffs, _residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        k = float(coeffs[0])
        alpha = float(coeffs[1]) if len(coeffs) > 1 else 0.0

        # Compute energy H(t) = 0.5 * p^2 / m + 0.5 * k * q^2 + 0.25 * alpha * q^4
        H = 0.5 * (p ** 2) / m + 0.5 * k * (q ** 2)
        if abs(alpha) > 1e-12:
            H += 0.25 * alpha * (q ** 4)

        # Verify energy conservation invariant
        h_range = float(np.ptp(H))
        h_mean = float(np.abs(np.mean(H))) + 1e-12
        drift = h_range / h_mean

        if drift > self.max_energy_drift:
            # Energy is not conserved within tolerance
            return []

        if abs(alpha) > 1e-6:
            formula = f"H(q, p) = 0.5*{1/m:.4f}*p^2 + 0.5*{k:.4f}*q^2 + 0.25*{alpha:.4f}*q^4"
            acc_formula = f"-{k:.4f}*q - {alpha:.4f}*q^3"
        else:
            formula = f"H(q, p) = 0.5*{1/m:.4f}*p^2 + 0.5*{k:.4f}*q^2"
            acc_formula = f"-{k:.4f}*q"

        cand_id = f"HAM-{hashlib.sha256(formula.encode()).hexdigest()[:8]}"

        def compute_score(test_df: pd.DataFrame) -> np.ndarray:
            tq = test_df[q_col].to_numpy(dtype=float) if q_col in test_df.columns else np.zeros(len(test_df))
            return -k * tq - alpha * (tq ** 3)

        proposal = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=cand_id,
            formula=acc_formula,
            strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(q_col, p_col),
            compute_score=compute_score,
            parameters={"mass": m, "spring_k": k, "nonlinear_alpha": alpha, "energy_drift": drift},
            metadata={
                "hamiltonian_formula": formula,
                "domain": "hamiltonian_mechanics",
                "symplectic_invariance": True,
                "relative_energy_drift": drift,
            },
        )
        return [proposal]


class SolenoidalFluidProposer(ProposerBase):
    """Discovers divergence-free (solenoidal) fluid velocity fields.

    Continuity equation for incompressible flow:
        div u = du/dx + dv/dy = 0

    Constructed via stream function psi(x, y):
        u(x, y) = d(psi)/dy
        v(x, y) = -d(psi)/dx
        => div u = d^2(psi)/(dx dy) - d^2(psi)/(dy dx) == 0 analytically.
    """

    def __init__(self, polynomial_order: int = 2) -> None:
        self.order = int(polynomial_order)

    @property
    def proposer_id(self) -> str:
        return "solenoidal_incompressible_fluid_proposer_v1"

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame] | pd.DataFrame,
        target_col: str = "u",
        x_col: str = "x",
        y_col: str = "y",
        u_col: str = "u",
        v_col: str = "v",
    ) -> Sequence[CandidateProposal]:
        """Fit stream function psi(x, y) and propose solenoidal velocity field."""
        df = train_data if isinstance(train_data, pd.DataFrame) else next(iter(train_data.values()))
        for c in (x_col, y_col, u_col, v_col):
            if c not in df.columns:
                return []
        if len(df) < 15:
            return []

        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        u_obs = df[u_col].to_numpy(dtype=float)
        v_obs = df[v_col].to_numpy(dtype=float)

        # Basis for psi(x, y):
        # psi_1 = x * y       => u = x,   v = -y    (hyperbolic / stagnation flow)
        # psi_2 = y^2 / 2     => u = y,   v = 0     (Couette shear flow)
        # psi_3 = x^2 / 2     => u = 0,   v = -x    (vortical flow)
        # psi_4 = x * y^2     => u = 2xy, v = -y^2
        # psi_5 = x^2 * y     => u = x^2, v = -2xy
        # Joint least squares on [u, v]^T = B * c
        u_basis = np.column_stack([
            x,              # d(xy)/dy
            y,              # d(y^2/2)/dy
            np.zeros_like(x), # d(x^2/2)/dy
            2 * x * y,      # d(xy^2)/dy
            x ** 2,         # d(x^2 y)/dy
        ])

        v_basis = np.column_stack([
            -y,             # -d(xy)/dx
            np.zeros_like(x), # -d(y^2/2)/dx
            -x,             # -d(x^2/2)/dx
            -y ** 2,        # -d(xy^2)/dx
            -2 * x * y,     # -d(x^2 y)/dx
        ])

        # Stack joint regression
        A = np.vstack([u_basis, v_basis])
        b_target = np.concatenate([u_obs, v_obs])

        coeffs, _residuals, _, _ = np.linalg.lstsq(A, b_target, rcond=None)
        c1, c2, c3, c4, c5 = [float(val) for val in coeffs]

        # Analytical divergence check:
        # div u = du/dx + dv/dy:
        # For each basis function, d(u_i)/dx + d(v_i)/dy:
        # (1) d(x)/dx + d(-y)/dy = 1 - 1 = 0
        # (2) d(y)/dx + d(0)/dy = 0 + 0 = 0
        # (3) d(0)/dx + d(-x)/dy = 0 + 0 = 0
        # (4) d(2xy)/dx + d(-y^2)/dy = 2y - 2y = 0
        # (5) d(x^2)/dx + d(-2xy)/dy = 2x - 2x = 0
        # Exactly 0 by mathematical construction!

        formula_u = f"{c1:.4f}*x + {c2:.4f}*y + {c4:.4f}*2*x*y + {c5:.4f}*x^2"
        formula_v = f"-{c1:.4f}*y - {c3:.4f}*x - {c4:.4f}*y^2 - {c5:.4f}*2*x*y"
        stream_fn = f"psi(x, y) = {c1:.4f}*x*y + 0.5*{c2:.4f}*y^2 + 0.5*{c3:.4f}*x^2 + {c4:.4f}*x*y^2 + {c5:.4f}*x^2*y"

        cand_id = f"FLUID-SOL-{hashlib.sha256(stream_fn.encode()).hexdigest()[:8]}"

        def compute_score(test_df: pd.DataFrame) -> np.ndarray:
            tx = test_df[x_col].to_numpy(dtype=float) if x_col in test_df.columns else np.zeros(len(test_df))
            ty = test_df[y_col].to_numpy(dtype=float) if y_col in test_df.columns else np.zeros(len(test_df))
            return c1 * tx + c2 * ty + 2 * c4 * tx * ty + c5 * (tx ** 2)

        proposal = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=cand_id,
            formula=formula_u,
            strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
            declared_scale_invariant=True,
            declared_nominal_dependencies=(x_col, y_col),
            compute_score=compute_score,
            parameters={"c_hyperbolic": c1, "c_couette": c2, "c_vortex": c3, "c_nl1": c4, "c_nl2": c5},
            metadata={
                "stream_function": stream_fn,
                "velocity_field_u": formula_u,
                "velocity_field_v": formula_v,
                "divergence_free": True,
                "domain": "incompressible_fluid_mechanics",
            },
        )
        return [proposal]


class CompartmentalSirProposer(ProposerBase):
    """Discovers epidemiological compartmental dynamics with exact mass conservation.

    Kermack-McKendrick SIR equations:
        dS/dt = -beta * S * I / N
        dI/dt = beta * S * I / N - gamma * I
        dR/dt = gamma * I

    Exact Invariant:
        d/dt (S + I + R) = 0 => S(t) + I(t) + R(t) = N (constant population).
    """

    def __init__(self, dt: float = 1.0, max_pop_drift: float = 0.02) -> None:
        self.dt = float(dt)
        self.max_pop_drift = float(max_pop_drift)

    @property
    def proposer_id(self) -> str:
        return "compartmental_sir_epidemiological_proposer_v1"

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame] | pd.DataFrame,
        target_col: str = "dI",
        s_col: str = "S",
        i_col: str = "I",
        r_col: str = "R",
    ) -> Sequence[CandidateProposal]:
        """Fit transmission and recovery rates with mass conservation guarantee."""
        df = train_data if isinstance(train_data, pd.DataFrame) else next(iter(train_data.values()))
        for c in (s_col, i_col, r_col):
            if c not in df.columns:
                return []
        if len(df) < 15:
            return []

        S = df[s_col].to_numpy(dtype=float)
        I_arr = df[i_col].to_numpy(dtype=float)
        R = df[r_col].to_numpy(dtype=float)

        total_N = S + I_arr + R
        N0 = float(np.mean(total_N))
        if N0 <= 0.0:
            return []

        # Population conservation check
        pop_drift = float(np.ptp(total_N) / N0)
        if pop_drift > self.max_pop_drift:
            return []

        # Numerical derivatives
        dS = np.gradient(S, self.dt)
        np.gradient(I_arr, self.dt)
        dR = np.gradient(R, self.dt)

        # Estimate gamma from dR/dt = gamma * I_arr
        denom_I = float(np.sum(I_arr ** 2))
        if denom_I < 1e-12:
            return []
        gamma = float(np.sum(dR * I_arr) / denom_I)
        gamma = max(0.001, gamma)

        # Estimate beta from dS/dt = -beta * (S * I_arr / N)
        interaction = S * I_arr / N0
        denom_SI = float(np.sum(interaction ** 2))
        if denom_SI < 1e-12:
            return []
        beta = float(-np.sum(dS * interaction) / denom_SI)
        beta = max(0.001, beta)

        r0 = beta / gamma
        formula = f"dI/dt = {beta:.4f} * (S * I_arr / {N0:.1f}) - {gamma:.4f} * I_arr"
        cand_id = f"SIR-{hashlib.sha256(formula.encode()).hexdigest()[:8]}"

        def compute_score(test_df: pd.DataFrame) -> np.ndarray:
            tS = test_df[s_col].to_numpy(dtype=float) if s_col in test_df.columns else np.zeros(len(test_df))
            tI = test_df[i_col].to_numpy(dtype=float) if i_col in test_df.columns else np.zeros(len(test_df))
            return beta * (tS * tI / N0) - gamma * tI

        proposal = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=cand_id,
            formula=formula,
            strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(s_col, i_col, r_col),
            compute_score=compute_score,
            parameters={"beta": beta, "gamma": gamma, "R0": r0, "population_N": N0},
            metadata={
                "infection_rate_beta": beta,
                "recovery_rate_gamma": gamma,
                "basic_reproduction_number_R0": r0,
                "conserved_total_population": N0,
                "population_drift": pop_drift,
                "domain": "mathematical_epidemiology",
            },
        )
        return [proposal]
