r"""Higher-Order Spatio-Temporal PDE Hypothesis Engine & JetFlow Bridge.

Status: experimental

Unites the JetFlow 2nd-order continuous flow / 4D Tensor-Train architecture with
symbolic discovery and epistemic referee adjudication of higher-order PDEs:
1. Spatial derivatives up to 4th order:
   - 1st order: u_x, u_y (advection / transport)
   - 2nd order: u_xx, u_yy, laplacian(u) (diffusion / wave tension)
   - 3rd order: u_xxx, u_yyy (Korteweg-de Vries dispersion)
   - 4th order: u_xxxx, biharmonic_lap2(u) (Kuramoto-Sivashinsky / Cahn-Hilliard / thin-plate)
2. Temporal orders k_t in {1, 2}:
   - 1st-order evolution: u_t = F(u, u_x, u_xx, u_xxx, u_xxxx, ...)
   - 2nd-order Jet2 hyperbolic evolution: u_tt = F(u, u_t, u_xx, u_xxxx, ...)
3. Strict AGENTS.md Section 1.1 Proposer vs. Referee separation:
   - HigherOrderPdeProposer fits exclusively on development data and emits CandidateProposal objects.
   - HigherOrderPdeReferee evaluates frozen coefficients on sealed holdout slices and enforces
     higher-order ultraviolet well-posedness (e.g. biharmonic damping sign for 4th-order PDEs).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.high_dim_pde import EpistemicPdeVerdict
from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)

HIGHER_ORDER_PROPOSER_ID = "jetflow_higher_order_pde_proposer"


@dataclass(frozen=True)
class HigherOrderPdeDiscoveryResult:
    """Discovered higher-order PDE hypothesis on development data."""
    temporal_order: int  # 1 for u_t, 2 for u_tt
    target_lhs: str      # "u_t" or "u_tt"
    formula: str
    active_terms: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    r2_dev: float
    residual_rmse: float
    bic: float
    spatial_dimension: int
    max_spatial_order: int


@dataclass(frozen=True)
class HigherOrderPdeAdjudication:
    """Epistemic referee verdict on a sealed holdout trajectory."""
    verdict: EpistemicPdeVerdict
    discovered: HigherOrderPdeDiscoveryResult
    frozen_oos_r2: float
    frozen_oos_rmse: float
    residual_whiteness_rho1: float
    ultraviolet_well_posed: bool
    rational_justification: str


def _spectral_derivative_1d(u: np.ndarray, dx: float, order: int, axis: int = -1) -> np.ndarray:
    """Compute exact periodic Fourier spectral derivative of arbitrary order along `axis`."""
    n = u.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ik_pow = (1j * k) ** order
    # Zero out Nyquist mode for odd derivative orders to preserve real symmetry
    if order % 2 == 1 and n % 2 == 0:
        ik_pow[n // 2] = 0.0
    shape = [1] * u.ndim
    shape[axis] = n
    u_hat = np.fft.fft(u, axis=axis)
    return np.fft.ifft(u_hat * ik_pow.reshape(shape), axis=axis).real


def _fd_derivative_1d_interior(u: np.ndarray, dx: float, order: int) -> np.ndarray:
    """Compute 4th-order-accurate finite-difference derivative on interior grid [3:-3]."""
    if order == 1:
        # 4th-order central: (u_{i-2} - 8 u_{i-1} + 8 u_{i+1} - u_{i+2}) / (12 dx)
        return (
            u[:, 1:-5] - 8.0 * u[:, 2:-4] + 8.0 * u[:, 4:-2] - u[:, 5:-1]
        ) / (12.0 * dx)
    if order == 2:
        # 4th-order central: (-u_{i-2} + 16 u_{i-1} - 30 u_i + 16 u_{i+1} - u_{i+2}) / (12 dx^2)
        return (
            -u[:, 1:-5]
            + 16.0 * u[:, 2:-4]
            - 30.0 * u[:, 3:-3]
            + 16.0 * u[:, 4:-2]
            - u[:, 5:-1]
        ) / (12.0 * (dx ** 2))
    if order == 3:
        # 4th-order 7-point central for u_xxx:
        # (u_{i-3} - 8 u_{i-2} + 13 u_{i-1} - 13 u_{i+1} + 8 u_{i+2} - u_{i+3}) / (8 dx^3)
        return (
            u[:, 0:-6]
            - 8.0 * u[:, 1:-5]
            + 13.0 * u[:, 2:-4]
            - 13.0 * u[:, 4:-2]
            + 8.0 * u[:, 5:-1]
            - u[:, 6:]
        ) / (8.0 * (dx ** 3))
    if order == 4:
        # 4th-order 7-point central for u_xxxx:
        # (-u_{i-3} + 12 u_{i-2} - 39 u_{i-1} + 56 u_i - 39 u_{i+1} + 12 u_{i+2} - u_{i+3}) / (6 dx^4)
        return (
            -u[:, 0:-6]
            + 12.0 * u[:, 1:-5]
            - 39.0 * u[:, 2:-4]
            + 56.0 * u[:, 3:-3]
            - 39.0 * u[:, 4:-2]
            + 12.0 * u[:, 5:-1]
            - u[:, 6:]
        ) / (6.0 * (dx ** 4))
    raise ValueError(f"Unsupported spatial derivative order {order}")


def build_higher_order_dictionary_1d(
    u: np.ndarray,
    dt: float,
    dx: float,
    periodic: bool = True,
    max_spatial_order: int = 4,
    include_nonlinear: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Construct u_t, u_tt, and higher-order spatial candidate dictionary for 1D field u(T, X)."""
    if u.ndim != 2:
        raise ValueError(f"Expected 2D array (T, X), got shape {u.shape}")
    n_t, n_x = u.shape
    if n_t < 5 or n_x < 8:
        raise ValueError(f"Grid (T={n_t}, X={n_x}) too small for 4th-order temporal/spatial stencils")

    # 4th-order central temporal derivatives on interior time steps [2:-2]
    u_t_full = (u[0:-4] - 8.0 * u[1:-3] + 8.0 * u[3:-1] - u[4:]) / (12.0 * dt)
    u_tt_full = (
        -u[0:-4] + 16.0 * u[1:-3] - 30.0 * u[2:-2] + 16.0 * u[3:-1] - u[4:]
    ) / (12.0 * (dt ** 2))
    u_mid = u[2:-2]

    terms: Dict[str, np.ndarray] = {}
    if periodic:
        u_core = u_mid
        u_t = u_t_full.reshape(-1)
        u_tt = u_tt_full.reshape(-1)
        terms["u"] = u_core.reshape(-1)
        terms["u_t"] = u_t
        for k in range(1, max_spatial_order + 1):
            name = "u_" + ("x" * k)
            terms[name] = _spectral_derivative_1d(u_core, dx, order=k, axis=-1).reshape(-1)
    else:
        u_core = u_mid[:, 3:-3]
        u_t = u_t_full[:, 3:-3].reshape(-1)
        u_tt = u_tt_full[:, 3:-3].reshape(-1)
        terms["u"] = u_core.reshape(-1)
        terms["u_t"] = u_t
        for k in range(1, max_spatial_order + 1):
            name = "u_" + ("x" * k)
            terms[name] = _fd_derivative_1d_interior(u_mid, dx, order=k).reshape(-1)

    if include_nonlinear:
        u_flat = terms["u"]
        terms["u^2"] = u_flat ** 2
        terms["u^3"] = u_flat ** 3
        if "u_x" in terms:
            terms["u*u_x"] = u_flat * terms["u_x"]
            terms["(u_x)^2"] = terms["u_x"] ** 2
        if "u_xx" in terms:
            terms["u*u_xx"] = u_flat * terms["u_xx"]

    return u_t, u_tt, terms


def build_higher_order_dictionary_2d(
    u: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    max_spatial_order: int = 4,
    include_nonlinear: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Construct u_t, u_tt, and 2D isotropic/anisotropic higher-order PDE dictionary for u(T, H, W)."""
    if u.ndim != 3:
        raise ValueError(f"Expected 3D array (T, H, W), got shape {u.shape}")
    n_t, h, w = u.shape
    if n_t < 5 or h < 8 or w < 8:
        raise ValueError(f"Grid (T={n_t}, H={h}, W={w}) too small for higher-order 2D stencils")

    u_t_full = (u[0:-4] - 8.0 * u[1:-3] + 8.0 * u[3:-1] - u[4:]) / (12.0 * dt)
    u_tt_full = (
        -u[0:-4] + 16.0 * u[1:-3] - 30.0 * u[2:-2] + 16.0 * u[3:-1] - u[4:]
    ) / (12.0 * (dt ** 2))
    u_mid = u[2:-2]

    u_t = u_t_full.reshape(-1)
    u_tt = u_tt_full.reshape(-1)
    u_flat = u_mid.reshape(-1)

    ux = _spectral_derivative_1d(u_mid, dx, order=1, axis=-1).reshape(-1)
    uy = _spectral_derivative_1d(u_mid, dy, order=1, axis=-2).reshape(-1)
    uxx = _spectral_derivative_1d(u_mid, dx, order=2, axis=-1)
    uyy = _spectral_derivative_1d(u_mid, dy, order=2, axis=-2)
    lap = (uxx + uyy).reshape(-1)

    terms: Dict[str, np.ndarray] = {
        "u": u_flat,
        "u_t": u_t,
        "u_x": ux,
        "u_y": uy,
        "laplacian(u)": lap,
    }

    if max_spatial_order >= 3:
        uxxx = _spectral_derivative_1d(u_mid, dx, order=3, axis=-1).reshape(-1)
        uyyy = _spectral_derivative_1d(u_mid, dy, order=3, axis=-2).reshape(-1)
        terms["u_xxx"] = uxxx
        terms["u_yyy"] = uyyy

    if max_spatial_order >= 4:
        # Biharmonic operator \nabla^4 u = \nabla^2 (\nabla^2 u)
        lap_2d = uxx + uyy
        lap_xx = _spectral_derivative_1d(lap_2d, dx, order=2, axis=-1)
        lap_yy = _spectral_derivative_1d(lap_2d, dy, order=2, axis=-2)
        terms["biharmonic_lap2(u)"] = (lap_xx + lap_yy).reshape(-1)

    if include_nonlinear:
        terms["u^3"] = u_flat ** 3
        terms["u*u_x"] = u_flat * ux
        terms["u*u_y"] = u_flat * uy
        terms["|grad(u)|^2"] = ux ** 2 + uy ** 2

    return u_t, u_tt, terms


def _stridge_regression(
    theta: np.ndarray,
    target: np.ndarray,
    term_names: Sequence[str],
    sparsity_threshold: float = 0.02,
    ridge_alpha: float = 1e-6,
    max_iter: int = 12,
) -> Tuple[Tuple[str, ...], Tuple[float, ...], np.ndarray]:
    """Column-normalized Sequentially Thresholded Ridge Regression (STRidge) with OLS refit."""
    n_samples, n_terms = theta.shape
    col_norms = np.linalg.norm(theta, axis=0) / np.sqrt(max(1, n_samples))
    valid_cols = col_norms > 1e-12
    if not np.any(valid_cols):
        return (), (), np.zeros_like(target)

    y_std = float(np.std(target))
    if y_std < 1e-12:
        return (), (), np.zeros_like(target)

    theta_norm = np.zeros_like(theta)
    theta_norm[:, valid_cols] = theta[:, valid_cols] / col_norms[valid_cols]
    y_norm = target / y_std

    active = valid_cols.copy()
    w_norm = np.zeros(n_terms, dtype=np.float64)

    for _ in range(max_iter):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        sub_x = theta_norm[:, idx]
        gram = sub_x.T @ sub_x + ridge_alpha * np.eye(len(idx))
        sub_w = np.linalg.solve(gram, sub_x.T @ y_norm)
        w_norm[:] = 0.0
        w_norm[idx] = sub_w
        # Threshold in normalized effect-size units
        keep = np.abs(w_norm) >= sparsity_threshold
        if np.array_equal(keep, active):
            break
        active = keep

    idx_final = np.where(active)[0]
    if len(idx_final) == 0:
        return (), (), np.zeros_like(target)

    # Unbiased OLS refit on original physical scales
    sub_orig = theta[:, idx_final]
    gram_orig = sub_orig.T @ sub_orig + (1e-10 * np.eye(len(idx_final)))
    coeffs_orig = np.linalg.solve(gram_orig, sub_orig.T @ target)
    pred = sub_orig @ coeffs_orig

    selected_names = tuple(term_names[i] for i in idx_final)
    selected_coeffs = tuple(float(c) for c in coeffs_orig)
    return selected_names, selected_coeffs, pred


def discover_higher_order_pde(
    u: np.ndarray,
    dt: float = 0.01,
    dx: float = 0.1,
    dy: Optional[float] = None,
    temporal_order: Optional[int] = None,
    max_spatial_order: int = 4,
    periodic: bool = True,
    include_nonlinear: bool = True,
    sparsity_threshold: float = 0.03,
    ridge_alpha: float = 1e-6,
) -> HigherOrderPdeDiscoveryResult:
    """Discover 1st-order (u_t) or 2nd-order (u_tt) higher-order PDE (up to 4th spatial order)."""
    if u.ndim == 2:
        spatial_dim = 1
        u_t, u_tt, terms = build_higher_order_dictionary_1d(
            u,
            dt=dt,
            dx=dx,
            periodic=periodic,
            max_spatial_order=max_spatial_order,
            include_nonlinear=include_nonlinear,
        )
    elif u.ndim == 3:
        spatial_dim = 2
        u_t, u_tt, terms = build_higher_order_dictionary_2d(
            u,
            dt=dt,
            dx=dx,
            dy=dy if dy is not None else dx,
            max_spatial_order=max_spatial_order,
            include_nonlinear=include_nonlinear,
        )
    else:
        raise ValueError(f"Expected 2D (T, X) or 3D (T, H, W) field, got {u.shape}")

    orders_to_test = [temporal_order] if temporal_order in (1, 2) else [1, 2]
    candidates: List[HigherOrderPdeDiscoveryResult] = []

    for t_ord in orders_to_test:
        lhs_name = "u_t" if t_ord == 1 else "u_tt"
        target = u_t if t_ord == 1 else u_tt
        # Exclude u_t from RHS when fitting 1st-order u_t = F(...)
        cand_names = [k for k in terms.keys() if not (t_ord == 1 and k == "u_t")]
        theta = np.column_stack([terms[k] for k in cand_names])

        active_terms, coeffs, pred = _stridge_regression(
            theta,
            target,
            cand_names,
            sparsity_threshold=sparsity_threshold,
            ridge_alpha=ridge_alpha,
        )
        resid = target - pred
        n_obs = len(target)
        mse = float(np.mean(resid ** 2))
        rmse = float(np.sqrt(mse))
        var_y = float(np.var(target))
        r2 = 1.0 - (mse / max(1e-12, var_y))
        k_params = len(active_terms)
        # Normalized BIC (in variance-normalized units so u_t and u_tt are directly comparable)
        rel_mse = max(1e-12, mse / max(1e-12, var_y))
        bic = float(n_obs * np.log(rel_mse) + k_params * np.log(max(2, n_obs)))

        max_ord_found = 0
        for name in active_terms:
            if "xxxx" in name or "biharmonic" in name:
                max_ord_found = max(max_ord_found, 4)
            elif "xxx" in name or "yyy" in name:
                max_ord_found = max(max_ord_found, 3)
            elif "xx" in name or "yy" in name or "laplacian" in name:
                max_ord_found = max(max_ord_found, 2)
            elif "_x" in name or "_y" in name or "grad" in name:
                max_ord_found = max(max_ord_found, 1)

        if active_terms:
            rhs_str = " + ".join(f"{c:+.4f}*{name}" for name, c in zip(active_terms, coeffs))
        else:
            rhs_str = "0.0"
        formula = f"{lhs_name} = {rhs_str}"

        candidates.append(
            HigherOrderPdeDiscoveryResult(
                temporal_order=t_ord,
                target_lhs=lhs_name,
                formula=formula,
                active_terms=active_terms,
                coefficients=coeffs,
                r2_dev=float(r2),
                residual_rmse=float(rmse),
                bic=bic,
                spatial_dimension=spatial_dim,
                max_spatial_order=max_ord_found,
            )
        )

    candidates.sort(key=lambda c: c.bic)
    return candidates[0]


class HigherOrderPdeProposer(ProposerBase):
    """Hypothesis proposer for higher-order PDEs (KdV, Kuramoto-Sivashinsky, Cahn-Hilliard, Wave) + JetFlow TT."""

    def __init__(
        self,
        dt: float = 0.01,
        dx: float = 0.1,
        temporal_order: Optional[int] = None,
        max_spatial_order: int = 4,
        periodic: bool = True,
        include_nonlinear: bool = True,
        sparsity_threshold: float = 0.03,
        ridge_alpha: float = 1e-6,
        strategy_archetype: StrategyArchetype = StrategyArchetype.CONTINUOUS_FACTOR,
    ) -> None:
        self._dt = dt
        self._dx = dx
        self._temporal_order = temporal_order
        self._max_spatial_order = max_spatial_order
        self._periodic = periodic
        self._include_nonlinear = include_nonlinear
        self._sparsity_threshold = sparsity_threshold
        self._ridge_alpha = ridge_alpha
        self._strategy_archetype = strategy_archetype

    @property
    def proposer_id(self) -> str:
        return HIGHER_ORDER_PROPOSER_ID

    def propose_from_field(
        self,
        dev_field: np.ndarray,
        compress_tt: bool = True,
    ) -> Tuple[HigherOrderPdeDiscoveryResult, Sequence[CandidateProposal], Optional[Any]]:
        """Discover higher-order PDE on development field (T, X) or (T, H, W) or video (T, C, H, W).

        Never inspects holdout data or decides its own acceptance (AGENTS.md Rule 1.1).
        """
        tt_summary: Optional[Dict[str, Any]] = None
        if dev_field.ndim == 4 and compress_tt:
            # 4D video/field tensor (T, C, H, W): compress via JetFlow 4D Tensor-Train & analyze channel 0
            import torch
            from jetflow_video.tensor_train import compress_video_latent_torch
            tt_obj = compress_video_latent_torch(torch.from_numpy(dev_field.astype(np.float32)), max_rank=8)
            tt_summary = {
                "bond_ranks": tt_obj.bond_ranks,
                "compression_ratio": float(tt_obj.compression_ratio),
                "memory_saving_pct": float(tt_obj.memory_saving_pct),
            }
            u_work = np.asarray(dev_field[:, 0, :, :], dtype=np.float64)
        elif dev_field.ndim == 4:
            u_work = np.asarray(dev_field[:, 0, :, :], dtype=np.float64)
        else:
            u_work = np.asarray(dev_field, dtype=np.float64)

        res = discover_higher_order_pde(
            u_work,
            dt=self._dt,
            dx=self._dx,
            temporal_order=self._temporal_order,
            max_spatial_order=self._max_spatial_order,
            periodic=self._periodic,
            include_nonlinear=self._include_nonlinear,
            sparsity_threshold=self._sparsity_threshold,
            ridge_alpha=self._ridge_alpha,
        )
        if not res.active_terms:
            return res, (), tt_summary

        cid_hash = hashlib.sha256(f"{res.formula}_{res.coefficients}".encode()).hexdigest()[:12]
        candidate_id = f"ho_pde_{cid_hash}"
        param_dict = {term: float(coeff) for term, coeff in zip(res.active_terms, res.coefficients)}

        def score_fn(eval_df: pd.DataFrame) -> np.ndarray:
            if "u" not in eval_df.columns:
                return np.zeros(len(eval_df))
            return np.nan_to_num(eval_df["u"].to_numpy(dtype=np.float64), nan=0.0)

        meta: Dict[str, Any] = {
            "temporal_order": res.temporal_order,
            "target_lhs": res.target_lhs,
            "max_spatial_order": res.max_spatial_order,
            "r2_dev": res.r2_dev,
            "residual_rmse": res.residual_rmse,
            "bic": res.bic,
            "active_terms": list(res.active_terms),
        }
        if tt_summary is not None:
            meta["jetflow_tt_summary"] = tt_summary

        proposal = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=candidate_id,
            formula=res.formula,
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=True,
            declared_nominal_dependencies=("x",) if res.spatial_dimension == 1 else ("x", "y"),
            compute_score=score_fn,
            parameters=param_dict,
            metadata=meta,
        )
        return res, (proposal,), tt_summary

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame] | pd.DataFrame,
        target_col: str = "u",
    ) -> Sequence[CandidateProposal]:
        """Standard ProposerBase DataFrame entrypoint."""
        df = next(iter(train_data.values())) if isinstance(train_data, dict) else train_data
        if target_col not in df.columns:
            return ()
        u_vals = df[target_col].to_numpy(dtype=np.float64)
        n_x = len(df["x"].unique()) if "x" in df.columns else 16
        n_t = len(u_vals) // max(1, n_x)
        if n_t < 5 or n_x < 8:
            return ()
        u_grid = u_vals[: n_t * n_x].reshape(n_t, n_x)
        _, proposals, _ = self.propose_from_field(u_grid, compress_tt=False)
        return proposals


class HigherOrderPdeReferee:
    """Epistemic Referee adjudicating higher-order PDE hypotheses on sealed holdout data."""

    def __init__(
        self,
        min_holdout_r2: float = 0.95,
        max_whiteness_rho1: float = 0.50,
        enforce_ultraviolet_stability: bool = True,
    ) -> None:
        self._min_holdout_r2 = min_holdout_r2
        self._max_whiteness_rho1 = max_whiteness_rho1
        self._enforce_ultraviolet_stability = enforce_ultraviolet_stability

    @staticmethod
    def check_ultraviolet_well_posedness(result: HigherOrderPdeDiscoveryResult) -> Tuple[bool, str]:
        """Verify high-wavenumber (ultraviolet) Fourier symbol damping for parabolic/biharmonic PDEs.

        For 1st-order-in-time PDEs u_t = sum c_k d^k u / dx^k:
        - A Fourier mode exp(i k x) evolves with real growth rate Re(lambda(k)) = -c_2 k^2 + c_4 k^4 - ...
        - If the highest even spatial derivative is 4th order (u_xxxx or biharmonic_lap2(u)), its
          coefficient c_4 MUST be negative (c_4 < 0, e.g. Kuramoto-Sivashinsky u_t = -u_xx - nu u_xxxx
          with nu > 0) so that Re(lambda(k)) = c_4 k^4 -> -infty as |k| -> infty.
        - If the highest even spatial derivative is 2nd order (u_xx or laplacian(u)) with no 4th-order
          stabilizer, its coefficient c_2 MUST be positive (c_2 > 0, forward heat equation).
        """
        if result.temporal_order != 1:
            return True, "Hyperbolic 2nd-order temporal evolution (wave/dispersive)."

        param_map = dict(zip(result.active_terms, result.coefficients))
        c4 = param_map.get("u_xxxx", param_map.get("biharmonic_lap2(u)", None))
        c2 = param_map.get("u_xx", param_map.get("laplacian(u)", None))

        if c4 is not None:
            if c4 > 0.0:
                return (
                    False,
                    f"Ultraviolet blowup: positive 4th-order biharmonic coefficient ({c4:+.4f}) "
                    "amplifies high-wavenumber modes as exp(+c4 * k^4 * t).",
                )
            return True, f"Stabilized 4th-order biharmonic damping ({c4:+.4f} * k^4 < 0)."

        if c2 is not None and c2 < 0.0:
            return (
                False,
                f"Thermodynamic arrow violated: negative 2nd-order diffusion ({c2:+.4f}) "
                "without 4th-order biharmonic stabilization.",
            )

        return True, "Well-posed dispersive/dissipative symbol."

    def adjudicate(
        self,
        result: HigherOrderPdeDiscoveryResult,
        holdout_u: np.ndarray,
        dt: float = 0.01,
        dx: float = 0.1,
        dy: Optional[float] = None,
        periodic: bool = True,
    ) -> HigherOrderPdeAdjudication:
        """Evaluate frozen discovered coefficients on a sealed holdout trajectory."""
        if holdout_u.ndim == 2:
            u_t, u_tt, terms = build_higher_order_dictionary_1d(
                holdout_u,
                dt=dt,
                dx=dx,
                periodic=periodic,
                max_spatial_order=max(4, result.max_spatial_order),
                include_nonlinear=True,
            )
        elif holdout_u.ndim == 3:
            u_t, u_tt, terms = build_higher_order_dictionary_2d(
                holdout_u,
                dt=dt,
                dx=dx,
                dy=dy if dy is not None else dx,
                max_spatial_order=max(4, result.max_spatial_order),
                include_nonlinear=True,
            )
        else:
            raise ValueError(f"Expected 2D or 3D holdout array, got {holdout_u.shape}")

        target = u_t if result.temporal_order == 1 else u_tt
        pred = np.zeros_like(target)
        for name, coeff in zip(result.active_terms, result.coefficients):
            if name in terms:
                pred = pred + coeff * terms[name]

        resid = target - pred
        mse = float(np.mean(resid ** 2))
        rmse = float(np.sqrt(mse))
        var_y = float(np.var(target))
        oos_r2 = 1.0 - (mse / max(1e-12, var_y))

        # Lag-1 autocorrelation of residuals
        if len(resid) > 2 and np.std(resid) > 1e-12:
            r0 = resid[:-1] - np.mean(resid[:-1])
            r1 = resid[1:] - np.mean(resid[1:])
            denom = float(np.linalg.norm(r0) * np.linalg.norm(r1))
            rho1 = float(np.dot(r0, r1) / denom) if denom > 1e-12 else 0.0
        else:
            rho1 = 0.0

        well_posed, uv_reason = self.check_ultraviolet_well_posedness(result)
        if self._enforce_ultraviolet_stability and not well_posed:
            return HigherOrderPdeAdjudication(
                verdict=EpistemicPdeVerdict.PREDICTIVE_WRONG_STRUCTURE,
                discovered=result,
                frozen_oos_r2=float(oos_r2),
                frozen_oos_rmse=rmse,
                residual_whiteness_rho1=rho1,
                ultraviolet_well_posed=False,
                rational_justification=uv_reason,
            )

        if oos_r2 >= self._min_holdout_r2:
            verdict = EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
            reason = f"Confirmed on sealed holdout: R^2={oos_r2:.6f} >= {self._min_holdout_r2}. {uv_reason}"
        elif oos_r2 >= 0.40:
            verdict = EpistemicPdeVerdict.PREDICTIVE_WRONG_STRUCTURE
            reason = f"Holdout R^2={oos_r2:.6f} below threshold {self._min_holdout_r2}."
        else:
            verdict = EpistemicPdeVerdict.REJECTED_NO_CANDIDATE
            reason = f"Holdout collapse: R^2={oos_r2:.6f}."

        return HigherOrderPdeAdjudication(
            verdict=verdict,
            discovered=result,
            frozen_oos_r2=float(oos_r2),
            frozen_oos_rmse=rmse,
            residual_whiteness_rho1=rho1,
            ultraviolet_well_posed=well_posed,
            rational_justification=reason,
        )
