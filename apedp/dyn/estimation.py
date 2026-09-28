"""APEDP-Dyn D4+D5: Parameter estimation with holdout gating.

D4 — Parameter estimation, separated from structural discovery:
structure → parameters are distinct problems (a misspecified structure
cannot be rescued by better parameters, and a correct structure can be
ruined by naively fitted ones).

D5 — Trajectory holdout: fit on one identification experiment, score on
a *different* trajectory (new IC and/or new input) before trusting the
parameters.  The holdout is scored by *simulation* — integrating the
fitted model from the holdout's initial conditions and comparing whole
trajectories — because a parameter set can interpolate derivatives well
yet integrate poorly (small coefficient errors accumulate).

Two fit modes:

* **derivative** (default): Savitzky-Golay differentiates the observed
  channel; one weight per RHS term is fitted against the acceleration
  channel.  Fast and accurate at moderate noise.  All channels pass
  through the same zero-phase filter: differentiating only the noisy
  channel while regressing against raw inputs correlates regressor
  noise with response noise and biases damping estimates severely
  (observed 4x on a mass-spring-damper).
* **weak_form**: for noisier data.  The ODE is integrated analytically
  over sliding windows, Δv = ∫x'' dt = Σ w_j ∫φ_j dt, so the response
  and regressors are *integrals* of the data — integrals smooth noise
  instead of amplifying it, and nothing is differentiated more than
  once (the weak-form/integral identification principle, Messenger &
  Bortz style).  Residual_rms / r_squared are in velocity units, not
  acceleration units, and are not comparable across modes.

Physical parameters (m, c, k for the canonical mechanical triple) are
*interpreted* from fitted weights rather than free-fitted::

    x'' = w_v v + w_x x + w_F F
    ⟺   m x'' + c v + k x = F   with
        m = 1/w_F,  c = -w_v/w_F,  k = -w_x/w_F

Usage::

    est = ParameterEstimator()
    res = est.fit_with_holdout(train, holdout, terms={"v": "v", "x": "x", "F": "F"})
    res.passed            # holdout trajectory within tolerance

    noisy = ParameterEstimator(mode="weak_form")
    fit = noisy.fit(trajectory, terms=TERMS)   # same API, robust mode
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from apedp.discovery_types import DictCompatibleResult
from apedp.dyn.types import TrajectoryData

logger = logging.getLogger("apedp.dyn.estimation")


def _trapz(y: np.ndarray, dt: float) -> float:
    """Trapezoidal integral (avoids the numpy-2-only ``np.trapezoid``)."""
    return float(0.5 * np.sum(y[1:] + y[:-1]) * dt)


# Weak-form prefilter: a generous fixed window (empirically validated on
# the mass-spring-damper at sigma/dt from 0.5 to 10 rad/s^2 — short
# prefilters leave residual noise in the library channels that the
# errors-in-variables mechanism turns into 30%+ attenuation of the
# forcing coefficient; window 61 keeps it under ~6%).
_WEAK_PREFILTER = 61


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EstimationResult(DictCompatibleResult):
    """Outcome of a per-term weighted fit of a second-order channel.

    Attributes:
        terms:         Basis names in design-matrix column order.
        weights:       Fitted coefficient per term (same order as terms).
        parameters:    Physically interpreted parameters (m, c, k) when the
                       fit contains the canonical mechanical triple
                       (``v``, ``x``, ``F``); otherwise empty.
        residual_rms:  Root-mean-square residual of the fit (acceleration
                       units in derivative mode; velocity units in
                       weak_form mode).
        r_squared:     Coefficient of determination against the fit
                       response of the configured mode.
        cond_number:   Condition number of the design matrix; large values
                       signal non-persistent excitation.
        n_samples:     Number of regression samples after filtering and
                       decimation.
        mode:          Fit mode: ``"derivative"`` or ``"weak_form"``.
        note:          Free-text diagnostics.
    """
    terms: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    residual_rms: float = float("inf")
    r_squared: float = 0.0
    cond_number: float = float("inf")
    n_samples: int = 0
    mode: str = "derivative"
    note: str | None = None

    def get(self, name: str, default: float = float("inf")) -> float:
        """Shorthand for a fitted physical parameter by name."""
        return self.parameters.get(name, default)

    def weight(self, name: str) -> float:
        """Fitted weight for a basis term by name (KeyError if absent)."""
        return self.weights[self.terms.index(name)]


@dataclass(frozen=True)
class HoldoutResult(DictCompatibleResult):
    """D5 verdict: fitted parameters scored on a *different* trajectory.

    The holdout trajectory is produced by integrating the fitted model
    from the holdout experiment's initial conditions (and, when present,
    its measured input), then compared against the observed holdout
    channel.

    Attributes:
        fit:           The underlying in-sample EstimationResult.
        mse:           Mean squared error of the simulated holdout x(t).
        rmse:          Root-mean-square holdout error.
        max_abs_error: Maximum absolute trajectory error.
        rel_rmse:      RMSE relative to the RMS of the holdout channel.
        n_points:      Holdout comparison points.
        passed:        True when rmse is within the holdout tolerance.
        note:          Free-text diagnostics.
    """
    fit: EstimationResult = field(default_factory=EstimationResult)
    mse: float = float("inf")
    rmse: float = float("inf")
    max_abs_error: float = float("inf")
    rel_rmse: float = float("inf")
    n_points: int = 0
    passed: bool = False
    note: str | None = None


# ── Estimator ────────────────────────────────────────────────────────────────


class ParameterEstimator:
    """Fits per-term weights for x'' = Σ w_i φ_i from noisy trajectories.

    The estimator is deliberately *structure-agnostic*: the caller (a
    structural discovery stage, or a test) proposes term templates; this
    class handles the numerics — consistent filtering, alignment,
    least-squares, conditioning diagnostics, physical interpretation of
    the canonical mechanical triple, and D5 trajectory-holdout scoring.

    Two modes:
      * ``mode="derivative"`` — SG-derivative regression (default).
      * ``mode="weak_form"``  — integral-form regression for noisier data.
    """

    def __init__(
        self,
        window: int = 41,
        polyorder: int = 3,
        decimate: int = 4,
        mode: str = "derivative",
        n_integral: int = 20,
    ) -> None:
        """Configure filtering / integration.

        Args:
            window:     SG window length in samples (odd, > polyorder).
            polyorder:  SG polynomial order.
            decimate:   Keep every n-th interior sample for the
                        regression (reduces residual autocorrelation;
                        1 disables).
            mode:       ``"derivative"`` or ``"weak_form"``.
            n_integral: Weak-form sliding-window length in samples
                        (>= 5); widened automatically to match the
                        prefilter, windows overlap by half.
        """
        if window % 2 == 0 or window <= polyorder:
            raise ValueError(
                f"window must be odd and greater than polyorder "
                f"(got window={window}, polyorder={polyorder})"
            )
        if mode not in {"derivative", "weak_form"}:
            raise ValueError(f"unknown mode {mode!r}")
        if n_integral < 5:
            raise ValueError(f"n_integral must be >= 5 (got {n_integral})")
        self.window = window
        self.polyorder = polyorder
        self.decimate = decimate
        self.mode = mode
        self.n_integral = n_integral

    # ── Channel construction ──────────────────────────────────────────

    def identification_channels(
        self,
        x: np.ndarray,
        forcing: np.ndarray | None,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build consistently filtered (a, v, x, F) channels.

        All channels pass through the same zero-phase SG kernel so the
        filtered signals still satisfy the true ODE.  ``forcing`` may be
        None for autonomous systems, in which case the F channel is all
        zeros.  Returns arrays evaluated on one shared index grid —
        misaligned regressors silently bias least squares.
        """
        x = np.asarray(x, dtype=float)
        if forcing is None:
            forcing = np.zeros_like(x)
        else:
            forcing = np.asarray(forcing, dtype=float)
        if x.shape != forcing.shape:
            raise ValueError(
                f"x and forcing shapes differ: {x.shape} vs {forcing.shape}"
            )
        if x.size <= self.window:
            raise ValueError(
                f"trajectory too short ({x.size} samples) for "
                f"SG window {self.window}"
            )

        a = savgol_filter(x, self.window, self.polyorder, deriv=2, delta=dt)
        v = savgol_filter(x, self.window, self.polyorder, deriv=1, delta=dt)
        x_f = savgol_filter(x, self.window, self.polyorder, deriv=0)
        F_f = savgol_filter(forcing, self.window, self.polyorder, deriv=0)

        # Interior trim (SG edge stencils are truncated), then decimate.
        half = self.window // 2
        idx = np.arange(half, x.size - half)
        if self.decimate > 1:
            idx = idx[:: self.decimate]
        return a[idx], v[idx], x_f[idx], F_f[idx], idx

    # ── Fitting ───────────────────────────────────────────────────────

    def fit(
        self,
        trajectory: TrajectoryData,
        terms: dict[str, str],
        *,
        state: str = "x",
        input_name: str | None = "F",
        interpret: bool = True,
    ) -> EstimationResult:
        """Fit one weight per named term using the configured mode.

        Args:
            trajectory: Identification trajectory; must contain the state
                        channel (and the input channel when
                        ``input_name`` is given).
            terms:      Basis name -> SymPy expression template over the
                        symbols ``x`` (position), ``v`` (velocity),
                        ``F`` (input), ``t`` (time), e.g.
                        ``{"v": "v", "x": "x", "sin": "sin(x)"}``.
            state:      Name of the observed second-order state channel.
            input_name: Name of the input channel, or None for an
                        autonomous system.
            interpret:  When True, interpret fitted weights as physical
                        parameters for the canonical mechanical triple.

        Returns:
            EstimationResult with weights, interpreted parameters, and
            fit diagnostics.  Never raises on numerical failure: a
            singular design matrix yields ``cond_number = inf`` and an
            explanatory note.
        """
        if self.mode == "weak_form":
            return self._fit_weak_form(
                trajectory, terms, state=state, input_name=input_name,
                interpret=interpret,
            )
        return self._fit_derivative(
            trajectory, terms, state=state, input_name=input_name,
            interpret=interpret,
        )

    def _fit_derivative(
        self,
        trajectory: TrajectoryData,
        terms: dict[str, str],
        *,
        state: str,
        input_name: str | None,
        interpret: bool,
    ) -> EstimationResult:
        """Derivative mode: regress the SG acceleration channel on terms."""
        t = trajectory.t
        dt = float(t[1] - t[0])
        if state not in trajectory.states:
            raise KeyError(
                f"state channel '{state}' not in trajectory "
                f"(have: {sorted(trajectory.states)})"
            )
        if input_name is not None and input_name not in trajectory.inputs:
            raise KeyError(
                f"input channel '{input_name}' not in trajectory "
                f"(have: {sorted(trajectory.inputs)})"
            )

        forcing = (
            trajectory.inputs[input_name] if input_name is not None else None
        )
        a_c, v_c, x_c, F_c, idx = self.identification_channels(
            trajectory.states[state], forcing, dt
        )
        t_c = t[idx]

        # Evaluate every term template on the shared grid.
        x_s, v_s, F_s, t_s = (
            sp.Symbol("x"), sp.Symbol("v"), sp.Symbol("F"), sp.Symbol("t"),
        )
        columns: list[np.ndarray] = []
        for name, template in terms.items():
            fn = sp.lambdify((x_s, v_s, F_s, t_s), template, "numpy")
            vals = np.asarray(
                [
                    fn(xc, vc, fc, tc)
                    for xc, vc, fc, tc in zip(x_c, v_c, F_c, t_c, strict=True)
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(vals)):
                return EstimationResult(
                    terms=list(terms),
                    note=f"term '{name}' produced non-finite values",
                )
            columns.append(vals)
        A = np.column_stack(columns)

        # Conditioning diagnostic *before* the fit: a non-persistently
        # excited experiment cannot identify the parameters, and the fit
        # should say so rather than emit confident garbage.
        cond = float(np.linalg.cond(A))
        if not np.isfinite(cond) or cond > 1e10:
            return EstimationResult(
                terms=list(terms),
                cond_number=cond if np.isfinite(cond) else float("inf"),
                n_samples=A.shape[0],
                note=(
                    f"design matrix ill-conditioned (cond={cond:.3g}): "
                    f"input lacks persistent excitation for these terms"
                ),
            )

        weights, *_ = np.linalg.lstsq(A, a_c, rcond=None)
        residual = A @ weights - a_c
        residual_rms = float(np.sqrt(np.mean(residual**2)))
        ss_tot = float(np.sum((a_c - a_c.mean()) ** 2))
        r_squared = (
            float(1.0 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else 0.0
        )

        parameters: dict[str, float] = {}
        note: str | None = None
        if interpret:
            parameters, note = self._interpret_mechanical(list(terms), weights)
        return EstimationResult(
            terms=list(terms),
            weights=[float(w) for w in weights],
            parameters=parameters,
            residual_rms=residual_rms,
            r_squared=r_squared,
            cond_number=cond,
            n_samples=int(A.shape[0]),
            mode="derivative",
            note=note,
        )

    # ── Weak-form (integral) mode ─────────────────────────────────────

    def _fit_weak_form(
        self,
        trajectory: TrajectoryData,
        terms: dict[str, str],
        *,
        state: str,
        input_name: str | None,
        interpret: bool,
    ) -> EstimationResult:
        """Integral-form regression for noisier data.

        Over sliding windows [t_i, t_{i+w}] the ODE is integrated
        analytically once:

            Δv = v(t_{i+w}) - v(t_i) = Σ w_j ∫ φ_j(x, v, F) dt

        The response is a *difference of first derivatives* (SG window
        15, poly 2 — cheap and noise-tolerant); every regressor is a
        trapezoidal integral of filtered channels, which smooths noise
        instead of amplifying it.  Nothing is ever differentiated
        twice.  Fitted weights are the same w_j as the derivative form,
        so mechanical interpretation applies unchanged; only the
        residual scale differs (velocity, not acceleration units).
        """
        t = trajectory.t
        dt = float(t[1] - t[0])
        if state not in trajectory.states:
            raise KeyError(
                f"state channel '{state}' not in trajectory "
                f"(have: {sorted(trajectory.states)})"
            )
        if input_name is not None and input_name not in trajectory.inputs:
            raise KeyError(
                f"input channel '{input_name}' not in trajectory "
                f"(have: {sorted(trajectory.inputs)})"
            )

        x_raw = np.asarray(trajectory.states[state], dtype=float)
        F_raw = (
            np.asarray(trajectory.inputs[input_name], dtype=float)
            if input_name is not None else None
        )

        # Generous fixed prefilter (see _WEAK_PREFILTER); the sliding
        # integrals do the remaining noise rejection.  Only first
        # derivatives are ever taken.
        w1 = min(_WEAK_PREFILTER, max(15, (x_raw.size // 4) | 1))
        x_f = savgol_filter(x_raw, w1, 2)
        v_f = savgol_filter(x_raw, w1, 2, deriv=1, delta=dt)
        # The input channel is *not* prefiltered: inputs are assumed
        # clean (measured with far higher SNR than the state, or known
        # by construction).  Filtering a clean, spectrally rich input
        # with SG(61, 2) attenuates its high-frequency content — an
        # error-in-variables on the F side that biases 1/w_F (hence m)
        # high by ~7%.  Callers with genuinely noisy inputs should
        # prefilter them with a filter matched to their own noise.
        F_f = F_raw if F_raw is not None else np.zeros_like(x_f)

        # Library arrays on the full filtered grid.
        x_s, v_s, F_s, t_s = (
            sp.Symbol("x"), sp.Symbol("v"), sp.Symbol("F"), sp.Symbol("t"),
        )
        lib: dict[str, np.ndarray] = {}
        for name, template in terms.items():
            fn = sp.lambdify((x_s, v_s, F_s, t_s), template, "numpy")
            vals = np.asarray(
                [
                    fn(xc, vc, fc, tc)
                    for xc, vc, fc, tc in zip(x_f, v_f, F_f, t, strict=True)
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(vals)):
                return EstimationResult(
                    terms=list(terms), mode="weak_form",
                    note=f"term '{name}' produced non-finite values",
                )
            lib[name] = vals

        # Sliding windows on the interior (SG edges excluded), spaced
        # half a window apart to limit overlap correlation.  Velocity
        # differences span w - 1 (>= n_integral - 1) samples: longer
        # differencing windows average the prefiltered derivative's
        # residual bias toward its mean, which is exactly v(t2)-v(t1).
        n = x_raw.size
        margin = w1 // 2
        # Integration windows at least as long as the prefilter: short
        # windows reintroduce the noise the prefilter removed.
        w = max(self.n_integral, w1)
        stride = max(1, w // 2)
        starts = np.arange(margin, n - 1 - w - margin, stride)
        if starts.size < len(terms) + 1:
            return EstimationResult(
                terms=list(terms), mode="weak_form",
                note=(
                    f"trajectory too short for weak-form windows "
                    f"({starts.size} windows, {len(terms)} terms)"
                ),
            )

        rows_y = np.empty(starts.size, dtype=float)
        rows_phi = np.empty((starts.size, len(terms)), dtype=float)
        names = list(terms)
        for row, i in enumerate(starts):
            j = i + w
            rows_y[row] = v_f[j] - v_f[i]
            for col, name in enumerate(names):
                rows_phi[row, col] = _trapz(lib[name][i : j + 1], dt)

        if not np.all(np.isfinite(rows_phi)) or not np.all(
            np.isfinite(rows_y)
        ):
            return EstimationResult(
                terms=names, mode="weak_form",
                note="non-finite values in weak-form regression",
            )

        cond = float(np.linalg.cond(rows_phi))
        if not np.isfinite(cond) or cond > 1e10:
            return EstimationResult(
                terms=names, mode="weak_form",
                cond_number=cond if np.isfinite(cond) else float("inf"),
                n_samples=int(starts.size),
                note=(
                    f"weak-form design ill-conditioned (cond={cond:.3g}): "
                    f"input lacks persistent excitation for these terms"
                ),
            )

        weights, *_ = np.linalg.lstsq(rows_phi, rows_y, rcond=None)
        residual = rows_phi @ weights - rows_y
        residual_rms = float(np.sqrt(np.mean(residual**2)))
        ss_tot = float(np.sum((rows_y - rows_y.mean()) ** 2))
        r_squared = (
            float(1.0 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else 0.0
        )

        parameters: dict[str, float] = {}
        note: str | None = None
        if interpret:
            parameters, note = self._interpret_mechanical(list(names), weights)
        return EstimationResult(
            terms=names,
            weights=[float(wt) for wt in weights],
            parameters=parameters,
            residual_rms=residual_rms,
            r_squared=r_squared,
            cond_number=cond,
            n_samples=int(starts.size),
            mode="weak_form",
            note=note,
        )

    # ── D5: trajectory holdout ────────────────────────────────────────

    def fit_with_holdout(
        self,
        train: TrajectoryData,
        holdout: TrajectoryData,
        terms: dict[str, str],
        *,
        state: str = "x",
        input_name: str | None = "F",
        threshold_rms: float | None = None,
        interpret: bool = True,
    ) -> HoldoutResult:
        """Fit on the training experiment, score on the holdout experiment.

        Args:
            train:   Identification trajectory (fit here).
            holdout: A *different* experiment — new initial conditions
                     and/or new input.  Scored by simulating the fitted
                     model from the holdout's initial conditions.
            terms:        Basis templates (see ``fit``).
            state:        Observed state channel name.
            input_name:   Input channel name, or None for autonomous.
            threshold_rms: Holdout tolerance on RMSE; None selects
                           ``0.10 * RMS(holdout channel)`` — loose enough
                           to absorb realistic initial-velocity
                           uncertainty (which even a perfect model
                           inherits from the holdout data), tight enough
                           to reject wrong dynamics.
            interpret:    Interpret weights mechanically (see ``fit``).

        Returns:
            HoldoutResult with ``passed`` set by the trajectory error.
        """
        fit = self.fit(
            train, terms, state=state, input_name=input_name,
            interpret=interpret,
        )
        return self.score_holdout(
            fit, holdout, terms,
            state=state, input_name=input_name, threshold_rms=threshold_rms,
        )

    def score_holdout(
        self,
        fit: EstimationResult,
        holdout: TrajectoryData,
        terms: dict[str, str],
        *,
        state: str = "x",
        input_name: str | None = "F",
        threshold_rms: float | None = None,
    ) -> HoldoutResult:
        """Score a fitted model on a holdout trajectory via simulation.

        The model is integrated from the holdout's initial conditions —
        this validates the parameters as *dynamics*, not as a curve fit.
        """
        if not fit.weights or not np.all(np.isfinite(fit.weights)):
            return HoldoutResult(
                fit=fit, note=f"fit unusable: {fit.note or 'no weights'}"
            )
        if len(fit.weights) != len(terms):
            raise ValueError(
                f"terms ({len(terms)}) do not match the fitted basis "
                f"({len(fit.weights)}); score a fit made against the same "
                f"terms, or use fit_with_holdout"
            )

        sim_x = self._simulate_check(
            terms, fit.weights, holdout, state=state, input_name=input_name
        )
        if sim_x is None:
            return HoldoutResult(fit=fit, note="holdout simulation failed")

        obs = np.asarray(holdout.states[state], dtype=float)
        n = min(obs.size, sim_x.size)
        err = sim_x[:n] - obs[:n]
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        rms_obs = float(np.sqrt(np.mean(obs[:n] ** 2)))
        # Default tolerance: 10% of the signal RMS, but never below the
        # *observation noise floor* — comparing a simulation against
        # noisy measurements means RMSE cannot drop below ~sigma, and a
        # model at the floor is doing as well as achievable.  sigma is
        # estimated from the holdout itself via a third difference
        # (noise dominates the highest frequencies).  The 1.5x factor
        # covers the estimator's inflation on smooth signals
        # (deterministic curvature leaks into the difference, ~1.75x
        # measured on the mass-spring-damper) without loosening the
        # gate enough to admit wrong structures.
        sigma_hat = float(
            1.4826 * np.median(np.abs(np.diff(obs[:n], n=3))) / np.sqrt(6.0)
        )
        default = max(0.10 * rms_obs, 1.5 * sigma_hat)
        threshold = (
            threshold_rms if threshold_rms is not None else default
        )
        return HoldoutResult(
            fit=fit,
            mse=mse,
            rmse=rmse,
            max_abs_error=float(np.max(np.abs(err))),
            rel_rmse=rmse / rms_obs if rms_obs > 0 else float("inf"),
            n_points=int(n),
            passed=bool(rmse < threshold),
            note=None if rmse < threshold else (
                f"holdout rmse {rmse:.3g} >= tolerance {threshold:.3g}"
            ),
        )

    # ── Simulation helpers ────────────────────────────────────────────

    def _simulate_check(
        self,
        terms: dict[str, str],
        weights: list[float],
        trajectory: TrajectoryData,
        *,
        state: str,
        input_name: str | None,
    ) -> np.ndarray | None:
        """Integrate the fitted x'' = Σ w_j φ_j over the trajectory span.

        Used by holdout scoring to validate parameters as dynamics.
        Returns the simulated position channel on the trajectory's time
        grid, or None on failure.
        """
        t = trajectory.t
        dt = float(t[1] - t[0])
        x_obs = np.asarray(trajectory.states[state], dtype=float)

        x_s, v_s, F_s, t_s = (
            sp.Symbol("x"), sp.Symbol("v"), sp.Symbol("F"), sp.Symbol("t"),
        )
        if input_name is None:
            for template in terms.values():
                if F_s in sp.sympify(template).free_symbols:
                    logger.debug(
                        "term %r references input F but system is autonomous",
                        template,
                    )
                    return None
        rhs = sum(
            wt * sp.sympify(tpl)
            for wt, tpl in zip(weights, terms.values(), strict=True)
        )

        if input_name is not None and input_name in trajectory.inputs:
            F_arr = np.asarray(trajectory.inputs[input_name], dtype=float)
            rhs_fn = sp.lambdify((t_s, x_s, v_s, F_s), rhs, "numpy")

            def dynamics(tt: float, y: np.ndarray) -> list[float]:
                Fv = float(np.interp(tt, t, F_arr))
                return [float(y[1]), float(rhs_fn(tt, y[0], y[1], Fv))]
        else:
            rhs_fn = sp.lambdify((t_s, x_s, v_s), rhs, "numpy")

            def dynamics(tt: float, y: np.ndarray) -> list[float]:
                return [float(y[1]), float(rhs_fn(tt, y[0], y[1]))]

        # Initial velocity: quadratic fit for a first estimate, then a
        # 1-D *shoot* — integrate over a short leading window and pick
        # the v0 minimising the residual there.  This removes the
        # estimator's IC bias from the holdout verdict so the score
        # reflects the fitted *dynamics* (frequency/decay over the full
        # horizon), not IC-estimation accuracy.  Wrong parameters still
        # accumulate error over the full window, where it counts.
        n_ic = min(41, x_obs.size)
        v0_est = float(np.polyfit(t[:n_ic] - t[0], x_obs[:n_ic], 2)[1])
        n_shoot = min(201, x_obs.size)
        t_shoot = t[:n_shoot]
        scale = max(1.0, float(np.max(np.abs(x_obs[:n_shoot]))))

        def shoot_cost(v0: float) -> float:
            sol = solve_ivp(
                dynamics, (float(t_shoot[0]), float(t_shoot[-1])),
                [float(x_obs[0]), v0],
                t_eval=t_shoot, max_step=dt / 2, rtol=1e-8, atol=1e-10,
            )
            if not sol.success or sol.y.shape[1] != t_shoot.size:
                return float("inf")
            return float(np.mean((sol.y[0] - x_obs[:n_shoot]) ** 2))

        from scipy.optimize import minimize_scalar
        try:
            opt = minimize_scalar(
                shoot_cost,
                bounds=(v0_est - 2.0 * scale, v0_est + 2.0 * scale),
                method="bounded",
                options={"xatol": 1e-6},
            )
            v0 = float(opt.x) if np.isfinite(opt.fun) else v0_est
        except (ValueError, FloatingPointError, OverflowError) as exc:
            logger.debug("holdout IC shoot failed: %s", exc)
            v0 = v0_est

        y0 = [float(x_obs[0]), v0]
        try:
            sol = solve_ivp(
                dynamics, (float(t[0]), float(t[-1])), y0,
                t_eval=t, max_step=dt / 2, rtol=1e-8, atol=1e-10,
            )
        except (ValueError, FloatingPointError, OverflowError) as exc:
            logger.debug("holdout simulation failed: %s", exc)
            return None
        if not sol.success or sol.y.shape[1] != t.size:
            return None
        result_array = sol.y[0]
        assert isinstance(result_array, np.ndarray)
        return result_array

    # ── Interpretation ────────────────────────────────────────────────

    @staticmethod
    def _interpret_mechanical(
        terms: list[str], weights: np.ndarray
    ) -> tuple[dict[str, float], str | None]:
        """Convert fitted weights into (m, c, k) for the mechanical triple.

        x'' = w_v v + w_x x + w_F F  ⟺  m x'' + c v + k x = F with
        m = 1/w_F, c = -w_v/w_F, k = -w_x/w_F.  Terms absent from the
        basis are treated as zero contribution (so a no-damping structure
        reports c = inf, correctly signalling non-identifiability of a
        parameter the structure does not contain).

        Returns (parameters, note); note is set when interpretation is
        degraded (vanished forcing coefficient).
        """
        index = {name: i for i, name in enumerate(terms)}
        w_F = weights[index["F"]] if "F" in index else 0.0
        if abs(w_F) < 1e-8:
            return (
                {"m": float("inf"), "c": float("inf"), "k": float("inf")},
                "forcing coefficient vanished; m, c, k not identifiable",
            )
        params = {
            "m": float(1.0 / w_F),
            "c": (
                float(-weights[index["v"]] / w_F)
                if "v" in index else float("inf")
            ),
            "k": (
                float(-weights[index["x"]] / w_F)
                if "x" in index else float("inf")
            ),
        }
        return params, None
