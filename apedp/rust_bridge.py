"""APEDP Rust Numerical & Validation Kernel Bridge.

Provides a unified interface with:
1. Rust Kernel (`apedp_kernel`) as the primary first-line numerical engine.
2. Pure Python reference engine as the transparent fallback.
3. Dual-backend differential cross-validation (`dual_validate=True`).

The kernel is a separate maturin package under `rust/apedp-kernel`, not built
by the setuptools root `pyproject.toml`, so it is absent unless someone ran
`maturin develop`. Item 1 is therefore conditional, not assumed: check
`is_rust_kernel_available()` rather than inferring it. When the module is
missing the Python reference engine answers every call and the results are
still correct -- which is exactly why that is easy to miss, so the CI job
asserts importability instead of trusting a skip.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Protocol, cast

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger("apedp.rust_bridge")

_RUST_AVAILABLE = False

# ------------------------------------------------------------------
# Compile/function caches and kernel-path observability.
#
# Two costs were paid on every call before these existed:
#   1. `kernel.compile(expr, vars)` re-parsed the same expression string at
#      every FFI crossing (per candidate, per generation).
#   2. The Python fallback re-parsed and re-`lambdify`'d identical expression
#      strings on every fallback evaluation.
# Both caches are keyed only on immutable strings, so entries carry no data
# layout assumptions and invalidation is unnecessary: a compiled program is
# a pure function of its text.
# ------------------------------------------------------------------
_MAX_COMPILE_CACHE = 4096
_kernel_compile_cache: OrderedDict[tuple[str, tuple[str, ...]], Any] = OrderedDict()
# The kernel the cache was compiled against. A compiled handle belongs to the
# kernel module that produced it, so when the owner changes (a test fixture
# swaps in a different kernel, a module is reloaded) the cache is cleared
# rather than re-served across kernel identities.
_kernel_compile_owner: list[Any] = [None]
_lambdify_cache: OrderedDict[str, tuple[tuple[str, ...], Any]] = OrderedDict()

# Kernel-path counters: populated per RustKernelBridge instance. These exist
# because a run cannot otherwise prove which path executed -- the fix this
# file's earlier audit called "a counter, not a log line". Benchmarks read
# `bridge.tallies` (or the `kernel_path` field now attached to every result
# dict) to cite the fast path honestly.
_KERNEL_PATH_RUST = "rust"
_KERNEL_PATH_FALLBACK = "fallback"

_missing_kernel_warned = False


class _RustKernel(Protocol):
    """Surface of the pyo3 `apedp_kernel` extension used by this bridge.

    The real module is a compiled extension with no stubs; this protocol gives
    mypy an accurate view of the handful of entry points we call.
    """

    def compile(self, expr: str, variables: list[str]) -> Any: ...
    def evaluate_batch(self, compiled: Any, data: list[np.ndarray]) -> np.ndarray: ...
    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, float, float, float, bool]: ...
    def score_candidates_parallel(
        self, compiled: list[Any], data: list[np.ndarray], target: np.ndarray
    ) -> list[Any]: ...
    def probe_singularity(
        self, compiled: Any, bounds: list[Any], num_grid_points: int
    ) -> tuple[bool, bool, float | None, float, str]: ...
    def compute_matrix_conditioning(
        self, cols: list[np.ndarray], rcond_threshold: float
    ) -> tuple[float, float, float, int, bool, list[float]]: ...
    def solve_linear_coefficients(
        self, cols: list[np.ndarray], target: np.ndarray, lambda_reg: float
    ) -> tuple[list[float], float, float, float, float, bool]: ...
    def evaluate_precision_boundary(
        self, separation_delta: float, condition_number: float, precision_bits: int
    ) -> tuple[str, float, float, float, bool]: ...
    def video_psnr(self, a: np.ndarray, b: np.ndarray) -> float: ...
    def video_ssim_frame(
        self, a_rows: list[np.ndarray], b_rows: list[np.ndarray], win: int, c1: float, c2: float
    ) -> float: ...
    def video_sliced_wasserstein2(
        self, x_rows: list[np.ndarray], y_rows: list[np.ndarray], dir_rows: list[np.ndarray]
    ) -> float: ...
    def video_warp_bilinear_2d(
        self, input: np.ndarray, vx: np.ndarray, vy: np.ndarray, channels: int, height: int, width: int
    ) -> np.ndarray: ...
    def video_compute_optical_flow_divergence(
        self, vx: np.ndarray, vy: np.ndarray, height: int, width: int, dx: float, dy: float
    ) -> np.ndarray: ...
    def video_detect_disocclusion_mask(
        self, div_flow: np.ndarray, divergence_threshold: float
    ) -> np.ndarray: ...
    def evolve_native(
        self,
        data: list[np.ndarray],
        variable_names: list[str],
        target: np.ndarray,
        population_size: int,
        generations: int,
        islands: int,
        migration_interval: int,
        max_depth: int,
        tournament_size: int,
        mutation_rate: float,
        crossover_rate: float,
        allow_trans: bool,
        seed: int,
    ) -> tuple[list[tuple[str, float, int, float, float]], int]: ...


try:
    import apedp_kernel

    _RUST_AVAILABLE = True
except ImportError:
    apedp_kernel = cast(ModuleType, None)  # only read behind use_rust/is_rust_kernel_available()


def is_rust_kernel_available() -> bool:
    """Returns True if the native apedp_kernel extension is loaded and functional."""
    return _RUST_AVAILABLE


class RustKernelBridge:
    """High-performance bridge between APEDP Python orchestration and Rust numerical kernel."""

    def __init__(self, force_fallback: bool = False, dual_validate: bool = False):
        global _missing_kernel_warned
        self.force_fallback = force_fallback
        self.dual_validate = dual_validate
        self.use_rust = _RUST_AVAILABLE and not force_fallback
        # Kernel-path counters (see the module block above). Per instance so a
        # benchmark run's tally cannot be polluted by other runs sharing the
        # module-level default bridge. `tallies` counts *public call outcomes*;
        # `last_path` records the most recent one.
        self.tallies = {_KERNEL_PATH_RUST: 0, _KERNEL_PATH_FALLBACK: 0}
        self.last_path: str | None = None
        if not self.use_rust and not force_fallback and not _missing_kernel_warned:
            # Loud, once, at construction -- not buried at debug level. The
            # bridge is correct without the kernel, but a benchmark that meant
            # to measure the fast path must NOT be able to silently measure
            # the Python reference engine instead.
            logger.warning(
                "apedp_kernel native extension not built; using the pure-Python "
                "reference engine (correct but slower; build with "
                "`maturin develop --release --manifest-path rust/apedp-kernel/Cargo.toml`)"
            )
            _missing_kernel_warned = True

    def _tally(self, path: str) -> str:
        """Records and returns the kernel path that answered this call."""
        self.tallies[path] = self.tallies.get(path, 0) + 1
        self.last_path = path
        return path

    def _kernel_compile(self, kernel: Any, expr_str: str, variables: Sequence[str]) -> Any:
        """LRU-cached FFI crossing to ``kernel.compile``.

        The compiled program is a pure function of ``(expr_str, variables)``,
        so the cache key is exactly that pair of immutable strings. Candidates
        repeat heavily across generations; without the cache every re-serving
        of one re-parsed the same expression inside the kernel.
        """
        if _kernel_compile_owner[0] is not kernel:
            _kernel_compile_cache.clear()
            _kernel_compile_owner[0] = kernel
        key = (expr_str, tuple(variables))
        cached = _kernel_compile_cache.get(key)
        if cached is not None:
            _kernel_compile_cache.move_to_end(key)
            return cached
        compiled = kernel.compile(expr_str, list(variables))
        _kernel_compile_cache[key] = compiled
        while len(_kernel_compile_cache) > _MAX_COMPILE_CACHE:
            _kernel_compile_cache.popitem(last=False)
        return compiled

    def evaluate_expression(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        data: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Evaluates expression over variable arrays using Rust first-line, Python fallback.

        Parameters
        ----------
        expr_str : str
            Expression string (e.g. 'x0**2 + sin(x1)').
        variable_names : Sequence[str]
            List of variable names matching the order in data.
        data : Sequence[np.ndarray]
            List of 1D float64 numpy arrays, one per variable.

        Returns
        -------
        np.ndarray
            1D float64 array of evaluated predictions.
        """
        data_f64 = [np.ascontiguousarray(d, dtype=np.float64) for d in data]

        kernel: _RustKernel | ModuleType = apedp_kernel
        if self.use_rust:
            try:
                compiled = self._kernel_compile(kernel, expr_str, list(variable_names))
                rust_out = kernel.evaluate_batch(compiled, data_f64)
                self._tally(_KERNEL_PATH_RUST)

                if self.dual_validate:
                    py_out = self._fallback_eval(expr_str, variable_names, data_f64)
                    diff = np.max(np.abs(rust_out - py_out))
                    if diff > 1e-10:
                        logger.warning(
                            "Dual-validation divergence for '%s': max diff = %e", expr_str, diff
                        )

                return rust_out
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust kernel eval failed for '%s' (%s), falling back to Python.", expr_str, e)
                self._tally(_KERNEL_PATH_FALLBACK)
                return self._fallback_eval(expr_str, variable_names, data_f64)

        self._tally(_KERNEL_PATH_FALLBACK)
        return self._fallback_eval(expr_str, variable_names, data_f64)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Computes MSE, MAE, R^2, and max residual with Rust first-line."""
        yt = np.ascontiguousarray(y_true, dtype=np.float64)
        yp = np.ascontiguousarray(y_pred, dtype=np.float64)

        if self.use_rust:
            try:
                mse, mae, r2, max_res, has_nan = cast(Any, apedp_kernel).compute_metrics(yt, yp)
                self._tally(_KERNEL_PATH_RUST)
                return {
                    "mse": mse,
                    "mae": mae,
                    "r_squared": r2,
                    "max_residual": max_res,
                    "has_nan_or_inf": has_nan,
                    "kernel_path": _KERNEL_PATH_RUST,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust compute_metrics failed (%s), falling back to Python.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        metrics = self._fallback_metrics(yt, yp)
        metrics["kernel_path"] = _KERNEL_PATH_FALLBACK
        return metrics

    def score_candidates_batch(
        self,
        candidate_exprs: Sequence[str],
        variable_names: Sequence[str],
        data: Sequence[np.ndarray],
        target: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Scores multiple candidate expressions in parallel across CPU cores using Rayon."""
        target_f64 = np.ascontiguousarray(target, dtype=np.float64)
        data_f64 = [np.ascontiguousarray(d, dtype=np.float64) for d in data]

        if self.use_rust:
            try:
                compiled_list = []
                valid_indices = []
                for idx, expr_str in enumerate(candidate_exprs):
                    try:
                        c = cast(Any, apedp_kernel).compile(expr_str, list(variable_names))
                        compiled_list.append(c)
                        valid_indices.append(idx)
                    except Exception:
                        # NOT a verdict. The Rust kernel refusing to compile one
                        # expression is a fallback to "this candidate is not
                        # evaluable by the fast path", and the remaining
                        # compiled_list is still scored normally. Nothing here
                        # derives a pass/fail from the failure: the caller
                        # scores what compiled and cannot see a candidate that
                        # was dropped as though it had scored well. The gap
                        # this leaves is observability -- a run cannot prove
                        # which path executed -- and the fix for that is a
                        # counter, not a log line.
                        #
                        # The same file's verdict-deriving paths are Class A and
                        # were fixed; see physics_constraints.py and
                        # singularity_probing.py for the _unverified() shape
                        # used where a check does decide something.
                        logger.debug("Skipping uncompilable candidate %r", expr_str, exc_info=True)

                if compiled_list:
                    raw_scores = cast(Any, apedp_kernel).score_candidates_parallel(
                        compiled_list, data_f64, target_f64
                    )
                    score_map = {}
                    for cid, mse, mae, r2, max_res, has_nan in raw_scores:
                        orig_idx = valid_indices[cid]
                        score_map[orig_idx] = {
                            "candidate_id": orig_idx,
                            "expression": candidate_exprs[orig_idx],
                            "mse": mse,
                            "mae": mae,
                            "r_squared": r2,
                            "max_residual": max_res,
                            "has_nan_or_inf": has_nan,
                            "kernel_path": _KERNEL_PATH_RUST,
                        }

                    results = []
                    for idx, expr_str in enumerate(candidate_exprs):
                        if idx in score_map:
                            results.append(score_map[idx])
                        else:
                            # Fallback for this individual expression
                            py_eval = self._fallback_eval(expr_str, variable_names, data_f64)
                            metrics = self._fallback_metrics(target_f64, py_eval)
                            metrics["candidate_id"] = idx
                            metrics["expression"] = expr_str
                            metrics["kernel_path"] = _KERNEL_PATH_FALLBACK
                            results.append(metrics)
                    self._tally(_KERNEL_PATH_RUST)
                    return results
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust parallel batch scoring failed (%s), falling back.", e)
                self._tally(_KERNEL_PATH_FALLBACK)

        # Fallback scoring sequentially in Python
        self._tally(_KERNEL_PATH_FALLBACK)
        results = []
        for idx, expr_str in enumerate(candidate_exprs):
            py_eval = self._fallback_eval(expr_str, variable_names, data_f64)
            metrics = self._fallback_metrics(target_f64, py_eval)
            metrics["candidate_id"] = idx
            metrics["expression"] = expr_str
            metrics["kernel_path"] = _KERNEL_PATH_FALLBACK
            results.append(metrics)
        return results

    def probe_singularity(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        domain_bounds: Sequence[tuple[float, float]],
        num_grid_points: int = 2000,
    ) -> dict[str, Any]:
        """Probes for poles and asymptotic blow-ups across domain bounds."""
        if self.use_rust:
            try:
                compiled = self._kernel_compile(kernel=apedp_kernel, expr_str=expr_str, variables=list(variable_names))
                is_sound, has_pole, pole_coord, max_deriv, risk_type = (
                    cast(Any, apedp_kernel).probe_singularity(compiled, list(domain_bounds), num_grid_points)
                )
                self._tally(_KERNEL_PATH_RUST)
                return {
                    "is_sound": is_sound,
                    "has_pole": has_pole,
                    "pole_coordinate": pole_coord,
                    "max_derivative": max_deriv,
                    "risk_type": risk_type,
                    "kernel_path": _KERNEL_PATH_RUST,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust probe_singularity failed (%s), falling back to Python.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        probe = self._fallback_singularity_probe(expr_str, variable_names, domain_bounds, num_grid_points)
        probe["kernel_path"] = _KERNEL_PATH_FALLBACK
        return probe

    def compute_matrix_conditioning(
        self,
        cols: Sequence[np.ndarray],
        rcond_threshold: float = 1e-12,
    ) -> dict[str, Any]:
        """Computes condition number kappa = sigma_max / sigma_min for design matrix columns."""
        col_slices = [np.ascontiguousarray(c, dtype=np.float64) for c in cols]
        if self.use_rust:
            try:
                res = cast(Any, apedp_kernel).compute_matrix_conditioning(
                    col_slices, rcond_threshold
                )
                cond, s_max, s_min, rank, is_well = res[0], res[1], res[2], res[3], res[4]
                s_list = list(res[5]) if len(res) > 5 else [s_max, s_min]
                self._tally(_KERNEL_PATH_RUST)
                return {
                    "condition_number": cond,
                    "max_singular_value": s_max,
                    "min_singular_value": s_min,
                    "effective_rank": rank,
                    "is_well_conditioned": is_well,
                    "singular_values": s_list,
                    "kernel_path": _KERNEL_PATH_RUST,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust compute_matrix_conditioning failed (%s), falling back.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        cond_rep = self._fallback_conditioning(col_slices, rcond_threshold)
        cond_rep["kernel_path"] = _KERNEL_PATH_FALLBACK
        return cond_rep

    def solve_linear_coefficients(
        self,
        basis_cols: Sequence[np.ndarray],
        target: np.ndarray,
        lambda_reg: float = 1e-9,
    ) -> dict[str, Any]:
        """Solves linear parameter least squares min_c ||A c - y||^2 + lambda ||c||^2 via QR."""
        col_slices = [np.ascontiguousarray(c, dtype=np.float64) for c in basis_cols]
        target_f64 = np.ascontiguousarray(target, dtype=np.float64)

        if self.use_rust:
            try:
                coeffs, mse, mae, r2, cond, is_well = cast(Any, apedp_kernel).solve_linear_coefficients(
                    col_slices, target_f64, lambda_reg
                )
                self._tally(_KERNEL_PATH_RUST)
                return {
                    "coefficients": np.array(coeffs, dtype=np.float64),
                    "mse": mse,
                    "mae": mae,
                    "r_squared": r2,
                    "condition_number": cond,
                    "is_well_conditioned": is_well,
                    "kernel_path": _KERNEL_PATH_RUST,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust solve_linear_coefficients failed (%s), falling back.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        linear = self._fallback_linear_solver(col_slices, target_f64, lambda_reg)
        linear["kernel_path"] = _KERNEL_PATH_FALLBACK
        return linear

    def evaluate_precision_boundary(
        self,
        separation_delta: float,
        condition_number: float = 1.0,
        precision_bits: int = 64,
    ) -> dict[str, Any]:
        """Evaluates identifiability boundary under machine precision scaling."""
        if self.use_rust:
            try:
                p_type, eps_mach, eps_star, obs, resolved = (
                    cast(Any, apedp_kernel).evaluate_precision_boundary(separation_delta, condition_number, precision_bits)
                )
                self._tally(_KERNEL_PATH_RUST)
                return {
                    "precision_type": p_type,
                    "machine_epsilon": eps_mach,
                    "theoretical_threshold": eps_star,
                    "empirical_transition_observed": obs,
                    "resolved": resolved,
                    "kernel_path": _KERNEL_PATH_RUST,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust evaluate_precision_boundary failed (%s), falling back.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        boundary = self._fallback_precision_boundary(separation_delta, condition_number, precision_bits)
        boundary["kernel_path"] = _KERNEL_PATH_FALLBACK
        return boundary

    def evolve_native(
        self,
        data: Sequence[np.ndarray],
        variable_names: Sequence[str],
        target: np.ndarray,
        population_size: int = 100,
        generations: int = 25,
        islands: int = 4,
        migration_interval: int = 5,
        max_depth: int = 4,
        tournament_size: int = 4,
        mutation_rate: float = 0.35,
        crossover_rate: float = 0.70,
        allow_trans: bool = True,
        allow_exp: bool = True,
        allow_misc: bool = True,
        allow_advanced: bool = True,
        polish_constants: bool = True,
        polish_top_k: int = 3,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], int]:
        """Runs high-throughput parallel multi-island evolutionary search in Rust."""
        data_f64 = [np.ascontiguousarray(d, dtype=np.float64) for d in data]
        target_f64 = np.ascontiguousarray(target, dtype=np.float64)

        if self.use_rust and hasattr(apedp_kernel, "evolve_native"):
            try:
                raw_pareto, total_evals = cast(Any, apedp_kernel).evolve_native(
                    data_f64,
                    list(variable_names),
                    target_f64,
                    population_size,
                    generations,
                    islands,
                    migration_interval,
                    max_depth,
                    tournament_size,
                    mutation_rate,
                    crossover_rate,
                    allow_trans,
                    allow_exp,
                    allow_misc,
                    allow_advanced,
                    polish_constants,
                    polish_top_k,
                    seed,
                )
                formatted = [
                    {
                        "expression": item[0],
                        "mse": item[1],
                        "complexity": item[2],
                        "scale": item[3],
                        "bias": item[4],
                        "kernel_path": _KERNEL_PATH_RUST,
                    }
                    for item in raw_pareto
                ]
                self._tally(_KERNEL_PATH_RUST)
                return formatted, total_evals
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust evolve_native failed (%s), falling back to Python.", e)
                self._tally(_KERNEL_PATH_FALLBACK)

        # Fallback to pure Python evolve
        self._tally(_KERNEL_PATH_FALLBACK)
        from apedp.evolutionary_search import SearchConfig, evolve

        x_dict = dict(zip(variable_names, data_f64, strict=False))
        cfg = SearchConfig(
            islands=islands,
            island_size=population_size // islands if islands > 0 else population_size,
            generations=generations,
            max_depth=max_depth,
            seed=seed,
        )
        res = evolve(x_dict, target_f64, config=cfg)
        formatted = [
            {"expression": expr, "mse": mse, "complexity": int(compl), "scale": 1.0, "bias": 0.0,
             "kernel_path": _KERNEL_PATH_FALLBACK}
            for expr, mse, compl in res.pareto_front
        ]
        return formatted, res.n_evaluations

    # ------------------ Video metrics (eval_gate first-line port) ------------------

    def video_psnr(self, a: np.ndarray, b: np.ndarray) -> float:
        """Peak SNR (dB) between two [0, 1] arrays; kernel first-line, NumPy fallback.

        Mirrors ``apedp_dyn.video.eval_gate.compute_psnr``: a shape mismatch
        raises identically on both paths, and a mean squared error at or
        below 1e-12 reports the same 100 dB ceiling (metric definition, not a
        referee threshold).
        """
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        if a_arr.shape != b_arr.shape:
            raise ValueError(f"Shape mismatch: {a_arr.shape} vs {b_arr.shape}")
        if self.use_rust:
            try:
                out = float(cast(Any, apedp_kernel).video_psnr(
                    np.ascontiguousarray(a_arr).reshape(-1),
                    np.ascontiguousarray(b_arr).reshape(-1),
                ))
                self._tally(_KERNEL_PATH_RUST)
                return out
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust video_psnr failed (%s), falling back to NumPy.", e)
        self._tally(_KERNEL_PATH_FALLBACK)
        mse = float(np.mean((a_arr - b_arr) ** 2))
        return 100.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))

    def video_ssim_frame(
        self,
        a_gray: np.ndarray,
        b_gray: np.ndarray,
        win: int = 8,
        c1: float = 1e-4,
        c2: float = 9e-4,
    ) -> float:
        """Mean windowed SSIM over ONE grayscale frame (2-D), kernel first-line.

        Defaults are the canonical squared stabilizers for unit dynamic range
        ((K1*L)^2 = 1e-4, (K2*L)^2 = 9e-4). Rows cross the boundary as
        zero-copy views of the frame buffer. A frame smaller than the window
        yields NaN on both paths (matching the reference's empty-block mean).
        """
        a_g = np.asarray(a_gray, dtype=np.float64)
        b_g = np.asarray(b_gray, dtype=np.float64)
        if a_g.shape != b_g.shape or a_g.ndim != 2:
            return float("nan")
        if self.use_rust:
            try:
                out = float(cast(Any, apedp_kernel).video_ssim_frame(
                    list(a_g), list(b_g), int(win), c1, c2
                ))
                self._tally(_KERNEL_PATH_RUST)
                return out
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust video_ssim_frame failed (%s), falling back to NumPy.", e)
        self._tally(_KERNEL_PATH_FALLBACK)
        h, w = a_g.shape
        vals = []
        for y in range(0, h - win + 1, win):
            for x in range(0, w - win + 1, win):
                pa, pb = a_g[y:y + win, x:x + win], b_g[y:y + win, x:x + win]
                ma, mb = float(pa.mean()), float(pb.mean())
                va, vb = float(pa.var()), float(pb.var())
                cov = float(((pa - ma) * (pb - mb)).mean())
                vals.append((2 * ma * mb + c1) * (2 * cov + c2)
                            / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))
        return float(np.mean(vals)) if vals else float("nan")

    def video_sliced_wasserstein(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_projections: int = 128,
        seed: int = 7,
    ) -> float:
        """Sliced Wasserstein-2 between (n, d) clouds; kernel first-line.

        The projection matrix is generated numpy-side exactly as the
        reference does (``RandomState(seed)``), so kernel and reference
        distances are identically seeded, not merely close.
        """
        x_mat = np.asarray(x, dtype=np.float64).reshape(-1, np.asarray(x).shape[-1])
        y_mat = np.asarray(y, dtype=np.float64).reshape(-1, np.asarray(y).shape[-1])
        if x_mat.shape[-1] != y_mat.shape[-1]:
            raise ValueError("Feature dimension mismatch")
        if x_mat.shape[0] < 2 or y_mat.shape[0] < 2:
            raise ValueError("Need at least 2 points per distribution")
        rng = np.random.RandomState(seed)
        d = x_mat.shape[-1]
        dirs = rng.normal(0.0, 1.0, size=(n_projections, d))
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12
        if self.use_rust:
            try:
                out = float(cast(Any, apedp_kernel).video_sliced_wasserstein2(
                    list(np.ascontiguousarray(x_mat)),
                    list(np.ascontiguousarray(y_mat)),
                    list(dirs),
                ))
                self._tally(_KERNEL_PATH_RUST)
                return out
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust video_sliced_wasserstein2 failed (%s), falling back to NumPy.", e)
        self._tally(_KERNEL_PATH_FALLBACK)
        # Reference implementation, INLINE (like every other bridge fallback):
        # a lazy import of eval_gate's routed `sliced_wasserstein` would
        # re-enter the bridge and recurse under force_fallback.
        total = 0.0
        for k in range(n_projections):
            px = np.sort(x_mat @ dirs[k])
            py = np.sort(y_mat @ dirs[k])
            n = min(len(px), len(py))
            total += float(np.mean((np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(px)), px) -
                                    np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(py)), py)) ** 2))
        return float(np.sqrt(total / n_projections))

    def video_warp_2d(
        self,
        image: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        *,
        periodic: bool = False,
    ) -> np.ndarray:
        """2D Bilinear Sub-Pixel Coordinate Warping of an image/latent field.

        Args:
            image: Array of shape (C, H, W) or (H, W).
            vx: Horizontal displacement array of shape (H, W).
            vy: Vertical displacement array of shape (H, W).
            periodic: wrap-around (toroidal) boundary conditions matching the
                fluid-domain generator's `floor(q) % size` backtrace indexing
                (float is floored first, the integer index wrapped, frac =
                q - floor(q)). Default False = border clamping.

        Returns:
            Warped array with the same shape as `image`.
        """
        arr = np.asarray(image, dtype=np.float64)
        is_2d = arr.ndim == 2
        if is_2d:
            arr = arr[np.newaxis, :, :]
        c, h, w = arr.shape
        vx_arr = np.asarray(vx, dtype=np.float64)
        vy_arr = np.asarray(vy, dtype=np.float64)
        if vx_arr.shape != (h, w) or vy_arr.shape != (h, w):
            raise ValueError(f"Flow shapes ({vx_arr.shape}, {vy_arr.shape}) must match image plane ({h}, {w})")

        if self.use_rust:
            try:
                if periodic:
                    fn = getattr(cast(Any, apedp_kernel), "video_warp_bilinear_2d_periodic", None)
                    if fn is None:
                        raise AttributeError("video_warp_bilinear_2d_periodic absent from kernel build")
                else:
                    fn = cast(Any, apedp_kernel).video_warp_bilinear_2d
                out = fn(
                    np.ascontiguousarray(arr).reshape(-1),
                    np.ascontiguousarray(vx_arr).reshape(-1),
                    np.ascontiguousarray(vy_arr).reshape(-1),
                    c, h, w,
                )
                self._tally(_KERNEL_PATH_RUST)
                res = np.asarray(out, dtype=np.float64).reshape(c, h, w)
                return res[0] if is_2d else res
            except Exception as e:  # noqa: BLE001
                logger.debug("Rust video_warp_bilinear_2d failed (%s), falling back to NumPy.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        # Vectorized NumPy fallback
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        if periodic:
            # Generator semantics: floor the backtrace coordinate FIRST, wrap
            # the integer index (numpy `%`), frac = q - floor(q). Wrapping the
            # float first would lose exactness for tiny negative backtraces.
            qx = x_grid + vx_arr
            qy = y_grid + vy_arr
            fx = np.floor(qx)
            fy = np.floor(qy)
            x0 = np.mod(fx.astype(np.int64), w)
            y0 = np.mod(fy.astype(np.int64), h)
            x1 = (x0 + 1) % w
            y1 = (y0 + 1) % h
            wx = qx - fx
            wy = qy - fy
        else:
            sx = np.clip(x_grid + vx_arr, 0.0, w - 1.0)
            sy = np.clip(y_grid + vy_arr, 0.0, h - 1.0)
            x0 = np.floor(sx).astype(np.int64)
            y0 = np.floor(sy).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, w - 1)
            y1 = np.clip(y0 + 1, 0, h - 1)
            wx = sx - x0
            wy = sy - y0

        warped = np.zeros_like(arr)
        for ch in range(c):
            plane = arr[ch]
            top = plane[y0, x0] * (1.0 - wx) + plane[y0, x1] * wx
            bottom = plane[y1, x0] * (1.0 - wx) + plane[y1, x1] * wx
            warped[ch] = top * (1.0 - wy) + bottom * wy
        return warped[0] if is_2d else warped

    def video_optical_flow_divergence(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0,
    ) -> np.ndarray:
        """Compute spatial divergence of optical flow: div(v) = d(vx)/dx + d(vy)/dy.

        Args:
            vx: Horizontal velocity field of shape (H, W).
            vy: Vertical velocity field of shape (H, W).
            dx: Spatial step size in x.
            dy: Spatial step size in y.

        Returns:
            Divergence field of shape (H, W).
        """
        vx_arr = np.asarray(vx, dtype=np.float64)
        vy_arr = np.asarray(vy, dtype=np.float64)
        if vx_arr.shape != vy_arr.shape:
            raise ValueError(f"Shape mismatch: {vx_arr.shape} vs {vy_arr.shape}")
        if vx_arr.ndim != 2:
            raise ValueError("Optical flow components must be 2-dimensional (H, W)")
        h, w = vx_arr.shape
        if h < 3 or w < 3:
            raise ValueError("Grid dimensions must be at least 3x3")

        if self.use_rust:
            try:
                out = cast(Any, apedp_kernel).video_compute_optical_flow_divergence(
                    np.ascontiguousarray(vx_arr).reshape(-1),
                    np.ascontiguousarray(vy_arr).reshape(-1),
                    h, w, dx, dy,
                )
                self._tally(_KERNEL_PATH_RUST)
                return np.asarray(out, dtype=np.float64).reshape(h, w)
            except Exception as e:  # noqa: BLE001
                logger.debug("Rust video_compute_optical_flow_divergence failed (%s), falling back to NumPy.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        dvx_dx = np.gradient(vx_arr, dx, axis=1)
        dvy_dy = np.gradient(vy_arr, dy, axis=0)
        return dvx_dx + dvy_dy

    def video_disocclusion_mask(
        self,
        div_flow: np.ndarray,
        divergence_threshold: float = 0.0,
    ) -> np.ndarray:
        """Detect disocclusion / expansion regions triggering residual texture injection.

        Args:
            div_flow: Divergence array of shape (H, W) or 1D.
            divergence_threshold: Threshold above which divergence triggers source injection.

        Returns:
            Disocclusion intensity array in [0.0, 1.0].
        """
        div_arr = np.asarray(div_flow, dtype=np.float64)
        orig_shape = div_arr.shape

        if self.use_rust:
            try:
                out = cast(Any, apedp_kernel).video_detect_disocclusion_mask(
                    np.ascontiguousarray(div_arr).reshape(-1),
                    divergence_threshold,
                )
                self._tally(_KERNEL_PATH_RUST)
                return np.asarray(out, dtype=np.float64).reshape(orig_shape)
            except Exception as e:  # noqa: BLE001
                logger.debug("Rust video_detect_disocclusion_mask failed (%s), falling back to NumPy.", e)

        self._tally(_KERNEL_PATH_FALLBACK)
        return np.clip(np.maximum(0.0, div_arr - divergence_threshold), 0.0, 1.0)

    # ------------------ Fallback Pure-Python Implementations ------------------

    def _fallback_eval(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        data: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Pure-Python / NumPy reference evaluator.

        The expression string is untrusted: it arrives from candidate search,
        CSV ingestion, and the REST layer, so it is parsed with the same
        allowlist policy as every other string-in path (:func:`safe_parse_expr`)
        and compiled with sympy's ``lambdify`` -- never handed to ``eval``.

        Passing ``{"__builtins__": {}}`` to ``eval`` is not a sandbox. Attribute
        traversal still reaches every class in the interpreter, so
        ``(1).__class__.__mro__[1].__subclasses__()`` executes and yields the
        live class table. The allowlist parser rejects that shape up front, and
        ``lambdify`` then emits code containing only the numeric calls the
        expression actually references.
        """
        shape = data[0].shape
        nan_result = lambda: np.full(shape, np.nan, dtype=np.float64)  # noqa: E731

        # The reference engine's own cost was re-parsing + re-lambdifying the
        # same expression on every call. The compiled callable is a pure
        # function of the expression text (pi/e are substituted into the body
        # by the parser), so it is cached by `expr_str` alone.
        cached_entry = _lambdify_cache.get(expr_str)
        if cached_entry is None:
            try:
                expr = safe_parse_expr(expr_str)
            except (TypeError, ValueError) as e:
                logger.debug("Rejected expression %r in Python fallback (%s).", expr_str, e)
                return nan_result()

            # ``lambdify`` builds a signature from the expression's free symbols
            # only, so pass exactly those; ``pi``/``e`` are already substituted
            # into the expression body by the parser.
            argnames = tuple(sorted(str(s) for s in expr.free_symbols))
            func = sp.lambdify(
                argnames,
                expr,
                modules="numpy",
                cse=False,
                docstring_limit=0,
            )
            cached_entry = (argnames, func)
            _lambdify_cache[expr_str] = cached_entry
            while len(_lambdify_cache) > _MAX_COMPILE_CACHE:
                _lambdify_cache.popitem(last=False)
        else:
            _lambdify_cache.move_to_end(expr_str)

        argnames, func = cached_entry
        # A variable named in the data shadows a same-named mathematical
        # constant, matching the previous hand-built environment.
        env: dict[str, Any] = {"pi": np.pi, "e": np.e}
        env.update(dict(zip(variable_names, data, strict=False)))

        try:
            res = func(*(env[name] for name in argnames))
            if np.isscalar(res):
                return np.full(shape, float(cast(Any, res)), dtype=np.float64)
            arr = np.asarray(res, dtype=np.float64)
            # ``lambdify`` broadcasts constants; broadcast to the caller's shape
            # so downstream ``invalid_mask`` arithmetic stays aligned.
            if arr.shape != shape:
                arr = np.broadcast_to(arr, shape)
            return np.ascontiguousarray(arr)
        except Exception:  # noqa: BLE001 - best-effort probe must not crash
            return nan_result()

    def _fallback_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Pure-Python / NumPy reference metrics."""
        if len(y_true) == 0 or len(y_true) != len(y_pred):
            return {
                "mse": float("nan"),
                "mae": float("nan"),
                "r_squared": float("nan"),
                "max_residual": float("nan"),
                "has_nan_or_inf": True,
            }

        residuals = y_true - y_pred
        has_nan = bool(np.isnan(residuals).any() or np.isinf(residuals).any())
        if has_nan:
            return {
                "mse": float("nan"),
                "mae": float("nan"),
                "r_squared": float("nan"),
                "max_residual": float("nan"),
                "has_nan_or_inf": True,
            }

        mse = float(np.mean(residuals**2))
        mae = float(np.mean(np.abs(residuals)))
        max_res = float(np.max(np.abs(residuals)))

        var_y = float(np.var(y_true))
        r2 = 1.0 - (mse / var_y) if var_y > 1e-15 else (1.0 if mse < 1e-15 else 0.0)

        return {
            "mse": mse,
            "mae": mae,
            "r_squared": r2,
            "max_residual": max_res,
            "has_nan_or_inf": False,
        }

    def _fallback_singularity_probe(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        domain_bounds: Sequence[tuple[float, float]],
        num_grid_points: int,
    ) -> dict[str, Any]:
        """Pure-Python / NumPy reference singularity probe."""
        if not domain_bounds:
            return {
                "is_sound": True,
                "has_pole": False,
                "pole_coordinate": None,
                "max_derivative": 0.0,
                "risk_type": "NONE",
            }

        min_x, max_x = domain_bounds[0]
        grid = np.linspace(min_x, max_x, num_grid_points + 1, dtype=np.float64)
        step = (max_x - min_x) / max(num_grid_points, 1)

        data = [grid]
        for b in domain_bounds[1:]:
            mid = (b[0] + b[1]) / 2.0
            data.append(np.full_like(grid, mid))

        y = self._fallback_eval(expr_str, variable_names, data)
        invalid_mask = ~np.isfinite(y) | (np.abs(y) > 1e6)

        has_pole = False
        pole_coord = None
        if np.any(invalid_mask):
            has_pole = True
            first_idx = np.where(invalid_mask)[0][0]
            pole_coord = float(grid[first_idx])

        diffs = np.abs(np.diff(y)) / step
        finite_diffs = diffs[np.isfinite(diffs)]
        max_deriv = float(np.max(finite_diffs)) if len(finite_diffs) > 0 else 0.0

        if not has_pole:
            for i in range(len(grid) - 1):
                d = diffs[i]
                if np.isfinite(d) and d > 1e4:
                    sign_flip = (y[i] * y[i + 1] < 0.0) and (abs(y[i]) > 50.0 or abs(y[i + 1]) > 50.0)
                    if sign_flip or abs(y[i]) > 500.0 or abs(y[i + 1]) > 500.0:
                        has_pole = True
                        pole_coord = float(grid[i])
                        break

        is_sound = not has_pole
        risk_type = "POLE_IN_DOMAIN" if has_pole else "NONE"

        return {
            "is_sound": is_sound,
            "has_pole": has_pole,
            "pole_coordinate": pole_coord,
            "max_derivative": max_deriv,
            "risk_type": risk_type,
        }

    def _fallback_conditioning(
        self,
        cols: Sequence[np.ndarray],
        rcond_threshold: float,
    ) -> dict[str, Any]:
        """Pure-Python / NumPy SVD condition number fallback.

        A conditioning probe must never crash: SVD non-convergence (or a
        non-finite design matrix, e.g. NaN Jacobian columns from sqrt of
        negative inputs) IS the degeneracy signal, reported as ill-conditioned.
        """
        if not cols:
            return {
                "condition_number": float("inf"),
                "max_singular_value": 0.0,
                "min_singular_value": 0.0,
                "effective_rank": 0,
                "is_well_conditioned": False,
            }
        try:
            mat = np.column_stack(cols)
            if not np.all(np.isfinite(mat)):
                raise np.linalg.LinAlgError("non-finite design matrix")
            s = np.linalg.svd(mat, compute_uv=False)
        except (np.linalg.LinAlgError, ArithmeticError, ValueError):
            logger.debug("SVD conditioning probe degenerate; reporting ill-conditioned", exc_info=True)
            return {
                "condition_number": float("inf"),
                "max_singular_value": 0.0,
                "min_singular_value": 0.0,
                "effective_rank": 0,
                "is_well_conditioned": False,
            }
        sigma_max = float(s[0]) if len(s) > 0 else 0.0
        sigma_min = float(s[-1]) if len(s) > 0 else 0.0
        cond = float(sigma_max / sigma_min) if sigma_min > 1e-15 else float("inf")
        rank = int(np.sum(s > sigma_max * rcond_threshold))
        is_well = cond < (1.0 / max(rcond_threshold, 1e-15))
        return {
            "condition_number": cond,
            "max_singular_value": sigma_max,
            "min_singular_value": sigma_min,
            "effective_rank": rank,
            "is_well_conditioned": is_well,
            "singular_values": [float(v) for v in s],
        }

    def _fallback_linear_solver(
        self,
        basis_cols: Sequence[np.ndarray],
        target: np.ndarray,
        lambda_reg: float,
    ) -> dict[str, Any]:
        """Pure-Python / NumPy least squares solver fallback."""
        mat = np.column_stack(basis_cols)
        k = mat.shape[1]
        if lambda_reg > 1e-15:
            # Regularized normal equations, with SVD-based downgrade when the
            # Cholesky/solve path fails on a degenerate system.
            ata = mat.T @ mat + lambda_reg * np.eye(k)
            aty = mat.T @ target
            try:
                coeffs = np.linalg.solve(ata, aty)
            except np.linalg.LinAlgError:
                logger.debug("Normal-equations solve failed; downgrading to lstsq", exc_info=True)
                coeffs, _, _, _ = np.linalg.lstsq(mat, target, rcond=None)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(mat, target, rcond=None)

        pred = mat @ coeffs
        residuals = target - pred
        mse = float(np.mean(residuals**2))
        mae = float(np.mean(np.abs(residuals)))
        var_y = float(np.var(target))
        r2 = 1.0 - (mse / var_y) if var_y > 1e-15 else (1.0 if mse < 1e-15 else 0.0)
        cond_rep = self._fallback_conditioning(basis_cols, 1e-12)

        return {
            "coefficients": coeffs,
            "mse": mse,
            "mae": mae,
            "r_squared": r2,
            "condition_number": cond_rep["condition_number"],
            "is_well_conditioned": cond_rep["is_well_conditioned"],
        }

    def _fallback_precision_boundary(
        self,
        separation_delta: float,
        condition_number: float,
        precision_bits: int,
    ) -> dict[str, Any]:
        """Pure-Python precision boundary calculation fallback."""
        if precision_bits == 32:
            eps_mach = float(np.finfo(np.float32).eps)
            p_name = "Float32 (Single Precision)"
        elif precision_bits == 64:
            eps_mach = float(np.finfo(np.float64).eps)
            p_name = "Float64 (Double Precision)"
        else:
            eps_mach = float(2.0 ** (-(precision_bits - 1)))
            p_name = f"CustomPrecision ({precision_bits}-bit)"

        eps_star = float(np.sqrt(2.0 * eps_mach))
        effective_noise = eps_mach * max(condition_number, 1.0)
        resolved = bool(separation_delta > effective_noise and separation_delta > eps_star)

        return {
            "precision_type": p_name,
            "machine_epsilon": eps_mach,
            "theoretical_threshold": eps_star,
            "empirical_transition_observed": resolved,
            "resolved": resolved,
        }


# Global default kernel instance
default_kernel = RustKernelBridge()
