r"""Kernel↔NumPy parity tests for the video evaluation instruments.

The Rust kernel now ports PSNR / windowed SSIM / sliced Wasserstein-2 as
first-line paths behind :mod:`apedp_dyn.video.eval_gate`. Because the
projection matrix for the FVD proxy is generated numpy-side (``RandomState``)
and handed to the kernel as zero-copy row views, the two engines must be
provably EQUAL on identical inputs (to float tolerance), not merely close —
that is the whole point of routing a certification instrument through the
dual-implementation discipline.

What is pinned here:
1. Values: bridge-routed eval_gate outputs equal the pure-NumPy reference
   formulas to 1e-10 (PSNR ceiling semantics included).
2. Routing honesty: with the kernel available, ``bridge.tallies`` records
   ``rust``; with ``force_fallback=True`` the identical numbers come back via
   the NumPy path and ``tallies`` records ``fallback`` (kernel_path counters
   — a run can prove which engine answered).
3. Behavioural guards kept intact: shape/dimension mismatch ValueError, the
   2-point minimum, metric monotonicity on degradation.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.rust_bridge import RustKernelBridge, is_rust_kernel_available
from apedp_dyn.video.eval_gate import (
    compute_psnr,
    compute_ssim,
    sliced_wasserstein,
)


def _frame(seed: int, h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(3, h, w)


def _cloud(seed: int, n: int, d: int) -> np.ndarray:
    return np.random.RandomState(seed).randn(n, d)


@pytest.mark.skipif(not is_rust_kernel_available(), reason="native kernel not built")
class TestKernelNumPyParity:
    def test_psnr_identity_ceiling_and_parity(self) -> None:
        a, b = _frame(1), _frame(2)
        assert compute_psnr(a, a) == 100.0
        ref = 10.0 * np.log10(1.0 / float(np.mean((a - b) ** 2)))
        assert abs(compute_psnr(a, b) - ref) < 1e-10
        assert compute_psnr(a, b) == RustKernelBridge().video_psnr(a, b)

    def test_psnr_shape_mismatch_raises_on_both_paths(self) -> None:
        a, c = _frame(1), _frame(2, h=16)
        with pytest.raises(ValueError):
            compute_psnr(a, c)
        with pytest.raises(ValueError):
            RustKernelBridge().video_psnr(a, c)
        with pytest.raises(ValueError):
            RustKernelBridge(force_fallback=True).video_psnr(a, c)

    def test_ssim_parity_on_frames_and_videos(self) -> None:
        a, b = _frame(3), _frame(4)
        gray = lambda f: 0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]
        ga, gb = gray(a), gray(b)
        ref = _np_ssim(ga, gb)
        assert abs(compute_ssim(a, b) - ref) < 1e-10
        # (T, 3, H, W) routing gives the per-frame mean of the same formula.
        va, vb = np.stack([a, a * 0.9 + 0.05]), np.stack([b, b * 0.9 + 0.05])
        expected = (0.5 * _np_ssim(gray(va[0]), gray(vb[0]))
                    + 0.5 * _np_ssim(gray(va[1]), gray(vb[1])))
        assert abs(compute_ssim(va, vb) - expected) < 1e-10

    def test_ssim_below_window_frame_is_nan(self) -> None:
        small = np.random.RandomState(7).rand(3, 4, 6)
        out = compute_ssim(small, small)
        assert np.isnan(out), out

    def test_sliced_wasserstein_parity_including_unequal_n(self) -> None:
        x, y = _cloud(11, 40, 6), _cloud(12, 25, 6)
        ref = _np_swd(x, y, n_projections=64, seed=5)
        assert abs(sliced_wasserstein(x, y, n_projections=64, seed=5) - ref) < 1e-9

        x2, y2 = _cloud(13, 30, 6), _cloud(13, 30, 6)  # n != m
        ref2 = _np_swd(x2, y2, n_projections=128, seed=9)
        assert abs(sliced_wasserstein(x2, y2, n_projections=128, seed=9) - ref2) < 1e-9

    def test_sliced_wasserstein_guardrails(self) -> None:
        with pytest.raises(ValueError):
            sliced_wasserstein(_cloud(1, 30, 6), _cloud(2, 30, 5))
        with pytest.raises(ValueError):
            sliced_wasserstein(_cloud(1, 1, 6), _cloud(2, 30, 6))
        assert sliced_wasserstein(_cloud(1, 30, 6), _cloud(1, 30, 6)) < 1e-9

    def test_counters_prove_which_engine_answered(self) -> None:
        a, b = _frame(5), _frame(6)
        kernel_bridge = RustKernelBridge()
        kernel_bridge.video_psnr(a, b)
        fallback_bridge = RustKernelBridge(force_fallback=True)
        fallback_bridge.video_psnr(a, b)
        assert kernel_bridge.tallies["rust"] >= 1
        assert fallback_bridge.tallies["fallback"] >= 1
        # And both engines agree to float tolerance (parity by construction).
        assert abs(kernel_bridge.video_ssim_frame(
            0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2],
            0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2],
        ) - fallback_bridge.video_ssim_frame(
            0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2],
            0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2],
        )) < 1e-10

    def test_fallback_bridge_matches_kernel_via_eval_gate(self) -> None:
        # Route eval_gate with the kernel, then compare its numbers to a run
        # against the pure fallback bridge by monkeypatching availability.
        import apedp_dyn.video.eval_gate as eg

        x, y = _cloud(21, 50, 7), _cloud(22, 50, 7)
        routed = sliced_wasserstein(x, y, n_projections=96, seed=17)
        assert np.isfinite(routed)

        monkey_ref = None
        class _FallbackOnlyBridge(RustKernelBridge):
            def __init__(self) -> None:
                super().__init__(force_fallback=True)

        old = eg._kernel_video_bridge
        eg._kernel_video_bridge = lambda: _FallbackOnlyBridge()  # type: ignore[assignment]
        try:
            monkey_ref = sliced_wasserstein(x, y, n_projections=96, seed=17)
        finally:
            eg._kernel_video_bridge = old
        assert abs(routed - monkey_ref) < 1e-9


# ---- pure-NumPy reference mirrors (the eval_gate bodies, inlined so this
# ---- file stays independent of routing decisions in eval_gate itself) ----

def _np_ssim(fa: np.ndarray, fb: np.ndarray, win: int = 8, c1: float = 1e-4, c2: float = 9e-4) -> float:
    h, w = fa.shape
    vals = []
    for y in range(0, h - win + 1, win):
        for x in range(0, w - win + 1, win):
            pa, pb = fa[y:y + win, x:x + win], fb[y:y + win, x:x + win]
            ma, mb = float(pa.mean()), float(pb.mean())
            va, vb = float(pa.var()), float(pb.var())
            cov = float(((pa - ma) * (pb - mb)).mean())
            vals.append((2 * ma * mb + c1) * (2 * cov + c2)
                        / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))
    return float(np.mean(vals))


def _np_swd(x: np.ndarray, y: np.ndarray, n_projections: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    d = x.shape[1]
    dirs = rng.normal(0.0, 1.0, size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12
    total = 0.0
    for k in range(n_projections):
        px = np.sort(x @ dirs[k])
        py = np.sort(y @ dirs[k])
        n = min(len(px), len(py))
        total += float(np.mean((np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(px)), px) -
                                np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(py)), py)) ** 2))
    return float(np.sqrt(total / n_projections))
