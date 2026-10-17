"""Periodic-boundary warp pins: torch / Rust kernel / NumPy fallback must all
reproduce the fluid generator's OWN advection semantics.

Property pinned (not tuned to green): the generator (`fluid_domain._advect`)
backtraces with `back = x - dt*v`, takes `floor(back) % size` for the integer
sample indices and `q - floor(q)` for the fractional weights. Every warp path
used by the learned-flow pipeline must be bit-class-equal to that formula on
wrap-heavy flows (several full cycles, negative backtraces, exact border
landings):

- `warp_frames(..., periodic=True)` (torch training loss + gradient path),
- `RustKernelBridge.video_warp_2d(..., periodic=True)` routed to the Rust
  `video_warp_bilinear_2d_periodic` kernel (DC-STT rollout),
- the same bridge under `force_fallback=True` (the documented fallback path).

Tolerances: 1e-12 absolute. All three implementations evaluate the SAME
formula up to floating-point operation order (reference computes the 4-term
form; engines compute top/bottom then lerp); measured deviation is < 1e-15,
so 1e-12 is the operation-order class bound, not a fitted threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apedp.rust_bridge import RustKernelBridge
from apedp_dyn.video.fluid_domain import _advect
from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport,
    DecoupledTransportConfig,
)
from apedp_dyn.video.learned_flow_head import warp_frames


def _wrap_heavy_flow(h: int, w: int, seed: int = 3):
    """Displacements spanning several full domain cycles (multi-wrap) with
    negative and positive backtraces and near-integer landings."""
    rng = np.random.RandomState(seed)
    vx = rng.uniform(-1.5 * w, 1.5 * w, size=(h, w))
    vy = rng.uniform(-1.5 * h, 1.5 * h, size=(h, w))
    vx[0, 0] = float(w)          # exact one full cycle
    vy[0, 1] = -float(h)         # exact negative full cycle
    vx[1, 0] = 0.5               # half-pixel landing
    vy[1, 1] = -0.5
    return vx, vy


def _reference(field: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    # DC-STT convention: warp samples at x + disp with disp = -dt*v; the
    # generator's (u, v) are (y-velocity, x-velocity).
    return _advect(field, u=vy, v=vx, dt=1.0)


def test_torch_periodic_warp_matches_generator_formula() -> None:
    import torch

    h, w = 9, 11
    rng = np.random.RandomState(0)
    field = rng.rand(3, h, w)
    vx, vy = _wrap_heavy_flow(h, w)
    ref = _reference(field, vx, vy)

    out = warp_frames(torch.from_numpy(field)[None],
                      torch.from_numpy(vx)[None], torch.from_numpy(vy)[None],
                      dt=1.0, periodic=True)[0].numpy()
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-12, f"torch periodic warp != generator formula ({err:.3e})"


def test_periodic_warp_is_not_clamped_at_the_border() -> None:
    # Sanity that the flag actually switches semantics: for the wrap-heavy
    # flow the periodic and clamped results must differ materially.
    import torch

    h, w = 9, 11
    rng = np.random.RandomState(1)
    field = rng.rand(3, h, w)
    vx, vy = _wrap_heavy_flow(h, w)
    per = warp_frames(torch.from_numpy(field)[None],
                      torch.from_numpy(vx)[None], torch.from_numpy(vy)[None],
                      dt=1.0, periodic=True)[0].numpy()
    cla = warp_frames(torch.from_numpy(field)[None],
                      torch.from_numpy(vx)[None], torch.from_numpy(vy)[None],
                      dt=1.0, periodic=False)[0].numpy()
    assert float(np.max(np.abs(per - cla))) > 1e-3


def test_bridge_fallback_periodic_matches_generator_formula() -> None:
    h, w = 9, 11
    rng = np.random.RandomState(2)
    field = rng.rand(3, h, w)
    vx, vy = _wrap_heavy_flow(h, w)
    ref = _reference(field, vx, vy)

    bridge = RustKernelBridge(force_fallback=True)
    out = bridge.video_warp_2d(field, -1.0 * vx, -1.0 * vy, periodic=True)
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-12, f"NumPy fallback periodic warp != generator formula ({err:.3e})"


def test_rust_kernel_periodic_matches_torch_and_generator() -> None:
    apedp_kernel = pytest.importorskip("apedp_kernel", reason="rust kernel not built/installed")
    import torch

    # Sharp pin: WITHOUT this binding the bridge silently falls back to NumPy
    # and the test would pass vacuously. The installed kernel must actually
    # carry the periodic warp.
    assert hasattr(apedp_kernel, "video_warp_bilinear_2d_periodic"), (
        "installed kernel build lacks video_warp_bilinear_2d_periodic; "
        "rebuild with maturin (see rust/apedp-kernel)"
    )

    h, w = 9, 11
    rng = np.random.RandomState(4)
    field = rng.rand(3, h, w)
    vx, vy = _wrap_heavy_flow(h, w)
    ref = _reference(field, vx, vy)

    bridge = RustKernelBridge(force_fallback=False)
    out = bridge.video_warp_2d(field, -1.0 * vx, -1.0 * vy, periodic=True)
    err_ref = float(np.max(np.abs(out - ref)))
    assert err_ref < 1e-12, f"Rust periodic warp != generator formula ({err_ref:.3e})"

    torch_out = warp_frames(torch.from_numpy(field)[None],
                            torch.from_numpy(vx)[None], torch.from_numpy(vy)[None],
                            dt=1.0, periodic=True)[0].numpy()
    err_parity = float(np.max(np.abs(out - torch_out)))
    assert err_parity < 1e-12, f"Rust vs torch periodic parity {err_parity:.3e}"


@pytest.mark.parametrize("use_rust", [False, True])
def test_dcstt_transport_step_periodic_matches_generator_advection(use_rust: bool) -> None:
    if use_rust:
        kernel = pytest.importorskip("apedp_kernel", reason="rust kernel not built/installed")
        assert hasattr(kernel, "video_warp_bilinear_2d_periodic"), (
            "installed kernel build lacks video_warp_bilinear_2d_periodic; "
            "the use_rust=True arm would silently exercise the NumPy fallback"
        )
    h, w = 9, 11
    rng = np.random.RandomState(5)
    field = rng.rand(3, h, w)
    vx, vy = _wrap_heavy_flow(h, w)
    ref = _reference(field, vx, vy)

    solver = DecoupledSpatioTemporalTransport(DecoupledTransportConfig(
        use_rust=use_rust, nu=0.0, enable_laplacian_dissipation=False,
        enforce_mass_conservation=False, periodic_warp=True,
    ))
    out, diag = solver.transport_step(field, vx, vy, dt=1.0)
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-12, (
        f"DC-STT transport_step (use_rust={use_rust}) periodic != generator advection "
        f"({err:.3e})"
    )
    assert diag["step_dt"] == 1.0
