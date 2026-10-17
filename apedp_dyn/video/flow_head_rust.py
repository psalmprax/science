r"""Rust-backed learned flow head: the boundary-hop layer.

Status: experimental

Mirrors the torch-side representations of
`apedp_dyn/video/learned_flow_head.py` so the SAME weights can be executed by
either engine, and drives the kernel's single-crossing transport step
(`apedp_kernel.flow_head_transport_step`) so a rollout makes exactly ONE
Python<->Rust hop per frame instead of one hop per kernel call (the measured
3x-class gap this landing attacks).

Design contract (pinned in tests/test_dyn_flow_head_rust.py):
- torch `nn.Conv2d` weights flatten row-major to (o, i, ky, kx) — the kernel's
  layer-major layout packs the same numbers in the same order, so packing is a
  plain concatenate.
- Kernel GELU uses an erf approximation (|err| <= 1.5e-7); the parity tolerance
  in tests accounts for it.
- Index selection (frame triples) stays a python/RNG concern; the kernel stays
  pure and parallel-safe (py.allow_threads in every entrypoint).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # kernel is optional at runtime: fall back loudly, never silently
    import apedp_kernel
    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in fallback mode
    apedp_kernel = None
    _RUST_AVAILABLE = False


@dataclass(frozen=True)
class RustFlowHeadSpec:
    """Flat mirror of `FlowHeadConfig` + kernel-side architecture fields."""

    context_channels: int = 2
    hidden_dim: int = 24
    n_layers: int = 3
    flow_scale: float = 4.0
    lambda_div: float = 0.10
    lambda_smooth: float = 0.01

    @property
    def weights_length(self) -> int:
        n = 0
        c_prev = self.context_channels * 2
        for _ in range(self.n_layers):
            n += self.hidden_dim * c_prev * 9 + self.hidden_dim
            c_prev = self.hidden_dim
        n += 2 * self.hidden_dim * 9 + 2
        return n


def pack_weights(state_dict: dict) -> np.ndarray:
    """Flatten a torch LearnedFlowHead state_dict into the kernel layout.

    torch Conv2d weight shape is (out, in, ky, kx) row-major: identical order
    to the kernel's per-layer (W, b) sequence.
    """
    weights: list[np.ndarray] = []
    keys = sorted(k for k in state_dict if k.startswith("net.") and k.endswith(".weight"))
    for wk in keys:
        bk = wk.replace(".weight", ".bias")
        weights.append(np.asarray(state_dict[wk], dtype=np.float64).ravel())
        weights.append(np.asarray(state_dict[bk], dtype=np.float64).ravel())
    packed = np.concatenate(weights)
    return np.ascontiguousarray(packed, dtype=np.float64)


def unpack_weights(packed: np.ndarray, spec: RustFlowHeadSpec) -> dict:
    """Inverse of `pack_weights`: kernel layout -> torch state_dict keys."""
    flat = np.asarray(packed, dtype=np.float64)
    out: dict = {}
    pos = 0
    c_prev = spec.context_channels * 2
    for layer in range(spec.n_layers):
        n_w = spec.hidden_dim * c_prev * 9
        out[f"net.{2 * layer}.weight"] = flat[pos:pos + n_w].reshape(spec.hidden_dim, c_prev, 3, 3)
        pos += n_w
        out[f"net.{2 * layer}.bias"] = flat[pos:pos + spec.hidden_dim]
        pos += spec.hidden_dim
        c_prev = spec.hidden_dim
        # the GELU modules sit at odd indices in the Sequential and carry no params
    n_w = 2 * spec.hidden_dim * 9
    out[f"net.{2 * spec.n_layers}.weight"] = flat[pos:pos + n_w].reshape(2, spec.hidden_dim, 3, 3)
    pos += n_w
    out[f"net.{2 * spec.n_layers}.bias"] = flat[pos:pos + 2]
    return out


def rust_train_flow_head(
    data: np.ndarray,          # (T, C_in, H, W) head-channel planes, float64
    frame_triples: np.ndarray, # (N, 3) int64/uint64 transitions
    spec: RustFlowHeadSpec,
    weights: np.ndarray,       # initial kernel-layout weights
    steps: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    dt: float = 1.0,
    deadline_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train in the kernel; returns (losses, trained_weights)."""
    if not _RUST_AVAILABLE:
        raise RuntimeError(
            "apedp_kernel is unavailable; build it with maturin develop --release"
        )
    _t, c_in, h, w = data.shape
    if c_in != spec.context_channels:
        raise ValueError(f"data has {c_in} head channels, spec expects {spec.context_channels}")
    flat = np.ascontiguousarray(data, dtype=np.float64).ravel()
    triples = np.ascontiguousarray(frame_triples, dtype=np.uint64).ravel()
    w0 = np.ascontiguousarray(weights, dtype=np.float64)
    deadline_ms = None if deadline_s is None else int(deadline_s * 1000.0)
    losses, w_out = apedp_kernel.flow_head_train(
        flat, triples, spec.context_channels, spec.hidden_dim, spec.n_layers,
        spec.flow_scale, h, w, w0, int(steps), float(lr), float(weight_decay),
        float(spec.lambda_div), float(spec.lambda_smooth), float(dt), deadline_ms,
    )
    return np.asarray(losses, dtype=np.float64), np.asarray(w_out, dtype=np.float64)


class RustFlowRollout:
    """Kernel-backed DC-STT rollout driven by head-predicted flow.

    ONE kernel crossing per step (transport step incl. Leray projection and
    optional viscosity); head history lives on the Python side (two small
    planes), which is the architecture that removes the per-kernel-call hop.
    """

    def __init__(self, spec: RustFlowHeadSpec, weights: np.ndarray,
                 dt: float = 1.0, enforce_hh: bool = True, hh_iters: int = 30,
                 nu: float = 0.0, use_dissipation: bool = False,
                 dx: float = 1.0, dy: float = 1.0) -> None:
        if not _RUST_AVAILABLE:
            raise RuntimeError(
                "apedp_kernel is unavailable; build it with maturin develop --release"
            )
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if w.size != spec.weights_length:
            raise ValueError(f"weights size {w.size} != spec length {spec.weights_length}")
        self.spec = spec
        self.weights = w
        self.dt = float(dt)
        self.enforce_hh = bool(enforce_hh)
        self.hh_iters = int(hh_iters)
        self.nu = float(nu)
        self.use_dissipation = bool(use_dissipation)
        self.dx = float(dx)
        self.dy = float(dy)

    def step(self, state: np.ndarray, head_prev: np.ndarray, head_curr: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One crossing: transport the state with head-predicted (optionally
        Leray-projected) flow. Torch-free by contract: only numpy buffers
        cross the boundary."""
        state_f = np.ascontiguousarray(state, dtype=np.float64)
        _c_in, h, w = head_prev.shape
        hp = np.ascontiguousarray(head_prev, dtype=np.float64).ravel()
        hc = np.ascontiguousarray(head_curr, dtype=np.float64).ravel()
        z_next, vx, vy, _rem = apedp_kernel.flow_head_transport_step(
            state_f.ravel(), hp, hc, self.spec.context_channels, self.spec.hidden_dim,
            self.spec.n_layers, self.spec.flow_scale, h, w, self.weights, self.dt,
            self.enforce_hh, self.hh_iters, self.nu, self.use_dissipation,
            self.dx, self.dy,
        )
        z_next = np.asarray(z_next, dtype=np.float64).reshape(state_f.shape)
        vx = np.asarray(vx, dtype=np.float64).reshape(h, w)
        vy = np.asarray(vy, dtype=np.float64).reshape(h, w)
        return z_next, vx, vy

    def rollout(self, state0: np.ndarray, context: np.ndarray, horizon: int
                ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Run `horizon` steps; returns (final_state, trajectory incl. context)."""
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        ctx = np.asarray(context, dtype=np.float64)
        c_full = int(ctx.shape[1]) if ctx.ndim == 4 else int(ctx[0].shape[0])
        hc = self._head_channels(c_full)
        head_prev = ctx[0][hc]
        head_curr = ctx[1][hc]
        state = ctx[1].copy()
        frames = [ctx[0].copy(), ctx[1].copy()]
        for _ in range(horizon):
            z_next, vx, vy = self.step(state, head_prev, head_curr)
            head_prev, head_curr = head_curr, z_next[hc]
            state = z_next
            frames.append(state.copy())
        self._last_flows = (vx, vy)  # exposed for diagnostics; not part of contract
        return state, frames

    @staticmethod
    def _head_channels(c_in: int) -> list[int]:
        # default head-channel slice used by the torch trainer: 1:3 when C>=3, else all
        return [1, 2] if c_in >= 3 else list(range(c_in))
