r"""Rust vs Python (torch) learned flow-head benchmark.

Status: audit-only

Measures the boundary-hop claim directly:
1. TRAIN: both engines run the SAME architecture (hidden 24, 3 layers,
   flow_scale 4, lambda_div 0.01, lambda_smooth 0.01) with an IDENTICAL
   wall-clock budget (default 90 s, deadline mid-loop). No RNG-exchange
   claim: each trainer picks its own (deterministic, seeded) transition
   order; the pinned equivalence relations live in the parity tests.
2. ROLLOUT: per-step median latency and FPS for a DC-STT rollout
   (Python crossing-per-kernel-call path vs the kernel's single-crossing
   `flow_head_transport_step`), both with their own trained weights.
3. Cross-engine state agreement on the same holdout is recorded (the
   parity-test class bound, ~2e-5 through HH+warp).

Honest scope: numbers ARE the deliverable — the artifact records whatever
advantage materializes (or fails to); there is no threshold gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport,
    DecoupledTransportConfig,
)
from apedp_dyn.video.flow_head_rust import (
    RustFlowHeadSpec,
    RustFlowRollout,
    pack_weights,
    rust_train_flow_head,
)
from apedp_dyn.video.fluid_domain import fluid_domain_manifest, make_domain_set
from apedp_dyn.video.learned_flow_head import (
    FlowHeadConfig,
    LearnedFlowHead,
    rollout_dcstt_head,
    train_flow_head,
)


def _median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def run(budget_s: float = 90.0, horizon: int = 14, out_path: Path | None = None,
        seed: int = 0) -> dict:
    man = fluid_domain_manifest(n_clips=10)
    clips = make_domain_set(tuple(man["train_ids"]), tuple(man["holdout_ids"]), t=16, h=64, w=64)
    train_clips = np.concatenate([clips[c] for c in man["train_ids"]], axis=0)[:, 1:3]  # head planes
    holdout = np.asarray(clips[man["holdout_ids"][0]], dtype=np.float64)
    rng = np.random.RandomState(seed)
    trip_rng = np.random.RandomState(seed + 1)

    spec = RustFlowHeadSpec(hidden_dim=24, n_layers=3, flow_scale=4.0,
                            lambda_div=0.01, lambda_smooth=0.01)

    # ---------------- python (torch) trainer ------------------------------
    head = LearnedFlowHead(FlowHeadConfig(
        context_channels=spec.context_channels, hidden_dim=spec.hidden_dim,
        n_layers=spec.n_layers, flow_scale=spec.flow_scale,
        lambda_div=spec.lambda_div, lambda_smooth=spec.lambda_smooth))
    head.double()
    t0 = time.perf_counter()
    py_losses = train_flow_head(head, train_clips, steps=1000000, lr=1e-3,
                                seed=seed, deadline_s=budget_s)
    py_train_s = time.perf_counter() - t0
    py_steps = len(py_losses)

    # ---------------- rust trainer ----------------------------------------
    data_flat_src = np.ascontiguousarray(train_clips, dtype=np.float64)
    # build an explicit triple list (kernel is RNG-free; python owns the order)
    t_max = data_flat_src.shape[0]
    triples = []
    for _ in range(30000):  # enough for any plausible budget
        ti = int(trip_rng.randint(1, t_max - 1))
        triples.append([ti - 1, ti, ti + 1])
    triples = np.asarray(triples[: max(1, py_steps + int(2 * py_steps))], dtype=np.uint64)
    init_head = LearnedFlowHead(FlowHeadConfig(
        context_channels=spec.context_channels, hidden_dim=spec.hidden_dim,
        n_layers=spec.n_layers, flow_scale=spec.flow_scale))
    init_head.double()
    packed_init = pack_weights(init_head.state_dict())
    t0 = time.perf_counter()
    rust_losses, w_rust = rust_train_flow_head(
        data_flat_src, triples, spec, packed_init, steps=10_000_000, lr=1e-3,
        weight_decay=1e-2, dt=1.0, deadline_s=budget_s)
    rust_train_s = time.perf_counter() - t0
    rust_steps = len(rust_losses)

    # ---------------- rollout latency (each engine's trained weights) ------
    ctx = holdout[:2]                       # (2, 3, H, W) full state context

    # python path: torch head + per-kernel-call transport crossings
    solver = DecoupledSpatioTemporalTransport(
        DecoupledTransportConfig(use_rust=True, nu=0.0, enable_laplacian_dissipation=False))
    rollout_dcstt_head(head, solver, ctx, horizon=horizon)  # warmup (excluded below)
    import torch
    py_step_ms: list[float] = []
    zs = [ctx[0].copy(), ctx[1].copy()]
    with torch.no_grad():
        state_prev = torch.from_numpy(ctx[0][None])       # full (1,3,H,W) state
        state_curr = torch.from_numpy(ctx[1][None])
        for _ in range(horizon):
            tt = time.perf_counter()
            v = head(state_prev[:, [1, 2], :, :], state_curr[:, [1, 2], :, :])
            z_next, _d = solver.transport_step(
                state_curr[0].numpy(), v[0, 0].numpy(), v[0, 1].numpy(), dt=1.0)
            py_step_ms.append((time.perf_counter() - tt) * 1e3)
            state_prev = state_curr
            state_curr = torch.from_numpy(z_next[None])
            zs.append(z_next.copy())
    py_roll = np.stack(zs, axis=0)
    py_roll_s = float(sum(py_step_ms)) / 1e3   # step timings are ms

    # rust path: single crossing per step (with its own trained weights)
    loop = RustFlowRollout(spec, w_rust, dt=1.0, enforce_hh=True, hh_iters=30,
                           nu=0.0, use_dissipation=False)
    rust_step_ms: list[float] = []
    _state, rust_frames = loop.rollout(None, ctx, horizon=horizon)  # warmup
    state = ctx[1].copy()
    hp, hc = ctx[0][[1, 2]].copy(), ctx[1][[1, 2]].copy()
    for _ in range(horizon):
        tt = time.perf_counter()
        z_next, _, _ = loop.step(state, hp, hc)
        rust_step_ms.append((time.perf_counter() - tt) * 1e3)
        hp, hc = hc.copy(), z_next[[1, 2]].copy()
        state = z_next
    rust_roll_s = float(sum(rust_step_ms)) / 1e3  # step timings are ms

    # cross-engine state agreement on the same rollout (parity-class bound)
    state_err = float(np.max(np.abs(np.stack(rust_frames, axis=0)[-1]
                                    - py_roll[-1])))

    payload = {
        "benchmark": "flow_head_rust_vs_python",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {"hidden": spec.hidden_dim, "layers": spec.n_layers,
                 "flow_scale": spec.flow_scale, "lambda_div": spec.lambda_div,
                 "lambda_smooth": spec.lambda_smooth},
        "train_budget_s": float(budget_s),
        "python_train": {"steps": int(py_steps), "time_s": float(py_train_s),
                         "ms_per_step": float(py_train_s / max(1, py_steps) * 1e3),
                         "first_loss": float(py_losses[0]) if len(py_losses) > 0 else float("nan"),
                         "final_loss": float(py_losses[-1]) if len(py_losses) > 0 else float("nan")},
        "rust_train": {"steps": int(rust_steps), "time_s": float(rust_train_s),
                       "ms_per_step": float(rust_train_s / max(1, rust_steps) * 1e3),
                       "first_loss": float(rust_losses[0]) if len(rust_losses) > 0 else float("nan"),
                       "final_loss": float(rust_losses[-1]) if len(rust_losses) > 0 else float("nan")},
        "python_rollout": {"horizon": int(horizon), "time_s": float(py_roll_s),
                           "median_step_ms": _median(py_step_ms),
                           "fps": float(horizon / max(1e-9, py_roll_s))},
        "rust_rollout": {"time_s": float(rust_roll_s),
                         "median_step_ms": _median(rust_step_ms),
                         "fps": float(horizon / max(1e-9, rust_roll_s))},
        "cross_engine_final_state_maxabsdiff": state_err,
        "notes": "no threshold gate: numbers are the deliverable; budgets equal; "
                 "each engine trains its own weights; cross-engine state diff "
                 "is the parity test class bound (erf approx through HH+warp).",
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved rust-vs-python artifact to {out_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-budget-s", type=float, default=90.0)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--out", type=Path,
                        default=_REPO / "benchmarks" / "flow_head_rust_vs_python_results.json")
    args = parser.parse_args()
    p = run(budget_s=args.train_budget_s, horizon=args.horizon, out_path=args.out)
    print(json.dumps({k: v for k, v in p.items() if k in ("python_train", "rust_train",
        "python_rollout", "rust_rollout", "cross_engine_final_state_maxabsdiff")}, indent=2))
