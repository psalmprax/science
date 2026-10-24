r"""Learned-Flow DC-STT Benchmark: transport REAL domain content with an
inferred velocity field instead of a hand-coded analytic scene flow.

Status: audit-only

Pipeline under test (first learned component in the DC-STT path):
    1. A small CNN (`apedp_dyn/video/learned_flow_head.LearnedFlowHead`)
       is trained ENTIRELY ON CPU, self-supervised on manifest train clips
       of the fluid domain: predict v(x) from two frames such that
       advecting the current frame with -v reproduces the next frame.
    2. At evaluation, roll DC-STT forward from 2 context frames of a
       HELD-OUT manifest clip using only head-predicted flow (nu=0.0 pure
       advection, matching the training dynamics).
    3. Score per horizon against the clip's true future frames, against:
         - persistence (copy of the last context frame),
         - integer-shift search baseline ("best constant translation").

Honest scope (stated, not hidden):
- "Real content" = domain content whose flow is NOT provided (must be
  inferred from pixels; the head must generalize across vortex configs).
  The clip generator itself remains synthetic; photorealism is unclaimed.
- Deadline-disciplined training (matched wall-clock class as the
  jet-vs-diffusion benchmark), deterministic seeds, holdout ids come from
  the same domain manifest split used by the matched-compute benchmark.
- AC-training floor: dye renormalization in the clip generator makes the
  self-supervised target approximate; the trained loss floor is reported.
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
from apedp_dyn.video.fluid_domain import (
    fluid_domain_manifest,
    make_domain_set,
    make_fluid_clip,
)
from apedp_dyn.video.learned_flow_head import (
    FlowHeadConfig,
    LearnedFlowHead,
    rollout_dcstt_head,
    train_flow_head,
)


def _best_integer_shift(prev: np.ndarray, curr: np.ndarray,
                        max_shift: int = 8) -> tuple[float, float]:
    """Classic translation estimate: integer shift maximizing similarity.

    Exhaustive search over +-max_shift px on the mean dye plane (cheap CPU
    baseline standing in for optical flow). Returns (vx, vy) in units of
    px/frame matching DC-STT's (dx, dy) convention.
    """
    h, w = prev.shape[-2], prev.shape[-1]
    ref = prev.mean(axis=0) if prev.ndim == 3 else prev
    cur = curr.mean(axis=0) if curr.ndim == 3 else curr
    best, best_score = (0.0, 0.0), -np.inf
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            ys, xs = np.clip(np.arange(h) + dy, 0, h - 1), np.clip(np.arange(w) + dx, 0, w - 1)
            score = float(-np.mean((cur[np.ix_(np.arange(h), np.arange(w))]
                                    - ref[np.ix_(ys, xs)]) ** 2))
            if score > best_score:
                best_score = score
                best = (float(dx), float(dy))
    return best


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def run_learned_flow_benchmark(train_budget_s: float = 90.0,
                               horizons: tuple[int, ...] = (4, 8, 14),
                               n_holdout: int = 2,
                               out_path: Path | None = None,
                               seed: int = 0,
                               flow_supervised: bool = False,
                               periodic: bool = True,
                               save_head: Path | None = None,
                               gain_correction: bool = False,
                               rollout_steps: int = 1,
                               ch0_policy: str = "persist") -> dict:
    # n_clips=10 mirrors run_jetflow_vs_diffusion_matched.py exactly, so the
    # shared holdout ids (fluid_clip_0003/0005) stay burnable and are never
    # used as training data here (AGENTS.md 1.2).
    man = fluid_domain_manifest(n_clips=10)
    t, h, w = int(man["t"]) if "t" in man else 16, 64, 64
    print("=" * 80)
    print("LEARNED-FLOW DC-STT: transport held-out clips with INFERRED flow (CPU)"
          + ("  [Stage-3 FLOW-SUPERVISED: analytic teacher on TRAIN clips only]"
             if flow_supervised else ""))
    print(f"config: periodic_warp={periodic}, gain_correction={gain_correction}, "
          f"rollout_steps={rollout_steps}, ch0_policy={ch0_policy}")
    print("=" * 80)

    clips = make_domain_set(tuple(man["train_ids"]), tuple(man["holdout_ids"]),
                            t=t, h=h, w=w)
    train_ids = list(man["train_ids"])
    train_clips = np.concatenate([clips[cid] for cid in train_ids], axis=0)

    flow_targets = None
    if flow_supervised:
        # Stage-3 teacher flows: the generator's own analytic field,
        # TRAIN-CLIP IDS ONLY (the holdout ids stay exclusively gate-side;
        # they never enter training nor the pseudo-flow set). Stacked in
        # EXACTLY the same clip order train_flow_head concatenates, in
        # head-units (vx, vy) = (v_gen, u_gen) — the swap pinned in tests.
        per_id = []
        for cid in train_ids:
            gseed = int(cid.replace("fluid_clip_", ""), 10)
            _, u_gen, v_gen = make_fluid_clip(gseed, t=t, h=h, w=w,
                                              return_flow=True)
            per_id.append(np.stack([v_gen, u_gen], axis=1))  # (T, 2, H, W)
        flow_targets = np.concatenate(per_id, axis=0)
        print(f"flow supervision: {flow_targets.shape} analytic teacher "
              "fields from the TRAIN ids only")

    # ---- CPU training under the repo's wall-clock deadline discipline ------
    head = LearnedFlowHead(FlowHeadConfig(predict_gain=gain_correction))
    head.double()
    t0 = time.perf_counter()
    losses = train_flow_head(head, train_clips, steps=100000, lr=3e-4,
                             seed=seed, progress_every=2000,
                             deadline_s=train_budget_s,
                             flow_targets=flow_targets,
                             frames_per_clip=t,
                             periodic=periodic,
                             rollout_steps=rollout_steps)
    train_s = time.perf_counter() - t0
    train_steps = len(losses)
    print(f"trained {train_steps} steps in {train_s:.1f}s; "
          f"loss {losses[0]:.5f} -> {losses[-1]:.5f} "
          + ("(per-step objective over a k-step model-own rollout; "
             if rollout_steps > 1 else "(advection-consistency; ")
          + ("gain head models the renormalization affine)" if gain_correction
             else "the generator's dye renorm is a factor out of head scope here)"))

    # nu=0.0 pure-advection transport for rollout (see module docstring);
    # the fluid generator's advection is periodic, so BOTH the training loss
    # warp and the rollout transport must be periodic (toroidal).
    solver = DecoupledSpatioTemporalTransport(
        DecoupledTransportConfig(use_rust=True, nu=0.0,
                                 enable_laplacian_dissipation=False,
                                 periodic_warp=periodic)
    )

    per_clip: dict[str, dict] = {}
    agg: dict[str, dict[str, list[float]]] = {k: {} for k in ("learned_flow", "persistence", "shift_search")}
    for cid in list(man["holdout_ids"])[:n_holdout]:
        clip = np.asarray(clips[cid], dtype=np.float64)
        context = clip[:2]
        truth = clip[2:]  # true future frames, (T-2, C, H, W)
        true_h = min(truth.shape[0], max(horizons))

        # learned-flow DC-STT rollout
        t1 = time.perf_counter()
        rollout = rollout_dcstt_head(head, solver, context, horizon=true_h,
                                     ch0_policy=ch0_policy)
        rollout_s = time.perf_counter() - t1

        # persistence + integer-shift-search baselines
        pers = np.repeat(context[1:2], true_h, axis=0)
        vx0, vy0 = _best_integer_shift(context[0], context[1])
        ones = np.ones((h, w), dtype=np.float64)
        shift_traj, _ = solver.generate_multiscene_trajectory(
            context[1], [(ones * vx0, ones * vy0)] * true_h, dt=1.0)
        # shift traj: (true_h+1, C, H, W); candidate frames 1:
        shift_frames = shift_traj[1:]

        rows = {}
        for hz in horizons:
            hz_eff = min(hz, true_h)
            row = {
                "horizon": int(hz_eff),
                "learned_flow_psnr_db": float(np.mean([_psnr(rollout[2 + i], truth[i]) for i in range(hz_eff)])),
                "persistence_psnr_db": float(np.mean([_psnr(pers[i], truth[i]) for i in range(hz_eff)])),
                "shift_search_psnr_db": float(np.mean([_psnr(shift_frames[i], truth[i]) for i in range(hz_eff)])),
            }
            rows[str(hz)] = row
            for k in ("learned_flow", "persistence", "shift_search"):
                agg[k].setdefault(str(hz), []).append(row[f"{k}_psnr_db"])
        per_clip[cid] = {
            "estimated_shift_vx_vy": [vx0, vy0],
            "rollout_time_s": float(rollout_s),
            "rollout_fps": float(true_h / max(1e-6, rollout_s)),
            "rows": rows,
        }
        print(f"{cid}: shift=({vx0:+.0f},{vy0:+.0f}) px/frame, rollout {rollout_s:.2f}s "
              f"({true_h / max(1e-6, rollout_s):.1f} FPS incl. python-transfer overhead)")
        for hz in horizons:
            r = rows[str(hz)]
            print(f"  H={r['horizon']:>3d}: learned={r['learned_flow_psnr_db']:6.2f}dB  "
                  f"persistence={r['persistence_psnr_db']:6.2f}dB  "
                  f"shift={r['shift_search_psnr_db']:6.2f}dB")

    aggregate = {k: {hz: float(np.mean(v)) for hz, v in d.items() if v}
                 for k, d in agg.items()}
    print("\nAGGREGATE (mean over holdout clips):")
    for k, d in aggregate.items():
        print(f"  {k:12s}", {hz: f"{v:.2f}dB" for hz, v in sorted(d.items(), key=lambda x: int(x[0]))})

    payload = {
        "benchmark": "learned_flow_dcstt" + ("_flow_supervised" if flow_supervised else ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "flow_supervised": bool(flow_supervised),
        "periodic_warp": bool(periodic),
        "gain_correction": bool(gain_correction),
        "rollout_steps": int(rollout_steps),
        "ch0_policy": str(ch0_policy),
        "train_budget_s": float(train_budget_s),
        "train_steps": int(train_steps),
        "train_s": float(train_s),
        "final_train_loss": float(losses[-1]),
        "first_train_loss": float(losses[0]),
        "params": int(sum(p.numel() for p in head.parameters())),
        "holdout_ids": list(list(man["holdout_ids"])[:n_holdout]),
        "per_clip": per_clip,
        "aggregate_psnr_db": aggregate,
        "honest_scope": __doc__.split("Honest scope")[1].strip() if "Honest scope" in (__doc__ or "") else "",
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved learned-flow results to {out_path}")
    if save_head is not None:
        # Persist the trained weights (packable into the Rust kernel via
        # flow_head_rust.pack_weights for the distillation-parity check).
        save_head.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(head.state_dict(), str(save_head))
        print(f"Saved trained head weights to {save_head}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-budget-s", type=float, default=90.0)
    parser.add_argument("--flow-supervised", action="store_true", default=False,
                        help="Stage-3: supervise the head with the generator's "
                             "analytic flow on TRAIN clips only")
    parser.add_argument("--save-head", type=Path, default=None,
                        help="optional .pt path: export the trained head state_dict")
    parser.add_argument("--clamped-boundary", action="store_true", default=False,
                        help="ablation: use clamped (border) warp instead of the "
                             "domain-correct periodic warp, in BOTH training and rollout")
    parser.add_argument("--gain-correction", action="store_true", default=False,
                        help="head also predicts a per-channel affine (s,b) applied "
                             "after transport (the generator's dye renorm gain)")
    parser.add_argument("--rollout-steps", type=int, default=1,
                        help="exposure-bias training: unroll k model-own steps per "
                             "training window (BPTT); 1 = single-transition (legacy)")
    parser.add_argument("--ch0-policy", choices=("transport", "persist"),
                        default="persist",
                        help="vorticity channel policy: 'transport' (legacy advect) "
                             "or 'persist' (freeze; domain-correct for ch0=|curl|)")
    parser.add_argument("--out", type=Path,
                        default=_REPO / "benchmarks" / "learned_flow_dcstt_results.json")
    args = parser.parse_args()
    run_learned_flow_benchmark(train_budget_s=args.train_budget_s, out_path=args.out,
                               flow_supervised=args.flow_supervised,
                               periodic=not args.clamped_boundary,
                               save_head=args.save_head,
                               gain_correction=args.gain_correction,
                               rollout_steps=args.rollout_steps,
                               ch0_policy=args.ch0_policy)
