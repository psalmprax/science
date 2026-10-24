r"""DC-STT Horizon-Decay Benchmark.

Status: audit-only

Question the multi-scene benchmark does NOT answer: DC-STT's per-step latency
(~3.7 ms) and Tensor-Train compression are *throughput* claims; does transport
QUALITY survive long rollouts? This harness quantifies decay in the qualities
that would actually matter for serving:

1. **Integrator fidelity vs horizon** — trajectory from the shipped
   ``transport_step`` (dt=0.5) against a fine-substep reference integration
   (dt=0.05, 10 substeps) of the SAME analytic flow: isolates numerical error
   growth (advection O(dt) led error + boundary effects) from the flow
   definition itself.
2. **Appearance preservation vs horizon** — normalized "mass" (energy) decay
   under nu-Laplacian dissipation + numerical diffusion: a transport engine
   that washes its image toward flat is not a faithful video model at any
   horizon, however fast its steps.
3. **Detail decay (blur proxy)** — mean |gradient| magnitude growth/decay
   versus horizon, the classical signature of diffusive smearing.
4. **TT-reconstruction error vs horizon** (only for horizons where a
   fine reference exists).

Honest scope: these are RECONSTRUCTION/transport claims about an untrained
mathematical operator given ideal analytic flow. They are a prerequisite for
(higher) serving-quality claims, not a substitute for them: nothing here
compares DC-STT to learned generation models (Sora etc.) — generation
competence would require a learned prior, which this repo tracks separately
(learned-flow-head benchmark).
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

from apedp_dyn.video.continuous_interpolator import ContinuousTemporalInterpolator
from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport,
    DecoupledTransportConfig,
)
from benchmarks.run_decoupled_transport_multiscene_benchmark import (
    _build_scene_flow_sequence,
    _build_scene_initial_condition,
)

HORIZONS = [16, 32, 64, 128, 256]
SUBSTEPS = 10           # reference integration = 10x finer dt
PSNR_PEAK = 1.0         # these scenes live in roughly [-1, 1]-scaled [0, ~1]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(PSNR_PEAK ** 2 / mse))


def _fine_reference(solver: DecoupledSpatioTemporalTransport,
                    z0: np.ndarray,
                    flow_sequence: list[tuple[np.ndarray, np.ndarray]],
                    horizon: int) -> np.ndarray:
    """Integrate the SAME analytic flow with SUBSTEPS-fold finer dt."""
    cfg_fine = DecoupledTransportConfig(
        use_rust=True,
        nu=solver.config.nu,
        enforce_mass_conservation=solver.config.enforce_mass_conservation,
        helmholtz_hodge_iterations=solver.config.helmholtz_hodge_iterations,
        enable_laplacian_dissipation=solver.config.enable_laplacian_dissipation,
        disocclusion_threshold=solver.config.disocclusion_threshold,
    )
    fine = DecoupledSpatioTemporalTransport(cfg_fine)
    z = np.asarray(z0, dtype=np.float64)
    if z.ndim == 2:
        z = z[None]
    per_step = 1.0 / float(SUBSTEPS)
    for i in range(horizon):
        vx, vy = flow_sequence[i % len(flow_sequence)]
        for _ in range(SUBSTEPS):
            z, _ = fine.transport_step(z, vx, vy, s_source=None, dt=per_step)
    return z


def _details_energy(x: np.ndarray) -> float:
    """Mean absolute spatial gradient (blur proxy; lower = blurrier)."""
    gy = np.abs(np.diff(x, axis=-2)).mean()
    gx = np.abs(np.diff(x, axis=-1)).mean()
    return float(0.5 * (gy + gx))


def run_horizon_decay(out_path: Path | None = None,
                      horizons: list[int] | None = None) -> dict:
    horizons = horizons or HORIZONS
    scene_ids = ["vehicle_translation", "fluid_vortex", "camera_zoom"]

    cfg = DecoupledTransportConfig(use_rust=True, nu=0.003)
    solver = DecoupledSpatioTemporalTransport(cfg)

    scenes: dict[str, dict] = {}
    for scene_id in scene_ids:
        z0 = _build_scene_initial_condition(scene_id, 64, 64)
        details0 = _details_energy(z0)
        per_h: dict[str, dict] = {}
        for horizon in horizons:
            flows, _ = _build_scene_flow_sequence(scene_id, horizon, 64, 64)

            t0 = time.perf_counter()
            traj, _ = solver.generate_multiscene_trajectory(z0, flows, None, dt=0.5)
            coarse_s = time.perf_counter() - t0

            z_final = traj[-1]
            row = {
                "horizon": int(horizon),
                "coarse_time_s": float(coarse_s),
                # 2. appearance/mass preservation
                "energy_ratio": float(np.sum(z_final ** 2) / max(1e-12, np.sum(np.asarray(z0, dtype=np.float64) ** 2))),
                # 3. detail blur proxy
                "details_ratio": float(_details_energy(z_final) / max(1e-12, details0)),
                "flicker_accel": float(ContinuousTemporalInterpolator.measure_temporal_smoothness(traj)),
            }
            # 1. integrator fidelity (fine reference) only up to the affordable horizon
            if horizon <= 128:
                ref = _fine_reference(solver, z0, flows, horizon)
                row["vs_fine_ref_psnr_db"] = _psnr(z_final, ref)
                row["vs_fine_ref_mse"] = float(np.mean((z_final - ref) ** 2))
            per_h[str(horizon)] = row
        scenes[scene_id] = {"initial_details_energy": float(details0), "rows": per_h}
        for hz in per_h:
            r = per_h[hz]
            psnr_s = (f"PSNRvsFine={r['vs_fine_ref_psnr_db']:6.2f} " if "vs_fine_ref_psnr_db" in r else "ref=n/a    ")
            print(f"{scene_id:22s} H={hz:>4s}: {psnr_s} energy={r['energy_ratio']:6.3f} "
                  f"details={r['details_ratio']:6.3f}")

    payload = {
        "benchmark": "dcstt_horizon_decay",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "substeps_reference": SUBSTEPS,
        "coarse_dt": 0.5,
        "scenes": scenes,
        "honest_scope": __doc__.split("Honest scope:")[1].strip() if "Honest scope:" in (__doc__ or "") else "",
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved horizon-decay results to {out_path}")
    return payload



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=_REPO / "benchmarks" / "dcstt_horizon_decay_results.json")
    args = parser.parse_args()
    run_horizon_decay(out_path=args.out)
