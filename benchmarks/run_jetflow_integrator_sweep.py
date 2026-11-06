"""Numerical JetFlow step-count sweep on analytic ODEs (not a video-quality benchmark).

Compares Jet2 against Euler at an equal velocity-evaluation budget and against a
fine-step RK4 reference. The analytic regimes expose both smooth-flow gains and
stiff-flow failure modes without relying on learned-model training.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PACKAGE = _REPO / "packages" / "jetflow-video"
for _path in (str(_REPO), str(_PACKAGE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
from jetflow_video.stepper import JetFlowConfig, JetFlowStepper

Velocity = Callable[[torch.Tensor, float], torch.Tensor]


def _velocity(regime: str) -> Velocity:
    if regime == "linear_time_varying":
        return lambda z, t: -0.8 * z + 0.2 * math.sin(math.pi * t)
    if regime == "smooth_nonlinear":
        return lambda z, t: -0.6 * z + 0.25 * torch.sin(z) * math.cos(2.0 * math.pi * t)
    if regime == "stiff_decay":
        return lambda z, _t: -8.0 * z
    raise ValueError(f"unknown velocity regime: {regime}")


def _rk4_reference(velocity: Velocity, initial: torch.Tensor, steps: int) -> torch.Tensor:
    """Integrate from t=1 to t=0 with classical fixed-step RK4."""
    z = initial.clone()
    dt = -1.0 / steps
    for index in range(steps):
        t = 1.0 + index * dt
        k1 = velocity(z, t)
        k2 = velocity(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = velocity(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = velocity(z + dt * k3, t + dt)
        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return z


def run_sweep(
    *,
    seeds: tuple[int, ...] = (7, 17, 27),
    step_counts: tuple[int, ...] = (1, 2, 4, 8),
    reference_steps: int = 4096,
    eps_fd: float = 1e-3,
) -> dict[str, object]:
    """Run reproducible analytic-flow comparisons and return JSON-safe results."""
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must be a non-empty tuple of integers")
    if not step_counts or any(steps < 1 for steps in step_counts):
        raise ValueError("step_counts must be positive")
    if reference_steps < 1 or eps_fd <= 0.0:
        raise ValueError("reference_steps and eps_fd must be positive")

    rows: list[dict[str, object]] = []
    for regime in ("linear_time_varying", "smooth_nonlinear", "stiff_decay"):
        velocity = _velocity(regime)
        for seed in seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = torch.randn((64,), generator=generator, dtype=torch.float64)
            reference = _rk4_reference(velocity, initial, reference_steps)
            for steps in step_counts:
                for method, order, method_steps in (
                    ("jet2", 2, steps),
                    ("euler_equal_evaluations", 1, 2 * steps),
                ):
                    calls = 0

                    def counted_velocity(
                        z: torch.Tensor, t: float, velocity_fn: Velocity = velocity
                    ) -> torch.Tensor:
                        nonlocal calls
                        calls += 1
                        return velocity_fn(z, t)

                    stepper = JetFlowStepper(JetFlowConfig(
                        num_steps=method_steps,
                        order=order,
                        eps_fd=eps_fd,
                        t_start=1.0,
                        t_end=0.0,
                    ))
                    start = time.perf_counter()
                    result = stepper.generate(counted_velocity, initial).final_latent
                    elapsed_s = time.perf_counter() - start
                    rmse = float(torch.mean((result - reference) ** 2).sqrt().item())
                    relative_l2 = float(
                        torch.linalg.vector_norm(result - reference)
                        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
                    )
                    rows.append({
                        "regime": regime,
                        "seed": seed,
                        "requested_steps": steps,
                        "method": method,
                        "integration_steps": method_steps,
                        "velocity_evaluations": calls,
                        "rmse_vs_rk4": rmse,
                        "relative_l2_vs_rk4": relative_l2,
                        "runtime_s": elapsed_s,
                    })

    return {
        "schema_version": "1.0",
        "scope": (
            "Synthetic analytic ODE integrator comparison only; not evidence of "
            "learned video quality, real-data generalization, or universal solver superiority."
        ),
        "protocol": {
            "direction": "t=1 to t=0",
            "reference": "classical fixed-step RK4",
            "reference_steps": reference_steps,
            "jet2_finite_difference_epsilon": eps_fd,
            "equal_evaluation_baseline": "Euler uses twice as many integration steps as Jet2",
            "seeds": list(seeds),
            "requested_jet2_step_counts": list(step_counts),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-steps", type=int, default=4096)
    parser.add_argument("--eps-fd", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=_REPO / "benchmarks" / "jetflow_integrator_sweep.json")
    args = parser.parse_args()
    results = run_sweep(reference_steps=args.reference_steps, eps_fd=args.eps_fd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results['rows'])} measurements to {args.out}")


if __name__ == "__main__":
    main()
