"""The Jet2 step's curvature has the right sign (fixed 26 November 2026; it was -dv/dt before)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "jetflow-video"))

from jetflow_video import stepper


@pytest.mark.parametrize("central", [False, True])
def test_one_step_on_a_curved_flow_is_second_order(central: bool) -> None:
    """dz/dt = z from t = 1 to t = 0: exact z / e; Euler 0; a second-order Taylor step 0.5 z."""
    s = stepper.JetFlowStepper(stepper.JetFlowConfig(order=2, use_central_diff=central))
    z = torch.randn(3, 4, dtype=torch.float64)
    out, _, a = s.step(lambda z_, t_: z_, z, 1.0, -1.0)
    assert torch.allclose(a, z, atol=1e-6)                 # dv/dt = (dv/dz) v = z
    assert torch.allclose(out, 0.5 * z, atol=1e-6)
    assert float((out - z / math.e).abs().max()) < float((0 * z - z / math.e).abs().max())


def test_a_flow_that_depends_on_time_alone_is_integrated_exactly() -> None:
    """dz/dt = t from t = 1 to 0: z(0) = z(1) - 1/2, which one second-order step gets right."""
    s = stepper.JetFlowStepper(stepper.JetFlowConfig(order=2))
    z = torch.zeros(5, dtype=torch.float64)
    out, _, _ = s.step(lambda z_, t_: torch.full_like(z_, t_), z, 1.0, -1.0)
    assert torch.allclose(out, torch.full_like(z, -0.5), atol=1e-6)


def test_forward_in_time_too() -> None:
    s = stepper.JetFlowStepper(stepper.JetFlowConfig(order=2))
    z = torch.zeros(5, dtype=torch.float64)
    out, _, _ = s.step(lambda z_, t_: torch.full_like(z_, t_), z, 0.0, 1.0)
    assert torch.allclose(out, torch.full_like(z, 0.5), atol=1e-6)


def test_more_steps_converge_to_the_exact_answer() -> None:
    s = stepper.JetFlowStepper(stepper.JetFlowConfig(order=2))
    z0 = torch.ones(2, dtype=torch.float64)
    errors = []
    for n in (1, 2, 4, 8):
        z, ts = z0.clone(), torch.linspace(1.0, 0.0, n + 1).tolist()
        for k in range(n):
            z, _, _ = s.step(lambda z_, t_: z_, z, ts[k], ts[k + 1] - ts[k])
        errors.append(float((z - z0 / math.e).abs().max()))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] / errors[-2] < 0.3                  # about a quarter: second order
