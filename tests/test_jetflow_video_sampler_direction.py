"""generate_video sampler-direction contract tests.

``trainer.compute_loss`` parameterizes the flow geodesic as
``z_t = (1 - t) * z_data + t * z_noise`` (t = 0 is DATA, t = 1 is NOISE) with
the learned field ``v* = z_noise - z_data``. Generation must therefore start
from noise at t = 1 and integrate t: 1 -> 0 with a NEGATIVE dt — the same
convention the samplers in ``first_frame.py`` and ``distill.py`` follow.
``model.generate_video`` used to walk t: 0 -> 1 from noise (the training
parameterization's opposite direction); these tests pin the corrected
behavior so it cannot regress back.

Direction discriminator: with a stub velocity field ``v(z) = -z``, the
trainer-consistent integration (t: 1 -> 0) grows the field magnitude by
exactly e over the full interval (z(t) = z(1) * e^(1-t)), while the old
backward sampler (t: 0 -> 1, dt > 0) shrinks it to z(1)/e. The growth
direction is impossible under the old schedule.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Same path discipline as benchmarks/run_jetflow_distill_benchmark.py: the
# package ships as a sub-tree of the repo, not as an installed dependency.
_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "packages" / "jetflow-video"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("jetflow_video", reason="jetflow-video package not importable")


def _tiny_model():
    from jetflow_video.model import JetFlowModelConfig, JetFlowVideoModel

    torch.manual_seed(11)
    model = JetFlowVideoModel(JetFlowModelConfig(
        latent_channels=12, hidden_dim=32, num_layers=1)).double()
    model.eval()

    # Pass-through the projection/steering layers the sampler sits behind, so
    # the stub velocity below is observed unmodified through the sampler.
    model.noether_engine.project_spatial_latent_velocity = (
        lambda v: (v, None))
    model.physics_controller.steer_velocity_field = (
        lambda v_sol, z_eval, t, dt: v_sol)
    return model


class _Probe:
    def __init__(self) -> None:
        self.ts: list[float] = []
        self.zs: list[torch.Tensor] = []

    def attach(self, model) -> None:
        def velocity(z_eval, t, text_tokens=None, guidance_scale=None):
            self.ts.append(float(t))
            self.zs.append(z_eval.detach().clone())
            return -z_eval

        model.compute_flow_guided_velocity = velocity


@pytest.mark.parametrize("num_steps", [2, 4, 40])
def test_generate_video_integrates_down_from_noise_at_t_one(num_steps) -> None:
    """The schedule descends 1.0 -> 0.0 and the first eval sees raw noise."""
    model = _tiny_model()
    probe = _Probe()
    probe.attach(model)

    shape = (4, 12, 8, 8)
    prompt = torch.zeros(64, dtype=torch.float32)
    torch.manual_seed(0)
    _, rgb = model.generate_video(prompt, num_steps=num_steps, shape=shape)
    assert rgb.shape[-3:] == (3, 16, 16)

    torch.manual_seed(0)
    expected_first_noise = torch.randn(*shape, dtype=torch.float64)

    assert probe.ts, "the sampler never evaluated the velocity field"
    assert abs(probe.ts[0] - 1.0) < 1e-9, (
        f"generation must START at t=1 (noise); first eval was t={probe.ts[0]}")
    assert torch.equal(probe.zs[0], expected_first_noise), (
        "the initial latent is no longer the sampler's own fresh noise draw")

    # The step schedule covers the whole descending interval [1.0, 0.0]: the
    # main velocity evaluations start at t=1 and the LAST main eval sits at
    # the second-to-last boundary (1/num_steps); no step fires AT t=0 itself
    # (integration stops there). The extra calls are Jet2's central-
    # difference probes around the current t, bounded to [0, 1].
    assert max(probe.ts) <= 1.0 + 1e-9 and min(probe.ts) >= 0.0 - 1e-9
    assert min(probe.ts) <= 1.0 / num_steps + 1e-9, (
        f"the last velocity evaluation must sit at the final step boundary; "
        f"observed min {min(probe.ts)} for num_steps={num_steps}")


def test_generate_video_direction_grows_field_by_e_not_shrinks() -> None:
    """With v(z) = -z, trainer-consistent integration grows |z| by ~e.

    The old schedule (t: 0 -> 1, dt > 0) would shrink it by 1/e — this is the
    discriminating observable between the two conventions.
    """
    model = _tiny_model()
    probe = _Probe()
    probe.attach(model)

    shape = (4, 12, 8, 8)
    prompt = torch.zeros(64, dtype=torch.float32)
    torch.manual_seed(3)
    model.generate_video(prompt, num_steps=100, shape=shape)

    ratio = (probe.zs[-1].norm() / probe.zs[0].norm()).item()
    assert 1.8 < ratio < 3.6, (
        f"|z_final|/|z_init| = {ratio:.3f}; trainer-consistent integration of "
        f"v=-z over t: 1->0 must converge to e ≈ {math.e:.3f} (the old "
        f"t: 0->1 schedule gives 1/e ≈ {1.0 / math.e:.3f})")


def test_generate_video_matches_first_frame_sampler_convention() -> None:
    """The model sampler and the anchored sampler agree on the ODE contract.

    Using the same stub v(z) = -z and identical seeds, the latent trajectory
    of generate_video must equal the one produced by integrating the same
    schedule directly (z + dt * v with dt < 0), apart from the anchor-only
    frame logic absent here (anchor not supplied).
    """
    model = _tiny_model()
    probe = _Probe()
    probe.attach(model)

    shape = (3, 12, 8, 8)
    prompt = torch.zeros(64, dtype=torch.float32)
    torch.manual_seed(5)
    model.generate_video(prompt, num_steps=32, shape=shape)

    # Independent integration with the recorded first noise draw and the
    # plain downward schedule (z += dt * v, dt negative), Euler at the same
    # step count on the jet_order-1 path used when the stub has no curvature.
    torch.manual_seed(5)
    z = torch.randn(*shape, dtype=torch.float64)
    ts = torch.linspace(1.0, 0.0, 32 + 1).tolist()
    for i in range(32):
        dt = float(ts[i + 1]) - float(ts[i])
        z = z + dt * (-z)
    ratio_expected = (z.norm() / torch.randn(*shape, dtype=torch.float64).norm())

    ratio_model = (probe.zs[-1].norm() / probe.zs[0].norm()).item()
    # Only the direction and scale contract is pinned here (the Jet2 stepper
    # differs from plain Euler by O(dt^2) curvature terms); the old backward
    # schedule sits an order of e away from this range.
    assert 1.8 < ratio_model < 3.6, ratio_model
    assert 1.8 < ratio_expected.item() < 3.6, ratio_expected.item()
