"""Unified JetFlow-Video Model: Continuous Spatio-Temporal PDE Video Generation.

Status: experimental

Integrates:
1. VideoVAEBridge: RGB pixel space (T, 3, H, W) <-> Latents (T, C, H', W').
2. TensorTrainVideo: 4D low-rank Matrix Product State video compression (75x-186x VRAM cut).
3. JetFlowStepper: 2nd-order convective acceleration flow stepper (2-4 step generation).
4. NoetherConservationEngine: Divergence-free solenoidal projection preventing object melting.
5. PhysicalSteeringController: Convection, viscosity dissipation, camera SE(3), and time dilation.
6. SemanticVideoConditioner: Fourier time + text cross-attention + AdaLN-Zero + Flow Guidance.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from apedp_dyn.video.jet_flow_stepper import (
    JetFlowConfig,
    JetFlowStepper,
)
from apedp_dyn.video.noether_engine import (
    NoetherConfig,
    NoetherConservationEngine,
)
from apedp_dyn.video.physics_knobs import (
    PhysicalSteeringConfig,
    PhysicalSteeringController,
)
from apedp_dyn.video.semantic_bridge import (
    SemanticConditioningConfig,
    SemanticVideoConditioner,
)
from apedp_dyn.video.tensor_train_video import (
    TensorTrainVideo,
    compress_video_latent,
)
from apedp_dyn.video.vae_bridge import (
    VideoVAEBridge,
)


@dataclass(frozen=True)
class JetFlowModelConfig:
    """Configuration for JetFlow-Video Continuous Generative PDE Model."""
    latent_channels: int = 12
    spatial_downsample: int = 2
    embedding_dim: int = 64
    num_heads: int = 4
    guidance_scale: float = 3.0
    bond_ranks: tuple[int, int, int] = (8, 8, 8)
    jet_order: int = 2
    lambda_acc: float = 0.5
    lambda_noether: float = 0.1
    use_rust_kernel: bool = True
    seed: int = 42


class JetFlowVideoModel:
    """End-to-End High-Dimensional Continuous PDE Video Generative Model."""

    def __init__(self, config: JetFlowModelConfig | None = None):
        self.config = config or JetFlowModelConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # 1. Video VAE Bridge
        self.vae_bridge = VideoVAEBridge(
            latent_channels=self.config.latent_channels,
            downsample_factor=self.config.spatial_downsample,
        )

        # 2. Semantic Conditioner (AdaLN-Zero + Multi-Head Cross-Attention)
        cond_config = SemanticConditioningConfig(
            latent_dim=self.config.latent_channels,
            text_dim=self.config.embedding_dim,
            guidance_scale=self.config.guidance_scale,
            num_heads=self.config.num_heads,
            fourier_frequencies=16,
        )
        self.conditioner = SemanticVideoConditioner(cond_config, seed=self.config.seed)

        # 3. High-Order Curvature Jet2 Stepper
        stepper_config = JetFlowConfig(
            order=self.config.jet_order,
            num_steps=2,
        )
        self.stepper = JetFlowStepper(stepper_config)

        # 4. Noether Conservation Engine
        self.noether_engine = NoetherConservationEngine(
            NoetherConfig(enforce_divergence_free=True, poisson_iterations=40)
        )

        # 5. Physical Steering Controller
        self.physics_controller = PhysicalSteeringController()

    def train_step(
        self,
        video_rgb: np.ndarray,
        prompt_embedding: np.ndarray,
        learning_rate: float = 1e-3,
    ) -> dict[str, float]:
        r"""Compute the 2nd-order Jet2 flow matching losses (DEPRECATED as a trainer).

        Certainty note: the NumPy conditioner in this module has frozen,
        random-initialized weights (w_time, w_text, w_q/k/v/out, w_adaln) and
        the only mutating parameter was a single bias vector updated by one
        scalar mean-gradient — this is not a trainable flow-matching model.
        The previous version pretended to train via that scalar update; it
        was removed. This method now computes and returns the Jet2 loss
        values on the frozen model (useful as a diagnostics/eval signal)
        and performs NO parameter update.

        Real training lives in ``packages/jetflow-video`` (``JetFlowTrainer``),
        which performs full-parameter flow matching with AdamW and exports
        trained weights for inference.

        Loss = L_flow(v_theta, v*) + lambda_acc * L_acc(a_theta, a*) + lambda_noether * L_div(\nabla \cdot v)
        """
        # Encode RGB video to latent
        z_data = self.vae_bridge.encode_video_to_latent(video_rgb)
        shape = z_data.shape  # (T, C, H', W')

        # Sample initial Gaussian noise z_1
        z_noise = self.rng.randn(*shape).astype(np.float64)

        # Sample flow matching time t in (0, 1)
        t = float(self.rng.uniform(0.05, 0.95))

        # Geodesic flow interpolation: z_t = (1 - t) * z_data + t * z_noise
        z_t = (1.0 - t) * z_data + t * z_noise
        target_v = z_noise - z_data

        # Prepare text tokens
        text_tokens = prompt_embedding.reshape(1, -1) if prompt_embedding.ndim == 1 else prompt_embedding

        # Predict conditional velocity
        pred_v = self.conditioner.compute_conditioned_velocity(
            z_latent=z_t,
            t=t,
            text_tokens=text_tokens,
        )

        # Velocity matching error
        flow_loss = float(np.mean((pred_v - target_v) ** 2))

        # Convective acceleration penalty
        dt_eps = 0.01
        pred_v_plus = self.conditioner.compute_conditioned_velocity(z_t, t + dt_eps, text_tokens)
        pred_v_minus = self.conditioner.compute_conditioned_velocity(z_t, max(0.0, t - dt_eps), text_tokens)
        temporal_acc = (pred_v_plus - pred_v_minus) / (2.0 * dt_eps)
        acc_loss = float(np.mean(temporal_acc ** 2))

        # Divergence penalty (Noether mass preservation)
        _, mean_div = self.noether_engine.project_spatial_latent_velocity(pred_v)
        div_loss = float(mean_div ** 2)

        # Total Jet2 loss
        total_loss = flow_loss + self.config.lambda_acc * acc_loss + self.config.lambda_noether * div_loss

        # Deprecation notice: no fake parameter mutation. All conditioner
        # weights are frozen on this path; the NumPy side is inference-only.
        warnings.warn(
            "JetFlowVideoModel.train_step is inference/diagnostics-only: the "
            "NumPy conditioner cannot be trained (frozen weights); real "
            "flow-matching training is in packages/jetflow-video JetFlowTrainer.",
            DeprecationWarning,
            stacklevel=2,
        )

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "acc_loss": acc_loss,
            "div_loss": div_loss,
        }

    def generate_video(
        self,
        prompt_embedding: np.ndarray,
        num_steps: int = 2,
        shape: tuple[int, int, int, int] = (16, 12, 16, 16),
        physical_knobs: PhysicalSteeringConfig | None = None,
        bond_ranks: tuple[int, int, int] | None = None,
    ) -> tuple[TensorTrainVideo, np.ndarray]:
        """Generate high-precision video from prompt embedding in 2 to 4 curvature steps.

        Returns:
            tt_video: Low-rank 4D Tensor-Train compressed representation.
            video_rgb: Decompressed RGB video frames (T, 3, H, W).
        """
        ranks = bond_ranks or self.config.bond_ranks
        controller = PhysicalSteeringController(physical_knobs) if physical_knobs else self.physics_controller

        # Initial random noise latent z_1 ~ N(0, I)
        z_curr = self.rng.randn(*shape).astype(np.float64)

        # Timestep schedule from 0.0 (prior noise) up to 1.0 (data latent)
        timesteps = np.linspace(0.0, 1.0, num_steps + 1)

        text_tokens = prompt_embedding.reshape(1, -1) if prompt_embedding.ndim == 1 else prompt_embedding

        # Jet2 Curvature Integration
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr  # Positive forward flow step

            def velocity_eval(z_eval: np.ndarray, t_eval: float) -> np.ndarray:
                v_model, _ = self.conditioner.compute_flow_guided_velocity(
                    z_latent=z_eval,
                    t=t_eval,
                    text_tokens=text_tokens,
                    guidance_scale=self.config.guidance_scale,
                )
                v_sol, _ = self.noether_engine.project_spatial_latent_velocity(v_model)
                return controller.steer_velocity_field(v_sol, z_eval, t=t_eval, dt=abs(dt))  # noqa: B023 - velocity_fn is consumed inside the same loop iteration it is defined

            z_next, _, _ = self.stepper.step(
                velocity_fn=velocity_eval,
                z_t=z_curr,
                t=t_curr,
                dt=dt,
            )
            z_curr = z_next

        # Apply continuous camera SE(3) transform across time frames if configured
        if physical_knobs and (abs(physical_knobs.camera_pan) > 1e-6 or abs(physical_knobs.camera_zoom) > 1e-6):
            t_frames = shape[0]
            for ti in range(t_frames):
                z_curr[ti] = controller.apply_camera_transform(z_curr[ti], t=float(ti) / max(1, t_frames - 1))

        # 4D Spatio-Temporal Tensor-Train Compression
        max_r = max(ranks) if isinstance(ranks, (tuple, list)) else int(ranks)
        tt_video = compress_video_latent(z_curr, max_rank=max_r)

        # Decode directly to RGB
        video_rgb = self.vae_bridge.decode_tt_to_video(tt_video)

        return tt_video, video_rgb

    def generate_continuous_slowmo(
        self,
        tt_video: TensorTrainVideo,
        target_fps: int = 60,
        source_fps: int = 24,
    ) -> np.ndarray:
        """Generate high-frame-rate / slow-motion video directly from compressed TT cores.

        Evaluates continuous timestamps tau in [0, 1] without expanding the 4D latent tensor.
        """
        source_frames = tt_video.shape[0]
        duration_sec = (source_frames - 1) / float(source_fps)
        total_target_frames = round(duration_sec * target_fps) + 1

        timestamps = np.linspace(0.0, 1.0, total_target_frames)
        rgb_frames = [
            self.vae_bridge.decode_continuous_frame_rgb(tt_video, float(tau))
            for tau in timestamps
        ]
        return np.stack(rgb_frames, axis=0)
