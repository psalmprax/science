"""JetFlow-Video: Continuous High-Dimensional Spatio-Temporal PDE Video Generation.

Status: experimental

Sub-project providing:
1. 4D Tensor-Train (TT / MPS) Latent Video Representation (`TensorTrainVideo`).
2. High-Order Curvature Flow Matching Stepper (`JetFlowStepper`).
3. Continuous Temporal Interpolator for Infinite FPS (`ContinuousTemporalInterpolator`).
"""
from apedp_dyn.video.continuous_interpolator import (
    ContinuousTemporalInterpolator,
)
from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport,
    DecoupledTransportConfig,
)
from apedp_dyn.video.jet_flow_model import (
    JetFlowModelConfig,
    JetFlowVideoModel,
)
from apedp_dyn.video.jet_flow_stepper import (
    JetFlowConfig,
    JetFlowStepper,
    JetFlowTrajectory,
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
    decompress_video_latent,
    evaluate_continuous_temporal_frame,
)
from apedp_dyn.video.vae_bridge import (
    ContinuousWaveletVAE,
    VideoVAEBridge,
)

__all__ = [
    "ContinuousTemporalInterpolator",
    "ContinuousWaveletVAE",
    "DecoupledSpatioTemporalTransport",
    "DecoupledTransportConfig",
    "JetFlowConfig",
    "JetFlowModelConfig",
    "JetFlowStepper",
    "JetFlowTrajectory",
    "JetFlowVideoModel",
    "NoetherConfig",
    "NoetherConservationEngine",
    "PhysicalSteeringConfig",
    "PhysicalSteeringController",
    "SemanticConditioningConfig",
    "SemanticVideoConditioner",
    "TensorTrainVideo",
    "VideoVAEBridge",
    "compress_video_latent",
    "decompress_video_latent",
    "evaluate_continuous_temporal_frame",
]

