r"""JetFlow-Video: Continuous High-Dimensional Spatio-Temporal PDE Video Generation & 4D Tensor-Train Compression.

Status: experimental

Slashes generative video latency to 2 to 4 curvature steps via 2nd-order Jet2 flow matching
and eliminates VRAM bottlenecks (75x - 186x compression) via 4D Tensor-Train decomposition.
"""
from __future__ import annotations

from jetflow_video.tensor_train import (
    TensorTrainVideo,
    compress_video_latent_torch,
    decompress_video_latent_torch,
    evaluate_continuous_temporal_frame_torch,
)
from jetflow_video.stepper import (
    JetFlowConfig,
    JetFlowTrajectory,
    JetFlowStepper,
)
from jetflow_video.noether import (
    NoetherConfig,
    NoetherConservationEngine,
)
from jetflow_video.physics import (
    PhysicalSteeringConfig,
    PhysicalSteeringController,
)
from jetflow_video.bridge import (
    ContinuousWaveletVAE,
    DiffusersVAEBridge,
    VideoVAEBridge,
)
from jetflow_video.model import (
    JetFlowModelConfig,
    JetFlowBackbone,
    JetFlowVideoModel,
)
from jetflow_video.trainer import (
    VideoDataset,
    JetFlowTrainer,
)

__version__ = "0.1.0"

__all__ = [
    # Tensor-Train Compression
    "TensorTrainVideo",
    "compress_video_latent_torch",
    "decompress_video_latent_torch",
    "evaluate_continuous_temporal_frame_torch",
    # Jet2 Curvature Stepper
    "JetFlowConfig",
    "JetFlowTrajectory",
    "JetFlowStepper",
    # Noether Solenoidal Projector
    "NoetherConfig",
    "NoetherConservationEngine",
    # Physics Controller
    "PhysicalSteeringConfig",
    "PhysicalSteeringController",
    # VAE Bridge
    "ContinuousWaveletVAE",
    "DiffusersVAEBridge",
    "VideoVAEBridge",
    # Core Model
    "JetFlowModelConfig",
    "JetFlowBackbone",
    "JetFlowVideoModel",
    # Trainer & Dataset
    "VideoDataset",
    "JetFlowTrainer",
]
