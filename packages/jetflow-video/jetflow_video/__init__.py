r"""JetFlow-Video: Continuous High-Dimensional Spatio-Temporal PDE Video Generation & 4D Tensor-Train Compression.

Status: experimental

Slashes generative video latency to 2 to 4 curvature steps via 2nd-order Jet2 flow matching
and eliminates VRAM bottlenecks (75x - 186x compression) via 4D Tensor-Train decomposition.
"""
from __future__ import annotations

from jetflow_video.tensor_train import (
    TTLinear,
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
    CausalConv3d,
    CausalResBlock3D,
    Causal3DSpatioTemporalVAE,
    ContinuousWaveletVAE,
    DiffusersVAEBridge,
    VideoVAEBridge,
)
from jetflow_video.depth_se3 import (
    se3_exp_map,
    compute_plucker_rays,
    PluckerRayEmbedder,
    DepthAware3DTransportEngine,
)
from jetflow_video.text_encoder import (
    CompositionalTextEncoder,
    encode_prompt_sequence,
    extract_physical_steering_from_prompt,
)
from jetflow_video.model import (
    JetFlowModelConfig,
    JetFlowBackbone,
    JetFlowVideoModel,
)
from jetflow_video.trainer import (
    VideoDataset,
    JetFlowTrainer,
    encode_prompt_deterministic,
)

__version__ = "0.2.0"

__all__ = [
    # Tensor-Train Compression & Factorized Layers
    "TTLinear",
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
    # VAE Bridge & 3D Causal Spatio-Temporal VAE
    "CausalConv3d",
    "CausalResBlock3D",
    "Causal3DSpatioTemporalVAE",
    "ContinuousWaveletVAE",
    "DiffusersVAEBridge",
    "VideoVAEBridge",
    # 3D Depth & SE(3) Plücker Ray Transport
    "se3_exp_map",
    "compute_plucker_rays",
    "PluckerRayEmbedder",
    "DepthAware3DTransportEngine",
    # Compositional Text Encoder
    "CompositionalTextEncoder",
    "encode_prompt_sequence",
    "extract_physical_steering_from_prompt",
    "encode_prompt_deterministic",
    # Core Model
    "JetFlowModelConfig",
    "JetFlowBackbone",
    "JetFlowVideoModel",
    # Trainer & Dataset
    "VideoDataset",
    "JetFlowTrainer",
]
