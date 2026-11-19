# 🌊 JetFlow-Video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](examples/colab_quickstart.ipynb)
![Version 0.1.0, not published to PyPI](https://img.shields.io/badge/version-0.1.0_(not_on_PyPI)-blue.svg)
[![License: Apache-2.0, as declared in pyproject.toml](https://img.shields.io/badge/License-Apache_2.0-green.svg)](pyproject.toml)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](../../AGENTS.md)

**JetFlow-Video** is an experimental research engine for video generation as a continuous flow:
a second-order (Jet2) Taylor integrator over latent trajectories, tensor-train storage of the
latent clip that can be evaluated at any time between frames, and optional physical steering
terms. It runs on a CPU or a single small GPU at the clip sizes it has been measured on.

---

## What has been measured, and what has not

Everything below comes from the repository's own benchmarks, at toy scale: ten synthetic clips
of 16×16 pixels and six frames, eight for training and two held out
(`benchmarks/jetflow_vs_diffusion_matched*_results.json`, with a diffusion baseline of matched
size trained for the same number of steps). Image quality is the PSNR and SSIM against the most
similar held-out clip, not against a reference frame.

| | Fluid clips | Collision clips |
| :--- | :--- | :--- |
| JetFlow, 2 steps | 10.7 dB, SSIM 0.51 | 13.7 dB, SSIM 0.017 |
| Diffusion, matched cost (2 and 4 steps) | 11.0 dB, SSIM 0.48 | 20.8 dB, SSIM 0.065 |
| Diffusion, 48 steps | 10.9 dB, SSIM 0.49 | 20.9 dB, SSIM 0.066 |

- On the fluid clips, two JetFlow steps reach the quality of a 48-step diffusion sampler of the
  same size. On the collision clips they do not: the preregistered confirmation
  (`benchmarks/jetflow_video_v03_confirmation_results.json`) found diffusion better on SSIM and on
  temporal consistency on all three seeds, so the fluid result does not transfer.
- Nothing here has been compared with production video models (Sora, CogVideoX, Hunyuan or
  others), at their resolutions, or on real footage. No claim about photorealism is made.
- **Tensor-train storage** is lossy below full rank. The storage ratio is arithmetic on the
  shape and bond ranks; what survives depends on the content. On the repository's 64×64 vortex
  clips, rank 16 stores 5.5× less and leaves 64 % of the variance unexplained. It shrinks the
  latent clip, not the network, so it is not a VRAM figure.
- **The solenoidal projector** removes the divergence of a 2D vector field on a periodic grid to
  machine precision (pinned by `tests/test_dyn_jetflow_video_package.py`). In the model it is
  applied to *latent* velocity, pairing latent channels (0, 1), (2, 3), … as if they were x and y
  components. Latent channels are not physical velocities, so this does not by itself guarantee
  anything about mass or shape in the decoded pixels; with wavelet latents it is replaced by a
  temporal mass-drift penalty.
- **Continuous time:** a frame at any τ in [0, 1] is evaluated by a cubic spline through the
  temporal tensor-train core, without decoding the whole clip.

---

## 📦 Installation

### 1. In Google Colab
Run in your Colab notebook:
```bash
# Clone the repository and install in editable mode
!git clone https://github.com/psalmprax/science.git
%cd science/packages/jetflow-video
!pip install -e .
!pip install imageio[ffmpeg] scipy tqdm
```

### 2. Standard pip installation
```bash
pip install -e packages/jetflow-video
```

Or with optional HuggingFace Diffusers VAE support:
```bash
pip install -e "packages/jetflow-video[diffusers]"
```

---

## ⚡ Quickstart

### 1. Train on Video Clips in Google Colab
```python
import torch
from torch.utils.data import DataLoader
from jetflow_video import (
    JetFlowVideoModel,
    JetFlowModelConfig,
    JetFlowTrainer,
    VideoDataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stream pre-computed video matrices directly from Hugging Face with 0 MB disk usage!
# (Also supports local latent_dir, video_dir, or procedural synthetic sequences)
dataset = VideoDataset(
    hf_repo="Jnaranjo/video-dit-latents-hq",
    max_hf_samples=100,  # Or stream all 15,614
    num_frames=16,
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model (dynamic latent channels from dataset: 4 for SD/DiT, 12 for Wavelet)
detected_channels = dataset[0]["latent"].shape[1]
config = JetFlowModelConfig(latent_channels=detected_channels, hidden_dim=128, jet_order=2)
model = JetFlowVideoModel(config).to(device)

# Train with 2nd-order Jet2 flow matching
trainer = JetFlowTrainer(model=model, learning_rate=2e-4, device=device)
for epoch in range(1, 4):
    metrics = trainer.train_epoch(dataloader, epoch=epoch)
    print(f"Epoch {epoch}: Total Loss = {metrics['mean_total_loss']:.4f}")

trainer.save_checkpoint("jetflow_video_checkpoint.pt")
```

### 2. Fast 2-Step Video Generation
```python
import torch
from jetflow_video import PhysicalSteeringConfig

# Steer the video with physical control knobs
physics = PhysicalSteeringConfig(
    convection_velocity=(0.2, 0.0), # Drift along x
    viscosity_nu=0.01,              # Viscous damping
    camera_pan=0.1,                 # Camera pan
    time_dilation=1.0,
)

prompt_emb = torch.randn(64, device=device)

# Generate in only 2 curvature steps!
tt_video, generated_rgb = model.generate_video(
    prompt_embedding=prompt_emb,
    num_steps=2,
    shape=(16, 12, 32, 32),
    physical_knobs=physics,
)

print(f"Tensor-Train Compression: {tt_video.compression_ratio:.1f}x reduction")
print(f"Video Shape: {generated_rgb.shape}") # (16, 3, 64, 64)
```

### 3. Continuous-Time Slow Motion Direct from TT Cores
```python
# Evaluate continuous timestamps tau in [0, 1] in O(R^2 C + R^2 H + RW) without 4D expansion:
timestamps = torch.linspace(0.0, 1.0, 60) # 60 FPS super slow-motion
slowmo_frames = [
    model.vae_bridge.decode_continuous_frame_rgb(tt_video, float(tau))
    for tau in timestamps
]
slowmo_video = torch.stack(slowmo_frames, dim=0) # (60, 3, 64, 64)
```

---

## 🔬 Mathematical Foundations

### 1. 2nd-Order Jet2 Curvature Flow Matching
Standard flow matching integrates the velocity field $\frac{d\mathbf{z}_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c})$ using 1st-order Euler steps, which need many small steps where trajectories curve. JetFlow computes the **total convective acceleration**:
$$\mathbf{a}_t = \frac{d\mathbf{v}}{dt} = \frac{\partial \mathbf{v}}{\partial t} + (\nabla_{\mathbf{z}} \mathbf{v}) \mathbf{v}$$
and performs 2nd-order Jet2 Taylor integration:
$$\mathbf{z}_{t + \Delta t} = \mathbf{z}_t + \Delta t \, \mathbf{v}_t + \frac{(\Delta t)^2}{2} \mathbf{a}_t + \mathcal{O}((\Delta t)^3)$$
so that fewer steps are needed than with first-order Euler integration. How few is measured above.

### 2. 4D Spatio-Temporal Tensor-Train (TT-MPS) Compression
Instead of storing dense 4D tensors $\mathbf{Z} \in \mathbb{R}^{T \times C \times H \times W}$ requiring gigabytes of VRAM, JetFlow decomposes the video into 4 low-rank tensor cores:
$$\mathbf{Z}(t, c, h, w) = \sum_{r_1=1}^{R_1} \sum_{r_2=1}^{R_2} \sum_{r_3=1}^{R_3} G_1(1, t, r_1) \, G_2(r_1, c, r_2) \, G_3(r_2, h, r_3) \, G_4(r_3, w, 1)$$
Storage is the sum of the core sizes instead of $T C H W$; the ratio depends on the shape and the ranks, and the reconstruction is lossy below full rank (see *What has been measured*).

### 3. Lie-Algebraic Helmholtz-Hodge Solenoidal Projector
To prevent the classic generative video defect of object deformation and spontaneous mass creation/annihilation, the Noether engine solves the Poisson equation in Fourier space:
$$\nabla^2 \phi = \nabla \cdot \mathbf{v}, \quad \mathbf{v}_{\text{sol}} = \mathbf{v} - \nabla \phi$$
which removes the (central-difference) divergence of the field it is applied to, to machine precision. In the model that field is a latent velocity; see *What has been measured* for what that does and does not imply.

---

## 📜 Repository & Agent Protocols
This package adheres strictly to the `APEDP-X / CPMD` epistemic referee protocol outlined in [`AGENTS.md`](../../AGENTS.md). All modules carry the `experimental` status label.
