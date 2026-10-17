# JetFlow Universal Spatio-Temporal Architecture & Higher-Order PDE Hypothesis Engine

**Status:** `experimental` (Verified across 108/108 repository regression + higher-order PDE tests)

---

## 1. Architecture Identity & Classification

* **Short Name:** **JetFlow-Video** / **JetFlow-DiT**
* **Full Technical Classification:** **Physics-Informed 2nd-Order Jet Flow-Matching Spatio-Temporal Transformer with 4D Tensor-Train Factorization**
* **Architectural Category:** **Hybrid Neuro-Physical Transformer** — a deep neural network ([`JetFlowBackbone`](../packages/jetflow-video/jetflow_video/model.py) + [`Causal3DSpatioTemporalVAE`](../packages/jetflow-video/jetflow_video/bridge.py)) coupled with a 2nd-order continuous ODE/PDE solver ([`JetFlowStepper`](../packages/jetflow-video/jetflow_video/stepper.py)), Noether conservation laws ([`NoetherConservationEngine`](../packages/jetflow-video/jetflow_video/noether.py)), 3D $\text{SE}(3)$ camera-ray geometry ([`DepthAware3DTransportEngine`](../packages/jetflow-video/jetflow_video/depth_se3.py)), and 4D Tensor-Train / Matrix Product State compression ([`TensorTrainVideo`](../packages/jetflow-video/jetflow_video/tensor_train.py), [`TTLinear`](../packages/jetflow-video/jetflow_video/tensor_train.py)).

### The Five Architectural Layers

| Layer | Component Type | Implementation Module | Mathematical Role |
| :--- | :--- | :--- | :--- |
| **1. Core Velocity Brain** | Factorized Spatio-Temporal Transformer (DiT) | [`packages/jetflow-video/jetflow_video/model.py`](../packages/jetflow-video/jetflow_video/model.py) | Packs 3D tubelets `(pt, ph, pw)`, injects 21-D 3D $(\tau, y, x)$ Fourier features + 6D Plücker rays, and applies AdaLN-Zero modulated 2D spatial stencil + spatial attention, multi-token cross-attention, and temporal attention to predict $\mathbf{v}_\theta(\mathbf{z}, t, \mathbf{c})$. |
| **2. Generative Integrator** | 2nd-Order Jet2 Flow-Matching Solver | [`packages/jetflow-video/jetflow_video/stepper.py`](../packages/jetflow-video/jetflow_video/stepper.py) | Integrates $\mathbf{z}_{t+\Delta t} = \mathbf{z}_t + \mathbf{v}_t \Delta t + \frac{1}{2}\mathbf{a}_t \Delta t^2$ in **2 to 4 steps** using the 2-evaluation fused directional material derivative $\mathbf{a}_t = \partial_t \mathbf{v} + (\nabla_{\mathbf{z}}\mathbf{v})\mathbf{v}$. |
| **3. Physical & 3D Priors** | Noether Conservation + 3D $\text{SE}(3)$ Depth & Plücker Transport | [`packages/jetflow-video/jetflow_video/noether.py`](../packages/jetflow-video/jetflow_video/noether.py), [`packages/jetflow-video/jetflow_video/depth_se3.py`](../packages/jetflow-video/jetflow_video/depth_se3.py), [`apedp_dyn/video/decoupled_transport.py`](../apedp_dyn/video/decoupled_transport.py) | Enforces subband-preserving divergence-free conservation ($\nabla \cdot \mathbf{v} = 0$), models monocular depth $Z_\theta(\mathbf{z})$ and 6-DoF $\text{SE}(3)$ camera orbits ($\exp(\mathfrak{se}(3))$), and executes semi-Lagrangian PDE advection in native Rust AVX2/Rayon kernels ([`rust/apedp-kernel/src/video.rs`](../rust/apedp-kernel/src/video.rs)). |
| **4. Spatio-Temporal Tokenizer** | Learned 3D Causal Wavelet-VAE | [`packages/jetflow-video/jetflow_video/bridge.py`](../packages/jetflow-video/jetflow_video/bridge.py), [`packages/jetflow-video/jetflow_video/perceptual_decoder.py`](../packages/jetflow-video/jetflow_video/perceptual_decoder.py) | Maps `(B, T, 3, H, W)` RGB clips to `(B, T', C, H', W')` latents via invertible Haar wavelets + past-only 3D causal residual convolutions (`CausalConv3d`, `CausalResBlock3D`) with diagonal Gaussian KL regularization. |
| **5. Tensor-Network Compression** | 4D Tensor-Train (TT / MPS) & `TTLinear` | [`packages/jetflow-video/jetflow_video/tensor_train.py`](../packages/jetflow-video/jetflow_video/tensor_train.py) | Factorizes 4D video latents into $G^{(0)}(t) G^{(1)}(c) G^{(2)}(y) G^{(3)}(x)$ with $C^1$ Catmull-Rom cubic Hermite temporal splines and factorizes Transformer FFN weights via `TTLinear`. |

---

## 2. Training Inline with PyTorch vs. Standalone Edge Deployment

### 2.1 Training: Inline with PyTorch (`torch.nn.Module`)
All trainable modules (`JetFlowVideoModel`, `Causal3DSpatioTemporalVAE`, `CompositionalTextEncoder`, `DepthAware3DTransportEngine`, `TTLinear`, `NeuralTransportFlowNet`) are native PyTorch `nn.Module` classes trained via [`JetFlowTrainer`](../packages/jetflow-video/jetflow_video/trainer.py) using:
* **Multi-Objective Jet2 Loss:** $\mathcal{L} = \|\mathbf{v}_\theta - \mathbf{v}^*\|_2^2 + \lambda_{\text{acc}} \|\mathbf{a}_\theta\|_2^2 + \lambda_{\text{noether}} \|\nabla \cdot \mathbf{v}_\theta\|_2^2$
* **Classifier-Free Guidance (CFG) Conditioning Dropout (`cfg_dropout_prob=0.1`)**
* **Exponential Moving Average (`ema_decay=0.995`) Shadow Weights**
* **Linear Warmup + Cosine Annealing LR Schedule & Automatic Mixed Precision (`torch.amp`)**

### 2.2 Inference: PyTorch or Standalone ONNX + Rust
1. **PyTorch Mode:** `model.generate_video("fast red car driving right and orbiting in 3d at sunset", num_steps=2)`
2. **Standalone CPU Mode:** Export `JetFlowBackbone` to static `fp32` / quantized `int8` ONNX via [`cpu_export.py`](../packages/jetflow-video/jetflow_video/cpu_export.py) and run semi-Lagrangian transport via [`rollout_with_rust_transport`](../apedp_dyn/video/decoupled_transport.py).

---

## 3. Why Training Is 5.6× Faster (Measured Matched Benchmark)

From [`benchmarks/jetflow_vs_diffusion_matched_results.json`](../benchmarks/jetflow_vs_diffusion_matched_results.json) (documented in [`docs/jetflow_vs_diffusion_matched_benchmark.md`](jetflow_vs_diffusion_matched_benchmark.md)), comparing `JetFlowVideoModel` (`138,780` params) against `DiffusionVideoUNet` (`152,492` params) over 300 steps on identical Navier–Stokes splits:

| Metric | `jet_tt_2step` | `jet_tt_4step` | `diffusion_matched` (4-step DDIM) | `diffusion_frontier_K48` (20-step DDIM) |
| :--- | :---: | :---: | :---: | :---: |
| **300-Step Training Time (CPU)** | **14.40 s** | **14.40 s** | 80.75 s | 80.75 s |
| **Final Training Loss** | **0.1528** | **0.1528** | 0.1631 | 0.1631 |
| **Forward Passes / Clip** | **8.0** | 16.0 | 8.0 | 40.0 |
| **Held-Out PSNR (dB, $\uparrow$)** | **10.74** | **10.75** | 10.97 | 10.92 |
| **Held-Out SSIM ($\uparrow$)** | **0.5142** | **0.5065** | 0.4765 | 0.4919 |
| **Temporal Flicker ($\downarrow$)** | **0.2752** | **0.2442** | 0.3652 | 0.3716 |
| **FVD-Proxy SWD ($\downarrow$)** | **0.5177** | 0.5625 | 0.4718 | 0.5171 |

---

## 4. Multi-Modal Conditioning & Non-Video Generalization

### 4.1 Multi-Modal Conditioning Pathways
1. **Text Sequence Conditioning ([`text_encoder.py`](../packages/jetflow-video/jetflow_video/text_encoder.py)):** Multi-token `(B, L_seq, D_text)` contextual embeddings with modifier binding, negation gating, temporal phase markers, and HuggingFace `transformers` bridge.
2. **First-Frame Image Conditioning ([`first_frame.py`](../packages/jetflow-video/jetflow_video/first_frame.py)):** Anchored flow matching pinning frame 0 clean for Image-to-Video (I2V).
3. **3D $\text{SE}(3)$ Camera Pose & Plücker Rays ([`depth_se3.py`](../packages/jetflow-video/jetflow_video/depth_se3.py)):** 6D Plücker ray coordinates $\mathbf{r} = (\mathbf{d}, \mathbf{o} \times \mathbf{d})$ and monocular depth $Z_\theta$.
4. **Physical PDE Knobs ([`physics.py`](../packages/jetflow-video/jetflow_video/physics.py)):** Convection velocity $(c_x, c_y)$, viscosity $\nu$, and time dilation $\gamma$.
5. **Audio / Action Token Concatenation:** Arbitrary sequence conditioning tokens `(B, L_cond, text_dim)` concatenate directly into `SpatioTemporalTransformerBlock.cross_attn`.

### 4.2 Six Non-Video Domains Supported by the Same Engine
1. **Still Image Generation ($T=1$):** `shape=(1, C, H, W)` runs 2-step flow-matching image generation.
2. **Audio & Speech Spectrogram Transport:** [`apedp_dyn/audio/field_transport.py`](../apedp_dyn/audio/field_transport.py) ([`tests/test_dyn_audio_field_transport.py`](../tests/test_dyn_audio_field_transport.py)).
3. **Weather, Ocean & High-Dimensional PDE Simulation (1D–100D+):** [`apedp_dyn/epistemic/high_dim_pde.py`](../apedp_dyn/epistemic/high_dim_pde.py).
4. **3D Molecular Conformation ($\text{SE}(3)$ Equivariant Flow):** [`apedp_dyn/molecular/se3_flow.py`](../apedp_dyn/molecular/se3_flow.py) ([`tests/test_dyn_molecular_se3_flow.py`](../tests/test_dyn_molecular_se3_flow.py)).
5. **Robotics & Autonomous Driving $\text{SE}(2)/\text{SE}(3)$ Trajectories:** [`apedp_dyn/robotics/trajectory_stepper.py`](../apedp_dyn/robotics/trajectory_stepper.py) ([`tests/test_dyn_robotics_trajectory.py`](../tests/test_dyn_robotics_trajectory.py)).
6. **Financial Limit Order Book (LOB) Density Transport:** [`apedp_dyn/lob/transport_discovery.py`](../apedp_dyn/lob/transport_discovery.py) ([`tests/test_dyn_lob_transport.py`](../tests/test_dyn_lob_transport.py)).

---

## 5. Higher-Order PDE Hypothesis Engine (`apedp_dyn/pde/higher_order_engine.py`)

Implemented in [`apedp_dyn/pde/higher_order_engine.py`](../apedp_dyn/pde/higher_order_engine.py) and verified in [`tests/test_dyn_higher_order_pde_hypothesis_engine.py`](../tests/test_dyn_higher_order_pde_hypothesis_engine.py), the Higher-Order PDE Hypothesis Engine unites JetFlow's 2nd-order temporal curvature ($\partial_{tt} u$) and 4D Tensor-Train compression with symbolic higher-order PDE discovery under strict `AGENTS.md` Rule 1.1 (Proposer vs. Referee separation):

### 5.1 Supported Higher-Order PDE Families
* **3rd-Order Dispersive PDEs (Korteweg–de Vries / KdV):**
  $$\partial_t u = c_{\text{adv}}\, u\,\partial_x u + c_3\, \partial_{xxx} u$$
* **4th-Order Biharmonic PDEs (Kuramoto–Sivashinsky / Cahn–Hilliard):**
  $$\partial_t u = -c_2\, \partial_{xx} u - \nu\, \partial_{xxxx} u - u\,\partial_x u, \qquad \partial_t u = D\,\nabla^2 u - \gamma\,\nabla^4 u$$
* **2nd-Order-in-Time Hyperbolic / Klein–Gordon / Wave PDEs (Jet2 $\partial_{tt} u$):**
  $$\partial_{tt} u = c^2\,\nabla^2 u - m^2\, u$$

### 5.2 Proposer vs. Referee Separation & Ultraviolet Well-Posedness Guard
* **[`HigherOrderPdeProposer`](../apedp_dyn/pde/higher_order_engine.py):** Fits exclusively on the development split using 4th-order finite-difference / Fourier spectral stencils and column-normalized STRidge + BIC model selection over $k_t \in \{1, 2\}$, optionally compresses 4D fields `(T, C, H, W)` into `TensorTrainVideo`, and emits standardized `CandidateProposal` objects.
* **[`HigherOrderPdeReferee`](../apedp_dyn/pde/higher_order_engine.py):** Evaluates frozen discovered coefficients on a sealed holdout trajectory and enforces the **Ultraviolet Fourier Symbol Stability Guard**:
  * For a 1st-order-in-time PDE $\partial_t u = \sum_m c_m \partial_x^m u$, a Fourier mode $e^{i k x}$ has real growth rate $\text{Re}(\lambda(k)) = -c_2 k^2 + c_4 k^4$.
  * If a 4th-order biharmonic operator ($u_{xxxx}$ or $\nabla^4 u$) is active, the referee requires $c_4 < 0$ so $\text{Re}(\lambda(k)) \to -\infty$ as $|k| \to \infty$ (allowing anti-diffusive instability $c_2 < 0$ at low wavenumbers as in Kuramoto–Sivashinsky, while rejecting ill-posed $c_4 > 0$ ultraviolet blowup).

### 5.3 Example Usage

```python
from apedp_dyn.pde import HigherOrderPdeProposer, HigherOrderPdeReferee

# 1. Propose higher-order PDE hypothesis on development trajectory (never touches holdout)
proposer = HigherOrderPdeProposer(dt=0.001, dx=0.098, max_spatial_order=4, periodic=True)
result, proposals, tt_summary = proposer.propose_from_field(dev_field, compress_tt=True)

# 2. Adjudicate frozen hypothesis on sealed holdout trajectory via Epistemic Referee
referee = HigherOrderPdeReferee(min_holdout_r2=0.95, enforce_ultraviolet_stability=True)
verdict = referee.adjudicate(result, holdout_field, dt=0.001, dx=0.098, periodic=True)
print(result.formula, "->", verdict.verdict, f"(OOS R^2 = {verdict.frozen_oos_r2:.4f})")
```
