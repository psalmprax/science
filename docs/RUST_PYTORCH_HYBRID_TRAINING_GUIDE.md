# Hybrid Rust ↔ PyTorch Usage Guide

**Status:** production pattern (documentation only — no runtime changes).
**Applies to:** `rust/apedp-kernel` (kernel crate), `apedp/rust_bridge.py`,
`apedp_dyn/video/learned_flow_head.py`, `apedp_dyn/video/decoupled_transport.py`,
`apedp_dyn/video/flow_head_rust.py`.
**Verified example:** the complete script in §7 was executed from the repo root
against the installed kernel; its real output is quoted in §7.

---

## 1. The idea in plain words

The Rust kernel is the factory: fast, deterministic, it does the heavy numeric
step (warping an image through a flow field, integrating a solver step).
PyTorch is the drafting office: it learns *what* flow to use. This guide shows
the standard way to connect them so that

1. training can run on numbers computed by the same code deployment will use,
2. gradients still flow into the neural network, and
3. inference is a plain, gradient-free call into the full Rust solver.

The pattern has three pieces: the **wheel** (§2), the **bridge** (§3), and the
**autograd wrapper** (§4). Physics ideas (PDE/divergence constraints) enter as
ordinary **loss terms** (§5) — never as hidden code.

---

## 2. Getting the Rust code into Python (the pip story)

`rust/apedp-kernel` is a [pyo3](https://pyo3.rs) crate. `maturin` compiles it
into a Python extension named `apedp_kernel` and packages it as a *wheel* —
a zip that pip copies into `site-packages`. After that, `import apedp_kernel`
loads native machine code, exactly like numpy.

```bash
pip install maturin
maturin build --release --manifest-path rust/apedp-kernel/Cargo.toml -o dist/
pip install dist/apedp_kernel-*.whl

# Development alternative (what this repo does locally): unzip the wheel and
# copy the shared object into site-packages:
#   unzip -o dist/apedp_kernel-*.whl -d /tmp/wheel && \
#   cp /tmp/wheel/apedp_kernel/libapedp_kernel.so ~/.local/lib/python3.12/site-packages/apedp_kernel/
```

Nothing about the training code depends on how the extension got there — the
bridge in §3 hides the whole question.

---

## 3. One import, two roads (the bridge with graceful fallback)

```python
from apedp.rust_bridge import RustKernelBridge

bridge = RustKernelBridge(force_fallback=False)
print(bridge.use_rust)   # True when the Rust extension is importable
```

The bridge is the toll booth: it *tries* the fast Rust road and, on any machine
where the extension was never built (say a fresh Colab), quietly falls back to
a slower NumPy implementation of the same math — training still runs, just
less fast. The two roads are pinned against each other in tests
(`tests/test_dyn_periodic_boundary_transport.py`,
`tests/test_dyn_learned_flow_head.py`) at a **1e-12** tolerance, so "fallback"
can never silently mean "different results".

---

## 4. Rust inside the training loop: the autograd bridge

PyTorch trains with two passes: **forward** (compute numbers) and **backward**
("blame": how should each number move to reduce the loss?). Rust knows nothing
about the blame pass. So we wrap the Rust kernel in a
`torch.autograd.Function` that says:

- *forward* — run in Rust (the exact math deployment will use);
- *backward* — recompute the same warp with the PyTorch twin function and let
  PyTorch's own autograd produce the gradient.

It is like the factory casting the part while the drafting office draws the
correction notes. The rows generated this way are bit-identical to the pure
torch path (parity 0 in the verified run below; pinned at 1e-12 in the tests).

```python
import torch
from apedp_dyn.video.learned_flow_head import warp_frames


class RustWarp(torch.autograd.Function):
    """Rust does the forward cast; the torch twin draws the gradient notes."""

    @staticmethod
    def forward(ctx, z, vx, vy, bridge, dt, periodic):
        # Rust speaks NumPy, one sample at a time: squeeze our batch of 1.
        # Sign convention: our `v` means "the wind blows this way"; the
        # bridge's displacement means "sample from where the air came from" —
        # same physics, opposite sign.
        ctx.save_for_backward(z.detach(), vx.detach(), vy.detach())
        ctx.dt, ctx.periodic = dt, periodic
        out = bridge.video_warp_2d(
            z.detach().numpy()[0],
            -dt * vx.detach().numpy()[0],
            -dt * vy.detach().numpy()[0],
            periodic=periodic,
        )
        return torch.from_numpy(out).unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_out):
        # Recompute with the torch twin on the saved numbers; autograd then
        # hands us the gradients for the flow fields. The context frame z is
        # fixed data -> no gradient for it (return None).
        z, vx, vy = ctx.saved_tensors
        vx_ = vx.clone().requires_grad_(True)
        vy_ = vy.clone().requires_grad_(True)
        with torch.enable_grad():
            out = warp_frames(z, vx_, vy_, dt=ctx.dt, periodic=ctx.periodic)
            gvx, gvy = torch.autograd.grad(out, (vx_, vy_), grad_out)
        return None, gvx, gvy, None, None, None
```

Any Rust numeric step (warp, RK45, PDE step) integrates the same way: forward
calls it, backward uses a torch twin. That single class is the whole bridge.

---

## 5. PDE ideas as ordinary loss terms (the physics nudge)

Physics constraints are not layers; they are penalty terms on the loss. A real
incompressible flow never piles up or disappears — the divergence measures
local piling-up, and punishing it keeps the estimated flow smooth, which also
stops the warp from smearing (blurring) the picture:

```python
# divergence (how much the flow compresses/expands locally), central differences
dvx = v[:, 0, :, 2:] - v[:, 0, :, :-2]
dvy = v[:, 1, 2:, :] - v[:, 1, :-2, :]
div = dvx[:, 1:-1, :] * 0.5 + dvy[:, :, 1:-1] * 0.5
loss_phys = 0.01 * (div ** 2).mean()

loss = loss_fit + loss_phys      # loss_fit = MSE against the true next frame
```

This is the same family as the `lambda_div` / `lambda_smooth` regularizers
inside `flow_consistency_loss`. Keep these terms in torch (GPU-able); they
shape the flow manifold, they do not add texture — anti-blur proper comes from
perceptual/sampling objectives, not from physics terms.

---

## 6. Deployment inference: the gradient-free Rust path

Serving needs no gradients, so the wrapper disappears and the *full* engine
runs: the multi-scale transport solver calling the Rust kernel per step.

```python
from apedp_dyn.video.learned_flow_head import rollout_dcstt_head
from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport, DecoupledTransportConfig,
)

solver = DecoupledSpatioTemporalTransport(DecoupledTransportConfig(
    use_rust=True, nu=0.0, enable_laplacian_dissipation=False, periodic_warp=True))

rollout = rollout_dcstt_head(head, solver, context_frames, horizon=4,
                             ch0_policy="persist")
```

Training and deployment therefore agree by construction: the training loss
warped with the same periodic convention the solver uses, and the parity tests
hold the two implementations together.

---

## 7. Verified reference run (complete script)

The script below was executed from the repo root with the kernel installed;
the quoted output underneath is from that actual run. It is a plumbing demo on
one toy clip (150 steps) — it proves the integration, not benchmark quality.

```python
"""DEMO: plugging the Rust kernels into a PyTorch training loop, end to end.

Run from the repo root:  python3 demo_hybrid_training.py
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = "."                              # repo root (adjust if run elsewhere)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from apedp.rust_bridge import RustKernelBridge
from apedp_dyn.video.learned_flow_head import (
    FlowHeadConfig, LearnedFlowHead, rollout_dcstt_head, warp_frames,
)
from apedp_dyn.video.decoupled_transport import (
    DecoupledSpatioTemporalTransport, DecoupledTransportConfig,
)
from apedp_dyn.video.fluid_domain import make_fluid_clip

# PART 1 — the pip story: the bridge tries Rust, falls back to NumPy.
bridge = RustKernelBridge(force_fallback=False)
print(f"[1] Rust kernel active: {bridge.use_rust}")

# PART 2 — the autograd bridge (see §4 for the annotated version).
class RustWarp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, vx, vy, bridge, dt, periodic):
        ctx.save_for_backward(z.detach(), vx.detach(), vy.detach())
        ctx.dt, ctx.periodic = dt, periodic
        out = bridge.video_warp_2d(
            z.detach().numpy()[0],
            -dt * vx.detach().numpy()[0],
            -dt * vy.detach().numpy()[0],
            periodic=periodic,
        )
        return torch.from_numpy(out).unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_out):
        z, vx, vy = ctx.saved_tensors
        vx_ = vx.clone().requires_grad_(True)
        vy_ = vy.clone().requires_grad_(True)
        with torch.enable_grad():
            out = warp_frames(z, vx_, vy_, dt=ctx.dt, periodic=ctx.periodic)
            gvx, gvy = torch.autograd.grad(out, (vx_, vy_), grad_out)
        return None, gvx, gvy, None, None, None

# Proof the two roads agree; they must, or training drifts from deployment.
zt = torch.randn(1, 2, 16, 16, dtype=torch.float64)
vxt = torch.randn(1, 16, 16, dtype=torch.float64) * 2
vyt = torch.randn(1, 16, 16, dtype=torch.float64) * 2
a = RustWarp.apply(zt, vxt, vyt, bridge, 1.0, True)
b = warp_frames(zt, vxt, vyt, dt=1.0, periodic=True)
print(f"[2] Rust-vs-torch parity max diff: {float((a - b).abs().max()):.2e}")

# PART 3 — data: a tiny simulated fluid (seed 2 = training-pattern seed;
# holdout seeds 3/5 belong to the evaluation registry).
clip, _, _ = make_fluid_clip(2, t=6, h=32, w=32, return_flow=True)
z_data = torch.from_numpy(np.asarray(clip[:, 1:3], dtype=np.float64))
print(f"[3] data: clip {clip.shape} -> head sees {tuple(z_data.shape)}")

# PART 4 — training loop: gradients flow THROUGH the Rust call.
torch.manual_seed(0)
head = LearnedFlowHead(
    FlowHeadConfig(hidden_dim=16, n_layers=2, lambda_div=0.0, lambda_smooth=0.0)
).double()
opt = torch.optim.AdamW(head.parameters(), lr=3e-3)

first = last = None
for step in range(150):
    ti = 1 + (step % (z_data.shape[0] - 2))     # tour the clip's transitions
    z_prev, z_curr, z_next = z_data[ti - 1:ti], z_data[ti:ti + 1], z_data[ti + 1:ti + 2]

    v = head(z_prev, z_curr)                    # (1, 2, H, W): where does dye go?
    z_pred = RustWarp.apply(z_curr, v[:, 0], v[:, 1], bridge, 1.0, True)
    loss_fit = F.mse_loss(z_pred, z_next)       # "did we land on the real frame?"

    dvx = v[:, 0, :, 2:] - v[:, 0, :, :-2]      # physics nudge (§5)
    dvy = v[:, 1, 2:, :] - v[:, 1, :-2, :]
    div = dvx[:, 1:-1, :] * 0.5 + dvy[:, :, 1:-1] * 0.5
    loss_phys = 0.01 * (div ** 2).mean()

    loss = loss_fit + loss_phys
    opt.zero_grad(set_to_none=True)
    loss.backward()                              # <- crosses RustWarp into the head
    opt.step()

    if step == 0:
        first = float(loss_fit)
    last = float(loss_fit)
    if step % 50 == 0:
        print(f"[4] step {step:3d}  fit-loss {float(loss_fit):.6f}  "
              f"div-penalty {float(loss_phys):.2e}")
print(f"[4] fit-loss {first:.6f} -> {last:.6f} (learning worked through Rust)")

# PART 5 — deployment-style inference: no gradients, full Rust solver (§6).
solver = DecoupledSpatioTemporalTransport(DecoupledTransportConfig(
    use_rust=True, nu=0.0, enable_laplacian_dissipation=False, periodic_warp=True))
roll = rollout_dcstt_head(head, solver, np.asarray(clip[:2]), horizon=4,
                          ch0_policy="persist")
truth = np.asarray(clip[2:6])
pers = np.repeat(np.asarray(clip[1:2]), 4, axis=0)
mse_l = float(np.mean((roll[2:] - truth) ** 2))
mse_p = float(np.mean((pers - truth) ** 2))
print(f"[5] 4-frame rollout MSE: learned {mse_l:.3e} vs persistence {mse_p:.3e}")
```

Observed output of that run:

```text
[1] Rust kernel active: True
[2] Rust-vs-torch parity max diff: 0.00e+00
[3] data: clip (6, 3, 32, 32) -> head sees (6, 2, 32, 32)
[4] step   0  fit-loss 0.001308  div-penalty 2.18e-06
[4] step  50  fit-loss 0.000946  div-penalty 3.19e-05
[4] step 100  fit-loss 0.000464  div-penalty 6.84e-05
[4] fit-loss 0.001308 -> 0.000488 (learning worked through Rust)
[5] 4-frame rollout MSE: learned 1.508e-03 vs persistence 2.424e-03
```

---

## 8. The rules of the boundary

1. **Torch owns the graph.** Every differentiable path stays in torch (CPU or
   CUDA). The torch twin in `backward` is the only gradient source.
2. **Rust owns the boundaries.** Data generation, kernel steps, metrics and
   verification, and frozen-model export. Rust is not a training engine.
3. **Every boundary gets a parity test.** Current pins:
   - warp / fallback parity: 1e-12
     (`tests/test_dyn_periodic_boundary_transport.py`,
     `tests/test_dyn_learned_flow_head.py`);
   - Rust flow-head forward ≤ 5e-6 (erf-based GELU grid), rollout-step parity
     ~1e-9 (`tests/test_dyn_flow_head_rust.py` docstring).
4. **The tiny frozen head is the only hand-ported model.**
   `apedp_dyn/video/flow_head_rust.py` (`RustFlowHeadSpec`, `pack_weights`,
   `rust_train_flow_head`) is a distillation/export target. Do not hand-port
   larger models — export them to a real runtime (ONNX Runtime / candle)
   instead.
5. **Holdout discipline applies to any data factory.** Train-pattern clips
   only; holdout slices stay in the evaluation registry (AGENTS.md §1.2).

---

## 9. Scaling up (the pattern, not the toy sizes)

- **GPU / Colab:** keep the structure of §4; let `forward`/`backward` use CUDA
  torch ops (the wrapper's shape is unchanged) and keep the Rust path as the
  CPU/audit twin. Beware fp64: this pipeline runs in double precision, which
  consumer GPUs throttle.
- **Data factory:** reuse the same bridge inside DataLoader workers to mint
  many deterministic training clips — the measured bottleneck of the flow head
  is data diversity, not steps.
- **Export:** once a model is frozen, prefer an inference runtime over
  hand-written Rust math; keep the transport/solver steps in Rust.
