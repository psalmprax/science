r"""JetFlow against plain flow matching and MeanFlow, at equal training time, by the number of network calls.

Status: experimental

The owner asked on 26 November whether JetFlow is a better way to train a video model, and later to
build the comparison. Everything here shares one latent space (the fixed wavelet autoencoder of
``bridge.ContinuousWaveletVAE``, no learned parameters), one clip shape, one batch size (one clip per
step, because JetFlow's backbone conditions correctly only at batch one) and one optimiser; only the
network, the objective and the sampler differ:

* ``fm-jetflow``: ``JetFlowVideoModel`` trained by flow matching, sampled by its own sampler (the Jet2
  second-order step and its latent velocity projection; its steering is the identity at default settings),
  as packaged and with the step's curvature sign corrected, and by plain Euler;
* ``fm-unet``: the repository's compact video UNet (``diffusion_baseline``) of about the same size,
  trained by the same flow matching, sampled by Euler and by the Jet2 step alone;
* ``meanflow-unet``: the same UNet with one more time input, trained by MeanFlow (Geng et al., 2025):
  it learns the average velocity u(z_t, r, t) over [r, t] from the identity
  u = v - (t - r) du/dt, the total derivative taken by one forward-mode product, so that one call
  u(z_1, 0, 1) goes from noise to data.

Conventions are the repository's: z_t = (1 - t) x + t e, v = e - x, sampling from t = 1 to t = 0. Every
model is conditioned on the same constant zero prompt in training and sampling (an unconditional model;
guidance off). Quality is the Frechet distance between generated and held-out real clips in the space
of the first principal components of the training clips (``frechet_pca``), with the distance of the
training clips themselves to the held-out ones as its floor, and a copying check
(``memorisation_ratio``).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from jetflow_video.bridge import ContinuousWaveletVAE
from jetflow_video.diffusion_baseline import (
    DiffusionVideoConfig,
    DiffusionVideoUNet,
    count_parameters,
    pick_matched_config,
)
from jetflow_video.model import JetFlowModelConfig, JetFlowVideoModel
from jetflow_video.noether import NoetherConfig, NoetherConservationEngine
from jetflow_video.stepper import JetFlowConfig, JetFlowStepper

FRAMES, SIZE, LATENT = 6, 16, 12
TEXT = torch.zeros(1, 2, 64)                                    #: the constant prompt every model sees
NOETHER = NoetherConservationEngine(NoetherConfig(enforce_divergence_free=True, preserve_wavelet_subbands=True))
JETFLOW = {"latent_channels": LATENT, "hidden_dim": 48, "num_layers": 2}   #: as in the earlier matched benchmark


# ---------------------------------------------------------------- networks

def build_jetflow() -> JetFlowVideoModel:
    model = JetFlowVideoModel(JetFlowModelConfig(**JETFLOW))  # type: ignore[arg-type]
    model.config.guidance_scale = 1.0
    return model


def unet_config() -> DiffusionVideoConfig:
    """The UNet width closest to the parameters JetFlow's forward uses, its backbone (within 15 %); the text
    encoder and depth engine it also holds are not used by these unconditional models."""
    return pick_matched_config(count_parameters(build_jetflow().backbone), latent_channels=LATENT, frames=FRAMES)


def _fourier(module: Any, t: Tensor) -> Tensor:
    """``FourierTimeEmbedding`` of a scalar tensor t, differentiable in t (forward-mode included)."""
    angles = t.reshape(1, 1) * module.freqs.unsqueeze(0).to(t.dtype) * (2.0 * math.pi)
    out: Tensor = module.mlp(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))
    return out


class MeanFlowUNet(DiffusionVideoUNet):
    """The compact UNet with the gap t - r embedded beside t; with r = t it is a velocity model."""

    def __init__(self, config: DiffusionVideoConfig) -> None:
        super().__init__(config)
        self.gap_embed = type(self.time_embed)(config.width, config.fourier_frequencies)

    def forward(self, z: Tensor, t: float | Tensor, text_tokens: Tensor | None = None,  # type: ignore[override]
                r: float | Tensor | None = None) -> Tensor:
        t_t = torch.as_tensor(t, dtype=z.dtype) if not isinstance(t, Tensor) else t
        r_t = t_t if r is None else (torch.as_tensor(r, dtype=z.dtype) if not isinstance(r, Tensor) else r)
        squeeze = z.ndim == 4
        if squeeze:
            z = z.unsqueeze(0)
        b, frames, c, h, w = z.shape
        x = self.input_proj(z.reshape(b * frames, c, h, w))
        cond = (_fourier(self.time_embed, t_t) + _fourier(self.gap_embed, t_t - r_t)).repeat(b * frames, 1)
        if text_tokens is not None:
            cond = cond + self.text_proj(text_tokens.mean(dim=1)).repeat_interleave(frames, dim=0)
        for block in self.blocks:
            x = block(x, cond)
        x = self.output_proj(self.output_norm(self.temporal(x, cond)))
        v = x.reshape(b, frames, c, h, w)
        return v.squeeze(0) if squeeze else v


class MeanFlowJetFlow(torch.nn.Module):
    """A hybrid: JetFlow's network trained by MeanFlow. Its backbone takes one time; the gap t - r reaches it
    as a learned shift of each latent channel (a Fourier embedding of the gap through a layer that starts at
    zero), so JetFlow's code is unchanged and the shift gets a gradient from the first step. With r = t the
    gap is zero and its embedding constant."""

    def __init__(self) -> None:
        super().__init__()
        self.jet = build_jetflow()
        self.backbone = self.jet.backbone
        self.gap_freqs = torch.nn.Parameter(torch.exp(-torch.arange(8.0) * math.log(1000.0) / 7), requires_grad=False)
        self.gap_proj = torch.nn.Linear(16, LATENT)
        torch.nn.init.zeros_(self.gap_proj.weight)
        torch.nn.init.zeros_(self.gap_proj.bias)

    def forward(self, z: Tensor, t: float | Tensor, text_tokens: Tensor | None = None,
                r: float | Tensor | None = None) -> Tensor:
        t_f = float(t)
        gap = torch.tensor([t_f - (t_f if r is None else float(r))], dtype=z.dtype)
        angles = gap.reshape(1, 1) * self.gap_freqs.unsqueeze(0) * (2.0 * math.pi)
        shift = self.gap_proj(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)).reshape(LATENT, 1, 1)
        out: Tensor = self.jet(z + shift, t_f, text_tokens=text_tokens if text_tokens is not None else TEXT)
        return out


def build(arm: str) -> torch.nn.Module:
    if arm == "fm-jetflow":
        return build_jetflow()
    if arm == "meanflow-jetflow":
        return MeanFlowJetFlow()
    if arm == "fm-unet":
        return DiffusionVideoUNet(unet_config())
    if arm == "meanflow-unet":
        return MeanFlowUNet(unet_config())
    raise ValueError(arm)


# ---------------------------------------------------------------- training

def train_flow_matching(model: torch.nn.Module, latents: Tensor, seconds: float, lr: float = 1e-3,
                        seed: int = 0) -> dict[str, Any]:
    """Flow matching, one clip per step, t uniform in (0, 1) (the repository's trainer uses [0.05, 0.95]; every
    sampler here starts at t = 1, and MeanFlow trains up to it, so the full range is used)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        x = latents[int(rng.integers(len(latents)))].unsqueeze(0)
        e = torch.randn_like(x)
        t = float(rng.uniform(1e-3, 1.0))
        loss = F.mse_loss(model(((1 - t) * x + t * e), t, text_tokens=TEXT), e - x)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        losses.append(float(loss.detach()))
    return {"steps": len(losses), "seconds": round(time.monotonic() - t0, 1), "final_loss": _tail(losses)}


def _logit_normal(rng: np.random.Generator, mean: float = -0.4, std: float = 1.0) -> float:
    return float(1.0 / (1.0 + math.exp(-rng.normal(mean, std))))


def meanflow_loss(model: torch.nn.Module, x: Tensor, t: float, r: float, power: float = 1.0,
                  derivative: str = "jvp") -> Tensor:
    """MeanFlow: u(z_t, r, t) against sg(v - (t - r)(v . grad_z u + d_t u)), weighted by
    sg(1 / (||error||^2 + 0.001)^power), power 1 (the paper's adaptive weight)."""
    e = torch.randn_like(x)
    z = (1 - t) * x + t * e
    v = e - x
    tt = torch.tensor(t, dtype=x.dtype)
    rr = torch.tensor(r, dtype=x.dtype)

    def u_fn(z_: Tensor, r_: Tensor, t_: Tensor) -> Tensor:
        out: Tensor = model(z_, t_, text_tokens=TEXT, r=r_)
        return out
    if derivative == "jvp":
        u, dudt = torch.func.jvp(u_fn, (z, rr, tt), (v, torch.zeros_like(rr), torch.ones_like(tt)))
    else:                                   # finite difference along (v, 0, 1); the target is not differentiated
        u = u_fn(z, rr, tt)
        eps = 1e-3 if t + 1e-3 <= 1.0 else -1e-3
        with torch.no_grad():
            dudt = (u_fn(z + eps * v, rr, tt + eps) - u.detach()) / eps
    target = (v - (t - r) * dudt).detach()
    err = ((u - target) ** 2).sum()                      # the squared norm of the error, as in the paper
    weight = (1.0 / (err.detach() + 1e-3) ** power) if power else 1.0
    loss: Tensor = weight * err
    return loss


def train_meanflow(model: torch.nn.Module, latents: Tensor, seconds: float, lr: float = 1e-3, seed: int = 0,
                   share_unequal: float = 0.25) -> dict[str, Any]:
    """MeanFlow, one clip per step; (t, r) logit-normal(-0.4, 1), r = t on 75 % of steps (the paper's)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        x = latents[int(rng.integers(len(latents)))].unsqueeze(0)
        a, b = _logit_normal(rng), _logit_normal(rng)
        t, r = max(a, b), min(a, b)
        if rng.random() >= share_unequal:
            r = t
        loss = meanflow_loss(model, x, t, r, derivative="fd" if isinstance(model, MeanFlowJetFlow) else "jvp")
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        losses.append(float(loss.detach()))
    return {"steps": len(losses), "seconds": round(time.monotonic() - t0, 1), "final_loss": _tail(losses)}


def _tail(xs: list[float]) -> float | None:
    return round(float(np.mean(xs[-200:])), 5) if xs else None


# ---------------------------------------------------------------- sampling

class Counter:
    """Counts the network's calls during sampling (installed once per model; ``reset`` between uses)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.calls = 0
        inner = model.backbone if hasattr(model, "backbone") else model
        original = inner.forward

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return original(*args, **kwargs)
        inner.forward = counted  # type: ignore[method-assign]

    def reset(self) -> None:
        self.calls = 0


_COUNTERS: dict[int, Counter] = {}


def counter_for(model: torch.nn.Module) -> Counter:
    if id(model) not in _COUNTERS:
        _COUNTERS[id(model)] = Counter(model)
    counter = _COUNTERS[id(model)]
    counter.reset()
    return counter


def _noise(n: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, FRAMES, LATENT, SIZE // 2, SIZE // 2, generator=g)


def jet2_corrected_step(velocity: Callable[[Tensor, float], Tensor], z: Tensor, t: float, dt: float,
                        eps: float = 1e-3) -> Tensor:
    """The second-order step z + dt v + dt^2 / 2 a with a = dv/dt along the flow, by one probe in the
    direction of integration. JetFlow's ``JetFlowStepper`` computes (v - v_probe) / h, which is -dv/dt
    (found by the independent review of 26 November); this is the step it means to take. Same two calls."""
    v = velocity(z, t)
    h = (-eps if t - eps >= 0.0 else eps) if dt < 0 else (eps if t + eps <= 1.0 else -eps)
    a = (velocity(z + h * v, t + h) - v) / h
    out: Tensor = z + dt * v + 0.5 * dt * dt * a
    return out


@torch.no_grad()
def sample(model: torch.nn.Module, how: str, steps: int, n: int, seed: int, noise: Tensor | None = None) -> Tensor:
    """``euler``; ``jet2`` (JetFlow's second-order step as packaged); ``jet2-corrected`` (the same step with
    the curvature's sign right); ``jetflow-sampler`` (JetFlow's latent velocity projection and its step as
    packaged: what ``generate_video`` does, from shared noise); ``jetflow-sampler-corrected``; ``meanflow``.
    One clip at a time; returns (n, T, C, H, W) latents."""
    model.eval()
    out = []
    ts = torch.linspace(1.0, 0.0, steps + 1).tolist()
    stepper = JetFlowStepper(JetFlowConfig(order=2, num_steps=steps))

    def plain(z_: Tensor, t_: float) -> Tensor:
        v: Tensor = model(z_, t_, text_tokens=TEXT)
        return v

    def projected(z_: Tensor, t_: float) -> Tensor:
        v: Tensor = NOETHER.project_spatial_latent_velocity(model(z_, t_, text_tokens=TEXT))[0]
        return v
    for z0 in (_noise(n, seed) if noise is None else noise):
        z = z0.clone()
        for k in range(steps):
            t, t_next = ts[k], ts[k + 1]
            dt = t_next - t
            if how == "euler":
                z = z + dt * plain(z, t)
            elif how == "jet2":
                z, _, _ = stepper.step(plain, z, t, dt)
            elif how == "jet2-corrected":
                z = jet2_corrected_step(plain, z, t, dt)
            elif how == "jetflow-sampler":
                z, _, _ = stepper.step(projected, z, t, dt)
            elif how == "jetflow-sampler-corrected":
                z = jet2_corrected_step(projected, z, t, dt)
            elif how == "meanflow":
                z = z + dt * model(z, t, text_tokens=TEXT, r=t_next)
            else:
                raise ValueError(how)
        out.append(z)
    return torch.stack(out)


# ---------------------------------------------------------------- data and metrics

VAE = ContinuousWaveletVAE(latent_channels=LATENT, downsample_factor=2)


def encode(clips: np.ndarray) -> Tensor:
    """(N, T, 3, H, W) RGB in [0, 1] -> (N, T, C, H/2, W/2) latents."""
    x = torch.from_numpy(np.asarray(clips, dtype=np.float32))
    return torch.stack([torch.stack([VAE.encode_frame(f) for f in clip]) for clip in x])


def decode(latents: Tensor) -> np.ndarray:
    frames = [[VAE.decode_frame(f, target_shape=(SIZE, SIZE)).detach().numpy() for f in clip] for clip in latents]
    return np.clip(np.asarray(frames, dtype=np.float64), 0.0, 1.0)


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    w, q = np.linalg.eigh((a + a.T) / 2)
    out: np.ndarray = (q * np.sqrt(np.clip(w, 0, None))) @ q.T
    return out


def frechet_pca(train: np.ndarray, held: np.ndarray, gen: np.ndarray, k: int = 16) -> float:
    """Frechet distance of ``gen`` to ``held`` in the first k principal components of ``train``."""
    tr = train.reshape(len(train), -1)
    mean = tr.mean(0)
    _, _, vt = np.linalg.svd(tr - mean, full_matrices=False)
    basis = vt[:k].T

    def feats(a: np.ndarray) -> np.ndarray:
        out: np.ndarray = (a.reshape(len(a), -1) - mean) @ basis
        return out
    f1, f2 = feats(held), feats(gen)
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    s1 = _sqrtm_psd(c1)
    cross = _sqrtm_psd(s1 @ c2 @ s1)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.trace(cross))


def memorisation_ratio(train: np.ndarray, held: np.ndarray, gen: np.ndarray) -> float:
    """Mean distance of a generated clip to its nearest training clip, over the same for held-out real
    clips; well below 1 means the model copies its training clips."""
    tr = torch.from_numpy(np.asarray(train, dtype=np.float64).reshape(len(train), -1))

    def nearest(a: np.ndarray) -> float:
        d = torch.cdist(torch.from_numpy(np.asarray(a, dtype=np.float64).reshape(len(a), -1)), tr)
        return float(d.min(dim=1).values.mean())
    return nearest(gen) / max(1e-12, nearest(held))


def temporal_roughness(clips: np.ndarray) -> float:
    """Mean |second difference in time| (the repository's temporal-consistency instrument)."""
    return float(np.abs(clips[:, 2:] - 2 * clips[:, 1:-1] + clips[:, :-2]).mean())


def toy_clips(kind: str, seeds: Iterable[int]) -> np.ndarray:
    if kind == "fluid":
        from apedp_dyn.video.fluid_domain import make_fluid_clip as make
    elif kind == "collision":
        from apedp_dyn.video.collision_domain import make_collision_clip as make
    else:
        raise ValueError(kind)
    return np.stack([np.asarray(make(s, t=FRAMES, h=SIZE, w=SIZE), dtype=np.float32) for s in seeds])


def video_windows(path: Path, windows: int = 4, stride: int = 3) -> list[np.ndarray]:
    """Up to ``windows`` evenly spaced clips of FRAMES frames, every ``stride``-th frame, resized to
    SIZE x SIZE by area, RGB in [0, 1], as (T, 3, H, W)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        frames.append(small[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
    cap.release()
    span = (FRAMES - 1) * stride + 1
    if len(frames) < span:
        return []
    starts = np.linspace(0, len(frames) - span, windows).round().astype(int)
    return [np.stack(frames[s:s + span:stride]) for s in dict.fromkeys(starts.tolist())]


def video_folder(root: Path, split: str) -> tuple[np.ndarray, list[str]]:
    """All windows of the ``.avi`` files under ``root/split`` (sorted), and the file each came from."""
    clips, names = [], []
    for path in sorted((root / split).rglob("*.avi")):
        for clip in video_windows(path):
            clips.append(clip)
            names.append(path.name)
    return np.stack(clips), names


def evaluate(model: torch.nn.Module, how: str, steps: int, n: int, seed: int, train: np.ndarray,
             held: np.ndarray) -> dict[str, Any]:
    counter = counter_for(model)
    t0 = time.monotonic()
    gen = decode(sample(model, how, steps, n, seed))
    took = time.monotonic() - t0
    return {"sampler": how, "steps": steps, "calls_per_clip": round(counter.calls / n, 2),
            "seconds_per_clip": round(took / n, 4), "frechet_pca": round(frechet_pca(train, held, gen), 4),
            "memorisation_ratio": round(memorisation_ratio(train, held, gen), 4),
            "temporal_roughness": round(temporal_roughness(gen), 5)}


Trainer = Callable[..., dict[str, Any]]
