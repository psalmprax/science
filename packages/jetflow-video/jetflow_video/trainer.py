r"""JetFlow-Video PyTorch Native Training Engine for High-Dynamic Video Generation.

Enables full video model training on Google Colab (CPU and NVIDIA T4/A100/L4 GPUs):
1. VideoDataset: loads video files (.mp4, .mov, .avi), directory of images, or pre-extracted .pt latents.
2. 2nd-order Jet2 Curvature Flow Matching Loss with Convective Acceleration and Noether Divergence penalties.
3. Mixed-precision (AMP) accelerated training loop with AdamW and Cosine Annealing.
4. Automatic checkpointing and sample generation.
"""
from __future__ import annotations

import os
import glob
import math
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from jetflow_video.model import JetFlowVideoModel, JetFlowModelConfig
from jetflow_video.bridge import VideoVAEBridge


_CONCEPT_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("concept:vehicle", ("car", "automobile", "vehicle", "sedan", "coupe", "truck", "bus", "train", "bike")),
    ("concept:motion_fast", ("fast", "rapid", "speed", "racing", "sprinting", "swift", "accelerating", "driving", "running")),
    ("concept:motion_slow", ("slow", "gentle", "drifting", "calm", "gliding", "parked", "still", "resting")),
    ("concept:fluid_water", ("ocean", "sea", "water", "waves", "river", "stream", "lake", "rain", "splash", "fluid")),
    ("concept:vortex_turbulence", ("vortex", "swirl", "turbulent", "turbulence", "whirlpool", "cyclone", "spiral", "eddy")),
    ("concept:fire_heat", ("fire", "flame", "lava", "explosion", "sparks", "ember", "burning", "smoke")),
    ("concept:lighting_warm", ("sunset", "sunrise", "golden", "warm", "sunlight", "dawn", "dusk", "glow")),
    ("concept:weather_storm", ("storm", "clouds", "thunder", "lightning", "wind", "fog", "mist", "snow")),
    ("concept:camera_motion", ("pan", "zoom", "tracking", "aerial", "drone", "cinematic", "closeup", "wide")),
    ("concept:nature_terrain", ("cliff", "mountain", "forest", "tree", "rock", "canyon", "valley", "beach", "coast")),
)


def encode_prompt_deterministic(prompt: str, dim: int = 64) -> torch.Tensor:
    """Encode a text caption or clip identifier into a deterministic L2-normalized semantic vector."""
    tokens = re.findall(r"[a-z0-9]+", prompt.lower()) or ["<empty>"]
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        t_vec = np.zeros(dim, dtype=np.float32)
        grams = ["w:" + tok]
        padded = "|^" + tok + "^|"
        grams.extend("t:" + padded[i:i + 3] for i in range(len(padded) - 2))
        for g in grams:
            digest = hashlib.md5(g.encode("utf-8")).hexdigest()  # noqa: S324
            idx = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8], 16) < 8 else -1.0
            t_vec[idx] += sign
        for concept_tag, vocab in _CONCEPT_GROUPS:
            if tok in vocab or any(tok.startswith(v) for v in vocab if len(v) >= 4):
                for slot in range(3):
                    c_digest = hashlib.md5(f"{concept_tag}:{slot}".encode("utf-8")).hexdigest()  # noqa: S324
                    c_idx = int(c_digest[:8], 16) % dim
                    c_sign = 1.0 if int(c_digest[8], 16) < 8 else -1.0
                    t_vec[c_idx] += 2.5 * c_sign
        n = float(np.linalg.norm(t_vec))
        if n > 0:
            vec += t_vec / n
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return torch.from_numpy(vec)


class VideoDataset(Dataset):
    """Dataset for training JetFlow video model on raw videos, cached latents, or synthetic sequences."""

    def __init__(
        self,
        video_dir: Optional[str] = None,
        latent_dir: Optional[str] = None,
        hf_repo: Union[str, List[str], None] = None,
        max_hf_samples: int = 100,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (64, 64),
        num_synthetic_samples: int = 0,
    ) -> None:
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.video_files: List[str] = []
        self.latent_files: List[str] = []
        self.hf_files: List[str] = []
        self.parquet_records: List[Dict[str, Any]] = []
        self._ram_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.num_synthetic_samples = num_synthetic_samples

        if hf_repo:
            import urllib.request
            import json
            import io
            repos = [r.strip() for r in hf_repo.split(",")] if isinstance(hf_repo, str) else list(hf_repo)
            for repo_id in repos:
                try:
                    url = f"https://huggingface.co/api/datasets/{repo_id}"
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = json.loads(resp.read().decode())
                        siblings = [s["rfilename"] for s in data.get("siblings", [])]
                        parquet_files = [f for f in siblings if f.endswith(".parquet")]
                        pt_files = [f for f in siblings if f.endswith(".pt")]

                        if parquet_files:
                            try:
                                import pandas as pd
                                pq_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{parquet_files[0]}"
                                req_pq = urllib.request.Request(pq_url, headers={"User-Agent": "Mozilla/5.0"})
                                with urllib.request.urlopen(req_pq, timeout=25) as p_resp:
                                    df = pd.read_parquet(io.BytesIO(p_resp.read()))
                                    limit = max_hf_samples or len(df)
                                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
                                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
                                    for i in range(min(limit, len(df))):
                                        row = df.iloc[i]
                                        if "video_tensor" in row:
                                            v_tens = row["video_tensor"]
                                            c_list = []
                                            for ch in v_tens:
                                                t_list = []
                                                for frame in ch:
                                                    t_list.append(np.array([r for r in frame], dtype=np.float32))
                                                c_list.append(np.stack(t_list, axis=0))
                                            arr = np.stack(c_list, axis=0)  # (3, T, H, W)
                                            rgb = (arr * std + mean).clip(0.0, 1.0)
                                            t_tensor = torch.from_numpy(rgb).permute(1, 0, 2, 3)  # (T, 3, H, W)
                                            self.parquet_records.append({
                                                "video_rgb": t_tensor,
                                                "name": str(row.get("name", row.get("caption", f"video_{i}"))),
                                            })
                            except Exception as pe:
                                print(f"[VideoDataset] Failed to parse parquet from {repo_id}: {pe}")

                        if pt_files:
                            urls = [
                                f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
                                for f in pt_files
                            ]
                            self.hf_files.extend(urls)
                except Exception as e:
                    print(f"[VideoDataset] Failed to query HF repo {repo_id}: {e}")

            if max_hf_samples and len(self.hf_files) > max_hf_samples:
                self.hf_files = self.hf_files[:max_hf_samples]

        if latent_dir and os.path.isdir(latent_dir):
            self.latent_files = sorted(glob.glob(os.path.join(latent_dir, "*.pt")))

        if video_dir and os.path.isdir(video_dir):
            for ext in ("*.mp4", "*.avi", "*.mov", "*.gif"):
                self.video_files.extend(glob.glob(os.path.join(video_dir, ext)))
            self.video_files.sort()

    def __len__(self) -> int:
        if self.parquet_records:
            return len(self.parquet_records)
        if self.hf_files:
            return len(self.hf_files)
        if self.latent_files:
            return len(self.latent_files)
        if self.video_files:
            return len(self.video_files)
        return max(1, self.num_synthetic_samples)

    def _generate_synthetic_video(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a multi-scene 3D physical/vortex/occlusion video with deterministic sequence embeddings."""
        from jetflow_video.text_encoder import encode_prompt_sequence
        t_steps, h, w = self.num_frames, self.frame_size[0], self.frame_size[1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32),
            torch.linspace(-1, 1, w, dtype=torch.float32),
            indexing="ij",
        )
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        theta = torch.atan2(y, x)

        scene_type = idx % 4
        frames = []
        phase_offset = (idx * 0.3) % (2.0 * math.pi)
        for t in range(t_steps):
            tau = float(t) / max(1, t_steps - 1)
            phi = float(t) / float(t_steps) * 2.0 * math.pi + phase_offset
            if scene_type == 0:
                # Scene 0: Turbulent fluid vortex spiral
                ch0 = 0.5 + 0.5 * torch.sin(4.0 * r - phi)
                ch1 = 0.5 + 0.5 * torch.cos(theta * 2.0 + phi)
                ch2 = 0.5 + 0.5 * torch.sin(3.0 * x + 2.0 * y - 2.0 * phi)
                caption = f"turbulent fluid vortex wave spiral phase {idx % 8}"
            elif scene_type == 1:
                # Scene 1: 3D rotating sphere with foreground/background depth occlusion
                cx = 0.45 * math.cos(phi)
                cy = 0.25 * math.sin(phi)
                dx = x - cx
                dy = y - cy
                dist_sq = dx * dx + dy * dy
                radius_sq = 0.18
                fg_mask = torch.sigmoid((radius_sq - dist_sq) * 35.0)
                z_sphere = torch.sqrt(torch.clamp(radius_sq - dist_sq, min=0.0))
                # 3D Lambertian normal shading rotating in yaw
                light_x = math.cos(phi * 0.5)
                light_y = 0.3
                light_z = math.sin(phi * 0.5)
                shade = torch.clamp((dx * light_x + dy * light_y + z_sphere * light_z) / math.sqrt(radius_sq), 0.1, 1.0)
                bg0 = 0.3 + 0.2 * torch.cos(3.0 * x - 0.5 * phi)
                bg1 = 0.4 + 0.2 * torch.sin(3.0 * y + 0.5 * phi)
                bg2 = 0.6 - 0.2 * y
                ch0 = fg_mask * (0.85 * shade) + (1.0 - fg_mask) * bg0
                ch1 = fg_mask * (0.25 * shade) + (1.0 - fg_mask) * bg1
                ch2 = fg_mask * (0.20 * shade) + (1.0 - fg_mask) * bg2
                caption = f"3d rotating red sphere orbiting in foreground over blue background {idx % 8}"
            elif scene_type == 2:
                # Scene 2: Fast vehicle translating across coastal horizon at warm sunset
                car_x = -0.6 + 1.2 * tau
                car_y = 0.25
                car_mask = torch.exp(-((x - car_x) ** 2 / 0.04 + (y - car_y) ** 2 / 0.015))
                sky = torch.clamp(0.7 - 0.4 * y, 0.0, 1.0)
                waves = 0.5 + 0.3 * torch.sin(8.0 * x - 3.0 * phi) * torch.sigmoid(y * 10.0)
                ch0 = torch.clamp(0.8 * sky + 0.9 * car_mask, 0.0, 1.0)
                ch1 = torch.clamp(0.45 * sky + 0.3 * waves * (1.0 - car_mask), 0.0, 1.0)
                ch2 = torch.clamp(0.25 * sky + 0.7 * waves * (1.0 - car_mask), 0.0, 1.0)
                caption = f"fast red car driving right along ocean coast at golden sunset {idx % 8}"
            else:
                # Scene 3: 3D camera orbit and dolly zoom over mountain terrain
                zoom = 1.0 + 0.35 * tau
                ux = x / zoom + 0.2 * math.sin(phi * 0.5)
                uy = y / zoom
                ridge = torch.sin(3.5 * ux) * torch.cos(2.5 * uy)
                ch0 = 0.4 + 0.35 * ridge
                ch1 = 0.55 + 0.35 * torch.sin(2.0 * ux + uy)
                ch2 = 0.5 + 0.3 * torch.cos(3.0 * uy - phi * 0.5)
                caption = f"aerial drone camera zoom and orbit over green mountain valley {idx % 8}"

            frame = torch.stack([ch0, ch1, ch2], dim=0).clamp(0.0, 1.0)
            frames.append(frame)

        video = torch.stack(frames, dim=0)  # (T, 3, H, W)
        prompt_emb = encode_prompt_deterministic(caption, dim=64)
        prompt_seq = encode_prompt_sequence(caption, max_seq_len=8, dim=64)
        return video, prompt_emb, prompt_seq

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._ram_cache:
            return self._ram_cache[idx]

        from jetflow_video.text_encoder import encode_prompt_sequence

        if self.parquet_records:
            rec = self.parquet_records[idx]
            video_rgb = rec["video_rgb"]
            if len(video_rgb) != self.num_frames:
                indices = np.linspace(0, len(video_rgb) - 1, self.num_frames).astype(int)
                video_rgb = video_rgb[indices]
            if self.frame_size and (video_rgb.shape[2] != self.frame_size[0] or video_rgb.shape[3] != self.frame_size[1]):
                video_rgb = torch.nn.functional.interpolate(
                    video_rgb, size=self.frame_size, mode="bilinear", align_corners=False
                )
            caption = str(rec.get("name", f"video_clip_{idx}"))
            item = {
                "video_rgb": video_rgb,
                "prompt_embedding": encode_prompt_deterministic(caption, dim=64),
                "prompt_tokens": encode_prompt_sequence(caption, max_seq_len=8, dim=64),
            }
            self._ram_cache[idx] = item
            return item

        if self.hf_files:
            import urllib.request
            import io
            import time
            u = self.hf_files[idx]
            for attempt in range(3):
                try:
                    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        t = torch.load(io.BytesIO(resp.read()), map_location="cpu").float()
                        if t.ndim == 4 and t.shape[0] in (4, 12, 16) and t.shape[1] > t.shape[0]:
                            t = t.permute(1, 0, 2, 3)
                        if self.frame_size and (t.shape[2] != self.frame_size[0] or t.shape[3] != self.frame_size[1]):
                            t = torch.nn.functional.interpolate(
                                t, size=self.frame_size, mode="bilinear", align_corners=False
                            )
                        caption = os.path.splitext(os.path.basename(u))[0].replace("_", " ")
                        item = {
                            "latent": t,
                            "prompt_embedding": encode_prompt_deterministic(caption, dim=64),
                            "prompt_tokens": encode_prompt_sequence(caption, max_seq_len=8, dim=64),
                        }
                        self._ram_cache[idx] = item
                        return item
                except Exception:
                    if attempt < 2:
                        time.sleep(0.3)
                    else:
                        video_rgb, prompt_emb, prompt_seq = self._generate_synthetic_video(idx)
                        item = {
                            "video_rgb": video_rgb,
                            "prompt_embedding": prompt_emb,
                            "prompt_tokens": prompt_seq,
                        }
                        self._ram_cache[idx] = item
                        return item

        if self.latent_files:
            latent_path = self.latent_files[idx]
            data = torch.load(latent_path, map_location="cpu")
            lat = data["latent"]
            if self.frame_size and (lat.shape[2] != self.frame_size[0] or lat.shape[3] != self.frame_size[1]):
                lat = torch.nn.functional.interpolate(
                    lat, size=self.frame_size, mode="bilinear", align_corners=False
                )
            default_caption = str(data.get("caption", os.path.splitext(os.path.basename(latent_path))[0].replace("_", " ")))
            item = {
                "latent": lat,  # (T, C, H', W')
                "prompt_embedding": data.get("prompt_embedding", encode_prompt_deterministic(default_caption, dim=64)),
                "prompt_tokens": data.get("prompt_tokens", encode_prompt_sequence(default_caption, max_seq_len=8, dim=64)),
            }
            self._ram_cache[idx] = item
            return item

        if self.video_files:
            try:
                import imageio.v3 as iio
                video_path = self.video_files[idx]
                raw_frames = iio.imread(video_path)
                # Ensure (T, H, W, 3)
                if raw_frames.ndim == 4:
                    t_total = raw_frames.shape[0]
                    # Uniform sampling of num_frames
                    indices = np.linspace(0, t_total - 1, self.num_frames).astype(int)
                    sampled = raw_frames[indices]
                    # Convert to PyTorch (T, 3, H, W) normalized [0, 1]
                    tensor = torch.from_numpy(sampled).permute(0, 3, 1, 2).float() / 255.0
                    if tensor.shape[2] != self.frame_size[0] or tensor.shape[3] != self.frame_size[1]:
                        tensor = torch.nn.functional.interpolate(
                            tensor, size=self.frame_size, mode="bilinear", align_corners=False
                        )
                    caption = os.path.splitext(os.path.basename(video_path))[0].replace("_", " ").replace("-", " ")
                    item = {
                        "video_rgb": tensor,
                        "prompt_embedding": encode_prompt_deterministic(caption, dim=64),
                        "prompt_tokens": encode_prompt_sequence(caption, max_seq_len=8, dim=64),
                    }
                    self._ram_cache[idx] = item
                    return item
            except Exception:
                pass  # Fall back to synthetic

        # Default: procedural multi-scene synthetic video
        video_rgb, prompt_emb, prompt_seq = self._generate_synthetic_video(idx)
        item = {
            "video_rgb": video_rgb,
            "prompt_embedding": prompt_emb,
            "prompt_tokens": prompt_seq,
        }
        self._ram_cache[idx] = item
        return item


class JetFlowTrainer:
    """Trainer orchestrating 2nd-order Jet2 flow matching optimization with CFG Dropout & EMA."""

    def __init__(
        self,
        model: JetFlowVideoModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        cfg_dropout_prob: float = 0.1,
        ema_decay: float = 0.995,
        warmup_steps: int = 10,
        max_steps: int = 1000,
        grad_accum_steps: int = 1,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = torch.cuda.is_available(),
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.ema_decay = float(ema_decay)
        self.warmup_steps = max(0, int(warmup_steps))
        self.max_steps = max(1, int(max_steps))
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.global_step = 0
        self.base_lr = float(learning_rate)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Exponential Moving Average (EMA) shadow parameters
        self.ema_state: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        self._backup_state: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update Exponential Moving Average shadow weights."""
        decay = self.ema_decay
        for k, v in self.model.state_dict().items():
            if k in self.ema_state and v.is_floating_point():
                self.ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                self.ema_state[k] = v.detach().clone()

    @torch.no_grad()
    def apply_ema_weights(self) -> None:
        """Swap EMA shadow weights into the active model for evaluation/generation."""
        self._backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema_state, strict=False)

    @torch.no_grad()
    def restore_online_weights(self) -> None:
        """Restore online training weights after EMA evaluation."""
        if self._backup_state is not None:
            self.model.load_state_dict(self._backup_state, strict=False)
            self._backup_state = None

    def _update_learning_rate(self) -> float:
        """Linear warmup followed by cosine annealing learning rate schedule."""
        step = self.global_step
        if self.warmup_steps > 0 and step < self.warmup_steps:
            scale = float(step + 1) / float(self.warmup_steps)
        else:
            progress = min(1.0, float(step - self.warmup_steps) / max(1.0, float(self.max_steps - self.warmup_steps)))
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.base_lr * scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def compute_loss(
        self,
        z_data: torch.Tensor,
        prompt_embedding: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        r"""Compute 2nd-order Jet2 curvature flow matching loss with CFG conditioning dropout.

        Loss = L_flow(v_theta, v*) + lambda_acc * L_acc(a_theta) + lambda_noether * L_div(\nabla \cdot v)
        """
        b, t_dim, c, h, w = z_data.shape
        device = z_data.device

        # Sample prior noise z_1 ~ N(0, I)
        z_noise = torch.randn_like(z_data)

        # Sample flow time t ~ U(0.05, 0.95)
        t = torch.rand(b, device=device) * 0.9 + 0.05

        # Geodesic flow interpolation: z_t = (1 - t) * z_data + t * z_noise
        t_bc = t.view(b, 1, 1, 1, 1)
        z_t = (1.0 - t_bc) * z_data + t_bc * z_noise
        target_v = z_noise - z_data

        # Normalize text_tokens to (B, L_seq, D_text) and apply Classifier-Free Guidance dropout
        text_tokens: Optional[torch.Tensor] = None
        if prompt_embedding is not None:
            if not (self.model.training and self.cfg_dropout_prob > 0.0 and torch.rand(1).item() < self.cfg_dropout_prob):
                if prompt_embedding.ndim == 2:
                    text_tokens = prompt_embedding.unsqueeze(1)
                elif prompt_embedding.ndim == 3:
                    text_tokens = prompt_embedding
                else:
                    text_tokens = prompt_embedding.view(b, 1, -1)

        # Predict conditional (or dropped unconditional) velocity v_theta
        pred_v = self.model(z_t, t, text_tokens=text_tokens)

        # 1. Flow matching loss
        flow_loss = F.mse_loss(pred_v, target_v)

        # 2. Convective material acceleration loss along the characteristic ray (z_t + dt * v, t + dt)
        # Along a straight geodesic z_t = (1 - t) z_data + t z_noise, total material acceleration is zero.
        if self.model.config.lambda_acc > 0:
            dt_eps = 0.01
            t_plus = torch.clamp(t + dt_eps, 0.0, 1.0)
            z_plus = z_t + dt_eps * pred_v.detach()
            with torch.no_grad():
                pred_v_target = self.model(z_plus, t_plus, text_tokens=text_tokens)
            material_acc = (pred_v_target - pred_v) / dt_eps
            acc_loss = torch.mean(material_acc ** 2)
        else:
            acc_loss = torch.tensor(0.0, device=device)

        # 3. Noether solenoidal divergence penalty
        div_loss = self.model.noether_engine.compute_divergence_loss(pred_v)

        # Total multi-objective loss
        total_loss = (
            flow_loss
            + self.model.config.lambda_acc * acc_loss
            + self.model.config.lambda_noether * div_loss
        )

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "acc_loss": acc_loss,
            "div_loss": div_loss,
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int = 1) -> Dict[str, float]:
        """Train for one full epoch over the dataset with EMA and cosine warmup."""
        self.model.train()
        total_losses = []
        flow_losses = []
        acc_losses = []
        div_losses = []

        self.optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        for step_idx, batch in enumerate(pbar):
            self._update_learning_rate()

            # Latents can either be pre-encoded or encoded on-the-fly via VAE
            if "latent" in batch:
                z_data = batch["latent"].to(self.device)
            else:
                videos_rgb = batch["video_rgb"].to(self.device)  # (B, T, 3, H, W)
                with torch.no_grad():
                    b, t_dim = videos_rgb.shape[:2]
                    v_flat = videos_rgb.view(b * t_dim, *videos_rgb.shape[2:])
                    lat_flat = self.model.vae_bridge.vae.encode_frame(v_flat)
                    z_data = lat_flat.view(b, t_dim, *lat_flat.shape[1:])

            # Prefer multi-token sequence embeddings when available
            if "prompt_tokens" in batch:
                prompt_emb = batch["prompt_tokens"].to(self.device)
            else:
                prompt_emb = batch["prompt_embedding"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    losses = self.compute_loss(z_data, prompt_emb)
                    scaled_loss = losses["total_loss"] / self.grad_accum_steps
                self.scaler.scale(scaled_loss).backward()
                if (step_idx + 1) % self.grad_accum_steps == 0 or (step_idx + 1) == len(dataloader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.update_ema()
                    self.global_step += 1
            else:
                losses = self.compute_loss(z_data, prompt_emb)
                scaled_loss = losses["total_loss"] / self.grad_accum_steps
                scaled_loss.backward()
                if (step_idx + 1) % self.grad_accum_steps == 0 or (step_idx + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.update_ema()
                    self.global_step += 1

            tot = losses["total_loss"].item()
            flo = losses["flow_loss"].item()
            acc = losses["acc_loss"].item()
            div = losses["div_loss"].item()

            total_losses.append(tot)
            flow_losses.append(flo)
            acc_losses.append(acc)
            div_losses.append(div)

            pbar.set_postfix({"total": f"{tot:.4f}", "flow": f"{flo:.4f}", "acc": f"{acc:.4f}"})

        return {
            "mean_total_loss": float(np.mean(total_losses)),
            "mean_flow_loss": float(np.mean(flow_losses)),
            "mean_acc_loss": float(np.mean(acc_losses)),
            "mean_div_loss": float(np.mean(div_losses)),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model and EMA checkpoint to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "ema_state_dict": self.ema_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.config,
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str, use_ema: bool = False) -> None:
        """Load model checkpoint from disk."""
        ckpt = torch.load(path, map_location=self.device)
        if use_ema and "ema_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["ema_state_dict"])
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            self.ema_state = ckpt["ema_state_dict"]
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = int(ckpt.get("global_step", 0))
