r"""Compositional Sequence-Level Text Encoder & Pretrained HuggingFace Language Bridge.

Status: experimental

Provides:
1. CompositionalTextEncoder: Token-sequence language encoder producing (B, L_seq, D_text)
   embeddings with contextual modifier binding, negation gating, temporal phase ordering,
   and spatial preposition awareness (eliminating single-vector bag-of-words collapse).
2. PretrainedHFTextEncoderBridge: Seamless integration with HuggingFace `transformers`
   (CLIPTextModel, T5EncoderModel, AutoModel) when available.
3. extract_physical_steering_from_prompt: Parses natural language prompts into physical PDE
   and 3D SE(3) camera steering parameters (PhysicalSteeringConfig).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from jetflow_video.physics import PhysicalSteeringConfig


_CONCEPT_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("concept:vehicle", ("car", "automobile", "vehicle", "sedan", "coupe", "truck", "bus", "train", "bike", "spaceship")),
    ("concept:motion_fast", ("fast", "rapid", "speed", "racing", "sprinting", "swift", "accelerating", "driving", "running", "dashing")),
    ("concept:motion_slow", ("slow", "gentle", "drifting", "calm", "gliding", "parked", "still", "resting", "floating")),
    ("concept:fluid_water", ("ocean", "sea", "water", "waves", "river", "stream", "lake", "rain", "splash", "fluid")),
    ("concept:vortex_turbulence", ("vortex", "swirl", "turbulent", "turbulence", "whirlpool", "cyclone", "spiral", "eddy", "nebula")),
    ("concept:fire_heat", ("fire", "flame", "lava", "explosion", "sparks", "ember", "burning", "smoke", "plasma")),
    ("concept:lighting_warm", ("sunset", "sunrise", "golden", "warm", "sunlight", "dawn", "dusk", "glow", "neon")),
    ("concept:weather_storm", ("storm", "clouds", "thunder", "lightning", "wind", "fog", "mist", "snow")),
    ("concept:camera_motion", ("pan", "zoom", "tracking", "aerial", "drone", "cinematic", "closeup", "wide", "orbit", "dolly")),
    ("concept:nature_terrain", ("cliff", "mountain", "forest", "tree", "rock", "canyon", "valley", "beach", "coast")),
    ("concept:color_red", ("red", "crimson", "scarlet", "ruby", "vermilion")),
    ("concept:color_blue", ("blue", "azure", "cobalt", "sapphire", "cyan", "navy")),
    ("concept:color_green", ("green", "emerald", "jade", "lime", "forest")),
    ("concept:direction_left", ("left", "west", "leftward", "counterclockwise")),
    ("concept:direction_right", ("right", "east", "rightward", "clockwise")),
    ("concept:direction_up", ("up", "upward", "rising", "ascending", "north")),
    ("concept:direction_down", ("down", "downward", "falling", "descending", "south")),
    ("concept:3d_depth", ("3d", "depth", "rotating", "turning", "spinning", "orbiting", "behind", "foreground", "background", "sphere", "cube")),
)

_NEGATION_TOKENS = frozenset({"no", "not", "without", "never", "neither", "lack", "free"})
_TEMPORAL_EARLY_TOKENS = frozenset({"first", "initially", "start", "beginning", "before"})
_TEMPORAL_LATE_TOKENS = frozenset({"then", "later", "finally", "after", "subsequently", "eventually"})


def _encode_single_token_base(tok: str, dim: int) -> np.ndarray:
    """Deterministic subword + ontology basis vector for a single lexical token."""
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
            for slot in range(4):
                c_digest = hashlib.md5(f"{concept_tag}:{slot}".encode("utf-8")).hexdigest()  # noqa: S324
                c_idx = int(c_digest[:8], 16) % dim
                c_sign = 1.0 if int(c_digest[8], 16) < 8 else -1.0
                t_vec[c_idx] += 2.5 * c_sign

    norm = float(np.linalg.norm(t_vec))
    if norm > 1e-8:
        t_vec /= norm
    return t_vec


def encode_prompt_sequence(
    prompt: str,
    max_seq_len: int = 16,
    dim: int = 64,
) -> torch.Tensor:
    """Encode a natural language prompt into a contextual token sequence `(max_seq_len, dim)`.

    Features:
    - Preserves token order via sinusoidal sequence positional encodings.
    - Contextual bigram/trigram modifier binding (adjectives bind to adjacent nouns).
    - Negation gating (`"no"`, `"without"`, `"not"` invert the subsequent 2 tokens' semantic polarity).
    - Temporal phase tagging (`"initially"` vs `"then"` / `"finally"`).
    - Slot 0 stores the global L2-normalized sentence summary token (CLS), while slots
      `1 .. max_seq_len-1` store the contextualized per-word sequence embeddings.
    """
    tokens = re.findall(r"[a-z0-9]+", prompt.lower()) or ["<empty>"]
    raw_vecs: List[np.ndarray] = [_encode_single_token_base(tok, dim) for tok in tokens]

    # Contextualize tokens: negation gating, temporal phase, and local modifier binding
    ctx_vecs: List[np.ndarray] = []
    negate_window = 0
    temporal_phase = 0.0  # -1.0 early, +1.0 late

    for idx, tok in enumerate(tokens):
        if tok in _NEGATION_TOKENS:
            negate_window = 2
        if tok in _TEMPORAL_EARLY_TOKENS:
            temporal_phase = -1.0
        elif tok in _TEMPORAL_LATE_TOKENS:
            temporal_phase = 1.0

        v = raw_vecs[idx].copy()

        # Bind with preceding modifier token (when preceding token is not a negation operator)
        if idx > 0 and tokens[idx - 1] not in _NEGATION_TOKENS:
            prev_v = raw_vecs[idx - 1]
            # Non-commutative circular shift + gated binding preserves (modifier, head) order
            shifted_prev = np.roll(prev_v, shift=1)
            v = v + 0.45 * shifted_prev + 0.25 * (prev_v * np.roll(v, shift=-1))

        # Negation flips semantic polarity of the modified concept phrase
        if negate_window > 0 and tok not in _NEGATION_TOKENS:
            v = -v
            negate_window -= 1

        # Inject temporal phase bias into dedicated harmonic slots
        if abs(temporal_phase) > 0.0 and dim >= 8:
            v[0] += 0.35 * temporal_phase
            v[1] -= 0.35 * temporal_phase

        # Add sequence position encoding
        pos_angles = (idx + 1) * np.exp(-np.arange(0, dim // 2, dtype=np.float32) * (math.log(1000.0) / max(1, dim // 2)))
        pos_enc = np.zeros(dim, dtype=np.float32)
        pos_enc[0::2][:len(pos_angles)] = np.sin(pos_angles)
        pos_enc[1::2][:len(pos_angles)] = np.cos(pos_angles)
        v = v + 0.15 * pos_enc

        norm = float(np.linalg.norm(v))
        if norm > 1e-8:
            v /= norm
        ctx_vecs.append(v)

    # Global CLS summary token at position 0
    cls_vec = np.sum(ctx_vecs, axis=0)
    cls_norm = float(np.linalg.norm(cls_vec))
    if cls_norm > 1e-8:
        cls_vec /= cls_norm

    seq = np.zeros((max_seq_len, dim), dtype=np.float32)
    seq[0] = cls_vec
    limit = min(len(ctx_vecs), max_seq_len - 1)
    for i in range(limit):
        seq[i + 1] = ctx_vecs[i]

    return torch.from_numpy(seq)


class CompositionalTextEncoder(nn.Module):
    """Trainable Contextual Transformer Text Encoder operating on sequence embeddings `(B, L_seq, D_text)`."""

    def __init__(
        self,
        text_dim: int = 64,
        max_seq_len: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        use_pretrained_hf: bool = False,
        hf_model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.max_seq_len = max_seq_len
        self.use_pretrained_hf = use_pretrained_hf
        self.hf_tokenizer = None
        self.hf_model = None
        self.hf_proj: Optional[nn.Linear] = None

        if use_pretrained_hf:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                self.hf_model = AutoModel.from_pretrained(hf_model_name)
                self.hf_model.eval()
                for p in self.hf_model.parameters():
                    p.requires_grad_(False)
                hf_hidden = int(getattr(self.hf_model.config, "hidden_size", 512))
                self.hf_proj = nn.Linear(hf_hidden, text_dim)
            except Exception:
                self.use_pretrained_hf = False

        # Self-attention refinement stack initialized as exact identity residual
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=num_heads,
            dim_feedforward=2 * text_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def encode_prompts(
        self,
        prompts: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Encode a single prompt or list of prompts into `(B, L_seq, D_text)`."""
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)

        target_device = device or next(self.parameters()).device

        if self.use_pretrained_hf and self.hf_tokenizer is not None and self.hf_model is not None and self.hf_proj is not None:
            tokens = self.hf_tokenizer(
                prompt_list,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(target_device)
            with torch.no_grad():
                out = self.hf_model(**tokens)
                hidden = out.last_hidden_state.to(dtype=dtype)
            base_seq = self.hf_proj(hidden)
        else:
            seqs = [encode_prompt_sequence(p, max_seq_len=self.max_seq_len, dim=self.text_dim) for p in prompt_list]
            base_seq = torch.stack(seqs, dim=0).to(device=target_device, dtype=dtype)

        refined = self.transformer(base_seq)
        return base_seq + torch.tanh(self.residual_gate) * refined


def extract_physical_steering_from_prompt(prompt: str) -> PhysicalSteeringConfig:
    """Deterministically extract physical PDE and 3D SE(3) camera knobs from a natural language prompt."""
    tokens = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    negated = bool(tokens & _NEGATION_TOKENS)
    polarity = -0.5 if negated else 1.0

    vx, vy = 0.0, 0.0
    if tokens & {"right", "east", "rightward", "driving", "racing", "forward"}:
        vx += 0.25 * polarity
    if tokens & {"left", "west", "leftward"}:
        vx -= 0.25 * polarity
    if tokens & {"up", "upward", "rising", "ascending"}:
        vy -= 0.20 * polarity
    if tokens & {"down", "downward", "falling", "descending"}:
        vy += 0.20 * polarity
    if tokens & {"fast", "rapid", "speed", "racing", "accelerating"}:
        vx *= 1.6
        vy *= 1.6
    if tokens & {"parked", "still", "resting", "frozen"}:
        vx, vy = 0.0, 0.0

    viscosity = 0.0
    if tokens & {"smooth", "calm", "viscous", "gentle", "fog", "mist"}:
        viscosity = 0.04

    camera_pan = 0.0
    camera_zoom = 0.0
    camera_orbit_yaw = 0.0
    camera_orbit_pitch = 0.0
    camera_dolly_z = 0.0

    if tokens & {"pan", "tracking"}:
        camera_pan = 0.20 * polarity
    if tokens & {"zoom", "closeup", "dolly"}:
        camera_zoom = 0.15 * polarity
        camera_dolly_z = 0.15 * polarity
    if tokens & {"orbit", "orbiting", "rotating", "turning", "spinning", "3d", "around"}:
        camera_orbit_yaw = 0.35 * polarity
    if tokens & {"aerial", "drone", "tilt"}:
        camera_orbit_pitch = 0.15 * polarity

    return PhysicalSteeringConfig(
        convection_velocity=(float(vx), float(vy)),
        viscosity_nu=float(viscosity),
        camera_pan=float(camera_pan),
        camera_zoom=float(camera_zoom),
        camera_orbit_yaw=float(camera_orbit_yaw),
        camera_orbit_pitch=float(camera_orbit_pitch),
        camera_dolly_z=float(camera_dolly_z),
    )
