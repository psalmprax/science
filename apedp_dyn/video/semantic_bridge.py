r"""Semantic Conditioning Bridge for Continuous Video Flow Matching.

Status: experimental

Connects the continuous spatio-temporal PDE velocity field to text prompt embeddings:
1. Continuous Fourier Time Embedding: e_t = [sin(\omega_k t), cos(\omega_k t)].
2. AdaLN-Zero Latent Modulation: modulates spatial latent channels with scale, shift, and gate.
3. Multi-Head Cross-Attention: attends from spatio-temporal video tokens to text tokens.
4. Classifier-Free Flow Guidance: steers the continuous trajectory toward text intent:
   v_guided(z, t, c) = v(z, t, \emptyset) + s * (v(z, t, c) - v(z, t, \emptyset)).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemanticConditioningConfig:
    """Configuration for Semantic Video Conditioning Bridge."""
    latent_dim: int = 16            # Number of latent channels per token (e.g. 16 or 64)
    text_dim: int = 64              # Text prompt embedding dimension
    guidance_scale: float = 3.0     # Classifier-free flow guidance scale s >= 1.0
    num_heads: int = 4              # Cross-attention heads
    fourier_frequencies: int = 16   # Number of temporal Fourier frequency bases


class SemanticVideoConditioner:
    r"""Cross-attention and AdaLN conditioning bridge for continuous video flow fields."""

    def __init__(
        self,
        config: SemanticConditioningConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or SemanticConditioningConfig()
        rng = np.random.default_rng(seed)

        d = self.config.latent_dim
        d_txt = self.config.text_dim
        n_freq = self.config.fourier_frequencies

        # Fixed geometric frequency scales for continuous time: \omega_k = \exp(-k * \ln(10000) / n_freq)
        self.freq_scales = np.exp(-np.arange(n_freq, dtype=np.float64) * (np.log(10000.0) / max(1, n_freq - 1)))

        # Time projection: maps 2 * n_freq -> d
        self.w_time = rng.normal(0, np.sqrt(2.0 / (2 * n_freq)), size=(2 * n_freq, d))
        self.b_time = np.zeros(d, dtype=np.float64)

        # Text pooled projection: maps d_txt -> d
        self.w_text = rng.normal(0, np.sqrt(2.0 / d_txt), size=(d_txt, d))
        self.b_text = np.zeros(d, dtype=np.float64)

        # AdaLN-Zero projection: maps d -> 6 * d (scale1, shift1, gate1, scale2, shift2, gate2)
        # Initialized with zero weights on gates to ensure identity mapping at initialization
        self.w_adaln = rng.normal(0, 0.02, size=(d, 6 * d))
        self.b_adaln = np.zeros(6 * d, dtype=np.float64)

        # Multi-Head Cross-Attention projections
        self.w_q = rng.normal(0, np.sqrt(2.0 / d), size=(d, d))
        self.w_k = rng.normal(0, np.sqrt(2.0 / d_txt), size=(d_txt, d))
        self.w_v = rng.normal(0, np.sqrt(2.0 / d_txt), size=(d_txt, d))
        self.w_out = rng.normal(0, np.sqrt(2.0 / d), size=(d, d))

    def embed_continuous_time(self, t: float) -> np.ndarray:
        r"""Compute continuous Fourier temporal embedding for normalized time t in [0, 1].

        Returns:
        - 1D vector of shape (latent_dim,).
        """
        t_val = float(np.clip(t, 0.0, 1.0))
        angles = t_val * self.freq_scales * 2.0 * np.pi
        fourier = np.concatenate([np.sin(angles), np.cos(angles)])  # shape (2 * n_freq,)
        return fourier @ self.w_time + self.b_time

    def embed_text_prompt(self, prompt: str, length: int = 8) -> np.ndarray:
        r"""Create a deterministic, process-stable text embedding from an input prompt string.

        Method: ``feature_hash_word_char_trigram_v1`` — the established hashing
        trick (Weinberger et al., 2009) over word and character-n-gram tokens.
        Each token's char-trigrams are hashed to stable signed indices via MD5
        (NOT the Python builtin ``hash()``, which is salted per process via
        PYTHONHASHSEED and made the old embeddings non-reproducible across
        runs). A small sentence-context term is added to every token row so
        that prompts sharing vocabulary embed near each other; rows are
        L2-normalized.

        These are *lexical* features, not CLIP/MiniLM-scale semantics: for
        real semantic conditioning, pass precomputed embedding arrays INTO
        the model (cross-attention consumes any (L, text_dim) tokens).

        Returns:
        - 2D array of shape (length, text_dim).
        """
        tokens = re.findall(r"[a-z0-9]+", prompt.lower()) or ["<empty>"]

        per_token = np.stack([self._token_feature(tok, self.config.text_dim) for tok in tokens], axis=0)
        # Sentence-context term: pooled lexical feature of the whole prompt
        context = per_token.mean(axis=0)

        rows = np.zeros((length, self.config.text_dim), dtype=np.float64)
        for j in range(length):
            tok_vec = per_token[j % len(tokens)]
            # Deterministic, process-stable construction: recurring rows cycle
            # token features with a descending context weight (no RNG).
            w_ctx = 0.15 if j < len(tokens) else 0.05 * (1.0 - j / max(1, length))
            rows[j] = tok_vec + w_ctx * context

        # Normalize across embedding dimension
        norms = np.linalg.norm(rows, axis=-1, keepdims=True)
        return rows / np.maximum(norms, 1e-8)

    @property
    def embedding_method(self) -> str:
        """Provenance tag of the text featurization in use."""
        return "feature_hash_word_char_trigram_v1"

    _SEMANTIC_CONCEPT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
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

    @classmethod
    def _token_feature(cls, token: str, dim: int) -> np.ndarray:
        r"""Deterministic hashing-trick + semantic concept group feature for one token."""
        vec = np.zeros(dim, dtype=np.float64)
        grams = ["w:" + token]
        padded = "|^" + token + "^|"
        grams.extend("t:" + padded[i:i + 3] for i in range(len(padded) - 2))
        for g in grams:
            digest = hashlib.md5(g.encode("utf-8")).hexdigest()  # noqa: S324 - MD5 for process-stable feature-index bucketing (documented); not security-sensitive
            idx = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8], 16) < 8 else -1.0
            vec[idx] += sign

        # Project synonymous physical/visual concepts onto shared semantic basis dimensions
        for concept_tag, vocab in cls._SEMANTIC_CONCEPT_GROUPS:
            if token in vocab or any(token.startswith(v) for v in vocab if len(v) >= 4):
                for slot in range(3):
                    c_digest = hashlib.md5(f"{concept_tag}:{slot}".encode("utf-8")).hexdigest()  # noqa: S324
                    c_idx = int(c_digest[:8], 16) % dim
                    c_sign = 1.0 if int(c_digest[8], 16) < 8 else -1.0
                    vec[c_idx] += 2.5 * c_sign

        n = float(np.linalg.norm(vec))
        return vec / n if n > 0 else vec

    def cross_attention(
        self,
        z_tokens: np.ndarray,
        text_tokens: np.ndarray,
    ) -> np.ndarray:
        r"""Multi-Head Cross-Attention from spatial latent tokens to text tokens.

        Parameters:
        - z_tokens: (N_spatial, latent_dim)
        - text_tokens: (L_text, text_dim)

        Returns:
        - (N_spatial, latent_dim) context features.
        """
        n_spatial, d = z_tokens.shape
        l_text, _d_txt = text_tokens.shape
        n_heads = self.config.num_heads
        d_head = d // n_heads

        # Projections
        q = (z_tokens @ self.w_q).reshape(n_spatial, n_heads, d_head)
        k = (text_tokens @ self.w_k).reshape(l_text, n_heads, d_head)
        v = (text_tokens @ self.w_v).reshape(l_text, n_heads, d_head)

        # Scale factor: 1 / sqrt(d_head)
        scale = 1.0 / np.sqrt(max(1, d_head))

        # Compute multi-head attention via batched BLAS matmul: (H, N, d_head) @ (H, d_head, L)
        q_b = q.transpose(1, 0, 2)  # (H, N, d_head)
        k_b = k.transpose(1, 2, 0)  # (H, d_head, L)
        v_b = v.transpose(1, 0, 2)  # (H, L, d_head)

        scores = (q_b @ k_b) * scale  # (H, N, L)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.maximum(np.sum(exp_scores, axis=-1, keepdims=True), 1e-12)
        out_heads = (attn_weights @ v_b).transpose(1, 0, 2)  # (N, H, d_head)

        out_flat = out_heads.reshape(n_spatial, d)
        return out_flat @ self.w_out

    def compute_conditioned_velocity(
        self,
        z_latent: np.ndarray,
        t: float,
        text_tokens: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Evaluate the continuous velocity field v(z, t, c) conditioned on text and time.

        Parameters:
        - z_latent: 4D video latent (T, C, H, W) or 2D token array (N, C).
        - t: continuous time coordinate in [0, 1].
        - text_tokens: optional text token array (L, text_dim). If None, evaluates unconditionally.

        Returns:
        - Continuous velocity array matching z_latent shape.
        """
        orig_shape = z_latent.shape
        d = self.config.latent_dim

        # Flatten spatial dimensions into token matrix (N_tokens, latent_dim)
        if z_latent.ndim == 4:
            t_dim, c_dim, h_dim, w_dim = orig_shape
            z_flat = z_latent.transpose(0, 2, 3, 1).reshape(-1, c_dim)
        elif z_latent.ndim == 2:
            z_flat = z_latent.copy()
        else:
            z_flat = z_latent.reshape(-1, d)

        # 1. Temporal embedding
        e_time = self.embed_continuous_time(t)

        # 2. Text embedding
        if text_tokens is not None and len(text_tokens) > 0:
            # Mean-pooled text vector
            text_pooled = np.mean(text_tokens, axis=0)
            e_text = text_pooled @ self.w_text + self.b_text
            cond_vector = e_time + e_text
        else:
            cond_vector = e_time

        # 3. AdaLN modulation parameters: 6 chunks of size d
        # (scale1, shift1, gate1, scale2, shift2, gate2)
        adaln_params = cond_vector @ self.w_adaln + self.b_adaln
        s1, sh1, g1, s2, sh2, g2 = np.split(adaln_params, 6)

        # 4. First block: modulated normalization
        norm_z = (z_flat - np.mean(z_flat, axis=-1, keepdims=True)) / (
            np.std(z_flat, axis=-1, keepdims=True) + 1e-6
        )
        modulated_z = norm_z * (1.0 + s1) + sh1

        # 5. Cross-attention if text tokens provided
        if text_tokens is not None and len(text_tokens) > 0:
            attn_out = self.cross_attention(modulated_z, text_tokens)
            h_state = modulated_z + np.tanh(g1) * attn_out
        else:
            h_state = modulated_z

        # 6. Second block: non-linear continuous flow dynamics
        norm_h = (h_state - np.mean(h_state, axis=-1, keepdims=True)) / (
            np.std(h_state, axis=-1, keepdims=True) + 1e-6
        )
        flow_dynamic = np.tanh(norm_h * (1.0 + s2) + sh2)
        v_tokens = -0.5 * z_flat + np.tanh(g2) * flow_dynamic

        # Reshape back to original dimensions
        if z_latent.ndim == 4:
            t_dim, c_dim, h_dim, w_dim = orig_shape
            v_out = v_tokens.reshape(t_dim, h_dim, w_dim, c_dim).transpose(0, 3, 1, 2)
        else:
            v_out = v_tokens.reshape(orig_shape)

        return np.asarray(v_out)

    def compute_flow_guided_velocity(
        self,
        z_latent: np.ndarray,
        t: float,
        text_tokens: np.ndarray | None,
        guidance_scale: float | None = None,
    ) -> tuple[np.ndarray, float]:
        r"""Compute Classifier-Free Flow Guided velocity:
        v_guided = v_uncond + s * (v_cond - v_uncond)

        Returns:
        - (guided_velocity, steering_magnitude)
        """
        s = self.config.guidance_scale if guidance_scale is None else float(guidance_scale)

        # Unconditional forward evaluation (null text)
        v_uncond = self.compute_conditioned_velocity(z_latent, t, text_tokens=None)

        if text_tokens is None or s == 0.0:
            return v_uncond, 0.0

        # Conditional forward evaluation
        v_cond = self.compute_conditioned_velocity(z_latent, t, text_tokens=text_tokens)

        # Steer trajectory in the direction of the semantic prompt
        steering = v_cond - v_uncond
        v_guided = v_uncond + s * steering
        steering_magnitude = float(np.linalg.norm(steering))

        return v_guided, steering_magnitude
