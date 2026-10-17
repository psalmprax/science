r"""Held-Out Video Evaluation Gate: PSNR / SSIM / Temporal Consistency / FVD Proxy.

Status: experimental (certification harness — audit-only weighting)

Preregistration discipline for the video fast-path: sample-quality claims are
certified ONLY relative to held-out real videos that the generative p.ipeline
never fit. This module provides the instruments; THRESHOLDS ARE NOT HARDCODED
HERE (repo rule 1.3: no arbitrary threshold literals) — callers pass their own
preregistered cut-lines to ``VideoEvalGate.certify``.

Instruments (all dependency-free, deterministic):
1. PSNR (per-video, dB).
2. SSIM (windowed, grayscale, 8x8 windows, no external models).
3. Temporal consistency: mean |D_t^2 u| acceleration across frames — a smooth
   physically advected clip must beat a per-frame-shuffled clip on this.
4. FVD proxy: sliced Wasserstein-2 distance between per-frame feature
   distributions of real vs generated video sets. HONEST LABEL: the canonical
   FVD uses I3D softmax features, which this dependency-free harness cannot
   load; the proxy uses a FROZEN deterministic feature extractor (fixed-seed
   random image projections + gradient statistics + motion statistics). It is
   a ranked-separation instrument (lower = distributionally closer), not an
   FVD clone. Do not cite the proxy's absolute value as "true FVD".
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _kernel_video_bridge():
    """The kernel-backed bridge when the native fast path is usable.

    Lazy import (function level) so importing this module never pulls the
    apedp package tree, and so a missing/unbuilt kernel silently degrades to
    the pure-NumPy reference bodies below — which stay in place verbatim as
    the fallback (bridge docstring discipline: same numbers, slower path).
    """
    try:
        from apedp.rust_bridge import default_kernel
    except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
        return None
    return default_kernel if default_kernel.use_rust else None


# ---------------------------------------------------------------------------
# Frame metrics
# ---------------------------------------------------------------------------


def compute_psnr(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> float:
    """Peak SNR in dB between two RGB videos/frames in [0, 1]."""
    a = np.asarray(original_rgb, dtype=np.float64)
    b = np.asarray(reconstructed_rgb, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    bridge = _kernel_video_bridge()
    if bridge is not None:
        return float(bridge.video_psnr(a, b))
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.asarray(frame, dtype=np.float64)
    if frame.ndim == 3 and frame.shape[0] == 3:
        # (3, H, W) RGB -> ITU-R BT.601 luma
        return np.asarray(np.asarray(np.asarray((0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).astype(np.float64))))
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.float64)
    raise ValueError(f"Unsupported frame shape {frame.shape}")


def compute_ssim(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray,
                 win: int = 8, c1: float = 0.01, c2: float = 0.03) -> float:
    """Mean windowed SSIM over grayscale; accepts (H, W), (3, H, W),\
 or (T, 3, H, W)."""
    a = np.asarray(original_rgb, dtype=np.float64)
    b = np.asarray(reconstructed_rgb, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError(f"Bad array shapes {a.shape} / {b.shape}")
    if a.ndim == 4:
        per = [compute_ssim(a[t], b[t], win=win, c1=c1, c2=c2) for t in range(a.shape[0])]
        return float(np.mean(per))
    a, b = _grayscale(a), _grayscale(b)

    frames = [(a, b)] if a.ndim == 2 else zip(a, b, strict=False)
    bridge = _kernel_video_bridge()
    scores = []
    for fa, fb in frames:
        if bridge is not None:
            scores.append(float(bridge.video_ssim_frame(fa, fb, win, c1, c2)))
            continue
        h, w = fa.shape
        vals = []
        for y in range(0, h - win + 1, win):
            for x in range(0, w - win + 1, win):
                pa = fa[y:y + win, x:x + win]
                pb = fb[y:y + win, x:x + win]
                ma, mb = float(pa.mean()), float(pb.mean())
                va, vb = float(pa.var()), float(pb.var())
                cov = float(((pa - ma) * (pb - mb)).mean())
                num = (2 * ma * mb + c1) * (2 * cov + c2)
                den = (ma ** 2 + mb ** 2 + c1) * (va + vb + c2)
                vals.append(num / den)
        scores.append(float(np.mean(vals)))
    return float(np.mean(scores))


def measure_temporal_consistency(video_rgb: np.ndarray) -> float:
    r"""Mean per-pixel temporal acceleration \|D_t^2 u\| over (T, 3, H, W).

    Inverse-score instrument: LOWER is smoother/more physically coherent;
    a synthetic advected clip must score below a frame-shuffled copy of
    itself (pinned in tests). Larger videos (more frames) dilute boundary
    effects naturally through the mean.
    """
    vid = np.asarray(video_rgb, dtype=np.float64)
    if vid.ndim != 4 or vid.shape[0] < 3:
        raise ValueError(f"Expected (T>=3, C, H, W) video, got {vid.shape}")
    d2 = vid[2:] - 2.0 * vid[1:-1] + vid[:-2]
    return float(np.mean(np.abs(d2)))


# ---------------------------------------------------------------------------
# FVD proxy (sliced Wasserstein over frozen deterministic features)
# ---------------------------------------------------------------------------

class _FrozenFrameFeaturizer:
    """Frozen deterministic per-frame feature extractor (fixed seed, no learned
    parameters, no data leakage possible). 4 blocks x 4096 dims: raw 64x64
    grayscale, x-gradient, y-gradient, and abs diffuse (frame-to-frame motion)
    against the within-video previous frame, all passed through a fixed
    random projection to ``feature_dim``.
    """

    _BLOCK = 4096           # 64x64 grayscale
    _N_BLOCKS = 4

    def __init__(self, feature_dim: int = 64, seed: int = 2601) -> None:
        rng = np.random.RandomState(seed)
        self.feature_dim = feature_dim
        self.proj = rng.normal(0.0, 1.0 / np.sqrt(64), size=(self._BLOCK * self._N_BLOCKS, feature_dim))

    @staticmethod
    def _down64(gray: np.ndarray) -> np.ndarray:
        """Deterministic crop-or-pad to exactly 64x64 (resolution-independent)."""
        g = np.zeros((64, 64), dtype=np.float64)
        h_min = min(gray.shape[0], 64)
        w_min = min(gray.shape[1], 64)
        g[:h_min, :w_min] = gray[:h_min, :w_min]
        return g

    def _frame_feature(self, frame: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        g = self._down64(_grayscale(frame))
        xg = np.zeros_like(g)
        xg[1:, :] = np.abs(g[1:, :] - g[:-1, :])
        yg = np.zeros_like(g)
        yg[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
        if prev is not None:
            pg = self._down64(_grayscale(prev))
            motion = np.abs(g - pg)
        else:
            motion = np.zeros_like(g)
        blocks = np.concatenate([g.ravel(), xg.ravel(), yg.ravel(), motion.ravel()])
        return blocks @ self.proj

    def features(self, videos: Sequence[np.ndarray]) -> np.ndarray:
        """Per-frame features over a set of videos; shape (n_frames_total, feature_dim)."""
        feats = []
        for vid in videos:
            arr = np.asarray(vid, dtype=np.float64)
            if arr.ndim != 4:
                raise ValueError(f"Expected (T, C, H, W) video, got {arr.shape}")
            prev = None
            for t in range(arr.shape[0]):
                feats.append(self._frame_feature(arr[t], prev))
                prev = arr[t]
        return np.asarray(feats, dtype=np.float64)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_projections: int = 128,
                       seed: int = 7) -> float:
    """Sliced Wasserstein-2 distance between two point clouds (n_i, d)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    y = np.asarray(y, dtype=np.float64).reshape(-1, y.shape[-1])
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimension mismatch")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("Need at least 2 points per distribution")
    bridge = _kernel_video_bridge()
    if bridge is not None:
        return float(bridge.video_sliced_wasserstein(x, y, n_projections=n_projections, seed=seed))
    rng = np.random.RandomState(seed)
    d = x.shape[1]
    dirs = rng.normal(0.0, 1.0, size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12
    total = 0.0
    for k in range(n_projections):
        px = np.sort(x @ dirs[k])
        py = np.sort(y @ dirs[k])
        # Match quantile positions by index over equal-length sorted pairs
        n = min(len(px), len(py))
        total += float(np.mean((np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(px)), px) -
                                np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(py)), py)) ** 2))
    return float(np.sqrt(total / n_projections))


def fvd_proxy(real_videos: Sequence[np.ndarray], generated_videos: Sequence[np.ndarray],
              n_projections: int = 128, seed: int = 7) -> dict[str, float]:
    """Sliced-Wasserstein FVD proxy between two video sets.

    Returns ``{"swd": d, "n_features_real": *, "n_features_gen": *}``.
    NOT an I3D-FVD clone; see module docstring for the honest label.
    """
    featurizer = _FrozenFrameFeaturizer(seed=2601)
    f_real = featurizer.features(real_videos)
    f_gen = featurizer.features(generated_videos)
    return {
        "swd": sliced_wasserstein(f_real, f_gen, n_projections=n_projections, seed=seed),
        "n_features_real": int(f_real.shape[0]),
        "n_features_gen": int(f_gen.shape[0]),
    }


# ---------------------------------------------------------------------------
# Held-out discipline + certification API
# ---------------------------------------------------------------------------

def held_out_split(ids: Sequence[str], holdout_fraction: float = 0.2,
                   seed: int = 13) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic, disjoint train/hold-out split with the test set kept out.

    The holdout set stays exclusive: no later caller may renew or shuffle it
    without changing the (ids, fraction, seed) tuple, mirroring the repo's
    burnable-holdout discipline (AGENTS.md 1.2) at the data-split level.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    ids = list(dict.fromkeys(ids))
    if len(ids) < 2:
        raise ValueError("Need at least 2 distinct ids")
    order = np.random.RandomState(seed).permutation(len(ids))
    n_hold = max(1, round(holdout_fraction * len(ids)))
    hold = [ids[i] for i in order[:n_hold]]
    train = [ids[i] for i in order[n_hold:]]
    assert not (set(hold) & set(train))
    return tuple(sorted(train)), tuple(sorted(hold))


@dataclass(frozen=True)
class EvalReport:
    psnr_db: float
    ssim: float
    temporal_consistency: float
    fvd_proxy_swd: float
    extras: dict[str, float]


def evaluate_video_set(
    real_videos: Sequence[np.ndarray],
    generated_videos: Sequence[np.ndarray],
    n_projections: int = 128,
    seed: int = 7,
) -> dict[str, float]:
    """Aggregate instrument bundle over paired real/generated video sets."""
    if len(real_videos) != len(generated_videos):
        raise ValueError("Sets must be paired 1:1 for this bundle")
    psnrs, ssims, conss = [], [], []
    for r, g in zip(real_videos, generated_videos, strict=False):
        psnrs.append(compute_psnr(r, g))
        ssims.append(compute_ssim(r, g))
        conss.append(measure_temporal_consistency(g))
    fvd = fvd_proxy(real_videos, generated_videos, n_projections=n_projections, seed=seed)
    return {
        "psnr_db_mean": float(np.mean(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "temporal_consistency_mean": float(np.mean(conss)),
        "fvd_proxy_swd": fvd["swd"],
        "n_features_gen": fvd["n_features_gen"],
    }


def certify(metrics: dict[str, float], preregistered: dict[str, float]) -> dict[str, object]:
    """Compare measured instruments against caller-supplied preregistered cuts.

    ``preregistered`` maps instrument name -> cut value:
      ``psnr_db_mean`` (>=), ``ssim_mean`` (>=), ``fvd_proxy_swd`` (<=),
      ``temporal_consistency_mean`` (<=).
    No thresholds are defined in this module (repo rule 1.3).
    """
    checks: dict[str, object] = {}
    all_pass = True
    for name, cut in preregistered.items():
        if name not in metrics:
            raise KeyError(f"Unknown instrument {name!r}; measured: {list(metrics)}")
        if name in ("psnr_db_mean", "ssim_mean"):
            ok = metrics[name] >= cut
        elif name in ("fvd_proxy_swd", "temporal_consistency_mean"):
            ok = metrics[name] <= cut
        else:
            raise ValueError(f"No direction defined for instrument {name!r}")
        checks[name] = {"measured": float(metrics[name]), "cut": float(cut), "pass": bool(ok)}
        all_pass = all_pass and bool(ok)
    return {"pass": all_pass, "checks": checks}
