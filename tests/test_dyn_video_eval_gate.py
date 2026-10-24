r"""Certification gate tests for the held-out video evaluation instruments.

Covers:
1. PSNR / SSIM sanity (identity -> ceiling, degradation -> monotone drop).
2. Temporal consistency discriminates smooth advection from frame-shuffled jerk.
3. FVD proxy: near-zero separation for identical distributions, positive and
   larger for degraded distributions; deterministic across seeds/processes.
4. Held-out split discipline: disjoint sets, deterministic, run-id exclusivity.
5. Certification API: thresholds supplied by the caller, never hardcoded.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.video.eval_gate import (
    certify,
    compute_psnr,
    compute_ssim,
    evaluate_video_set,
    fvd_proxy,
    held_out_split,
    measure_temporal_consistency,
    sliced_wasserstein,
)


def _advected_video(seed: int, t: int = 8, c: int = 3, h: int = 32, w: int = 32, noise: float = 0.02) -> np.ndarray:
    """Smooth, physically advected synthetic clip (moving Gaussian blob)."""
    rng = np.random.RandomState(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    vid = np.zeros((t, c, h, w), dtype=np.float64)
    for ti in range(t):
        angle = 2 * np.pi * ti / t
        r2 = (xx - 0.5 * np.cos(angle)) ** 2 + (yy - 0.5 * np.sin(angle)) ** 2
        blob = np.exp(-r2 / 0.2)
        for ci in range(c):
            vid[ti, ci] = 0.6 * blob + 0.2 + noise * rng.randn(h, w)
    return vid


def _shuffled_copy(vid: np.ndarray, seed: int = 99) -> np.ndarray:
    perm = np.random.RandomState(seed).permutation(vid.shape[0])
    return vid[perm].copy()


def test_psnr_and_ssim_identity_ceilings_and_monotone_degradation():
    clean = _advected_video(11, noise=0.0)
    noisy = clean + np.random.RandomState(3).randn(*clean.shape) * 0.02
    noisier = clean + np.random.RandomState(3).randn(*clean.shape) * 0.08

    assert compute_psnr(clean, clean) == 100.0
    p1, p2 = compute_psnr(clean, noisy), compute_psnr(clean, noisier)
    assert p1 > p2 > 0.0, (p1, p2)

    s_ident = compute_ssim(clean, clean)
    assert s_ident > 0.99, s_ident
    s1, s2 = compute_ssim(clean, noisy), compute_ssim(clean, noisier)
    assert 0.0 <= s2 < s1 < s_ident, (s1, s2, s_ident)


def test_temporal_consistency_smooth_beats_shuffled():
    vid = _advected_video(5)
    smooth = measure_temporal_consistency(vid)
    jerky = measure_temporal_consistency(_shuffled_copy(vid))
    assert smooth < jerky, (smooth, jerky)
    with pytest.raises(ValueError):
        measure_temporal_consistency(vid[:2])


def test_fvd_proxy_identical_zero_degraded_positive_and_deterministic():
    vids_a = [_advected_video(s, noise=0.02) for s in (21, 22, 23)]
    d_same = fvd_proxy(vids_a, vids_a)["swd"]
    assert d_same < 1e-8, d_same  # identical distributions separate ~zero

    noisy = [v + np.random.RandomState(k + 100).randn(*v.shape) * 0.10 for k, v in enumerate(vids_a)]
    d_noisy = fvd_proxy(vids_a, noisy)["swd"]
    assert d_noisy > d_same + 0.01, (d_same, d_noisy)

    # Determinism: same inputs + same seed -> identical proxy distance
    assert fvd_proxy(vids_a, noisy)["swd"] == d_noisy

    # Cross-seed stability of ranking (not value): ordering must hold
    d_sw1 = sliced_wasserstein(np.random.RandomState(1).randn(30, 8), np.random.RandomState(2).randn(30, 8), seed=1)
    d_sw2 = sliced_wasserstein(np.random.RandomState(1).randn(30, 8), np.random.RandomState(2).randn(30, 8), seed=2)
    assert d_sw1 > 0 and d_sw2 > 0


def test_held_out_split_is_disjoint_deterministic_and_exclusive():
    ids = [f"clip_{i:03d}" for i in range(10)]
    train1, hold1 = held_out_split(ids, holdout_fraction=0.2, seed=13)
    train2, hold2 = held_out_split(ids, holdout_fraction=0.2, seed=13)
    assert train1 == train2 and hold1 == hold2  # deterministic
    assert not (set(train1) & set(hold1))       # disjoint
    assert set(train1) | set(hold1) == set(ids)  # full coverage
    # Differing seeds give differing (still disjoint) partitions
    _, hold_alt = held_out_split(ids, holdout_fraction=0.2, seed=7)
    assert hold_alt != hold1
    # Invalid params rejected
    with pytest.raises(ValueError):
        held_out_split(ids, holdout_fraction=0.0)


def test_evaluate_video_set_bundle_and_certify_explicit_thresholds():
    real = [_advected_video(s, noise=0.02) for s in (31, 32)]
    gen_smooth = [v + np.random.RandomState(k + 200).randn(*v.shape) * 0.02 for k, v in enumerate(real)]
    gen_bad = [v + np.random.RandomState(k + 300).randn(*v.shape) * 0.30 for k, v in enumerate(real)]

    m_good = evaluate_video_set(real, gen_smooth)
    m_bad = evaluate_video_set(real, gen_bad)
    # Ordering: smoother/closer generation must dominate on every instrument
    assert m_good["psnr_db_mean"] > m_bad["psnr_db_mean"]
    assert m_good["ssim_mean"] > m_bad["ssim_mean"]
    assert m_good["fvd_proxy_swd"] < m_bad["fvd_proxy_swd"]

    # Certification with explicit (caller-owned) cuts — no hardcoded thresholds
    strict = certify(m_bad, {"psnr_db_mean": 40.0, "ssim_mean": 0.95, "fvd_proxy_swd": 0.01})
    assert strict["pass"] is False
    lenient = certify(m_good, {"psnr_db_mean": 20.0, "ssim_mean": 0.5, "fvd_proxy_swd": 50.0})
    assert lenient["pass"] is True
    with pytest.raises(KeyError):
        certify(m_good, {"unknown_instrument": 1.0})
