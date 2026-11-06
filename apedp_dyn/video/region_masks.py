"""Region masks derived from the picture, not from fixed image coordinates.

The picture-to-video demo used to animate three hard-coded bands: the sky was
always the top 40 % (``y < 0.40``), the ocean always the bottom 40 % (``y >
0.60``) and the car always the middle 30 %. Any input image got that treatment,
so a photograph of a forest was animated as if it had a sky band and a surf
band, and the "preserved" object was preserved because of its row index rather
than its appearance.

This module replaces the coordinates with a measurement of the latent itself:
per-row texture energy (the mean absolute horizontal gradient across channels).
The two rows whose energy steps most sharply split the frame into three bands,
the upper band gets the smooth drift, the lower band the oscillatory flow, and
the middle band is preserved by the divergence-free projection. What the bands
*mean* is a decision this module does not make: it reports the row boundaries
and the energies it used, and the caller names them.
"""
from __future__ import annotations

import numpy as np

# A cut is evidence of a boundary only if it stands out from the row-to-row
# noise, and only if it sits in the same place across the frame. The photograph
# audit (docs/REGION_MASK_AUDIT.md) measured which of the two predicts a wrong
# split: both, and the band ordering besides. These are the values it used.
MIN_CUT_DECISIVENESS = 2.0
MAX_HALF_DISAGREEMENT = 0.06


class UnusableRegionSplit(RuntimeError):
    """The frame has no defensible three-band horizontal split.

    Raised only when a caller asks for it (``require_reliable=True``). The
    default returns the split with ``evidence["reliable"] = False`` and the
    reasons, because a caller that only gets an exception cannot show what it
    found.
    """


def row_texture_energy(latent: np.ndarray) -> np.ndarray:
    """Mean absolute horizontal gradient per row, averaged over channels.

    Parameters
    ----------
    latent : np.ndarray
        ``(C, H, W)`` or ``(H, W)`` array. Colour channels are averaged, so a
        smooth sky and a busy ocean separate even when their brightness matches.
    """
    arr = np.asarray(latent, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected (C, H, W) or (H, W), got shape {arr.shape}")
    if arr.shape[-1] < 3:
        raise ValueError("need at least 3 columns to measure a gradient")
    # (C, H, W-1) -> mean over channels and columns leaves one value per row.
    grad = np.abs(np.diff(arr, axis=-1))
    return grad.mean(axis=(0, 2))


def _describe_order(means: tuple[float, float, float]) -> str:
    upper, middle, lower = means
    if upper < middle < lower:
        return "increasing"
    if upper > middle > lower:
        return "decreasing"
    return "non_monotone"


def _best_two_cuts(
    energy: np.ndarray, min_len: int,
) -> tuple[int, int, float]:
    """The two rows whose adjacent step is largest, each band at least min_len."""
    h = energy.size
    step = np.abs(np.diff(energy, prepend=energy[0]))
    best = (-np.inf, min_len, h - min_len)
    for b1 in range(min_len, h - 2 * min_len + 1):
        for b2 in range(b1 + min_len, h - min_len + 1):
            score = step[b1] + step[b2]
            if score > best[0]:
                best = (score, b1, b2)
    return best[1], best[2], best[0]


def _evidence(
    latent: np.ndarray,
    energy: np.ndarray,
    step: np.ndarray,
    b1: int,
    b2: int,
    smoothing: int,
    min_band_fraction: float,
) -> dict:
    """Measured support for the split, and the reasons to distrust it.

    Three things a row split cannot express, each measured rather than assumed:
    whether the cuts stand out from row-to-row noise, whether the left and right
    halves of the frame agree on where the cuts are (a diagonal boundary gives
    two different answers), and whether the bands are ordered by texture at all.
    """
    arr = np.asarray(latent)
    h = energy.size
    median_step = float(np.median(step))
    decisiveness = (
        float(step[b1] + step[b2]) / median_step if median_step > 0.0 else 0.0)
    min_len = max(1, round(min_band_fraction * h))
    width = arr.shape[-1]
    half_cuts: dict[str, tuple[int, int]] = {}
    for side, columns in (("left", slice(0, width // 2)),
                          ("right", slice(width // 2, width))):
        half_energy = _smooth(row_texture_energy(arr[..., columns]), smoothing)
        half_cuts[side] = _best_two_cuts(half_energy, min_len)[:2]
    disagreement = max(
        abs(half_cuts["left"][0] - half_cuts["right"][0]),
        abs(half_cuts["left"][1] - half_cuts["right"][1]),
    ) / h
    means = (float(energy[:b1].mean()), float(energy[b1:b2].mean()),
             float(energy[b2:].mean()))
    order = _describe_order(means)

    reasons: list[str] = []
    if decisiveness < MIN_CUT_DECISIVENESS:
        reasons.append(
            f"the two cuts are {decisiveness:.2f}x the median row step "
            f"(need {MIN_CUT_DECISIVENESS:g}x): they sit inside the noise")
    if disagreement > MAX_HALF_DISAGREEMENT:
        reasons.append(
            f"the left and right halves disagree by {disagreement:.3f} of the "
            f"height (limit {MAX_HALF_DISAGREEMENT:g}): one row boundary does "
            "not describe the frame everywhere")
    if order == "non_monotone":
        reasons.append(
            "band texture is non-monotone, so the bands are not an ordered "
            "stack and 'the smoothest band is the sky' has no meaning")
    return {
        "reliable": not reasons,
        "reasons": reasons,
        "cut_decisiveness": round(decisiveness, 3),
        "half_disagreement": round(disagreement, 4),
        "texture_order": order,
        "half_cuts": {side: [cuts[0] / h, cuts[1] / h]
                      for side, cuts in half_cuts.items()},
    }


def _smooth(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or profile.size < window:
        return profile
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(profile, kernel, mode="same")


def derive_region_masks(
    latent: np.ndarray,
    *,
    smoothing: int = 5,
    min_band_fraction: float = 0.15,
    ramp_fraction: float = 0.15,
    require_reliable: bool = False,
) -> dict:
    """Split the frame into three texture bands and build soft masks.

    The split maximises the two strongest adjacent-row steps in the smoothed
    energy profile, subject to each band being at least ``min_band_fraction`` of
    the height. Masks are ramps of width ``ramp_fraction`` around each boundary
    and are normalised so they sum to one at every row.

    Returns
    -------
    dict
        ``masks`` (``upper``/``middle``/``lower``, each ``(H, W)``), the
        ``boundaries`` as height fractions, the ``energy`` profile used, and an
        ``evidence`` block (``reliable``, ``reasons``, ``cut_decisiveness``,
        ``half_disagreement``, ``texture_order``, ``half_cuts``), so a record
        can show both what the split was derived from and whether anything
        supports it.

    Raises
    ------
    UnusableRegionSplit
        Only when ``require_reliable=True`` and ``evidence["reliable"]`` is
        False. The default returns the split with the reasons attached.
    """
    if not 0.0 < min_band_fraction < 0.5:
        raise ValueError("min_band_fraction must lie in (0, 0.5)")
    if not 0.0 < ramp_fraction < 0.5:
        raise ValueError("ramp_fraction must lie in (0, 0.5)")
    energy = _smooth(row_texture_energy(latent), smoothing)
    h = energy.size
    min_len = max(1, round(min_band_fraction * h))
    if h < 3 * min_len + 2:
        raise ValueError(f"height {h} is too small for three bands")

    step = np.abs(np.diff(energy, prepend=energy[0]))
    b1, b2, _ = _best_two_cuts(energy, min_len)
    evidence = _evidence(
        latent, energy, step, b1, b2, smoothing, min_band_fraction)
    if require_reliable and not evidence["reliable"]:
        raise UnusableRegionSplit(
            "no defensible three-band split: " + "; ".join(evidence["reasons"]))

    rows = np.arange(h, dtype=np.float64)
    ramp = max(1.0, ramp_fraction * h)

    def _down(cut: int) -> np.ndarray:
        # 1 above the cut, 0 below it, ramped over `ramp` rows
        return np.clip(0.5 + (cut - rows) / ramp, 0.0, 1.0)

    def _up(cut: int) -> np.ndarray:
        return np.clip(0.5 + (rows - cut) / ramp, 0.0, 1.0)

    w_upper = _down(b1)
    w_lower = _up(b2)
    w_middle = np.clip(1.0 - w_upper - w_lower, 0.0, 1.0)
    total = w_upper + w_middle + w_lower
    total[total <= 0] = 1.0
    w_upper, w_middle, w_lower = w_upper / total, w_middle / total, w_lower / total
    w = np.asarray(latent).shape[-1]

    def _broadcast(weights: np.ndarray) -> np.ndarray:
        return np.repeat(weights[:, None], w, axis=1)

    return {
        "masks": {
            "upper": _broadcast(w_upper),
            "middle": _broadcast(w_middle),
            "lower": _broadcast(w_lower),
        },
        "boundaries": {"upper_to_middle": round(b1 / h, 4),
                       "middle_to_lower": round(b2 / h, 4)},
        "energy": {"min": round(float(energy.min()), 6),
                   "max": round(float(energy.max()), 6),
                   "upper_band_mean": round(float(energy[:b1].mean()), 6),
                   "middle_band_mean": round(float(energy[b1:b2].mean()), 6),
                   "lower_band_mean": round(float(energy[b2:].mean()), 6)},
        "method": "row_texture_energy_argmax_steps",
        "semantics": "texture rank only; the caller names the bands",
        "evidence": evidence,
    }


__all__ = [
    "MAX_HALF_DISAGREEMENT",
    "MIN_CUT_DECISIVENESS",
    "UnusableRegionSplit",
    "derive_region_masks",
    "row_texture_energy",
]
