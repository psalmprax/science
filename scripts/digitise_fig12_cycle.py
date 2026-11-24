#!/usr/bin/env python3
r"""Digitise Fig. 12(a) of arXiv:2507.16152v2 from the PDF, instead of reading it by eye.

Fig. 12(a) plots the logical clock cycle time (in units of ``tau_echo``) against
photon loss, for code distances L = 4, 6, ... 22. The caption quotes a single point
(209 tau_echo at L = 3) and calls the scaling "linear with L". It is linear, but
affine, and the slope is the quantity this script measures.

Method
------
1. Render page 20 at 500 dpi with ``pdftoppm``, or accept an already-rendered PNG.
2. Crop the left column, locate the y-spine, and cluster the nine y-tick-label rows
   (250 ... 2250, about 105.5 px apart). That gives an exact row -> tau_echo
   calibration; using the gridlines instead fails, because most of them are minor
   gridlines at 50 tau_echo, not majors.
3. Detect the ten coloured curve centres in a window at low photon loss
   (x = 8%-45% of the plot width). The topmost curve is the *largest* L, not the
   smallest.
4. Least-squares fit ``tau = a + b L`` and compare with the anchored model
   (209 tau_echo at L = 3) used by :mod:`apedp_dyn.quantum.sparrow_photonics`.

The ten extracted points are pinned in
``tests/test_dyn_sparrow_deterministic_photonics.py``, so a change to that model has
to reproduce the published figure.

Usage
-----
    python3 scripts/digitise_fig12_cycle.py --pdf /path/to/2507.16152v2.pdf
    python3 scripts/digitise_fig12_cycle.py --png /path/to/page20.png
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PAGE = 20
DPI = 500

#: y tick labels of Fig. 12(a), read top to bottom.
TICK_VALUES = (2250, 2000, 1750, 1500, 1250, 1000, 750, 500, 250)

#: Values this script extracted on 2026-11-02 from 2507.16152v2, kept here so a
#: re-run reports drift instead of silently changing the model.
FIG12A_DIGITISED = {
    4: 310.9,
    6: 524.1,
    8: 742.1,
    10: 961.3,
    12: 1179.3,
    14: 1397.3,
    16: 1612.9,
    18: 1826.2,
    20: 2037.0,
    22: 2245.5,
}


#: N per curve in Fig. 12(b), top to bottom, with the slope ratios measured on
#: 2026-11-02 (each curve's slope divided by the N = 2 slope).
FIG12B_N_ORDER = (12, 10, 8, 6, 4, 2)
FIG12B_SLOPE_RATIOS = {12: 5.006, 10: 4.195, 8: 3.396, 6: 2.604, 4: 1.805, 2: 1.000}


def _cluster(rows: np.ndarray, gap: float) -> list[float]:
    """Collapse runs of adjacent rows (a thick line or a text glyph) to one centre."""
    out: list[float] = []
    cur = [float(rows[0])]
    for value in rows[1:]:
        if value - cur[-1] <= gap:
            cur.append(float(value))
        else:
            out.append(float(np.mean(cur)))
            cur = [float(value)]
    out.append(float(np.mean(cur)))
    return out


def digitise(image_path: Path) -> tuple[dict[int, float], float, float, float]:
    """Return the ten (L, tau_echo) points, the affine fit, and its rms residual."""
    image = np.asarray(Image.open(image_path).convert("RGB")).astype(float) / 255.0
    height, width, _ = image.shape
    panel = image[int(height * 0.09):int(height * 0.34), int(width * 0.05):int(width * 0.50)]
    h, w, _ = panel.shape

    brightest, darkest = panel.max(2), panel.min(2)
    black = brightest < 0.30
    coloured = (brightest - darkest > 0.20) & (brightest > 0.35)

    column_counts, row_counts = black.sum(0), black.sum(1)
    left = int(np.argmax(column_counts[: w // 2]))
    right = w // 2 + int(np.argmax(column_counts[w // 2:]))
    top = int(np.argmax(row_counts[: h // 2]))
    bottom = h // 2 + int(np.argmax(row_counts[h // 2:]))

    label_rows = np.where(black[top:bottom, left - 150:left - 15].sum(1) > 2)[0] + top
    labels = _cluster(label_rows, gap=20)
    if len(labels) != len(TICK_VALUES):
        raise RuntimeError(f"expected {len(TICK_VALUES)} y-tick labels, found {len(labels)}")
    calibration = np.polyfit(labels, list(TICK_VALUES), 1)

    x_from = left + int(0.08 * (right - left))
    x_to = left + int(0.45 * (right - left))
    curve_rows = np.where(coloured[top + 4:bottom - 3, x_from:x_to].mean(1) > 0.7)[0] + top + 4
    curves = _cluster(curve_rows, gap=5)
    if len(curves) != 10:
        raise RuntimeError(f"expected 10 curves, found {len(curves)}")

    distances = list(range(22, 3, -2))  # topmost curve is the largest L
    taus = {L: float(np.polyval(calibration, row)) for L, row in zip(distances, curves)}

    x = np.array(list(taus.keys()), dtype=float)
    y = np.array(list(taus.values()), dtype=float)
    design = np.vstack([np.ones_like(x), x]).T
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    rms = float(np.sqrt(np.mean((intercept + slope * x - y) ** 2)))
    return taus, float(intercept), float(slope), rms


def digitise_panel_b(image_path: Path) -> dict[int, float]:
    """Return Fig. 12(b)'s per-N slope ratios, each divided by the N = 2 slope.

    Eq. (11) is linear in ``(2 n_i^max + 1) tau_echo``, so the plotted slopes have to
    scale as ``2N + 1``. Measuring two columns and dividing by the N = 2 slope tests
    that without converting pixels to tau_echo at all, which is why this check needs no
    y calibration. The L factor is common to every curve, so this panel cannot fix it.
    """
    image = np.asarray(Image.open(image_path).convert("RGB")).astype(float) / 255.0
    height, width, _ = image.shape
    page = image[int(height * 0.09):int(height * 0.34), int(width * 0.05):int(width * 0.55)]
    brightest, darkest = page.max(2), page.min(2)
    black = brightest < 0.30
    coloured = (brightest - darkest > 0.18) & (brightest > 0.30)

    spines = _cluster(np.where(black.sum(0) > 0.5 * page.shape[0])[0], gap=6)
    if len(spines) < 4:
        raise RuntimeError(f"expected four panel spines, found {len(spines)}")
    left, right = int(spines[-2]), int(spines[-1])

    horizontals = _cluster(np.where(black.sum(1) > 0.5 * page.shape[1])[0], gap=6)
    if len(horizontals) < 2:
        raise RuntimeError("could not find the panel's top and bottom spines")
    top, bottom = int(horizontals[0]), int(horizontals[-1])

    def bands(column: int) -> list[float]:
        rows = np.where(coloured[top + 8:bottom - 8, column])[0] + top + 8
        if len(rows) == 0:
            raise RuntimeError(f"no curve pixels in column {column}")
        # Seed the run but not the output list: seeding both counts the first band twice.
        centres: list[float] = []
        run = [float(rows[0])]
        for value in rows[1:]:
            if value - run[-1] <= 6:
                run.append(float(value))
            else:
                centres.append(float(np.mean(run)))
                run = [float(value)]
        centres.append(float(np.mean(run)))
        return centres

    # The legend sits inside the axes on the right, so its colour swatches add a fake
    # band to columns that pass through it. Take the widest pair of columns that gives
    # exactly one band per curve, rather than hard-coding a pair that may not.
    b1: list[float] = []
    b2: list[float] = []
    for frac_left, frac_right in ((0.30, 0.80), (0.25, 0.72), (0.22, 0.78), (0.30, 0.68)):
        c1 = left + int(frac_left * (right - left))
        c2 = left + int(frac_right * (right - left))
        b1, b2 = bands(c1), bands(c2)
        if len(b1) == len(FIG12B_N_ORDER) and len(b2) == len(FIG12B_N_ORDER):
            break
    else:
        raise RuntimeError(
            f"expected {len(FIG12B_N_ORDER)} curves per column, found {len(b1)} and {len(b2)}"
        )

    slopes = [(y2 - y1) / (c2 - c1) for y1, y2 in zip(b1, b2)]
    reference = abs(slopes[-1])
    return {N: abs(s) / reference for N, s in zip(FIG12B_N_ORDER, slopes)}


def render(pdf: Path, out_dir: Path) -> Path:
    if shutil.which("pdftoppm") is None:
        sys.exit("pdftoppm not found; install poppler-utils or pass --png")
    stem = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-f", str(PAGE), "-l", str(PAGE), "-r", str(DPI), "-png", str(pdf), str(stem)],
        check=True,
    )
    return next(out_dir.glob("page*.png"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdf", type=Path, help="arXiv PDF to render")
    parser.add_argument("--png", type=Path, help="already-rendered page 20 PNG")
    args = parser.parse_args(argv)
    if not args.pdf and not args.png:
        parser.error("pass --pdf or --png")

    with tempfile.TemporaryDirectory() as tmp:
        image = args.png if args.png is not None else render(args.pdf, Path(tmp))
        taus, intercept, slope, rms = digitise(Path(image))
        try:
            panel_b = digitise_panel_b(Path(image))
        except Exception as exc:  # panel (b) is a secondary check
            panel_b = None
            print(f"warning: Fig. 12(b) not digitised: {exc}")

    print(" L   tau_echo")
    for L in sorted(taus):
        print(f"{L:3d} {taus[L]:8.1f}")

    print(f"\nfit: tau = {intercept:.1f} + {slope:.2f} L   (rms {rms:.1f} tau_echo)")
    print("\ndrift against the pinned extraction:")
    for L in sorted(FIG12A_DIGITISED):
        pinned = FIG12A_DIGITISED[L]
        print(f"  L={L:2d} pinned {pinned:7.1f}  this run {taus[L]:7.1f}  drift {taus[L] - pinned:+.1f}")

    print(f"\nL=16 -> {intercept + 16 * slope:.0f} tau_echo = "
          f"{(intercept + 16 * slope) * 29e-9 * 1e6:.2f} us at 29 ns")
    anchored = 209.0 + 107.5 * 13
    print(f"anchored model 209 + 107.5 (L - 3) at L=16 -> {anchored:.1f} tau_echo")

    if panel_b is not None:
        print("\nFig. 12(b) slope ratios against Eq. (11)'s (2N + 1)/5:")
        for N in FIG12B_N_ORDER:
            drift = panel_b[N] - FIG12B_SLOPE_RATIOS[N]
            print(f"  N={N:2d}  measured {panel_b[N]:6.3f}  pinned {FIG12B_SLOPE_RATIOS[N]:6.3f} "
                  f"expected {(2 * N + 1) / 5:6.3f}  drift {drift:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
