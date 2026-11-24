#!/usr/bin/env python3
"""Fail when the digitised Fig. 12(a) cycle points drift from the pinned extraction.

This is the CI half of ``scripts/digitise_fig12_cycle.py``. It digitises the figure
and compares each measured ``tau_echo`` against the pinned value, exiting non-zero if
any point moves further than the tolerance.

Tolerance: 4.0 tau_echo, about 1.7 rendered pixels at 500 dpi. Re-running against the
same PDF is bit-identical (drift 0.0), but a different poppler version can shift a
detected curve centre by a pixel or two, which is 2.4 units per pixel. A real change
to the figure or to the extraction logic moves points by tens of units, so 4.0
separates renderer noise from a genuine change.

Exit codes: 0 in tolerance, 1 drift, 2 the figure could not be digitised at all
(missing PDF, missing poppler, or a detection that did not find nine labels and ten
curves). In CI, 2 is also a build failure: a check that cannot run must not pass.

Usage
-----
    python3 scripts/check_fig12_drift.py --pdf /path/to/2507.16152v2.pdf
    python3 scripts/check_fig12_drift.py --png /path/to/page20.png
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# The sibling module is imported by file location too (tests load this script by
# path), so put ``scripts/`` on the path rather than relying on the caller's cwd.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from digitise_fig12_cycle import FIG12A_DIGITISED, digitise, render  # noqa: E402

#: Maximum allowed |measured - pinned| per point, in units of tau_echo.
TOLERANCE_TAU_ECHO = 4.0


def compare(
    measured: dict[int, float],
    pinned: dict[int, float] | None = None,
    tolerance: float = TOLERANCE_TAU_ECHO,
) -> tuple[list[dict[str, float | int | bool]], bool]:
    """Return per-distance drift rows and whether every point is in tolerance."""
    reference = FIG12A_DIGITISED if pinned is None else pinned
    rows: list[dict[str, float | int | bool]] = []
    for distance in sorted(reference):
        if distance not in measured:
            rows.append({"L": distance, "pinned": reference[distance],
                         "measured": float("nan"), "drift": float("inf"), "passed": False})
            continue
        drift = measured[distance] - reference[distance]
        rows.append(
            {
                "L": distance,
                "pinned": reference[distance],
                "measured": measured[distance],
                "drift": drift,
                "passed": abs(drift) <= tolerance,
            }
        )
    return rows, all(bool(row["passed"]) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, help="arXiv PDF to render")
    parser.add_argument("--png", type=Path, help="already-rendered page 20 PNG")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_TAU_ECHO,
                        help=f"max drift per point in tau_echo (default {TOLERANCE_TAU_ECHO})")
    args = parser.parse_args(argv)
    if not args.pdf and not args.png:
        parser.error("pass --pdf or --png")

    try:
        with TemporaryDirectory() as tmp:
            if args.png is not None:
                image = args.png
                digest = hashlib.sha256(args.png.read_bytes()).hexdigest()[:16]
            else:
                digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()[:16]
                image = render(args.pdf, Path(tmp))
            measured, intercept, slope, rms = digitise(Path(image))
    except SystemExit:
        raise
    except Exception as exc:  # detection or rendering failure
        print(f"FAIL: could not digitise Fig. 12(a): {exc}")
        return 2

    rows, ok = compare(measured, tolerance=args.tolerance)
    print(f"source sha256[:16] = {digest}")
    print(f"fit: tau = {intercept:.1f} + {slope:.2f} L   rms {rms:.1f} tau_echo")
    print(f"tolerance: {args.tolerance} tau_echo\n")
    print("  L     pinned   measured     drift   verdict")
    for row in rows:
        print(
            f"{row['L']:3d} {row['pinned']:10.1f} {row['measured']:10.1f} "
            f"{row['drift']:+9.1f}   {'ok' if row['passed'] else 'DRIFT'}"
        )

    if ok:
        print("\nPASS: every digitised point is within tolerance.")
        return 0

    print("\nFAIL: the figure no longer matches the pinned extraction.")
    print("If arXiv replaced the preprint, re-run scripts/digitise_fig12_cycle.py,")
    print("update apedp_dyn/quantum/sparrow_photonics.py, the spec's clock section")
    print("and the decoder/wall-clock figures in the docs, then update the pins here.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
