"""The Riemann verifier's off-line scan must cover the height it was asked about.

Both tests exist because of a real defect: the off-line grid was built as
`(100..400) * 0.1`, i.e. t in [10, 40], regardless of `max_t`, and the engine
then printed `OFF_LINE_GRID_SCAN_PASS`. A run at T=300 reported a pass over a
window 260 units shorter than the question. The source-level test catches a
revert without needing a Rust toolchain; the behavioural test pins what the
engine actually emits.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN_RS = ROOT / "rust" / "riemann_verifier" / "src" / "main.rs"
BIN = ROOT / "rust" / "riemann_verifier" / "target" / "release" / "riemann_verifier"


def test_scan_window_is_not_a_hardcoded_range():
    src = MAIN_RS.read_text(encoding="utf-8")
    assert "(100..400)" not in src, (
        "the off-line scan grid is hardcoded to t in [10, 40] again; it must follow max_t"
    )
    assert "t in [10.0, 40.0]" not in src
    # The window must be derived from the requested height, not a constant.
    assert "max_t.max(GRID_SCAN_T_LO" in src
    assert '"OFF_LINE_GRID_SCAN_PASS"' in src


def _run(max_t: float) -> dict:
    out = subprocess.run(
        [str(BIN), str(max_t), "2000", "--json"],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    ).stdout
    return json.loads(out)


@pytest.mark.skipif(not BIN.is_file(), reason="riemann_verifier binary not built")
def test_scan_covers_the_requested_height_at_full_resolution():
    r = _run(300.0)
    assert r["max_t"] == 300.0
    lo, hi = r["grid_scan_window"]
    assert lo == 10.0
    assert hi >= r["max_t"], (hi, r["max_t"])
    assert r["grid_scan_covers_requested_height"] is True
    assert r["grid_scan_coarsened"] is False
    assert r["grid_scan_step"] == pytest.approx(0.1)
    assert r["grid_scan_samples"] >= 2900
    assert r["grid_scan_status"] == "OFF_LINE_GRID_SCAN_PASS"


@pytest.mark.skipif(not BIN.is_file(), reason="riemann_verifier binary not built")
def test_a_coarsened_scan_never_reports_a_bare_pass():
    """The window is kept and the resolution degrades; the weaker claim is labelled."""
    r = _run(5000.0)
    lo, hi = r["grid_scan_window"]
    assert lo == 10.0 and hi >= 5000.0
    assert r["grid_scan_coarsened"] is True
    assert r["grid_scan_step"] > 0.1
    assert r["grid_scan_samples"] == 20_000
    assert r["grid_scan_status"] == "OFF_LINE_GRID_SCAN_PASS_COARSENED", (
        "a coarsened scan must not be reported as a full-resolution PASS"
    )
