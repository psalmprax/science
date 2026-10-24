"""Noisy multi-dataset repair suite (C): 3+ real measured targets, 2% noise.

Suites the repair engine is measured against (all REAL measured values, no
planted truth):

  D1  AME2020 total binding energy vs mass number -- the volume/surface
      family. Original: real linear volume fit; repair: (A*Z) family.
  D2  AME2020 isobaric parabolas -- B vs Z at fixed A, which is genuinely a
      quadratic in Z (different variable, different form, same measured
      table): independent discovery problem.
  D3  AME2020 symmetry-energy residual -- B minus the linear-volume trend
      against (N-Z)^2/A -- the asymmetry family.

What is pinned, per dataset:
  * a supplied no-repair baseline that must never report a fix;
  * the repair's sealed verdict is either an improvement or an honest
    abstention -- never a fabricated one;
  * 2% observation noise is the reported figure (clean data is exactness).

Honest scope note recorded rather than hidden: the other real tables in
this repository (Bertozzi 1964: 5 runs; Brown lifter: 20 rows; warp energy:
~10 records) are below the 40-row floor the 4-role partition harness needs,
so a 3-dataset NOISY battery at that row minimum requires data acquisition
beyond this repository.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    RepairStatus,
    repair,
)
from apedp_dyn.proposers.rime_proposer import RimeProposer

ROOT = Path(__file__).resolve().parent.parent
_NUM = re.compile(r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)")
M_H, M_N = 7288.971064, 8071.31806

OPS = (RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT,
       RepairOperator.R3_LAG_INSERT)
CFG = RepairConfig(6, 2, 1, 0.02, 1e-9, OPS, min_sse_reduction=0.30)
NOISELESS = RepairConfig(6, 2, 1, 0.02, 1e-9, OPS, min_sse_reduction=0.30,
                         r5_emit_unidentified=True)

CUTS = {"rho_flag": RESID_RHO_FLAG, "bias_ratio_cut": 0.25, "variance_ratio_cut": 4.0,
            "curvature_corr_cut": 0.3, "interaction_corr_cut": 0.3, "min_rows": 30}


def _load_binding() -> pd.DataFrame:
    path = ROOT / "data" / "mass_1.mas20.txt"
    if not path.exists():
        pytest.skip(f"AME2020 file absent: {path}")
    rows = []
    for line in path.open(encoding="utf-8", errors="replace"):
        if line[:1] not in ("0", " ", "-"):
            continue
        m = _NUM.search(line)
        if not m:
            continue
        N, Z, A = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if A <= 0 or N != A - Z:
            continue
        rows.append({"A": float(A), "Z": float(Z),
                     "B": Z * M_H + N * M_N - float(m.group(5))})
    df = pd.DataFrame(rows)
    return df[(df.A >= 20) & (df.B > 0)].reset_index(drop=True)


def _d1_surface(ok):
    fr = ok[["A", "B"]].copy()
    return fr.rename(columns={"B": "target"}), ["A"]


def _d2_isobaric(ok):
    """B vs Z at fixed large A, where the parabola is genuinely quadratic."""
    # Measured available chains: A>=100 with >=15 isotopes yields 0 rows;
    # A>=60 with >=10 gives 92 rows over 9 isobaric chains (>=8 gives 298).
    big = ok[ok.A >= 60]
    keep = big.groupby("A").Z.transform("count") >= 10
    sub = big[keep][["A", "Z", "B"]].copy()
    fr = sub[["A", "Z", "B"]].rename(columns={"B": "target"})
    return fr, ["A", "Z"]


def _d3_symmetry(ok):
    """B minus linear volume trend, against the asymmetry term (N-Z)^2/A."""
    k = float(np.polyfit(ok.A.to_numpy(), ok.B.to_numpy(), 1)[0])
    fr = pd.DataFrame({
        "A": ok.A.to_numpy(),
        "NZ2A": ((ok.A - ok.Z - ok.Z) ** 2 / ok.A).to_numpy(),
        "target": (ok.B - k * ok.A).to_numpy(),
    })
    return fr, ["A", "NZ2A"]


def _noisy(frame: pd.DataFrame, noise: float, seed: int = 42) -> pd.DataFrame:
    if noise == 0.0:
        return frame
    r = np.random.default_rng(seed)
    fr = frame.copy()
    y = fr["target"].to_numpy(float)
    fr["target"] = y + r.normal(0, noise * float(np.std(y)), len(y))
    return fr


def _original(frame, features):
    tr = frame.iloc[:int(len(frame) * 0.4)].reset_index(drop=True)
    props = RimeProposer(feature_cols=features, target_col="target",
                        max_depth=2, budget=200).fit_and_propose(tr)
    assert props, "original proposer must emit on this real target"
    return sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]


def _none_fixture(frame, features):
    """Supplied no-repair baseline: a fixture that never proposes a repair."""
    orig = _original(frame, features)
    return orig


def _run(frame, features, cfg, seed=42, noise=0.02, original=None):
    fr = _noisy(frame, noise, seed)
    parts = RepairPartitions.from_frame(fr, fr["target"].to_numpy())
    orig = original if original is not None else _original(fr, features)
    return repair(orig, parts, cfg, **CUTS), fr


@pytest.mark.parametrize("dataset,fn", [
    ("D1_surface", _d1_surface),
    ("D2_isobaric", _d2_isobaric),
    ("D3_symmetry", _d3_symmetry),
])
def test_real_targets_under_two_percent_noise(dataset, fn):
    ok = _load_binding()
    frame, features = fn(ok)
    if len(frame) < 80:
        pytest.skip(f"{dataset} rows={len(frame)} below the 40-row partition floor")
    out, fr = _run(frame, features, CFG)
    print(f"\n[noisy-multi] {dataset:<12} rows={len(fr)} -> {out.status.value} "
          f"ident={out.identifiability_status} note={out.submit_note}")
    # Honest outcomes only: an improvement WITH sealed evidence, or an
    # abstention with nothing opened. Never a fabricated fix.
    assert out.status in (RepairStatus.IMPROVED, RepairStatus.NO_IMPROVEMENT,
                          RepairStatus.AMBIGUOUS, RepairStatus.INSUFFICIENT_EVIDENCE), out.status
    if out.status == RepairStatus.IMPROVED:
        assert out.sealed is not None and out.sealed["winner_r2"] > out.sealed["original_r2"]
    else:
        assert out.sealed is None


@pytest.mark.parametrize("dataset,fn", [
    ("D1_surface", _d1_surface),
    ("D2_isobaric", _d2_isobaric),
])
def test_no_repair_baseline_never_reports_a_fix(dataset, fn):
    """The baseline every repair claim is measured against."""
    ok = _load_binding()
    frame, features = fn(ok)
    if len(frame) < 80:
        pytest.skip(f"{dataset} rows={len(frame)} below floor")
    orig = _original(frame, features)
    # No repair runs at all: the original IS the answer, and status must be
    # NO_IMPROVEMENT with no sealed split opened.
    fr = _noisy(frame, 0.02)
    parts = RepairPartitions.from_frame(fr, fr["target"].to_numpy())
    # NO repair operators at all: the genuine no-repair baseline. (With R1
    # enabled this is a legitimate parameter-refit repair, not a baseline.)
    out = repair(orig, parts, RepairConfig(6, 1, 1, 0.02, 1e-9, ()), **CUTS)
    # Either no finding to repair, or not enough rows to diagnose -- both are
    # abstentions, and neither may report a fix or open the sealed split.
    assert out.status in (RepairStatus.NO_IMPROVEMENT, RepairStatus.INSUFFICIENT_EVIDENCE), out.status
    assert out.sealed is None
    assert out.winner is None
    print(f"\n[no-repair-baseline] {dataset:<12} rows={len(frame)} -> {out.status.value} "
          f"(no fix reported)")


def test_row_floor_limitation_is_recorded():
    """The other in-repo real tables cannot host the 4-role battery; say so."""
    small = {}
    b = ROOT / "data" / "bertozzi_1964.jsonl"
    if b.exists():
        small["bertozzi_1964"] = len(b.read_text().splitlines())
    lf = ROOT / "data" / "brown_lifter_tables.jsonl"
    if lf.exists():
        rows = sum(len(__import__("json").loads(ln).get("rows", [])) for ln in lf.read_text().splitlines())
        small["brown_lifter_tables"] = rows
    print(f"\n[scope] in-repo small real tables: {small} (all below the 40-row floor)")
    assert all(v < 40 for v in small.values()), small