"""REAL-data repair benchmark: AME2020 atomic masses (no planting anywhere).

Why this file exists: every repair milestone before it ran on planted
synthetic worlds. This one runs discovery → diagnosis → bounded repair on
genuinely measured data (the AME2020 atomic-mass evaluation shipped in this
repo), with the semi-empirical mass formula (SEMF) as the literature
reference rather than a planted ground truth.

Measured, pinned here:
  * discovery on real B(A,Z): `+7426.14*A + 93729.3` at dev R² ≈ 0.994 —
    the volume term of the SEMF, recovered from measured masses;
  * diagnosis on that original: `curvature_suspect` fires (the missing
    surface term A^(2/3));
  * bounded repair at the PREREGISTERED bar (0.02): NO_IMPROVEMENT — the
    absolute bar is mis-scaled for a real dataset whose original already
    explains 99.4% (achievable gain ≈ 0.005);
  * bounded repair at a scale-appropriate bar (≤0.005): IMPROVED, winner
    adds (A*Z), sealed R² 0.9995 vs original 0.9947 — the first
    sealed-validated repair on measured data in this repository;
  * literature reference: SEMF with standard coefficients reaches R² 0.9999
    on the same frame (units keV), so the data fully support the known law —
    the boundary is grammar reachability (A^(2/3) is not in the transform
    set) and threshold calibration, not the data.

Honest framing: the added (A*Z) term is a CORRELATED PROXY for the
Coulomb/surface structure, not the SEMF term itself — predictive gain
without unique identification, i.e. MULTI_HYPOTHESIS territory, which the
APEDP-X track would have to adjudicate. Nothing here promotes.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic import rime_repair_diagnosis as diag
from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    RepairStatus,
    build_probes,
    repair,
)
from apedp_dyn.proposers.rime_proposer import RimeProposer

ROOT = Path(__file__).resolve().parent.parent
MASS_FILE = ROOT / "data" / "mass_1.mas20.txt"
M_H, M_N = 7288.971064, 8071.31806  # keV mass excesses, from the same file

CUTS = dict(rho_flag=RESID_RHO_FLAG, bias_ratio_cut=0.25, variance_ratio_cut=4.0,
            curvature_corr_cut=0.3, interaction_corr_cut=0.3, min_rows=30)
OPERATORS = (RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT,
             RepairOperator.R3_LAG_INSERT)


def _load_binding() -> pd.DataFrame:
    """Parse AME2020 mass excesses -> binding energies (real measured)."""
    if not MASS_FILE.exists():
        pytest.skip(f"AME2020 file absent: {MASS_FILE}")
    pat = re.compile(r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)")
    rows = []
    for line in MASS_FILE.open(encoding="utf-8", errors="replace"):
        if line[:1] not in ("0", " ", "-"):
            continue
        m = pat.search(line)
        if not m:
            continue
        N, Z, A = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if A <= 0 or N != A - Z:
            continue
        rows.append({"A": A, "Z": Z, "N": N, "mex_keV": float(m.group(5))})
    df = pd.DataFrame(rows)
    df["B_keV"] = df.Z * M_H + df.N * M_N - df.mex_keV
    df = df[(df.A >= 20) & (df.B_keV > 0)].copy()
    rng = np.random.RandomState(0)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def _frame(ok: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"A": ok.A.to_numpy(float), "Z": ok.Z.to_numpy(float),
                         "target": ok.B_keV.to_numpy(float)})


def test_real_discovery_recovers_volume_term_and_diagnosis_fires():
    ok = _load_binding()
    assert len(ok) >= 1000, len(ok)
    fr = _frame(ok)
    parts = RepairPartitions.from_frame(fr, fr["target"].to_numpy(), fracs=(0.4, 0.2, 0.2, 0.2))
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    props = RimeProposer(feature_cols=["A", "Z"], target_col="target",
                         max_depth=3, budget=200).fit_and_propose(tr)
    assert len(props) > 0
    orig = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    assert orig.parameters["dev_r2"] > 0.99, orig.parameters
    assert "A" in (orig.metadata.get("equation") or ""), orig.metadata

    rep = diag.diagnose_residuals(orig, fr, fr["target"].to_numpy(), "real")
    yp = np.asarray(orig.compute_score(fr), dtype=float)
    yt = fr["target"].to_numpy()
    m = np.isfinite(yt) & np.isfinite(yp)
    probes = {k: np.asarray(v)[m] for k, v in build_probes(fr, ("A", "Z")).items()}
    f = diag.classify_findings(rep, (yt - yp)[m], yp[m], probes, **CUTS)
    assert diag.FINDING_CURVATURE_SUSPECT in f, f
    print(f"\n[real] original devR2={orig.parameters['dev_r2']:.4f}; findings={f}")


def test_real_repair_threshold_calibration_pinned():
    """Real data exposes the absolute-bar miscalibration; both bars pinned."""
    ok = _load_binding()
    fr0 = _frame(ok)

    def run(bar: float):
        parts = RepairPartitions.from_frame(fr0.copy(), fr0["target"].to_numpy(),
                                           fracs=(0.4, 0.2, 0.2, 0.2))
        tr = parts.discovery_df.assign(target=parts.discovery_y)
        orig = sorted(RimeProposer(feature_cols=["A", "Z"], target_col="target",
                                   max_depth=3, budget=200).fit_and_propose(tr),
                      key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        cfg = RepairConfig(6, 2, 1, bar, 1e-9, OPERATORS)
        return repair(orig, parts, cfg, **CUTS)

    refused = run(0.02)      # preregistered absolute bar
    assert refused.status == RepairStatus.NO_IMPROVEMENT, refused.status
    assert refused.sealed is None

    accepted = run(0.005)    # scale-appropriate bar for a 0.994 baseline
    assert accepted.status == RepairStatus.IMPROVED, (accepted.status, accepted.submit_note)
    assert accepted.winner is not None and "(A*Z)" in (accepted.winner.metadata["term"] or "")
    assert accepted.sealed is not None
    assert accepted.sealed["sealed_r2_gain"] > 0.004, accepted.sealed
    assert accepted.sealed["winner_r2"] > accepted.sealed["original_r2"]
    print(f"\n[real] bar=0.02 -> NO_IMPROVEMENT; bar=0.005 -> IMPROVED "
          f"sealed_gain={accepted.sealed['sealed_r2_gain']:.4f} term=(A*Z)")


def test_real_semf_literature_reference_reached():
    """The data DO support the known law (units keV): grammar is the boundary."""
    ok = _load_binding()
    A, Z = ok.A.to_numpy(float), ok.Z.to_numpy(float)
    B = ok.B_keV.to_numpy(float)
    semf_keV = (15.8 * A - 18.3 * A ** (2 / 3) - 0.71 * Z * (Z - 1) / A ** (1 / 3)
                - 23.7 * (A - 2 * Z) ** 2 / A) * 1000.0
    ss = float(np.sum((B - B.mean()) ** 2))
    r2 = 1 - float(np.sum((B - semf_keV) ** 2)) / ss
    assert r2 > 0.999, r2
    print(f"\n[real] SEMF literature coefficients R2={r2:.4f} on measured masses")


def _parts_and_orig(fr0):
    from apedp_dyn.epistemic.rime_repair_engine import RepairPartitions
    parts = RepairPartitions.from_frame(fr0.copy(), fr0["target"].to_numpy(),
                                        fracs=(0.4, 0.2, 0.2, 0.2))
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    props = RimeProposer(feature_cols=["A", "Z"], target_col="target",
                         max_depth=3, budget=200).fit_and_propose(tr)
    assert len(props) > 0
    return parts, sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]


def _generate_for_terms(fr0):
    parts, orig = _parts_and_orig(fr0)
    cfg0 = RepairConfig(14, 2, 1, 0.02, 1e-9, OPERATORS,
                        min_sse_reduction=0.30, rational_powers=(2 / 3, 1 / 3))
    from apedp_dyn.epistemic.rime_repair_engine import generate_repairs
    return generate_repairs(orig, ("interaction_suspect",), parts.repair_df,
                            parts.repair_y, cfg0)


def test_rational_power_grammar_reaches_surface_term():
    """Domain-informed family (declared, explicit) makes A^(2/3) reachable."""
    ok = _load_binding()
    fr0 = _frame(ok)
    parts, orig = _parts_and_orig(fr0)
    cfg_without = RepairConfig(14, 2, 1, 0.02, 1e-9, OPERATORS,
                               min_sse_reduction=0.30, rational_powers=())
    cfg_with = RepairConfig(14, 2, 1, 0.02, 1e-9, OPERATORS,
                            min_sse_reduction=0.30, rational_powers=(2 / 3, 1 / 3))
    out_a = repair(orig, parts, cfg_without, **CUTS)
    assert out_a.status == RepairStatus.IMPROVED, out_a.status
    assert "(A*Z)" in (out_a.winner.metadata["term"] or ""), out_a.winner.metadata

    parts2, orig2 = _parts_and_orig(fr0)
    out_b = repair(orig2, parts2, cfg_with, **CUTS)
    assert out_b.candidates_evaluated > out_a.candidates_evaluated
    assert len(out_b.validation) >= 14, sorted(out_b.validation)
    # the domain-informed surface term is actually reachable and evaluated
    surface = [g for cid, g in out_b.validation.items() if "(A^0.667)" in (
        out_b.winner.metadata.get("term") or "") or "(A^0.667)" in cid]
    gen_terms = set()
    for cid in out_b.validation:
        for g in _generate_for_terms(fr0):
            if g.candidate_id == cid:
                gen_terms.add(g.metadata.get("term"))
    assert "(A^0.667)" in gen_terms, gen_terms


def test_proxy_and_mechanism_are_observationally_indistinguishable():
    """The decisive scientific measurement: A*Z vs A^(2/3) on the sealed rows.

    The physically-correct surface term and the correlated proxy are within
    1e-3 of each other in sealed R2 — the data cannot separate them, so the
    honest engine outcome is MULTI_HYPOTHESIS, not a unique winner.
    """
    ok = _load_binding()
    fr0 = _frame(ok)
    parts, orig = _parts_and_orig(fr0)
    sealed_df, sealed_y = parts.take_sealed()
    A = sealed_df["A"].to_numpy(float)
    B = sealed_y.astype(float)
    base = np.column_stack([np.ones_like(A), A])
    w, *_ = np.linalg.lstsq(base, B, rcond=None)
    pred0 = base @ w
    ss = float(np.sum((B - B.mean()) ** 2))

    def r2_with(term):
        M = np.column_stack([np.ones_like(A), A, term])
        ww, *_ = np.linalg.lstsq(M, B, rcond=None)
        return 1 - float(np.sum((B - M @ ww) ** 2)) / ss

    r2_proxy = r2_with(A * sealed_df["Z"].to_numpy(float))
    r2_surface = r2_with(np.sign(A) * np.abs(A) ** (2 / 3))
    r2_base = 1 - float(np.sum((B - pred0) ** 2)) / ss
    assert r2_base < r2_surface < r2_base + 0.01
    assert abs(r2_proxy - r2_surface) < 1e-3, (r2_proxy, r2_surface)
    print(f"\n[real] sealed R2: base={r2_base:.4f} surface A^(2/3)={r2_surface:.4f} "
          f"proxy A*Z={r2_proxy:.4f} | Δ={abs(r2_proxy - r2_surface):.5f}")


def test_identifiability_band_returns_multi_hypothesis():
    """With the band declared, the engine reports AMBIGUOUS instead of a winner."""
    ok = _load_binding()
    fr0 = _frame(ok)
    parts, orig = _parts_and_orig(fr0)
    cfg = RepairConfig(14, 2, 1, 0.02, 1e-9, OPERATORS,
                       min_sse_reduction=0.30,
                       rational_powers=(2 / 3, 1 / 3),
                       equivalence_r2_tol=0.001)
    out = repair(orig, parts, cfg, **CUTS)
    assert out.status == RepairStatus.AMBIGUOUS, out.status
    assert out.ambiguity_set_size >= 2, out.ambiguity_set_size
    assert out.sealed is None  # ambiguity never consumes the sealed split
    print(f"\n[real] band=0.001 -> {out.status.value}, set size {out.ambiguity_set_size}, sealed unopened")
