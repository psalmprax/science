"""RIME Repair on QEC threshold data: gate-validity boundary (pinned behavior).

Measured behavior (seeds fixed): a degraded LINEAR original on the
quadratic threshold law yields an R2 square candidate at validation
R²=0.9545 / gain +3.26 — yet the outcome is NO_IMPROVEMENT because the
diagnostic gate raises a fresh temporal flag on the validation rows.

Root cause, measured: p is LOG-SPACED, so quadratic extrapolation from the
repair split grows smoothly into the validation split; the growing
extrapolation misfit is autocorrelated in index order and trips the
whiteness screen. The screen assumes stationarity across evaluation rows —
valid for time series, misfiring under geometric-spacing extrapolation.

This test PINS the conservative behavior (refuse rather than promote) and
documents the validity condition. It does not change gate semantics: the
safe failure direction is refusal, and any respecification needs its own
preregistered benchmark. Consistent with the sandbox's own MULTI stance
between the algebraically identical threshold_ratio / plain_quad forms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    RepairStatus,
    repair,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal
from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.sandbox.qec_threshold import generate_qec_threshold_data

CUTS = dict(rho_flag=RESID_RHO_FLAG, bias_ratio_cut=0.25,
            variance_ratio_cut=4.0, curvature_corr_cut=0.3, min_rows=30)

CONFIG = RepairConfig(
    max_candidates_per_round=6,
    max_rounds=2,
    max_added_terms=1,
    min_validation_r2_gain=0.02,
    equivalence_tol=1e-9,
    operators=(RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT, RepairOperator.R3_LAG_INSERT),
)


def _qec_frame(n_points: int = 600, seed: int = 42):
    p, obs = generate_qec_threshold_data(n_points=n_points, seed=seed)
    return pd.DataFrame({"p": p, "target": obs})


def test_qec_degraded_linear_repair_refused_despite_high_r2():
    df = _qec_frame()
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    dtr, ytr = parts.discovery_df, parts.discovery_y
    A = np.column_stack([dtr["p"].to_numpy(), np.ones(len(dtr))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)

    def lin_score(dfx, _w=w):
        return _w[0] * np.asarray(dfx["p"], dtype=float) + _w[1]

    orig = CandidateProposal(proposer_id="fixture_degraded_linear", candidate_id="lin_p",
                             formula=f"{w[0]:+.6g}*(p){w[1]:+.6g}", strategy_archetype=None,
                             declared_scale_invariant=True, declared_nominal_dependencies=("p",),
                             compute_score=lin_score, parameters={}, metadata={"fixture": True})
    out = repair(orig, parts, CONFIG, **CUTS)
    # Pinned conservative behavior: the strong R2 candidate exists (check the
    # gate table) but the fresh temporal flag vetoes promotion.
    assert out.status == RepairStatus.NO_IMPROVEMENT, out.status
    r2_rows = [g for cid, g in out.validation.items() if "R2_TERM_INSERT" in cid]
    assert len(r2_rows) == 1
    assert r2_rows[0].val_r2 > 0.9, r2_rows[0]  # strong yet refused
    assert r2_rows[0].new_temporal_flag is True  # the veto, explicitly
    assert out.winner is None and out.sealed is None  # sealed never opened
    print(f"\n[qec] R2 val-R2={r2_rows[0].val_r2:.4f} refused via new_temporal_flag; sealed=shut")


def test_qec_rime_discovery_repair_disciplined():
    df = _qec_frame()
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    props = RimeProposer(feature_cols=["p"], target_col="target", max_depth=3, budget=200).fit_and_propose(tr)
    assert len(props) > 0
    orig = sorted(props, key=lambda p: p.parameters["dev_r2"], reverse=True)[0]
    out = repair(orig, parts, CONFIG, **CUTS)
    # Either principled non-repair (sealed shut) or a sealed-validated win.
    assert out.status in (RepairStatus.IMPROVED, RepairStatus.NO_IMPROVEMENT,
                         RepairStatus.AMBIGUOUS, RepairStatus.INSUFFICIENT_EVIDENCE), out.status
    if out.status == RepairStatus.IMPROVED:
        assert out.sealed is not None and out.sealed["sealed_r2_gain"] > 0
    else:
        assert out.sealed is None
    print(f"\n[qec-discovery] orig={orig.metadata.get('equation')} outcome={out.status.value} evals={out.candidates_evaluated}")
