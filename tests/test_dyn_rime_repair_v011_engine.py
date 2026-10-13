"""RIME Repair engine benchmark (§7): planted defects + controls, no-repair baseline.

Categories: parameter error → R1 refit; missing curvature → R2 insertion;
missing lag → R3 insertion; irreducible noise → no repair; equivalent
equations → ambiguity preserved (R4 machinery). Every test records outcome
status, sealed improvement over the frozen original, candidates evaluated,
and rounds used. A repair that cannot beat the original on sealed rows is a
failure, reported as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic import rime_repair_diagnosis as diagmod
from apedp_dyn.epistemic.experiment_ledger import HoldoutContaminationError
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    RepairStatus,
    cluster_equivalent,
    repair,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal
from apedp_dyn.proposers.rime_proposer import RimeProposer

BIAS_RATIO_CUT = 0.25
VARIANCE_RATIO_CUT = 4.0
CURVATURE_CORR_CUT = 0.3
MIN_ROWS = 30

CONFIG = RepairConfig(
    max_candidates_per_round=6,
    max_rounds=2,
    max_added_terms=1,
    min_validation_r2_gain=0.02,
    equivalence_tol=1e-9,
    operators=(RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT, RepairOperator.R3_LAG_INSERT),
)

CUTS = dict(rho_flag=RESID_RHO_FLAG, bias_ratio_cut=BIAS_RATIO_CUT,
            variance_ratio_cut=VARIANCE_RATIO_CUT, curvature_corr_cut=CURVATURE_CORR_CUT,
            min_rows=MIN_ROWS)


def _discover(df: pd.DataFrame, **kw) -> CandidateProposal:
    kw.setdefault("feature_cols", ["x"])
    kw.setdefault("target_col", "target")
    kw.setdefault("max_depth", 1)
    kw.setdefault("budget", 120)
    props = RimeProposer(**kw).fit_and_propose(df)
    assert len(props) > 0, "discovery must emit on planted structure"
    return sorted(props, key=lambda p: p.parameters["dev_r2"], reverse=True)[0]


def test_repair_parameter_error_refit():
    # Discovery regime y=3x[t-2]+10; later splits shift intercept by +5.
    # R1 must refit; structure is already right.
    rng = np.random.RandomState(21)
    n = 1000
    x = rng.randn(n)
    shift = np.where(np.arange(n) < 400, 0.0, 5.0)
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + shift[2:] + 0.01 * rng.randn(n - 2)
    df = pd.DataFrame({"x": x, "target": y})
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    assert parts.spans["discovery"] == (0, 400)
    orig = _discover(parts.discovery_df.assign(target=parts.discovery_y))
    out = repair(orig, parts, CONFIG, **CUTS)
    assert out.status == RepairStatus.IMPROVED, out.submit_note
    assert out.winner is not None and out.winner.metadata["operator"] == "R1_PARAM_REFIT"
    assert out.targeted_finding == diagmod.FINDING_INTERCEPT_BIAS
    assert out.sealed is not None and out.sealed["sealed_r2_gain"] > 0
    assert out.sealed["winner_r2"] >= out.sealed["original_r2"]  # no-repair baseline
    assert out.ambiguity_set_size == 1
    print(f"\n[param] sealed gain={out.sealed['sealed_r2_gain']:.4f} evals={out.candidates_evaluated} rounds={out.rounds_used}")


def test_repair_missing_curvature_inserts_square():
    rng = np.random.RandomState(22)
    n = 1000
    x = rng.randn(n)
    y = 2.0 * x + 0.8 * x**2 + 0.05 * rng.randn(n)
    df = pd.DataFrame({"x": x, "target": y})
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    orig = _discover(parts.discovery_df.assign(target=parts.discovery_y), max_depth=1)
    out = repair(orig, parts, CONFIG, **CUTS)
    assert out.status == RepairStatus.IMPROVED, out.submit_note
    assert out.winner is not None and out.winner.metadata["operator"] == "R2_TERM_INSERT"
    assert out.targeted_finding == diagmod.FINDING_CURVATURE_SUSPECT
    assert out.sealed is not None and out.sealed["sealed_r2_gain"] > 0
    print(f"\n[curv] sealed gain={out.sealed['sealed_r2_gain']:.4f} evals={out.candidates_evaluated} rounds={out.rounds_used}")


def test_repair_missing_lag_inserts_lag():
    # Design rationale (measured): omitted lags of the SAME driver are
    # absorbed by OLS into ~white residuals (orthogonality), and anything
    # expressible as a depth-1 seed RIME finds in discovery. The honest R3
    # case is EMERGENT lag dynamics: the discovery regime has no lag signal
    # (x2 irrelevant), the later regime gains a 2*x2[t-1] term
    # (nonstationarity). The independent AR driver survives unabsorbed in
    # the residuals, and R3 reaches it because repair terms derive from ALL
    # declared search features. Stated as emergent-lag repair, nothing else.
    rng = np.random.RandomState(23)
    n = 1000
    x1 = rng.randn(n)
    w = np.zeros(n + 200)
    for t in range(1, n + 200):
        w[t] = 0.9 * w[t - 1] + rng.randn()
    x2 = w[200:]
    lagterm = np.where(np.arange(n) < 400, 0.0, 2.0 * np.roll(x2, 1))
    y = 3.0 * x1 + lagterm + 5.0 + 0.05 * rng.randn(n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    props = RimeProposer(feature_cols=["x1", "x2"], target_col="target", max_depth=1, budget=120).fit_and_propose(tr)
    assert len(props) > 0
    orig = sorted(props, key=lambda p: p.parameters["dev_r2"], reverse=True)[0]
    out = repair(orig, parts, CONFIG, **CUTS)
    assert out.status == RepairStatus.IMPROVED, (out.status, out.submit_note, out.validation)
    assert out.winner is not None and out.winner.metadata["operator"] == "R3_LAG_INSERT"
    assert out.targeted_finding == diagmod.FINDING_TEMPORAL_DEPENDENCE
    assert out.sealed is not None and out.sealed["sealed_r2_gain"] > 0
    print(f"\n[lag] sealed gain={out.sealed['sealed_r2_gain']:.4f} evals={out.candidates_evaluated} rounds={out.rounds_used}")


def test_repair_noise_abstains_and_sealed_stays_shut():
    rng = np.random.RandomState(99)
    n = 600
    df = pd.DataFrame({"x": rng.randn(n), "target": rng.randn(n)})
    props = RimeProposer(feature_cols=["x"], target_col="target", max_depth=2, budget=120).fit_and_propose(df)
    assert len(props) == 0  # no original → no repair, no baseline to beat
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    parts.take_sealed()  # single legitimate access
    with pytest.raises(HoldoutContaminationError):
        parts.take_sealed()  # §4 enforcement: sealed never reopens


def test_repair_equivalence_preserved_by_clustering():
    # Two formula-distinct, prediction-identical fixtures + one outsider.
    rng = np.random.RandomState(31)
    n = 200
    x = rng.randn(n)
    base = 3.0 * x + 10.0
    mk = lambda form, off: CandidateProposal(
        proposer_id="fixture", candidate_id=f"fix_{form}", formula=form,
        strategy_archetype=None, declared_scale_invariant=True,
        compute_score=(lambda df, _b=base[:len(df)] if len(base) >= len(df) else base: np.asarray(_b[:len(df)], dtype=float) + off),
        parameters={}, metadata={})
    df = pd.DataFrame({"x": x})
    y = base + rng.randn(n) * 0.01
    a = mk("a", 0.0)
    b = mk("b", 0.0)  # identical predictions to a
    c = mk("c", 5.0)  # outsider
    clusters = cluster_equivalent([a, b, c], df, y, tol=1e-9)
    sizes = sorted(len(cl) for cl in clusters)
    assert sizes == [1, 2], sizes  # ambiguity preserved, outsider separate


def test_repair_adequate_original_not_falsely_repaired():
    rng = np.random.RandomState(11)
    n = 1000
    x = rng.randn(n)
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + 0.01 * rng.randn(n - 2)
    df = pd.DataFrame({"x": x, "target": y})
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    orig = _discover(parts.discovery_df.assign(target=parts.discovery_y))
    out = repair(orig, parts, CONFIG, **CUTS)
    # Adequate original: either no finding (NO_IMPROVEMENT, zero candidates)
    # or R1 refit attempted and rejected by the gates. Never a false repair.
    assert out.status in (RepairStatus.NO_IMPROVEMENT, RepairStatus.INSUFFICIENT_EVIDENCE), out.status
    assert out.winner is None
    assert out.sealed is None  # sealed never opened without a validated winner
    print(f"\n[adequate] status={out.status.value} evals={out.candidates_evaluated}")
