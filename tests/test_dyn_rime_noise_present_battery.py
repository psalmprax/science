"""Noise-present battery: every headline number must have a noisy counterpart.

Why this file exists. The clean-data results for the rational/Padé work are
exact by construction -- the tested functional form IS the generator, so R2
= 1.000000 and sealed RMSE ~ 4e-17 are exactness demonstrations, not
discovery under measurement. This suite records the same pipelines at 0.5/2/5/10
percent observation noise and pins three rules:

  1. MONOTONE DEGRADATION: reported fit must fall as noise rises, so the
     clean-data figure can never be misquoted as the noisy one.
  2. RECOVERY SURVIVES REAL NOISE: the QEC lag-zero square -- a genuine
     capability, since the frozen grammar cannot express it -- must remain
     structurally recoverable at 2% noise.
  3. NO QUOTABLE PHANTOM: at >=2% noise the Padé operator DOES fit a
     denominator on pure quadratic data (measured b*k = 62, 221). The default
     policy must refuse to emit it; the audit path keeps the evidence visible
     but unquotable.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from apedp_dyn.epistemic import rime_repair_diagnosis as diagmod
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    generate_repairs,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype
from apedp_dyn.proposers.rime_proposer import RimeProposer

NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10)
P = np.linspace(0.001, 0.025, 400)
C_NUM, C_DEN = 1000.0, 2000.0

R5_OPS = (RepairOperator.R1_PARAM_REFIT, RepairOperator.R5_PADE_SATURATION)
R5_CONFIG = RepairConfig(6, 1, 1, 0.02, 1e-9, R5_OPS)
R5_CONFIG_AUDIT = RepairConfig(6, 1, 1, 0.02, 1e-9, R5_OPS, r5_emit_unidentified=True)


def _pade_observations(noise: float, seed: int = 42) -> np.ndarray:
    clean = (C_NUM * P ** 2) / (1.0 + C_DEN * P ** 2)
    if noise == 0.0:
        return clean
    r = np.random.default_rng(seed)
    return clean + r.normal(0, noise * clean.std(), P.size)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1 - float(np.sum((y_true - y_pred) ** 2)) / ss if ss > 0 else float("nan")


def _parent(k: float) -> CandidateProposal:
    return CandidateProposal(
        proposer_id="t", candidate_id="c", formula=f"{k:.4e} * p^2",
        strategy_archetype=StrategyArchetype.TACTICAL, declared_scale_invariant=True,
        declared_nominal_dependencies=("p",),
        compute_score=lambda df, _k=k: _k * (df["p"].to_numpy() ** 2),
        parameters={}, metadata={})


def _pade(frame: pd.DataFrame, k: float, config=None) -> list[CandidateProposal]:
    reps = generate_repairs(_parent(k), [diagmod.FINDING_CURVATURE_SUSPECT], frame,
                            frame["target"].to_numpy(), config or R5_CONFIG)
    return [r for r in reps if r.metadata.get("operator") == "R5_PADE_SATURATION"]


def test_pade_noise_curve_is_monotone_and_recorded():
    """Rule 1: clean-data R2 can never stand in for the noisy one."""
    curve = []
    for noise in NOISE_LEVELS:
        obs = _pade_observations(noise)
        df = pd.DataFrame({"p": P, "target": obs})
        tr, te = df.iloc[:len(df) // 2], df.iloc[len(df) // 2:]
        k = float(np.polyfit(tr["p"] ** 2, tr["target"], 1)[0])
        # Diagnostic curve: needs the candidates the default refuses to emit.
        cand = _pade(tr, k, R5_CONFIG_AUDIT)
        assert cand, noise
        r2 = _r2(te["target"].to_numpy(), cand[0].compute_score(te))
        curve.append((noise, r2))
        refused = _pade(tr, k, R5_CONFIG) == []
        print(f"\n[pade-noise] noise={noise:.3f} holdoutR2={r2:.6f} default-emits={not refused}")
    r2s = [r for _n, r in curve]
    assert r2s[0] > 0.9, curve                       # exact construction
    assert all(a >= b - 1e-9 for a, b in itertools.pairwise(r2s)), curve
    assert r2s[-1] < r2s[0] - 1e-3, curve             # must visibly fall
    assert 0.5 < r2s[1] < 1.0, curve                  # the honest 2% figure


def test_qec_square_recovers_structurally_at_two_percent_noise():
    """Rule 2: a genuine capability must survive real observation noise."""
    r = np.random.default_rng(7)
    clean = 1000.0 * P ** 2
    y = clean + r.normal(0, 0.02 * clean.std(), P.size)
    frame = pd.DataFrame({"p": P, "target": y})
    props = RimeProposer(feature_cols=["p"], target_col="target", max_depth=1,
                         budget=200, static_mode=True,
                         square_seeds=True).fit_and_propose(frame)
    assert props, "square_seeds must still recover at 2% noise"
    best = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    eq = best.metadata.get("equation") or ""
    holdout_r2 = _r2(y[len(y) // 2:], best.compute_score(frame)[len(y) // 2:])
    print(f"\n[qec-noise] 2% noise -> {eq}  holdoutR2={holdout_r2:.4f}")
    assert holdout_r2 > 0.95, holdout_r2
    assert "[t-" not in eq and "state_" not in eq, eq   # still the lag-0 form


def test_no_phantom_rationality_is_quotable():
    """Rule 3: a phantom denominator must not be emitted as a candidate."""
    r = np.random.default_rng(9)
    clean = C_NUM * P ** 2
    tr = pd.DataFrame({"p": P, "target": clean}).iloc[:len(P) // 2]
    k = float(np.polyfit(tr["p"] ** 2, tr["target"], 1)[0])
    assert _pade(tr, k, R5_CONFIG) == [], "clean data must emit no Pade candidate"
    for noise in (0.02, 0.05):
        y = clean + r.normal(0, noise * clean.std(), P.size)
        t = pd.DataFrame({"p": P, "target": y}).iloc[:len(P) // 2]
        kk = float(np.polyfit(t["p"] ** 2, t["target"], 1)[0])
        assert _pade(t, kk, R5_CONFIG) == [], noise      # phantom refused
        audited = _pade(t, kk, R5_CONFIG_AUDIT)
        if audited:
            q = audited[0]
            den = q.parameters.get("b", 0.0) * kk
            print("\n[phantom] noise={:.2f} phantom b*k={:.3e} status={} (refused by default)".format(noise, den, q.parameters["identifiability_status"]))
            if den >= 1e-3:
                assert q.parameters["identifiability_status"] == "PREDICTIVE_ONLY_ILL_CONDITIONED"
