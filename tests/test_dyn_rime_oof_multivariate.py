"""Registered pins for RIME-OOF-MV (PREREG-RIME-OOF-MV-2026-10-17). audit-only.

Pins the proposer-layer outcome of the registered run
(`benchmarks/rime_oof_multivariate_results.json`, report
`docs/RIME_OOF_MV_REPORT.md`): RimeProposer recovers the in-family control, and
on each multivariate out-of-family law it emits confident surrogates (no
abstention) whose wide-range R2 collapses. RimeAdProposer is excluded here by
design (56-77 s per multivariate fit); its registered NO_SIGNAL outcome lives
in the benchmark JSON.

Tolerance bands are set from the registered run (deterministic defaults) with
margin; a failure means the registered outcome moved and the study must be
re-run/re-preregistered rather than patched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.run_rime_oof_multivariate_probe import LAWS, _r2, make_pop, run_rime


def _best_by_test(props, df_te, df_fr, df_wide):
    scored = []
    for cand in props:
        te = _r2(df_te["target"].to_numpy(float), np.asarray(cand.compute_score(df_te), float).ravel())
        fr = _r2(df_fr["target"].to_numpy(float), np.asarray(cand.compute_score(df_fr), float).ravel())
        wd = _r2(df_wide["target"].to_numpy(float), np.asarray(cand.compute_score(df_wide), float).ravel())
        scored.append((te, fr, wd, cand))
    return max(scored, key=lambda t: (t[0] if np.isfinite(t[0]) else -9))


@pytest.fixture(scope="module")
def rime_runs():
    out = {}
    for name, law in LAWS.items():
        df_tr = make_pop(600, law).iloc[:400].reset_index(drop=True)
        df_te = make_pop(600, law).iloc[400:].reset_index(drop=True)
        df_fr = make_pop(601, law)
        df_wide = make_pop(603, law, scale=2.5)
        props, _ms = run_rime(df_tr)
        out[name] = (props, _best_by_test(props, df_te, df_fr, df_wide))
    return out


def test_control_recovers_exactly(rime_runs) -> None:
    props, (test, _fresh, wide, cand) = rime_runs["control_in_family"]
    assert len(props) > 0
    assert test >= 0.99 and wide >= 0.95
    assert "x1[t-1] * x2[t-1]" in cand.formula
    assert "x3[t-2]" in cand.formula
    assert cand.parameters.get("identifiability_status") == "UNIQUE_WINNER"


def test_no_abstention_on_any_out_of_family_law(rime_runs) -> None:
    for name in ("oof_fn_of_interaction", "oof_max_switch", "oof_amplitude_mod"):
        props, _ = rime_runs[name]
        assert len(props) > 0, f"{name}: proposer abstained (registered outcome: emitted)"


def test_sin_of_interaction_surrogate_collapses_out_of_range(rime_runs) -> None:
    _props, (test, _fresh, wide, cand) = rime_runs["oof_fn_of_interaction"]
    assert test >= 0.85
    assert wide <= 0.70
    assert (test - wide) >= 0.20
    assert "tanh" in cand.formula


def test_max_switch_surrogate_emitted_but_partial(rime_runs) -> None:
    _props, (test, _fresh, wide, _cand) = rime_runs["oof_max_switch"]
    assert test >= 0.45
    assert wide <= 0.75


def test_amplitude_modulation_surrogate_collapses_out_of_range(rime_runs) -> None:
    _props, (test, _fresh, wide, cand) = rime_runs["oof_amplitude_mod"]
    assert test >= 0.55
    assert wide <= 0.35
    assert (test - wide) >= 0.30
    assert "tanh" in cand.formula
