"""Grammar-space sandbox pins (doc §81): registry, domains, causality,
combos, promotion pipeline, frozen §80 record. Builders cross-checked
against the established e5_market panels (identical series).
"""

import numpy as np
import pandas as pd
import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.sandbox import grammar_space as G


def _btc():
    frames = M.load_frames()
    return M.build_asset_panel(frames["raw"], "BTC")


def test_registry_and_combos():
    assert set(G.PRIMITIVES) == {"r_t", "V_t"}
    assert set(G.BINARY_OPS) == {"+", "-", "*", "/"}
    assert set(G.COMBOS) == {"M_P", "M_RV", "M_A", "M_P+RV", "M_P+A", "M_RV+A", "M_P+RV+A"}
    assert G.COMBOS["M_P+RV+A"] == ("H-P", "H-RV", "H-A")


def test_domain_checks():
    assert G.check_log(np.array([1.0, 2.0])).ok
    assert not G.check_log(np.array([1.0, 0.0])).ok  # x>0, not log(x)>0
    assert not G.check_div(np.ones(3), np.array([1.0, 0.0, 1.0])).ok
    assert not G.check_zscore(np.ones(5)).ok
    assert G.check_zscore(np.array([1.0, 2.0, 3.0])).ok


def test_builders_match_established_panels():
    btc = _btc()
    feats = G.build_features(btc)
    np.testing.assert_allclose(feats["sig"], btc["sig"].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(feats["rankV"], btc["rankV"].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(feats["HA"], btc["HA"].to_numpy(), equal_nan=True)
    tgt = G.build_target(btc)
    d5 = pd.Series(np.abs(btc["r"].to_numpy())).shift(-1).rolling(5).mean().to_numpy()
    np.testing.assert_allclose(tgt, d5, equal_nan=True)


def test_causality_warmup_nans():
    btc = _btc()
    feats = G.build_features(btc)
    assert bool(np.isnan(feats["rankV"][:39]).all())  # window-40 needs history
    assert bool(np.isnan(feats["HA"][:59]).all())  # min-history 60 for z-scores


def test_promotion_pipeline():
    table = G.decision_summary()
    assert table["H-P"].level == 4 and "HELD" in table["H-P"].verdict
    assert table["H-RV"].level == 1 and table["H-RV"].failed_at == "L2-incremental"
    assert table["H-A"].level == 1 and table["H-A"].failed_at == "L2-incremental"
    # Later gates inadmissible: full-house evidence except L2 still held at L2.
    ev = G.GateEvidence(True, True, True, False, True, True, True)
    v = G.promote("X", ev)
    assert v.level == 1 and v.failed_at == "L2-incremental"
    # Full pass promotes.
    v2 = G.promote("X", G.GateEvidence(True, True, True, True, True, True, True))
    assert v2.level == 5 and "PROMOTED" in v2.verdict


def test_incremental_equations_named():
    b = G.BatterySummary()
    assert any("A|P,RV" in e for e in b.incremental_eqs)
    assert any("RV|P,A" in e for e in b.incremental_eqs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
