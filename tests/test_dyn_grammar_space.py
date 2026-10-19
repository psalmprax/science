"""Grammar-space sandbox pins (doc §81): registry, domains, causality,
combos, promotion pipeline, frozen §80 record. Builders cross-checked
against the established e5_market panels (identical series).
"""

import numpy as np
import pandas as pd
import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.sandbox import multi_grammar_sandbox as MG
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
    # Forward-only d5 target (Finding 1 fix, causal contract): the aggregate
    # over s[t+1 .. t+5], i.e. computed by rolling first and shifting -5. The
    # pins never accept the old backward window s.shift(-1).rolling(5), which
    # folded s[t-3 .. t-1] -- 80% past/current data -- into the "future" target.
    d5 = pd.Series(np.abs(btc["r"].to_numpy())).rolling(5).mean().shift(-5).to_numpy()
    np.testing.assert_allclose(tgt, d5, equal_nan=True)
    # The trailing 5 rows must be NaN: an aligned t admits no future window.
    assert bool(np.isnan(tgt[-5:]).all())
    assert bool(np.isfinite(tgt[:-5]).all())


@pytest.mark.parametrize(
    "fit",
    [
        lambda panel: MG._fit_grammar_candidate(panel, "H-P"),
        lambda panel: MG._fit_combo_candidate(panel, "M_P"),
    ],
)
def test_candidate_fit_purges_forward_target_overlap(fit):
    rng = np.random.default_rng(23)
    r = rng.normal(0, 0.01, 120)
    changed = r.copy()
    changed[60] += 1.0
    panels = [pd.DataFrame({"r": values}) for values in (r, changed)]

    scores = [fit(panel) for panel in panels]

    assert all(score is not None for score in scores)
    weights = [score.provenance["gate4_status"]["weights"] for score in scores]
    np.testing.assert_allclose(weights[0], weights[1])


def test_pooled_fit_purges_forward_target_overlap(monkeypatch):
    import apedp_dyn.sandbox.multi_asset_signals_sandbox as signals

    rng = np.random.default_rng(29)
    universe = {
        asset: pd.DataFrame({"r": rng.normal(0, 0.01, 120)})
        for asset in ("A", "B")
    }
    changed = {asset: panel.copy() for asset, panel in universe.items()}
    changed["A"].loc[60, "r"] += 1.0
    monkeypatch.setattr(signals, "SIGNAL_NAMES", ["test_signal"])
    monkeypatch.setattr(signals, "calc_signals_for_asset",
                        lambda panel, median: np.arange(len(panel), dtype=float)[:, None])
    monkeypatch.setattr(MG, "adjudicate", lambda *args, **kwargs: None)

    losses = []
    for panels in (universe, changed):
        monkeypatch.setattr(signals, "load_universe", lambda _: (panels, list(panels)))
        _, candidates = MG.run_pooled_multi_asset_discovery(bootstrap=False)
        assert candidates
        losses.append(candidates[0].train_loss)

    np.testing.assert_allclose(losses[0], losses[1])


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
