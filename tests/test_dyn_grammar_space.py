"""Grammar-space sandbox pins (doc §81): registry, domains, causality,
combos, promotion pipeline, frozen §80 record. Builders cross-checked
against the established e5_market panels (identical series).
"""

import numpy as np
import pandas as pd
import pytest

import apedp_dyn.epistemic.active_disambiguation.e5_market as M
from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.epistemic.novelty_adjudication import evaluate_predictive_incrementality
from apedp_dyn.sandbox import grammar_space as G
from apedp_dyn.sandbox import multi_grammar_sandbox as MG


def _btc():
    frames = M.load_frames()
    return M.build_asset_panel(frames["raw"], "BTC")


def test_registry_and_combos():
    assert set(G.PRIMITIVES) == {"r_t", "V_t"}
    assert set(G.BINARY_OPS) == {"+", "-", "*", "/", "min", "max"}
    assert {"sqrt", "tanh"} <= set(G.UNARY_OPS)
    assert G.check_sqrt(np.array([0.0, 4.0])).ok
    assert not G.check_sqrt(np.array([0.0, -1.0])).ok
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
        monkeypatch.setattr(signals, "load_universe",
                            lambda _path, p=panels: (p, list(p)))  # p: bind loop var (B023)
        _, candidates = MG.run_pooled_multi_asset_discovery(bootstrap=False)
        assert candidates
        losses.append(candidates[0].train_loss)

    np.testing.assert_allclose(losses[0], losses[1])


def _synthesized_panel(seed: int = 11, n: int = 600):
    """Panel whose volume column matters and whose returns are inert.

    `r` is pure noise (so none of the return-driven hypotheses is secretly the
    truth) and `V` is a wide log-normal, which is what makes a linear-in-V and
    a sqrt(V) law distinguishable rather than near-collinear.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, n)
    V = rng.lognormal(3.0, 0.5, n)
    panel = pd.DataFrame({
        "r": r, "V": V, "ofi": r + rng.normal(0.0, 0.005, n),
        "price": 100.0 * np.exp(np.cumsum(r)),
    })
    return panel, rng


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, float) ** 2)))


def _holdout_ols(feat: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Second-half holdout predictions under the sandbox's own split.

    Mirrors `MG._fit_grammar_candidate`: train on `[0, n//2 - 5)`, score on
    `[n//2, n)`. Used for the linear-in-V control, which is deliberately NOT a
    member of the frozen grammar; the caller asserts the mirrored figure equals
    the sandbox's reported holdout RMSE, so this helper cannot drift away from
    the implementation it is standing in for.
    """
    n = len(target)
    n_train = n // 2
    train_end = max(0, n_train - 5)
    X = np.column_stack([np.ones(n), feat])
    w, *_ = np.linalg.lstsq(X[:train_end], target[:train_end], rcond=None)
    return X[n_train:] @ w, target[n_train:]


def test_h_sqrtv_negative_control_on_a_linear_level_volume_law():
    """Negative control: H-SQRTV is not selected when the law is linear in V.

    The concave level-volume hypothesis is only useful if it beats a
    linear-in-level law on data that is actually concave. This pin plants
    `y = 1 + 2V + noise` and asks two referee questions under the sandbox's own
    protocol (same train/holdout split as `_fit_grammar_candidate`, holdout-RMSE
    ranking, Gate-4 incrementality at the canonical alpha):

      (a) on the linear law the linear-in-V control is selected and H-SQRTV is
          rejected as an increment (DM statistic on the wrong side of zero);
      (b) on `y = 1 + 2sqrt(V) + noise` the same machinery selects H-SQRTV and
          certifies it incremental -- so the control is a discriminator, not a
          candidate that can never win.

    Measured context (this fixture): H-SQRTV holdout RMSE is ~86x the linear
    control on the linear law and the linear control is ~8.6x H-SQRTV on the
    concave law. Within the *declared* frozen pool (which has no linear-in-V
    member) H-SQRTV is the closest available approximation to a linear law --
    that is model starvation, not a discovery, which is exactly why the
    control has to stand outside the pool for this question.
    """
    panel, rng = _synthesized_panel()
    n = len(panel)
    V = panel["V"].to_numpy(float)
    alpha = DIEBOLD_MARIANO_P_VALUE_THRESHOLD

    def arm(target: np.ndarray):
        cand = MG._fit_grammar_candidate(panel, "H-SQRTV", target=target)
        assert cand is not None, "H-SQRTV could not be fitted on the synthetic panel"
        resid = np.asarray(cand.provenance["gate4_status"]["holdout_residuals"], float)
        pred_linear, y_ho = _holdout_ols(V, target)
        # Same holdout block and the same all-finite mask: the residual vector
        # the sandbox reports must line up row-for-row with the control's.
        assert resid.size == y_ho.size
        rmse_sqrtv = _rmse(resid)
        rmse_linear = _rmse(y_ho - pred_linear)
        # The mirror is only honest if it reproduces the sandbox's number.
        np.testing.assert_allclose(cand.holdout_rmse, rmse_sqrtv, atol=0.0, rtol=1e-12)
        verdict = evaluate_predictive_incrementality(
            y_ho, y_ho - resid, pred_linear, alpha=alpha)
        selected = min((("H-SQRTV", rmse_sqrtv), ("LINEAR_V", rmse_linear)),
                       key=lambda kv: kv[1])[0]
        return rmse_sqrtv, rmse_linear, verdict, selected

    noise = 0.05

    # (a) linear law: the concave family must lose, both in ranking and in Gate 4.
    y_linear = 1.0 + 2.0 * V + rng.normal(0.0, noise, n)
    rmse_sqrtv, rmse_linear, verdict, selected = arm(y_linear)
    assert selected == "LINEAR_V"
    assert rmse_sqrtv > 10.0 * rmse_linear, (rmse_sqrtv, rmse_linear)
    assert verdict.verdict != "PREDICTIVELY_INCREMENTAL", verdict.verdict
    assert verdict.dm_statistic < 0.0, verdict.dm_statistic

    # (b) concave law: same machinery, opposite answer (the control bites).
    y_sqrt = 1.0 + 2.0 * np.sqrt(V) + rng.normal(0.0, noise, n)
    rmse_sqrtv_c, rmse_linear_c, verdict_c, selected_c = arm(y_sqrt)
    assert selected_c == "H-SQRTV"
    assert rmse_linear_c > 4.0 * rmse_sqrtv_c, (rmse_linear_c, rmse_sqrtv_c)
    assert verdict_c.verdict == "PREDICTIVELY_INCREMENTAL", verdict_c.verdict


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
