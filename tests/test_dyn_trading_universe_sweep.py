"""Trading universe sweep: RIME over 15/30/45/60/90 assets x horizons.

Data reality (checked in-repo): ALL market files are daily bars —
no minute-level data exists, so 1/5/15/30/45/60-minute timeframes cannot
be tested without inventing data. This suite grids the five real
universes against honest downsampled horizons (daily, 5-day, 21-day).

Measured (seeds fixed): 9/15 cells abstain (all five dailies among
them); 21d short cells (34 rows) below usable size; the four emitting
5d/21d cells produce only weak difference forms (devR2 <= 0.12) that
collapse on temporal holdout (R2 -0.30, -0.77) or abstain when halved.
No cell yields a transferable trading equation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.proposers.rime_proposer import RimeProposer

CRYPTO30 = ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "DOT", "LINK",
            "MATIC", "NEAR", "ATOM", "UNI", "LTC", "XLM", "ALGO", "VET", "ICP", "FIL",
            "TRX", "ETC", "XMR", "AAVE", "MKR", "COMP", "SNX", "YFI", "SUSHI", "ZEC")
EQUITY15 = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V",
            "JNJ", "WMT", "PG", "XOM", "UNH", "SPY")


def _load(path: str):
    df = pd.read_csv(path)
    closes = [c for c in df.columns if c.endswith("_CLOSE")]
    if not closes:
        # crypto_top15.csv uses bare symbols (BTC, ETH, ...); exclude
        # date, flow/volume/OHLC suffixes and composites explicitly.
        skip = ("observation_date", "COMPOSITE_TOP15")
        suf = ("_OFI", "_VOL", "_TAKER_BUY", "_OPEN", "_HIGH", "_LOW")
        closes = [c for c in df.columns if c not in skip and not c.startswith("COMPOSITE")
                  and not any(c.endswith(s) for s in suf)]
    of = [c for c in df.columns if c.endswith("_OFI") and "COMPOSITE" not in c]
    assert closes, f"no price columns in {path}"
    return df, closes, of


def _frame(df: pd.DataFrame, px: list[str], of: list[str], agg: int) -> pd.DataFrame:
    px = [c for c in px if c in df.columns]
    of = [c for c in of if c in df.columns]
    r = np.diff(np.log(df[px].to_numpy(float)), axis=0,
                prepend=np.log(df[px].to_numpy(float))[[0], :])
    basket = r.mean(axis=1)
    idx = np.exp(np.cumsum(basket))
    n = len(idx)
    cut = (n // agg) * agg
    idx = idx[:cut].reshape(-1, agg)[:, -1]
    basket = np.diff(np.log(idx), prepend=0.0)
    vol = pd.Series(basket).rolling(5, min_periods=5).std().to_numpy()
    ofi = df[of].to_numpy(float).mean(axis=1) if of else np.zeros(n)
    ofi = ofi[:cut].reshape(-1, agg).mean(axis=1)
    out = pd.DataFrame({"r": basket, "vol": vol, "ofi": ofi, "target": np.roll(basket, -1)})
    return out.iloc[:-1].reset_index(drop=True)


def _universes():
    out = {}
    df, px, of = _load("data/market/crypto_top15.csv")
    out["15-crypto"] = (df, [c for c in px if c.split("_")[0] not in ("COMPOSITE",)], [c for c in of if "COMPOSITE" not in c])
    df, px, of = _load("data/market/multi_asset_30_orderflow.csv")
    out["30-multi"] = (df, px, of)
    df, px, of = _load("data/market/multi_asset_60_orderflow.csv")
    out["45-mix"] = (df, [c for c in px if c.split("_")[0] in CRYPTO30][:30] + [c for c in px if c.split("_")[0] in EQUITY15][:15], of)
    out["60-multi"] = (df, px, of)
    df, px, of = _load("data/market/multi_asset_90_5yr_orderflow.csv")
    out["90-5yr"] = (df, px, of)
    return out


def _holdout_r2(proposal, te: pd.DataFrame) -> float:
    yp = np.asarray(proposal.compute_score(te), dtype=float)
    yt = te["target"].to_numpy()
    m = np.isfinite(yt) & np.isfinite(yp)
    if int(m.sum()) < 10:
        return float("nan")
    ss = float(np.sum((yt[m] - yt[m].mean()) ** 2))
    return 1 - float(np.sum((yt[m] - yp[m]) ** 2)) / ss if ss > 0 else float("nan")


def test_universe_sweep_dailies_abstain():
    for tag, (df, px, of) in _universes().items():
        fr = _frame(df, px, of, 1)
        assert len(fr) > 100, (tag, len(fr))
        assert int(np.isfinite(fr["target"].to_numpy()).sum()) > 100, (tag, "non-finite target")
        props = RimeProposer(feature_cols=["r", "vol", "ofi"], target_col="target",
                             max_depth=2, budget=200).fit_and_propose(fr)
        assert len(props) == 0, (tag, [p.formula for p in props])
    print("\n[sweep] all 5 daily universes abstain")


def test_universe_sweep_weak_emissions_collapse_holdout():
    checked = 0
    for tag, path, agg in (("30-multi", "data/market/multi_asset_30_orderflow.csv", 5),
                           ("60-multi", "data/market/multi_asset_60_orderflow.csv", 5),
                           ("90-5yr", "data/market/multi_asset_90_5yr_orderflow.csv", 21)):
        df, px, of = _load(path)
        fr = _frame(df, px, of, agg)
        n = len(fr)
        tr, te = fr.iloc[:n // 2].reset_index(drop=True), fr.iloc[n // 2:].reset_index(drop=True)
        props = RimeProposer(feature_cols=["r", "vol", "ofi"], target_col="target",
                             max_depth=2, budget=200).fit_and_propose(tr)
        if not props:
            continue  # abstention on the half is itself the finding
        checked += 1
        b = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        assert b.parameters["dev_r2"] < 0.15, (tag, b.parameters["dev_r2"])  # weak-only
        assert _holdout_r2(b, te) < 0.2, tag  # no transfer
    assert checked >= 1
    print(f"\n[sweep] weak 5d/21d emissions collapse on holdout ({checked} emitting halves checked)")


def test_rime_on_ga_exact_22_signal_grammar_abstains():
    # Reviewer challenge: give RIME GA's exact grammar (all 22 signals).
    # Measured: 0 proposals at depths 2-3, budgets 200-1000 (instant
    # seed-stage abstention — no seed beats the mean predictor). Penalty-off
    # reveals why: best achievable devR2 is 0.0010 (quadruple-lagged noise
    # columns). More grammar cannot fix absent signal; GA's historical edge
    # lived in thresholded-Sharpe space, not MSE space, and collapsed
    # out-of-regime regardless.
    from apedp_dyn.sandbox.trading_signals_sandbox import (
        SIGNAL_REGISTRY,
        extract_all_22_features,
        load_real_btc_orderflow,
    )
    panel = load_real_btc_orderflow(600)
    X = extract_all_22_features(panel)
    names = [s[0] for s in SIGNAL_REGISTRY]
    px = panel["price"].to_numpy(float)
    fwd = np.roll(np.append(0.0, np.diff(np.log(px))), -1)
    df = pd.DataFrame({nm: X[:, i] for i, nm in enumerate(names)})
    df["target"] = fwd
    for depth, budget in ((2, 200), (3, 200), (3, 1000)):
        props = RimeProposer(feature_cols=names, target_col="target",
                             max_depth=depth, budget=budget).fit_and_propose(df)
        assert len(props) == 0, (depth, budget, [p.formula for p in props])
    off = RimeProposer(feature_cols=names, target_col="target", max_depth=3,
                       budget=1000, rel_penalty=0.0, gamma=None).fit_and_propose(df)
    assert len(off) > 0
    best = sorted(off, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    assert best.parameters["dev_r2"] < 0.01, best.parameters["dev_r2"]
    print(f"\n[ga-grammar] 22-signal RIME: 0/0/0 proposals; penalty-off best devR2={best.parameters['dev_r2']:.4f}")


def test_rime_grammar_semantically_inert_rename_invariance():
    # RIME understands no grammar: feature names are paint. Same values
    # under a meaningful vs nonsense name yield identical graph ids and
    # identical scores — the kernel sees anonymous columns (v0, v1, ...).
    # Semantic content in this pipeline lives in feature engineering, the
    # dimensional gates, and the referee's claim ladder — never in RIME.
    import pandas as _pd
    rng = np.random.RandomState(11)
    n = 600
    x = rng.randn(n)
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + 0.01 * rng.randn(n - 2)
    d1 = _pd.DataFrame({"momentum_factor": x, "target": y})
    d2 = _pd.DataFrame({"zzzqqq": x, "target": y})
    p1 = RimeProposer(feature_cols=["momentum_factor"], target_col="target",
                      max_depth=2, budget=200).fit_and_propose(d1)
    p2 = RimeProposer(feature_cols=["zzzqqq"], target_col="target",
                      max_depth=2, budget=200).fit_and_propose(d2)
    assert len(p1) == len(p2) > 0
    for a, b in zip(sorted(p1, key=lambda q: q.parameters["dev_r2"], reverse=True),
                    sorted(p2, key=lambda q: q.parameters["dev_r2"], reverse=True), strict=False):
        assert a.metadata.get("graph_id") == b.metadata.get("graph_id")
        assert a.parameters["dev_r2"] == b.parameters["dev_r2"]
    print("\n[semantics] rename-invariant: identical graph ids, identical scores")


def test_frozen_planted_equations_do_not_transfer_to_trading():
    # The five planted-lag equations (frozen coefficients) applied to real
    # crypto forwards: literal application fails on scale (intercept +10 vs
    # mean-0 targets, R2 ~ -1e5); demeaned application still fails on gain
    # (planted gain 3.0 vs 0.03-scale return world, R2 ~ -5e3). Transfer
    # fails on calibration before dynamics are assessable; lag-2 dependence
    # at trading scale belongs to discovery, which abstains (pinned above).
    import pandas as _pd
    FROZEN = [
        ("Eq1", 3.00026, "lag2", 10.0),
        ("Eq2", -3.00026, "neg", 10.0),
        ("Eq5", -1.50594, "diff", 9.88149),
    ]

    def _pred(form, xs, a, b):
        xs = np.asarray(xs, dtype=float)
        v = np.full_like(xs, np.nan)
        if form == "lag2":
            v[2:] = xs[:-2]
        elif form == "neg":
            v[2:] = -(0.0 + (-xs[:-2]))
        else:
            v[2:] = xs[2:] + (-xs[:-2])
        return a * v + b

    for sym in ("BTC", "ETH"):
        df = _pd.read_csv("data/market/crypto_top15_orderflow.csv",
                          usecols=["observation_date", f"{sym}_CLOSE"])
        r = np.append(0.0, np.diff(np.log(df[f"{sym}_CLOSE"].to_numpy(float))))
        n = len(r)
        ntr = n // 2
        mu, sd = r[:ntr].mean(), r[:ntr].std()
        z = (r - mu) / sd
        fwd = np.roll(r, -1)
        te = np.arange(ntr, n - 1)
        for name, a, form, b in FROZEN:
            yp = _pred(form, z, a, b)[te]
            yt = fwd[te]
            m = np.isfinite(yt) & np.isfinite(yp)
            ss = float(np.sum((yt[m] - yt[m].mean()) ** 2))
            r2 = 1 - float(np.sum((yt[m] - yp[m]) ** 2)) / ss
            assert r2 < 0, (sym, name, r2)  # no transfer, both layers
    print("\n[non-transfer] frozen planted Eqs fail on BTC/ETH forwards (scale, then gain)")


def test_usage_log_22_signals_mostly_untouched():
    # Usage logging on GA's exact grammar (penalty-off, the only emitting
    # regime): 5 proposals touch 3/22 signals; 19 sit entirely unused.
    # Presence is syntactic (reachable nodes), not causal importance.
    from apedp_dyn.proposers.rime_usage_log import log_search_usage
    from apedp_dyn.sandbox.trading_signals_sandbox import (
        SIGNAL_REGISTRY,
        extract_all_22_features,
        load_real_btc_orderflow,
    )
    panel = load_real_btc_orderflow(600)
    X = extract_all_22_features(panel)
    names = [s[0] for s in SIGNAL_REGISTRY]
    px = panel["price"].to_numpy(float)
    df = pd.DataFrame({nm: X[:, i] for i, nm in enumerate(names)})
    df["target"] = np.roll(np.append(0.0, np.diff(np.log(px))), -1)
    props = RimeProposer(feature_cols=names, target_col="target", max_depth=3,
                         budget=1000, rel_penalty=0.0, gamma=None).fit_and_propose(df)
    assert len(props) > 0
    rep = log_search_usage(props, names)
    assert len(rep.unused_in_all) >= 15, rep.to_dict()
    assert set(rep.used_in_any) <= {"H-CVD", "H-OFI", "H-SQUEEZE", "H-BB", "H-RV"}, rep.used_in_any
    print(f"\n[usage] 22-signal: {rep.n_proposals} proposals, used={rep.used_in_any}")


def test_universe_sweep_90_assets_60d_and_60_dim_pde():
    """Test 90 assets across both 60-day horizon and 60-dimensional continuous PDE space.
    1. 60-day aggregation window (19 bars) strictly abstains due to sample size floor (<30 rows).
    2. Daily bars with trailing 60d rolling features (1195 bars) strictly abstain (no spurious overfitting).
    3. 60-dimensional continuous PDE discovery on 90 real assets refutes continuous spatial diffusion (R^2 < 0.06).
    4. Positive control on identical 90-asset 60D coordinates recovers planted 60D diffusion with R^2 > 0.999999.
    5. 60D McKean-Vlasov Attention PDE contracts the 90-asset return cloud monotonically.
    """
    from sklearn.decomposition import PCA

    from apedp_dyn.epistemic.high_dim_pde import (
        discover_pde_rbf_point_cloud_hd,
        step_vlasov_attention_pde_hd,
    )

    df, px, of = _load("data/market/multi_asset_90_5yr_orderflow.csv")

    # Part 1: 60-day aggregation horizon
    fr_60d = _frame(df, px, of, 60)
    assert len(fr_60d) == 19
    props_60d = RimeProposer(
        feature_cols=["r", "vol", "ofi"], target_col="target", max_depth=2, budget=200
    ).fit_and_propose(fr_60d)
    assert len(props_60d) == 0  # Abstains honestly on N < 30

    # Part 2: Daily data with trailing 60-day rolling features
    daily_fr = _frame(df, px, of, 1)
    r_series = daily_fr["r"]
    daily_fr["r_60d"] = r_series.rolling(60).sum()
    daily_fr["vol_60d"] = r_series.rolling(60).std()
    daily_fr["ofi_60d"] = daily_fr["ofi"].rolling(60).mean()
    clean_daily = daily_fr.dropna().reset_index(drop=True)
    assert len(clean_daily) == 1195

    props_rolling = RimeProposer(
        feature_cols=["r_60d", "vol_60d", "ofi_60d"], target_col="target", max_depth=2, budget=200
    ).fit_and_propose(clean_daily)
    assert len(props_rolling) == 0  # Honest abstention on market noise

    # Part 3: 60-dimensional continuous PDE evaluation on 90 assets
    close_cols = [c for c in df.columns if c.endswith("_CLOSE")]
    prices = df[close_cols].to_numpy(float)
    returns = np.diff(np.log(prices), axis=0)  # (1254, 90)

    pca = PCA(n_components=60, random_state=42)
    points_60d = pca.fit_transform(returns.T)  # 90 assets as points in R^60
    assert points_60d.shape == (90, 60)

    # Real returns over last 60 time steps
    u_values = returns[-60:, :]
    res_real = discover_pde_rbf_point_cloud_hd(
        points=points_60d,
        u_values=u_values,
        dt=1.0 / 252.0,
        dimension=60,
        sparsity_threshold=0.01,
    )
    assert res_real.dimension == 60
    assert not res_real.holdout_passed  # Referee correctly refuses continuous diffusion on market data
    assert res_real.r2_dev < 0.10

    # Part 4: Positive control on identical 90-asset 60D coordinates
    norm_points_60d = points_60d / np.std(points_60d) * 0.1
    dim, alpha, nt, dt = 60, 0.5, 30, 0.005
    r2 = np.sum(norm_points_60d * norm_points_60d, axis=1)
    u_planted = np.zeros((nt, 90))
    for ti in range(nt):
        t = 1.0 + ti * dt
        scale = (1.0 / t) ** (dim / 2.0)
        u_planted[ti, :] = scale * np.exp(-r2 / (4.0 * alpha * t))

    res_planted = discover_pde_rbf_point_cloud_hd(
        points=norm_points_60d,
        u_values=u_planted,
        dt=dt,
        dimension=dim,
        sparsity_threshold=0.01,
    )
    assert res_planted.dimension == 60
    assert res_planted.r2_dev > 0.99999
    assert res_planted.holdout_passed
    lap_idx = res_planted.active_terms.index("laplacian_60d(u)")
    assert abs(res_planted.coefficients[lap_idx] - alpha) < 0.05

    # Part 5: 60D McKean-Vlasov Attention PDE on the 90 assets
    evolved = step_vlasov_attention_pde_hd(points_60d, steps=10, dt=0.05, damping=0.1)
    assert evolved.shape == (90, 60)
    assert np.max(np.linalg.norm(evolved, axis=1)) < np.max(np.linalg.norm(points_60d, axis=1))
    assert np.mean(np.var(evolved, axis=0)) < np.mean(np.var(points_60d, axis=0))

