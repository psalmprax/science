"""Intraday crypto sweep: RIME over repo-defined crypto universes × exchange
intervals, on genuinely fetched minute bars.

Data: `data/market/intraday/intraday_crypto_bars.csv` (Binance public
klines, 1000 bars/symbol/interval, 5m/15m/30m/1h/1m, written by
`tools/fetch_intraday_crypto.py` with its own PROVENANCE.json). The frozen
daily corpus is untouched. Skips when the artifact is absent.

Scope honesty (recorded, not papered over):
  * 45m is not a Binance interval; 60/90-asset universes need the equity
    legs, which were unavailable (Yahoo intraday HTTP 429 at fetch time) —
    so the intraday universes are the repo's crypto legs: 14 / 26 / 39.
  * Equal BAR COUNTS cover different calendar spans (1000 1m ≈ 17h,
    1000 1h ≈ 42d) — the interval axis is confounded with span by design.

Measured, pinned here:
  * real basket forward returns: 0 proposals in all 15 universe×interval
    cells (both seed orders) — no grammar-expressible edge at any tested
    resolution;
  * null worlds: 0 proposals;
  * planted positive control (lag-1 dependence on BTC): recovered at
    dev R²>0.98 at every interval — the pipeline works on minute bars;
  * per-position coverage follows the 60-form budget arithmetic exactly:
    with 39 inputs, `coverage_first` keeps lag-1 for inputs 0..20
    (39 levels + 21 = 60) and positions 21+ remain generation gaps —
    a strict improvement over legacy (whole inputs lost), with the residual
    boundary quantified rather than asserted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    CRYPTO_15_ASSETS,
    CRYPTO_30_ASSETS,
    CRYPTO_45_ASSETS,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "market" / "intraday" / "intraday_crypto_bars.csv"
INTERVALS = ("1m", "5m", "15m", "30m", "1h")


def _require_data() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip(f"intraday artifact absent: {DATA}")
    return pd.read_csv(DATA)


def _universe_map(df: pd.DataFrame) -> dict[str, list[str]]:
    have = set(df["symbol"].unique())
    return {
        "u15": [s for s in CRYPTO_15_ASSETS if s in have],
        "u30": [s for s in CRYPTO_30_ASSETS if s in have],
        "u45": [s for s in CRYPTO_45_ASSETS if s in have],
    }


def _frames(df: pd.DataFrame, interval: str, syms: list[str]):
    d = df[df.interval == interval].pivot(index="open_time_ms", columns="symbol",
                                          values="close").sort_index()[syms].dropna()
    r = np.log(d).diff().iloc[1:]
    X = r.iloc[:-1].reset_index(drop=True)
    X.columns = [f"{c}_r" for c in X.columns]
    return d, X, r.mean(axis=1).shift(-1).iloc[:-1].to_numpy()


def test_intraday_real_returns_abstain_all_cells():
    df = _require_data()
    for iv in INTERVALS:
        for uname, syms in _universe_map(df).items():
            _d, X, tgt = _frames(df, iv, syms)
            fr = X.copy()
            fr["target"] = tgt
            tr = fr.iloc[: len(fr) // 2].reset_index(drop=True)
            feats = [c for c in tr.columns if c != "target"]
            for order in ("legacy", "coverage_first"):
                props = RimeProposer(feature_cols=feats, target_col="target", max_depth=2,
                                     budget=200, seed_order=order).fit_and_propose(tr)
                assert len(props) == 0, (iv, uname, order, len(props))
    print("\n[intraday] 15/15 cells abstain on real minute-bar returns (both seed orders)")


def test_intraday_null_worlds_abstain():
    df = _require_data()
    uni = _universe_map(df)
    for iv in INTERVALS:
        _d, X, _t = _frames(df, iv, uni["u45"])
        total = 0
        for s in range(3):
            rng = np.random.RandomState(200 + s)
            fr = X.copy()
            fr["target"] = rng.randn(len(X))
            props = RimeProposer(feature_cols=list(X.columns), target_col="target",
                                 max_depth=2, budget=200,
                                 seed_order="coverage_first").fit_and_propose(fr)
            total += len(props)
        assert total == 0, (iv, total)
    print("\n[intraday] null worlds: 0 proposals across all intervals")


def test_intraday_planted_control_recovers_at_every_interval():
    df = _require_data()
    uni = _universe_map(df)
    for iv in INTERVALS:
        _d, X, _t = _frames(df, iv, uni["u45"])
        col = X["BTC_r"].to_numpy(float)
        rng = np.random.RandomState(7)
        y = np.zeros(len(X))
        y[1:] = 2.0 * col[:-1] + 0.05 * float(np.nanstd(col[:-1])) * rng.randn(len(X) - 1)
        fr = X.copy()
        fr["target"] = y
        tr = fr.iloc[: len(fr) // 2].reset_index(drop=True)
        props = RimeProposer(feature_cols=list(X.columns), target_col="target",
                             max_depth=2, budget=200).fit_and_propose(tr)
        assert len(props) > 0, iv
        best = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        assert best.metadata.get("inputs_used_names") == ["BTC_r"], (iv, best.metadata)
        assert best.parameters["dev_r2"] > 0.98, (iv, best.parameters["dev_r2"])
    print("\n[intraday] planted control recovered (dev R2>0.98) at all 5 intervals")


def test_intraday_position_coverage_matches_budget_arithmetic():
    # 39 inputs: catalogue budget 60 holds 39 levels + the first 21 lag-1
    # forms, so coverage_first keeps lag-1 for inputs 0..20 and positions
    # 21+ remain generation gaps. Legacy loses whole inputs instead.
    df = _require_data()
    uni = _universe_map(df)
    _d, X, _t = _frames(df, "5m", uni["u45"])
    feats = list(X.columns)
    n_in = len(feats)
    keep = 60 - n_in  # lag-1 slots left after the level family
    assert 0 < keep < n_in, (n_in, keep)

    def planted(pos: int):
        col = X[feats[pos]].to_numpy(float)
        rng = np.random.RandomState(11)
        y = np.zeros(len(X))
        y[1:] = 2.0 * col[:-1] + 0.05 * float(np.nanstd(col[:-1])) * rng.randn(len(X) - 1)
        fr = X.copy()
        fr["target"] = y
        return fr.iloc[: len(fr) // 2].reset_index(drop=True)

    def best_used(pos: int, order: str):
        tr = planted(pos)
        props = RimeProposer(feature_cols=feats, target_col="target", max_depth=2,
                             budget=200, seed_order=order).fit_and_propose(tr)
        if not props:
            return None, float("nan")
        b = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
        used = b.metadata.get("inputs_used_names") or []
        return (used[0] if used else None), b.parameters["dev_r2"]

    inside, outside = min(keep - 1, n_in - 1), keep + 2
    if inside < n_in:
        u, r2 = best_used(inside, "coverage_first")
        assert u == feats[inside] and r2 > 0.98, (inside, u, r2)
    if outside < n_in:
        u, _ = best_used(outside, "coverage_first")
        assert u != feats[outside], (outside, u)
    print(f"\n[intraday] coverage_first keeps lag-1 for inputs 0..{keep-1}; "
          f"position {inside} recovered, {outside} still a gap")


def test_intraday_bar_count_span_confound_documented():
    prov = DATA.with_name("PROVENANCE.json")
    if not prov.exists():
        pytest.skip("no intraday PROVENANCE.json")
    import json
    notes = json.loads(prov.read_text()).get("notes", [])
    assert any("confounded" in n for n in notes), notes
    df = _require_data()
    for iv in INTERVALS:
        sub = df[df.interval == iv]
        assert len(sub) > 0
        span_ms = int(sub["open_time_ms"].max() - sub["open_time_ms"].min())
        assert span_ms > 0
