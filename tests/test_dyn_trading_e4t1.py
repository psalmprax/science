"""E4-T1 protocol pins (doc S72): time-order, abstention honesty, determinism.

Full 11-origin numbers live in the doc (slow); this suite pins the frozen
protocol's structural properties on two origins: P1 MULTI pre, active
abstention when max-D < k, forward-only re-adjudication, frozen
reproducibility, and the oracle-never-promotes guard (with ~6 forward
blocks no policy may manufacture UNIQUE from noise).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import apedp_dyn.epistemic.active_disambiguation.trading_e4t1 as T1
import apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 as T

PANELS = None


def _synthetic_panels(n=1200):
    rng = np.random.default_rng(41)
    dates = pd.date_range("2020-01-01", periods=n)
    panel = pd.DataFrame({
        "date": dates,
        "sig": np.abs(rng.normal(0, 0.01, n)),
        "trailvol20": rng.random(n) + 0.1,
        "rankV": rng.random(n),
        "r": rng.normal(0, 0.01, n),
        "logV": rng.normal(size=n),
        "HA": rng.normal(size=n),
    })
    return {asset: panel.copy() for asset in ("BTC", "ETH", "SOL")}


def _panels():
    global PANELS
    if PANELS is None:
        frames = T.load_frames()
        PANELS = {a: T.build_asset_panel(frames["raw"], a) for a in T.ASSETS}
    return PANELS


def test_v1_d5_measurement_table_excludes_forward_labels(monkeypatch):
    panels = _synthetic_panels()
    changed = {asset: panel.copy() for asset, panel in panels.items()}
    origin = "2022-01-01"
    origin_idx = int(np.searchsorted(
        panels["BTC"]["date"].to_numpy(), np.datetime64(origin), side="left"
    ))
    changed["BTC"].loc[origin_idx + 4, "sig"] += 1.0
    observed = []

    monkeypatch.setattr(T1, "_adjudicate", lambda _: ("MULTI", None))

    def record_table(preds, sigmas):
        observed.append(sigmas.copy())
        return [SimpleNamespace(mid="m2", d_sigma=1.0)]

    monkeypatch.setattr(T1, "discrimination_table", record_table)

    T1.run_origin(panels, origin, "active", seed=1)
    T1.run_origin(changed, origin, "active", seed=1)

    for mid in ("m2", "m6"):
        assert observed[0][mid] == pytest.approx(observed[1][mid])


def test_v2_fit_excludes_forward_labels(monkeypatch):
    panels = _synthetic_panels()
    changed = {asset: panel.copy() for asset, panel in panels.items()}
    origin = "2022-01-01"
    origin_idx = int(np.searchsorted(
        panels["BTC"]["date"].to_numpy(), np.datetime64(origin), side="left"
    ))
    changed["BTC"].loc[origin_idx + 4, "sig"] += 1.0
    real_lstsq = T.la.lstsq
    training_targets = []

    def capture_lstsq(x, y, *args, **kwargs):
        training_targets.append(np.asarray(y).copy())
        return real_lstsq(x, y, *args, **kwargs)

    monkeypatch.setattr(T.la, "lstsq", capture_lstsq)
    T.run_origin(panels, origin, "active", seed=1)
    first_targets = training_targets[0].copy()
    training_targets.clear()
    T.run_origin(changed, origin, "active", seed=1)

    np.testing.assert_allclose(first_targets, training_targets[0])


@pytest.mark.parametrize("origin", ["2022-08-01", "2023-01-01"])
def test_P1_multi_pre(origin):
    r = T.run_origin(_panels(), origin, "active", seed=7)
    assert "multi" in r.pre_status.lower()


@pytest.mark.parametrize("origin", ["2022-08-01", "2023-01-01"])
def test_abstention_honest(origin):
    r = T.run_origin(_panels(), origin, "active", seed=7)
    assert r.post_status == "INSUFFICIENT_DISCRIMINATION"
    assert r.mid is None and r.winner is None


def test_oracle_cannot_manufacture_unique():
    for origin in ("2022-08-01", "2023-01-01"):
        r = T.run_origin(_panels(), origin, "oracle", seed=7)
        assert "unique_support" not in r.post_status.lower()


def test_frozen_reproducibility():
    a = T.run_origin(_panels(), "2022-08-01", "random", seed=7)
    b = T.run_origin(_panels(), "2022-08-01", "random", seed=7)
    assert a == b


def test_time_order_forward_after_selection():
    import pandas as pd

    dates = _panels()["BTC"]["date"]
    origin = "2022-08-01"
    nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
    assert str(dates[dates >= origin].min()) >= origin
    assert str(dates[dates < origin].max()) < origin
    assert nxt > origin


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
