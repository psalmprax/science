"""The survivorship-free panel builder, against a fake archive.

The real archive is not reachable from every network (the build runs on
GitHub's runners), so these tests serve a small archive from memory: a listed
pair, a pair delisted half-way, a pegged pair and a leveraged token, with the
2025 change from millisecond to microsecond timestamps and a tampered file.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_point_in_time_crypto_panel as builder

from apedp_dyn.market.universe import point_in_time_membership


def _kline_zip(name: str, bars: list[tuple[dt.date, float, float, float]], micro: bool) -> bytes:
    text = io.StringIO()
    writer = csv.writer(text)
    for day, close, volume, taker in bars:
        stamp = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())
        stamp = stamp * 1_000_000 if micro else stamp * 1000
        writer.writerow([stamp, close, close, close, close, volume, stamp, close * volume, 10, taker, close * taker, 0])
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(name.replace(".zip", ".csv"), text.getvalue())
    return out.getvalue()


class FakeArchive:
    """Enough of data.binance.vision and its S3 listing for the builder."""

    def __init__(self, series: dict[str, dict[dt.date, tuple[float, float, float]]]):
        self.objects: dict[str, bytes] = {}
        for symbol, bars in series.items():
            by_month: dict[str, list] = {}
            for day, (close, volume, taker) in sorted(bars.items()):
                by_month.setdefault(f"{day:%Y-%m}", []).append((day, close, volume, taker))
            for month, rows in by_month.items():
                name = f"{symbol}-1d-{month}.zip"
                key = f"{builder.KLINE_PREFIX}{symbol}/1d/{name}"
                blob = _kline_zip(name, rows, micro=month >= "2025-01")
                self.objects[key] = blob
                self.objects[f"{key}.CHECKSUM"] = f"{hashlib.sha256(blob).hexdigest()}  {name}\n".encode()
        self.symbols = sorted(series)

    def __call__(self, url: str) -> bytes:
        url.encode("ascii")  # what urllib does to the request line; a raw non-ASCII key fails here
        parsed = urlparse(url)
        if url.startswith(builder.LISTING):
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            prefix = query["prefix"]
            if query.get("delimiter") == "/":
                body = "".join(f"<CommonPrefixes><Prefix>{prefix}{s}/</Prefix></CommonPrefixes>" for s in self.symbols)
            else:
                body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in sorted(self.objects)
                               if k.startswith(prefix) and not k.endswith(".CHECKSUM"))
            return f"<ListBucketResult><IsTruncated>false</IsTruncated>{body}</ListBucketResult>".encode()
        return self.objects[unquote(parsed.path.lstrip("/"))]


def _daily(start: str, end: str, close: float = 10.0, volume: float = 100.0, taker: float = 60.0):
    day, last, out = dt.date.fromisoformat(start), dt.date.fromisoformat(end), {}
    while day <= last:
        out[day] = (close, volume, taker)
        close *= 1.01
        day += dt.timedelta(days=1)
    return out


@pytest.fixture
def archive():
    return FakeArchive({
        "AAAUSDT": _daily("2024-11-01", "2025-03-31"),
        "GONEUSDT": _daily("2024-11-01", "2025-01-15", close=5.0, volume=500.0, taker=200.0),
        "USDCUSDT": _daily("2024-11-01", "2025-03-31", close=1.0),
        "BTCUSDT": _daily("2024-11-01", "2025-03-31", close=100.0, volume=1.0, taker=0.5),
        "BTCUPUSDT": _daily("2024-11-01", "2025-03-31"),
        "AAABTC": _daily("2024-11-01", "2025-03-31"),
    })


def _build(tmp_path, archive):
    return builder.build(tmp_path, dt.date(2024, 11, 1), dt.date(2025, 3, 31), dt.date(2025, 3, 1),
                         fetch=archive, workers=2)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())))


def test_a_delisted_pair_is_kept_up_to_its_last_bar(tmp_path, archive):
    manifest = _build(tmp_path, archive)
    assert manifest["assets"] == ["AAA", "BTC", "GONE"]
    assert manifest["last_bar"]["GONE"] == "2025-01-15"
    assert manifest["stopped_trading_before_end"] == ["GONE"]
    panel = _read(tmp_path / "panel_development.csv.gz")
    gone = panel.set_index("observation_date")["GONE_CLOSE"]
    assert gone.loc["2025-01-15"] > 0 and pd.isna(gone.loc["2025-01-16"])


def test_pegged_and_leveraged_pairs_get_no_column_and_are_listed(tmp_path, archive):
    manifest = _build(tmp_path, archive)
    assert manifest["excluded_pairs"] == ["BTCUPUSDT", "USDCUSDT"]
    panel = _read(tmp_path / "panel_development.csv.gz")
    assert not any(c.startswith(("USDC_", "BTCUP_")) for c in panel.columns)


def test_microsecond_timestamps_from_2025_land_on_the_right_day(tmp_path, archive):
    _build(tmp_path, archive)
    holdout = _read(tmp_path / "panel_holdout.csv.gz")
    assert holdout["observation_date"].iloc[0] == "2025-03-01"
    assert holdout["observation_date"].iloc[-1] == "2025-03-31"
    assert holdout["AAA_CLOSE"].notna().all()


def test_the_holdout_is_a_separate_file_starting_at_the_split(tmp_path, archive):
    manifest = _build(tmp_path, archive)
    development = _read(tmp_path / "panel_development.csv.gz")
    assert development["observation_date"].max() == "2025-02-28"
    for name in ("panel_development.csv.gz", "panel_holdout.csv.gz"):
        assert manifest[name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert json.loads((tmp_path / "manifest.json").read_text())["holdout_start"] == "2025-03-01"


def test_order_flow_imbalance_is_buy_minus_sell_over_total(tmp_path, archive):
    _build(tmp_path, archive)
    panel = _read(tmp_path / "panel_development.csv.gz").set_index("observation_date")
    assert panel.loc["2024-11-01", "AAA_OFI"] == pytest.approx((60 - 40) / 100)
    assert panel.loc["2024-11-01", "GONE_OFI"] == pytest.approx((200 - 300) / 500)


def test_a_file_that_does_not_match_its_published_digest_is_refused(tmp_path, archive):
    key = next(k for k in archive.objects if k.endswith(".zip"))
    archive.objects[key] = archive.objects[key] + b"tampered"
    with pytest.raises(ValueError, match="does not match the published"):
        _build(tmp_path, archive)


def test_the_universe_rule_admits_the_delisted_pair_while_it_traded(tmp_path, archive):
    """The point of the panel: point-in-time membership can now hold an asset that later died."""
    _build(tmp_path, archive)
    panel = _read(tmp_path / "panel_development.csv.gz")
    members = point_in_time_membership(panel, top_n=1, lookback_bars=10, rebalance_every=10)
    dates = panel["observation_date"].to_numpy()
    gone = members["GONE"].to_numpy()
    assert gone[dates < "2025-01-10"].any(), "the larger pair was never ranked in"
    assert not gone[dates > "2025-01-15"].any(), "a pair was held after its last bar"


def test_manifest_marks_the_panel_as_not_hindsight_selected(tmp_path, archive):
    selection = _build(tmp_path, archive)["universe_selection"]
    assert selection["hindsight_selected"] is False
    assert selection["includes_assets_that_stopped_trading"] is True


def test_the_fetcher_refuses_any_host_but_the_archive():
    with pytest.raises(ValueError, match="outside the archive"):
        builder.http_get("https://example.com/data/spot/x.zip", attempts=1)
    with pytest.raises(ValueError, match="outside the archive"):
        builder.http_get("file:///etc/passwd", attempts=1)


@pytest.mark.parametrize("symbol, expected", [
    ("BTCUPUSDT", None), ("ETHDOWNUSDT", None), ("BNBBULLUSDT", None),
    ("JUPUSDT", "JUP"), ("SYRUPUSDT", "SYRUP"), ("SUPERUSDT", "SUPER"),
    ("USDCUSDT", None), ("BTCUSDT", "BTC"), ("BTCBUSD", None),
])
def test_only_derivatives_of_a_listed_base_count_as_leveraged(symbol, expected):
    listed = frozenset({"BTC", "ETH", "BNB", "JUP", "SYRUP", "SUPER", "USDC"})
    assert builder.eligible(symbol, listed) == expected


def test_a_pair_whose_symbol_is_not_ascii_is_fetched(tmp_path):
    """The archive lists 币安人生USDT. Its first run on GitHub (2026-11-17) failed
    after ten minutes: the key went into the request line unencoded and urllib
    refused it. The fake refuses the same way, so this fails without the fix."""
    fake = FakeArchive({
        "AAAUSDT": _daily("2024-11-01", "2025-02-28"),
        "币安人生USDT": _daily("2024-12-01", "2025-02-28", close=0.2, volume=9000.0, taker=5000.0),
    })
    manifest = builder.build(tmp_path, dt.date(2024, 11, 1), dt.date(2025, 2, 28), dt.date(2025, 2, 1), fetch=fake)
    with gzip.open(tmp_path / "panel_development.csv.gz", "rt", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    assert "币安人生_CLOSE" in header and "AAA_CLOSE" in header
    assert any("币安人生USDT" in f["file"] for f in manifest["source_files"])


def test_archive_urls_are_ascii_and_round_trip():
    key = f"{builder.KLINE_PREFIX}币安人生USDT/1d/币安人生USDT-1d-2026-01.zip"
    url = builder.archive_url(key)
    url.encode("ascii")
    assert unquote(url[len(builder.ARCHIVE) + 1:]) == key
    plain = f"{builder.KLINE_PREFIX}BTCUSDT/1d/BTCUSDT-1d-2026-01.zip"
    assert builder.archive_url(plain) == f"{builder.ARCHIVE}/{plain}"
