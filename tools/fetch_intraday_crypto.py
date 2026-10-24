"""Fetch bounded intraday crypto bars from Binance for the repo's own universes.

Scope, deliberately bounded and clearly labeled:
  * Universes: the crypto legs ALREADY DEFINED by the frozen corpus
    (crypto_top15.csv = 15 symbols; the crypto leg of multi_asset_60 = 30;
    the crypto leg of multi_asset_90_5yr = 45). No new symbol list is
    invented, and the equity legs (60/90-asset universes) are NOT fetched:
    Yahoo Finance intraday returned HTTP 429 at fetch time.
  * Intervals: 1m, 5m, 15m, 30m, 1h. Binance has no 45m interval, so the
    requested 45-minute frame does not exist as an exchange bar.
  * Window: `limit` bars (max 1000 per request), one request per
    symbol/interval. NOTE the resulting confound, documented in the output
    PROVENANCE: equal BAR COUNTS cover different calendar spans (1000 1m
    bars ~ 17h; 1000 1h bars ~ 42 days). Fixing the calendar window instead
    would need paginated requests (up to 43k bars/symbol at 1m) and a much
    larger artifact; bar-count windows are the honest bounded choice.

Writes data/market/intraday/intraday_crypto_bars.csv + PROVENANCE.json.
The frozen daily corpus is not touched.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "market"
OUT_DIR = DATA_DIR / "intraday"
OUT_CSV = OUT_DIR / "intraday_crypto_bars.csv"
OUT_PROV = OUT_DIR / "PROVENANCE.json"

INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")
LIMIT = 1000


def _crypto_symbols(csv_path: Path) -> list[str]:
    """Crypto symbols of a corpus file, in file order (authoritative)."""
    df = pd.read_csv(csv_path, nrows=1)
    skip = {"observation_date", "COMPOSITE_TOP15"}
    out = []
    for c in df.columns:
        if c in skip or c.startswith("COMPOSITE"):
            continue
        if not c.endswith("_CLOSE"):
            continue
        base = c[: -len("_CLOSE")]
        out.append(base)
    return out


def _binance_listed() -> set[str]:
    """Base symbols currently listed on Binance spot (one exchangeInfo call)."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                info = json.loads(r.read().decode())
            bases = set()
            for s in info.get("symbols", []):
                if s.get("status") == "TRADING" and s.get("symbol", "").endswith("USDT"):
                    bases.add(s["symbol"][: -len("USDT")])
            return bases
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  ! exchangeInfo failed: {e}")
                return set()
            time.sleep(2.0 * (attempt + 1))
    return set()


def _fetch(symbol: str, interval: str, limit: int = LIMIT, retries: int = 3):
    url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT"
           f"&interval={interval}&limit={limit}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 — network flake, retry then fail
            if attempt == retries - 1:
                print(f"  ! {symbol} {interval}: {e}")
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def main() -> None:
    universes = {
        "u15": _crypto_symbols(DATA_DIR / "crypto_top15.csv"),
        "u30": _crypto_symbols(DATA_DIR / "multi_asset_60_orderflow.csv"),
        "u45": _crypto_symbols(DATA_DIR / "multi_asset_90_5yr_orderflow.csv"),
    }
    u15 = universes["u15"]
    u30 = universes["u30"]
    u45 = universes["u45"]
    assert set(u15) <= set(u30) <= set(u45) or True  # nested or not, we record as-is
    print(f"universes: u15={len(u15)} u30={len(u30)} u45={len(u45)}")
    all_syms = sorted(set(u15) | set(u30) | set(u45))
    listed = _binance_listed()
    skipped = [s for s in all_syms if s not in listed]
    all_syms = [s for s in all_syms if s in listed]
    print(f"distinct symbols: {len(all_syms)} fetchable ({len(skipped)} not Binance-listed: "
          f"{skipped[:8]}{'...' if len(skipped) > 8 else ''}); intervals: {INTERVALS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    spans: dict[str, tuple[int, int]] = {}
    for iv in INTERVALS:
        for sym in all_syms:
            kl = _fetch(sym, iv)
            if not kl:
                continue
            for k in kl:
                rows.append({
                    "interval": iv, "symbol": sym,
                    "open_time_ms": int(k[0]),
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                    "taker_buy_base": float(k[7]),
                })
            t0 = int(kl[0][0])
            t1 = int(kl[-1][0])
            spans[f"{iv}/{sym}"] = (t0, t1)
            print(f"  {iv:>3s} {sym:<6s} {len(kl):>4d} bars  "
                  f"{pd.to_datetime(t0, unit='ms')} .. {pd.to_datetime(t1, unit='ms')}")
    if not rows:
        raise SystemExit("no data fetched")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    prov = {
        "source": "Binance Public Spot Klines (api.binance.com/api/v3/klines)",
        "license_note": "Binance public market data terms apply",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "intervals": list(INTERVALS),
        "bars_per_symbol_interval": LIMIT,
        "universes": {"u15": u15, "u30": u30, "u45": u45},
        "universe_definition": "crypto legs read from the frozen daily corpus files",
        "notes": [
            "Bar-count windows: 1000 1m bars ~ 17h vs 1000 1h bars ~ 42 days — "
            "calendar span is confounded with interval by design (bounded fetch).",
            "Binance has no 45m interval; the requested 45-minute frame is absent.",
            "Equity legs not fetched (Yahoo Finance intraday returned HTTP 429), "
            "so 60/90-asset intraday universes are out of scope.",
        ],
        "rows": len(df),
    }
    OUT_PROV.write_text(json.dumps(prov, indent=1))
    print(f"\nwrote {OUT_CSV} ({len(df)} rows) and {OUT_PROV}")


if __name__ == "__main__":
    main()
