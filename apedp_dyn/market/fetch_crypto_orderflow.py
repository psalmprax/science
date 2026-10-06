"""Fetch historical daily order flow data (taker buy vs sell volume) from Binance.

Downloads daily klines for all 15 crypto assets from 2021-01-01 to 2023-12-31:
- Extracts total volume and taker buy base asset volume.
- Computes Order Flow Imbalance (OFI) = (V_buy - V_sell) / V_total.
- Computes equal-weighted COMPOSITE_TOP15 OFI.
- Synchronizes with existing data/market/crypto_top15.csv prices.
"""
from __future__ import annotations

import csv
import datetime
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "market"
CRYPTO_FILE = DATA_DIR / "crypto_top15.csv"
ORDERFLOW_FILE = DATA_DIR / "crypto_top15_orderflow.csv"
PROVENANCE_FILE = DATA_DIR / "PROVENANCE.json"

CRYPTO_ASSETS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "AVAX",
    "DOGE",
    "DOT",
    "LINK",
    "MATIC",
    "NEAR",
    "ATOM",
    "UNI",
    "LTC",
)


def fetch_symbol_klines(symbol: str, start_dt: str = "2021-01-01", end_dt: str = "2023-12-31") -> dict[str, dict[str, float]]:
    """Fetch daily klines with taker volumes from Binance public API."""
    pair = f"{symbol}USDT"
    start_ts = int(datetime.datetime.strptime(start_dt, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.datetime.strptime(end_dt, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc).timestamp() * 1000)

    url1 = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1d&startTime={start_ts}&limit=1000"
    req1 = urllib.request.Request(url1, headers={"User-Agent": "Mozilla/5.0"})
    # S310: the scheme is a literal https:// Binance public endpoint, not
    # user input, so there is no file:/custom-scheme surface here.
    with urllib.request.urlopen(req1, timeout=15) as resp:  # noqa: S310
        batch1 = json.loads(resp.read().decode())

    next_ts = batch1[-1][6] + 1
    url2 = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1d&startTime={next_ts}&endTime={end_ts}&limit=1000"
    req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req2, timeout=15) as resp:  # noqa: S310
        batch2 = json.loads(resp.read().decode())

    all_k = batch1 + batch2
    out: dict[str, dict[str, float]] = {}
    for k in all_k:
        open_time = k[0]
        dt = datetime.datetime.fromtimestamp(open_time / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        close_px = float(k[4])
        vol = float(k[5])
        taker_buy = float(k[9])
        taker_sell = max(0.0, vol - taker_buy)
        ofi = float(np.clip((taker_buy - taker_sell) / vol, -1.0, 1.0)) if vol > 0 else 0.0
        out[dt] = {
            "close": close_px,
            "volume": vol,
            "taker_buy": taker_buy,
            "taker_sell": taker_sell,
            "ofi": ofi,
        }
    return out


def main():
    print(f"Fetching Binance order flow for {len(CRYPTO_ASSETS)} assets...")
    asset_data: dict[str, dict[str, dict[str, float]]] = {}
    for sym in CRYPTO_ASSETS:
        print(f"  Fetching {sym}USDT...")
        asset_data[sym] = fetch_symbol_klines(sym)
        time.sleep(0.15)

    # Read original crypto_top15.csv rows to preserve exact existing price columns
    orig_rows: list[dict[str, str]] = []
    with open(CRYPTO_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_orig = list(reader.fieldnames or [])
        orig_rows = list(reader)

    dates = [r["observation_date"] for r in orig_rows]
    print(f"Synchronizing with existing {len(dates)} dates in {CRYPTO_FILE}...")

    # Compute per-day composite OFI and per-asset OFI
    enriched_rows: list[dict[str, Any]] = []
    orderflow_detailed_rows: list[dict[str, Any]] = []

    ofi_keys = [f"{sym}_OFI" for sym in CRYPTO_ASSETS] + ["COMPOSITE_TOP15_OFI"]
    new_fieldnames = fieldnames_orig + ofi_keys

    for r in orig_rows:
        dt = r["observation_date"]
        new_r = dict(r)
        detailed_r: dict[str, Any] = {"observation_date": dt}

        day_ofis: list[float] = []
        for sym in CRYPTO_ASSETS:
            d_sym = asset_data[sym].get(dt)
            if d_sym is not None:
                sym_ofi = d_sym["ofi"]
                day_ofis.append(sym_ofi)
                new_r[f"{sym}_OFI"] = f"{sym_ofi:.6f}"
                detailed_r[f"{sym}_CLOSE"] = f"{d_sym['close']:.4f}"
                detailed_r[f"{sym}_VOL"] = f"{d_sym['volume']:.2f}"
                detailed_r[f"{sym}_TAKER_BUY"] = f"{d_sym['taker_buy']:.2f}"
                detailed_r[f"{sym}_OFI"] = f"{sym_ofi:.6f}"
            else:
                new_r[f"{sym}_OFI"] = "0.000000"
                detailed_r[f"{sym}_OFI"] = "0.000000"

        comp_ofi = float(np.mean(day_ofis)) if day_ofis else 0.0
        new_r["COMPOSITE_TOP15_OFI"] = f"{comp_ofi:.6f}"
        detailed_r["COMPOSITE_TOP15_OFI"] = f"{comp_ofi:.6f}"

        enriched_rows.append(new_r)
        orderflow_detailed_rows.append(detailed_r)

    # Write enriched crypto_top15.csv
    print(f"Writing enriched crypto dataset with OFI to {CRYPTO_FILE}...")
    with open(CRYPTO_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    # Write detailed orderflow file
    print(f"Writing detailed orderflow breakdown to {ORDERFLOW_FILE}...")
    detailed_fieldnames = list(orderflow_detailed_rows[0].keys())
    with open(ORDERFLOW_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_fieldnames)
        writer.writeheader()
        writer.writerows(orderflow_detailed_rows)

    # Update PROVENANCE.json
    if PROVENANCE_FILE.exists():
        with open(PROVENANCE_FILE, encoding="utf-8") as f:
            prov = json.load(f)
    else:
        prov = {}

    prov["crypto_top15.csv"] = {
        "source": "Binance Public Spot Klines (BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE, DOT, LINK, MATIC, NEAR, ATOM, UNI, LTC)",
        "url": "https://api.binance.com/api/v3/klines",
        "coverage": "2021-01-01 to 2023-12-31 (daily closes + Order Flow Imbalance (OFI), 15 synchronized assets + composite)",
        "features": [
            "observation_date",
            "15 spot closing prices",
            "COMPOSITE_TOP15 price index",
            "15 asset Order Flow Imbalances (OFI = (V_buy - V_sell) / V_total)",
            "COMPOSITE_TOP15_OFI equal-weighted basket order flow",
        ],
        "fetched": datetime.date.today().isoformat(),
        "license_note": "Binance public market data terms apply",
    }
    prov["crypto_top15_orderflow.csv"] = {
        "source": "Binance Public Spot Klines",
        "url": "https://api.binance.com/api/v3/klines",
        "coverage": "2021-01-01 to 2023-12-31 (daily volume, aggressive taker buy volume, taker sell volume, OFI)",
        "fetched": datetime.date.today().isoformat(),
        "license_note": "Binance public market data terms apply",
    }

    with open(PROVENANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print("Done! Order flow successfully integrated and synchronized.")


if __name__ == "__main__":
    main()
