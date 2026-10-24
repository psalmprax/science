import datetime
import time
from pathlib import Path
import httpx
import numpy as np
import pandas as pd

DATA_DIR = Path("data/market")
FILE_30 = DATA_DIR / "multi_asset_30_orderflow.csv"
FILE_60 = DATA_DIR / "multi_asset_60_orderflow.csv"

# Load existing 30-asset dataframe to extract the exact calendar of 753 dates
df_30 = pd.read_csv(FILE_30)
target_dates = set(df_30["observation_date"])
dates_list = df_30["observation_date"].tolist()
print(f"Loaded existing 30-asset dataset: {len(df_30)} rows, dates from {dates_list[0]} to {dates_list[-1]}")

NEW_CRYPTOS = [
    "BCH", "XLM", "TRX", "ETC", "FIL", "VET", "THETA", "ALGO",
    "AAVE", "MKR", "EOS", "XMR", "KSM", "ZEC", "MANA"
]

NEW_EQUITIES = [
    "QQQ", "IWM", "GLD", "TLT", "BAC", "HD", "CVX", "LLY",
    "ABBV", "COST", "MRK", "KO", "PEP", "AMD", "NFLX"
]

print(f"Adding {len(NEW_CRYPTOS)} cryptos and {len(NEW_EQUITIES)} equities -> 60 total assets")

# 1. Fetch new Cryptos from Binance
crypto_dfs = {}
headers = {"User-Agent": "Mozilla/5.0"}

for sym in NEW_CRYPTOS:
    print(f"Fetching Binance klines for {sym}USDT...")
    all_klines = []
    # Call 1: 2021 to 2022
    s1 = int(datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    e1 = int(datetime.datetime(2022, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    url1 = f"https://api.binance.com/api/v3/klines?symbol={sym}USDT&interval=1d&startTime={s1}&endTime={e1}&limit=1000"
    r1 = httpx.get(url1, headers=headers, timeout=10.0)
    if r1.status_code == 200:
        all_klines.extend(r1.json())
    
    # Call 2: 2023
    s2 = int(datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    e2 = int(datetime.datetime(2023, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    url2 = f"https://api.binance.com/api/v3/klines?symbol={sym}USDT&interval=1d&startTime={s2}&endTime={e2}&limit=1000"
    r2 = httpx.get(url2, headers=headers, timeout=10.0)
    if r2.status_code == 200:
        all_klines.extend(r2.json())

    # Build per-symbol dataframe
    c_dates, closes, vols, highs, lows, taker_buys = [], [], [], [], [], []
    for k in all_klines:
        d = datetime.datetime.fromtimestamp(k[0] / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        c_dates.append(d)
        highs.append(float(k[2]))
        lows.append(float(k[3]))
        closes.append(float(k[4]))
        vols.append(float(k[5]))
        taker_buys.append(float(k[9]))

    df_c = pd.DataFrame({
        "observation_date": c_dates,
        f"{sym}_CLOSE": closes,
        f"{sym}_VOL": vols,
        f"{sym}_HIGH": highs,
        f"{sym}_LOW": lows,
        f"{sym}_TAKER_BUY": taker_buys,
    }).drop_duplicates(subset=["observation_date"])

    v_arr = df_c[f"{sym}_VOL"].to_numpy(float)
    tb_arr = df_c[f"{sym}_TAKER_BUY"].to_numpy(float)
    ofi_arr = (2.0 * tb_arr - v_arr) / np.maximum(v_arr, 1e-12)
    df_c[f"{sym}_OFI"] = np.clip(ofi_arr, -1.0, 1.0)

    # Filter to target dates
    df_c = df_c[df_c["observation_date"].isin(target_dates)].set_index("observation_date")
    crypto_dfs[sym] = df_c

# 2. Fetch new Equities from Yahoo Finance
equity_dfs = {}
p1 = int(datetime.datetime(2021, 1, 1).timestamp())
p2 = int(datetime.datetime(2023, 12, 31, 23, 59, 59).timestamp())

for sym in NEW_EQUITIES:
    print(f"Fetching Yahoo Finance chart for {sym}...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d"
    r = httpx.get(url, headers=headers, timeout=10.0)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch {sym}: {r.status_code}")
    res = r.json()["chart"]["result"][0]
    timestamps = res["timestamp"]
    quote = res["indicators"]["quote"][0]
    dates = [datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps]

    df_e = pd.DataFrame({
        "observation_date": dates,
        f"{sym}_CLOSE": quote["close"],
        f"{sym}_VOL": quote["volume"],
        f"{sym}_HIGH": quote["high"],
        f"{sym}_LOW": quote["low"],
        f"{sym}_OPEN": quote.get("open", quote["close"]),
    }).dropna().drop_duplicates(subset=["observation_date"])

    ret = df_e[f"{sym}_CLOSE"].pct_change().fillna(0.0).to_numpy()
    vol = df_e[f"{sym}_VOL"].to_numpy(float)
    vol_mean = pd.Series(vol).rolling(20, min_periods=1).mean().to_numpy()
    ofi = np.sign(ret) * (vol / np.maximum(vol_mean, 1e-12))
    df_e[f"{sym}_OFI"] = np.clip(ofi, -3.0, 3.0)
    df_e[f"{sym}_TAKER_BUY"] = vol * (0.5 + 0.25 * np.sign(ret))

    df_e = df_e[df_e["observation_date"].isin(target_dates)].set_index("observation_date")
    equity_dfs[sym] = df_e

# 3. Merge everything into the 60-asset dataset
print("Merging all 60 assets...")
df_merged = df_30.set_index("observation_date")

for sym in NEW_CRYPTOS:
    df_merged = df_merged.join(crypto_dfs[sym], how="left")

for sym in NEW_EQUITIES:
    df_merged = df_merged.join(equity_dfs[sym], how="left")

# Forward fill any potential single holiday gaps
df_merged = df_merged.ffill().bfill().reset_index()

print(f"Final 60-asset dataset shape: {df_merged.shape}")
df_merged.to_csv(FILE_60, index=False)
print(f"Saved successfully to {FILE_60}")
