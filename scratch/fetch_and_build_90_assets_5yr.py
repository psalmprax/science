import datetime
import time
from pathlib import Path
import httpx
import numpy as np
import pandas as pd

DATA_DIR = Path("data/market")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILE_90 = DATA_DIR / "multi_asset_90_5yr_orderflow.csv"

# 45 Cryptocurrencies
CRYPTOS = [
    # Top 30
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "UNI", "LTC", "ATOM", "NEAR", "BCH", "XLM", "TRX", "ETC", "FIL",
    "VET", "THETA", "ALGO", "AAVE", "MKR", "EOS", "XMR", "KSM", "ZEC", "MANA",
    # 15 Additional Liquid 5-Year Tokens
    "SAND", "AXS", "CRV", "SNX", "COMP", "GRT", "SUSHI", "1INCH", "ENJ", "BAT",
    "CHZ", "FTM", "ZIL", "ICX", "ONT"
]

# 45 Equities, ETFs & Macro Assets
EQUITIES = [
    # Top 30 Existing
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "UNH", "SPY", "QQQ", "IWM", "GLD", "TLT", "BAC",
    "HD", "CVX", "LLY", "ABBV", "COST", "MRK", "KO", "PEP", "AMD", "NFLX",
    # 15 Additional Traditional / Sectors / Macro Assets
    "INTC", "CRM", "ORCL", "ADBE", "CSCO", "QCOM",
    "XLE", "XLF", "XLV", "XLI", "XLU", "XLK", "XLP",
    "EEM", "USO"
]

print(f"Target Universe: {len(CRYPTOS)} Cryptos + {len(EQUITIES)} Equities = {len(CRYPTOS) + len(EQUITIES)} Assets")
print("Target Horizon: 5 Full Years (2021-01-01 to 2025-12-31)")

headers = {"User-Agent": "Mozilla/5.0"}
p1 = int(datetime.datetime(2021, 1, 1).timestamp())
p2 = int(datetime.datetime(2025, 12, 31, 23, 59, 59).timestamp())

# 1. Fetch SPY to establish master trading days calendar
print("Fetching master US equity calendar via SPY...")
r_spy = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1={p1}&period2={p2}&interval=1d", headers=headers, timeout=15.0)
if r_spy.status_code != 200:
    raise RuntimeError("Failed to fetch SPY master calendar")
res_spy = r_spy.json()["chart"]["result"][0]
ts_list = res_spy["timestamp"]
dates_master = [datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in ts_list]
# Filter to 2021-01-01 through 2025-12-31
master_calendar = [d for d in dates_master if "2021-01-01" <= d <= "2025-12-31"]
master_calendar_set = set(master_calendar)
print(f"Master Calendar established: {len(master_calendar)} trading days (from {master_calendar[0]} to {master_calendar[-1]})")

# 2. Fetch all 45 Equities/ETFs
equity_data = {}
for i, sym in enumerate(EQUITIES):
    print(f"[{i+1}/{len(EQUITIES)}] Fetching Yahoo Finance: {sym}...")
    success = False
    for attempt in range(3):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d"
            r = httpx.get(url, headers=headers, timeout=15.0)
            if r.status_code == 200:
                res = r.json()["chart"]["result"][0]
                timestamps = res["timestamp"]
                q = res["indicators"]["quote"][0]
                dates = [datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps]
                
                df_e = pd.DataFrame({
                    "observation_date": dates,
                    f"{sym}_CLOSE": q["close"],
                    f"{sym}_HIGH": q["high"],
                    f"{sym}_LOW": q["low"],
                    f"{sym}_VOL": q["volume"],
                }).drop_duplicates(subset=["observation_date"])
                
                # Align to master calendar
                df_aligned = pd.DataFrame({"observation_date": master_calendar}).merge(df_e, on="observation_date", how="left")
                # Clean nulls using forward fill
                df_aligned = df_aligned.ffill().bfill()
                
                # Synthetic OFI / taker buy proxy
                c = df_aligned[f"{sym}_CLOSE"].to_numpy(float)
                v = df_aligned[f"{sym}_VOL"].to_numpy(float)
                ret = np.append(0.0, np.diff(np.log(np.maximum(c, 1e-8))))
                vol_smooth = pd.Series(v).rolling(5, min_periods=1).mean().to_numpy()
                tb = v * np.clip(0.5 + 2.5 * ret, 0.1, 0.9)
                ofi = (2.0 * tb - v) / np.maximum(v, 1e-12)
                
                df_aligned[f"{sym}_TAKER_BUY"] = tb
                df_aligned[f"{sym}_OFI"] = np.clip(ofi, -1.0, 1.0)
                
                equity_data[sym] = df_aligned.set_index("observation_date")
                success = True
                break
            else:
                time.sleep(0.5)
        except Exception as exc:
            time.sleep(1.0)
    if not success:
        raise RuntimeError(f"Failed to fetch equity {sym}")

# 3. Fetch all 45 Cryptocurrencies from Binance
crypto_data = {}
# Chunk timestamps: Binance returns max 1000 candles per call
# 5 years is ~1826 days -> 2 calls per crypto: (2021-2023) and (2023-2025)
t_chunks = [
    (int(datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000),
     int(datetime.datetime(2023, 6, 30, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000)),
    (int(datetime.datetime(2023, 7, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000),
     int(datetime.datetime(2025, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000)),
]

for i, sym in enumerate(CRYPTOS):
    print(f"[{i+1}/{len(CRYPTOS)}] Fetching Binance spot: {sym}USDT...")
    all_klines = []
    for s_ms, e_ms in t_chunks:
        success = False
        for attempt in range(3):
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={sym}USDT&interval=1d&startTime={s_ms}&endTime={e_ms}&limit=1000"
                r = httpx.get(url, headers=headers, timeout=15.0)
                if r.status_code == 200:
                    all_klines.extend(r.json())
                    success = True
                    break
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(1.0)
        if not success:
            raise RuntimeError(f"Failed to fetch Binance crypto {sym}")
        time.sleep(0.1)
    
    # Process klines
    c_dates, closes, vols, highs, lows, tb_vols = [], [], [], [], [], []
    for k in all_klines:
        d = datetime.datetime.fromtimestamp(k[0] / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        c_dates.append(d)
        highs.append(float(k[2]))
        lows.append(float(k[3]))
        closes.append(float(k[4]))
        vols.append(float(k[5]))
        tb_vols.append(float(k[9]))
    
    df_c = pd.DataFrame({
        "observation_date": c_dates,
        f"{sym}_CLOSE": closes,
        f"{sym}_HIGH": highs,
        f"{sym}_LOW": lows,
        f"{sym}_VOL": vols,
        f"{sym}_TAKER_BUY": tb_vols,
    }).drop_duplicates(subset=["observation_date"])
    
    # Align to master calendar
    df_aligned = pd.DataFrame({"observation_date": master_calendar}).merge(df_c, on="observation_date", how="left")
    df_aligned = df_aligned.ffill().bfill()
    
    v_arr = df_aligned[f"{sym}_VOL"].to_numpy(float)
    tb_arr = df_aligned[f"{sym}_TAKER_BUY"].to_numpy(float)
    ofi_arr = (2.0 * tb_arr - v_arr) / np.maximum(v_arr, 1e-12)
    df_aligned[f"{sym}_OFI"] = np.clip(ofi_arr, -1.0, 1.0)
    
    crypto_data[sym] = df_aligned.set_index("observation_date")

# 4. Merge all 90 assets into one synchronized dataframe
print("Merging all 90 assets into single unified dataframe...")
master_df = pd.DataFrame({"observation_date": master_calendar}).set_index("observation_date")

for sym in CRYPTOS:
    master_df = master_df.join(crypto_data[sym])

for sym in EQUITIES:
    master_df = master_df.join(equity_data[sym])

master_df = master_df.reset_index()

# Quality Checks
null_counts = master_df.isnull().sum().sum()
print(f"Final Merged Shape: {master_df.shape} (Rows: {len(master_df)}, Cols: {master_df.shape[1]})")
print(f"Total Null Values: {null_counts}")
assert null_counts == 0, f"Found {null_counts} nulls in merged dataset!"

# Save to disk
master_df.to_csv(FILE_90, index=False)
print(f"Successfully saved 90-asset 5-year dataset to {FILE_90}!")
