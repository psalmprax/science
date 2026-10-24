import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    calc_signals_for_asset, SIGNAL_NAMES, simulate_multi_asset_portfolio
)

df_all = pd.read_csv('data/market/multi_asset_30_orderflow.csv')

crypto_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'LINK', 'MATIC', 'NEAR', 'ATOM', 'UNI', 'LTC']
equity_symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'XOM', 'UNH', 'SPY']
all_symbols = crypto_symbols + equity_symbols

universe_30 = {}
for s in all_symbols:
    p = df_all[f'{s}_CLOSE'].to_numpy(float)
    r = np.zeros(len(p))
    r[1:] = np.diff(np.log(p))
    v = df_all[f'{s}_VOL'].to_numpy(float)
    h = df_all[f'{s}_HIGH'].to_numpy(float) if f'{s}_HIGH' in df_all.columns else p * 1.01
    l = df_all[f'{s}_LOW'].to_numpy(float) if f'{s}_LOW' in df_all.columns else p * 0.99
    tb = df_all[f'{s}_TAKER_BUY'].to_numpy(float) if f'{s}_TAKER_BUY' in df_all.columns else v * 0.5
    ofi = df_all[f'{s}_OFI'].to_numpy(float) if f'{s}_OFI' in df_all.columns else np.zeros(len(p))
    universe_30[s] = pd.DataFrame({'price': p, 'r': r, 'V': v, 'high': h, 'low': l, 'taker_buy': tb, 'ofi': ofi})

returns_30 = np.column_stack([universe_30[s]['r'].to_numpy() for s in all_symbols])
prices_30 = np.column_stack([universe_30[s]['price'].to_numpy() for s in all_symbols])
median_ret_30 = np.median(returns_30, axis=1)

vols_30 = np.zeros_like(returns_30)
for i in range(len(all_symbols)):
    r_series = pd.Series(returns_30[:, i])
    vols_30[:, i] = r_series.rolling(20, min_periods=5).std().bfill().to_numpy()

T, N = len(df_all), len(all_symbols)
K = len(SIGNAL_NAMES)
features_3d = np.zeros((T, N, K))
for i, s in enumerate(all_symbols):
    features_3d[:, i, :] = calc_signals_for_asset(universe_30[s], median_ret_30)

n_train = T // 2
f_ho = features_3d[n_train:]
r_ho = returns_30[n_train:]
v_ho = vols_30[n_train:]
p_ho = prices_30[n_train:]

# Dynamic 30-Asset In-Sample Genetic Algorithm Evolution (No hardcoded weights)
from apedp_dyn.sandbox.multi_asset_signals_sandbox import run_multi_asset_genetic_algorithm
best_chrom = run_multi_asset_genetic_algorithm(
    features_3d[:n_train], returns_30[:n_train], vols_30[:n_train],
    pop_size=80, generations=35, seed=42
)
eff_w = best_chrom.mask.astype(float) * best_chrom.weights
thr_30 = best_chrom.threshold
scores_30_ho = f_ho @ eff_w  # (T_ho, 30)


# Extract non-linear sizing terms for Master Sizing Phi
idx_sqz = SIGNAL_NAMES.index('H-SQUEEZE (BB/Keltner Squeeze)')
idx_hurst = SIGNAL_NAMES.index('H-HURST (Hurst Regime Memory)')
idx_asym = SIGNAL_NAMES.index('H-ASYM (GJR Panic Leverage)')

sqz_ho = f_ho[:, :, idx_sqz]
hurst_ho = f_ho[:, :, idx_hurst]
asym_vols_ho = np.maximum(v_ho + 0.5 * np.maximum(f_ho[:, :, idx_asym], 0.0) * v_ho, 0.005)

# 1. Backtest 30-Asset Unified Portfolio
ret_rp_ls, sr_rp_ls, acc_rp_ls, pnl_rp_ls, tr_rp_ls, pf_rp_ls, dd_rp_ls, wr_rp_ls = simulate_multi_asset_portfolio(
    scores_30_ho, r_ho, v_ho, mode='cross_sectional_ls', sizing='risk_parity', threshold=thr_30
)

ret_eq_ls, sr_eq_ls, acc_eq_ls, pnl_eq_ls, tr_eq_ls, pf_eq_ls, dd_eq_ls, wr_eq_ls = simulate_multi_asset_portfolio(
    scores_30_ho, r_ho, v_ho, mode='cross_sectional_ls', sizing='equal', threshold=thr_30
)

ret_phi_ls, sr_phi_ls, acc_phi_ls, pnl_phi_ls, tr_phi_ls, pf_phi_ls, dd_phi_ls, wr_phi_ls = simulate_multi_asset_portfolio(
    scores_30_ho, r_ho, v_ho, mode='cross_sectional_ls', sizing='master_snr', threshold=thr_30,
    sqz_2d=sqz_ho, hurst_2d=hurst_ho, asym_vols_2d=asym_vols_ho
)

ret_rp_dir, sr_rp_dir, acc_rp_dir, pnl_rp_dir, tr_rp_dir, pf_rp_dir, dd_rp_dir, wr_rp_dir = simulate_multi_asset_portfolio(
    scores_30_ho, r_ho, v_ho, mode='directional', sizing='risk_parity', threshold=thr_30
)

# 2. Test 30-asset equation on 15 Crypto sub-portfolio
scores_crypto = scores_30_ho[:, :15]
ret_c_rp, sr_c_rp, _, pnl_c_rp, tr_c_rp, _, dd_c_rp, wr_c_rp = simulate_multi_asset_portfolio(
    scores_crypto, r_ho[:, :15], v_ho[:, :15], mode='cross_sectional_ls', sizing='risk_parity', threshold=thr_30
)

# 3. Test 30-asset equation on 15 Equities sub-portfolio
scores_equity = scores_30_ho[:, 15:]
ret_e_rp, sr_e_rp, _, pnl_e_rp, tr_e_rp, _, dd_e_rp, wr_e_rp = simulate_multi_asset_portfolio(
    scores_equity, r_ho[:, 15:], v_ho[:, 15:], mode='cross_sectional_ls', sizing='risk_parity', threshold=thr_30
)

# Benchmarks Buy & Hold
bnh_30 = float(np.mean((p_ho[-1, :] - p_ho[0, :]) / p_ho[0, :]) * 100.0)
bnh_crypto = float(np.mean((p_ho[-1, :15] - p_ho[0, :15]) / p_ho[0, :15]) * 100.0)
bnh_equity = float(np.mean((p_ho[-1, 15:] - p_ho[0, 15:]) / p_ho[0, 15:]) * 100.0)

print('=' * 100)
print('OUT-OF-SAMPLE RESULTS: 30-ASSET MULTI-ASSET UNIVERSAL EQUATION (377 HOLDOUT DAYS)')
print('=' * 100)
dash = '-' * 100
print(f'{"Portfolio Strategy":<42} | {"Return (%)":<10} | {"P&L ($)":<13} | {"Sharpe":<7} | {"Max DD":<8} | {"Trades"}')
print(dash)
print(f'{"30-Asset GA L/S Risk Parity":<42} | {ret_rp_ls:+9.2f}% | ${pnl_rp_ls:+12.2f} | {sr_rp_ls:+6.2f} | {dd_rp_ls:7.1f}% | {tr_rp_ls}')
print(f'{"30-Asset GA L/S Equal Weight":<42} | {ret_eq_ls:+9.2f}% | ${pnl_eq_ls:+12.2f} | {sr_eq_ls:+6.2f} | {dd_eq_ls:7.1f}% | {tr_eq_ls}')
print(f'{"30-Asset GA L/S Master Sizing Phi":<42} | {ret_phi_ls:+9.2f}% | ${pnl_phi_ls:+12.2f} | {sr_phi_ls:+6.2f} | {dd_phi_ls:7.1f}% | {tr_phi_ls}')
print(f'{"30-Asset GA Directional Risk Parity":<42} | {ret_rp_dir:+9.2f}% | ${pnl_rp_dir:+12.2f} | {sr_rp_dir:+6.2f} | {dd_rp_dir:7.1f}% | {tr_rp_dir}')
print(dash)
print(f'{"-> Crypto Sub-Portfolio (15 Coins)":<42} | {ret_c_rp:+9.2f}% | ${pnl_c_rp:+12.2f} | {sr_c_rp:+6.2f} | {dd_c_rp:7.1f}% | {tr_c_rp}')
print(f'{"-> Equity Sub-Portfolio (15 Stocks)":<42} | {ret_e_rp:+9.2f}% | ${pnl_e_rp:+12.2f} | {sr_e_rp:+6.2f} | {dd_e_rp:7.1f}% | {tr_e_rp}')
print(dash)
print(f'{"Benchmark: 30-Asset Buy & Hold Basket":<42} | {bnh_30:+9.2f}% | ${100000*(bnh_30/100):+12.2f} | {"--":>6} | {"--":>7} | 30')
print(f'{"Benchmark: 15-Crypto Buy & Hold Basket":<42} | {bnh_crypto:+9.2f}% | ${100000*(bnh_crypto/100):+12.2f} | {"--":>6} | {"--":>7} | 15')
print(f'{"Benchmark: 15-Equity Buy & Hold Basket":<42} | {bnh_equity:+9.2f}% | ${100000*(bnh_equity/100):+12.2f} | {"--":>6} | {"--":>7} | 15')
print('=' * 100)
