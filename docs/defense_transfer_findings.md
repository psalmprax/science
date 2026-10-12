# Defense-Transfer Findings: Baseline (B) vs Hysteresis (H)

Frozen predictors, pre-registered thresholds, 5 bps costs throughout.
Research record only — not financial advice; hypothetical simulations.

## 1. Configurations

| ID | Predictor | Direction | Thresholds |
|----|-----------|-----------|------------|
| A-short | Score_30 4-term (0.8111/−1.4272/0.5984/0.6740) | short iff S<0 | none |
| A-hyst | same | hysteresis | train-half 75th/50th pct of |S|, frozen |
| B | 15-asset GA (0.828·RV + 0.556·BB + 0.292·MOM) | symmetric ± | none |
| B-hyst | same | symmetric hysteresis | train-half 75th/50th pct, frozen |
| B-expo | same as B | baseline timing × exposure scale | matched to H exposure |

## 2. Equation B symmetric, DEFENSE-TRANSFER-v1 (2021–2023 panels)

| U | B net/DD | H net/DD | E net/DD | H−E net (pp) |
|---|----------|----------|----------|--------------|
| 15 | +106.39 / 30.8 | +78.28 / 62.2 | +38.92 / 14.7 | +39.36 |
| 30 | +45.10 / 28.0 | +26.32 / 23.0 | +20.75 / 13.4 | +5.57 |
| 60 | +24.32 / 54.0 | +14.18 / 22.1 | +13.52 / 27.0 | +0.66 |
| 90 | −25.56 / 66.6 | −9.18 / 26.5 | −11.39 / 31.5 | +2.21 |

H beats exposure-matched E on net 4/4; on drawdown 2/4 (60, 90). U=15 DD blowout (62.2%) stands as counterexample.

## 3. Short-only 4-term panels (2021–2023)

| U | Base net | Hyst net | Turnover cut |
|---|----------|----------|--------------|
| 30 | −28.58% | −5.79% | −66% |
| 60 | −22.75% | −6.89% | −66% |
| 90 | −11.40% | −9.89% | −73% |

Per-asset equal weight: 30: 2470W/2810L; 60: 5120W/5733L; 90: 7993W/8532L. All sub-coin-flip.

## 4. Tail-concentration audit (Equation B, log-return space)

Top-5 P&L share ~half under both rules on every universe; removing top 5 roughly halves positives, deepens negatives. Skew positive everywhere (+4 to +7); decreases under H on 15/30/60, not 90. Winners held 3–4× longer than losers under both rules.

## 5. Forward / independent windows

| Window | Base net | Hyst net | Note |
|--------|----------|----------|------|
| 2024–25, 90-asset panel | — | — | frozen 4-term: crypto −28.1%, equity −14.7% |
| 2026-01–10 BTC daily | +5.41% | +4.35% | first positive OOS; single asset |
| 2026 cohort ETH/SOL/BNB/XRP | mixed | H wins net 4/4 vs expo | untouched data |

## 6. Intraday (hourly BTC, 3000 bars)

Long/short: −26.06% vs buy-hold −21.24%. Short-only: −14.94%, DD 17.19%, 288 holds × 5.4 bars. 4-universe short-only: all negative net.

## 7. Ledger verdicts

* Hysteresis reduces turnover: SUPPORTED (66–73% cuts everywhere).
* Hysteresis reduces drawdown: MIXED (2/4 vs exposure-matched; U=15 blowout).
* Hysteresis adds net beyond exposure: SUPPORTED 4/4 on Equation B (margins +39/+5.6/+0.7/+2.2pp).
* Predictor has robust edge: NOT ESTABLISHED.
* Combined system profitable across universes: NOT SUPPORTED (U=90 negative).
* Rank-invariant version: FALSIFIED by ablation.
* Regime gate: UNRESOLVED (no-op design failure, frozen).
* Universal law: NOT SUPPORTED.

## 8. Full equations (as executed)

Score_30: S = 0.8111·BB − 1.4272·OBV_MOM + 0.5984·TAKER_MOM + 0.6740·MASTER_SNR, thr 0.02.
Equation B: S = 0.828·RV + 0.556·BB + 0.292·MOM, thr per-universe train-half 75th/50th pct.

* BB: z((P − Lower)/(Upper − Lower)), 20-day SMA ± 2σ, causal z-score.
* OBV_MOM: OBV accumulates V·sgn(ΔP); z of 10-bar pct change.
* TAKER_MOM: z(z(CVD)·z(MOM5)); CVD = (2·tb − V)/V; MOM5 = 5-bar pct change.
* MASTER_SNR: z(z(MOM5)/σ_asym²); σ_asym = max(0.01 + 0.3|r| + 0.5·I(r<0)|r|, 1e-4).
* RV: trailing percentile rank of V.
* All z(·) are causal (trailing-only) standardizations.
* Base short-only: p = −1 iff S < 0 else 0. Symmetric base: ±1 by sign(S).
* Hysteresis: enter iff |S| beyond entry pct; hold until |S| recovers past exit pct.
* Exposure control: baseline timing × mean|H|/mean|B| (≤1), costs on scaled turnover.

## 9. Standing caveats

* Most panel windows inspected repeatedly — exploratory except 2026 cohort + prospective freeze (`scratch/prospective_defense_v1.json`, PENDING).
* Single-asset transfer fails directionally and economically.
* OFI proxied from taker-buy imbalance on single-asset runs.
* Fee model 5 bps, zero impact/slippage beyond spread.
* Sharpe_ac figures exclude flip costs where noted; net is the decision metric.
