# Preregistration: CPMD-ST1 — Synthetic Trading Test of OLS-T Slope

**Document ID:** PREREG-CPMD-ST1-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_synth_trading.py` (committed SEPARATELY after)
**Standing:** v0.15 slope estimation is PRESERVED untouched; profitability is a
  NEW hypothesis. OLS-T winning estimation gates implies nothing about returns.
**Real markets:** not involved. No promotion path to real trading from ST1 alone.

## Synthetic market (known rules, hidden regimes)

- Hidden 2-state Markov chain: PERSIST (AR(1) φ=+0.3) / REVERT (φ=−0.3),
  P(stay)=0.98, on Gaussian shocks σ=0.01. 3000 steps/path.
- OLS-T (frozen: lags 4–12 log-log slope of |return| decay curve, past
  200 observations ONLY, strictly causal) classifies regime per step:
  slope < median(training) → PERSIST else REVERT. Median frozen on burn-in.

## Trading rule (frozen before run)

- PERSIST → momentum: position = sign(past-5-day return).
- REVERT → reversal: position = −sign(past-5-day return).
- Costs 2bp per side per trade; positions ∈ {−1, 0, +1}; flat allowed only
  when OLS-T abstains (CI width > 1.0 — same cap).

## Baselines

Buy-hold, always-momentum, always-reversal, random-sign (seeded).

## Evaluation (seeds 20000+, 30 paths)

Net Sharpe, total net return, max drawdown, turnover per path; OLS-T rule
vs each baseline by paired Wilcoxon (α=0.05).

## Gates (ST1 passes iff ALL hold)

1. Positive mean net Sharpe across paths.
2. Beats buy-hold AND best naive baseline on net Sharpe (Wilcoxon p<0.05).
3. Max drawdown < 2× buy-hold drawdown (survivability).
No gate changes after results. Failure = OLS-T not a trading signal here.
