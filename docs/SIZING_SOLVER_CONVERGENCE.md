# Sizing solver convergence audit (2026-11-07)

`apedp_dyn.sandbox.finance_l2_demo.size_positions` sizes a position by solving a
cost-penalised mean-variance problem over a forecast `μ`, a covariance `Σ`, a fee
and a stated risk aversion `γ`:

```
maximise  w'μ − (γ/2) w'Σw − c‖w − w_prev‖₁      subject to  w ≥ 0,  Σw ≤ cap
```

The solver is proximal gradient on the capped simplex with a fixed step
`1 / (γ λ_max)`. That step is cheap and monotone, but its convergence rate
depends on the covariance condition number `κ = λ_max / λ_min`, so the audit
measures the rate rather than asserting it.

## What is measured

`scripts/audit_sizing_solver_convergence.py` builds correlated SPD covariances
(random orthogonal eigenbasis, eigenvalues `logspace(0, −log10 κ)`) for
`n = 2, 4, 8`, `κ = 10² … 10¹²`, and previous weights that are either zero or a
random portfolio. For each of the 36 cases it records:

- the analytic KKT violation of the returned weights,
- whether that violation is at most `1e-10` (`converged`),
- the iterations used,
- the objective gap and maximum weight delta against a 200,000-iteration run at
  tolerance `1e-15`,
- gross exposure and whether the cap binds.

The record is `benchmarks/sizing_solver_convergence.json`.

## Findings

**All 36 cases converge**, where `converged` now means an analytic KKT
violation of at most `1e-10`. The worst residual in the record is **2.0e-12**
(`n = 8`, `κ = 1e2`, nonzero previous weights). The record still carries the
iteration count and the objective gap against a 200,000-iteration run, but the
stopping rule no longer depends on the gradient loop alone.

The grid exposed two different failures, which is why the fix is not a larger
iteration cap:

1. **The fixed step can exhaust its budget.** `n = 8`, `κ = 1e4` and `κ = 1e12`
   with zero previous weights hit 4,000 iterations. The active set and the L1
   sign pattern were already settled; solving the KKT system on that face takes
   both to machine precision.
2. **The fixed-point residual can report convergence at an L1 kink.** For
   `n = 8`, `κ = 1e12` with nonzero previous weights, the proximal map's
   fixed-point residual was 9.9e-13 while the analytic KKT violation was
   **3.9e-04** and the objective was **0.0077** against the optimum's
   **0.0147**. Two coordinates flip sign as they cross their previous weights,
   so a sign-flipping active-set loop cycles instead of converging.

`size_positions` now reports the analytic KKT violation and, when the loop has
not met it, (a) solves the active-set KKT system on the identified face and
(b) if that still cycles, solves the smooth `t >= |w − previous|` QP and snaps
the result onto any kink it finds. The kink snap matters: a coordinate resting
at its previous weight satisfies an interval of stationarity conditions, not an
equality, so the residual minimizes the violation over the feasible multiplier
instead of averaging the interval.

Condition number alone does not predict the iteration count: many high-`κ` cases
terminate in tens of iterations because the optimum sits on a vertex or the cap
face and the projection resolves it, while moderate `κ` with `n = 8` can need
thousands. The audit therefore records the whole grid rather than a headline
`κ` threshold.

## Real-market walk-forward

`scripts/audit_sizing_real_market.py` applies the rule out of sample to the
repository's real panels: `data/market/equity_top15.csv` (14 US equities,
2021-2023) and `data/market/crypto_top15.csv` (14 Binance spot tokens,
2021-2023). The forecast is the trailing 126-day mean, the covariance is the
sample covariance of the same window, and the book is rebalanced every 5 days
with the fee charged on L1 turnover. The benchmark asset (SPY or BTC) is
excluded from the investable universe and shown only as a reference.

| panel | fee | ann. return | ann. vol | Sharpe | max DD | avg turnover | no-trade | fees | max KKT |
|---|---|---|---|---|---|---|---|---|---|
| equity | 0 | 50.70% | 35.02% | 1.35 | -21.08% | 0.294 | 60.8% | 0.00% | 1.2e-14 |
| equity | 5 | 38.31% | 36.21% | 1.08 | -31.50% | 0.123 | 82.4% | 0.77% | 3.2e-15 |
| equity | 20 | 40.44% | 40.57% | 1.03 | -37.49% | 0.041 | 92.0% | 1.02% | 7.5e-15 |
| equity | equal-weight hold | 11.81% | 21.27% | 0.63 | -24.77% | -- | -- | 0.20% | -- |
| equity | SPY hold | 3.96% | 18.42% | 0.30 | -25.36% | -- | -- | -- | -- |
| crypto | 0 | 43.14% | 95.45% | 0.84 | -76.49% | 0.373 | 17.6% | 0.00% | 1.4e-13 |
| crypto | 5 | 54.06% | 100.35% | 0.90 | -75.06% | 0.195 | 48.2% | 1.88% | 1.4e-13 |
| crypto | 20 | 22.39% | 94.62% | 0.67 | -76.75% | 0.088 | 66.8% | 3.39% | 1.4e-13 |
| crypto | equal-weight hold | -8.82% | 79.60% | 0.32 | -80.51% | -- | -- | 0.20% | -- |
| crypto | BTC hold | -10.64% | 61.28% | 0.12 | -76.63% | -- | -- | -- | -- |

All six runs report **0 unconverged rebalances** (125 equity, 193 crypto); the
worst residual is 1.4e-13. The trailing mean produces concentrated books (1-2
positions on average) and the annualized figures come from 2.5 years, so the
returns are a description of this path, not evidence of edge. Fees are charged
on turnover; the benchmark rows are gross.

The real panels found three defects the synthetic grid did not:

1. **Previous-weight dust.** A coordinate sold to zero in an earlier solve can
   retain ~1e-18 in `previous`; the sold-out KKT rule treated that as a held
   position and charged the full fee for re-entering it.
2. **Kink-only multiplier.** On a cap-binding book whose active coordinates are
   all at their previous weights, the multiplier is an interval. Averaging the
   interval invented violations in 18 of the 24 flagged rebalances.
3. **Partial trade versus no trade.** The other 6 flagged rebalances had a
   partial split whose utility was below simply holding the previous book; an
   independent multi-start QP found the no-trade book better by 4.5e-5 to
   3.2e-4 in utility.

## Reproduce

```
.venv/bin/python scripts/audit_sizing_solver_convergence.py
.venv/bin/python scripts/audit_sizing_real_market.py
```

The script always exits 0: it is a measurement, and a non-converged case is a
result to read, not a reason to hide the record. Rerun it after changing the
step rule, the tolerance or the default iteration budget, and update this page
and the JSON together.

## Limitations

- The long run is a reference, not an exact solution; a case where both runs stop
  short would understate the gap, though the reported KKT residual would expose
  it.
- The synthetic audit covers the demo sizing rule at its default `γ = 1` and
  5 bps fee; the real-market run sweeps 0, 5 and 20 bps at `γ = 1`.
- The real-market walk-forward is a mechanical check with one forecast rule
  (trailing mean). Its returns are a single 2021-2023 path, not evidence of
  edge; the concentrated books and high crypto drawdowns are reported to show
  the rule's behaviour, not a strategy result.
- The real panels are well conditioned (median `κ` 77 equity, 170 crypto), so
  the synthetic `κ = 1e12` cases remain the conditioning stress test.
- `converged` means the analytic KKT violation is at most `1e-10`; the loop's
  own `1e-12` weight-change tolerance is a stopping heuristic, not the
  convergence claim.
