# Preregistration: CPMD-ST3 / v0.16 — Mechanism Discrimination

**Document ID:** PREREG-CPMD-ST3-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_synth_trading3.py` (committed SEPARATELY after)
**Frozen evidence:** ST2 records, OLS-200 rule, positions alignment
  (`position[t]` earns `ret[t+1]`), costs 2bp/unit/side, seeds 20000–20029.
  No estimator/window/threshold changes. ST2 verdicts stand regardless.
**Primary endpoint:** Test A circular-shift null p-value (α=0.05, one-sided).

## Test A — forward-return information (gross, costs excluded)

- Frozen OLS-200 positions on the 30 ST2 paths; statistic = aggregate gross
  next-step return Σ_paths Σ_t position[t]·ret[t+1].
- Null: 1000 replicates; per replicate, circular-shift each path's position
  series by an independent uniform offset (preserves churn/autocorr/exposure,
  breaks signal timing). p = fraction of null aggregates ≥ observed.
- Clustering: statistic aggregated at path level; shifts independent per path.

## Test B — cost damage (identical frozen positions)

- Fee ladder {0, 1bp, 2bp (=ST2 actual), 5bp} per exposure unit traded
  (reversal +1→−1 = 2 units). Turnover = Σ|Δpos| in exposure units.
- Record per fee: net Sharpe, net return, max DD, turnover.
- Break-even cost: fee at which aggregate net return = 0 (linear interp).

## Test C — minimum-hold intervention (hold ≥ 5 steps after each flip)

- Intervention FIXED here (5 steps; no search): after a flip, ignore new
  signals for 5 steps. Slope estimation untouched.
- Exploratory: ST2 paths (20000+). Confirmatory: 30 FRESH paths (40000+).
- Report gross/net Sharpe + turnover for original vs hold-5 on both sets.

## Interpretation (frozen)

| Evidence | Permitted conclusion |
|---|---|
| Test A p ≥ 0.05 | Forward-return information NOT demonstrated |
| Test A p < 0.05 but net ≤ 0 at 2bp | Signal exists; costs dominate → turnover material |
| Hold-5 improves net on fresh paths | Churn reduction helps; NOT promotability |
| Gross absent + costs worsen | Both contribute; neither exclusive |
| Wide CIs / noisy | INSUFFICIENT_EVIDENCE; no forced verdict |

No promotion follows any ST3 outcome. v0.15 estimator stays protected.
