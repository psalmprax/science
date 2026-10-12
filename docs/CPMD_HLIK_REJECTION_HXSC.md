# H-LIK Real-Market Rejection + H-XSC Proposal

**H-LIK status: synthetic survivor, real-market hypothesis REJECTED.**
Frozen config ($W=200$, $L=6$, $\lambda=1.0$) stays frozen — no rescue tuning.
Supplementary 200d/300d cuts confirm; they do not revise the verdict.

## 1. Final H-LIK verdict table

| Evidence stage | Verdict |
|---|---|
| Synthetic discovery, gen-D | Passed (10/10, all $p < 0.005$) |
| Synthetic controls, gen-E | Correct abstention (10/10) |
| Fresh-seed confirmation, gen-D + gen-F | Passed (10/10 each) |
| Real equity panel (752d × 14, 2021–2023) | Rejected: IC $-0.0285$, MSE ratio $1.043$ |
| Real crypto panel (1094d × 15, 2021–2023) | Rejected: IC $-0.0227$, MSE ratio $1.044$ |
| Supplementary last-200d | Equity $+0.0167$ (p=0.255), crypto $-0.0222$ — no edge |
| Supplementary last-300d | Equity $-0.0024$, crypto $-0.0504$ — no edge |
| Trading evaluation | Not run; correctly blocked by 3A failure |

Statistical note: high shift-null p-values mean no support for positive
prediction. Negative ICs do NOT establish an inverse signal — do not flip
the sign. MSE ratios above $1$ mean worse than forecasting zero.

## 2. Proposed H-XSC: Cross-Sectional Shock Concentration (UNTESTED)

A different mechanism and target: risk state, not directional alpha.
Candidate construction, not an established law.

Standardized returns:

$$z_{i,t} = \frac{r_{i,t}}{\sigma_{i,t} + \epsilon}$$

Each asset's share of absolute cross-sectional shock:

$$q_{i,t} = \frac{|z_{i,t}|}{\epsilon + \displaystyle\sum_{j=1}^{N}|z_{j,t}|}$$

Normalized shock concentration ($0$ = even spread, $1$ = single-asset shock):

$$\boxed{H_t = \frac{\displaystyle\sum_{i=1}^{N} q_{i,t}^{2} - 1/N}{1 - 1/N}}$$

Hypothesis in full:

$$\boxed{H_t \text{ predicts higher future cross-sectional dispersion}}$$

with the future target preregistered (at H-XSC prereg time) as the next
five observations' mean cross-sectional variance:

$$\boxed{D_{t+1:t+5} = \frac{1}{5}\sum_{k=1}^{5} \left( \frac{1}{N}\sum_{i=1}^{N} \left(r_{i,t+k} - \bar{r}_{t+k}\right)^2 \right), \qquad \bar{r}_{t+k} = \frac{1}{N}\sum_{i=1}^{N} r_{i,t+k}}$$

Controls (frozen at prereg): current dispersion $D_{t-4:t}$ and market-wide
shock magnitude $|m_t|$. Primary test: does $H_t$ add information beyond
both (incremental $R^2$ / residual IC with dependence-preserving null)?

## 3. Anti-synthetic-only rule for H-XSC

Formulate + freeze on real-market development dates; confirm on untouched
dates (+ additional assets where feasible). Synthetic data for
implementation checks only — synthetic success cannot qualify for
real-market promotion. Gate: beat historical-dispersion baseline +
zero-incremental-information control + permutation null, or reject.
No window search on confirmation data.

## Disposition

H-LIK: closed as failed real-market hypothesis; synthetic result preserved
in scope. H-XSC: proposal only — needs its own preregistration before any
number is computed. Economic utility would still require a further stage
even if dispersion prediction succeeds.
