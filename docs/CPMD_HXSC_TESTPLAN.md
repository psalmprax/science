# H-XSC Test Plan: Cross-Sectional Shock Concentration (UNEXECUTED)

**Status:** next independent candidate. H-LIK untouched (rejected, frozen).
This document is the test protocol — NOT a preregistration and NOT results.
Execution requires a separate prereg commit before any number is computed.

## 1. Frozen hypothesis

A market shock concentrated in few assets precedes a different
cross-sectional risk environment than a broadly spread shock. Direction
frozen: HIGHER concentration forecasts HIGHER subsequent dispersion.
Candidate construction; not a law; novelty unclaimed.

Standardized return ($\sigma_{i,t}^{(20)}$ = trailing-20 std, past data only):

$$z_{i,t} = \frac{r_{i,t}}{\sigma_{i,t}^{(20)} + \epsilon}, \qquad \epsilon = 10^{-8}$$

Asset share of absolute cross-sectional shock:

$$q_{i,t} = \frac{|z_{i,t}|}{\epsilon + \displaystyle\sum_{j=1}^{N}|z_{j,t}|}$$

Normalized shock concentration ($0$ = even, $1$ = single-asset dominance):

$$\boxed{H_t = \frac{\displaystyle\sum_{i=1}^{N} q_{i,t}^{2} - 1/N}{1 - 1/N}}$$

Prediction target — mean cross-sectional variance over next five observations:

$$\boxed{D_{t+1:t+5} = \frac{1}{5}\sum_{k=1}^{5} \left( \frac{1}{N}\sum_{i=1}^{N} \left(r_{i,t+k} - \bar{r}_{t+k}\right)^2 \right)}, \qquad \bar{r}_{t+k} = \frac{1}{N}\sum_{i=1}^{N} r_{i,t+k}$$

Frozen controls (must be beaten incrementally): current dispersion

$$D_{t-4:t} = \frac{1}{5}\sum_{k=0}^{4} \left( \frac{1}{N}\sum_{i=1}^{N} \left(r_{i,t-k} - \bar{r}_{t-k}\right)^2 \right)$$

market-wide shock magnitude $|m_t|$ with $m_t = \operatorname{median}_i(z_{i,t})$,
and recent market volatility (trailing-20 std of median returns).

## 2. Stages (no advantage)

**Stage 1 — predictive discovery.** Freeze equation, 5-obs horizon,
cleaning rules, metric (incremental $R^2$ / residual IC of $H_t$ beyond the
three controls, dependence-preserving null). Pass = significant increment.

**Stage 2 — independent confirmation.** Genuinely untouched period or panel;
preregistered loss comparison vs frozen baseline (historical-dispersion
autoregression); uncertainty accounting for overlapping 5-day targets and
shared shocks (block bootstrap / Newey-West with lag ≥ 4). Pass = out-of-sample win.

**Stage 3 — economic utility, if earned.** Fixed risk-management rule
(e.g., scale exposure by $1/\sqrt{\hat{D}}$) vs appropriate baseline on net
portfolio outcomes. A risk forecast is NOT directional alpha.

## 3. Data safeguard (frozen rule)

The 2021–2023 panels are extensively examined: results on those dates are
EXPLORATORY for H-XSC unless the prereg justifies otherwise. Confirmation
requires untouched dates (new fetch beyond 2023-12) or a preregistered
held-out split declared before any H-XSC number is computed — never chosen
after seeing results.

## 4. Outcome table

| Outcome | Decision |
|---|---|
| No incremental predictive information | `REJECTED` |
| Discovery passes, fresh confirmation fails | `REJECTED` |
| Replicated prediction, no economics yet | Eligible for Stage 3 |
| Competing formulations survive unseparated | `MULTI_HYPOTHESIS` |
| Cannot distinguish improvement from noise | `INSUFFICIENT_EVIDENCE` |

Direction of research: a different market structure (risk state), not
another form of failed return predictor. Same rejection standard as H-LIK.
