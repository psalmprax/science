# CPMD-HLIK Milestone: First Candidate to Survive Discovery + Confirmation

**Status:** H-LIK earned entry to Stage 3 economic evaluation. NOT promoted.
**Chain:** prereg `d8e7101` → implementation → results `17b40fe`.
**Precision note:** displayed `p=0.000` means 0/200 shift-nulls exceeded the
observation (null resolution 1/200). Recorded below as $p < 0.005$ one-sided.

## The equations, in full

Standardized asset return ($N = 10$ assets):

$$z_{i,t} = \frac{r_{i,t}}{\sigma_{i,t} + \epsilon}, \qquad \epsilon = 10^{-8}$$

with $\sigma_{i,t}$ the trailing-20 standard deviation.

Leave-one-out common-market shock (excludes predicted asset $i$):

$$\boxed{u_{i,t} = \operatorname{median}_{j \ne i}(z_{j,t})}$$

Asset-specific lag-response kernel: ridge regression of next-step own
standardized return on the distributed shock history, window $W = 200$,
max lag $L = 6$, regularization $\lambda = 1.0$, past data only:

$$\boxed{\hat{\boldsymbol{\beta}}_{i,t} = \arg\min_{\boldsymbol{\beta}} \left( \sum_{s=t-W}^{t-1} \left( z_{i,s+1} - \sum_{\ell=0}^{L-1} \beta_{\ell}\, u_{i,s-\ell} \right)^2 + \lambda \sum_{\ell=0}^{L-1} \beta_{\ell}^{2} \right)}$$

Forecast of next return:

$$\boxed{\hat{r}_{i,t+1} = \sigma_{i,t} \sum_{\ell=0}^{L-1} \hat{\beta}_{i,\ell,t}\; u_{i,t-\ell}}$$

The kernel allows different lags different strengths and signs — unlike
H-DAR's scalar discrepancy:

$$D_{i,t} = \frac{\displaystyle\sum_{\ell=0}^{L-1} \lambda^{\ell} e_{i,t-\ell}}{\displaystyle\sum_{\ell=0}^{L-1} \lambda^{\ell}}, \qquad \hat{r}_{i,t+1}^{\mathrm{(H\text{-}DAR)}} = \sigma_{i,t}\, G_t\, \tanh(\gamma D_{i,t})$$

## The survival record (all p-values one-sided shift-null, 200 shifts)

| Stage | gen-D (delay) | gen-E (instant control) | gen-F (new pattern) |
|---|---|---|---|
| Discovery | 10/10, all $p < 0.005$ | 10/10 abstain (min $p = 0.090$) | — |
| Confirmation (fresh seeds) | 10/10, all $p \leq 0.005$ | 10/10 abstain (all $\geq 0.345$) | 10/10, all $p \leq 0.005$ (gate: ≥8/10) |

Contrast H-DAR on identical gates: gen-D 4/10 (rejected), gen-E clean
abstention. Same strictness; representational difference did the work.

## Canonical claim (bounded)

> H-LIK's distributed lag-response kernel detected planted delayed-response
> structure (two delay patterns) with panel-shift-null significance on every
> tested panel, while abstaining on every instant-adjustment control panel.
> Economic validity, real-market predictability, and profitability are
> untested. Stage 3 requires a new preregistration.

## Non-claims

- No trading result: costs, sizing, turnover, drawdown unevaluated.
- No mechanism proof: predictive recovery ≠ causal identification (v0.19–v0.21).
- No real-market inference of any kind.
- The five prior rejections stand; H-LIK faces identical standards next.
