# H-LIK Equation Reference (frozen: W=200, L=6, λ=1.0)

**Code:** `apedp_dyn/sandbox/cpmd_hlik.py` (verified below against implementation).
**Scope:** synthetic hypothesis — no real-market or trading claim.

## 1. Forecasting equation (core)

$$\boxed{\hat{r}_{i,t+1} = \sigma_{i,t} \sum_{\ell=0}^{L-1} \hat{\beta}_{i,\ell,t}\; u_{i,t-\ell}}$$

Expanded to show each lag's own coefficient:

$$\hat{r}_{i,t+1} = \sigma_{i,t}\left( \hat{\beta}_{i,0,t}\,u_{i,t} + \hat{\beta}_{i,1,t}\,u_{i,t-1} + \cdots + \hat{\beta}_{i,L-1,t}\,u_{i,t-L+1} \right)$$

Coefficients may be positive, negative, or near zero per lag.

## 2. Components

A. Standardized asset return ($\sigma_{i,t}$ = trailing-20 std, $\epsilon = 10^{-8}$):

$$z_{i,t} = \frac{r_{i,t}}{\sigma_{i,t} + \epsilon}$$

B. Leave-one-out common-market shock (median over the other $N-1$ assets):

$$u_{i,t} = \operatorname{median}_{j \ne i}(z_{j,t})$$

C. Distributed-lag coefficients: ridge regression of next-step own
standardized return on shock history, estimation window $W = 200$,
regularization $\lambda = 1.0$, strictly past data:

$$\boxed{\hat{\boldsymbol{\beta}}_{i,t} = \arg\min_{\boldsymbol{\beta}} \left( \sum_{s=t-W}^{t-1} \left( z_{i,s+1} - \sum_{\ell=0}^{L-1} \beta_{\ell}\, u_{i,s-\ell} \right)^2 + \lambda \sum_{\ell=0}^{L-1} \beta_{\ell}^{2} \right)}$$

Solved via the normal equations $(X'X + \lambda I)\hat{\beta} = X'Y$
with a $200 \times 6$ design matrix per asset per step.

## 3. Distinction from H-DAR

H-LIK estimates a sequence of lag-specific responses; H-DAR compressed
delay into one scalar discrepancy $D_{i,t}$ gated by coherence $G_t$.
Synthetic evidence favors the distributed representation on the tested
generators; real-market prediction and profitability are unestablished.
