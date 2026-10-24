# H-LIK Novelty Ledger: Established Tools vs Proposed Hypothesis

**Rule:** familiar mathematics can still carry a falsifiable market claim;
a new formula alone is not a discovery. Novelty of construction is
unestablished; the hypothesis must earn its keep empirically.

## Established components (not our contribution)

- **Distributed-lag regression:** past values of explanatory variables
  predicting a future value — standard econometrics.
- **Ridge regression:** regularized coefficient estimation — standard statistics.
- **Cross-asset lead–lag / slow response to market information:** documented
  research area (e.g., NBER literature on gradual information diffusion).
- **Leave-one-out aggregates, rolling estimation, return forecasting:**
  standard empirical-finance machinery.

## Proposed combination (ours, novelty vs literature unestablished)

The specific H-LIK formula: leave-one-asset-out median market shock +
distributed ridge-estimated lag kernel (frozen $W=200$, $L=6$, $\lambda=1.0$)
+ rolling re-estimation + next-return forecast:

$$\hat{r}_{i,t+1} = \sigma_{i,t} \sum_{\ell=0}^{L-1} \hat{\beta}_{i,\ell,t}\; u_{i,t-\ell}, \qquad u_{i,t} = \operatorname{median}_{j \ne i}(z_{j,t})$$

No claim is made that this exact formulation is novel. The empirical
proposition — that this representation recovers delayed responses and
generalizes across delay patterns — is what passed synthetic testing
(gen-D 10/10, gen-F 10/10, gen-E abstains) and what Stage 3 must evaluate
economically.

## What H-LIK contributed

Evidence that distributed shock history captures planted delay structure
that H-DAR's scalar discrepancy missed under identical gates. Limits:
construction novelty unproven; trading validity untested.

## Recommendation (preserved)

Keep H-LIK a surviving synthetic hypothesis; proceed to preregistered
economic evaluation; never describe it as a novel financial law. The
scientific contribution to seek is an untested falsifiable market
relationship — not a new arrangement of familiar tools.
