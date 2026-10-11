# CPMD v0.15 Milestone: Bounded, Reproduced, Protected Result

**Verdict: 8/8 CONFIRMED** (results commit `4e97e5d`; prereg `f9baef8`
committed before implementation — provenance chain intact).
**Boundary verified at full precision:** lags6–15/nl=0.20 median width =
$0.999494 \leq 1.0$ — a genuine pass with margin $0.0005$, i.e., no safety
margin. The envelope edge is sharp.

## The validated procedure, in full

Target — finite-window effective slope over fixed lags
$\tau_1, \dots, \tau_n$:

$$\alpha_{\mathrm{FW}}^{\star} = -\frac{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})(z_i^{\star} - \bar{z}^{\star})}{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}, \qquad x_i = \ln \tau_i, \;\; z_i^{\star} = \ln y^{\star}(\tau_i)$$

where $y^{\star}$ is the noiseless generating curve. Base window lags
4–12 gives $\alpha_{\mathrm{FW}}^{\star} = 0.5619$ for
$y^{\star}(\tau) = 0.05\,(\tau + 8.0)^{-1.2}$.

Point estimator (log-log OLS on measurements $y_i$):

$$\hat{\alpha}_{\mathrm{FW}} = -\frac{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})(z_i - \bar{z})}{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}, \qquad z_i = \ln y_i$$

Uncertainty (Student-t, $df = n - 2$):

$$\boxed{CI_{95\%} = \hat{\alpha}_{\mathrm{FW}} \pm t_{0.975,n-2}\,\frac{s}{\sqrt{S_{xx}}}, \qquad s = \sqrt{\frac{\sum_{i=1}^{n}(z_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2}{\,n-2\,}}, \qquad S_{xx} = \sum_{i=1}^{n}(x_i - \bar{x})^2}$$

with $t_{0.975,7} = 2.3646$ (lags 4–12), $t_{0.975,5} = 2.5706$
(lags 3–10), $t_{0.975,8} = 2.3060$ (lags 6–15).

## The 8/8 confirmation record

| Cell | Coverage (/200) | Median width | Bias | RMSE |
|---|---|---|---|---|
| base 4–12, nl=0.05 | 195 | 0.22 | +0.0004 | 0.043 |
| base 4–12, nl=0.10 | 189 | 0.44 | +0.0171 | 0.086 |
| base 4–12, nl=0.15 | 191 | 0.65 | −0.0014 | 0.139 |
| base 4–12, nl=0.20 | 188 | 0.83 | −0.0153 | 0.178 |
| lags 3–10, nl=0.10 | 191 | 0.44 | −0.0020 | 0.086 |
| lags 3–10, nl=0.20 | 195 | 0.86 | +0.0047 | 0.172 |
| lags 6–15, nl=0.10 | 191 | 0.49 | +0.0051 | 0.110 |
| lags 6–15, nl=0.20 | 192 | 0.999494 | +0.0003 | 0.218 |

Gates per cell: coverage $\geq 184/200$, median width $\leq 1.0$,
$\lvert\mathrm{bias}\rvert \leq 0.1$, RMSE $\leq 0.3$. All pass;
disjoint seed block per cell (9000s–10400s); zero estimator changes.

## Canonical claim (protected)

> CPMD demonstrated recovery of a finite-window effective power-law
> slope using log-log OLS with Student-t confidence intervals, under
> the tested independent-noise conditions up to noise level 0.20, with
> successful independent-seed replication and transfer across the
> specified lag windows (lags 3–10, 4–12, 6–15).

## Explicit non-claims (protection boundary)

- Global parameters of $y(\tau) = A\,(\tau + \tau_0)^{-\alpha}$: unresolved.
- Correlated errors (AR(1)/MA): outside the envelope (v0.14: 2/8).
- Noise above 0.20, untested windows/regimes: abstain.
- No newly discovered governing equation; no sealed evaluation; no market.
- Option B (correlated-noise procedure) is a separate track: its failure
  cannot erase this result, its success cannot expand it without evidence.
