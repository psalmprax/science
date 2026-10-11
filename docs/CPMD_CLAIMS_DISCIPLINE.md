# CPMD Claims Discipline: Recovered vs Repaired vs Discovered

**Purpose:** exact language for what CPMD has and has not established.
**Status as of v0.14** (supersedes any "outstanding" labels on independent
eval / correlated noise / transfer — those ran in v0.14; sealed eval and
real-market promotion remain outstanding and unauthorized).

## The two equations, in full

The fitted effective relationship (what OLS-T estimates over lags 4–12):

$$\boxed{\hat{y}(\tau) = e^{\hat{\beta}_0} \, \tau^{-\hat{\alpha}_{\mathrm{FW}}}}$$

where the finite-window effective exponent is the sign-flipped log-log
OLS slope:

$$\hat{\alpha}_{\mathrm{FW}} = -\hat{\beta}_1 = -\frac{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})(z_i - \bar{z})}{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}, \qquad x_i = \ln \tau_i, \;\; z_i = \ln y_i$$

The original generating equation (whose global parameters resisted
identification since Track B):

$$\boxed{y(\tau) = A \, (\tau + \tau_0)^{-\alpha}}$$

with the pointwise effective slope it implies at lag $\tau$:

$$\boxed{\alpha_{\mathrm{eff}}(\tau) = -\frac{d\log y(\tau)}{d\log \tau} = \alpha\,\frac{\tau}{\tau + \tau_0}}$$

The first equation is a power-law **approximation** valid over the chosen
window. The second contains the global exponent $\alpha$ and offset
$\tau_0$ that corr $+0.983$ confounding keeps unidentified. Fitting the
first has not recovered the parameters of the second.

## What each word means here

**Recovered — yes, with qualifications.**
A finite-window effective slope is estimable by log-log OLS with
Student-t intervals inside the envelope (independent noise, nl ≤ 0.20,
tested windows). This is recovery of a specified statistical
relationship, not of the global exponent. Truth recovered against: the
finite-window projection

$$\alpha_{\mathrm{FW}}^{\star} = -\frac{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})(z_i^{\star} - \bar{z}^{\star})}{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}, \qquad z_i^{\star} = \ln\!\left(0.05\,(\tau_i + 8.0)^{-1.2}\right)$$

which equals $0.5619$ on lags 4–12 — distinct from the pointwise
$\alpha_{\mathrm{eff}}(8) = 0.6000$.

**Repaired — only the inference procedure.**
v0.13 replaced the normal critical value ($1.96$) with the Student-t
value ($t_{0.975,7} = 2.3646$). That repaired an uncertainty-calculation
flaw. It did not repair the shifted-power-law equation and did not solve
the $(\alpha, \tau_0)$ identification problem.

**Discovered — not in the strong sense.**
The power-law family was predefined, not found. The results show it is a
useful effective description over a stated window. No new governing law
or mechanism has been established. Further: adjudicating a predefined
candidate correctly (which APEDP-X did) is not discovering it.

## Canonical log wording (v0.14-updated)

> CPMD v0.13–v0.14 demonstrated a recoverable finite-window effective
> power-law relationship in synthetic data, using log-log OLS with
> Student-t uncertainty intervals within a prespecified low-noise
> operating envelope (independent noise, nl ≤ 0.20; lag transfer holds on
> tested windows). The global shifted-power-law equation remains
> unresolved. Correlated-noise robustness failed (AR(1)/MA cells);
> sealed evaluation and real-market promotion remain unauthorized.

## Rule for future wording

- "Recovered an effective relationship" — permitted now, with envelope attached.
- "Repaired the equation" — reserved for a correction to mathematical structure.
- "Discovered a governing equation" — reserved for identification +
  independent validation + reproducibility, none of which CPMD holds.
