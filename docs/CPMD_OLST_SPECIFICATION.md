# CPMD OLS-T Specification (v0.13, frozen by v0.14)

**Status:** reference method. Validated only inside the operating envelope
(noise ≤ 0.20, independent errors, tested lag windows). Not a global claim.
**Code:** `apedp_dyn/sandbox/cpmd_effective_slope.py` (`slope_ols`),
`apedp_dyn/sandbox/cpmd_envelope.py` (`ols_parts`, `T_CRIT = 2.3646`).
**Critical value verified:** `scipy.stats.t.ppf(0.975, 7) = 2.3646`.

OLS-T is ordinary least-squares regression on log-transformed measurements
with a Student-t confidence interval. It estimates the **finite-window
effective slope**, not the global exponent of the shifted power law.

## 1. The regression equation

Given positive measurements $y_i$ at positive lag values $\tau_i$,
transform the data:

$$x_i = \ln(\tau_i), \qquad z_i = \ln(y_i)$$

Fit the straight-line model:

$$\boxed{z_i = \beta_0 + \beta_1 x_i + \varepsilon_i}$$

Equivalently, in the original variables:

$$\boxed{\ln(y_i) = \beta_0 + \beta_1 \ln(\tau_i) + \varepsilon_i}$$

Here $\beta_0$ is the intercept, $\beta_1$ is the fitted log-log slope,
and $\varepsilon_i$ is the residual of observation $i$.

## 2. The effective-slope point estimate

The OLS slope estimate is:

$$\hat{\beta}_1 = \frac{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})(z_i - \bar{z})}{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

The OLS intercept estimate is:

$$\hat{\beta}_0 = \bar{z} - \hat{\beta}_1 \bar{x}$$

Since the target is a positive decay exponent, the finite-window slope
estimate is the sign-flipped slope:

$$\boxed{\hat{\alpha}_{\mathrm{FW}} = -\hat{\beta}_1}$$

The corresponding fitted curve is an ordinary power law over the window
— it is not the original shifted-power-law model:

$$\boxed{\hat{y}(\tau) = e^{\hat{\beta}_0} \, \tau^{-\hat{\alpha}_{\mathrm{FW}}}}$$

## 3. The Student-t confidence interval

Residual standard error:

$$s = \sqrt{\frac{\displaystyle\sum_{i=1}^{n}\left(z_i - \hat{\beta}_0 - \hat{\beta}_1 x_i\right)^2}{n - 2}}$$

Sum of squares of the regressor:

$$S_{xx} = \sum_{i=1}^{n}(x_i - \bar{x})^2$$

Standard error of the slope (sign flip does not change it):

$$SE(\hat{\alpha}_{\mathrm{FW}}) = \frac{s}{\sqrt{S_{xx}}}$$

Base configuration: $n = 9$ observations (lags 4–12), two fitted
coefficients, hence $n - 2 = 7$ residual degrees of freedom. The 95%
Student-t interval is:

$$\boxed{\hat{\alpha}_{\mathrm{FW}} \pm t_{0.975,7} \cdot \frac{s}{\sqrt{S_{xx}}}}$$

with the verified critical value:

$$t_{0.975,7} = 2.3646$$

In full:

$$\boxed{CI_{95\%} = \left(\hat{\alpha}_{\mathrm{FW}} - 2.3646\,\frac{s}{\sqrt{S_{xx}}}, \quad \hat{\alpha}_{\mathrm{FW}} + 2.3646\,\frac{s}{\sqrt{S_{xx}}}\right)}$$

Interval width (gated quantity):

$$W = 2 \times 2.3646 \times \frac{s}{\sqrt{S_{xx}}}$$

This is the v0.13 correction: the critical value is $2.3646$, not the
normal $1.96$. Using $1.96$ with estimated variance at $df = 7$ (the
retired OLS-N control) understates uncertainty by a factor of
$1.96 / 2.3646 \approx 0.83$.

Transfer configurations use their own critical values with the same
formulas: lags 3–10 ($n = 8$, $df = 5$, $t_{0.975,5} = 2.5706$),
lags 6–15 ($n = 10$, $df = 8$, $t_{0.975,8} = 2.3060$).

## 4. The complete decision rule

| Component | Rule |
|---|---|
| Target | Finite-window log-log slope over a fixed lag window |
| Point estimate | $\hat{\alpha}_{\mathrm{FW}} = -\hat{\beta}_1$ |
| Uncertainty | 95% Student-t interval, $df = n - 2$ |
| Coverage gate | $\geq 184/200$ per condition |
| Precision gate | Median interval width $W \leq 1.0$ |
| Point gates | $\lvert\mathrm{bias}\rvert \leq 0.1$, RMSE $\leq 0.3$ |
| Operating envelope | PASS at noise $\leq 0.20$ (independent errors, tested designs) |
| Beyond envelope | Abstain under the frozen rule |
| Correlated errors | Envelope does not cover AR(1)/MA dependence (v0.14: 2/8) |

Limitations: Student-t intervals carry classical guarantees only under the
regression assumptions (independent homoskedastic Gaussian errors, correct
mean structure). The noise boundary is established for the tested
simulation grid, not as an observable real-world threshold. Sealed
evaluation unopened; real-market promotion not authorized.

In one sentence: OLS-T estimates the negative slope of $\ln(y)$ against
$\ln(\tau)$, quantifies its uncertainty with a Student-t critical value at
$n - 2$ degrees of freedom, and returns a usable result only when the
preregistered accuracy, coverage, precision, and operating-domain gates
are all satisfied.
