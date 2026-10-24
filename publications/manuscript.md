# A Falsification-Oriented Adaptive Framework for Log-Volume Volatility Forecasting

## Abstract

Accurate volatility forecasting in high-frequency financial markets is difficult because volatility is persistent, nonstationary, heterogeneous across assets, and subject to rapid changes in information arrival and market conditions. This paper examines a compact volume-derived predictor,

$$
\boxed{
H_{A,t}
=
\ln\!\left(
\operatorname{RollMean}_{N}(V_t)
\right)
=
\ln\!\left(
\frac{1}{N}
\sum_{i=0}^{N-1}V_{t-i}
\right)
}
$$

where \(V_t\) denotes strictly backward-looking traded volume and \(N\) is a fixed rolling window. The feature is evaluated not as a standalone trading rule, but as an incremental component of an adaptive continuous-volatility forecasting model.

The central methodological contribution is a falsification-oriented evaluation design. Instead of treating a positive out-of-sample improvement as sufficient evidence, the candidate is subjected to a sequence of prospective tests addressing alternative explanations including functional-form dependence, redundancy with baseline predictors, numerical instability, leverage contamination, cross-asset instability, and post-hoc regime explanations.

Under the locked adaptive walk-forward protocol, the addition of \(H_A\) increased the reported out-of-sample forecasting score by \(0.031\), \(0.036\), and \(0.022\) for BTC, SOL, and ETH, respectively. The corresponding baseline and augmented scores were 0.259 versus 0.290 for BTC, 0.264 versus 0.300 for SOL, and 0.240 versus 0.263 for ETH. In the block-permutation procedure, zero of 200 permutations exceeded the observed improvement for each asset. Thus, under an add-one empirical permutation convention, the exceedance probability is approximately \(1/201\approx0.005\).

A transformation battery comprising sine, cosine, tangent, square-root, absolute-value, square, exponential, and hyperbolic-tangent transformations did not improve the linear specification. The final bounded transformation produced changes of \(-0.009\), approximately \(0.000\), and \(-0.003\) for BTC, SOL, and ETH, respectively. Ex-ante orthogonalization of \(H_A\) against the baseline predictor space produced the same fitted out-of-sample improvements to four decimal places, consistent with the invariance of the jointly refitted linear model under a change of basis.

A separate diagnostic program rejected pole-bearing ratio alternatives. These alternatives produced values approaching \(\pm 10^9\), caused an initial singular-value-decomposition failure, and provided no incremental forecasting value after stabilization. By contrast, the actual \(H_A\) feature is pole-free because it is the logarithm of a positive rolling volume measure. Its Cook's-distance profile showed a maximum value of 0.029 and only approximately 0.1% of observations above the project's 1% diagnostic threshold, providing no indication that the observed contribution is driven by a small number of high-leverage observations.

Cross-asset transportability was heterogeneous and only partially replicable. BTC and ETH retained positive source-level transport in a prospective period, while BNB lost its earlier transportability classification and SOL and AVAX became positive. A subsequent moderator program failed to identify a stable mechanism: the initially suggestive beta relationship was prospectively falsified; synchronization was untestable because the available universe lacked sufficient variation; stress did not generate positive transfer; and trend remained negative in all three regimes. The moderator search was formally terminated.

The evidence therefore supports a deliberately narrow conclusion. The linear log-rolling-volume structure provides reproducible incremental information for continuous volatility forecasting within the tested adaptive walk-forward scope. Its usefulness does not depend on a bounded nonlinear transformation, obvious pole pathology, or a small number of high-leverage observations. However, the evidence does not establish universal cross-asset transportability, a general transportability mechanism, return-direction forecasting, or profitable trading performance.

**Keywords:** realized volatility; volatility forecasting; trading volume; cryptocurrency markets; adaptive estimation; walk-forward validation; QLIKE; model falsification; cross-asset transportability; computational finance

---

# 1. Introduction

Volatility forecasting is a central problem in financial econometrics, with applications in risk management, derivatives pricing, portfolio allocation, market surveillance, and capital allocation. High-frequency observations permit increasingly precise measurement of realized volatility, but forecasting remains difficult because volatility exhibits persistence, long memory, structural change, and substantial cross-asset heterogeneity. The use of intraday data for volatility measurement and forecasting has become a major area of empirical finance, with realized-volatility methods providing a foundation for modern forecasting systems.

The heterogeneous autoregressive model of realized volatility (HAR-RV) established a particularly influential parsimonious approach in which realized volatility is represented through components operating over multiple horizons. Corsi's formulation demonstrated that simple multi-horizon structures can achieve strong forecasting performance while remaining computationally tractable.

Trading volume represents a natural additional information source because it captures aspects of market participation and information arrival that are not fully represented by return history alone. Prior empirical work has explicitly investigated the role of volume in volatility forecasting, including evidence that incorporating trading volume can improve volatility forecasts in ARCH-family models. More recent work has continued to investigate the volume-volatility relationship and the extent to which different decompositions of trading activity improve realized-volatility forecasts.

The literature therefore establishes that volume can contain useful volatility information. That observation, however, does not by itself establish that a particular volume representation contains incremental information beyond an existing forecasting system, nor that an apparent improvement will survive prospective evaluation.

This paper studies a particularly compact representation:

$$
H_{A,t}
=
\ln
\left(
\frac{1}{N}
\sum_{i=0}^{N-1}V_{t-i}
\right).
$$

The feature contains two operations only: backward-looking averaging and a logarithmic transform. It therefore has considerably less structural complexity than many machine-learning forecasting systems. This is relevant because recent cryptocurrency volatility research increasingly employs nonlinear and deep-learning models, including recurrent neural architectures designed to capture complex high-frequency volatility dynamics.

The contribution of the present study is not to claim that a simple log-volume feature is universally superior to sophisticated models. Instead, the contribution lies in a deliberately adversarial validation framework.

A candidate feature is not considered credible merely because it improves an in-sample fit or one favorable out-of-sample period. It must survive attempts to demonstrate that its apparent value originates from:

1. an advantageous nonlinear transformation;
2. redundancy with existing baseline predictors;
3. numerical instability;
4. extreme observations or leverage;
5. source-selection artifacts;
6. regime-specific explanations invented after observing the data.

This produces a research distinction between predictive evidence and explanatory evidence. The former can survive while the latter remains unresolved.

The paper therefore addresses the following questions.

### Research Question 1

Does \(H_A\) provide incremental out-of-sample information for continuous volatility forecasting beyond the locked baseline predictor set?

### Research Question 2

Is the forecasting contribution robust to plausible functional transformations?

### Research Question 3

Does ex-ante orthogonalization alter the incremental contribution, thereby revealing whether the candidate occupies a genuinely distinct component of the fitted predictor space?

### Research Question 4

Can the observed effect be explained by numerical pathology or extreme-leverage observations?

### Research Question 5

Does the predictive contribution transport consistently across assets?

### Research Question 6

Can a stable market-state moderator explain when transportability occurs?

The study's principal result is positive but deliberately constrained: \(H_A\) provides incremental predictive information within the tested adaptive walk-forward scope. Its cross-asset mechanism remains unidentified.

---

# 2. Relation to Existing Literature

## 2.1 Realized volatility and high-frequency forecasting

The use of high-frequency observations to construct realized volatility established a framework in which intraday variation can be aggregated into a more informative volatility measure than low-frequency squared returns alone. Andersen et al. demonstrate the practical forecasting value of logarithmic realized-volatility dynamics using high-frequency exchange-rate data.

Corsi subsequently proposed HAR-RV as a parsimonious multi-horizon representation capable of reproducing important empirical properties of volatility and delivering useful forecasts.

The present study is consistent with this literature in treating volatility forecasting as a time-series prediction problem rather than as a return-direction classification problem.

## 2.2 Trading volume as a forecasting variable

Existing work directly supports investigating trading volume as an auxiliary volatility predictor. Anatolyev and Ponomarev (2023) incorporate log-transformed average trading volumes over 1-, 5-, and 22-day windows into HAR-RV-type models and report out-of-sample forecasting gains on 41 equities at the daily horizon, with effects attenuating at weekly and monthly horizons. The literature has also considered the inclusion of volume in ARCH-family models with reported forecasting gains in some settings.

Other research has explored decomposition of trading activity into components with different information-flow characteristics and investigated their contribution to realized-volatility forecasts.

The present study differs in three respects. First, it focuses on a very compact functional representation of rolling volume, \(\ln(\operatorname{RollMean}(V))\), at intraday (15-minute) resolution on cryptocurrency markets. Second, it treats the feature as a candidate subject to falsification rather than as a variable whose inclusion is justified solely through conventional coefficient significance. Third, it subjects the candidate to permutation, replication, horizon, window-ablation, transformation, orthogonalization, pole-diagnostic, and cross-market transfer tests within one pre-registered battery.

## 2.3 Forecast evaluation

Forecast evaluation is particularly important when volatility itself is measured imperfectly. Patton shows that some loss functions have desirable robustness properties when volatility forecasts are evaluated against noisy volatility proxies and identifies QLIKE as belonging to an important class of robust loss functions.

The present study therefore emphasizes out-of-sample evaluation and retains the project's locked forecasting metric rather than selecting a metric after observing the result.

## 2.4 Cryptocurrency volatility forecasting

Cryptocurrency markets provide a demanding forecasting environment characterized by high sampling frequency and substantial volatility shifts. Recent studies have employed increasingly flexible machine-learning and probabilistic architectures for high-frequency cryptocurrency volatility forecasting. No machine-learning benchmark model was implemented in the present replication package; direct comparison against deep-learning volatility architectures is proposed as future work rather than claimed here.

The purpose of the present study is not to compete with every possible deep-learning architecture. It instead asks whether an extremely compact predictor can deliver reproducible incremental information under an adaptive evaluation framework.

The novelty therefore resides in the combination of compact structural feature plus adaptive estimation plus falsification-oriented validation.

---

# 3. Data

## 3.1 Market universe

The study uses 15-minute cryptocurrency spot observations from the Binance public market-data API (no authentication required): BTCUSDT, SOLUSDT, and ETHUSDT as primary assets, with BNB, XRP, DOGE, ADA, AVAX, LINK, NEAR, ATOM, UNI, LTC, and PAXG (PAX Gold) used for transportability analysis. Sample: 2024-01-01 through 2026-10-03/04, UTC bar timestamps, approximately 96,600 bars per asset (2024 training: ~35,100 bars; 2025+ evaluation: ~61,500 bars). Data accessed 2026-10-03/04. Equity index data (S&P 500, Nasdaq-100, hourly, Yahoo Finance) and single-name equities (AAPL, MSFT, NVDA) were used only for specified transfer experiments.

## 3.2 Information-set discipline

All candidate variables are constructed using only information available at the forecasting origin. For observation \(t\), the feature satisfies \(H_{A,t} = g(V_t,V_{t-1},\ldots,V_{t-N+1})\) and contains no future observations. Any rolling normalization, standardization, estimation, or transformation used elsewhere in the pipeline follows the same backward-looking information-set restriction.

## 3.3 Volume definition

\(V_t\) denotes per-bar quote volume in USDT (Binance kline field 7). All rolling means, logarithms, imbalance fractions, and volume z-scores in the study derive from this quote-volume series. No exchange-aggregated or index-level volume is used.

---

# 4. Candidate Structure

The candidate is defined as

$$
\boxed{
H_{A,t}
=
\ln\!\left(
\operatorname{RollMean}_{N}(V_t)
\right)
}
$$

or equivalently,

$$
\boxed{
H_{A,t}
=
\ln\!\left(
\frac{1}{N}
\sum_{i=0}^{N-1}V_{t-i}
\right),
}
$$

with the fixed window \(N=40\) bars (10 hours at 15-minute resolution) as the canonical crypto instantiation. Window-ablation variants \(N \in \{20, 80, 120, 150, 200\}\) (and finer grids for equities) are treated as separate frozen specifications, never tuned on evaluation data.

The candidate has no denominator involving price distance or volatility. This point becomes important later because several alternative ratio formulations were investigated and found to contain pole-related numerical pathologies. Those pathologies do not describe the actual \(H_A\) structure.

---

# 5. Forecasting Model

Let \(X_{0,t}\) denote the locked baseline predictor vector: trailing-12-bar realized volatility, contemporaneous absolute return \(|r_t|\), and Amihud illiquidity \(|r_t|/V_t\).

The target is \(Y_{t} = \operatorname{std}(r_{t+1},\ldots,r_{t+16})\), the realized volatility over the next 16 fifteen-minute bars (4-hour horizon).

The baseline forecast is \(\widehat{Y}^{(0)} = \beta_0^{(0)} + \boldsymbol{\beta}^{(0)\top}X_{0,t}\). The augmented model is

$$
\boxed{
\widehat{Y}_{t+h}
=
\beta_0
+
\boldsymbol{\beta}^{\top}X_{0,t}
+
\gamma
\ln\!\left(
\frac{1}{N}
\sum_{i=0}^{N-1}V_{t-i}
\right).
}
$$

Coefficients are estimated by ordinary least squares: frozen fit on calendar-2024 data for the primary OOS evaluation (2025+), and refit on rolling 90-day windows with 30-day test blocks for the adaptive walk-forward evaluation.

---

# 6. Adaptive Walk-Forward Evaluation

The model is estimated recursively under the locked walk-forward protocol: training on rolling 90-day windows, forecasting the following 30 days, stepping through 2025+. At forecast origin \(t\), the estimation sample contains only observations with \(s\leq t\), so forecasts are functions only of the information set \(\mathcal I_t\).

The important methodological distinction is between an adaptive estimator and a permanently frozen coefficient vector. The study evaluates whether the candidate survives changing parameter estimates across time rather than assuming stationarity of its coefficients over the entire historical sample.

---

# 7. Evaluation Metrics and Promotion Gates

The primary forecasting evaluation uses out-of-sample \(R^2\) increment over the locked baseline, with QLIKE as the proper scoring rule for the walk-forward horse race:

$$
QLIKE_t
=
\frac{\sigma_t^2}{\widehat{\sigma}_t^2}
-
\log
\left(
\frac{\sigma_t^2}{\widehat{\sigma}_t^2}
\right)
-1,
$$

with lower loss indicating better forecasting performance. Variance forecasts below 5% of the in-sample median are floored at that level before QLIKE evaluation (documentedwinsorization necessitated by near-zero squared OLS forecasts).

The project's promotion framework separately records positive incremental fit, the associated out-of-sample score, and the block-permutation result. The minimum reporting structure is: \(\Delta R^2>0\), \(R^2>0\), and \(p_{\text{block}}<0.05\) (200 block permutations of 96 bars, add-one convention \(p=(b+1)/201\)). These gates are displayed independently rather than collapsed into one statistic.

---

# 8. Primary Results

The continuous-volatility experiment produced the following results.

| Asset | Baseline \(V_0\) | \(V_0+H_A\) |  Increment |
| ----- | ---------------: | ----------: | ---------: |
| BTC   |            0.259 |       0.290 | **+0.031** |
| SOL   |            0.264 |       0.300 | **+0.036** |
| ETH   |            0.240 |       0.263 | **+0.022** |

Thus the incremental contributions are positive in all three primary assets. In the block-permutation procedure, zero of 200 permutations exceeded the observed improvement for each asset (empirical \(p\le1/301\approx0.0033\) under the stated design; the manuscript's \(1/201\) convention is retained as an approximation).

The primary finding is therefore that \(H_A\) provides positive incremental out-of-sample forecasting information in all three primary assets.

---

# 9. Functional-Form Robustness

A candidate can appear predictive because a particular nonlinear transformation happens to fit the sample. The transformation battery evaluated sine, cosine, tangent, square-root, absolute-value, square, exponential, and hyperbolic-tangent transformations of standardized \(H_A\). None improved upon the linear representation in the locked prospective tests. Orthogonalized re-evaluation (transformations residualized against linear \(H_A\)) collapsed all apparent gains to ≤0.001 or negative, confirming the periodic positives were linear-component stowaways.

| Asset | Linear H-A increment | Change after \(\tanh\) |
| ----- | -------------------: | ---------------------: |
| BTC   |               +0.031 |                 −0.009 |
| SOL   |               +0.036 |    approximately 0.000 |
| ETH   |               +0.022 |                 −0.003 |

Saturation therefore does not provide evidence for a superior functional representation. The surviving specification remains linear in \(H_A$. This result is informative because it closes a plausible transformation pathway without increasing model complexity merely for the sake of optimization.

---

# 10. Orthogonalization Test

To determine whether \(H_A\) is simply redundant with the baseline predictors, the candidate was residualized against the baseline predictor space. The results were identical to four decimal places. The mathematical reason is that, for a linear model that is refitted after residualization, the span is invariant under the change of basis. The result is therefore attributional confirmation, not a second forecasting equation.

---

# 11. Numerical Pathology Tests

## 11.1 Pole-bearing alternatives

A separate candidate family containing ratios of price deviation to quantities that can approach zero was examined. Its observed range reached approximately \([-10^9,+10^9]\), with poles on both sides of moving-average crossings. The first conventional OLS attempt produced an SVD non-convergence failure, establishing estimator-level numerical instability.

## 11.2 Stabilized rerun

The alternatives were subsequently stabilized (robust median/IQR scaling) and rerun: V0=0.2370, +f=0.2369, +finv=0.2370, with winsorized differences of +0.000 and −0.002. The pole-bearing candidates therefore did not retain meaningful incremental forecasting value after stabilization. The rank correlation between the reciprocal pair was +0.49 rather than the −1.0 that a positive-domain-only analysis would predict, because the sign break at MA crossings voids the reciprocal symmetry.

---

# 12. Diagnostics of the Actual H-A Feature

The actual candidate does not contain the denominator responsible for the ratio poles. Because rolling volume is strictly positive, the logarithm does not generate a zero-crossing singularity. Influence diagnostics: max Cook's distance 0.029 with ~0.1% of 35,000 observations above the 1% threshold. The observed contribution is therefore not evidently concentrated in a tiny number of high-leverage observations.

---

# 13. Cross-Asset Transportability

Source-level transportability was evaluated using mean incremental directional accuracy and positive-transfer fraction over 13 targets. The initial experiment showed BTC/ETH/BNB transportable and XRP/DOGE/ADA not, with PAXG isolated. The prospective replication retained BTC/ETH (plus SOL/AVAX) while BNB dropped out, indicating transportability is regime-dependent rather than a fixed source trait. Detailed per-asset, per-period transport tables are retained in the project ledger.

---

# 14. Falsification of Candidate Transportability Explanations

Beta (prospective group test: low −0.022, high −0.025, ∂T/∂β not established), synchronization (range [0.60, 0.89], insufficient variation — unresolved, not rejected), stress (all regimes negative: LOW −0.058, NORMAL −0.051, HIGH −0.017), and trend (DOWN −0.035, FLAT −0.042, UP −0.016) all failed to explain transfer. The moderator search was formally terminated per the prespecified stopping rule.

---

# 15. Prespecified Stopping Rule

The moderator program was closed after beta was prospectively falsified, synchronization was untestable, stress failed, and trend failed. No fifth moderator was introduced. This stopping rule is methodologically important because unrestricted moderator search can convert exploratory research into repeated multiple-testing.

---

# 16. Discussion

## 16.1 Principal contribution

The primary contribution is not the claim that the logarithm of rolling volume is a universal law of volatility. The evidence supports a narrower finding: a compact volume predictor providing incremental forecasting information under the tested adaptive walk-forward framework, with credibility deriving from positive prospective performance plus survival of targeted diagnostics.

## 16.2 Why simplicity matters

The surviving structure contains only rolling mean and logarithm. No high-order polynomial, saturation function, periodic basis, reciprocal term, or manually optimized nonlinear interaction is required. The transformation battery provides empirical support for retaining the simpler representation.

## 16.3 Why orthogonalization matters

The orthogonalization test does not create a new predictor. Because the refitted linear predictor space is invariant under the change of basis, the test provides attributional confirmation that the incremental contribution cannot be dismissed as an artifact of an arbitrary correlated coordinate representation.

## 16.4 Numerical robustness

Pole diagnostics establish that unstable denominators, not useful predictive information, drive the apparent behavior of ratio-family alternatives. The actual \(H_A\) feature is structurally pole-free with a clean leverage profile.

## 16.5 Transportability remains the major unresolved question

Some sources travel repeatedly. Some do not. Some change status. No tested moderator explains the pattern. This is not a weakness to hide; it is part of the result. The evidence supports conditional predictive usefulness rather than universal transportability.

---

# 17. Relationship to Directional Classification

The continuous-volatility model should not be conflated with directional-volatility classification (balanced accuracy ~0.70 on persistence alone; H-A adds directionally only on a heterogeneous 5/14 assets) or with return-direction prediction (no validated content). Volatility prediction is not return prediction, and neither converts to trading profitability without costs, execution, and risk treatment.

---

# 18. Economic Interpretation

A statistically useful volatility forecast does not automatically imply an economically profitable trading strategy. Volatility-targeted long/flat translation (daily rebalance, 5bps costs, frozen forecasts) beat buy-and-hold on net return in 2/3 assets during the 2026 downtape but with noise-grade Sharpe ratios (+0.17/−0.07/+0.30) and severe drawdowns (64–141%). Economic value is therefore not established. An economic evaluation should constitute a separate, fully frozen experiment.

---

# 19. Robustness and Reproducibility Requirements

For publication, the complete replication package should contain: the frozen H-A definition (N=40); exact data construction (Binance 15m klines, quote volume, UTC, alignment and missing-data rules as implemented in the ledger scripts); frozen baseline definitions; the forecast target (next-16-bar realized volatility) and horizon; walk-forward schedule (90d-train/30d-test) and information-set rules; statistical gates (ΔR², block-permutation p with 96-bar blocks × 200 draws, R², QLIKE with 5%-median floor); permutation design and seeds (42/0/6/7/9/11 as logged per experiment); and code/data snapshots. This allows a reader to reproduce not only the positive result but also the negative results.

---

# 20. Limitations

The study has several important limitations. The strongest continuous forecasting evidence is concentrated in three primary assets. The mechanism is unknown. Synchronization variation was insufficient. Findings are specific to 15-minute sampling (with 1-hour equity replication at longer memory scales). Uncertainty intervals come from a moving-block bootstrap (96-bar blocks, 1000 resamples): BTC ΔR² +0.031, 95% CI [+0.021, +0.042]; SOL +0.036 [+0.022, +0.052]; ETH +0.022 [+0.013, +0.033]. Benchmark breadth is limited: a GARCH(1,1) horse race was implemented (H-A wins on MSE and QLIKE for BTC/SOL; see project ledger §36) while HAR-RV and machine-learning benchmarks were not implemented and are proposed as future work. Economic value is not established. Although the investigation uses a frozen battery and explicit stopping rules, the number and ordering of exploratory tests must be fully disclosed to distinguish confirmatory from diagnostic evidence; the project ledger (800+ lines, 60+ sections) provides that record.

---

# 21. Required Final Statistical Package Before Submission

Completed in this revision: dependence-aware confidence intervals via moving-block bootstrap (see §20); GARCH-family benchmark comparison (see §36 of the project ledger); Diebold-Mariano-type comparison (BTC +36.7 / SOL +3.5 without HAC correction — directional, not calibrated). Remaining: HAR-RV direct comparison (proposed, not implemented); cross-period replication on a fully fresh sample (data through 2026-10; proposed); locked economic evaluation with costs (translation exercise completed without promotion — see §24/§45 of the ledger).

---

# 22. Final Model Specification

The current frozen feature is \(H_{A,t} = \ln(\frac{1}{N}\sum_{i=0}^{N-1}V_{t-i})\) with \(N=40\), and the generic augmented forecasting model includes baseline controls plus \(\gamma \ln(\frac{1}{N}\sum V)\). The orthogonal attributional representation is \(H_{A,t}^{\perp} = H_{A,t} - \mathcal P_{X_0}(H_{A,t})\). No transformation is currently promoted above the linear representation. No pole-bearing formulation is promoted. No additional moderator is promoted.

---

# 23. Conclusions

This study evaluated a compact log-rolling-volume structure for continuous volatility forecasting under an adaptive walk-forward framework. The structure generated positive incremental forecasting performance for BTC, SOL, and ETH (+0.031, +0.036, +0.022). The result survived a block-permutation evaluation (0/200 exceedances each), a broad functional-form battery, orthogonalization, pole diagnostics, and (partially) cross-asset transfer, while moderator explanations (beta, synchronization, stress, trend) were falsified or left unresolved under frozen prospective designs. The final position is neither universal predictability nor identified mechanism: the linear log-rolling-volume structure provides reproducible incremental information for continuous volatility forecasting within the tested adaptive walk-forward scope, while its cross-asset transportability mechanism remains unidentified.

---

# References

Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and Forecasting Realized Volatility. *Econometrica, 71*(2), 579–625.

Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics, 7*(2), 174–196.

Patton, A. J. (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies. *Journal of Econometrics, 160*(1), 246–256.

Anatolyev, S., & Ponomarev, S. (2023). Does Trading Volume Help Forecast Volatility? Working paper. https://pages.nes.ru/sanatoly/Papers/VolVol.pdf

*No machine-learning benchmark models were implemented in this study; ML comparison is proposed future work. No dedicated cryptocurrency realized-volatility benchmark studies were used as comparators; proposed future work.*
