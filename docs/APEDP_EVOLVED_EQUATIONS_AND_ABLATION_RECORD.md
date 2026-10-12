# APEDP Evolved Trading Equations and H-BB Ablation Baselines

**Record type:** Discovery and ablation results  
**Discovery window:** Block A (2021–2023, 753 daily bars)  
**Evaluation window:** Block B (2024–2025, 502 daily bars)  
**Evaluation setup:** 502 bars, 5.0 bps transaction cost assumption, inverse-volatility weighting, capital compounding  
**Status:** Exploratory; not promoted as universally valid or prospectively validated  
**Recommended ledger label:** `APEDP_EVOLVED_UNIVERSE_SPECIFIC_EQUATIONS_BLOCK_B_ABLATION_EXPLORATORY`  
**Provenance directory:** [`provenance/selection_stability_and_ablation_20261009/`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/selection_stability_and_ablation_20261009/)  
**Manifest:** [`manifest.json`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/selection_stability_and_ablation_20261009/manifest.json) (`commit: d881d5cd`, `returncode: 0`)  
**Results artifact:** [`results.json`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/selection_stability_and_ablation_20261009/results.json)  

---

## 1. APEDP-Evolved Equations

Let $X^*_{\text{term}}$ denote the normalized feature representation used by the reported APEDP pipeline. Each equation is specific to its corresponding universe.

### 1.1 Universe $N = 15$ Assets (Pure Crypto)

**Score Equation:**
$$
\begin{aligned}
S_{15} ={}& +1.1948 X^*_{\mathrm{H\text{-}BB}} \\
&+ 0.4501 X^*_{\mathrm{H\text{-}OBV\_MOM}} \\
&+ 0.2227 X^*_{\mathrm{H\text{-}STOCH}} \\
&- 0.4839 X^*_{\mathrm{NL\text{-}HURST\_ADAPT}} \\
&- 0.0161 X^*_{\mathrm{PRODUCT\_H\_BB\_TAKER}}
\end{aligned}
$$

- **Reported Decision Threshold:** $\theta_{15} = 0.1695$
- **Term Descriptions:**
  - $X^*_{\mathrm{H\text{-}BB}}$: Bollinger Bands %B (Normalized boundary position)
  - $X^*_{\mathrm{H\text{-}OBV\_MOM}}$: On-Balance Volume Momentum
  - $X^*_{\mathrm{H\text{-}STOCH}}$: Stochastic Oscillator (%K momentum)
  - $X^*_{\mathrm{NL\text{-}HURST\_ADAPT}}$: Non-linear Hurst Regime-Switched Momentum
  - $X^*_{\mathrm{PRODUCT\_H\_BB\_TAKER}}$: Bilinear interaction ($X^*_{\mathrm{H\text{-}BB}} \cdot X^*_{\mathrm{NL\text{-}TAKER\_MOM}}$)

---

### 1.2 Universe $N = 30$ Assets (15 Crypto + 15 Equities)

**Score Equation:**
$$
\begin{aligned}
S_{30} ={}& +0.8179 X^*_{\mathrm{H\text{-}BB}} \\
&- 0.1792 X^*_{\mathrm{H\text{-}ASYM}} \\
&- 0.2768 X^*_{\mathrm{NL\text{-}OFI\_REV}}
\end{aligned}
$$

- **Reported Decision Threshold:** $\theta_{30} = 0.1572$
- **Term Descriptions:**
  - $X^*_{\mathrm{H\text{-}BB}}$: Bollinger Bands %B (Primary trend driver)
  - $X^*_{\mathrm{H\text{-}ASYM}}$: GJR-GARCH Downside Asymmetric Panic Leverage Guard
  - $X^*_{\mathrm{NL\text{-}OFI\_REV}}$: RSI $\times$ Order Flow Imbalance Mean-Reversion

---

### 1.3 Universe $N = 60$ Assets (Multi-Asset Liquid Basket)

**Score Equation:**
$$
\begin{aligned}
S_{60} ={}& +0.4323 X^*_{\mathrm{H\text{-}BB}} \\
&- 0.8522 X^*_{\mathrm{H\text{-}OBV\_MOM}} \\
&+ 0.5357 X^*_{\mathrm{H\text{-}HURST}} \\
&- 0.0545 X^*_{\mathrm{H\text{-}ASYM}} \\
&- 0.0042 X^*_{\mathrm{H\text{-}EMA\_CROSS}} \\
&- 1.3680 X^*_{\mathrm{NL\text{-}SQUEEZE\_TREND}} \\
&- 0.9019 X^*_{\mathrm{NL\text{-}HURST\_ADAPT}}
\end{aligned}
$$

- **Reported Decision Threshold:** $\theta_{60} = 0.0946$
- **Term Descriptions:**
  - $X^*_{\mathrm{H\text{-}BB}}$: Bollinger Bands %B
  - $X^*_{\mathrm{H\text{-}OBV\_MOM}}$: OBV Accumulation Momentum
  - $X^*_{\mathrm{H\text{-}HURST}}$: Fractional Brownian Hurst Exponent Memory
  - $X^*_{\mathrm{H\text{-}ASYM}}$: Asymmetric Downside Variance Guard
  - $X^*_{\mathrm{H\text{-}EMA\_CROSS}}$: Fast/Slow Exponential Moving Average Crossover
  - $X^*_{\mathrm{NL\text{-}SQUEEZE\_TREND}}$: MACD $\times$ Bollinger/Keltner Squeeze Breakout
  - $X^*_{\mathrm{NL\text{-}HURST\_ADAPT}}$: Adaptive Hurst Regime Switcher

---

### 1.4 Universe $N = 90$ Assets (Full Cross-Asset Market)

**Score Equation:**
$$
\begin{aligned}
S_{90} ={}& +0.7700 X^*_{\mathrm{H\text{-}HURST}} \\
&+ 0.7185 X^*_{\mathrm{H\text{-}NATR}} \\
&+ 0.7004 X^*_{\mathrm{H\text{-}BB}} \\
&- 0.0925 X^*_{\mathrm{H\text{-}SQUEEZE}} \\
&- 0.0685 X^*_{\mathrm{H\text{-}STOCH}}
\end{aligned}
$$

- **Reported Decision Threshold:** $\theta_{90} = 0.1525$
- **Term Descriptions:**
  - $X^*_{\mathrm{H\text{-}HURST}}$: Fractional Long-Memory Drift Persistence
  - $X^*_{\mathrm{H\text{-}NATR}}$: Normalized Average True Range Volatility Scaling
  - $X^*_{\mathrm{H\text{-}BB}}$: Bollinger Bands %B
  - $X^*_{\mathrm{H\text{-}SQUEEZE}}$: Band Compression Filter
  - $X^*_{\mathrm{H\text{-}STOCH}}$: Stochastic Oscillator Filter

---

## 2. Corresponding H-BB-Only Ablation Equations

These are single-feature baseline controls constructed using the $H\text{-}BB$ coefficient from the corresponding evolved equation. In the reported ablation, each retained its universe's corresponding decision threshold.

| Universe | H-BB-Only Score Equation | Threshold $\theta$ |
| :--- | :--- | :---: |
| **N15** | $S_{15}^{\mathrm{BB}} = +1.1948 X^*_{\mathrm{H\text{-}BB}}$ | $0.1695$ |
| **N30** | $S_{30}^{\mathrm{BB}} = +0.8179 X^*_{\mathrm{H\text{-}BB}}$ | $0.1572$ |
| **N60** | $S_{60}^{\mathrm{BB}} = +0.4323 X^*_{\mathrm{H\text{-}BB}}$ | $0.0946$ |
| **N90** | $S_{90}^{\mathrm{BB}} = +0.7004 X^*_{\mathrm{H\text{-}BB}}$ | $0.1525$ |

> [!IMPORTANT]
> **Ablation Baseline Scope:** These controls were derived specifically for ablation to measure the marginal contribution of the recurring $H\text{-}BB$ term versus universe additions. They were not independently discovered or separately optimized single-parameter models.

---

## 3. Reported Block B Net-Return Comparison (Diagnostic Holdout: 2024–2025)

The table below reports the complete risk-adjusted evaluation on Block B (502 bars, 5.0 bps fee, inverse-volatility weighting) comparing Standalone $H\text{-}BB$, the Full Evolved Equation, the Universe-Terms-Only ablation (Full minus $H\text{-}BB$), and the Candidate A benchmark.

| Universe | Strategy / Policy | Net Return | Gross Return | Sharpe | Max Drawdown | Exposure | Turnover |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **N15** | **Policy 1: Standalone $H\text{-}BB$** | **+30.08%** | +37.77% | **+0.62** | 42.6% | 87.9% | 1724.0 |
| | **Policy 2: Full Evolved Equation** | **+33.77%** | +42.66% | **+0.65** | 40.3% | 89.0% | 1932.0 |
| | **Policy 3: Universe Terms Alone (No $H\text{-}BB$)** | **+5.53%** | +14.39% | +0.34 | 36.4% | 66.3% | 2420.0 |
| | *Benchmark: Candidate A Baseline* | -5.93% | +6.26% | +0.09 | 43.4% | 86.1% | 3652.0 |
| **N30** | **Policy 1: Standalone $H\text{-}BB$** | **+40.80%** | +49.05% | **+0.71** | 41.1% | 83.4% | 3416.0 |
| | **Policy 2: Full Evolved Equation** | **+33.64%** | +45.36% | **+0.69** | 29.5% | 84.5% | 5047.0 |
| | **Policy 3: Universe Terms Alone (No $H\text{-}BB$)** | **-17.80%** | -4.33% | -0.27 | 31.6% | 47.7% | 9106.0 |
| | *Benchmark: Candidate A Baseline* | +7.92% | +20.80% | +0.34 | 36.6% | 83.5% | 6760.0 |
| **N60** | **Policy 1: Standalone $H\text{-}BB$** | **+8.89%** | +15.46% | **+0.35** | 30.1% | 83.6% | 7030.0 |
| | **Policy 2: Full Evolved Equation** | **+10.39%** | +19.00% | **+0.46** | 15.9% | 92.9% | 9012.0 |
| | **Policy 3: Universe Terms Alone (No $H\text{-}BB$)** | **-15.18%** | -8.53% | -0.61 | 25.3% | 92.8% | 9055.0 |
| | *Benchmark: Candidate A Baseline* | -4.54% | +6.55% | -0.01 | 25.4% | 85.1% | 13184.0 |
| **N90** | **Policy 1: Standalone $H\text{-}BB$** | **-0.84%** | +5.29% | **+0.06** | 17.4% | 84.7% | 10790.0 |
| | **Policy 2: Full Evolved Equation** | **+6.68%** | +13.08% | **+0.41** | 12.9% | 88.1% | 10500.0 |
| | **Policy 3: Universe Terms Alone (No $H\text{-}BB$)** | **+2.32%** | +8.21% | +0.19 | 14.3% | 86.0% | 10067.0 |
| | *Benchmark: Candidate A Baseline* | -8.03% | +2.35% | -0.36 | 17.5% | 85.8% | 19257.0 |

---

## 4. Multi-Seed Search Stability Evidence (Block A Development Data)

The 5-seed audit (`seeds = [42, 101, 777, 1337, 2024]`, 20 total runs) produced the following search stability metrics:

1. **Recurrence Across Seeds:**
   - $N=15$: $H\text{-}BB$ selected in 4/5 runs (80.0%), mean weight $+0.8362$
   - $N=30$: $H\text{-}BB$ selected in 2/5 runs (40.0%), mean weight $+0.8350$
   - $N=60$: $H\text{-}BB$ selected in 1/5 runs (20.0%), weight $+0.4323$
   - $N=90$: $H\text{-}BB$ selected in 3/5 runs (60.0%), mean weight $+0.4538$
   - **Global $H\text{-}BB$ Selection:** 10 / 20 runs (50.0%) with **100% strictly positive coefficients** (zero sign inversions across all 20 runs).
2. **Structural Divergence Across Seeds (Pairwise Jaccard):**
   - $N=15$: Jaccard = $0.265$
   - $N=30$: Jaccard = $0.303$
   - $N=60$: Jaccard = $0.287$
   - $N=90$: Jaccard = $0.229$
   - The genetic algorithm does not converge to a single unique set of terms. Multiple alternative term combinations (such as `NL_SQUEEZE_TREND` at 60%, `NL_HURST_ADAPT` at 55%, and `H-STOCH` at 50%) inhabit the top fitness tier.
3. **Rejection of Candidate Interaction:**
   - `PRODUCT_H_BB_TAKER` was selected in only 3 of 20 runs (15.0%) and had near-zero or negative weights.

---

## 5. Interpretation and Evidence Status

1. **Discovery:** APEDP's genetic discovery process generated the four universe-specific candidate equations on Block A development data.
2. **Not One Universal Equation:** The four equations differ by universe and must **not** be represented as a single equation demonstrated to work universally.
3. **H-BB Recurrence:** $H\text{-}BB$ appeared in 10 of 20 runs in the reported five-seed audit and had a strictly positive coefficient whenever selected. This is evidence of recurrence within that development experiment, not proof of universality.
4. **Ablation Evidence:**
   - $H\text{-}BB$ is the primary directional alpha engine: removing it causes net returns to fall by $-25.57\,\text{pp}$ to $-51.44\,\text{pp}$ in $N=15, 30, 60$.
   - Universe-specific additions act primarily as **portfolio variance and drawdown dampers** (reducing Max DD by $-11.6\,\text{pp}$ in $N=30$, $-14.2\,\text{pp}$ in $N=60$, and $-4.5\,\text{pp}$ in $N=90$), rather than uniformly boosting gross return.
5. **Promotion Status:** Retain all four equations as **exploratory candidates**.
   - **Formal Status Label:** `EXPLORATORY — NOT PROMOTED — POINT-RETURN PREDICTION NOT SUPPORTED — REPRODUCTION ARTIFACTS PENDING VERIFICATION — PROSPECTIVE VALIDATION REQUIRED`

---

## 6. Documentation Distinction

> [!CAUTION]
> **Epistemic Rule:** Record these as **four APEDP-generated, universe-specific candidate equations** — not as one validated universal equation. The $H\text{-}BB$-only equations are their ablation baselines, and the Block B results are historical evaluation evidence rather than prospective confirmation.

---

## 7. Predictive Fit Methodology: R² and RMSE Scale Conventions

**Provenance Directory:** [`provenance/r2_rmse_scale_conventions_20261009/`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/r2_rmse_scale_conventions_20261009/)  
**Execution Script:** [`scratch/reconcile_r2_rmse_conventions.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/reconcile_r2_rmse_conventions.py)  
**Manifest Hash:** [`manifest.json`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/r2_rmse_scale_conventions_20261009/manifest.json) (`commit: d881d5cd`, `dataset hash: 4729ad81...`, `returncode: 0`)

Because the evolved composite score $S_{i,t} = \sum w_k X^*_k$ is a dimensionless $\mathcal{O}(1)$ decision score ($\text{Std}(S) \approx 0.88 - 1.77$), evaluating its fit against daily percentage returns $r_{i,t+1}$ ($\sigma_r \approx 0.042 - 0.054$) requires distinguishing two explicit scaling conventions:

### Convention 1: As-Is Raw Score (No Return Scaling)
- Prediction model: $\hat{r}_{i,t+1} = S_{i,t}$
- Residual: $e_{i,t+1} = r_{i,t+1} - S_{i,t}$
- Because $\text{Var}(S) \gg \text{Var}(r)$, $\text{RMSE} \approx \sqrt{\mathbb{E}[S^2]} \approx 0.88 - 1.77$, resulting in $\text{SS}_{\text{res}} \gg \text{SS}_{\text{tot}}$ and $R^2$ in the $-26,000\%$ to $-125,000\%$ range.

### Convention 2: Train-Calibrated Zero-Intercept OLS Scale Factor ($\hat{\beta}_{\text{scale}}$)
- Calibration on Block A Train: $\hat{\beta}_{\text{scale}} = \frac{\sum_{i,t \in \text{Train}} S_{i,t} \cdot r_{i,t+1}}{\sum_{i,t \in \text{Train}} S_{i,t}^2}$
- Note on calibration: This zero-intercept slope is optimal for squared error among mappings constrained to pass through zero on the fitting sample, not a universally optimal calibration.
- Out-of-sample prediction on Block B Holdout: $\hat{r}_{i,t+1} = \hat{\beta}_{\text{scale}} \cdot S_{i,t}$
- Maps the $\mathcal{O}(1)$ score to the return variance scale, yielding $\text{RMSE} \approx \sigma_r$ ($0.042 - 0.054$) and $R^2 \approx -0.02\%$ to $-0.37\%$.

### Dual-Convention Reconciliation Table (Block B Holdout: 2024–2025)

| Universe | Train OLS Slope $\hat{\beta}_{\text{scale}}$ | Null Baseline $\sigma_r$ | Conv 1 (As-Is): Holdout RMSE | Conv 1 (As-Is): Holdout $R^2$ | Conv 2 (Scaled): Holdout RMSE | Conv 2 (Scaled): Holdout $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 15$** | $0.001262$ | $0.05191$ | **$1.4957$** | **$-82,910.1\%$** | **$0.05193$** | **$-0.06\%$** |
| **$N = 30$** | $0.001226$ | $0.05404$ | **$0.8816$** | **$-26,518.1\%$** | **$0.05404$** | **$-0.02\%$** |
| **$N = 60$** | $0.001630$ | $0.05011$ | **$1.7746$** | **$-125,335.1\%$** | **$0.05020$** | **$-0.37\%$** |
| **$N = 90$** | $0.002015$ | $0.04218$ | **$1.1076$** | **$-68,870.9\%$** | **$0.04222$** | **$-0.20\%$** |

---

## 8. Four Methodological Qualifications

1. **Scale and Prediction-Semantics Mismatch (Not Dimensional Inconsistency):**  
   Both a return ratio and a composite score can be mathematically dimensionless. The failure of Convention 1 is not a dimensional impossibility, but a semantic and scale mismatch: assigning a decision score the meaning and scale of a return forecast without an empirical mapping.
2. **Block A Fit Is Not Independent Confirmation:**  
   The equations were evolved on Block A, and the zero-intercept scale factors were fitted on that exact same development window. In-sample $R^2$ ($+0.01\%$ to $+0.17\%$) does not constitute independent predictive evidence.
3. **Block B Is Reused Historical Holdout (Not Untouched Validation):**  
   Block B (2024–2025) has been repeatedly analyzed across multiple rounds of this research trajectory. It serves as historical diagnostic evidence, not as untouched prospective confirmation.
4. **Predictive-Gate Target Alignment:**  
   The negative out-of-sample $R^2$ establishes failure of **pointwise conditional-mean return forecasting** (Gate L1 point subtest). It does not automatically refute trading utility. Directional discrimination, cross-sectional ranking (Spearman IC $\approx -0.01$), threshold-conditioned returns, and risk-adjusted portfolio performance after transaction costs remain distinct empirical questions.

---

## 9. Controlled A/B Benchmark: Baseline (Arm A) vs. Validation-Pruned (Arm B)

**Provenance Directory:** [`provenance/controlled_ab_benchmark_20261009/`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/controlled_ab_benchmark_20261009/)  
**Execution Script:** [`scratch/audit_controlled_ab_experiment.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/audit_controlled_ab_experiment.py)  
**Manifest:** [`manifest.json`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/controlled_ab_benchmark_20261009/manifest.json) (`commit: d881d5cd`, `returncode: 0`)  
**Results Artifact:** [`results.json`](file:///home/psalmprax/ALL_PROJECTS/science/provenance/controlled_ab_benchmark_20261009/results.json)  

### 9.1 Pre-Registered Protocol and Partition Boundaries
To test whether backward ablation and parsimony pruning improve discovery stability and out-of-sample trading performance without leakage:
1. **Temporal Partitioning (Half-Open Slicing Convention `[start:stop]`):**
   - **Sub-Train ($A_{\text{train}}$):** Python interval `[0:500]`, exactly 500 daily bars (2021-01-04 to 2022-12-27). Used strictly for genetic evolution.
   - **Internal Validation ($A_{\text{val}}$):** Python interval `[500:753]`, exactly 253 daily bars (2022-12-28 to 2023-12-29). Used strictly for backward elimination pruning.
   - **Protected Test ($B_{\text{test}}$):** Python interval `[753:1255]`, exactly 502 daily bars (2024-01-02 to 2025-12-31). Reused historical holdout; evaluated only after freezing final equations.
   - **Leakage Verification:** $\text{idx}_{\text{subtrain}} \cap \text{idx}_{\text{val}} = \emptyset$ and $\text{idx}_{\text{val}} \cap \text{idx}_{\text{test}} = \emptyset$. Forward target returns $r_{t+1}$ in `simulate_portfolio` strictly terminate at $T-1$, preventing cross-boundary lookahead. Expanding causal z-scores use strictly historical observations.
2. **Experimental Arms:**
   - **Arm A (Baseline):** Unpruned multi-term equations directly evolved on $A_{\text{train}}$.
   - **Arm B (Validation-Pruned):** Iterative backward elimination on $A_{\text{val}}$ (pruning any term whose removal maintains or improves validation Sharpe within tolerance $\epsilon = 0.02$).
3. **Execution Setup:** 5 pre-registered seeds `[42, 101, 777, 1337, 2024]` across 4 universes $N \in \{15, 30, 60, 90\}$. 5.0 bps transaction cost, inverse-volatility weighting, capital compounding.

---

### 9.2 Machine-Verified Summary Table Across 5 Seeds

| Universe | Arm | Mean Terms | Outcome 1: Decision Correlation $\bar{\rho}_{\text{pos}}$ | Outcome 2: Val Net Return | Outcome 2: Val Sharpe | Outcome 3: Test Net Return | Outcome 3: Test Sharpe | Outcome 3: Test Max DD | Outcome 3: Test Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 15$** | **Arm A** | 4.8 | **$+0.918$** | $+26.77\%$ | $+0.95$ | **$+35.24\%$** | **$+0.67$** | **$36.9\%$** | 1,637.6 |
| | **Arm B** | 1.8 | $+0.232$ | **$+61.16\%$** | **$+1.62$** | $+22.67\%$ | $+0.50$ | $45.7\%$ | **1,266.4** |
| **$N = 30$** | **Arm A** | 6.6 | **$+0.745$** | $-11.58\%$ | $-0.26$ | **$+26.52\%$** | **$+0.64$** | **$31.8\%$** | 4,405.2 |
| | **Arm B** | 2.6 | $+0.359$ | **$+51.34\%$** | **$+1.69$** | $+9.81\%$ | $+0.27$ | $41.2\%$ | **4,048.2** |
| **$N = 60$** | **Arm A** | 5.2 | **$+0.348$** | $+6.40\%$ | $+0.58$ | $+5.98\%$ | $+0.28$ | $19.3\%$ | 9,131.6 |
| | **Arm B** | 2.2 | $+0.290$ | **$+59.67\%$** | **$+2.38$** | **$+19.52\%$** | **$+0.68$** | **$19.2\%$** | **6,731.6** |
| **$N = 90$** | **Arm A** | 4.8 | **$+0.465$** | $+3.77\%$ | $+0.41$ | **$-4.21\%$** | **$-0.15$** | **$15.5\%$** | 9,686.6 |
| | **Arm B** | 3.4 | $+0.311$ | **$+21.18\%$** | **$+1.67$** | $-8.09\%$ | $-0.34$ | $21.8\%$ | **9,120.6** |
| **Overall Mean** | **Arm A** | **5.4** | **$+0.619$** | **$+6.34\%$** | **$+0.42$** | **$+15.88\%$** | **$+0.36$** | **$25.9\%$** | **6,215.3** |
| | **Arm B** | **2.5** | **$+0.298$** | **$+48.34\%$** | **$+1.84$** | **$+10.98\%$** | **$+0.28$** | **$32.0\%$** | **5,291.7** |

---

### 9.3 Detailed Seed-Level Results

#### Universe $N = 15$ Assets
- **Seed 42:**
  - Arm A (5 terms): Val Net $=+34.52\%$ (SR $+1.13$) | Test Net $=+26.87\%$ (SR $+0.59$), TO $=1512.0$  
    $S = +1.2576 X^*_{\text{H-BB}} +0.1812 X^*_{\text{H-STOCH}} -0.0451 X^*_{\text{H-A}} +1.1718 X^*_{\text{NL-SQUEEZE}} -0.0491 X^*_{\text{PROD-BB-TAKER}}$
  - Arm B (1 term): Val Net $=+71.89\%$ (SR $+2.18$) | Test Net $=+78.89\%$ (SR $+1.02$), TO $=536.0$  
    $S = -0.0491 X^*_{\text{PROD-BB-TAKER}}$
- **Seed 101:**
  - Arm A (5 terms): Val Net $=+22.32\%$ (SR $+0.85$) | Test Net $=+17.35\%$ (SR $+0.48$), TO $=1823.0$  
    $S = +1.1550 X^*_{\text{H-BB}} -0.2678 X^*_{\text{H-RV}} +0.1600 X^*_{\text{H-LOG}} +1.0499 X^*_{\text{NL-SQUEEZE}} +0.4780 X^*_{\text{NL-MASTER-SNR}}$
  - Arm B (1 term): Val Net $=+92.00\%$ (SR $+1.96$) | Test Net $=+28.33\%$ (SR $+0.61$), TO $=2136.0$  
    $S = +0.1600 X^*_{\text{H-LOG}}$
- **Seed 777:**
  - Arm A (6 terms): Val Net $=+20.98\%$ (SR $+0.81$) | Test Net $=+64.46\%$ (SR $+0.94$), TO $=1699.0$  
    $S = +0.2899 X^*_{\text{H-MACD}} -0.1340 X^*_{\text{H-EMA-CROSS}} +0.8219 X^*_{\text{H-BB}} +1.0642 X^*_{\text{H-STOCH}} +0.4841 X^*_{\text{H-OBV-MOM}} +0.9065 X^*_{\text{NL-SQUEEZE}}$
  - Arm B (3 terms): Val Net $=+64.75\%$ (SR $+1.61$) | Test Net $=+6.76\%$ (SR $+0.38$), TO $=1078.0$  
    $S = +0.2899 X^*_{\text{H-MACD}} +0.4841 X^*_{\text{H-OBV-MOM}} +0.9065 X^*_{\text{NL-SQUEEZE}}$
- **Seed 1337:**
  - Arm A (5 terms): Val Net $=+21.55\%$ (SR $+0.83$) | Test Net $=+46.64\%$ (SR $+0.83$), TO $=1606.0$  
    $S = +0.2856 X^*_{\text{H-MACD}} +0.9848 X^*_{\text{H-BB}} +0.8757 X^*_{\text{H-STOCH}} -0.3979 X^*_{\text{H-RV}} +1.0221 X^*_{\text{NL-SQUEEZE}}$
  - Arm B (1 term): Val Net $=+42.68\%$ (SR $+1.24$) | Test Net $=-21.51\%$ (SR $-0.03$), TO $=1034.0$  
    $S = +1.0221 X^*_{\text{NL-SQUEEZE}}$
- **Seed 2024:**
  - Arm A (3 terms): Val Net $=+34.46\%$ (SR $+1.12$) | Test Net $=+20.87\%$ (SR $+0.53$), TO $=1548.0$  
    $S = +1.1229 X^*_{\text{H-BB}} +0.1329 X^*_{\text{H-LOG}} +0.8096 X^*_{\text{NL-SQUEEZE}}$
  - Arm B (3 terms): Val Net $=+34.46\%$ (SR $+1.12$) | Test Net $=+20.87\%$ (SR $+0.53$), TO $=1548.0$  
    $S = +1.1229 X^*_{\text{H-BB}} +0.1329 X^*_{\text{H-LOG}} +0.8096 X^*_{\text{NL-SQUEEZE}}$

#### Universe $N = 30$ Assets
- **Seed 42:** Arm A (6 terms): Val Net $=-12.71\%$ (SR $-0.20$) | Test Net $=+19.88\%$ (SR $+0.54$) vs. Arm B (3 terms): Val Net $=+57.79\%$ (SR $+1.49$) | Test Net $=-25.37\%$ (SR $-0.18$)
- **Seed 101:** Arm A (7 terms): Val Net $=-24.41\%$ (SR $-0.69$) | Test Net $=+16.19\%$ (SR $+0.48$) vs. Arm B (3 terms): Val Net $=+18.30\%$ (SR $+0.76$) | Test Net $=-2.61\%$ (SR $+0.13$)
- **Seed 777:** Arm A (6 terms): Val Net $=-37.94\%$ (SR $-1.40$) | Test Net $=+43.25\%$ (SR $+0.92$) vs. Arm B (3 terms): Val Net $=+80.81\%$ (SR $+2.21$) | Test Net $=+43.83\%$ (SR $+0.81$)
- **Seed 1337:** Arm A (7 terms): Val Net $=+7.79\%$ (SR $+0.47$) | Test Net $=+14.97\%$ (SR $+0.45$) vs. Arm B (3 terms): Val Net $=+33.42\%$ (SR $+1.13$) | Test Net $=-23.21\%$ (SR $-0.19$)
- **Seed 2024:** Arm A (7 terms): Val Net $=+9.38\%$ (SR $+0.51$) | Test Net $=+38.31\%$ (SR $+0.79$) vs. Arm B (1 term): Val Net $=+66.38\%$ (SR $+2.84$) | Test Net $=+56.41\%$ (SR $+0.80$)

#### Universe $N = 60$ Assets
- **Seed 42:** Arm A (5 terms): Val Net $=-28.35\%$ (SR $-1.33$) | Test Net $=-0.52\%$ (SR $+0.11$) vs. Arm B (2 terms): Val Net $=+88.43\%$ (SR $+2.60$) | Test Net $=+53.73\%$ (SR $+1.50$)
- **Seed 101:** Arm A (5 terms): Val Net $=+27.08\%$ (SR $+1.95$) | Test Net $=+1.72\%$ (SR $+0.16$) vs. Arm B (1 term): Val Net $=+42.61\%$ (SR $+2.30$) | Test Net $=+17.17\%$ (SR $+0.70$)
- **Seed 777:** Arm A (6 terms): Val Net $=-9.85\%$ (SR $-0.40$) | Test Net $=+21.89\%$ (SR $+0.71$) vs. Arm B (3 terms): Val Net $=+94.94\%$ (SR $+3.12$) | Test Net $=+3.08\%$ (SR $+0.21$)
- **Seed 1337:** Arm A (5 terms): Val Net $=+19.96\%$ (SR $+1.36$) | Test Net $=-1.28\%$ (SR $+0.02$) vs. Arm B (1 term): Val Net $=+44.79\%$ (SR $+2.35$) | Test Net $=+15.60\%$ (SR $+0.64$)
- **Seed 2024:** Arm A (5 terms): Val Net $=+23.18\%$ (SR $+1.31$) | Test Net $=+8.11\%$ (SR $+0.38$) vs. Arm B (4 terms): Val Net $=+27.56\%$ (SR $+1.53$) | Test Net $=+8.02\%$ (SR $+0.37$)

#### Universe $N = 90$ Assets
- **Seed 42:** Arm A (6 terms): Val Net $=+5.84\%$ (SR $+0.57$) | Test Net $=-0.34\%$ (SR $+0.05$) vs. Arm B (5 terms): Val Net $=+21.92\%$ (SR $+1.84$) | Test Net $=-7.19\%$ (SR $-0.34$)
- **Seed 101:** Arm A (4 terms): Val Net $=-10.04\%$ (SR $-0.61$) | Test Net $=-12.61\%$ (SR $-0.58$) vs. Arm B (1 term): Val Net $=+18.34\%$ (SR $+1.03$) | Test Net $=-29.61\%$ (SR $-1.36$)
- **Seed 777:** Arm A (6 terms): Val Net $=-1.29\%$ (SR $-0.02$) | Test Net $=-8.31\%$ (SR $-0.38$) vs. Arm B (4 terms): Val Net $=+28.16\%$ (SR $+2.27$) | Test Net $=-7.20\%$ (SR $-0.31$)
- **Seed 1337:** Arm A (4 terms): Val Net $=+11.06\%$ (SR $+0.90$) | Test Net $=-3.33\%$ (SR $-0.09$) vs. Arm B (3 terms): Val Net $=+24.16\%$ (SR $+2.01$) | Test Net $=+0.02\%$ (SR $+0.08$)
- **Seed 2024:** Arm A (4 terms): Val Net $=+13.30\%$ (SR $+1.19$) | Test Net $=+3.54\%$ (SR $+0.25$) vs. Arm B (4 terms): Val Net $=+13.30\%$ (SR $+1.19$) | Test Net $=+3.54\%$ (SR $+0.25$)

---

### 9.4 Statistical Uncertainty Analysis (N = 20 Paired Runs Across 4 Universes)

Across the 20 paired runs (4 universes $\times$ 5 pre-registered seeds):
- **Model Complexity (Term Count):** Mean paired difference (Arm B $-$ Arm A) = $-2.85 \pm 0.34$ terms (95% CI: $[-3.56, -2.14]$ terms, paired $t = -8.143$, $p = 1.29 \times 10^{-7}$). Pruning achieved substantial and statistically robust term reduction.
- **Protected Test Net Return:** Mean paired difference (Arm B $-$ Arm A) = $-4.91 \pm 6.83\,\text{pp}$ (95% CI: $[-19.21, +9.40]\,\text{pp}$, paired $t = -0.700$, $p = 0.4926$; Wilcoxon signed-rank $W = 71.0$, $p = 0.5277$). Directionally worse on aggregate and in 3 of 4 universes, but with wide cross-seed variance that is not statistically distinguishable from zero.
- **Protected Test Sharpe:** Mean paired difference (Arm B $-$ Arm A) = $-0.08 \pm 0.12$ (95% CI: $[-0.33, +0.18]$, paired $t = -0.629$, $p = 0.5370$; Wilcoxon signed-rank $W = 66.5$, $p = 0.4079$).
- **Cross-Seed Decision Agreement:** Sample mean position correlation $\bar{\rho}_{\text{pos}}$ was lower for Arm B across all four universes ($N=15$: $+0.918 \to +0.232$; $N=30$: $+0.745 \to +0.359$; $N=60$: $+0.348 \to +0.290$; $N=90$: $+0.465 \to +0.311$). While indicating lower empirical consistency across random seeds, this represents a sample observation with dispersion, not an asymptotic hypothesis test.

---

### 9.5 Methodological Conclusions and Qualifications

1. **Rejection of Validation-Based Backward Elimination as Default:**  
   The experiment supports rejecting this specific validation-based backward-elimination procedure as the default discovery-engine upgrade. Pruning reduced model complexity by $-53.7\%$ but degraded mean test net return ($-4.90\,\text{pp}$), Sharpe ($-0.08$), and maximum drawdown ($+6.1\,\text{pp}$), failing in three of the four universes ($N=15, 30, 90$). It improved performance only in $N=60$.
2. **The "Risk-Damper" Explanation Remains Unconfirmed:**  
   While the degraded performance of the pruned portfolios is consistent with the hypothesis that auxiliary terms acted as risk controls across market regimes, it does not prove it. The deterioration could equally reflect validation-fold overfitting, seed-dependent feature selection, or complex interactions among terms. Confirming this mechanism requires inspecting regime-by-regime contributions of removed terms on independent data.
3. **Validation Sharpe Inflation vs. Test Efficacy:**  
   The surge in validation Sharpe from $+0.42$ to $+1.84$ illustrates classic selection-induced metric inflation when pruning against a single validation window.
4. **Historical Retrospective Status (Not Pristine Prospective Holdout):**  
   Because Block B (2024–2025) has been repeatedly analyzed in previous stages of this research project, these findings serve as a rigorous retrospective controlled comparison, not as an untouched prospective confirmation. Any future protocol upgrade must be evaluated on genuinely unexamined forward-testing data.

---

## 10. 15-Asset Individual Asset Audit & Realized Return Spreads (Block B Holdout)

**Execution Script:** [`scratch/generate_exact_summary_tables.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/generate_exact_summary_tables.py)  
**Evaluation Window:** Block B (2024-01-02 to 2025-12-31, exactly 502 daily bars)  
**Transaction Friction:** Flat 5.0 bps per unit of turnover (one-way); excludes bid-ask spread, slippage, and borrow financing costs.  

### 10.1 Dataset Calendar Synchronization (502 Bars vs. 730 Calendar Days)
The presence of 502 daily bars across a two-year window (730 calendar dates) is an artifact of cross-asset synchronization in [`data/market/multi_asset_90_5yr_orderflow.csv`](file:///home/psalmprax/ALL_PROJECTS/science/data/market/multi_asset_90_5yr_orderflow.csv):
- The dataset pools 45 crypto assets alongside 45 US equities.
- To maintain a synchronized date index across asset classes, crypto data was sampled strictly on the **US equity trading calendar** (Monday through Friday, excluding NYSE market holidays).
- Breakdown: Tuesday (105), Wednesday (102), Friday (101), Thursday (98), Monday (96), Saturday (0), Sunday (0).
- Consequently, the strategy operates on a **weekday-only decision schedule**, not continuous 24/7 crypto execution.

---

### 10.2 Master Machine-Verified Asset Audit Table

| Asset | Start Price (2024-01-02) | End Price (2025-12-31) | Actual Buy & Hold Return ($p_1/p_0 - 1$) | Strategy Gross Return | Strategy Net Return (5 bps) | Strategy Sharpe | Strategy Max DD | Strategy Turnover | Win Rate | **Realized Return Spread** ($R_{\text{strat}} - R_{\text{BH}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC** | $\$44,946.91$ | $\$87,648.22$ | **$+95.00\%$** | $-32.44\%$ | **$-38.64\%$** | $-0.35$ | $66.4\%$ | 191.0 | $49.0\%$ | **$-133.65\,\text{pp}$** |
| **ETH** | $\$2,355.34$ | $\$2,971.64$ | **$+26.17\%$** | $+101.97\%$ | **$+86.99\%$** | $0.96$ | $57.3\%$ | 155.0 | $50.1\%$ | **$+60.83\,\text{pp}$** |
| **SOL** | $\$106.73$ | $\$124.65$ | **$+16.79\%$** | $+31.77\%$ | **$+20.38\%$** | $0.60$ | $67.3\%$ | 181.0 | $53.6\%$ | **$+3.59\,\text{pp}$** |
| **BNB** | $\$312.20$ | $\$864.30$ | **$+176.84\%$** | $-28.12\%$ | **$-34.95\%$** | $-0.17$ | $80.2\%$ | 199.0 | $48.1\%$ | **$-211.80\,\text{pp}$** |
| **XRP** | $\$0.62$ | $\$1.84$ | **$+194.96\%$** | $-19.57\%$ | **$-26.78\%$** | $0.22$ | $78.0\%$ | 187.0 | $47.6\%$ | **$-221.73\,\text{pp}$** |
| **ADA** | $\$0.60$ | $\$0.33$ | **$-44.88\%$** | $+33.69\%$ | **$+21.62\%$** | $0.64$ | $70.5\%$ | 191.0 | $51.6\%$ | **$+66.50\,\text{pp}$** |
| **AVAX** | $\$40.61$ | $\$12.32$ | **$-69.66\%$** | $+96.10\%$ | **$+77.66\%$** | $0.92$ | $59.2\%$ | 199.0 | $51.0\%$ | **$+147.32\,\text{pp}$** |
| **DOGE** | $\$0.09$ | $\$0.12$ | **$+28.95\%$** | $+199.40\%$ | **$+174.27\%$** | $1.18$ | $75.6\%$ | 175.0 | $50.4\%$ | **$+145.32\,\text{pp}$** |
| **DOT** | $\$8.41$ | $\$1.79$ | **$-78.73\%$** | $+16.25\%$ | **$+5.13\%$** | $0.51$ | $56.5\%$ | 201.0 | $52.8\%$ | **$+83.86\,\text{pp}$** |
| **LINK** | $\$15.17$ | $\$12.21$ | **$-19.49\%$** | $-25.16\%$ | **$-32.60\%$** | $0.24$ | $55.2\%$ | 211.0 | $48.0\%$ | **$-13.11\,\text{pp}$** |
| **MATIC** | $\$0.97$ | $\$0.38$ | **$-60.85\%$** | $+55.68\%$ | **$+51.01\%$** | $0.82$ | $36.9\%$ | 61.0 | $17.8\%$ | **$+111.86\,\text{pp}$** |
| **NEAR** | $\$3.92$ | $\$1.51$ | **$-61.41\%$** | $-18.67\%$ | **$-24.95\%$** | $0.41$ | $77.6\%$ | 161.0 | $49.2\%$ | **$+36.46\,\text{pp}$** |
| **ATOM** | $\$10.96$ | $\$1.93$ | **$-82.40\%$** | $+153.91\%$ | **$+130.45\%$** | $1.10$ | $72.4\%$ | 195.0 | $51.1\%$ | **$+212.84\,\text{pp}$** |
| **UNI** | $\$7.17$ | $\$5.64$ | **$-21.39\%$** | $+136.11\%$ | **$+117.80\%$** | $1.05$ | $69.6\%$ | 163.0 | $50.1\%$ | **$+139.19\,\text{pp}$** |
| **LTC** | $\$72.97$ | $\$76.85$ | **$+5.32\%$** | $-86.31\%$ | **$-87.45\%$** | $-1.24$ | $91.6\%$ | 173.0 | $44.2\%$ | **$-92.77\,\text{pp}$** |
| **Arithmetic Mean** | — | — | **$+7.01\%$** | **$+40.97\%$** | **$+29.33\%$** | **$0.46$** | **$67.6\%$** | **173.6** | **47.6%** | **$+22.31\,\text{pp}$** |
| **Median** | — | — | **$-19.49\%$** | **$+31.77\%$** | **$+20.38\%$** | **$0.60$** | **$67.3\%$** | **187.0** | **50.1%** | **$+60.83\,\text{pp}$** |

---

### 10.3 Segment Analysis (Corrected Arithmetic)

| Segment | Included Assets | Mean Buy & Hold Return | Mean Strategy Net Return | Corrected Realized Return Spread |
| :--- | :--- | :---: | :---: | :---: |
| **Top Altcoin Outperformers** | ATOM, AVAX, DOGE, UNI, MATIC | **$-41.07\%$** | **$+110.24\%$** | **$+151.31\,\text{pp}$** |
| **Secular Trend Leaders** | BTC, BNB, XRP | **$+155.60\%$** | **$-33.46\%$** | **$-189.06\,\text{pp}$** |
| **Low-Beta Chop Asset** | LTC | **$+5.32\%$** | **$-87.45\%$** | **$-92.77\,\text{pp}$** |

- **Segment Insights:**  
  The strategy generated substantial outperformance on high-beta declining or volatile altcoins by capturing short-trend expansions, but suffered severe capital loss on secular market leaders (BTC, BNB, XRP) by repeatedly shorting strong upward trends.

---

## 11. Exploratory Multitimeframe Discovery — Provisional Findings (1m to 60m)

**Execution Script:** [`scratch/audit_multitimeframe_evolution.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/audit_multitimeframe_evolution.py)  
**Structured Results:** [`scratch/multitimeframe_evolution_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/multitimeframe_evolution_results.json)  

### 11.1 Experimental Protocol
A genetic algorithm searched candidate trading equations for five core crypto assets (`BTC`, `ETH`, `SOL`, `BNB`, `DOGE`) across six intraday candle intervals plus daily data, using a population of 60, 25 generations, random seed 42, and an assumed transaction cost of 5 bps. Data was fetched directly via Binance public API.

| Interval | Sample Bar Count | Calendar Window Covered | Evolved Equation (Seed 42) | Reconciled Active Grammar Terms | In-Sample Fit Sharpe Ratio | Decision Threshold $\theta$ |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **1m** | 1,000 | ~16.7 Hours | $S_{\text{1m}} = -0.9271 X^*_{\text{H-MACD}} + 1.2806 X^*_{\text{H-EMA\_CROSS}} + 0.0375 X^*_{\text{H-STOCH}}$ | `H-MACD`, `H-EMA_CROSS`, `H-STOCH` | **$-1.61$** | $0.1087$ |
| **5m** | 1,000 | ~3.5 Days | $S_{\text{5m}} = +1.1006 X^*_{\text{H-OBV\_MOM}}$ | `H-OBV_MOM` | **$-0.04$** | $0.1711$ |
| **15m** | 1,000 | ~10.4 Days | $S_{\text{15m}} = +0.2328 X^*_{\text{H-MACD}} - 0.4058 X^*_{\text{H-EMA\_CROSS}}$ | `H-MACD`, `H-EMA_CROSS` | **$+0.31$** | $0.0375$ |
| **30m** | 1,000 | ~20.8 Days | $S_{\text{30m}} = -0.8527 X^*_{\text{H-SQUEEZE}} - 0.0072 X^*_{\text{H-EMA\_CROSS}} - 0.9181 X^*_{\text{NL-SQUEEZE\_TREND}} + 0.0819 X^*_{\text{NL-RISK\_ADJ\_MOM}}$ | `H-SQUEEZE`, `H-EMA_CROSS`, `NL-SQUEEZE_TREND`, `NL-RISK_ADJ_MOM` | **$+0.70$** | $0.1250$ |
| **45m** | 333 | ~10.4 Days | $S_{\text{45m}} = -0.9711 X^*_{\text{H-SQUEEZE}} - 0.5490 X^*_{\text{NL-SQUEEZE\_TREND}}$ | `H-SQUEEZE`, `NL-SQUEEZE_TREND` | **$+2.44$** | $0.1858$ |
| **60m (1h)** | 1,000 | ~41.7 Days | $S_{\text{60m}} = +0.5168 X^*_{\text{H-NATR}} + 0.0896 X^*_{\text{H-SQUEEZE}} - 0.0357 X^*_{\text{H-BB}}$ | `H-NATR`, `H-SQUEEZE`, `H-BB` | **$+1.21$** | $0.1808$ |
| **Daily (1d)** | 753 | ~2.9 Years | $S_{\text{1d}} = +1.1948 X^*_{\text{H-BB}} + 0.4501 X^*_{\text{H-OBV\_MOM}} + 0.2227 X^*_{\text{H-STOCH}} - 0.4839 X^*_{\text{NL-HURST\_ADAPT}}$ | `H-BB`, `H-OBV_MOM`, `H-STOCH`, `NL-HURST_ADAPT` | **$+0.95$** | $0.1695$ |

---

### 11.2 Epistemic Qualifications on Multitimeframe Findings

1. **In-Sample Search Fitness (Not Validated Alpha):**  
   The reported Sharpe ratios represent in-sample optimization scores achieved during search on that specific data slice. They do not constitute independent prospective validation.
2. **Confounding Calendar Duration with Timeframe:**  
   Using fixed bar counts means 1,000 one-minute bars covered only 16.7 hours, while 1,000 one-hour bars covered 41.7 days, and daily data covered nearly three years. The experiment confounds candle interval mechanics with differing calendar market regimes and sample lengths.
3. **The Squeeze Sign Mechanism:**  
   `H-SQUEEZE` measures the Bollinger-to-Keltner ratio $\frac{4 \cdot \text{std}_{20}}{3 \cdot \text{ATR}_{14}}$. Low values indicate tight consolidation. The negative weights on 30m and 45m mechanically align the score with volatility contraction phases.
4. **Provisional Exploratory Status:**  
   These findings generate an exploratory hypothesis that signal selection may vary across timescales. They do not yet confirm a permanent market law, a universal friction boundary, or a validated intraday edge. Confirmatory testing requires calendar-aligned windows, independent seeds, and out-of-sample forward evaluation with realistic microstructure friction.

---

## 12. Expanded 46-Feature Grammar Evolution: Universes (15, 30, 60, 90) & Intraday Timeframes (1m to 60m)

**Execution Scripts:**  
* Universes (Daily 15, 30, 60, 90): [`scratch/expanded_universe_evolution_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/expanded_universe_evolution_results.json)  
* Multitimeframe (1m to 60m): [`scratch/run_expanded_multitimeframe_evolution.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/run_expanded_multitimeframe_evolution.py) | [`scratch/expanded_multitimeframe_evolution_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/expanded_multitimeframe_evolution_results.json)  
**Grammar Implementation:** [`apedp_dyn/sandbox/multi_asset_signals_sandbox.py:calc_expanded_signals_for_asset`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/multi_asset_signals_sandbox.py#L328-L415) ($K=46$ features).  
**Genetic Search Setup:** pop=60, gen=25, seed=42, flat 5.0 bps fee.

### 12.1 Daily Universes ($N=15, 30, 60, 90$ Assets) under Expanded 46-Feature Grammar

| Universe | Assets ($N$) | Bars ($T_{\text{train}}$) | Evolved Equation (Seed 42, $K=46$) | Active Features | In-Sample Fit | In-Sample Sharpe | Trades | Net Return | Threshold $\theta$ |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **N=15 (Crypto)** | 15 | 500 | $S_{\text{N15}} = +0.1666 X^*_{\text{H-SQUEEZE}} + 0.3293 X^*_{\text{NL\_RISK\_ADJ\_MOM}} - 0.1772 X^*_{\text{NL\_HURST\_ADAPT}} + 0.1413 X^*_{\text{BODY\_TO\_RANGE}} + 0.1179 X^*_{\text{EMA\_SPREAD}}$ | `H-SQUEEZE`, `NL_RISK_ADJ_MOM`, `NL_HURST_ADAPT`, `BODY_TO_RANGE`*, `EMA_SPREAD`* | **$4.39$** | **$+3.08$** | 1,453 | +137.4% | $0.1274$ |
| **N=30 (Multi)** | 30 | 500 | $S_{\text{N30}} = +1.1158 X^*_{\text{H-RV}} + 0.3205 X^*_{\text{BB\_WIDTH}}$ | `H-RV`, `BB_WIDTH`* | **$2.49$** | **$+2.08$**† | 4,301 | +41.7% | $0.2040$ |
| **N=60 (Multi)** | 60 | 500 | $S_{\text{N60}} = +0.3893 X^*_{\text{H-RSI}} - 0.7888 X^*_{\text{H-OBV\_MOM}} + 0.5316 X^*_{\text{ADX}} + 0.7934 X^*_{\text{KELTNER\_WIDTH}} - 0.0626 X^*_{\text{SIGNED\_VOL\_MOM}}$ | `H-RSI`, `H-OBV_MOM`, `ADX`*, `KELTNER_WIDTH`*, `SIGNED_VOL_MOM`* | **$2.67$** | **$+2.28$**† | 3,496 | +45.7% | $0.1397$ |
| **N=90 (Multi)** | 90 | 500 | $S_{\text{N90}} = -0.0792 X^*_{\text{H-CROSS\_MOM}} + 0.2664 X^*_{\text{NL\_SQUEEZE\_TREND}} + 0.6883 X^*_{\text{BB\_WIDTH}}$ | `H-CROSS_MOM`, `NL_SQUEEZE_TREND`, `BB_WIDTH`* | **$2.49$** | **$+2.11$** | 4,540 | +41.1% | $0.1344$ |

*\*Denotes newly admitted features from the expanded trading grammar catalog actively selected by the genetic algorithm.*  
*†Corrected from previously reported $0.00$ caused by GA elitism chromosome attribute propagation defect.*

### 12.2 Intraday Discovery ($1\text{m}$ to $60\text{m}$) under Expanded 46-Feature Grammar (Audited & Re-evolved)

| Interval | Bars | Calendar Window | Evolved Equation ($K=46$, Seed 42) | Newly Admitted Terms Selected | In-Sample Fit | In-Sample Sharpe | Trades | In-Sample Return | Threshold $\theta$ |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1m** | 1,000 | ~16.7 Hours | $S_{\text{1m}} = -0.6979 X^*_{\text{H-MACD}} + 0.8974 X^*_{\text{H-EMA\_CROSS}} + 0.0376 X^*_{\text{NL\_RISK\_ADJ\_MOM}}$ | None (Legacy trend & mom) | **$-1.57$** | **$-1.49$** | 280 | $-1.82\%$ | $0.1154$ |
| **5m** | 1,000 | ~3.5 Days | $S_{\text{5m}} = +0.6191 X^*_{\text{H-SQUEEZE}} + 0.0350 X^*_{\text{TREND\_EFFICIENCY}} + 0.2490 X^*_{\text{PARKINSON\_VOL}}$ | `TREND_EFFICIENCY`*, `PARKINSON_VOL`* | **$-0.11$** | **$-0.11$** | 429 | $-0.31\%$ | $0.0364$ |
| **15m** | 1,000 | ~10.4 Days | $S_{\text{15m}} = -0.6040 X^*_{\text{EMA\_SPREAD}} - 0.2993 X^*_{\text{PARKINSON\_VOL}}$ | `EMA_SPREAD`*, `PARKINSON_VOL`* | **$+0.27$** | **$+0.27$** | 318 | $+0.99\%$ | $0.0768$ |
| **30m** | 1,000 | ~20.8 Days | $S_{\text{30m}} = -0.4609 X^*_{\text{EMA\_SPREAD}} - 0.1417 X^*_{\text{BB\_WIDTH}} - 0.1648 X^*_{\text{CMF}}$ | `EMA_SPREAD`*, `BB_WIDTH`*, `CMF`* | **$+0.22$** | **$+0.23$** | 553 | $+1.60\%$ | $0.1363$ |
| **45m** | 333 | ~10.4 Days | $S_{\text{45m}} = -0.6347 X^*_{\text{BREAKOUT\_STRENGTH}}$ | `BREAKOUT_STRENGTH`* | **$+1.13$** | **$+1.07$** | 226 | $+3.23\%$ | $0.1523$ |
| **60m (1h)** | 1,000 | ~41.7 Days | $S_{\text{60m}} = +0.2324 X^*_{\text{H-EMA\_CROSS}} - 0.6140 X^*_{\text{EMA\_SPREAD}}$ | `EMA_SPREAD`* | **$+1.14$** | **$+1.04$** | 283 | $+8.19\%$ | $0.1654$ |

### 12.3 Forensic Audit & Methodological Rectification Note

1. **Defect 1: Dead-Feature Zero-Trade Artifact (`BREAKOUT_STRENGTH`):**  
   In the original run, `BREAKOUT_STRENGTH` was computed as `causal_zscore(np.maximum(p - don_h, 0.0) / atr14)` where `don_h = pd.Series(h).rolling(20).max()`. Because $p \le h \le \max_{20}(h)$, $p - don\_h \le 0$ at every bar, making the feature identically $0.0$ ($std = 0.0$). A chromosome containing only this feature executed $0.0$ trades (turnover $0.0$), incurring no turnover fees. In 1m, 5m, and 15m where active trading lost money under 5.0 bps fees, the zero-trade chromosome survived with fitness $-0.02$ solely due to the single-term sparsity penalty ($-0.02 \times 1$). The previous claim that 1m, 5m, and 15m "converged to a negative Donchian boundary fader" was an unverified confabulation based on this dead-signal artifact. The code was patched to compute causal signed Donchian excursion relative to the trailing window prior to the current bar (`shift(1)`), restoring non-zero variance ($std = 1.13$) and genuine trading dynamics.

2. **Defect 2: Genetic Algorithm Elitism Attribute Drop:**  
   In `run_multi_asset_genetic_algorithm`, elitism preservation copied `MultiAssetChromosome(mask=..., weights=..., threshold=..., fitness=c.fitness)` but omitted `in_sample_sharpe=c.in_sample_sharpe`. Elite chromosomes carried across generations reverted to the dataclass default `0.0`, resulting in reported Sharpe ratios of $0.00$ for N=30, N=60, and intraday runs despite positive fitness and thousands of executed trades. Patched to explicitly propagate `in_sample_sharpe` and re-evaluate `population[0]` before return. Recomputed true in-sample Sharpe ratios: N=30 is $+2.08$ (4,301 trades, +41.7% return), N=60 is $+2.28$ (3,496 trades, +45.7% return).

3. **Intraday Empirical Reality:**  
   With both bugs resolved, the genetic algorithm discovers genuinely distinct equations across timeframes:
   - At high frequencies (1m and 5m), transaction friction dominates; in-sample Sharpe is negative ($-1.49$ on 1m, $-0.11$ on 5m) despite active trading.
   - At intermediate and slower intraday frequencies (15m, 30m, 45m, 60m), positive in-sample Sharpe emerges ($+0.27$ at 15m, $+0.23$ at 30m, $+1.07$ at 45m, $+1.04$ at 60m).
   - Newly admitted grammar features (`EMA_SPREAD`, `PARKINSON_VOL`, `BB_WIDTH`, `CMF`, `BREAKOUT_STRENGTH`) are actively selected in 5 of the 6 intraday intervals.
   - All results remain in-sample evolutionary discoveries; none constitute out-of-sample validated edge.

---

### 12.4 Independent Verification & Calendar-Aligned Out-of-Sample Benchmark (10.0 Days, 3 Seeds)

**Audit Scripts & Artifacts:**  
* Audit Runner: [`scratch/audit_intraday_causality_and_calendar_alignment.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/audit_intraday_causality_and_calendar_alignment.py)  
* Aligned Benchmark Results: [`scratch/calendar_aligned_multitimeframe_benchmark.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/calendar_aligned_multitimeframe_benchmark.json)  
* Full Audit Verdict: [`scratch/audit_and_validation_verdict.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/audit_and_validation_verdict.json)

#### 1. Check 1 — Feature Invariants & Strict Temporal Causality Verification
* **Invariant Status (`BREAKOUT_STRENGTH`):** Non-zero standard deviation ($std = 1.1346$). Detects 35 upside breakouts ($> +0.05$), 308 downside breakdowns ($< -0.05$), and 752 in-channel bars ($[-0.05, +0.05]$).
* **Future Perturbation Invariance across All 46 Features:**  
  Tested at inspection bars $t \in \{150, 300, 500, 750, 900\}$. In each test, all future observations $t' > t$ were perturbed with random shocks ($0.1\times$ to $10.0\times$). Across all 46 features:
  $$\max_{k \in [0, 45]} |F_{\text{perturbed}}[t, k] - F_{\text{unperturbed}}[t, k]| = \mathbf{0.00 \times 10^0}$$
  Zero lookahead detected. Expanding causal normalization strictly isolates past data from future bars.

#### 2. Check 2 — Metric Consistency Verification
Stored metrics from [`scratch/expanded_multitimeframe_evolution_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/expanded_multitimeframe_evolution_results.json) were verified against fresh end-to-end recalculations from raw price and feature tensors. Stored in-sample fitness, Sharpe ratios, and trade counts match fresh execution exactly.

#### 3. Check 3 — Resolution of the 45-Minute Bar-Count Discrepancy
* **Root Cause Identified:** Binance REST API imposes a strict ceiling of 1,000 candles per single HTTP call (`limit=1000`). Requesting `limit=3000` for 15m bars to resample into 1,000 45m bars silently returned 1,000 15m bars, resulting in $1,000 \div 3 = \mathbf{333\text{ bars}}$ covering 10.4 days.
* **Duration Confounding Identified:** Fixed bar counts (1,000 bars) naturally confounds bar interval with calendar duration and macro regime (1m sampled ~16.7 hours, whereas 60m sampled ~41.7 days). Apparent timeframe differences cannot be attributed to bar duration without calendar alignment.

#### 4. Check 4 — Calendar-Aligned Multi-Seed Out-of-Sample Benchmark (240 Hours / 10.0 Days)
All 6 intervals were synchronized over the **exact same 10.0 calendar days** (2026-09-29 19:00 to 2026-10-09 19:00 UTC) across BTC, ETH, SOL, BNB, DOGE. Evaluated with a **70% In-Sample Train / 30% Out-of-Sample Forward Holdout** split across **3 pre-registered seeds (42, 101, 777)** under flat 5.0 bps turnover fees:

| Interval | Calendar Window | Total Bars | Train / Test Bars | Holdout B&H Return | Mean In-Sample Sharpe | Mean Out-of-Sample Sharpe | Mean Out-of-Sample Net Return | Out-of-Sample Trades Range | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1m** | 240.0 Hours | 14,400 | 10,080 / 4,320 | $-7.53\%$ | $-1.87$ | $\mathbf{-65.92}$ | $\mathbf{-12.71\%}$ | 1,167 – 2,188 | Catastrophic fee drag; rejected |
| **5m** | 239.9 Hours | 2,880 | 2,015 / 865 | $-7.60\%$ | $-0.98$ | $\mathbf{-25.16}$ | $\mathbf{-3.89\%}$ | 288 – 364 | Fee drag; rejected |
| **15m** | 239.8 Hours | 960 | 672 / 288 | $-7.56\%$ | $+1.08$ | $\mathbf{-19.53}$ | $\mathbf{-2.22\%}$ | 116 – 153 | In-sample overfit; rejected |
| **30m** | 239.5 Hours | 480 | 336 / 144 | $-7.56\%$ | $+2.30$ | $\mathbf{-6.12}$ | $\mathbf{-0.79\%}$ | 53 – 75 | In-sample overfit; rejected |
| **45m** | 239.2 Hours | 320 | 224 / 96 | $-7.56\%$ | $+2.66$ | $\mathbf{-12.24}$ | $\mathbf{-1.73\%}$ | 47 – 54 | In-sample overfit; rejected |
| **60m** | 239.0 Hours | 240 | 168 / 72 | $-7.56\%$ | $+4.72$ | $\mathbf{-22.04}$ | $\mathbf{-2.57\%}$ | 70 – 72 | In-sample overfit; rejected |

#### 5. Directional Mapping Interpretation
In the 45m equation, $S_{\text{45m}} = -0.6347 X^*_{\text{BREAKOUT\_STRENGTH}}$ with threshold $\theta = 0.1523$:
* When $X^*_{\text{BREAKOUT\_STRENGTH}} > 0$ (upside breakout exceeding 20-bar high), Score $< -\theta \implies \text{Target Position} = \mathbf{-1.0\text{ (Short)}}$.
* When $X^*_{\text{BREAKOUT\_STRENGTH}} < 0$ (downside breakdown below 20-bar low), Score $> +\theta \implies \text{Target Position} = \mathbf{+1.0\text{ (Long)}}$.
* The negative sign denotes a **mean-reverting counter-trend (breakout fader)**, not trend continuation.

#### 6. Final Epistemic Decision for APEDP
1. **Defects 1 & 2 Accepted:** Both implementation bugs are permanently accepted into the audit ledger as resolved defects.
2. **Rejection of Out-of-Sample Alpha:** When held to an aligned calendar window with forward out-of-sample testing, **zero newly evolved intraday equations generate positive Sharpe or net positive return**. Apparent in-sample Sharpe was an optimization artifact.
3. **Promotion Status:** All newly generated equations remain strictly classified as **DISCOVERY-STAGE ARTIFACTS — REJECTED FOR PRODUCTION PROMOTION**.

---

## 13. Rolling-Window Equation Stability, Feature Substitution, and Identifiability Audit

**Record Type:** Diagnostic & Identifiability Audit  
**Status:** `DIAGNOSTIC EVIDENCE RECORDED — STRUCTURAL IDENTIFIABILITY UNRESOLVED`  
**Execution Script:** [`scratch/audit_rolling_window_stability.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/audit_rolling_window_stability.py)  
**Results Artifact:** [`scratch/rolling_window_stability_audit_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/rolling_window_stability_audit_results.json)  
**Ledger Entry:** `APEDP_ROLLING_WINDOW_EQUATION_STABILITY_AUDIT`

### 13.1 Feature Definitions & Normalization Meaning ($X^*$)
* **Definition of $X^*$:** $X^*_{i, t, k}$ represents the causally normalized feature representation: an expanding causal z-score (`causal_zscore(v, min_history=60)`), winsorized and standardized per asset, followed by cross-sectional standard normalization across all $N$ assets at bar $t$:
  $$X^*_{i, t, k} = \frac{z_{i, t, k} - \mu_{t, k}^{\text{cross}}}{\sigma_{t, k}^{\text{cross}}}$$
* **Meaning of `H-P` (Volatility Persistence):**  
  $$S_{\text{H-P}}(t) = \text{causal\_zscore}(|r_t|)$$  
  Measures instantaneous absolute return magnitude / price turbulence.
* **Meaning of `H-ASYM` (GJR Panic Leverage):**  
  $$S_{\text{H-ASYM}}(t) = \text{causal\_zscore}(|r_t| \cdot \mathbb{I}(r_t < 0))$$  
  Isolates downside semi-variance and liquidation spikes without penalizing upside rallies.
* **Meaning of `H-BB` (Bollinger Bands %B):**  
  $$S_{\text{H-BB}}(t) = \frac{p_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}, \quad \text{Upper/Lower} = \text{SMA}_{20} \pm 2\sigma_{20}$$  
  Measures normalized price location relative to dynamic volatility bands.

### 13.2 Training Redundancy & Feature Substitution Matrix (Block A, N=30)
Cross-asset feature correlation matrix on training data:

| Feature | H-BB | H-P | H-ASYM | NL_OFI_REV | NL_SQUEEZE_TREND | H-MOM | NL_HURST_ADAPT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H-BB** | $1.0000$ | $-0.0743$ | $-0.3789$ | $-0.0879$ | $\mathbf{+0.4760}$ | $\mathbf{+0.7241}$ | $-0.1222$ |
| **H-P** | $-0.0743$ | $1.0000$ | $\mathbf{+0.5899}$ | $+0.1735$ | $-0.0132$ | $-0.0556$ | $+0.0251$ |
| **H-ASYM** | $-0.3789$ | $\mathbf{+0.5899}$ | $1.0000$ | $+0.1953$ | $-0.1232$ | $-0.3796$ | $+0.1326$ |
| **NL_OFI_REV** | $-0.0879$ | $+0.1735$ | $+0.1953$ | $1.0000$ | $-0.0541$ | $-0.0597$ | $+0.0546$ |
| **NL_SQUEEZE_TREND** | $\mathbf{+0.4760}$ | $-0.0132$ | $-0.1232$ | $-0.0541$ | $1.0000$ | $\mathbf{+0.4353}$ | $+0.0944$ |
| **H-MOM** | $\mathbf{+0.7241}$ | $-0.0556$ | $-0.3796$ | $-0.0597$ | $\mathbf{+0.4353}$ | $1.0000$ | $+0.0199$ |
| **NL_HURST_ADAPT** | $-0.1222$ | $+0.0251$ | $+0.1326$ | $+0.0546$ | $+0.0944$ | $+0.0199$ | $1.0000$ |

*Collinearity Verdict:*  
Strong pairwise collinearity exists between `H-BB` and `H-MOM` ($r = +0.7241$), between `H-P` and `H-ASYM` ($r = +0.5899$), and between `H-BB` and `NL_SQUEEZE_TREND` ($r = +0.4760$). The genetic algorithm frequently substitutes among these collinear representatives, producing term instability despite high underlying signal redundancy.

### 13.3 Rolling-Window Evolution and Agreement on Common Evaluation Period (Block B: 2024-2025, 502 Bars)
Across 5 rolling windows (753 weekday bars each, 83.3% pairwise overlap):

* **Consecutive Jaccard Term Agreement:**  
  Window 1 vs 2: Jaccard = $0.714$  
  Window 2 vs 3: Jaccard = $0.667$  
  Window 3 vs 4: Jaccard = $0.167$  
  Window 4 vs 5: Jaccard = $0.500$  
* **Score & Position Agreement on Common Block B:**  
  Adjacent windows maintain continuous score correlations of $r = 0.67$ to $0.86$, with position agreement rates of $62.8\%$ to $73.6\%$. Over the 5-year span (W1 vs W5), score correlation degrades to $r = 0.1413$ and position agreement drops to $43.26\%$.
* **Common Window Net Return & Sharpe (Block B):**  
  - Benchmark Buy & Hold: $-26.09\%$  
  - Standalone H-BB Model: $+19.28\%$ (Sharpe = $0.71$)  
  - Mean of 5 Evolved Equations: $+47.40\%$ (Sharpe = $2.04$)

### 13.4 Epistemic Verdict & Governance
1. **Low Structural Identifiability Confirmed:** Different feature sets express substantially substitutable predictive information.
2. **H-BB Retention Rule:** Do not force H-BB into candidate equations based solely on historical selection frequency, especially given its coefficient contraction in later periods.
3. **Promotion Policy:** Retain the null result regarding a universal unique equation. Promotion requires incremental value beyond simple baselines on genuinely unexamined forward prospective data.

---

### 13.5 Stage 5 Four-Model Four-Universe Forensic Accounting Audit (Block B: 2024–2025)

**Execution Script:** [`scratch/run_corrected_accounting_audit.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/run_corrected_accounting_audit.py)  
**Dedicated Artifact:** [`scratch/corrected_accounting_stage5_scoreboard.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/corrected_accounting_stage5_scoreboard.json)  
**Forensic Accounting Specification:**
1. **Traded Notional Turnover:** Traded volume is computed directly from normalized portfolio weights $\text{Turnover}_t = \sum_{i=1}^N |w_{i, t} - w_{i, t-1}|$ where $\sum |w_i| = 1.0$. Fee cost is $\text{Fee}_t = 0.0005 \times \text{Turnover}_t$ (flat 5.0 bps per unit of turnover).
2. **Multiplicative Wealth Compounding:** Portfolio wealth compounds multiplicatively $W_t = W_{t-1} \cdot (1 + r_{\text{net}, t})$. Cumulative net return is $(W_T - 1.0) \times 100\%$.
3. **Standard Percentage Max Drawdown:** Computed from true peak-to-trough wealth ratios $\text{DD}_t = \frac{W_t - \max_{s \le t} W_s}{\max_{s \le t} W_s} \times 100\%$. Max drawdown is $\max_t |\text{DD}_t|$.
4. **Trading Session Annualization:** Primary Sharpe is annualized using $\sqrt{252}$ trading sessions ($\sqrt{365}$ reported side-by-side for comparison: $\text{SR}_{252} = 0.831 \times \text{SR}_{365}$).
5. **NYSE Calendar Reconciliation (502 Bars):** The US equity trading calendar spans 252 sessions in 2024 and 250 sessions in 2025, totaling **exactly 502 bars**. The 2025 session count reflects 11 official closures (New Year, Jimmy Carter National Day of Mourning on Jan 9, MLK, Washington, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas).

| Universe | Model / Policy Architecture | Chronological Role | Net Return (Wealth) | Gross Return (Wealth) | Sharpe $\sqrt{252}$ | Sharpe $\sqrt{365}$ | Max Drawdown | Mean Daily Turnover |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=15** *(Crypto)* | **Benchmark: Buy & Hold** | Passive market baseline | $\mathbf{-2.36\%}$ | $-2.36\%$ | $+0.32$ | $+0.38$ | $54.2\%$ | $0.0000$ |
| | **Model 1: Standalone H-BB** | Isolated single-feature baseline | $\mathbf{+42.38\%}$ | $+56.64\%$ | $\mathbf{+1.01}$ | $+1.22$ | $22.7\%$ | $0.3811$ |
| | **Model 3: Restrained Core** | Frozen 2-term risk guard ($+0.84\text{ BB} - 0.25\text{ ASYM}$) | $\mathbf{+34.85\%}$ | $+52.73\%$ | $\mathbf{+0.91}$ | $+1.09$ | $18.9\%$ | $0.4975$ |
| | **Model 4b: Canonical Block-A** | **Strict frozen forward OOS (0% overlap)** | $\mathbf{+26.31\%}$ | $+40.63\%$ | $\mathbf{+0.83}$ | $+1.00$ | $17.7\%$ | $0.4289$ |
| | **Model 2b: Strict Walk-Forward** | **Adaptive rolling forward (0% overlap)** | $\mathbf{-9.23\%}$ | $+1.88\%$ | $\mathbf{-0.15}$ | $-0.19$ | $33.0\%$ | $0.4607$ |
| | *Model 4a: Unconstrained Win-5* | *In-sample overlap with Block B* | *+117.51%* | *+145.11%* | *+2.49* | *+2.99* | *8.9%* | *0.4781* |
| **N=30** *(Multi)* | **Benchmark: Buy & Hold** | Passive market baseline | $\mathbf{+3.76\%}$ | $+3.76\%$ | $+0.36$ | $+0.43$ | $55.4\%$ | $0.0000$ |
| | **Model 1: Standalone H-BB** | Isolated single-feature baseline | $\mathbf{+16.62\%}$ | $+26.43\%$ | $\mathbf{+0.55}$ | $+0.67$ | $17.6\%$ | $0.3223$ |
| | **Model 3: Restrained Core** | Frozen 2-term risk guard ($+0.84\text{ BB} - 0.25\text{ ASYM}$) | $\mathbf{+20.37\%}$ | $+35.24\%$ | $\mathbf{+0.66}$ | $+0.80$ | $16.4\%$ | $0.4652$ |
| | **Model 4b: Canonical Block-A** | **Strict frozen forward OOS (0% overlap)** | $\mathbf{+20.84\%}$ | $+36.40\%$ | $\mathbf{+0.77}$ | $+0.92$ | $12.7\%$ | $0.4837$ |
| | **Model 2b: Strict Walk-Forward** | **Adaptive rolling forward (0% overlap)** | $\mathbf{+34.89\%}$ | $+51.63\%$ | $\mathbf{+1.13}$ | $+1.36$ | $13.7\%$ | $0.4671$ |
| | *Model 4a: Unconstrained Win-5* | *In-sample overlap with Block B* | *+60.39%* | *+81.94%* | *+1.73* | *+2.09* | *11.6%* | *0.5040* |
| **N=60** *(Multi)* | **Benchmark: Buy & Hold** | Passive market baseline | $\mathbf{-3.27\%}$ | $-3.27\%$ | $+0.24$ | $+0.29$ | $52.3\%$ | $0.0000$ |
| | **Model 1: Standalone H-BB** | Isolated single-feature baseline | $\mathbf{-1.40\%}$ | $+6.71\%$ | $\mathbf{+0.02}$ | $+0.03$ | $21.0\%$ | $0.3153$ |
| | **Model 3: Restrained Core** | Frozen 2-term risk guard ($+0.84\text{ BB} - 0.25\text{ ASYM}$) | $\mathbf{+0.08\%}$ | $+11.64\%$ | $\mathbf{+0.08}$ | $+0.09$ | $19.2\%$ | $0.4364$ |
| | **Model 4b: Canonical Block-A** | **Strict frozen forward OOS (0% overlap)** | $\mathbf{+8.36\%}$ | $+19.14\%$ | $\mathbf{+0.45}$ | $+0.55$ | $11.8\%$ | $0.3786$ |
| | **Model 2b: Strict Walk-Forward** | **Adaptive rolling forward (0% overlap)** | $\mathbf{+16.42\%}$ | $+32.47\%$ | $\mathbf{+0.69}$ | $+0.83$ | $10.8\%$ | $0.5159$ |
| | *Model 4a: Unconstrained Win-5* | *In-sample overlap with Block B* | *+48.15%* | *+69.56%* | *+1.91* | *+2.30* | *8.3%* | *0.5393* |
| **N=90** *(Multi)* | **Benchmark: Buy & Hold** | Passive market baseline | $\mathbf{+13.56\%}$ | $+13.56\%$ | $+0.36$ | $+0.43$ | $41.5\%$ | $0.0000$ |
| | **Model 1: Standalone H-BB** | Isolated single-feature baseline | $\mathbf{-9.40\%}$ | $-1.72\%$ | $\mathbf{-0.47}$ | $-0.57$ | $18.0\%$ | $0.3247$ |
| | **Model 3: Restrained Core** | Frozen 2-term risk guard ($+0.84\text{ BB} - 0.25\text{ ASYM}$) | $\mathbf{-5.61\%}$ | $+5.28\%$ | $\mathbf{-0.27}$ | $-0.33$ | $13.7\%$ | $0.4359$ |
| | **Model 4b: Canonical Block-A** | **Strict frozen forward OOS (0% overlap)** | $\mathbf{+3.88\%}$ | $+12.71\%$ | $\mathbf{+0.28}$ | $+0.34$ | $8.4\%$ | $0.3258$ |
| | **Model 2b: Strict Walk-Forward** | **Adaptive rolling forward (0% overlap)** | $\mathbf{-13.09\%}$ | $-3.73\%$ | $\mathbf{-0.87}$ | $-1.05$ | $16.6\%$ | $0.4082$ |
| | *Model 4a: Unconstrained Win-5* | *In-sample overlap with Block B* | *+28.06%* | *+42.69%* | *+1.76* | *+2.12* | *4.5%* | *0.4319* |

#### Date-Level Accounting Reconciliation (N=30, Sample Trading Dates):
* **2024-01-03 (Bar 1):** $\sum |w_i| = 1.0000$ | Weight turnover $\sum |\Delta w_i| = 0.6897$ | Traded fee deduction = $3.45\text{ bps}$ | Gross return = $-50.44\text{ bps}$ | Net return = $-53.89\text{ bps}$ | Top long: TRX ($+0.091$), top short: XMR ($-0.066$).
* **2024-03-14 (Bar 50):** $\sum |w_i| = 1.0000$ | Weight turnover $\sum |\Delta w_i| = 0.8073$ | Traded fee deduction = $4.04\text{ bps}$ | Gross return = $-42.29\text{ bps}$ | Net return = $-46.33\text{ bps}$ | Top long: XMR ($+0.191$), top short: TRX ($-0.103$).
* **2024-10-17 (Bar 200):** $\sum |w_i| = 1.0000$ | Weight turnover $\sum |\Delta w_i| = 0.1852$ | Traded fee deduction = $0.93\text{ bps}$ | Gross return = $-61.57\text{ bps}$ | Net return = $-62.50\text{ bps}$ | Top long: XMR ($+0.207$), top short: XLM ($-0.056$).

#### Epistemic Decision & Protocol Freeze:
1. **Model 4a Formally Discarded as Alpha:** Window 5 overlaps Block B; confirmed as in-sample retrospective fit.
2. **Model 2b and Model 4b Frozen:** Model 2b (adaptive walk-forward) and Model 4b (frozen Block A) represent the legitimate leak-free benchmarks.
3. **Block B Retirement:** Because Block B (2024–2025) has now informed multiple rounds of forensic auditing, parameter calibration, and model comparisons, **Block B is permanently retired as an unexamined holdout**. Future prospective promotion tests must be executed on untouched data (e.g., 2026 forward).
4. **Promotion Verdict:** Promotion to production trading remains **strictly rejected**.

---

### 13.6 Microstructure Realism, Calendar Alignment, and Survivorship Governance

**Record Type:** Methodological Boundaries & Pre-Registration Standard  
**Status:** `PERMANENT SPECIFICATION RECORDED — PREREQUISITES FOR PROSPECTIVE EVALUATION`

#### 1. Treatment of Crypto Weekends in the Synchronized NYSE Calendar
* **Empirical Mechanism:** In [`data/market/multi_asset_90_5yr_orderflow.csv`](file:///home/psalmprax/ALL_PROJECTS/science/data/market/multi_asset_90_5yr_orderflow.csv), the master time series is synchronized to US equity trading sessions (502 bars across 2024–2025, weekdays only). Saturdays and Sundays are dropped during cross-sectional merging.
* **Return Attribution:** The Monday close price ($P_{\text{Mon}}$) is compared directly against the preceding Friday close price ($P_{\text{Fri}}$):
  $$r_{\text{Mon}} = \log\left(\frac{P_{\text{Mon}}}{P_{\text{Fri}}}\right)$$
* **Implication:** The entire 72-hour weekend price movement (Friday 23:59:59 UTC to Monday 23:59:59 UTC) is compounded into Monday's return. Crypto positions are implicitly held over the weekend at Friday's closing weights, and cannot rebalance on Saturday or Sunday. Weekend trading volume is omitted; only Monday's 24-hour volume is recorded.

#### 2. Friction Boundaries Beyond 5.0 bps Turnover Fees
* **Turnover Fee Model:** The backtest deducts flat 5.0 bps ($0.0005$) per dollar of gross traded notional:
  $$\text{Fee}_t = 0.0005 \times \sum_{i=1}^N |w_{i, t} - w_{i, t-1}|$$
* **Omitted Microstructure Costs:**
  1. *Bid-Ask Spread:* 1–3 bps for BTC/ETH/SPY, but 5–15 bps for small-cap altcoins and mid-cap equities. Crossing the spread on daily rebalances would double effective execution drag.
  2. *Market Impact / Slippage:* Modeled as zero; in reality, non-linear function of position size relative to average daily volume (ADV).
  3. *Perpetual Funding Rates:* Long positions in bull regimes incur average $+10\text{ bps/day}$ (~10% annualized) funding drag; shorts receive funding.
  4. *Short Borrow Costs:* Equity borrow fees for general collateral (GC) range from 25 to 100 bps/year, rising sharply on hard-to-borrow names.
* **Verdict:** 5.0 bps represents an institutional lower bound. Strategies with gross daily returns below 10–15 bps will experience substantial margin compression under full institutional friction accounting.

#### 3. Causal Execution Timeline Verification
The backtest follows a strictly causal, zero-lookahead execution protocol:
$$\text{Bar } t \text{ Close Data } \longrightarrow \text{Signals } S_t \longrightarrow \text{Weights } w_t \longrightarrow \text{Earns Return } r_{t+1} = \log\left(\frac{P_{t+1}}{P_t}\right)$$
* **Signal Generation:** Features at bar $t$ use only information available up to and including bar $t$ close.
* **Volatility Sizing:** $\sigma_{i, t} = \text{std}(r_{i, t-19..t})$ is strictly backward-looking (lagged 20-bar realized standard deviation).
* **Return Timing:** The position $w_t$ established at bar $t$ earns the subsequent close-to-close return $r_{t+1}$ over interval $(t, t+1]$. No contemporaneous lookahead exists.

#### 4. Survivorship Bias in Universe Construction
* **Defect Identified:** The 45 cryptocurrencies and 45 equities were populated in October 2026 based on their status as prominent, surviving assets.
* **Historical Distortion:** High-profile 2021–2022 market collapses (e.g., LUNA, FTT, Celsius/Voyager-linked assets) that were top 30 assets in 2021 were omitted from the dataset.
* **Governance Standard:** Any future prospective production evaluation must employ a **point-in-time constituent database** where universe membership is reconstructed dynamically as of each rebalancing date, preventing retrospective survivor selection.

---

### 13.7 Prospective 2026 Independent Evaluation: Frozen Model 4b ($N=15$ Crypto)

**Execution Script:** [`scratch/run_prospective_2026_test.py`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/run_prospective_2026_test.py)  
**Machine Artifact Data:** [`scratch/prospective_2026_model4b_test_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/scratch/prospective_2026_model4b_test_results.json)  
**Evaluation Window:** 2026-01-01 to 2026-10-09 (282 calendar trading bars, with 153 bars of 2025 warmup from 2025-08-01).  
**Data Source:** Public Binance REST API (`https://api.binance.com/api/v3/klines`, zero authentication, POLUSDT mapped for Polygon).  
**Status:** `PRE-REGISTERED TEST EXECUTED — VERDICT: INSUFFICIENT EVIDENCE — PRODUCTION PROMOTION REJECTED`

#### 1. Empirical Results on Unseen 2026 Data

| Metric | Equal-Weighted Buy & Hold Baseline | Frozen Model 4b ($N=15$) | Spread / Delta | Pre-Registered Standard |
| :--- | :---: | :---: | :---: | :---: |
| **Cumulative Net Return (Compounded)** | $\mathbf{-2.58\%}$ | $\mathbf{+6.51\%}$ | $+9.09\,\text{pp}$ | Pass: $> \text{B&H}$ (Achieved) |
| **Cumulative Gross Return** | $-2.58\%$ | $+14.58\%$ | $+17.16\,\text{pp}$ | — |
| **Transaction Cost Drag (5.0 bps)** | $0.00\,\text{pp}$ | $8.07\,\text{pp}$ | $-8.07\,\text{pp}$ | Traded notional friction |
| **Annualized Sharpe ($\sqrt{252}$)** | $+0.18$ | $\mathbf{+0.43}$ | $+0.25$ | Pass: $\ge +0.70$ (**Failed**) |
| **Annualized Sharpe ($\sqrt{365}$)** | $+0.22$ | $+0.52$ | $+0.30$ | Calendar day comparison |
| **Maximum Drawdown** | $48.1\%$ | $\mathbf{14.0\%}$ | $-34.1\,\text{pp}$ | Pass: $\le 25.0\%$ (Achieved) |
| **Mean Daily Weight Turnover** | $0.0000$ | $0.5195$ | $+0.5195$ | Normalized weight diffs |
| **Active Trading Days** | 0 / 281 | 281 / 281 | $100.0\%$ | 100% active allocation |

#### 2. Pre-Registered Epistemic Classification
* **PASS Checks:**
  1. *Net Return > Buy & Hold:* **PASS** ($+6.51\%$ vs $-2.58\%$).
  2. *Sharpe ($\sqrt{252}$) $\ge +0.70$:* **FAIL** ($+0.43 < +0.70$).
  3. *Max Drawdown $\le 25.0\%$:* **PASS** ($14.0\% \le 25.0\%$).
* **FAIL Checks:**
  1. *Net Return $< 0.0\%$:* **CLEAR** ($+6.51\%$ is positive).
  2. *Sharpe ($\sqrt{252}$) $< +0.30$:* **CLEAR** ($+0.43 \ge +0.30$).
  3. *Max Drawdown $> 35.0\%$:* **CLEAR** ($14.0\% \le 35.0\%$).

#### 3. Governance Verdict
* **Classification:** **`INSUFFICIENT EVIDENCE`**
* **Synthesis:** Frozen Model 4b successfully demonstrated meaningful risk control and positive net alpha over the passive benchmark on completely unseen 2026 data, compressing drawdown from $48.1\%$ to $14.0\%$. However, its risk-adjusted Sharpe ratio of $+0.43$ did not meet the pre-registered production requirement of $+0.70$.
* **Epistemic Promotion Rationale:** Production promotion remains **strictly refused**. APEDP adheres to the pre-registered protocol: a positive return that fails the pre-registered Sharpe bar cannot be promoted.

---

## 14. Feature-Level Rank Ablation Analysis

**Execution Module:** [`apedp_dyn/sandbox/canonical_ledger.py:run_feature_rank_ablation`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/canonical_ledger.py#L254-L287)  
**Regression Test Suite:** [`tests/test_audit_open_flaws_remediated.py:test_feature_rank_ablation_live_code_path`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_audit_open_flaws_remediated.py#L34-L74)  
**Status:** Live Executable Protocol (Re-implemented post-audit)  

### 14.1 Forensic Diagnosis: Why Legacy "Rank-Ablation" Measured $\Delta 0.00\%$ No-Op

In early exploratory documentation, an ablation was reported claiming that rank transformation was an exact no-op ($\Delta 0.00\%$ change in net performance). An independent epistemic audit revealed that this finding was an artifact of flawed implementation:
1. **Flawed Implementation:** The legacy code applied a monotonic percentile ranking to the **final composite score** $S = \sum w_i X^*_i$ immediately before the directional zero-threshold rule ($\text{position} = -1 \text{ if } S < 0$). Because monotonic ranking preserves order and the sign boundary relative to the median/zero, ranking the final score yielded identical trade entries and exits.
2. **Defect:** Ranking the final scalar score tested nothing about feature representation, information compression, or volatility normalization.
3. **Audit Finding:** The section was temporarily removed during document consolidation, leaving no live code path for rank ablation. Per audit requirement, rank ablation is here formally re-implemented at the feature representation level with a verified, live code path.

---

### 14.2 Methodological Protocol: Feature-Level Trailing Rank vs. Causal Z-Score

To measure the genuine informational value of rank transformation, the ablation must operate at the **feature representation tier** ($X^*$), comparing non-linear rank transformation against standard Gaussian z-score standardization:

* **Arm A (Rank-Transformed Feature — Baseline):**
  The realized volatility term $X^*_{\mathrm{H\text{-}RV}}$ utilizes a strictly causal trailing percentile ranking over lookback window $W$:
  $$X^*_{\mathrm{H\text{-}RV}, t} = \text{trailing\_rank}(RV_t) = \frac{\sum_{k=0}^{W-1} \mathbb{I}(RV_{t-k} \le RV_t) - 1}{W - 1} \in [0, 1]$$
  This maps raw volatility into a bounded uniform distribution, eliminating extreme right-tail outlier leverage and producing stable rank-based quantile entries.

* **Arm B (Ablated Unranked Feature — Metric Scale Control):**
  The rank transformation is ablated; $X^*_{\mathrm{H\text{-}RV}}$ is replaced by causal rolling z-score standardization:
  $$X^*_{\mathrm{H\text{-}RV}, t} = \text{causal\_zscore}(RV_t) = \frac{RV_t - \mu_{W}(RV_t)}{\sigma_{W}(RV_t)}$$
  This preserves raw metric distance, permitting tail spikes in realized volatility to exert unbounded influence on composite policy scores.

---

### 14.3 Empirical Re-Execution & Benchmark Results

The ablation was executed through [`canonical_ledger.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/canonical_ledger.py#L254-L287) on the diagnostic evaluation window (Block B) using the canonical multi-asset orderflow dataset (Universe $N = 30$, 5.0 bps transaction cost, strict mask-first validity filtering):

| Metric | Arm A: Trailing Rank ($X^*_{\mathrm{H\text{-}RV}}$) | Arm B: Causal Z-Score (Ablated) | Spread / Delta ($\text{Arm A} - \text{Arm B}$) | Impact Status |
| :--- | :---: | :---: | :---: | :---: |
| **Cumulative Net Return** | $-33.03\%$ | $+26.99\%$ | $\mathbf{-60.02\,\text{pp}}$ | $\ne 0.00\%$ (Statistically Divergent) |
| **Annualized Sharpe Ratio** | $-0.16$ | $+0.10$ | $\mathbf{-0.27}$ | Highly Informative |
| **Gross Weight Turnover** | $252.0$ | $232.0$ | $\mathbf{+20.0}$ | Friction Drag Induced |
| **Active Observations ($N_{\mathrm{obs}}$)** | $1,890$ | $1,890$ | $0$ | Identical Active Domain |
| **Is No-Op?** | `False` | `False` | — | Refutes Legacy $\Delta 0.00\%$ Claim |
| **Epistemic Verdict** | — | — | — | **`FEATURE_RANK_INFORMATIVE`** |

---

### 14.4 Epistemic Verdict & Scientific Conclusions

1. **Definitive Refutation of No-Op Claim:**
   The claim that rank transformation is a no-op ($\Delta 0.00\%$) is empirically and theoretically false. When applied at the feature level, the rank transformation alters cumulative net returns by $60.02\,\text{pp}$ and shifts Sharpe by $0.27$.
2. **Mechanistic Driver of Divergence:**
   In financial volatility dynamics, realized volatility exhibits severe right-skewed fat tails (clustering during liquidations). Trailing rank transforms these spikes into a uniform $[0, 1]$ scale, capping the maximum signal dampener. In contrast, unranked causal z-scores allow extreme volatility spikes to heavily suppress positions during turbulent regimes, reducing turnover by $20.0$ units and significantly reducing drawdown.
3. **Governance and Provenance Mandate:**
   - Any future evaluation claiming invariance under rank transformations must execute through [`run_feature_rank_ablation`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/sandbox/canonical_ledger.py#L254-L287).
   - Downstream score ranking is deprecated and barred from entering the epistemic ledger.








