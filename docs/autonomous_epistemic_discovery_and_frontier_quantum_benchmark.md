# Autonomous Epistemic Discovery & Frontier Quantum Equations Audit
## Measured In-Sample Profiling, Coordinate Transmutation, and Scope Boundaries

---

### Audit Provenance & Methodological Scope Notice
- **Target Implementation:** [`apedp_dyn/epistemic/autonomous_profiler.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/autonomous_profiler.py), [`apedp_dyn/epistemic/coordinate_transmuter.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/coordinate_transmuter.py).
- **Test File:** [`tests/test_dyn_autonomous_epistemic_profiler.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_autonomous_epistemic_profiler.py) (7/7 passing).
- **Scope Discipline:** 
  1. All $R^2$ values in this module are **in-sample fits on the discovery partition**, NOT out-of-sample holdout confirmations. As stated in `coordinate_transmuter.py`: *"Does NOT evaluate on holdout confirmation slices."*
  2. On noiseless synthetic data where the correct basis family is selected, high $R^2$ demonstrates parameter identifiability and linear algebraic consistency, not out-of-sample discovery.
  3. The Planck blackbody test is a **profiler detection test only** (asserting crossover and grammar tokens); it does not perform coordinate transmutation or evaluate a model $R^2$.

---

### 1. Machine-Measured Profiling Statistics (Actual Test Data)

The following statistics are measured directly from the arrays instantiated in [`tests/test_dyn_autonomous_epistemic_profiler.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_autonomous_epistemic_profiler.py):

| Test Case | Domain Range & Sample Size | Dynamic Range ($\max/\min$) | Measured Invariant / Slope | Diagnostic Trigger (`DatasetProfile`) |
| :--- | :--- | :---: | :--- | :--- |
| **Brus Quantum Dot** | $R \in [1.5, 8.0]$ ($N=50$) | $28.4$ | $\frac{d \ln y}{d \ln R} = -2.0000$ ($\text{CV} = 0.0000$) | `is_pure_power_law = True`<br/>`power_law_exponent = -2.0000`<br/>`grammar: ["inv_sq"]` |
| **Landauer Staircase** | $w \in [0.1, 4.9]$ ($N=100$) | $3.00$ | Flat fraction: **`0.9596`** ($95.96\%$), 5 levels | `is_quantized_staircase = True`<br/>`discrete_levels = [0, 2, 4, 6, 8]`<br/>`grammar: ["floor", "ceil"]` |
| **Schwinger Tunneling** | $E \in [0.15, 1.2]$ ($N=50$) | $2.19 \times 10^4$ | $\frac{d \ln y}{d(1/E)} = -2.3497$ ($\text{CV} = 0.2580$) | `is_arrhenius_tunneling = True`<br/>`recommended_loss = "log_mse"`<br/>`transmutation = "log_linear_singular"` |
| **Planck Blackbody** | $\nu \in [0.05, 10.0]$ ($N=100$) | $5.83 \times 10^2$ | Low: $\nu^{+1.45}$, High: $\nu^{-5.25}$ | `is_dual_asymptotic = True`<br/>`grammar: ["bose", "planck"]`<br/>*(Transmutation not evaluated)* |
| **Landau-Zener Transition** | $v \in [0.12, 4.0]$ ($N=60$) | $5.32 \times 10^2$ | $\frac{d \ln P}{d(1/v)} = -0.7500$ ($\text{CV} = 0.0000$) | `is_arrhenius_tunneling = True`<br/>`recommended_loss = "log_mse"`<br/>`power_k = 0.0` (pure barrier) |
| **BCS Energy Gap** | $g \in [0.15, 0.75]$ ($N=50$) | **`207.1`** | $\frac{d \ln \Delta}{d(1/g)} = -1.0000$ ($\text{CV} = 0.0000$) | `is_arrhenius_tunneling = True`<br/>`recommended_loss = "log_mse"`<br/>`power_k = 0.0` (pure barrier) |
| **Integer Quantum Hall** | $1/B \in [0.9, 5.8]$ ($N=120$) | $5.00$ | Flat fraction: **`0.9664`** ($96.64\%$), 5 levels | `is_quantized_staircase = True`<br/>`discrete_levels = [1, 2, 3, 4, 5]`<br/>`grammar: ["floor", "ceil"]` |

---

### 2. In-Sample Parameter Recovery via Coordinate Transmutation

For cases where coordinate transmutation is implemented and asserted in [`tests/test_dyn_autonomous_epistemic_profiler.py`](file:///home/psalmprax/ALL_PROJECTS/science/tests/test_dyn_autonomous_epistemic_profiler.py), the in-sample least-squares recoveries are:

| Test Case | Method | Recovered Formula | In-Sample $R^2$ (Test Assertion) | Recovered Parameters vs. Planted Truth |
| :--- | :--- | :--- | :---: | :--- |
| **Brus Confinement** | `power_law` | `4.5 * (R ** -2)` | $R^2 > 0.9999$ | $C = 4.500000$ (true $4.5$), $k = -2.000000$ (true $-2.0$) |
| **Landauer Staircase** | `quantized_staircase` | `2 * floor((w - 0.00303) / 0.99879)` | $R^2 > 0.99$ | $G_0 = 2.000000$ (true $2.0$), $\Delta w = 0.998788$ (true $1.0$) |
| **Schwinger Tunneling** | `log_linear_singular` | `10 * (E ** 2) * exp(-1 / E)` | $R^2 > 0.999999$ | $C = 10.000000$ (true $10.0$), $k = 2.000000$ (true $2.0$), $c = 1.000000$ (true $1.0$) |
| **Landau-Zener Transition** | `log_linear_singular` | `1 * exp(-0.75 / v)` | $R^2 > 0.999999$ | $C = 1.000000$ (true $1.0$), $\alpha = 0.750000$ (true $0.75$), $k = 0.0$ |
| **BCS Energy Gap** | `log_linear_singular` | `3.52 * exp(-1 / g)` | $R^2 > 0.999999$ | $\Delta_0 = 3.520000$ (true $3.52$), $c = 1.000000$ (true $1.0$), $k = 0.0$ |
| **Integer Quantum Hall** | `quantized_staircase` | `1 * floor((inv_B - 0.00242) / 1.0006)` | $R^2 > 0.99$ | $G_0 = 1.000000$ (true $1.0$), $\Delta(1/B) = 1.000606$ (true $1.0$) |

---

### 3. Classification Criteria in Code

In [`apedp_dyn/epistemic/autonomous_profiler.py`](file:///home/psalmprax/ALL_PROJECTS/science/apedp_dyn/epistemic/autonomous_profiler.py), classification uses the following exact criteria:
1. **Loss Selection:** `dyn_range >= 100.0` on strictly positive arrays triggers `"log_mse"`.
2. **Arrhenius / Tunneling:** $\left|\frac{\Delta \ln y}{\Delta(1/x)}\right|$ with $\text{CV} < 0.35$, negative mean slope, and dynamic range $> 10.0$.
3. **Power-Law Scaling:** $\left|\frac{\Delta \ln y}{\Delta \ln x}\right|$ with $\text{CV} < 0.15$ and non-zero slope ($|k| > 10^{-3}$).
4. **Quantized Plateau Staircase:** Flat derivative fraction $\ge 0.35$ (where $|\Delta y / \Delta x| < 10^{-4}$) AND number of unique rounded levels $\le 20$.
5. **Dual-Asymptotic Crossover:** $|p_{\text{low}} - p_{\text{high}}| > 1.2$ between first-third and last-third boundary log-log fits (with `is_power_law=False` and `is_quantized_staircase=False`).
