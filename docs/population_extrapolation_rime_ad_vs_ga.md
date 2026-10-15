# Multi-Population Extrapolation Benchmark: RIME-AD vs GA/GP
## Cross-Distribution Generalization, Parameter Shifts, and Noise Distractor Challenge

---

### Benchmark Protocol & Data Provenance
- **Ground Truth Mechanism:** $y[t] = 2.0 \cdot x_1[t-1] x_2[t-1] + 3.0 \cdot x_3[t-2] + 10.0 + \epsilon$
- **Populations Evaluated:**
  1. **Pop A-Train ($N=400$):** Discovery partition (seed 500, Gaussian inputs).
  2. **Pop A-Test ($N=200$):** In-distribution holdout slice.
  3. **Pop B ($N=600$):** Fresh realization from the same generating distribution (seed 501).
  4. **Pop C ($N=600$):** True physical parameter shift ($c_1 = 2.0 \to 2.5$, $c_2 = 3.0 \to 2.0$, seed 502).
  5. **Pop D ($N=600$):** Out-of-Distribution heavy-tailed inputs ($x \sim \text{Laplace}(0, 1.5)$).
  6. **Distractor Challenge:** Discovery under 7 irrelevant Gaussian noise channels ($x_4 \dots x_{10}$).
- **Machine Result Artifact:** [`benchmarks/population_extrapolation_results.json`](file:///home/psalmprax/ALL_PROJECTS/science/benchmarks/population_extrapolation_results.json).

---

### 1. Cross-Population Extrapolation Generalization Table

| Model / Search Architecture | In-Sample Fit (Pop A-Train $R^2$) | In-Dist Holdout (Pop A-Test $R^2$) | Fresh Seed (Pop B $R^2$) | Parameter Shift (Pop C $R^2$) | Heavy-Tail OOD (Pop D $R^2$) | Execution Time (ms) | Epistemic Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GA-Small** (Linear + Lags) | `0.6809` | `0.6450` | `0.7006` | `0.4185` | `0.3212` | `1354.7 ms` | **REFUSED** (Missing Interaction) |
| **GA-Large** (Polynomial + Cross Basis) | `0.9998` | `0.9998` | `0.9998` | `0.8961` | `1.0000` | `2137.7 ms` | **PASSED** (Pre-Planted Basis) |
| **RIME-AD** (Native Rust Additive Discovery) | `0.9998` | `0.9998` | `0.9998` | `0.8961` | `1.0000` | `8.7 ms` | **PASSED** (Autonomous Recovery) |

---

### 2. Discovered Equations & Structural Recoveries

- **RIME-AD Discovered Formula:**
  `+1 * (((2.995647 * x3[t-2]) + (1.996585 * (x1[t-1] * x2[t-1])))) +9.99877`
  *Active Inputs Used:* `['x1', 'x2', 'x3']` (Identified true parents $x_1, x_2, x_3$).
- **GA-Large Selected Features:**
  `['1', 'x1', 'x1[t-1]', 'x2', 'x2[t-1]', 'x3', 'x3[t-1]', 'x3[t-2]', 'x1^2', 'x1[t-1]^2', 'x1[t-2]^2', 'x2^2', 'x2[t-1]^2', 'x2[t-2]^2', 'x3^2', 'x1[t-1]*x2[t-1]', 'x1[t-2]*x2[t-2]', 'x1*x2', 'x1*x3', 'x2*x3']`

---

### 3. Distractor Noise Feature Challenge (7 Noise Channels: $x_4 \dots x_{10}$)

| Metric / Property | Measured Value | Standard Ground Truth | Evaluation |
| :--- | :---: | :---: | :---: |
| **Distractor Search Time** | `26.2 ms` | $< 2500 \text{ ms}$ | **Passed** |
| **Dev Partition Fit ($R^2$)** | `0.999969` | $> 0.9990$ | **Passed** |
| **Admitted Signal Inputs** | `['x1', 'x2', 'x3']` | `['x1', 'x2', 'x3']` | **Exact** |
| **Admitted Distractor Channels** | `[]` | `[]` (Zero noise channels) | **0% False Discovery** |

---

### 4. Key Analytical Takeaways
1. **Parity on In-Distribution & Fresh Realizations:** Both RIME-AD and GA-Large achieve near-perfect recovery ($R^2 = 0.9998$ vs $0.9998$) on Population A and Population B.
2. **Graceful Degradation on Parameter Shift:** When physical parameters shift ($c_1: 2.0 \to 2.5, c_2: 3.0 \to 2.0$), both engines transfer gracefully without catastrophic blowup ($R^2 = 0.8961$).
3. **Zero False Discovery on Distractors:** Under 7 pure noise distractor channels, RIME-AD's complexity-penalized additive search (`gamma=1.0`) strictly filters out all 7 noise features, admitting only the true generating variables $[x_1, x_2, x_3]$.
4. **Execution Speed Advantage:** RIME-AD evaluates candidate mechanisms in native Rust in **`8.7 ms`**, whereas multi-generation GA requires thousands of evaluations taking **`2137.7 ms`** (~246x speedup).