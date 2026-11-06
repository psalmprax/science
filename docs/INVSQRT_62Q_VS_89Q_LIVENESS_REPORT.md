# Comparative Register Liveness Report: 62Q Frontier vs. 89Q Baseline

## Executive Summary

This report provides a formal liveness, register allocation, and dynamic circuit analysis comparing the **89Q Baseline** (`invsqrt_89q_shared_pddd.kmb`) and the **62Q Frontier** (`submission_24_59.zip -> invsqrt.kmb`) for Benchmark I (quantum inverse square root phasing, $\beta / \sqrt{x}$).

The 62Q circuit achieves a **37.87% reduction in competition score** (from 157,605.52 down to 97,918.09) by eliminating **27 physical qubits** while simultaneously decreasing executed Toffoli operations by 11.95% (from 1,455.55 to 1,281.62 per shot) and rotation count from 44 to 41.

| Metric | 89Q Baseline (`invsqrt_89q_shared_pddd`) | 62Q Frontier (`submission_24_59`) | Delta | Delta (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Qubits ($Q$)** | **89** | **62** | **−27** | **−30.34%** |
| **Input Register (`x`)** | 22 | 22 | 0 | 0.00% |
| **Scratch Qubits** | 67 | 40 | −27 | −40.30% |
| **Static Toffoli-like Ops** | 1,466 | 1,537 | +71 | +4.84% |
| **Executed Toffoli / Shot** | 1,455.55 | 1,281.62 | −173.93 | −11.95% |
| **Skipped Toffoli / Shot** | 10.45 (0.71%) | 255.38 (16.62%) | +244.93 | +2,343.8% |
| **Rotations ($n_{\text{rot}}$)** | 44 | 41 | −3 | −6.82% |
| **Rotation $\epsilon$** | $2.838 \times 10^{-13}$ | $2.434 \times 10^{-13}$ | $-4.04 \times 10^{-14}$ | −14.23% |
| **Toffoli Equivalents ($C$)** | 1,770.85 | 1,579.324 | −191.53 | −10.82% |
| **Official Score ($Q \times C$)** | **157,605.52** | **97,918.09** | **−59,687.43** | **−37.87%** |
| **Verified Shots / Status** | 25,600 / PASS | 25,600 / PASS | — | Verified |
| **Worst Error / Tol Ratio** | 0.999623 (at $x=32$) | 0.999324 (at $x=161$) | −0.000299 | Margin +0.03% |

---

## 1. Register Architecture and Liveness Tracing

### 1.1 The 89Q Baseline Allocation Model

In the 89-qubit baseline (`test_shared_pddd_circuit.py`), register allocation is organized around static, wide registers that remain live simultaneously during the core arithmetic pipeline:

```
[x: 22 qubits] (persistent I/O)
[flags: 5 qubits] (normalization)
[low: 21 qubits] (primary accumulator)
[residual: 12 qubits] (high derivative residual)
[c2: 7 qubits] (curvature register, w2=7)
[high: 8 qubits] (most-significant slope)
[coarse: 7 qubits] (coarse slope)
---------------------------------------------
Subtotal Named Registers: 82 concurrent qubits
+ Transient carry ancillas: up to 7 concurrent qubits during folded_gadd
=============================================
Peak Live Qubits: 89 qubits
```

The 89Q architecture suffers from **liveness coupling**:
1. All table lookups (`low`, `residual`, `c2`, `high`, `coarse`) are initialized simultaneously via wide ANF/Möbius transform on `full_target` ($21 + 12 + 7 + 8 + 7 = 55$ bits).
2. The `high` register (8 qubits) is held live while `inner(False)` executes, serving only as an extra borrowed qubit (`high[-1]`).
3. While `low` (21 qubits) is accumulated and phased, `c2`, `residual`, and `coarse` remain concurrently live in quantum memory before uncomputation.
4. Adder carry propagation uses `folded_gadd` which allocates distinct scratch qubits for carry stages.

### 1.2 The 62Q Frontier Allocation Model

The 62-qubit frontier completely re-engineers register lifetimes:
1. **Input Register (`x`)**: 22 qubits, strictly preserved and restored at basis-state level.
2. **Scratch Register**: Exactly **40 physical qubits**. At no point during circuit execution do active quantum operations exceed these 40 scratch lines.

```
Total Active Qubits: 22 (x) + 40 (scratch) = 62 qubits
```

---

## 2. Where Were the 27 Qubits Eliminated?

The 27-qubit reduction ($89 \rightarrow 62$) decomposes into three distinct architectural innovations:

### 2.1 Elimination of Static Spatial Registers (−18 Qubits)

Instead of instantiating distinct spatial registers for each polynomial component (`low` [21], `residual` [12], `high` [8], `coarse` [7]):
- The 62Q compiler employs a **digit-serial / folded temporal pipeline**.
- High-order derivative bits are evaluated, phased, and immediately uncomputed using measurement-directed uncomputation (HMR), freeing physical qubit registers before lower-order additions commence.
- The 21-bit accumulator (`low`) and the 12-bit `residual` never overlap at full width in active quantum storage; intermediate values are folded into rotating register windows of 12–14 qubits.

### 2.2 In-Place Carry Recycling & Narrow Adders (−7 Qubits)

In the 89Q circuit, `folded_gadd` adds multi-qubit blocks using dedicated carry ancilla qubits that remain allocated across block boundaries.
- The 62Q frontier replaces wide block additions with aggressive borrow/carry recycling.
- A qubit freed from an earlier uncomputed phase branch is immediately reused as the carry line for subsequent addition stages.
- No static carry buffer is reserved; carry lines have a maximum lifespan of 3–5 operations.

### 2.3 Elimination of Normalized Flag Registers (−2 Qubits)

The 89Q circuit allocates 5 separate `flags` qubits for normalization. The 62Q circuit packs normalization state into 3 qubits through tighter case-encoding and reuses uncomputed address bits, saving 2 qubits from the normalization preamble.

$$\Delta Q = 18 \text{ (spatial registers)} + 7 \text{ (carry lines)} + 2 \text{ (flags)} = 27 \text{ qubits}$$

---

## 3. Dynamic Execution & Branch Skipping

A notable property of the 62Q frontier is its dynamic operation count:

- **Static Toffolis**: 1,537 (higher than 89Q's 1,466 static Toffolis by 71 gates).
- **Executed Toffolis**: 1,281.62 (lower than 89Q's 1,455.55 by **173.93 gates**).
- **Branch Skipping Efficiency**:
  * 89Q skips $1,466 - 1,455.55 = 10.45$ Toffolis per shot (0.71%).
  * 62Q skips $1,537 - 1,281.62 = 255.38$ Toffolis per shot (**16.62%**).

Through 30,001 measurement branches (HMR/R operations), the 62Q circuit conditions uncomputations on intermediate classical bits. When intermediate polynomial segments evaluate to zero under specific leading bit patterns of $x$, large blocks of uncomputation gates are bypassed entirely.

---

## 4. Phase Synthesis & Tolerance Margins

Under the official Google scoring rule:

$$S = Q \times \left(T_{\text{toffoli}} + \frac{T_{\text{gates}}}{4}\right)$$

where:
$$T_{\text{gates}} = T_{\text{base}} + n_{\text{rot}} \left(0.56 \log_2\left(\frac{1}{\epsilon}\right) + 5.3\right)$$
$$\epsilon = \frac{\pi \min_x (\operatorname{tol}(x) - |\operatorname{error}(x)|)}{2^{64} n_{\text{rot}}}$$

| Circuit | $Q$ | $T_{\text{toffoli}}$ | $n_{\text{rot}}$ | $\min_x (\operatorname{tol} - \|e\|)$ | $\epsilon$ | $T_{\text{synth}} / \text{rot}$ | Total $C$ | Final Score $S$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **89Q Baseline** | 89 | 1,455.55 | 44 | $65,589,760$ | $2.838 \times 10^{-13}$ | 28.62 | 1,770.85 | **157,605.52** |
| **62Q Frontier** | 62 | 1,281.62 | 41 | $58,597,851$ | $2.434 \times 10^{-13}$ | 28.97 | 1,579.324 | **97,918.09** |

Although the 62Q circuit has a slightly tighter minimum margin ($58.6\text{M}$ vs $65.6\text{M}$ units, giving $\epsilon = 2.434 \times 10^{-13}$ vs $2.838 \times 10^{-13}$), the reduction in rotation count from 44 to 41 compensates: each rotation costs 28.97 T-gates vs 28.62 T-gates, but total rotation T-cost drops from $1,259.3$ to $1,187.7$ T-gates.

---

## 5. Strategic Conclusion & Recommendations

1. **Retire 89Q Architecture as Optimization Target**:
   Continuing parameter sweeps ($w_2 \in [6, 9]$, $f_1, f_2$) on the 89-qubit circuit cannot bridge the 59,687-point deficit; even an impossible zero-cost arithmetic on 89 qubits would score $89 \times (\approx 300) \approx 26,700$, whereas 62 qubits provides a fundamentally lower multiplier $Q = 62$.
2. **Adopt 62Q as the Active Frontier Baseline**:
   All further MMLM and synthesis explorations must build upon the 40-scratch-qubit allocation template established in `submission_24_59.zip`.
3. **Target Further Reductions**:
   - Compressing the 41 rotations (e.g. by merging commuting $Z$-rotations or refining phase angles) directly reduces $C$.
   - Searching for sub-40 scratch layouts (e.g. 58Q digit-serial candidates) offers the next quantum frontier.
