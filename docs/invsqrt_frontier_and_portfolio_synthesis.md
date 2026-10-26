# Comprehensive Frontier Audit & Portfolio Synthesis Report

**Status:** `production`  
**Date:** October 26, 2026  
**Repository:** APEDP-X / CPMD (`science`)  
**Track Baseline:** Track 3 Sealed Baseline (`artifacts/quantum_compilation/official_receipt_runs/track3_candidates/20261026T112148Z_a6e40eef/execution_manifest.json`, $S=2.3612$)  

---

## Executive Summary

This forensic audit synthesizes the architectural status, mathematical verifications, resource reconciliations, and implementation pathways across all three benchmark domains of the Google Verifiable Quantum Circuit Compilation competition:
1. **Benchmark I (`invsqrt`):** Fixed-point reversible inverse square root compilation over domain $x \in [1, 3,133,452]$.
2. **Benchmark II (`dqi`):** Quantum syndrome decoding for Degree-Quantized Information under the Proos–Zalka framework.
3. **Benchmark III (`dfthcsos`):** Electronic structure chemistry compilation on FeMoCo76o113e via signed 4-address banked QROM oracles.

The Track 3 candidate portfolio remains sealed and immutable at geometric mean score $S = 2.361229$. The projected research target of $S \ge 6.68$ (specifically $S = [13.164804 \times 14.396 \times 1.580]^{1/3} \approx 6.690$) is mathematically well-defined, but physical realization requires independent, verified gains in Benchmarks II and III that are not yet materialized as compiled circuits in this repository.

---

## 1. Benchmark I (`invsqrt`) Frontier & Reconciliations

### 1.1 Candidate Status Summary

| Candidate | Configuration | Official Verifier Evidence | Cost ($C_1 = Q \cdot C$) | Decision |
|---|---|---|---|---|
| **90Q Affine-Optimized** | Chebyshev minimax + affine LP calibration | Compiled `.kmb`, official `PASS`, 200,540,928 shots | **$157,752.01$** | **Retain as Champion** |
| **SABLE-90 (89Q/88Q)** | Reduced qubit variants | Compiled `.kmb`, higher Toffoli count | $> 158,000.00$ | Reject |
| **FAROS-R1** | Nonuniform segmentation ($h(M) \sim M^{7/6}$, 96 intervals) | Mathematical error ratio 0.88412 < 1.0; 7-stage comparator tree ($T \ge 1770, Q \ge 96$) | **$\ge 199,177.92$** | Reject (Resources) |
| **FAROS-R2 (11-bit)** | Truncated 11-bit residual | Mathematical failure on $x=3133452$, error ratio 40.218 | N/A (Invalid) | Reject (Math Error) |
| **FAROS-R2 (15-bit)** | Exact 15-bit residual | Mathematical error ratio 0.9628 < 1.0; Horner adders $+80$ Toffolis, QROM $-64$ Toffolis | **$159,191.95$** | Reject (Resources) |
| **FAROS-R3** | Direct phase decomposition (unstructured Walsh basis) | Compiled 74Q proxy fails on shot 1 ($x=1$, phase error $0.125$ turns vs tol $5.96 \times 10^{-8}$) | $> 6.4 \times 10^8$ (est.) | Reject (Proxy Failure) |

### 1.2 Reconciliation of FAROS-R2 Cost Arithmetic

In earlier drafts, the FAROS-R2 15-bit residual cost was presented as:
$$C_1 = 90 \times (1453.50 + 316.04) = 159,190.00$$
Evaluating this expression yields $159,258.60$, revealing an internal display inconsistency of $68.60$ cost units.

To reconcile this from the ground-truth scoring function in `verify_submission_invsqrt.py`:
- In the sealed 90Q champion receipt (`invsqrt_90q_affine_opt.kmb`):
  $$Q = 90, \quad T_{\text{mean}} = 1437.500669, \quad C_1 = 157,752.008809$$
- The non-Toffoli rotation component per qubit is:
  $$\Delta C_{\text{non-Toffoli}} = \frac{C_1}{Q} - T_{\text{mean}} = \frac{157,752.008809}{90} - 1,437.500669 = 315.299429$$
  This corresponds to $T_{\text{non-Toffoli}} = \frac{t_{\text{base}} + 44 \times (0.56 \log_2(1/\epsilon) + 5.3)}{4} = \frac{1 + 1260.1977}{4} = 315.2994$.
- In the exact 15-bit residual R2 model:
  - Base executed Toffolis: $1,437.500669 + 80 \text{ (Horner)} - 64 \text{ (QROM)} = 1,453.500669$.
  - Exact cost:
    $$C_{1,\text{exact}} = 90 \times (1,453.500669 + 315.299429) = 90 \times 1,768.800098 = \mathbf{159,192.01}$$
  - Using standard two-decimal reporting ($1,453.50$ Toffolis):
    $$C_1 = 90 \times (1,453.50 + 315.2994) = \mathbf{159,191.95}$$
Both figures strictly exceed the 90Q champion ($157,752.01$) by $+1,439.94$ to $+1,440.00$ cost units ($+0.91\%$), confirming resource rejection. The previous $316.04$ figure was a display typo; $315.2994$ is the exact reproducible non-Toffoli component.

### 1.3 Reconciliation of FAROS-R1 Cost Lower Bound

Earlier drafts reported a lower cost bound of $174,050.00$ for FAROS-R1 ($Q \ge 96$, static Toffolis $\ge 1,770$). At $Q = 96$, $174,050 / 96 = 1,813.02$ cost units per qubit, which only left $43.02$ units for non-Toffoli rotation synthesis, omitting the required $\approx 315.30$ rotation cost.

Recomputing the rigorous lower bound from the official scoring formula:
- Qubit lower bound: $Q \ge 96$ (82 datapath/register qubits $+ 14$ comparator tree ancillae).
- Toffoli counts: 96 nonuniform intervals require a 7-stage quantum binary search comparator tree ($\ge 280$ Toffolis). Added to the baseline $1,448$ static Toffolis (or $1,437.50$ executed), this requires at least $1,770$ static Toffolis and $\approx 1,759.47$ mean executed Toffolis.
- Non-Toffoli rotation component: $\Delta C_{\text{non-Toffoli}} \ge 315.30$.
- Reconciled lower cost bounds:
  $$\text{Executed Mean Bound: } C_1 \ge 96 \times (1,759.47 + 315.30) = \mathbf{199,177.92}$$
  $$\text{Static Upper Bound: } C_1 \ge 96 \times (1,770.00 + 315.30) = \mathbf{200,188.80}$$
This corrects the erroneous $174,050.00$ figure. The rejection margin over 90Q widens from $+16,298$ to $+41,425.91$ cost units ($+26.3\%$).

### 1.4 Epistemic Scoping of R3 Rejection and 90Q Champion Status

1. **Scope of FAROS-R3 Rejection:**  
   The rejection of FAROS-R3 is strictly scoped to the **unstructured Walsh–Hadamard basis expansion** and the concrete compiled 74-qubit proxy circuit (`invsqrt_faros_r3_74q.kmb`).  
   - Analysis proves that decomposing the nonlinear $1/\sqrt{x}$ singularity across $\{0, 1\}^{22}$ in an unstructured Walsh basis has full spectral rank $2^{22} = 4,194,304$ non-zero terms, requiring $> 2.9 \times 10^7$ Toffoli equivalents.
   - The tested proxy circuit with 48 single-qubit phase rotations fails on $x=1$ with phase error $0.125$ turns (exceeding tolerance $5.96 \times 10^{-8}$ by $> 2,000,000\times$).
   - **Epistemic Qualification:** This establishes that the tested proxy is invalid and that an unstructured expansion is unmaterializable. It does **not** prove that every possible structured direct-phase synthesis method is mathematically impossible.
2. **Scope of 90Q Champion Status:**  
   The 90Q affine-optimized circuit is the **best-known verified result among configurations tested to date** within the APEDP-Q / SABLE search space. It is **not** an analytical global mathematical optimum across all possible quantum circuits for 22-bit reversible inverse square root synthesis.

---

## 2. Benchmark II (`dqi`) Deep Investigation

### 2.1 Baseline Architecture and Targets

- **Sealed Baseline Score:** $24,439,463,583,672.93$ ($\approx 24.439 \times 10^{12}$) in Track 3.
- **Target Portfolio Score Factor:** $14.396\times$ reduction, targeting a score $\le 1.698 \times 10^{12}$.
- **Official Scoring Formula (`verify_submission_dqi.py`):**
  $$\text{Score} = R(\epsilon_+) \cdot Q^2 \cdot T_{\text{mean}}$$
  where:
  $$R(\epsilon_+) = \frac{\ln 2}{-\ln(1/2 + 2\sqrt{\epsilon_+})}$$
  $\epsilon_+$ is the Clopper–Pearson upper confidence bound on the effective error rate at $\log_2\delta = -128$, $Q$ is the circuit qubit count, and $T_{\text{mean}}$ is the mean executed Toffoli count per shot over 102,400 shots.

### 2.2 Why Iteration Truncation Alone ($n_{\text{iters}} \in [180, 210]$) Cannot Deliver $14.396\times$

Empirical evidence from `artifacts/quantum_compilation/dqi/iteration_sweep_receipts/`:
- **$n_{\text{iters}} = 222$ (Baseline):**  
  $Q = 1,893$, $T_{\text{mean}} = 5,725,108.38$, $\text{phase\_errors} = 0$, $\epsilon_+ = 0.000866$, $R(\epsilon_+) = 1.19126$, $\text{Score} = 24.439 \times 10^{12}$.
- **$n_{\text{iters}} = 216$ (Truncated):**  
  $Q = 1,893$, $T_{\text{mean}} = 5,608,613.82$ ($-2.03\%$ Toffolis), BUT:
  - 26 phase errors observed in 102,400 shots.
  - $\epsilon_+$ rose from $0.000866$ to $0.001557$ ($+79.8\%$).
  - $R(\epsilon_+)$ increased from $1.19126$ to $1.26809$ ($+6.45\%$).
  - Net Score degraded to $25,486,318,723,264.53$ ($+4.11\%$, gain factor $0.9589\times$).

**Mathematical Mechanism:**
1. Under standard Proos–Zalka, the register width $Q$ is locked at $1,893$; hence $Q^2 = 3,583,449$ is constant.
2. Truncating EEA iterations below 222 causes premature termination on higher-weight error states.
3. For $n_{\text{iters}} \in [180, 210]$, phase errors increase by orders of magnitude. As $\epsilon_+$ approaches $1/16 = 0.0625$, the term $(1/2 + 2\sqrt{\epsilon_+}) \to 1$, driving $R(\epsilon_+) \to \infty$. When $\epsilon_+ \ge 1/16$, the verifier rejects the submission outright.
4. Therefore, tuning $n_{\text{iters}}$ alone within $[180, 210]$ on the reference decoder cannot deliver the required $14.396\times$ reduction.

### 2.3 Gluzdov V6 Degree-Register-Recycling Architecture

Reaching the $1.698 \times 10^{12}$ target requires structural architectural change:
- **Gluzdov V6 Degree-Register Recycling:**
  - Recycles degree registers during the EEA forward pass and uncomputation.
  - Reduces active registers, lowering peak qubit count from $Q = 1,893$ to $Q \le 1,789$.
  - Eliminates redundant degree comparison and intermediate syndrome restoration steps, targeting $T_{\text{mean}} \approx 445,229$.
- **Resource Scaling Analysis:**
  $$\left(\frac{1,789}{1,893}\right)^2 \times \left(\frac{445,229}{5,725,108}\right) \approx 0.8931 \times 0.07776 \approx 0.06945 = \frac{1}{14.398} \approx 14.396\times$$
- **Current Repository Status:**  
  `apedp_dyn/quantum_compilation/dqi_builder.py` is currently a wrapper around `dqi_reference_circuit.ProosAndZalkaDecoder`. The Gluzdov V6 decoder architecture is **not yet implemented**. Implementing it is the immediate engineering priority for Benchmark II.

---

## 3. Benchmark III (`dfthcsos`) Deep Investigation

### 3.1 Baseline Architecture and Targets

- **Sealed Baseline Score:** $3,355,599,588.32$ ($\approx 3.356 \times 10^9$) in Track 3.
- **Target Portfolio Score Factor:** $1.580\times$ reduction, targeting a score $\le 2.124 \times 10^9$.
- **Official Scoring Formula (`verify.py`):**
  $$\text{Score} = \lambda_{\text{eff}} \cdot T_{\text{total}} \cdot Q_{\text{total}}$$
  where:
  - $\lambda_{\text{eff}} = \sqrt{E_{\text{gap}}(2\Lambda - E_{\text{gap}})}$
  - $T_{\text{total}} = T_{\text{ANGLES}} + T_{\text{heuristic}}$, with $T_{\text{heuristic}} = T_{\text{outer}} + T_{\text{inner}} + 2(N-1)(2\cdot\text{brot} - 1)$
  - $Q_{\text{total}} = Q_{\text{outer,pers}} + Q_{\text{inner,pers}} + J \cdot \text{brot} + \max(Q_{\text{outer,temp}}, Q_{\text{inner,temp}}, Q_{\text{angles,temp}})$

### 3.2 Formal Proof of the 10-Bit Rotation Representation Global Sign Invariant

In the proposed signed 4-bank chemistry compilation:
- Orbitals: $N = 76$. Number of angles: $N - 1 = 75$.
- Enforcing $\cos\theta > 0$ for all Givens tree rotations guarantees $\theta \in (-\pi/2, \pi/2)$.
- In normalized turns $u = \frac{\theta}{2\pi} \pmod 1$:
  - For $\theta \in [0, \pi/2)$: $u \in [0, 0.25)$.
  - For $\theta \in (-\pi/2, 0)$: $u \in (0.75, 1.0)$.
- In 10-bit fixed-point quantization ($\text{brot} = 10$, $w = \lfloor 1024 u \rceil \pmod{1024}$):
  - If $u \in [0, 0.25) \implies w \in [0, 256)$. In binary: $\text{bit}_9 = 0, \text{bit}_8 = 0 \implies \text{bit}_9 == \text{bit}_8 == 0$.
  - If $u \in (0.75, 1.0) \implies w \in [768, 1024)$. In binary: $\text{bit}_9 = 1, \text{bit}_8 = 1 \implies \text{bit}_9 == \text{bit}_8 == 1$.
- **Conclusion:** For all valid Givens angles with $\cos\theta > 0$, the identity $\boxed{\text{bit}_9 == \text{bit}_8}$ holds identically.
- **Compression Implication:** Only 9 bits (bits $0..8$) need to be stored in the banked delta tables. Bit 9 is reconstructed without overhead by copying bit 8.

### 3.3 Exact 128-Qubit Cap Satisfaction

The verifier strictly enforces that each `ANGLE_k` circuit satisfies $\text{num\_qubits} \le 128$ (`QUBIT_CAP`).
With $J = 3$, block size $B = 4$ (3 delta entries), $\text{brot} = 10$, and 9-bit delta compression:
- Selection register ($a \in [0, 3976)$): $12$ qubits ($\lceil\log_2 3976\rceil$).
- Base word registers ($J = 3, \text{brot} = 10$): $3 \times 10 = 30$ qubits.
- Banked delta registers ($3 \text{ deltas} \times 3 \text{ words} \times 9 \text{ bits}$): $81$ qubits.
- Shared scratch workspace: $5$ qubits.
- **Peak Angle Circuit Qubits:**
  $$Q_{\text{peak}} = 12 + 30 + 81 + 5 = \mathbf{128} == \text{QUBIT\_CAP}$$
  $$\text{Headroom} = 128 - 128 = \mathbf{0} \text{ qubits}$$

**State-Preparation Workspace:**  
The official competition specification explicitly clarifies that the 128-qubit cap applies exclusively to the `ANGLE_k` circuits (`max_k ANGLE_k.num_qubits <= 128`). State-preparation registers ($Q_{\text{total}} = 661$, including $544$ temporary qubits) execute outside the angle circuits and do not violate this gate.

### 3.4 Forensic Audit of the 4,363 Toffoli Discrepancy

An audit of the external chemistry architecture claim reveals an arithmetic discrepancy between the literal structural formula and the promoted cost:
- Architecture parameters: $K = \lceil 75 / 3 \rceil = 25$ circuits, 2 passes (forward $U_T$ and reverse $U_T^\dagger$).
- Per-circuit pass Toffolis: $1,129$.
- Angle Toffolis total: $25 \times 1,129 \times 2 = 56,450$.
- Non-angle heuristic Toffolis at $\text{brot} = 10$:
  $846 \text{ (outer)} + 9,461 \text{ (inner)} + 2 \times 75 \times (2 \times 10 - 1) = 846 + 9,461 + 2,850 = 13,157$.
- **Literal Formula Total:**
  $$T_{\text{literal}} = 56,450 + 13,157 = \mathbf{69,607} \text{ Toffolis}$$
- **External Claimed Total:**
  $$T_{\text{claimed}} = \mathbf{65,244} \text{ Toffolis}$$
- **Discrepancy:**
  $$\Delta T = 69,607 - 65,244 = \mathbf{4,363} \text{ Toffolis}$$

**Impact on Chemistry Score and Gain:**
- At $\lambda_{\text{eff}} = 49.25$ and $Q_{\text{total}} = 661$:
  - Claimed cost ($T = 65,244$): $2,123,803,639.40 \implies \mathbf{1.580\times}$ gain.
  - Literal formula cost ($T = 69,607$): $2,266,003,679.75 \implies \mathbf{1.481\times}$ gain.
- **Additional Robustness Risk:** If Reed–Muller angle buffering requires 2,432 phase-fixup Toffolis, total Toffolis rise to $72,039$, increasing cost to $2,345,214,040$ and reducing gain to $\mathbf{1.431\times}$.
- **Status:** Benchmarked chemistry promotion is held at `HOLD_FOR_ARITHMETIC_AND_RECEIPT_RECONCILIATION`. The 25 circuits must be compiled and simulated through Kickmix to obtain the exact verified Toffoli count before asserting any gain factor.

---

## 4. Synthesis of Overall Portfolio Trajectory

| Benchmark | Current Verified Track 3 Baseline | Current Verified Factor | Target Research Factor | Target Cost | Implementation Status |
|---|---|---|---|---|---|
| **I: `invsqrt`** | $157,752.01$ | **$13.164804\times$** | $13.164804\times$ | $157,752.01$ | **COMPLETED & VERIFIED** (90Q Affine Champion) |
| **II: `dqi`** | $24.439 \times 10^{12}$ | **$1.000\times$** | $14.396\times$ | $1.698 \times 10^{12}$ | **PENDING** (Gluzdov V6 architecture needed; sweep alone fails) |
| **III: `dfthcsos`** | $3.356 \times 10^9$ | **$1.000\times$** | $1.580\times$ | $2.124 \times 10^9$ | **PENDING** (Signed 4-bank tables verified; `.kmb` compiler needed; 4,363 discrepancy audit) |
| **Portfolio $S$** | **$2.3612$** | — | **$6.690$** | — | Track 3 Sealed; awaiting independent II & III receipts |

### Conclusion & Operational Rules
1. The 90Q affine-optimized circuit remains the verified champion for Benchmark I. FAROS-R1, FAROS-R2, and SABLE-90 (89Q/88Q) are formally terminated as active contenders.
2. In accordance with the Three-Role Protocol and Anti-Snooping Holdout Discipline, the portfolio score $S = 2.3612$ will remain unmodified until physical `.kmb` binaries for Benchmarks II and III pass official verifiers under fresh seeds.
