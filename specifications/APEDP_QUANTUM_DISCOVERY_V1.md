# APEDP-Q Quantum Discovery Architecture & Epistemic Specification V1
**Status:** Canonical Research Specification  
**Version:** 1.0.0  
**Repository:** APEDP-X / CPMD (`apedp_dyn`)  
**Domain:** Quantum Circuit Compilation, Finite-Field Tensor Discovery, and Epistemic Referee Discipline  
**Date:** 2026-11-04  

---

## 1. Executive Summary & Research Mandate

This specification formalizes the **APEDP Quantum Discovery Architecture (V1)**, establishing rigorous mathematical foundations, cryptographic provenance, and experimental controls for automated quantum circuit optimization.

The platform targets three benchmark tracks from the Google Verifiable Quantum Circuit Compilation benchmark:
1. **Benchmark I (invsqrt):** Reversible piecewise-polynomial / QROM reciprocal square root on input domain $x \in [1, 3\,133\,452]$.
2. **Benchmark II (DQI):** Reversible Reed-Solomon syndrome decoder over $GF(2^{12})$ for parameters $(m=4095, n=70, b=12)$.
3. **Benchmark III (DFTHCSOS):** Double-factorized truncated second-quantized active-space Hamiltonian simulation (FeMoCo $76o, 113e$).

This specification incorporates **four fundamental technical corrections**, reconciles the $GF(2^{12})$ multiplier incumbent hierarchy, enforces independent accounting of mathematical tensor rank versus reversible quantum circuit execution cost, establishes strict proposer-referee separation, and defines the pre-registered protocol for **MMLM-POC-001** (Multi-Mathematics Language Model Proof of Concept).

---

## 2. Pinned Benchmark Contracts & Reference Resource Costs

All benchmark evaluations are scored against frozen, authoritative reference costs. Arbitrary threshold literals are strictly prohibited.

### 2.1 Benchmark Contracts Summary

| Benchmark Track | Problem Contract | Reference Cost ($C_{\text{ref}}$) | Scoring Formulation | Acceptance Criterion |
|---|---|---|---|---|
| **Benchmark I: `invsqrt`** | Full domain $x \in [1, 3\,133\,452]$, 64 shots/input (200,540,928 total shots), exact integer reference phase. | `2076774.233715282` | $C = Q \cdot T_{\text{equiv}}$ | Zero failure shots; phase error $\le \text{tol}$. |
| **Benchmark II: `dqi`** | $GF(2^{12})$ RS MBUC syndrome decoder $(m=4095, n=70, b=12)$, 102,400 shots. | `24439473914136.668` | $C = R(\varepsilon_+) \cdot Q^2 \cdot T_{\text{mean}}$ | Clopper-Pearson $\varepsilon_+ < 1/16$ at $1 - 2^{-128}$. |
| **Benchmark III: `dfthcsos`** | FeMoCo active space ($76o, 113e$), Givens-rotation state preparation and Trotter step. | `3355599588.3155365` | $C_{\text{chem}} = \lambda_{\text{eff}} \cdot Q_{\text{total}} \cdot T_{\text{total}}$ | Eigenvalue shift and energy within chemical accuracy. |

### 2.2 Frozen Benchmark Submission Archive

To protect long-term experimental provenance against transient `/tmp` purging:
- **Primary Archive Checkpoint:** `artifacts/checkpoints/submission_24_59.zip` (staged from `/tmp/submission_24_59.zip`).
- **Archive File Size:** `28,246,615` bytes (~28.2 MB).
- **Contents, re-measured on 2026-11-05** (`unzip -p <archive> <entry> | sha256sum`; re-run this before trusting any row — all three hashes previously pinned here had drifted from the bytes in the archive):
  - `invsqrt.kmb` (479,055 bytes; SHA-256: `af4698478fbad775a5a271690cf4c2d8783caf0b8d12048106d1973c4ac6862e`) — the **62-qubit** Benchmark I build: official `PASS` over 3,133,452 inputs × 64 shots = 200,540,928 shots at **97,921.5795**, 1,281.6647 executed / 1,537 static Toffolis, 41 rotations, worst error/tolerance 0.99932377 at $x = 161$ (receipt: `benchmarks/quantum_compilation/invsqrt_62q_full_domain_receipt.json`). This is **not** the submitted 89-qubit champion; that is a different file (`ce04c9d54dc42e42455444022b6652a6efb0be2630aca7b485d3efd4e58fa9de`, 245,010 bytes, `artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd.kmb`, 157,601.0808 at the same shot budget).
  - `dqi.kmb` (529,710,608 bytes; SHA-256: `748b01a7a07662c3f7e389b844e313a3d55dfb3b3d993414e5ad3dfac6fe154c`) — **a different, better decoder, verified rather than inherited:** it is not the 222-iteration decoder pinned at `959471d05a26373b4f14ca0cae2d8639ea25f18170d9f6532139aa51ccc04995` by `config/quantum_artifact_manifest.json` and `tests/test_quantum_artifact_manifest.py`, and it matches no `.kmb` held on disk. Re-verified on 2026-11-05 at the official 102,400 shots it is `PASS` with 0 value / 0 phase errors and scores **375,374,174,093.81** (Q = 891, T = 396,919.33, max 397,192, 58,336,057 ops) against the pinned build's 24,439,463,583,672.93 (Q = 1,893, T = 5,725,108.38) — **65.11× lower**, and 65.11× the 24,439,473,914,136.668 reference cost. The archive therefore holds a Benchmark II entry that the Track-3 baseline of 24.439e12 does not reflect; resealing the portfolio on it is a decision, not a consequence (receipt: `benchmarks/quantum_compilation/dqi_archived_artifact_reverify.json`).
  - `dfthcsos/` (15 angle `.kmb` files + `submission.npz`, 1,863,164 bytes; submission SHA-256: `ddce9a5bd7d67969e9b7ba4266fe1ae4c8946344553a6dbb7e1355d657bfc397`) — **a different, better bundle, verified:** not the `276d464f…` submission in `artifacts/quantum_compilation/dfthcsos/arm1/baseline_receipt.json`, and not the 12-bit family in `docs/DFTHCSOS_HANDOVER.md`. Verified on 2026-11-05 against the packaged `FeMoCo76o113e.fcidump` (128,128,348 bytes, sha256 `e5a48af7…`, both pinned by `scoreinfra/reference.py`) it is `PASS` at 78,080 shots with score **311,465,857.07** ($\lambda_{\text{eff}}$ 47.1847, `T` 15,459, `Q` 427, 15 circuits, `brot` 8, `batch_size` 5) against the Track-3 baseline of 3,355,599,588.32 — **10.77× lower**, and 6.82× better than the $1.580\times$ target. Receipt: `benchmarks/quantum_compilation/dfthcsos_archived_artifact_verify.json`.
- **Unpacked Verification Tree:** `/tmp/check_24_59/` (reproduce with `unzip -o artifacts/checkpoints/submission_24_59.zip -d /tmp/check_24_59`; the tree is transient by design, so the archive bytes and the hashes above are the record, not the unpacked path)
- **Composite Gain Metric:** $S = 20.9865$ — the geometric mean of the three sealed benchmark factors, $[13.177411 \times 65.106932 \times 10.773571]^{1/3}$, which is $3.136\times$ the $6.692$ target. Benchmark I is rebuilt (`ce04c9d5…`, pinned champion $157{,}601.0808$); Benchmarks II and III are **attested** on the binaries staged in `submission_24_59.zip` (`dqi.kmb` `748b01a7…`, $Q$ 891; `dfthcsos/submission.npz` `ddce9a5b…`, $Q$ 427), each passing its official verifier at the official budget under the default seed and again under salt `reseal-2026-11-05` — see [`docs/ARTIFACT_ATTESTATION_RULE.md`](../docs/ARTIFACT_ATTESTATION_RULE.md). The earlier $24.59\times$ label is withdrawn: it reproduces only as the cube root of **four** factors ($13.177411 \times 1.609466 \times 65.106932 \times 10.773571 = 14{,}876.6 \Rightarrow 24.59$), i.e. it counted the 62-qubit `invsqrt` row as a separate factor under a cube root and so overstated the portfolio by exactly $1.6095^{1/3} = 1.172\times$.

---

## 3. Mathematical Foundation of the $GF(2^{12})$ Field

The finite field $GF(2^{12})$ is defined in accordance with the KickMix little-endian polynomial basis specification:

$$GF(2^{12}) \cong GF(2)[u] / (u^{12} + u^6 + u^4 + u + 1)$$

### 3.1 Field Specification Properties
- **Modulus Integer:** `4179` (decimal)
- **Binary Representation:** `0b1000001010011`
- **Set Bit Exponents:** $\{12, 6, 4, 1, 0\}$, corresponding to $u^{12} + u^6 + u^4 + u^1 + u^0$.
- **Field Size:** $|GF(2^{12})| = 2^{12} = 4096$.
- **Multiplicative Order:** $2^{12} - 1 = 4095 = 3^2 \times 5 \times 7 \times 13$.
- **Irreducibility:** $P(u)$ is irreducible over $GF(2)$ ($\gcd(P(u), u^{2^d} - u) = 1$ for all $1 \le d \le 6$).
- **Primitive Element:** The root $u$ (represented as integer `2`, bit 1) has order exactly $4095$, since $u^{4095} \equiv 1 \pmod{P(u)}$ and $u^{4095/p} \not\equiv 1 \pmod{P(u)}$ for all prime divisors $p \in \{3, 5, 7, 13\}$.

### 3.2 Multiplication Tensor Definition
Let basis elements be $e_i = u^i$ for $i \in \{0, \dots, 11\}$. Any field elements $a, b \in GF(2^{12})$ are represented as polynomials $a(u) = \sum_{i=0}^{11} a_i u^i$ and $b(u) = \sum_{j=0}^{11} b_j u^j$.

The exact multiplication tensor $T \in GF(2)^{12 \times 12 \times 12}$ maps input bits to product bits:

$$c_k = \bigoplus_{i=0}^{11} \bigoplus_{j=0}^{11} T_{ijk} a_i b_j \pmod 2$$

where $T_{ijk}$ is the coefficient of $u^k$ in the reduced basis element product:

$$T_{ijk} = [u^k] \left( u^{i+j} \pmod{u^{12} + u^6 + u^4 + u + 1} \right)$$

Because $u^i \cdot u^j = u^j \cdot u^i$, $T$ satisfies exact symmetry $T_{ijk} = T_{jik}$.

---

## 4. Architectural Corrections & Invariants

### 4.1 Correction A: Karatsuba Baseline Reconciliation & The 51-Toffoli Incumbent
Previous internal drafts assumed Karatsuba with 54 nonlinear operations was the reference comparison point. The October 2026 Google Quantum AI publication establishes the true baseline hierarchy:

| Multiplier Implementation | Nonlinear Count | Category | Epistemic Role |
|---|---|---|---|
| **Schoolbook** | 144 | Naive Bilinear | Upper-bound reference baseline ($12 \times 12$). |
| **Published Karatsuba Reference** | 63 | Recursive Divide-and-Conquer | Reproduces the official KickMix benchmark implementation. |
| **Independent Karatsuba ($\mu(2)=3, \mu(3)=6$)** | 54 | Exhaustive Base-Case Composed | Composed $3 \times 3 \times 6 = 54$ using exhaustive sub-product minima. |
| **Published Hand-Tuned Incumbent** | 51 | Literature Incumbent | Existing strongest comparison point (Montgomery 6-term composition / tower-field). |

**Crucial Invariant:** The 51-Toffoli count is an existing implementation result, **not a mathematical proof of minimal tensor rank**.  
The true research objective for APEDP-MMLM is:
$$\text{Discover a machine-verified } GF(2^{12}) \text{ multiplication construction with } < 51 \text{ Toffoli-equivalent operations}.$$

### 4.2 Correction B: Tensor Rank vs. Reversible Quantum Circuit Cost
A bilinear tensor decomposition of rank $R$:

$$T_{ijk} = \bigoplus_{r=1}^R a_{r, i} b_{r, j} c_{r, k} \pmod 2$$

expresses the multiplication using $R$ nonlinear AND operations $m_r = (\sum_i a_{ri} a_i) \land (\sum_j b_{rj} b_j)$.

However, **mathematical tensor rank $R$ does not equal total quantum circuit cost**. The complete physical circuit budget must record:
1. **Mathematical Tensor Rank $R$:** The number of bilinear products $(\alpha_r \cdot a)(\beta_r \cdot b)$.
2. **CNOT Networks:** Linear combination input fan-in trees ($\sum a_{ri} a_i$) and output target accumulation ($\sum c_{rk} m_r$).
3. **Workspace & Ancilla Footprint:** Temporary AND registers and scratch windows (e.g. 19 clean qubits with level-by-level measure-and-reset vs 42 uncomputed qubits).
4. **Measurement-Based Uncomputation:** Opcodes such as `del_and` and `mr` that clear scratch qubits for zero Toffoli cost with classical phase correction, versus reversible unitary uncomputation which doubles Toffoli counts.
5. **Circuit Depth:** Total two-qubit gate depth and $T$-depth under realistic quantum hardware scheduling constraints.

**Mandate:** Every candidate discovery engine must report $(R_{\text{tensor}}, T_{\text{toffoli}}, C_{\text{cnot}}, Q_{\text{workspace}}, D_{\text{depth}})$ independently.

### 4.3 Correction C: Ledger vs. Referee Epistemic Separation
A ledger is an immutable historical record; **it is not an epistemic referee**. The system enforces strict separation of responsibilities:

```
+-----------------------------------------------------------------------------------+
|                            APEDP-X Epistemic Engine                              |
|   - Adjudicates evidence sufficiency, DSR hurdle (>= 0.95), Diebold-Mariano p    |
|   - Enforces promotion lifecycle stages (Stage 1 through Stage 5)                |
+-----------------------------------------------------------------------------------+
        ^                                               ^
        |                                               |
+------------------------------------+   +------------------------------------------+
|    Official Benchmark Verifier     |   |   Independent Mathematical Checker       |
| - Verifies contract adherence      |   | - Reconstructs exact tensor over GF(2)   |
| - Runs 102,400 shots (DQI)         |   | - Verifies irreducible polynomial fields |
| - Evaluates zero-failure shots     |   | - Proves circuit equivalence algebra     |
+------------------------------------+   +------------------------------------------+
        \                                               /
         v                                             v
+-----------------------------------------------------------------------------------+
|                        FullLineageExperimentLedger                                |
|   - Records complete immutable experiment lineage, trial counts, and hashes       |
|   - Permanently burns holdout slices upon out-of-sample failure                   |
|   - CANNOT decide mathematical truth; preserves audit trail and provenance        |
+-----------------------------------------------------------------------------------+
```

### 4.4 Correction D: Exact Integer Numerical Screening for `invsqrt`
The official `invsqrt` verifier tests 64 randomized measurement outcomes across all $3\,133\,452$ inputs ($200\,540\,928$ shots total) with zero tolerance for failure, comparing against exact 64-bit integer reference phase arithmetic.

The candidate screening model (`InvsqrtNumericalModel`) must:
- Implement bit-exact integer and fixed-point arithmetic matching the verifier.
- Mirror the verifier's exact normalization conventions and coefficient rounding.
- Evaluate the entire domain $x \in [1, 3\,133\,452]$ using vectorized NumPy operations.
- Calculate worst-case relative phase error $\max_x |\Delta \phi(x)| / \text{tol}$.
- Act as a high-throughput rejection filter before committing to expensive verifier runs.

---

## 5. Protected Chemistry Boundaries (Track III: DFTHCSOS)

To preserve benchmark integrity:
1. **Protected Symmetry Shifts:** $S$ and $\beta_1$ defined in `FeMoCo76o113e.fixed_shifts.npz` (SHA-256 `7b7facf63fe35a9d9d3d00787ffdb700c863fa845fe4724d7f7c54778151cd87`) are protected constants. `load_fixed_shifts()` is an immutable boundary; submitted replacements are ignored by the verifier.
2. **Factorization Allowlist Representation:** In candidate parameter allowlists, `H2_I` strictly designates the permitted double-factorization representation matrix, **not** the immutable input two-body Hamiltonian tensor $H_{pqrs}$.
3. **Optimization Scope:** Candidate search is confined to permitted factorization ranks, Givens-rotation angles, and truncation thresholds while preserving exact electronic structure invariants.

---

## 6. MMLM-POC-001: Experimental Protocol & Pre-Registration

### 6.1 Research Hypothesis
> **Hypothesis:** A Multi-Mathematics Language Model (MMLM) reasoning across abstract algebra, tensor decomposition, and reversible circuit synthesis can discover a machine-verifiable $GF(2^{12})$ multiplication implementation requiring fewer nonlinear operations than conventional search under an identical computational budget.

### 6.2 Experimental Arms (Matched Budget)

All three arms receive the identical starting tensor $T \in GF(2)^{12 \times 12 \times 12}$ and an equal budget (CPU/GPU time, maximum candidate evaluations, and solver search depth):

| Arm | Search Paradigm | Engine Components |
|---|---|---|
| **Arm A: Conventional Search** | Automated Combinatorial / Algebraic | Flip-graph local search, Boolean SAT rank search (`z3`/`pysat`), alternating minimization over $GF(2)$. |
| **Arm B: Frontier LLM Agent** | Model-Directed Search Guidance | Frontier model directing search heuristic transitions, proposing tensor slice permutations, and selecting mutation bases. |
| **Arm C: Mathematical Heuristics** | Structured Heuristic Search | Montgomery 6-term Karatsuba composition, tower-field embeddings $GF((2^6)^2)$, randomized local restarts. |

### 6.3 Pre-Registered Outcome Matrix

The empirical outcome strictly determines subsequent architectural commitments:

```
                                  [ MMLM-POC-001 Execution ]
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
           [ Verified Sub-51 Found ]                               [ No Sub-51 Found ]
                     |                                                   |
         +-----------+-----------+                           +-----------+-----------+
         |                       |                           |                       |
   [ Arm B Only ]         [ Both Arm A & B ]           [ Arm A Only ]          [ Neither Arm ]
         |                       |                           |                       |
    OUTCOME 1               OUTCOME 2                   OUTCOME 3               OUTCOME 4
(LLM Value Proved)     (Advantage Unproven)       (LLM Layer Rejected)      (Inconclusive)
         |                       |                           |                       |
Commit to Full MMLM     Refine Benchmark &          Do NOT build MMLM;      Reassess Formulation;
Mathematical Architecture Increase Problem Rigor     Rely on Classical      Audit Search Methods
                                                          Solvers
```

### 6.4 Deliberately Deferred Scope
The following components are deferred pending successful validation of MMLM-POC-001:
- Universal Math Intermediate Representation (Math-IR).
- Eight concurrent mathematical domain sub-agents.
- Model fine-tuning (LoRA / GRPO training). All initial experiments use frontier models with tool seams.

---

## 7. Discovery Engine Specification: `field_tensor_search.py`

The implementation module `apedp_dyn/quantum/field_tensor_search.py` provides the canonical programmatic interface:

1. `build_field_spec() -> FieldSpec`: Returns verified field parameters and proves irreducibility and primitivity.
2. `build_multiplication_tensor() -> np.ndarray`: Constructs the exact $12 \times 12 \times 12$ binary tensor $T_{ijk}$.
3. `verify_tensor_decomposition(tensor, decomp) -> (bool, list)`: Reconstructs $\hat{T}$ and performs coefficient-by-coefficient verification over $GF(2)$.
4. `build_schoolbook_reference() -> TensorDecomposition`: Generates the 144-product reference decomposition.
5. `build_karatsuba_reference() -> TensorDecomposition`: Generates the 63-product published reference Karatsuba decomposition.
6. `build_54_reference() -> TensorDecomposition`: Generates the 54-product Karatsuba base-case composition.
7. `load_published_51_reference() -> TensorDecomposition`: Reproduces the published 51-Toffoli incumbent decomposition.
8. `search_flip_graph(decomp, max_steps, seed)`: Explores the flip-graph of valid decompositions over $GF(2)$.
9. `search_sat_rank(target_rank, time_limit, seed)`: Encodes and searches for decompositions using Boolean satisfiability.
10. `compile_reversible_multiplier(decomp, controlled) -> ReversibleCircuitSpec`: Produces executable reversible circuit instructions.
11. `evaluate_multiplier(circuit, rank) -> dict`: Measures full quantum resources (rank, Toffoli, CNOT, ancillas, depth).
12. `generate_reproducibility_manifest(...) -> dict`: Emits a SHA-256 hashed provenance certificate.
13. `InvsqrtNumericalModel`: Exact integer and fixed-point candidate screening engine for Benchmark I.

---

## 8. Epistemic Assurance Sign-Off

This document constitutes the frozen research contract for quantum circuit discovery in APEDP-X. Any modification to threshold constants, benchmark contracts, or baseline hierarchies requires updating this specification, re-minting its SHA-256 hash, and recording the rationale in `CHANGELOG.md`.
