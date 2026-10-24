# Formal Pre-Registration: Common Abstention Wrapper Tournament Protocol

**Status:** PRE-REGISTERED (FROZEN PRIOR TO WRAPPED TOURNAMENT EXECUTION)  
**Protocol Version:** 1.0.0  
**Target Architecture:** Common Scientific Decision Layer (`CompetitorAbstentionWrapper`)  
**Registered Date:** 2026-09-24  
**Principal Invariants:** Frozen Stage A/B/C Gates (`INV-1` to `INV-10`), Identifiability Threshold $\kappa \le 10^6$, Stability Threshold $S \ge 0.15$, Separation Margin $1.15$, Neural Adversarial Reference.

---

## 1. Executive Scientific Rationale

In Phase 4 tournament evaluation, conventional symbolic regression engines (AI Feynman 2.0, LLM-Guided SR, gplearn, Neural AutoML) unconditionally output single unhedged predictions (`UNIQUE_SUPPORT`), generating a **36.7% false discovery rate** (11/30 false discoveries) on ambiguous ($\mathcal{B}_2$) and singular ($\mathcal{B}_4$) problems.

A legitimate methodological critique of this finding is:
> *"Conventional regressors were evaluated without an epistemic state machine. If competitors are equipped with the same abstention mechanism, how do their false discovery rates, recovery rates, and OOD generalization compare?"*

This pre-registration protocol resolves that critique by placing the **APEDP Scientific Decision Layer** as a common, standardized abstention wrapper around all competitor paradigms:

```text
                  CANDIDATE SEARCH PARADIGM
           (AI Feynman / LLM-SR / gplearn / APEDP)
                             │
                             ▼
                     Proposed Candidates
                             │
            ┌──────────────────────────────────┐
            │  APEDP SCIENTIFIC DECISION LAYER │
            │  (Standardized Common Wrapper)   │
            ├──────────────────────────────────┤
            │ 1. Stage A: Units & Roots Audit  │
            │ 2. Stage B: Identifiability & S  │
            │ 3. Stage C: Epistemic Machine    │
            │ 4. Neural Adversarial Ceiling    │
            └────────────────┬─────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
         CERTIFIED       AMBIGUOUS        REJECTED
```

---

## 2. Pre-Registered Hypotheses

1. **Hypothesis 1 (False Discovery Convergence):**
   Wrapping external competitors in APEDP's Decision Layer will suppress competitor false discoveries from 36.7% down to $\le 10\%$, proving that false certainty in symbolic regression is an artifact of missing decision layers rather than search failure alone.

2. **Hypothesis 2 (Sensitivity vs Abstention Frontier):**
   When external competitors are forced to meet Stage A mathematical soundness and Stage B identifiability/stability criteria, candidate quality differences will become evident:
   - Poorly conditioned or unphysical polynomials found by GP will be rejected (`INSUFFICIENT_EVIDENCE` or `NO_VALID_CANDIDATE`).
   - High-quality physical decompositions found by AI Feynman / LLM-SR will achieve valid certification.

3. **Hypothesis 3 (APEDP Core IP Differentiation):**
   The tournament will demonstrate that APEDP's primary intellectual property is the **Scientific Decision & Certification Layer**, which functions effectively as an operating system / certification layer for any underlying equation search engine.

---

## 3. Protocol & Invariant Constraints

1. **Frozen Decision Hyperparameters:**
   - Coordinate coupling threshold: $\tau = 0.08$
   - Candidate separation margin: $\text{ratio} = 1.15$
   - Parameter condition number limit: $\kappa_\text{max} = 10^6$
   - Minimum bootstrap parameter stability: $S_\text{min} = 0.15$
   - Dimensional validation: Dimensional Constant Synthesis enabled.

2. **Evaluation Metrics ($\mathbf{M}^\text{wrapped}$):**
   $$\mathbf{M}^\text{wrapped} = \Big( R_\text{rec},\; R_\text{false},\; P_\text{epi},\; E_\text{OOD},\; R_\text{dim},\; T_\text{wall} \Big)$$
   - No scalar score collapse.
   - Evaluated blindly against hidden ground truth via `IndependentEvaluator`.

3. **Problem Corpus:**
   Balanced 30-problem corpus ($6 \times 5 = 30$, Seed=42, SHA-256 verified) in `benchmark_corpus/`.
