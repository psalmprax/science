# Transformer Proposer — Design Q&A Record

**Status:** Design discussion record (`audit-only`). No implementation commitment.
No transformer proposer exists in the Rust kernel today; the only transformer
code on disk is the JAX/Flax `SymbolicTransformer` in
[`apedp/neural_discovery/architectures.py`](../apedp/neural_discovery/architectures.py).

**Date:** 2026-10-16

---

## Context

The questions below concern a possible future integration: a learned
(transformer-based) hypothesis proposer inside APEDP-X, implemented in or
alongside the Rust kernel. This record captures the answers given during design
discussion so the reasoning is preserved independently of any implementation.

Relevant on-disk facts:

- `apedp/neural_discovery/architectures.py` already implements
  `MultiHeadAttention`, `TransformerBlock`, `SymbolicTransformer` (encoder over
  expression sequences), and `TransformerDecoder` — in **JAX/Flax (Python)**,
  with training/support modules in `training.py`, `active_learning.py`,
  `uncertainty.py`, `integration.py`.
- The Rust kernel (`rust/apedp-kernel`) currently has **no ML dependency**
  (no `candle`, `tch-rs`, or `ort` in `Cargo.toml`).
- Governing constraints: strict proposer/referee separation, permanent holdout
  burning, and SHA-256-pinned preregistered thresholds (see `AGENTS.md`).

---

## Q1 — Can this system be turned into a transformer with a Rust program?

**Answer: as a proposer — yes; as the whole system — no, and it should not be.**

### Where a transformer fits

Layer 1 (hypothesis generation) only: a learned proposal policy over expression
trees, ensembled with RIME-AD / GA proposers. It emits `CandidateProposal`
objects. The deterministic referees (sole-survivor fidelity guard, residual
whiteness, bootstrap tiebreak, Lean bridge) still adjudicate. This preserves the
epistemic firewall: the transformer may suggest, never promote.

### Why a full conversion is wrong

- The system's value is *interpretable, Lean-verifiable equations with
  fail-closed abstention*. A transformer outputs weights and token
  probabilities; it can propose but cannot prove, and does not naturally
  abstain.
- **Holdout integrity**: transformer training data must be synthetic (via
  `benchmark_corpus_synthesis/`), never benchmark equations that overlap
  holdout or confirmation slices.
- **Determinism**: inference with frozen weights and `temperature=0` (or
  seeded sampling) is reproducible and ledger-compatible. Training is a
  one-time, frozen, hash-pinned event.

### Realistic Rust path

1. Train in JAX (existing `neural_discovery` code).
2. Export ONNX; quantize to int8 for deployment.
3. Run inference in the kernel via `ort` (ONNX Runtime bindings) or `candle`
   (pure Rust).
4. Expose to Python via the existing PyO3 bridge (`apedp/rust_bridge.py`
   pattern); wire as `transformer_proposer` with hash-pinned weights.

Pure-Rust *training* (`burn`/`candle`) is possible but a large effort for
marginal gain; `tch-rs` should be avoided (see Q2).

---

## Q2 — Will the transformer be heavy in size?

**Answer: no — not for this task. Expression proposal is far smaller than
language modeling.**

### Why it stays small

- Token vocabulary ~50–200 (operators, variables, small constants).
- Expression sequences ~20–100 tokens.
- Task-appropriate model sizes: ~2–30M parameters. Literature systems
  (NeSymReS; d'Ascoli et al. 2022, "End-to-end symbolic regression with
  transformers") operate in this range — orders of magnitude below GPT-2-small
  (124M).

### Footprint estimates (6 layers, d_model 256)

| Form | Weight size |
| :--- | :--- |
| fp32 | ~40 MB |
| fp16 | ~20 MB |
| int8 (quantized deployment) | ~10 MB |

| Runtime option | Binary cost |
| :--- | :--- |
| `candle` (pure Rust) | small |
| `ort` (ONNX Runtime) | ~15 MB |
| `tch-rs` (libtorch) | 1–2 GB — **avoid if size matters** |

### Inference cost

CPU-only ~1–10 ms per candidate; a 1,000-candidate batch in ~1 s. Against
RIME's current 10–35 ms per task this is real but acceptable overhead. GPU is
unnecessary.

Perspective: JAX/Flax (already a training dependency) is hundreds of MB; a
10–20 MB frozen, SHA-256-pinned int8 ONNX sidecar is light by comparison.

---

## Q3 — If we build a proposer/proposal, will the transformer self-improve with the engineering loop?

**Answer: bounded yes — but the loop that improves it must not be the
confirmation loop.**

### Safe self-improvement channels

| Signal | Why safe |
| :--- | :--- |
| Synthetic corpus growth (`benchmark_corpus_synthesis/`) | Unlimited, uncontaminated; better generator → better training data → retrain |
| Training-split feedback (candidates scored only on in-sample data) | Never touches confirmation slices |
| Active learning (`neural_discovery/active_learning.py`) | Queries points for the surrogate, not the ledger |
| Proposal-quality metrics (validity rate, diversity, parse rate) | Measures form, not outcomes |

### Unsafe channel (the trap)

Feeding the proposer adjudication verdicts / holdout R² / promotion outcomes as
reward breaks the constitution in two ways:

1. **Contamination** — burned slices leak back in; the proposer trains on data
   the firewall burned. The anti-snooping discipline dies quietly.
2. **Reflexive Goodhart** — a proposer optimizing "what passes the referee"
   learns referee loopholes (fidelity guard edges, whiteness artifacts),
   collapsing the proposer/referee separation that is architectural
   principle #1.

### Governed retrain pipeline (the correct shape)

1. Collect **form-only** proposal statistics (validity, diversity, parse rate).
2. Fine-tune on synthetic + training-split data only.
3. Re-freeze weights; SHA-256-pin the artifact (same discipline as
   `specifications/canonical_preregistered_spec_v1.json`).
4. Harness A/B vs incumbent on sealed benchmarks.
5. Promote only through the same gates as any component, under
   `self_improvement.py` Level 3 (pipeline) and `ConstitutionalAuditor`.

If true closed-loop co-evolution is wanted, the correct shape is co-evolution
*within training data* (adversarial generators, harder synthetic tasks,
rejection sampling off in-sample fit) — never off holdout verdicts.

---

## Q4 — Would a transformer help this project greatly?

**Answer: no — moderately at best, and only as an auxiliary proposer/ranker.**

### Why the expected gain is modest

1. **The bottleneck is not proposal speed.** Five-gate results show RIME-AD
   already achieves exact recovery on its target suite (R² = 1.0; Brus where
   the native GA collapses). The binding constraints are *transfer,
   identifiability, and holdout economics* — an epistemic bottleneck, not a
   generation bottleneck.
2. **Out-of-distribution tension.** The project exists to discover *unknown*
   laws; a transformer prior is trained on *known* distributions
   (`benchmark_corpus_synthesis/`). Anything genuinely novel is
   out-of-distribution by construction — exactly where these models degrade.
   Rigorous SR benchmarks (SRBench and follow-ups) show neural symbolic
   regression does not dominate well-tuned GP on hard/extrapolation tasks.
3. **New contamination surface.** Training data vs. holdout governance is a
   fresh firewall to maintain — cost without a matching epistemic return.
4. **Engineering cost.** New dependency, hash-pinned weights, retrain
   pipeline, drift monitoring. Real work for uncertain benefit.

### Where it could plausibly earn its keep (incremental)

- Dataset → representation/seed *ranking* (which of RIME's 8 representations
  to try first) — amortizes the expensive multi-representation fan-out.
- Fast triage of candidate plausibility before expensive gates.
- Shortcutting repeated benchmark families (amortized search on familiar
  equation classes).

### What would help this project greatly instead

- Second-degree Jet relations (Planck quadratic Riccati follow-up).
- The discrete/number-theory engine.
- Real transfer machinery (cross-regime retention, extrapolation guards).

These are direct capability gaps; the transformer is a convenience layer.

### Bottom line

If ever built: small, frozen, int8, proposer-only, form-only training signal —
and expect it to shave search cost, not unlock new science.

---

## Summary of the recommended design shape

- **Scope:** `transformer_proposer` (Layer 1 only), ensembled with RIME-AD/GA.
- **Training:** JAX/Flax (existing code), on synthetic corpora only.
- **Deployment:** int8 ONNX sidecar (~10 MB), `ort`/`candle` inference in the
  Rust kernel, PyO3-exposed; weights hash-pinned and frozen.
- **Determinism:** frozen weights, greedy/seeded decoding; inference logged.
- **Self-improvement:** governed retrain pipeline on form-only signals;
  holdout verdicts are constitutionally excluded from the training loop.
- **Referees:** unchanged, symbolic, deterministic, fail-closed.

## Non-claims

- This document does **not** claim a transformer proposer exists or is
  planned for a release; it records design reasoning only.
- Size and latency figures are engineering estimates from published
  literature and standard runtime footprints, not measurements on this
  codebase.
