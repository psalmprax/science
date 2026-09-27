# Preregistration: Frozen Comparative Benchmark — APEDP vs External Equation Discovery

**Document ID:** PREREG-APEDP-EXTBENCH-60-2026-09-26
**Date:** September 26, 2026
**Status:** FROZEN — problem taxonomy, counts, noise regimes, metrics, statistics,
and success criteria below are fixed. Corpus generation follows the freeze.

**System freeze record:**
- APEDP HEAD: `e563be11e2c501fd81f9c73f7ec7163ec22eced3` (2026-09-23).
  Worktree DIRTY at freeze (uncommitted session work); the freeze covers the
  worktree state — a clean commit is required before competitor runs, otherwise
  this round is void. `git status` snapshot archived with the corpus manifest.
- Python 3.12.3; numpy 2.5.3; scipy 1.18.1; sympy 1.14; scikit-learn 1.9.1;
  pandas 3.0.5; Pint 0.26.1; pytest 9.1.1.
- Competitor pins: **PySR 2.5.0** (requires Julia; separate container),
  **PySINDy 2.1.0**, **AI-Feynman @ a05bc4a** (`a05bc4a5be23d6eb3e1d0b2f7eb1ab5b78a920ad`).
- Wall-clock budget per problem per competitor: 600 s; CPU 4; memory 16 GB
  (container-enforced at execution; recorded per run).

---

## 1. Scientific Objective

Determine, under identical conditions, whether frozen APEDP demonstrates measurable
advantages or distinct capabilities (family recovery, parameter accuracy, OOD
generalization, abstention integrity) relative to external equation-discovery
systems — not to prove absolute superiority. An N=1 demonstration or an internal
tournament against APEDP's own paradigms cannot quantify comparative standing;
only a frozen, preregistered, independently evaluated benchmark can.

## 2. Freeze Procedure (order matters — violation voids the round)

1. **System freeze.** Record the APEDP git commit hash, `pyproject.toml` dependency
   pins, all `numerical_policy.py` constants, and the Stage C/D configuration.
   No APEDP code, constant, prompt, or adapter changes after this point until
   results are published.
2. **Preregistration freeze.** This document graduates from DRAFT to FROZEN with an
   assigned ID, fixing: problem taxonomy (§3), counts, noise regimes, metrics (§5),
   statistical tests, and success criteria (§7) — *before* corpus generation.
3. **Problem freeze.** Generate the corpus from fresh seeds *after* preregistration.
   Ground truth lives in `benchmark_corpus/hidden/`; competitors (including APEDP's
   own run harness) see only `public/` (train/val CSVs + spec). Follows the existing
   `benchmarks/tournament_runner.py` / `benchmarks/independent_evaluator.py` pattern
   (public/hidden split, blind evaluation, Wilcoxon/Holm).
4. **Environment freeze.** Each competitor runs in its own pinned container (§4)
   with identical CPU count, memory cap, and wall-clock budget per problem.

## 3. Problem Set (proposed: 60 problems)

| Block | Families | N | Stress axes |
|---|---|---|---|
| P: polynomials / power laws | polynomial deg 1–4, pure power incl. off-grid exponents | 10 | σ ∈ {0, 1.5%, 5%}; N ∈ {20, 40, 80} |
| E: exponentials / logs | exp decay incl. off-grid rates, log boundary layer | 10 | same noise/grid + domain-restriction cases |
| R: rationals | saturation rational, off-grid rational (808-class) | 8 | noise seeds ×5 for one 808-class item (stability probe) |
| O: oscillatory | damped / undamped incl. off-grid ω | 8 | phase non-separability, noisy-residual distractors |
| C: composite envelopes | rational × exponential, multi-scale | 8 | high-noise, sparse (N=20) |
| X: extrapolation | subset of P/E/R re-split | 8 | far-OOD error as primary metric |
| A: abstention adversaries | out-of-grammar, underspecified (N≤8), unidentifiable-at-noise-floor | 8 | correct abstention (C3/C5) scored, not fit quality |

All generating laws, seeds, and noise realizations are fixed at corpus generation
and hidden until evaluation. The `apedp/ingestion` Feynman suite (dimensionally
validated, reproducible generation) may supply additional block-P/E items provided
each item is re-pinned under this protocol's seeds.

## 4. Competitors & Adapters

- **APEDP** (frozen hash), **PySR** (pinned version), **PySINDy** (pinned version),
  **AI-Feynman** (pinned commit).
- Each competitor gets an **adapter** in `benchmarks/`, frozen with the protocol.
  Adapter contract — input: public CSVs + budget; output: (a) expression string,
  (b) predictions on the provided train/val/near/far grids, (c) wall-clock used,
  (d) abstention flag where supported.
- Competitors run with **default or author-recommended settings**; any non-default
  choice must be preregistered with justification. No per-problem tuning by the
  experimenter. File-based grid predictions keep the evaluator reusable for systems
  with no Python API.

## 5. Metrics (fixed before running — no additions after)

Per problem, computed blindly by `IndependentEvaluator` against hidden truth:

1. **Family accuracy** — canonical family match via `SymbolicEquivalenceEngine`
   (never raw string match).
2. **Parameter error** — max relative error on matched families (reported; not
   ranked where families differ).
3. **Predictive MSE** — val / near-OOD / far-OOD (far-OOD weighted highest in
   block X).
4. **Abstention score** (block A only) — correct C3/C5 vs hallucinated fit; false
   fits penalized harder than abstentions.
5. **Budget compliance** — over-budget runs disqualified per problem, reported as
   failures.
6. **Proof-tier audit** — APEDP's P-tier distribution reported for firewall
   verification (informational, not ranked).

Statistics: paired Wilcoxon signed-rank per metric + Holm correction across metrics
(existing evaluator precedent). Report effect sizes, not just p-values. The 808-class
stability probe reports `family_stability` / `mse_gap` (`apedp.bootstrap_stability`)
as a *diagnostic*, not a ranking metric.

## 6. Anti-Tuning Rules (load-bearing)

1. **No APEDP modifications** after the first competitor score is observed — code,
   constants, prompts, and adapters alike.
2. **Hidden data is never opened** until all competitors' outputs are locked.
   Violation voids the round.
3. **Bug fixes require a new round**: preregistration amendment + fresh seeds +
   re-run of *all* competitors (never APEDP alone).
4. **Category B outcomes are reported as ambiguity**, never counted as wins for
   any system.
5. **Negative results publish**: a lost block ships with unchanged analysis.

## 7. Success Criteria (predefined)

- **Primary**: family-accuracy advantage on blocks P–C with Holm-corrected
  significance, OR abstention-precision advantage on block A.
- **Secondary**: far-OOD MSE advantage on block X; parameter-error advantage
  conditional on matched families.
- **Firewall check**: zero P1/P2 claims derived from finite numerical evidence
  across all APEDP outputs.

## 8. Reporting

Results in Verification-Matrix form (engine invariants / proof tiers / empirical
synthesis kept separate), plus a threats-to-validity section following the
`PHYSICAL_BENCHMARK_METHODOLOGY_AND_THREATS_TO_VALIDITY.md` precedent. Raw outputs,
seeds, container pins, and hashes are archived with the report for independent
re-execution.
