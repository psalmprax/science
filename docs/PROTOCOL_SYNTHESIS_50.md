# Protocol: SYNTH-50 Autonomous Model Synthesis Tournament

**Document ID:** PROTOCOL-APEDP-SYNTH50-2026-10-01
**Date:** October 1, 2026
**Status:** RESOLVED — §5.4 records the refereeing result. All four conditions of
the criterion fixed in §5.3 pass, and **near-OOD promotion is permitted for
SYNTH-50**. Two caveats travel with that verdict and are stated in §5.4: the
far-OOD conditions covered 31 of 50 problems and none of class 5, and the
single regression (S17, 7.92x) cleared the 10x bound by only 1.26x.

---

## 0. What this document is, and what it is not

It would be easy to call this a preregistration. It is not one, and the
difference matters.

**What was inspected before this document was written.** The full SYNTH-50
corpus has been read: the manifest, all 50 `hidden/ground_truth.json` files,
and the class layout in §1. S11's outcome under both promotion settings is
already known (§5.2). The EXT-BENCH-60 referee results in §5.1 are known.

**What was *not* known.** The aggregate effect of near-OOD promotion across
SYNTH-50 has not been measured. That measurement is running against the
criterion fixed in §5.3.

So this is a protocol with a pre-committed decision rule, not a preregistration
with pre-committed predictions. The distinction is not pedantry: a
preregistration's value is that the *criterion* was fixed before the data
arrived, and that is exactly what §5.3 does. S11 cannot serve as evidence for
the rule it motivated, and §5.2 says so in the document rather than in a
footnote.

Where this protocol contradicts the existing test suite, that is recorded in
§6 rather than resolved silently.

---

## 1. What SYNTH-50 measures

50 problems, `S01`–`S50`, in five classes of ten. The corpus is checked in
under `benchmark_corpus_synthesis_50/` with `checksums.sha256` and a manifest.

| Class | Name | Problems | Truth family |
|---|---|---|---|
| 1 | Multi-Regime Envelopes | S01–S10 | varies |
| 2 | Logarithmic & Boundary Layers | S11–S20 | `logarithmic_boundary_layer` (all ten) |
| 3 | Critical Exponents & Scaling | S21–S30 | varies |
| 4 | Modulated Transport | S31–S40 | varies |
| 5 | Epistemic Abstention Adversaries | S41–S50 | varies |

Each problem ships four public splits — `train.csv`, `validation.csv`,
`near_ood.csv`, `far_ood.csv` — plus `problem_spec.json`, with truth in
`hidden/ground_truth.json` (`ground_truth_formula`,
`canonical_structural_family`, `oracle_epistemic_class`, `expected_state`,
`true_parameters`).

Oracle composition, read from `hidden/`:

| Oracle class | n | `expected_state` | n |
|---|---|---|---|
| C1 | 40 | `UNIQUE_SUPPORT` | 40 |
| C3 | 2 | `INSUFFICIENT_EVIDENCE` | 2 |
| C5 | 8 | `MISSING_STRUCTURE` | 8 |

The benchmark therefore asks two questions at once: *can the engine recover the
right structure*, and *does it decline when the data cannot support a claim*.
Ten of fifty problems are abstention adversaries.

**Class 2 is a concentration risk for the headline metric.** All ten problems
share one truth family, so class 2 alone is 20% of any family-accuracy figure
and turns it into substantially a five-way question rather than a fine-grained
one. Any protocol reporting "family accuracy" on SYNTH-50 must report the
per-class breakdown beside it, or a single class moving 10/10 → 9/10 reads as a
corpus-wide result.

---

## 2. Data contract

The selector may see `train`, `validation`, and `near_ood`. It may not see
`far_ood`, and it may not read `hidden/`.

`far_ood` is the referee split: it is the judge for §5, and handing the selector
the split it is scored on turns that number into a training target. This is the
same rule `benchmarks/external_60_adapters.py` enforces for EXT-BENCH-60, where
block X re-splits P/E/R and far-OOD is block X's primary metric.

### 2.1 A resolved question: does far-OOD change the verdict?

`SynthesizedModel.far_ood_mse` is passed into `CandidateProfile` for stage-C
adjudication, and `epistemic_state` is set from `decision.state`. That looks
like a leak, and it was worth checking rather than asserting.

`far_ood_mse` appears exactly twice in `apedp/decision/stage_c.py`: the
`CandidateProfile` field default, and the `ood_tested` boolean

```python
ood_tested = math.isfinite(best.near_ood_mse) or math.isfinite(best.far_ood_mse)
```

It does **not** participate in choosing `best` (which runs on holdout error,
parameter conditioning, and bootstrap stability) and does **not** set the state.
Supplying far-OOD flips whether the adjudication rationale claims extrapolation
was tested; it changes neither the selected model nor the epistemic verdict.

**Consequence.** The SYNTH-50 tests, which pass `x_far_ood, y_far_ood` into
`synthesize`, are not producing contaminated verdicts. They are producing a
*different rationale* than the corpus configuration implies, and they contradict
EXT-BENCH-60's adapter, which withholds far-OOD deliberately. This protocol
adopts the EXT-BENCH-60 position: withhold it (§6 records the resulting test
change).

---

## 3. Selection rule

Unchanged and not reopened by this document:

1. Rank by `val_mse`.
2. Break ties with a noise-calibrated parsimony rule: prefer strictly fewer
   parameters among candidates within `max(0.15 * best.val_mse, train_mse/n_val)`,
   and never let near-OOD *disqualify* a better-in-domain model by more than
   `1.5x + noise floor`.
3. `allow_near_ood_promote` defaults to the engine constant. **§5 governs it.**

---

## 4. Metrics

### 4.0 Which taxonomy "family" means — resolved first, because it decides everything

There are two incompatible family vocabularies in this repository, and the
field name in the SYNTH-50 truth files invites the wrong one.

- The **engine taxonomy** is `SynthesizedModel.structural_family`:
  `logarithmic_boundary_layer`, `rational_transition`, `critical_scaling_divergence`,
  `damped_oscillation`, …
- The **canonicalizer taxonomy** is
  `benchmarks.evaluate_external_60.canonical_family()` (backed by
  `SymbolicEquivalenceEngine`): `transcendental_composite`,
  `rational_or_power_law`, `logistic_saturation`, `polynomial_deg_4`, …

**They share no value.** Measured: `canonical_family()` returns
`transcendental_composite` for `1.5*x**0.5*ln(1.0 + 0.8*x**1.0)`, for
`2.0*x**1.5`, and for the S11 expression recovered correctly with promotion on.
The eight `canonical_structural_family` values across SYNTH-50's hidden files
are `bessel_j0_non_elementary`, `critical_scaling_divergence`,
`damped_oscillation`, `high_entropy_oscillation`, `logarithmic_boundary_layer`,
`piecewise_shock_discontinuity`, `rational_exponential_envelope`,
`underspecified_data_regime` — none of which the canonicalizer emits.

So the truth field named `canonical_structural_family` is canonical in the
*engine's* taxonomy, not the canonicalizer's. **Scoring SYNTH-50 with the
EXT-BENCH-60 scorer would mark every correct answer wrong**, and would do so
without error: the comparison is a string inequality between two disjoint
vocabularies, which is a clean zero rather than a crash.

This protocol scores SYNTH-50 in the **engine taxonomy**: predicted
`structural_family` against the truth file's `canonical_structural_family`,
both sides the same vocabulary. Every family figure in §5.4 must state which
taxonomy it used.

### 4.1 The metrics

- **Family accuracy** — predicted `structural_family` against
  `canonical_structural_family` in the engine taxonomy, reported per class and
  overall.
- **Epistemic calibration** — predicted state against `expected_state`;
  over-claiming (`UNIQUE_SUPPORT` where `MISSING_STRUCTURE` is expected) is
  counted separately from under-claiming, because they are not symmetric
  errors.
- **Far-OOD NRMSE** — `RMSE / std(truth)` on `far_ood.csv` against the hidden
  formula. Grids whose `std` falls below `1e-2 * max(|y|, 1)` are reported as
  saturated and excluded from the ratio; exclusions are counted, never dropped
  quietly.
- **Parameter accuracy** — recovered values against `true_parameters`, relative
  error, for problems where the family is recovered.

---

## 5. Near-OOD promotion

### 5.1 Prior evidence (EXT-BENCH-60, not SYNTH-50)

Refereed on all 60 EXT-BENCH problems with far-OOD as the judge. Promotion ON
changes 5 families (C41, C43, P08, R25, X52). Across the round: far-OOD better
on 13, worse on 5, unchanged on 34; median far-OOD MSE 3.731e-06 → 2.189e-06.

Two specific costs, both invisible to a median:

- **C41** far-OOD NRMSE 0.0153 → 0.347, a 23x regression.
- **X52** became a *new abstention* on a problem whose `expected_state` is
  `UNIQUE_SUPPORT` — promotion cost a correct answer rather than merely a
  better-worse trade.

A rule that helps on median and costs 23x on one problem and a correct answer
on another is not a clear improvement, and the aggregate alone does not say so.

### 5.2 Why S11 is the motivating case

`test_class_s2_logarithmic_recovery` asserts that S11 recovers
`logarithmic_boundary_layer`. It does not: the top model is

```
1.591772465*x**1.625202228 / (0.8132087704*x**0.8183343958 + 1.0)   # rational_transition
```

while the truth is `1.5 * x**0.5 * ln(1.0 + 0.8 * x**1.0)`. With promotion ON
the truth is selected — `1.51490307042953*x**0.5*log(0.781593962963559*x**1.0 + 1.0)`,
val MSE 9.553e-04, parameters within 2.30% of truth on the worst of four — and
all three of the test's assertions pass.

Independently reproduced by the §5.4 referee run, which also measures the split
the selector never sees: **far-OOD NRMSE 0.1386 → 0.006415, a 21.6x
improvement**. The mechanism is visible in the numbers — near-OOD MSE falls from
1.20e-03 to 3.86e-05 (31x), far enough past the parsimony tiebreak's 2x
promotion factor to fire, while validation MSE is essentially unchanged
(9.4555e-04 → 9.5529e-04, the promoted model 1.0% *worse* in-domain).

That last figure is the whole case: the two candidates are separated by ~1% on
validation, which is inside the tie margin, and the tie is broken by near-OOD
evidence that happens to be right.

**This is the case that motivated §5.3, so it cannot be evidence for §5.3.** It
is stated here so a reader does not mistake the motivating example for
supporting evidence. One problem out of fifty is not a result.

### 5.3 Pre-committed decision rule — FIXED BEFORE THE MEASUREMENT

The referee run executes all 50 problems in both conditions, differing in
`allow_near_ood_promote` only, with `far_ood` withheld from the selector.

**Promotion is permitted for SYNTH-50 if and only if all four hold:**

- **(a)** family accuracy does not decrease;
- **(b)** median far-OOD NRMSE over non-saturated grids does not increase;
- **(c)** no individual problem's far-OOD NRMSE regresses by **more than 10x**
  its no-promotion value — the C41 lesson, which a median would have hidden;
- **(d)** no problem whose `expected_state` is `UNIQUE_SUPPORT` becomes an
  abstention — the X52 lesson, which is a lost correct answer rather than a
  degraded one.

Otherwise promotion stays off and S11 remains a recorded disagreement between
the engine and the test, which is a legitimate state to be in: the test asserts
a capability no protocol authorises.

Committing to criterion (c) at 10x rather than something looser is deliberate. It
was chosen from EXT-BENCH-60's measured failure (23x), where a bound anywhere
between 1x and 23x would have produced the same verdict on that evidence.

### 5.4 Result

All 50 problems, both conditions, 0 errors. Judge: far-OOD NRMSE against the
hidden formula, with the far-OOD split withheld from the selector.

| | condition | result | verdict |
|---|---|---|---|
| (a) | family accuracy must not decrease | 0.560 → **0.640** (28/50 → 32/50) | **PASS** |
| (b) | median far-OOD NRMSE must not increase | 0.0617 → **0.0210** (n=31) | **PASS** |
| (c) | no problem regresses beyond 10x | 8 better, 1 worse, 22 equal, 19 unscoreable; worst 7.92x | **PASS** |
| (d) | no `UNIQUE_SUPPORT` problem becomes an abstention | 0 new abstentions | **PASS** |

**VERDICT: promotion is PERMITTED for SYNTH-50.**

Per class, family accuracy — the breakdown §1 requires, since class 2 is one
family worth 20% of the corpus:

| class | name | OFF | ON | |
|---|---|---|---|---|
| 1 | Multi-Regime Envelopes | 6/10 | 7/10 | |
| 2 | Logarithmic & Boundary Layers | **1/10** | **4/10** | the largest move |
| 3 | Critical Exponents & Scaling | 6/10 | 6/10 | |
| 4 | Modulated Transport | 10/10 | 10/10 | |
| 5 | Epistemic Abstention Adversaries | 5/10 | 5/10 | unchanged |

Families gained: S10, S11, S12, S16, S18. Families lost: S19. No epistemic state
changed value on any problem.

#### How close (c) actually was

The single regression is **S17 at 7.92x** (far-OOD NRMSE 0.0074 → 0.0588),
inside the 10x bound by a factor of 1.26. The criterion passes, but it passes
narrowly, and any threshold between 1x and 7.92x would have produced the
opposite verdict on this evidence. The 10x bound was chosen from EXT-BENCH-60's
23x failure, where it was uninformative; applied to SYNTH-50 it is close to
binding. A future round should re-derive it rather than inherit it.

#### Coverage of the far-OOD conditions — read this before quoting (b) or (c)

Conditions (b) and (c) were evaluated on **31 of 50 problems**, and on **none of
class 5**. The 19 exclusions are:

- **9 saturated grids** (S03, S07, S08, S09, S10, S33, S35, S37, S38) — the
  far-OOD `y` barely varies, so the NRMSE denominator is float noise. Excluded
  under §4 as written, counted here rather than dropped quietly.
- **10 with no evaluable curve** (S41–S50) — which is *exactly* class 5, the
  abstention adversaries. They produce no numeric expression to score. That is
  correct behaviour, not a gap, but it means far-OOD NRMSE says nothing at all
  about the class whose whole purpose is abstention.

Conditions (a) and (d) are defined for all 50 and were evaluated on all 50.
So the verdict rests on two conditions covering the whole corpus and two
covering 62% of it, with the missing 38% being the problems least able to
produce a far-OOD number at all. Note also that **S10 is among the saturated
exclusions and is in the families-gained list** — that gain has no far-OOD
corroboration.

#### S19: far-OOD and the family label disagree

S19 is the mirror image of S11, and the most consequential single result here.

| | family | far-OOD NRMSE |
|---|---|---|
| promotion OFF | `logarithmic_boundary_layer` (**correct**) | 0.3588 |
| promotion ON | `rational_transition` (**wrong**) | **0.0176** |

Promotion made S19 extrapolate **20x better** on the far-OOD split while
producing the wrong structural family. One of the two judges is wrong here.
The far-OOD split is supposed to be outside the observed domain, so a model
that extrapolates it 20x better is doing something real; yet the corpus labels
`logarithmic_boundary_layer` as the truth for this problem.

That is a question about the corpus, not about the selection rule, and this
protocol does not resolve it. It is recorded here because (a) and (b) point in
opposite directions on the same problem, and any single headline number hides
that. Family accuracy is the criterion SYNTH-50 was built to measure; far-OOD
NRMSE is the criterion used to referee the mechanism. Where they disagree, the
labelled family is the one the benchmark is asking about.

#### Consequence for S11

With promotion permitted, SYNTH-50 callers pass `allow_near_ood_promote=True`
and `test_class_s2_logarithmic_recovery` passes. The engine default stays
`False` (§6 of `docs/NEAR_OOD_PROMOTION.md`), so no other caller's selection
changes — the same scoping `08fccb0` established for EXT-BENCH-60.

**This is a decision about SYNTH-50 only.** EXT-BENCH-60's frozen protocol
still pins promotion off at its adapter, and nothing here disturbs it: the
EXT-BENCH referee measured a different corpus and found a 23x regression
(C41) and a lost correct answer (X52) that SYNTH-50 does not exhibit.

---

## 6. Known defects in the current test suite

Recorded rather than fixed here, because each changes what the suite means.

1. **Six of fifty problems are tested.** `S11, S21, S31, S41, S44, S49` — one or
   two per class. Class 2 is represented solely by S11, so a class-2 claim is a
   single data point standing for ten problems.
2. **The tests pass `far_ood` into `synthesize`**, contradicting §2. The
   measured effect is limited to the `ood_tested` rationale flag (§2.1), but the
   configuration should match the contract.
3. **The tests assert epistemic state under a configuration the protocol did
   not sanction.** They pass far-OOD and omit the promotion flag, i.e. they run
   a configuration that exists nowhere else.
4. **`xfail_strict` is on**, so S11 is a hard failure rather than a recorded
   limitation. §5.4 permits promotion, so S11 should pass once SYNTH-50
   callers pass the flag; until that wiring exists red is correct.
5. **The truth field is named `canonical_structural_family` but is not in the
   canonicalizer's taxonomy** (§4.0). The name is the defect: it reads as
   "the canonicalizer's answer" and is not, so any future attempt to score
   SYNTH-50 by reusing the EXT-BENCH-60 scorer will report near-zero family
   accuracy and look like a discovery capability result rather than a
   vocabulary mismatch.

---

## 7. Reproducing §5.4

```
python3 /tmp/referee_synth50.py     # scratch harness, writes /tmp/referee_synth50/
```

One JSON record per (problem, condition). Nothing is written into
`benchmark_corpus_synthesis_50/`. If the harness is relocated into the repo it
must land under `benchmarks/` and gain the same degenerate-input guards as
`tools/coverage_audit.py` — a referee that reports nothing must not be
indistinguishable from one that found nothing.