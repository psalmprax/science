# Preregistration: RIME §16e — Multivariate Referee-Side Scope Refusal

**Document ID:** PREREG-RIME-SCOPE-MV-2026-10-17
**Status:** DESIGN FROZEN for review; implementation committed SEPARATELY after.
**Relation:** closes the coverage boundary recorded by the registered RIME-OOF-MV
study (commit `674b13c`, `docs/RIME_OOF_MV_REPORT.md` finding 3): spec §16d
rival probes are univariate (`feature_col="p"`, positive axis) and cannot
adjudicate multivariate lagged frames, so multivariate out-of-family worlds
currently have no registered refusal path while `RimeProposer` emits confident
surrogates (H1 confirmed) that collapse under range shift (H2 confirmed).
**Deviation rule (from §16d):** changing the refusal wiring, the probe families
or the seed rule later means this document is superseded by a NEW prereg — the
suite is re-run, never quietly patched.

## What is added (and what is NOT touched)

- NEW module (planned): `apedp_dyn/epistemic/scope_refusal_mv.py` — probe
  CONSTRUCTION only, for multivariate frames.
- `apedp_dyn/epistemic/scope_refusal.py` (§16d) is FROZEN and NOT modified.
  Adjudication reuses its generic `adjudicate_scope` / `scored_from_predict`
  contracts unchanged. §16e is additive.
- Anti-snooping is unchanged and strict: the referee sees only (dev, holdout)
  frames, the emitted candidate's frozen predictor, and the declared noise
  scale. No world label, planted truth, grammar declaration or dataset tag
  enters — refusal is decided by `adjudicate()` outcomes alone.

## Frozen rival-probe families (fit on dev only; evaluated on holdout)

All three are linear-in-coefficients given declared discrete structure grids,
fitted by ordinary least squares on dev, member selected by minimum dev MSE.
Feature columns are the declared multivariate frame columns (lag columns are
explicit columns, e.g. `x1_l1 = x1[t-1]`). EVERY probe basis additionally
contains the full linear nuisance span `[x_1, …, x_F]` (plus intercept): rivals
must be able to absorb additive nuisance terms while remaining non-monomial in
the interaction structure, otherwise a truth like `4·sin(x1·x2) + 1.5·x3` could
not be rivaled for reasons unrelated to its out-of-family structure. Families
with no valid member on the given frame (e.g. `probe_abs_linear2` on `F = 1`)
are skipped; if NO probe can be built, adjudication is undefined and the
harness raises.

1. **`probe_periodic_quadratic`** — basis `[sin(c·q), cos(c·q), x_1..x_F, 1]`
   over `q ∈ {x_i·x_j (i≤j)}` (pairs and squares) and frequency `c` from
   `C0 = {0.5, 1.0, 1.5, 2.0, 3.0}` plus two deterministic refinement passes
   (pass 1: step 0.1 in [c*−0.3, c*+0.3]; pass 2: step 0.02 in [c*−0.1, c*+0.1];
   `c*` = best dev member so far; `c > 0` strictly). Guard: `q` must have
   finite dev entries and dev variance > 1e-12.
2. **`probe_abs_linear2`** — basis `[|w·x|, x_1..x_F, 1]` over
   `w ∈ {e_i ± e_j, i < j}` (STRICTLY ≥ 2 active components; single-axis `w`
   is excluded because `|x| = x` ties trivially on sign-constant axes and would
   wrongfully refuse true power recoveries; `w·x` itself is absorbed by the
   linear span). Exact rival for pairwise max/switch structure:
   `max(x_i, x_j) = (x_i + x_j + |x_i − x_j|)/2`.
3. **`probe_am_product`** — basis `[sin(c·x_j)·x_k, cos(c·x_j)·x_k, x_1..x_F, 1]`
   over ordered pairs `(j, k)`, `j ≠ k`, same frequency grid/refinement as
   family 1. Classical amplitude-modulated family (oscillation gated by another
   channel).

Family-distinctness: all three differ in KIND from the proposer's declared
static monomial algebra and from §16d's univariate exp/log probes.

## Frozen emitted-candidate rule

Emitted = `RimeProposer(house config from the RIME-OOF-MV prereg)`
`.fit_and_propose(aligned_dev)` → the engine's LIST-TOP proposal, converted via
`scored_from_predict(dev, hold, noise_sigma)`. List-top is used deliberately
(the engine's own dev-BIC ranking); picking by holdout R² would be snooping.
`adjudicate_scope(emitted, probes, holdout_scale)` uses §16d defaults
(`bootstrap=True`, `n_boot=1000`, `bootstrap_seed=7`); promotion requires
UNIQUE_SUPPORT against EVERY probe.

## Two-sided validation suite (frozen)

Worlds and guards, fresh noise seeds {700, 701, 702} (one repetition each of
the full suite per seed):

- **W1–W3 (must DECLINE):** the three registered out-of-family laws as aligned
  frames — `4·sin(x1_l1·x2_l1) + 1.5·x3_l1 + 2`;
  `2·max(x1_l1, x2_l1) + 1.5·x3_l1`; `x3_l1·sin(1.5·x1_l1) + 2`. Train rows
  0:400, holdout 400:600, noise 0.05. Requirement: list-top emitted candidate
  DECLINED by at least one probe per world per seed.
- **G1 (must NOT decline):** control_in_family (aligned) — emitted list-top
  must survive all probes un-declined.
- **G2 (must NOT decline):** §16c pinned monomial worlds — actual inventory is
  d2/d3/d6 across log/uniform/irregular grids; guard emissions are the fitted
  true-exponent monomials on `d2_C1000_uniform_1x` (k=2),
  `d3_C100_uniform_1x` (k=3), `d6_C1e12_log_1x` (k=6), and the
  `d2_C1000_log_1x` k=2 pin from the §16d suite — all must be promoted.
- **G3 (no promotion regardless):** null/noise world — record abstention; no
  probe may induce a promotion path where the engine abstained.

**Family-set rule (frozen):** the final §16e probe set = every declared family
that (a) passes all G guards on all seeds and (b) contributes ≥ 1 W decline.
Families failing (a) are excluded by this rule — mechanically, with the
exclusion recorded in the report; the rule is not re-tuned per world.

**Pass criteria:** ALL W declined AND ALL G promoted on ALL three seeds, and
determinism (same seed → identical verdicts and probe coefficients across
re-runs). ANY failure ⇒ §16e as specified FAILS this prereg: it is reported as
a negative result and a new prereg (v2) is required before any modification.

## Metrics and artifacts (planned)

- Report per (world, seed): emitted list-top formula/R², per-probe dev/holdout
  fits, adjudication records, verdict, which probe(s) declined.
- Artifacts (committed separately, after implementation):
  `apedp_dyn/epistemic/scope_refusal_mv.py`,
  `benchmarks/run_scope_refusal_mv_validation.py`,
  `benchmarks/scope_refusal_mv_validation_results.json`,
  `docs/RIME_SCOPE_MV_REPORT.md`,
  `tests/test_dyn_rime_scope_refusal_mv.py`.
- This prereg itself is frozen now, BEFORE any implementation run.

## Non-goals

No modification of §16d or any frozen referee module; no claims beyond the
declared families and worlds; no real-data claims; no promotion of the
mechanism into production pipelines before the two-sided criteria pass and the
report is reviewed.
