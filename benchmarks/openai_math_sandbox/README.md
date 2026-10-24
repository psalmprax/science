# OpenAI-Math Sandbox (`benchmarks/openai_math_sandbox/`)

A sandbox mirror of OpenAI's 2026-10-06 release *"Sharing AI progress in
mathematics"* (~4000 open problems attempted, 722 manuscripts published, 3
later retracted; the repo now lists 719 across 372 result families). The
source pipeline is `attempt -> publish claim -> independent verification ->
retraction accounting`. This sandbox runs the same loop for the subset of
problems APEDP-X can machine-check, and declares everything else
`UNSUPPORTED` with a reason — the point is to measure, with real numbers,
what is missing from this system relative to that release.

## Full-catalog ingestion (all 372 families / 719 manuscripts)

```bash
python3 benchmarks/openai_math_sandbox/ingest_openai_catalog.py --from-dir <cache>   # regenerate
```

`catalog/manuscripts_manifest.json` records every result family and all 719
manuscripts (titles, PDF paths, discipline, Lean-formalization flag), ingested
from a SHA-256-pinned snapshot in `catalog/source_snapshot/`
(`CONTENTS.md` / `history.md` / `README.md` / `overview.pdf` /
`formalization.yaml` of `openai/math`, retrieved 2026-10-16).
`catalog/CATALOG_AUDIT.md` is the human-readable audit. Pinned by
`tests/test_openai_math_catalog.py` (7/7).

Accounting found by the ingestion (all discrepancies recorded, not hidden):

- **722 → 719** = 3 withdrawn manuscripts (named in `history.md`, absent from
  the catalogue): a sign error in the Weil-classes paper cascaded into two
  dependents. 14 further manuscripts were revised and 13 citation-updated.
- Family numbering is historical, not contiguous: 372 live families over
  001–377 with retired numbers 45/61/70/123/163.
- **Formalization counts do not reconcile upstream**: `history.md` states
  300/719 top-line results formalized (~42%), while `formalization.yaml`
  enumerates 173 source papers / 200 main-result declarations (173 joined to
  catalogue manuscripts, 136/372 families). Both numbers are reported; the
  delta is left unresolved rather than assumed away.
- The yaml `review.status` is `unchecked` and entries are `agent`-produced —
  it is not an independent verification layer, and APEDP-X does not run Lean.
- One yaml source path is stale after a revision ("Taming implies
  compatibility on four-manifolds" still points at the September directory);
  it is joined by title and the stale path is on record.
- **APEDP-X verdict over the whole catalog: 372/372 families are
  `uncheckable_proof_level`** (no proof engine). Probe tags mark 40 families
  where finite numerical illustration is possible and 35 with explicit
  constructions — illustration, never verification.

## Run

```bash
python3 benchmarks/openai_math_sandbox/run_sandbox.py          # full (~2 s)
python3 benchmarks/openai_math_sandbox/run_sandbox.py --fast   # CI ranges
```

Prerequisite for `NT-1`/`NT-2`: release binaries in `rust/goldbach_verifier`
and `rust/riemann_verifier` (`cargo build --release` in each crate). If a
binary is absent the task reports `unavailable` — never `verified`.

The report is written to `sandbox_report.json` and the exit code is non-zero
iff any checker ran and missed its threshold (`failed` = would-be retraction).

## Measured run (2026-10-16)

| ID | Status | Certificate headline | Time |
|----|--------|----------------------|------|
| NT-1 | verified | 999,999 evens in [4, 2e6], 0 violations; ceiling `C2_PREDICTIVE` (universal claim abstains) | 29 ms |
| NT-2 | verified | 13 zeros isolated from scratch, calibration `PASS` (max Δ = 1.15e-9), tally `APPROXIMATE_TALLY_MATCH` (S(T) remainder uncertified), ψ(x) R² = 0.99894 | 20 ms |
| LAW-1 | verified | Planck Riccati exact: jet RMSE 8.60e-07, trajectory R² = 0.999999999759 | 0.7 ms |
| LAW-2 | verified | harmonic y'' = -4y exact: jet RMSE 6.11e-14 | 0.5 ms |
| GA-1 | verified | `sin(2x)` MSE 0.0 (recovered as `-1.0*sin(-2.0*x)`), `2cos(1.3x)` MSE 0.0 | 36 ms |
| PDE-1 | verified | Burgers frozen-coefficient holdout R² = 0.999772, coefficients (-0.9962 adv, 0.1112 diff), 672 holdout samples | 1.6 s |
| LEAN-1 | unsupported | per-result templated Lean files in-tree; no automated at-scale formalization pipeline | — |
| OPEN-RH / OPEN-BSD / OPEN-HODGE / OPEN-KAKEYA / OPEN-ALGO / OPEN-NS / OPEN-SCALE | unsupported | no engine path; reasons in `problems.json` | — |

**Accounting: attempted 6, verified 6, failed 0, unavailable 0, unsupported 8.**

## What is missing (the gap analysis this sandbox exists to expose)

1. **Proof-level mathematics.** Every `verified` above is a finite-range
   numeric certificate or an exact within-family recovery. There is no
   proof-search engine, so the headline categories of the OpenAI release
   (quasi-Riemann hypothesis variant, partial BSD/Hodge, Kakeya in R⁴,
   asymptotic algorithmic improvements, Navier-Stokes) are registered
   `unsupported` — APEDP-X can check candidate structure numerically but
   cannot construct or state proofs.
2. **Originality adjudication.** Nothing in the pipeline checks whether a
   recovered law is already known; there is no literature/novelty referee.
3. **Coverage skew.** The machine-checkable portfolio is number-theory
   certificates plus ODE/PDE closed-form recovery. No algebraic-geometry,
   combinatorics, topology, or complexity engines exist to even attempt
   those families.
4. **Autonomy and scale.** The system is single-problem and
   human-in-the-loop: this sandbox runs ~6 checks per invocation against
   OpenAI's ~4000 attempted problems. Holdout budget is deliberately scarce.
5. **Formalization.** Lean artifacts are per-result templates (14+ under
   `apedp_dyn/lean_formal/`), not an automated discovery-to-Lean pipeline.
6. **Compute parity.** Commodity-scale seconds here vs. ~3 hours of frontier
   model thinking per average result there.

## Integrity contrast with the source release

- OpenAI's release shipped 722 manuscripts and later retracted 3, and never
  publicly reconciled 722 vs 719. This sandbox enforces the opposite by
  construction: `run_sandbox.run_all` raises if any task is reported
  `verified` without a non-empty certificate; a checker that runs and misses
  its threshold becomes `failed` (the retraction count); every verified task
  carries an explicit `claim_ceiling` (finite range only, uncertified tally,
  within-family exactness); and where a re-fit metric could inflate a number
  (PDE-1), the frozen-coefficient metric is the one reported.
- Pinned by `tests/test_openai_math_sandbox.py`: registry integrity, the
  headline-gap contract, the no-certificate-no-verified rule, deterministic
  law checks, PDE conditional semantics, and binary-missing tolerance.
