# Representation-aware uncomputation: a checklist

**Status:** method note, derived from measurement · **Scope:** transforms that replace an
uncompute with a measurement-based clear, and what a synthesizer must prove before one may
be promoted · **Evidence:** `scripts/gf54_mul_reduced.py`,
`benchmarks/quantum_compilation/dqi_official_candidate.json`,
`benchmarks/quantum_compilation/dqi_submission_score.json`,
`benchmarks/quantum_compilation/dqi_arm_submission_score.json`. Discussion:
[`DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md) §§3, 5, 8, 9.

The one transformation in this repository that worked, worked for a reason worth reusing:

> Find the semantic precondition of an inverse primitive. Redesign the intermediate
> representation so it holds. Prove value, phase and workspace. Price it.

`del_gf2_mul` costs **0 Toffolis** and clears a product register *only if* the register
holds the reduced field product `lhs * rhs`. The old emitter never met that: it accumulated
an unreduced `2n-1`-bit polynomial sum and folded afterwards, so the primitive had no valid
fixup and the route appeared closed. Accumulating the product **already reduced** — CNOTs,
which are free — puts exactly `lhs * rhs` in a 12-qubit window, and the primitive applies.
Measured: 66 Toffolis per controlled call against 75, `Q` unchanged at 1,893, 0 value and 0
phase errors at 102,400 shots, score ratio 0.949054. The saving is not a cheaper gate; it is
a *different intermediate object* that makes a zero-cost gate legal.

## 1. What the three measured cases say

| clear | precondition | representation that meets it | cost | evidence |
|---|---|---|---|---|
| `del_and(c1, c2, t)` | `t` holds `c1 AND c2`, controls known | the AND itself, one target | **0 T** | Experiment D2: 0 phase flips on `\|0>` *and* on `\|1>` |
| `del_gf2_mul(lhs, rhs, w)` | `w` holds `lhs * rhs`, **already reduced** | product accumulated reduced into a 12-qubit window | **0 T**, window under the peak | Experiment D2/E: 66 vs 75 per call, `Q` unchanged |
| `hmr(t)` | **none** — an X-basis measurement | — | 0 T but **wrong** | Experiment D2: on `\|1>` it wrote 1 on 24/64 shots and flipped the phase on those 24; used as a clear it gave 37/64 phase errors |

The third row is the checklist's justification. `hmr` looks cheaper than every alternative
and is not a clear at all: its outcome is random, so no classically conditioned fixup can
restore the phase. A synthesizer that ranks candidate clears by Toffoli count alone will
choose it.

## 2. The checklist

Nine gates. Each names the failure it prevents, and every one of them has been the actual
cause of a wrong result at least once in this project.

1. **Name the primitive and quote its precondition from the source, not from memory.**
   *Prevents:* assuming a measurement is a universal clear (the `hmr` route, 37/64).
2. **State what the register holds *as a mathematical object* at the clear point.**
   *Prevents:* calling a scratch lane "scratch" when it holds a sum of products, which is
   not what any fixup primitive consumes.
3. **Prove the register is in the precondition's form on every path.** Both operand-parity
   paths, and every early exit. *Prevents:* a design that is correct on the measured path
   and wrong on the other.
4. **Check the phase contract: is the fixup conditioned on bits derived from known inputs?**
   If any conditioning bit is a measurement outcome, the clear is a randomiser. *Prevents:*
   the exact defect that withdrew Experiment D's 0.952.
5. **Check the qubit ledger against the existing peak, not against zero.** The reduced
   window fits under 1,893; the design that kept 54 AND targets alive would have cost +54 on
   `Q`, and `Q` is squared in the score. *Prevents:* a `T` saving that a `Q` penalty erases.
6. **Predict the Toffoli delta per call and pin the arithmetic identity before building.**
   `9 per call × 32,408 calls = 291,672`, matched to the static stream exactly.
   *Prevents:* a headline that is a decomposition nobody checked.
7. **Verify value, junk and phase — all three — on both paths, on the pinned extension.**
   *Prevents:* "syndromes restored" being reported as correctness; 13 of 21 records in this
   repository had exactly that gap (`correctness_audit.json`).
8. **Only then score: run the competition's `verify()` at its own budget.** A ratio measured
   on 64 hand-scored shots is a rule application, not a submission.
9. **Record what was *not* measured, and on which build.** Experiment F's arm row is real
   and is not a receipt; the record says so in three places.

Gates 3, 4 and 7 are the ones that decide correctness; 5 and 6 decide whether the saving is
real; 8 and 9 decide whether the claim is honest.

## 3. Where this plugs into the synthesizer

- **Candidates.** `apedp_dyn/quantum_compilation/candidates.py` carries
  `CircuitCandidate` and `CandidateStatus`. A transform that clears by measurement should
  emit gates 1–6 as structured fields on the candidate; a candidate missing 3, 4 or 7 must
  not be labelled correct, and `CandidateStatus` is the place to refuse it.
- **Referees.** Gates 7 and 8 are `referee.run_dqi_verifier` and
  `referee.dqi_verification_scope` — the only scoring path, and the only one whose numbers
  may be quoted.
- **Contracts.** `apedp_dyn/quantum_compilation/contracts.py` still decides which build may
  produce a receipt. A transform that only runs on a local rebuild is a `experimental`
  result, not a submission (Experiment F).
- **Reference implementation.** `scripts/gf54_mul_reduced.py` is the worked example: a
  reduced-product accumulator whose every clear is a fixup primitive, with the phase and
  workspace obligations discharged in `tests/test_gf54_mul_reduced.py`.

## 4. What the checklist does not claim

It is not a minimality proof: 66 Toffolis per call is cheaper than 75, and the 54-AND
base case is not proved optimal (the dimension bound suggests 23 at wider inputs, and none
is in hand). The external comparison has now been run, but only on the piece a
Clifford+T optimizer can express: given the multiply's forward AND netlist, PyZX 0.10.7
leaves the T-count at **378 → 378** (54 × 7) and only removes Clifford work, and its
T-reducing pipeline did not return within 90 s at 99 qubits; our clears are 0 Toffoli
because they are measurements, so a unitary optimizer's honest figure is 108 ANDs
(forward + adjoint), not 54 (`benchmarks/quantum_compilation/gf_mul_external_optimizer.json`).
It grants no licence to read a qubit's value into a classical bit: the pinned binary
cannot, and no clear in this checklist needs to.

## 5. Pins

`tests/test_representation_aware_uncomputation.py` pins the three precondition rows above
against the records that measured them (`del_and` 0 flips, `hmr` 24/64 and not a read,
`del_gf2_mul` 66 vs 75 with `Q` unchanged), and pins that the nine gates, the three
correctness gates and the plug-in points are present in this note. A future edit that drops
gate 3, 4 or 7, or that presents `hmr` as a clear, fails there.
