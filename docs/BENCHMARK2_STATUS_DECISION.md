# Benchmark II: the status decision

**Status:** decision record, 2026-10-30 · **Supersedes:** the implicit "still open" reading
of [`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) and
[`DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md) · **Evidence:** the six
records those notes cite, each measured by the competition's own `verify()` at 102,400
shots where a score is quoted.

This note does not add a measurement. It records what the measurements already decide, so
that the next reader does not have to re-derive it — and so that "the benchmark is still
open" stops being the default reading of a folder full of records.

## D1. The pinned competition binary is frozen at 0.949054

The best correct circuit the pinned build can compile is the reference with the
reduced-window 54-AND multiply: **`Q` = 1,893, 5,433,438.2 Toffolis per shot, 0 value and
0 phase errors at 102,400 shots, score 2.319438e13, score ratio 0.949054**. That is the
only submission-grade result on the pinned build, and it is **13.66×** above the derived
target of 1.6975e12.

No further pinned-binary circuit work is planned. The reason is arithmetic, not
preference: the multiply is 42.45% of the reference's static Toffoli stream and Experiment
E already took the part of that share the pinned primitive allows (9 of 75 per controlled
call); deleting *all* of it, which no emitter can do, would still leave the reference
8.29× out. A general-purpose optimizer does not close it either: PyZX 0.10.7 leaves the
multiply's forward T-count at 378 → 378.

## D2. `MR` is the only condition that reopens it

[`MR_PATCH_REQUEST.md`](MR_PATCH_REQUEST.md) asks for one opcode — Z-basis measure-and-reset
— with no verifier change. It is the only surface change that moves the order of magnitude:
4.142× on the reference (to 3.48× out) and 4.5898× on the arm. It is specified, patched
(`scripts/kickmix_mr_opcode.patch`) and locally built
(`scripts/build_kickmix_mr.py`), and the arm that uses it has been scored:
**5.324728e12** for the six-lever arm (3.137× the target) and **2.990339e12** for the
campaign's stacked arm (**1.762×** the target).

The decision: **the benchmark stays closed until an upstream build ships `MR` or an
equivalent Z-basis read.** If that happens, the reopen is mechanical — rebuild the pinned
line against the new surface, re-run `scripts/measure_dqi_submission_score.py` and
`scripts/measure_dqi_campaign_arm_submission_score.py`, and re-score. If it does not, the
pinned line stays at 0.949054 and nothing else in this repository is "in progress" on it.

## D3. No more field-arithmetic optimisation

The multiply is no longer where the gap is. Even with every product and the controlled add
removed — neither physically available — the arm is still 1.062× the target, because
472,768.2 Toffolis per shot are not field products at all. After the campaign the remaining
240,785.8 are the swap, the shift, `shift_step`, `update_flag` and the rest of the multiply
window.

**Corrected 2026-11-04 (§7.13 of the reachability note).** The 1.062× above prices the six-lever
arm's 472,768.2 at the campaign arm's width. The campaign arm's own residual is 299,953.2, so
freeing *its* products scores **0.688×** and the target sits 145,275.8 *above* this arm's floor.
The decision below stands, on a different ground: reaching 445,229 needs the surviving 386,061.7
of products at 37.6% of their cost — about 15.6 ANDs per multiply, below the `2n − 1 = 23` lower
bound — and at that bound the arm is 1.019× at 1,913, 0.998× only with the 20 qubits returned.

The decision: no further multiply/AND-base-case work is planned, and the 23-AND dimension
bound is recorded as the only remaining field-arithmetic lever rather than as a task. The
next class of change is a decoder algorithm — a shorter EEA, a different error locator, or
a construction that does not evaluate 4,095 positions per shot — and its measured plan is
[`EEA_AND_PREDICATE_PLAN.md`](EEA_AND_PREDICATE_PLAN.md). That work is *not* started.

## D2a. The blocked dependency, named

D2 says the benchmark reopens when upstream ships `MR`. That is a plan with no owner, and
it should be read as one:

| what is needed | who owns it | state |
|---|---|---|
| `mr` (Z-basis measure-and-reset) in the pinned extension | **upstream, not this repository** | specified, patched, built locally; no upstream commit, no issue filed, no maintainer contact in the tree |
| `gf_product` (54-AND product policy) on the pinned binary | **upstream** | absent as an attribute and as a gate stream; the emitter's own clear uses `mr` |
| `h` on the builder | **upstream** | absent, which is why `h; hmr` cannot compose into `mr` |

So the honest status of this line is **blocked**, not **in progress**. The 4.142× / 4.320×
figures are measurements on a local rebuild and are labelled a surrogate everywhere they
appear; nothing in the repository moves the pinned score until one of those three surfaces
exists. `MR_PATCH_REQUEST.md` is the request; filing it is a delivery step this repository
cannot perform, and that is stated there rather than implied here.

## What continues, and on what evidence

Two things from this line of work do continue, and neither is a Toffoli count:

- **The method.** [`REPRESENTATION_AWARE_UNCOMPUTATION.md`](REPRESENTATION_AWARE_UNCOMPUTATION.md)
  states and gates it: find the inverse primitive's precondition, redesign the intermediate
  representation to meet it, prove value/phase/workspace, price it. It produced the only
  correct improvement the pinned build has.
- **The referee.** The three-check discipline is now enforced at runtime
  (`CircuitCandidate.correctness_evidence`, `EvidenceIncompleteError`,
  `ledger.record_verification` refusing to promote an unsupported claim), because 13 of 21
  records in this repository once claimed correctness on partial evidence and one of them
  was wrong on 37 of 64 shots.

## What would overturn a decision

- **D1:** a pinned-build circuit whose `verify()` score is below 1.6975e12, or below
  2.319438e13 with a smaller `Q`.
- **D2:** an upstream `MR` (or any Z-basis read) in the shipped extension; or a
  measurement-free construction that reproduces the arm's saving without one.
- **D3:** a field-arithmetic result that reaches the 23-AND bound at 12 bits, or reduces the
  campaign arm's non-product 299,953.2 (the six-lever 472,768.2, not that arm's own) — the two
  things §2 of the reachability note says would be needed, and neither is in hand.

## Pins

`tests/test_benchmark2_status_decision.py` pins this note to the records it summarises: the
frozen pinned score and ratio come from `dqi_submission_score.json`, the arm scores from
`dqi_arm_submission_score.json` and `dqi_campaign_arm_submission_score.json`, the
multiply-share and non-product figures from the reachability note, and the three decisions
each name their reopen condition. A future edit that reopens the benchmark without one of
those conditions, or that quotes a frontier figure the records do not carry, fails there.
