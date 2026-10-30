# Experiment D: what the pinned competition binary can compile

**Status:** measured, and the earlier framing is corrected · **Scope:** the DQI
decoder at the pinned instance (m=4095, n=70, GF(2^12), 64 shots) and the two
`kickmix` builds — the pinned competition extension that may produce a receipt
and the local `MR` rebuild that may not · **Evidence:**
`benchmarks/quantum_compilation/dqi_official_compatibility.json`, produced by
`scripts/measure_dqi_official_compatibility.py`; builds on
[`DQI_RESOURCE_RECONCILIATION.md`](DQI_RESOURCE_RECONCILIATION.md) §2 and
[`DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md).

`DQI_RESOURCE_RECONCILIATION.md` §2 records one sentence — `arm on the pinned
binary: built = False, AttributeError: 'Builder' object has no attribute 'mr'` —
and calls the missing `MR` opcode "the only blocker between 686,014.8 and a
receipt". Two things in that sentence are wrong, and this experiment measures
both:

1. it is not one blocker. The arm needs **two** builder surfaces it does not
   have: `mr` (for the reads every lever is gated on) and `gf_product` (for the
   54-AND product policy all the published arm figures are measured with). The
   second was not recorded before this run.
2. it is not true that nothing can be compiled. One candidate can, and this
   experiment builds, executes and costs it: the reference algorithm with the
   54-AND multiply emitted as *gates* and cleared with `hmr` instead of `mr`.

What no candidate can have is the levers. That distinction is the whole result:
the arm's −43.84% is a *read-dependent* saving, and the pinned binary has no
Z-basis measurement.

## 1. The two surfaces, measured

| surface | pinned competition binary | local `MR` rebuild | what it is for |
|---|---|---|---|
| `mr` (Z-basis measure + reset) | **no** | yes | every lever's classical condition |
| `h` (Hadamard) | **no** | no | would let `hmr` be composed into `mr` (§3) |
| `hmr` (X-basis measure + Z reset) | yes | yes | measurement-based clears |
| `gf_product` (product policy) | **no** | yes | the 54-AND multiply the arm is measured with |
| `bit_store`, `push_condition` | yes | yes | the classical condition machinery itself |
| `init_and`, `del_and`, `cswap` | yes | yes | the gate set |
| `counting_op_executions` (per-instruction counts) | **no** | yes | the campaign's region ledger |

So the pinned binary *has* the classical-control machinery and *has* a
measurement instruction — it simply cannot measure in the Z basis, and cannot be
asked for a product policy. The arm fails on the first read it takes, which is
why the recorded error names `mr` and stops there.

## 2. What `mr` is worth, and what `gf_product` is worth

Both figures are already measured elsewhere; this run re-measures the policy one
on the same instance so the two are comparable:

| lever | worth per shot | source |
|---|---|---|
| the five gates (read-then-gate) | **535,452.0** (1,221,466.8 → 686,014.8) | `dqi_schedule_campaign.json` |
| the 54-AND product policy | **523,849.1** on the reference (5,725,071.6 → 5,201,222.5) | this record, `reference_and54_local_only` |
| the six levers alone | 4,503,604.8 (5,725,071.6 → 1,221,466.8) | `gf54_native_opcode.json` |

The policy is not a detail: it is worth more than the entire gate campaign, and
it is unreachable on the pinned binary in both of its forms — as an attribute
(the assignment is rejected) and as a gate stream (the emitter clears its
workspace with `mr`, §3).

## 3. Why `hmr` cannot stand in for `mr`

`hmr` is an X-basis measurement followed by a Z-basis reset, so *H then `hmr`* is
`H·H·MR = MR`. That identity is exact, and
[`VQC_AMPLITUDE_TIER.md`](VQC_AMPLITUDE_TIER.md) §5 measures it at amplitude level:
on basis and superposed inputs alike, `h; hmr` and `mr` produce the same branch
structure. It needs an `H` gate, and the builder has none — measured here as
`'Builder' object has no attribute 'h'`. `hmr` on its own measures in the X basis,
so on a basis-state qubit it returns a *random* outcome (also measured at
amplitude level). It therefore cannot supply a *read*.

It can supply a **clear**, and the pinned binary already knows that: `del_and`
costs **0 Toffolis** on both builds, because an AND's target is a basis-state
qubit that can be measured away instead of uncomputed. That is the crack the
candidate in §5 walks through.

## 4. The identical-circuit check

The two builds are different *counting backends* for the same circuit, and this
record re-establishes it on the same 64 sampled syndromes at the same instance:

| | `Q` | `T_mean`/shot |
|---|---|---|
| pinned, reference (built-in multiply) | 1,893 | **5,725,071.609375** |
| local `MR` rebuild, same circuit | 1,893 | **5,725,071.609375** |

Per-shot arrays are identical element for element. The pinned row also
reproduces the recorded reference constant exactly, so the comparison in §5 is
against a floor that is not in dispute.

## 5. The one candidate the pinned binary can compile

`Gf54MulMixin` (`scripts/gf54_mul.py`) replaces `ixor_gf2_mul` with the 54-AND
circuit and clears its workspace with `mr`. Its clears are *clears*, not reads, so
substituting `hmr` for `mr` there is sound (§3) — and that substitution needs no
builder change at all, because the emitter calls the builder's `mr` method.

| | `Q` | `T_mean`/shot | score ratio to the shipped reference | correct |
|---|---|---|---|---|
| reference (built-in multiply) | 1,893 | 5,725,071.609375 | 1.000 | yes |
| **candidate: reference + gate-stream 54-AND + `hmr` clears** | **1,896** | **5,433,399.609375** | **0.952** | yes |
| reference + 54-AND as a *policy* | 1,912 | 5,201,222.531250 | 0.925 | yes — **local only** |
| the arm + policy + five gates | 1,913 | 686,014.800 | — | **not compilable** |

The candidate is **−5.09% `T`** and **+3 qubits**, so its score is **4.79% better**
than the shipped reference: `(T/T₀)·(Q/Q₀)² = 0.949054 × 1.003172 = 0.952064`.
It restores the syndromes on all 64 shots, it uses no instruction the pinned
binary lacks, and it is the first candidate in this repository that the
competition build can construct at all.

It is also 7.92× worse than the arm. The candidate is the *floor*, not a
contender: it proves the route exists (a clear can be a measurement where a read
cannot) while leaving the 43.84% where it is, behind a read.

## 6. What is still out of reach, and why that is structural

The arm's saving is not quantum-controlled work that could be made cheaper; it is
work that is *skipped*, under a condition that comes from a measurement of the
data. A quantum control cannot express it — the campaign's own control arm is the
proof: the reference's every gate is quantum-controlled and every one is executed,
which is what a 5,725,071.6 static-equals-executed figure means.

So the receipt question has a precise answer:

- **no lever** can be compiled by the pinned binary;
- **one candidate** can, at 5,433,399.6 per shot and `Q` = 1,896 — a 4.79% score
  improvement, not the 14.4× the portfolio target asks for;
- the gap between the two is `mr`, and `mr` is not an optimisation detail: it is
  the instruction that makes an executed count depend on the data.

`apedp_dyn/quantum_compilation/contracts.py` remains the rule: only the pinned
extension may produce a receipt, and the pinned extension cannot build the arm.
This experiment does not change that, and it does not claim a submission.

## 7. Pins

`tests/test_dqi_official_compatibility.py` pins: the pinned reference row at
1,893/5,725,071.609375 and the identical per-shot arrays on both builds; the
surface inventory (`mr` no, `h` no, `hmr` yes, `gf_product` no, classical
controls yes); the arm's failure on one build and success on the other; `del_and`
costing 0; the candidate's `Q` = 1,896 and `T` = 5,433,399.609375 with its
syndromes restored; and the record's own statement that no submission is claimed.

A future edit that calls the missing `MR` the only blocker, quotes the arm as a
submission, or reports a candidate without its score ratio fails there.
