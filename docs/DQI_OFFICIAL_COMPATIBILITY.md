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
2. it is not true that nothing can be built. One candidate can, and this
   experiment builds, executes and costs it: the reference algorithm with the
   54-AND multiply emitted as *gates* and cleared with `hmr` instead of `mr`.

What no candidate can have is the levers. That distinction is the whole result:
the arm's −43.84% is a *read-dependent* saving, and the pinned binary has no
Z-basis measurement.

**Amended by Experiment D2 (§3, §5).** The candidate *builds* and *executes*, and
it is not a valid submission: `hmr` is not phase-safe, so it returns the wrong
phase on 37 of 64 shots and the verifier charges each one. The corrected result
is narrower than the one above: **no correct circuit cheaper than the shipped
reference can be compiled by the pinned binary**, and the cheapest correct
circuit it can compile is the reference itself. Experiment D's `0.952` score
ratio is withdrawn; its Toffoli count stands.

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
qubit that can be measured away instead of uncomputed.

**Correction (Experiment D2).** This section originally concluded that `hmr` could
supply a clear, and §5 built a candidate on that. It cannot: `hmr` is not
*phase-safe*, and this circuit is scored on phase. Measured one qubit at a time
(`scripts/measure_dqi_official_candidate.py`, `clear_phase_safety`), 200 shots
each:

| clear | on `|0>` | on `|1>` | measured ones on `|1>` |
|---|---|---|---|
| `hmr` | 0 phase flips | **97 phase flips** | 97 |
| `del_and` | 0 phase flips | **0 phase flips** | — |

The 97 flips *are* the 97 measurement outcomes: an X-basis measurement of `|1>`
puts a factor of −1 on the branch that measures 1, and kickmix tracks that factor
as a phase flip. `del_and` measures the same way and then applies a classically
conditioned `CZ(control1, control2)` — the phase fixup — which is why the
reference's own measurement-based uncomputation is phase-correct. The fixup needs
the value to be a *known* AND, so `del_and` clears an AND and nothing else.

`mr` is therefore load-bearing for the same reason it was: it is the only clear
whose outcome is the state's own value. A clear of a window holding a *sum* of
products, or of a scratch lane holding a syndrome, has no fixup to apply.
That is the crack §5 walked through, and it is not load-bearing.

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

## 5. The candidate the pinned binary can compile — and its phase

`Gf54MulMixin` (`scripts/gf54_mul.py`) replaces `ixor_gf2_mul` with the 54-AND
circuit and clears its workspace with `mr`. Substituting `hmr` for `mr` there
needs no builder change at all, because the emitter calls the builder's `mr`
method — and it produces a circuit the pinned binary *executes*.

Whether the pinned binary can compile a **valid submission** is a different
question, and it is the one the score asks. Every arm below was run on the pinned
extension at the pinned instance, 64 shots, and scored with the verifier's own
rule (`phase_err = (phase_flipped ^ phases) & ~val_err`):

| | `Q` | `T_mean`/shot | score ratio to the shipped reference | value errors | **phase errors** |
|---|---|---|---|---|---|
| reference (built-in multiply) | 1,893 | 5,725,071.609375 | 1.000 | 0/64 | **0/64** |
| reference + 54-AND gate stream, `hmr` clears | 1,896 | 5,433,399.609375 | 0.952 | 0/64 | **37/64** |
| forward EEA on a CNOT copy, scratch cleared by `hmr` | 2,736 | 3,438,873.609375 | 1.255 | 0/64 | **35/64** |
| reference + 54-AND as a *policy* | 1,912 | 5,201,222.531250 | 0.925 | — | — **local only** |
| the arm + policy + five gates | 1,913 | 686,014.800 | — | — | **not compilable** |

So the 54-AND candidate's `T` is real and its **score ratio 0.952 is withdrawn**:
37 of 64 shots return the wrong phase, which is the coin flip §3 measures, and the
verifier charges each one as a failing shot. It is compilable and it is *not* a
submission. The second row tests the largest lever the first row leaves alone —
run the EEA on a CNOT copy of the syndromes, so nothing has to be restored and no
qubit's value is read, which is exactly what the `clear_eea` lever needs — and it
fails the same way, at 35/64, for the same reason: its clears are `hmr`.

(It also costs 843 more qubits, because the whole scratch stays live until the
single clear at the end, so `(Q/Q₀)²` alone puts it at 1.255: worse than the
reference even if its phase were right.)

**The cheapest correct circuit the pinned binary can compile is the shipped
reference, 5,725,071.609375 at `Q` = 1,893.** That is the measured floor, not a
conjecture: every cheaper construction in this repository needs a clear whose
value is a function of the data, and the only phase-safe clear the binary has
(`del_and`) needs the value to be a *known AND*. A 54-AND multiply whose windows
were cleared with `del_and` instead would have to keep all 54 ANDs live — 54
scratch qubits against the 19-qubit rolling window — and `+54` on `Q` costs 5.8%
in `(Q/Q₀)²` against the 5.1% its `T` saves, so that route is at best a wash and
is not claimed here as a measurement.

## 6. What is still out of reach, and why that is structural

The arm's saving is not quantum-controlled work that could be made cheaper; it is
work that is *skipped*, under a condition that comes from a measurement of the
data. A quantum control cannot express it — the campaign's own control arm is the
proof: the reference's every gate is quantum-controlled and every one is executed,
which is what a 5,725,071.6 static-equals-executed figure means.

So the receipt question has a precise answer:

- **no lever** can be compiled by the pinned binary;
- **no correct candidate cheaper than the reference** can be compiled either: the
  two that build execute, and both fail on phase (37/64 and 35/64);
- the gap between the reference and the arm is `mr`, and `mr` is not an
  optimisation detail: it is the instruction that makes an executed count depend
  on the data, and its phase behaviour is what makes a measurement usable as a
  clear rather than as a randomiser.

`apedp_dyn/quantum_compilation/contracts.py` remains the rule: only the pinned
extension may produce a receipt, and the pinned extension cannot build the arm.
This experiment does not change that, and it does not claim a submission.

## 7. Pins

`tests/test_dqi_official_compatibility.py` pins: the pinned reference row at
1,893/5,725,071.609375 and the identical per-shot arrays on both builds; the
surface inventory (`mr` no, `h` no, `hmr` yes, `gf_product` no, classical
controls yes); the arm's failure on one build and success on the other; `del_and`
costing 0; the candidate's `Q` = 1,896 and `T` = 5,433,399.609375; and the
record's own statement that no submission is claimed.

`tests/test_dqi_official_candidate.py` pins Experiment D2: `hmr` flips the tracked
phase on |1> and `del_and` does not; the reference's 0/64 phase errors; the
54-AND candidate's 37/64 and the copy-EEA candidate's 35/64 at `Q` = 2,736; and
that no record quotes a score ratio for a candidate with phase errors.

A future edit that calls the missing `MR` the only blocker, quotes the arm as a
submission, quotes 0.952 as an attainable score, or reports a candidate without
its phase errors fails there.
