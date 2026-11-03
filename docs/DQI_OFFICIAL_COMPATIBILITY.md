# Experiment D: what the pinned competition binary can compile

**Status:** measured, and the earlier framing is corrected · **Scope:** the DQI
decoder at the pinned instance (m=4095, n=70, GF(2^12), 64 shots) and the two
`kickmix` builds — the pinned competition extension that may produce a receipt
and the local `MR` rebuild that may not · **Evidence:**
`benchmarks/quantum_compilation/dqi_official_compatibility.json`, produced by
`scripts/measure_dqi_official_compatibility.py`; builds on
[`DQI_RESOURCE_RECONCILIATION.md`](DQI_RESOURCE_RECONCILIATION.md) §2 and
[`DQI_MOVEMENT_SCHEDULE.md`](DQI_MOVEMENT_SCHEDULE.md).
Experiment E (§8) closes this record's last open item with
`benchmarks/quantum_compilation/dqi_submission_score.json`, produced by
`scripts/measure_dqi_submission_score.py`.

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
phase on 37 of 64 shots and the verifier charges each one. Experiment D's `0.952`
score ratio is withdrawn; its Toffoli count stands.

What replaces it is better, not narrower. There is a **second phase-safe clear** in
the pinned binary — `del_gf2_mul` (`gen_gf_unmul`), whose fixups are computed from
the two factors instead of from an AND — so the 54-AND multiply can be emitted with
no `mr`, no `hmr` and no extra qubits, provided its window is the thing that
primitive knows how to clear: a **reduced-product** register. Accumulating the
field product already reduced (`scripts/gf54_mul_reduced.py`) does exactly that.
Measured at the pinned instance: **`Q` 1,893, 5,433,399.609375 per shot, 0 value
errors, 0 phase errors, junk clean — a 5.09% score improvement**, the first correct
candidate cheaper than the shipped reference that the competition build can compile.

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

It can supply a **clear**, and the pinned binary already knows that: `del_and` and
`del_gf2_mul` both cost **0 Toffolis** on both builds, because a basis-state
register can be measured away instead of uncomputed — **as long as the value is
known**: `del_and`'s fixup is `CZ(control1, control2)`, and `del_gf2_mul`'s is the
masked parity of `lhs * rhs` (§5). A sum of products that is neither is the case
`hmr` tempts you into, and §3's table is why that temptation is a dephasing.

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
| **reference + reduced-window 54-AND (`del_and` + `del_gf2_mul`)** | **1,893** | **5,433,399.609375** | **0.9491** | **0/64** | **0/64** |
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

The first row is the one that stands. `scripts/gf54_mul_reduced.py` accumulates the
field product **already reduced** — CNOTs, so free — into a 12-qubit window instead
of an unreduced `2n-1`-bit polynomial window, which makes the one register that has
to be cleared hold exactly `lhs * rhs`. That is `del_gf2_mul`'s precondition, and its
measurement is followed by classically conditioned `CZ` fixups whose parity is
computed from the factors, so the tracked phase is restored rather than left to the
measurement outcome. Every AND target is cleared by `del_and` on the same footing.
The result costs **66 Toffolis per controlled call** against the built-in 75, and
**no extra qubits at all**: the 12-qubit window fits under the reference's peak, so
`Q` stays at 1,893 and the whole 5.09% `T` saving reaches the score. (The earlier
estimate that this route needed 54 live AND targets, `+54` on `Q`, was for a
different design — keeping the ANDs to clear an unreduced window — and is not what
this one does.)

What remains true is the boundary: **no lever can be compiled**, because a lever's
condition is a *read*, and no phase-safe clear supplies one. The saving this
candidate carries is the multiply's, not the schedule's.

## 6. What is still out of reach, and why that is structural

The arm's saving is not quantum-controlled work that could be made cheaper; it is
work that is *skipped*, under a condition that comes from a measurement of the
data. A quantum control cannot express it — the campaign's own control arm is the
proof: the reference's every gate is quantum-controlled and every one is executed,
which is what a 5,725,071.6 static-equals-executed figure means.

So the receipt question has a precise answer:

- **no lever** can be compiled by the pinned binary;
- **one correct candidate cheaper than the reference** can be, at 5,433,399.609375
  per shot and `Q` = 1,893 — a 5.09% score improvement, not the 14.4× the portfolio
  target asks for, and not a submission until it is run through the verifier's
  shot sample end to end;
- the gap between that candidate and the arm is `mr`: not an optimisation detail but
  the instruction that makes an executed count depend on the data. Its phase
  behaviour is separately load-bearing — it is what makes a measurement usable as a
  clear rather than as a randomiser, which is why the fixup primitives are the only
  clears this candidate uses.

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
54-AND candidate's 37/64 and the copy-EEA candidate's 35/64 at `Q` = 2,736; the
reduced-window candidate at `Q` = 1,893 and 5,433,399.609375 with **0/64 value and
0/64 phase errors** (score ratio 0.949054), as the only correct arm cheaper than the
reference; and that no record quotes a score ratio for a candidate with phase
errors.

`tests/test_gf54_mul_reduced.py` pins the emitter itself on the pinned extension:
the product correct on both paths, junk clean, the tracked phase untouched, 54 and
66 Toffolis, no `MR ` or standalone `H` in the stream, and every `HMR` accompanied
by a classically conditioned `CZ`.

A future edit that calls the missing `MR` the only blocker, quotes the arm as a
submission, quotes 0.952 as an attainable score, reports a candidate without its
phase errors, or claims the `hmr` clears are safe fails there.

## 8. Experiment E: the score the competition's own verifier assigns

§5 scored the candidate with the verifier's *rule*, applied by hand to 64 shots, and
§6 ended on the one thing that was left: "not a submission until it is run through
the verifier's shot sample end to end". This section runs `verify()` itself, from
`verifiers/dqi/verify_submission_dqi.py` — **every scoring line unmodified**, with
one *diagnostic-only* stub for `reaction_depth` (§8, "One line of `verify()` is not
run") — on both circuits at the competition's own budget — **102,400 shots** (400
batches of 256) at `log2_delta = -128`. Record:
`benchmarks/quantum_compilation/dqi_submission_score.json`.

The stub is not a modification of the verifier's logic and the record carries the
exact patch (`verifier_patch`: the replaced line, the replacement, both SHA-256s and
the reason). `tests/test_dqi_submission_score.py` proves the claim instead of
asserting it: the patch applies byte-exactly to the shipped file, `reaction_depth`
occurs once in the whole module, and an AST comparison shows the patched module
differs from the shipped one in exactly one dict value inside `verify`'s return —
so it cannot reach the `PASS` decision, the value or phase checks, `eps_plus`, `R`
or the score.

The budget is not decoration. `eps_plus` at `log2_delta = -128` is
`1 - 2^(-128/n)`, so even a *zero-error* run is refused below ~1,375 shots: at 256
shots the same passing circuits get `eps_plus = 0.2929` and the verifier raises
"Effective error rate … >= 1/16". The smallest batch-aligned budget a zero-error run
can pass at is **1,536** shots, and the record verifies at the full official
102,400 — a fraction of the official budget of 1.0.

| | `Q` | static `T` | executed `T`/shot | value errors | phase errors | `eps_plus` | `R` | **score** |
|---|---|---|---|---|---|---|---|---|
| reference (built-in multiply) | 1,893 | 5,725,461 | 5,725,108.382109 | 0/102,400 | 0/102,400 | 8.660587e-4 | 1.191260 | **2.443946e13** |
| reduced-window 54-AND (`del_and` + `del_gf2_mul`) | 1,893 | 5,433,789 | 5,433,438.229658 | 0/102,400 | 0/102,400 | 8.660587e-4 | 1.191260 | **2.319438e13** |

Both `PASS`. The score ratio is **0.949054** (0.94906 to five places) — a **5.09%
improvement**, and the ratio is *exactly* the Toffoli ratio: `Q` is equal, both runs
have zero errors, so they share `eps_plus` and `R`, and the comparison carries no
error-rate or qubit penalty in either direction. In the executed-mean column the
candidate reads 38.6 Toffolis above D2's 5,433,399.609375 (the reference 36.8 above
its own 5,725,071.609375); the static column is the exact one, and those
differences are shot noise on the classically conditioned Toffolis, not a change of
circuit.

### Where the 5.09% comes from, to the gate

The candidate replaces only `ixor_gf2_mul`. The reference emits **32,408** multiply
calls, and at 75 Toffoli-like gates per call against 66 that is `9 x 32,408 =
291,672`. Measured on the static streams: 5,725,461 − 5,433,789 = **291,672**,
exactly. Measured on the executed means: **291,670.152** per shot — 1.85 gates below the
static delta, which is the same quantity to 6.4e-6 relative, and the 1.85 is the
conditioned Toffolis. The operation deltas agree to that place: `CCX` differs by
29,867,023,611 across 102,400 shots, i.e. 291,670.152 per shot, and no other
Toffoli-like operation moves at all (`CSWAP` does not appear among the deltas). The
large `CX`, `HMR` and `CZ` deltas are not savings but the shape of the change: the
multiply is emitted as gates, so the candidate's stream is 104,555,982 operations
against 86,051,014, and what those extra Clifford-like gates buy is the Toffoli.
No gate is saved anywhere but in the multiply, which is **42.45%** of the
reference's static stream: the candidate removes **5.0943%** of that stream
(5.0946% of the executed mean).

### Why the levers are still out, measured rather than asserted

§6 said a lever cannot compile because its condition is a *read*. That is now a
property of the binary surface, not an inference. Of the **29** methods in the
pinned extension whose signature mentions a classical bit (`km.b`), exactly **one**
writes a bit from qubit data — `hmr` — and it is not a read: on a qubit prepared in
`|1>` it wrote 1 on **24 of 64** shots and flipped the tracked phase on exactly
those 24, so its outcome is a measurement and not the qubit's value. Every other
method only consumes a bit as a condition. The classical-condition machinery is
there (`push_condition`, `bit_store`); the instruction that would put a qubit's
value into it is not.

Two further consequences, stated because they are the honest boundary of the
record:

- **Absolute score, not just the ratio.** The competition's published target is
  `R(eps_plus) * 1,789^2 * 445,229 = 1.6975e12`. The candidate's `2.319438e13` is
  **13.66x above** that target; the reference is 14.40x above. The 5.09% is what
  the pinned binary permits, and the remaining factor is the schedule — reachable
  only through `mr`.
- **One line of `verify()` is not run.** `reaction_depth` is a reported diagnostic,
  not part of the score; on the 86M-operation reference it peaked at 13.9 GB and
  136 s, which gets the process killed here. It is stubbed in a temporary copy and
  relocated into `verifier_patch` in the record; every scoring line is the shipped
  one. The run is therefore the competition verifier with a documented
  diagnostic-only shim, not a literal unmodified invocation — and the shim's
  isolation from the verdict is pinned by test, not by this sentence.

For reference, the verifier took 125.1 s on the reference and 140.3 s on the
candidate at 102,400 shots (builds 15.4 s and 80.7 s), so this is a run anyone with
the pinned bundle can repeat.

**The 1,789/445,229 target used above is not derived.**
`scripts/measure_dqi_recycling_offsets.py` reads the decoder's own `n_A`, `n_B`, `n_a`,
`n_b` after every EEA sub-step: the a-pair is constant at `n + 1` rather than `n`, and
the b-pair varies, so only `n_a` can be recycled — 8 qubits at `n = 70`, a supported
peak of **1,885**, not 1,789. Degree recycling changes which registers exist and never
which gates run, so it says nothing about 445,229. Both figures are targets the campaign
measures against; see `RECYCLING_MEASUREMENT` in
`apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py`.

### Pins

`tests/test_dqi_submission_score.py` pins: both circuits `PASS` with zero value and
zero phase errors at the official 102,400 shots and the same `eps_plus` and `R`;
the 1,536-shot floor and the 256-shot refusal; `score = R * Q^2 * T` recomputed from
each row's own numbers; the score ratio equal to the Toffoli ratio at `Q` = 1,893
and equal to 0.94906; the decomposition `9 x 32,408 = 291,672` against both the
static and the executed streams, and `CCX / shots` equal to the executed delta; the
29-method bit surface with `hmr` the only writer and not a read; and the record's
`reaction_depth` patch.

A future edit that quotes a score ratio for this candidate other than its Toffoli
ratio, reports a budget below 102,400 as official, claims the levers can be
compiled, or describes `hmr` as a read fails there.

## 9. Experiment F: the arm's score, measured instead of multiplied out

§6 and §8 leave the other half of the frontier as a product: the six-lever arm
executes 1,221,466.8 Toffolis per shot at `Q` = 1,913, so its score *would be*
`R * Q^2 * T`. That product had never been run through `verify()`, and the arm cannot
run on the pinned binary at all. This section measures it on the local `MR` rebuild
(`scripts/build_kickmix_mr.py`: the vendored sources at their recorded commit plus
`scripts/kickmix_mr_opcode.patch`), at the competition's own budget — **102,400
shots**, `log2_delta = -128`, the 54-AND product policy. Record:
`benchmarks/quantum_compilation/dqi_arm_submission_score.json`.

| | `Q` | static `T` | executed `T`/shot | value errors | phase errors | `R` | **score** |
|---|---|---|---|---|---|---|---|
| pinned reference (Experiment E) | 1,893 | 5,725,461 | 5,725,108.382109 | 0/102,400 | 0/102,400 | 1.191260 | **2.443946e13** |
| **six-lever arm, MR rebuild** | **1,913** | 3,132,510 | **1,221,408.044756** | **0/102,400** | **0/102,400** | 1.191260 | **5.324728e12** |
| derived Benchmark II target | 1,789 | — | 445,229 | — | — | 1.191260 | 1.6975e12 |
| sourced projection at the measured peak | 1,885 | — | 445,229 | — | — | 1.191260 | 1.8846e12 |

Three things this settles that no earlier record could:

- **The arm is correct by the verifier's own standard.** Value *and* phase errors on
the sampled shots, with its classical-condition reads (`mr`) in the stream — not a
hand-applied rule on 64 shots.
- **The arm reproduces its recorded row.** 1,221,408.044756 against 1,221,466.8 is a
relative difference of 4.8e-5; `T` is a mean over sampled syndromes, and the static
count of 3,132,510 is exact.
- **The arm's score is measured, and it is 4.5898x the reference's** — a score ratio
of **0.2179**, exactly the `Q^2 T` ratio the reachability note computed by hand
(1,913² × 1,221,408.0 against 1,893² × 5,725,108.4). Both rows have zero errors, so
`eps_plus` and `R` are shared and carry no penalty either way.

**It is still not a win.** The target is `R * 1,789^2 * 445,229`; at `Q` = 1,913 the
same score needs `T` = **509,088**, and the arm executes 1,221,408 — **2.743x** on the
Toffoli count, **1.143x** on the qubit term, **3.137x** on the score. So `MR` is
necessary and not sufficient: the remaining factor is the decoder algorithm, not an
opcode.

**This is not a receipt.** `apedp_dyn/quantum_compilation/contracts.py` lets only the
pinned extension produce one, and the pinned extension has no `mr`; the record says so
in `status`, in `kickmix_role` and in `scope`, and no submission is claimed. The run
costs 6.7 s to build and 236.8 s to verify, so it is repeatable by anyone holding the
vendored sources.

### Pins

`tests/test_dqi_arm_submission_score.py` pins: `PASS` with zero value and zero phase
errors at 102,400 shots; `Q` = 1,913 and `T` within 0.15% of 1,221,466.8 with the
exact static 3,132,510; `score = R * Q^2 * T` to 1e-12; the like-for-like ratio
0.2179 (= 4.5898x) against the pinned reference at the same budget; the target gap
3.137x with `T` needed 509,088; and the record's own statement that it is not a
receipt.

A future edit that quotes the arm as a submission, reports its score without the
build it was measured on, or claims it reaches the target fails there.

## The two numbers the target row cannot supply

- **`Q` = 1,789 needs 96 qubits that do not exist.** Five degree registers at 8 qubits each
  is 40; one is recyclable and has been counted (1,885). The other four are 32, so the floor
  under any recycling is **1,853**. 96 > 32: closed by counting registers, not by a missing
  derivation.
- **`T_mean` = 445,229 is permanently unsourced.** No circuit produces it. The measured arms
  are 686,014.8, 1,221,466.8 and 472,768.2, and the last of those already assumes every field
  product and every controlled add are free, which they are not. The projection is therefore
  one measured factor (`Q`) times one aim (`T`), and says so in `target_basis`.

Committed artifact: `benchmarks/quantum_compilation/dqi_design_projection.json`, written by
`scripts/write_dqi_design_projection.py` and pinned to the live function by
`tests/test_dqi_gluzdov_v6.py`.

## The floor by register count

`peak = 1,853 + 8k` over `k` materialised degree registers, anchored on the two measured
points (`k = 5` → 1,893, `k = 4` → 1,885). The **zero-register floor** is 1,853, so even
allocating none of them leaves 64 qubits to find: 1,789 is not a register-count question.
Model: `model_degree_register_floor()`; carried in
`benchmarks/quantum_compilation/dqi_design_projection.json`.

## Nothing measured runs at the floor

The narrowest scored circuit is 1,893 qubits and both `MR` arms run at 1,913, so 1,853 is a
model rather than a build. Rescaled by `(1853/Q)²` at their measured Toffoli counts, the arms
score **2.222e+13**, **4.996e+12**
and **2.806e+12** — the best of them still
**1.489×** the target. Width is worth ~6% of the score, not
the gap. Model: `model_score_at_qubit_floor()`.

## Width levers sum to 55 of the 64 qubits needed

Measured width savings here are 32 qubits of degree-register recycling plus 23 from the
workspace window (42 → 19), which is **55 against the 64** the target needs — and the window
figure is an upper bound, since it was measured at 1,935 qubits. The remaining 9 qubits are
data-path width: one polynomial array is 852 qubits. Model: `model_data_path_floor()`.

## One array, if the algorithm allowed it

One polynomial array is 852 bits: held singly, the peak would be **1,001** and `Q` would stop
binding. But the EEA reads both remainders every round, so the dropped one must be re-derived
from the syndrome — one extra division scaffold per round, 46,656 Toffolis, which over 222
rounds is **2.81×** the reference at **0.786×** its score. Favourable arithmetic, unproven
construction; the assumption to falsify is that re-derivation costs one division and nothing
else. Model: `model_single_polynomial_array()`.

