# Experiment C: the A/B exchange cannot be scheduled away, and here is why

**Status:** measured, negative, with a correction to the campaign it builds on ·
**Scope:** the wholesale `(A, B)` exchange of the DQI decoder's synchronized EEA —
64,224 executed Toffolis per shot in the 686,014.8 incumbent (72 firings of
892) — at the pinned instance (m=4095, n=70, GF(2^12), 64 shots, 222 rounds) ·
**Evidence:** `benchmarks/quantum_compilation/dqi_movement_schedule.json`,
produced by `scripts/measure_dqi_movement_schedule.py` on the local `MR` rebuild;
builds on [`EEA_SCHEDULE_CAMPAIGN.md`](EEA_SCHEDULE_CAMPAIGN.md) §4 and answers
its §5 obligation for the exchange.

`EEA_SCHEDULE_CAMPAIGN.md` §4 enumerates what the 686,014.8 arm still spends:
386,061.7 on field products and 299,953.2 on non-product work, the largest
untouched items being the exchange's 64,224 and the shift's remaining 96,584.6.
It closes both "for this schedule" and names the only remaining route — **a
different data-movement schedule** — noting that it may return a measured zero.

This is that experiment, and it returns a measured zero. The schedule is a
function of the shot, not of the round index, so the exchange cannot be resolved
when the circuit is built and cannot be removed from the emitted stream. What the
experiment adds beyond the negative is *which* invariant the schedule does have
(the number of exchange rounds is fixed at 72 for every shot; their positions
differ in 10 distinct ways across the 64 shots), which is a much stronger
statement than "it did not work".

## 1. The change under test

The exchange costs 892 Toffolis per firing because it is a **controlled
permutation** of 892 qubits: an uncomputation or a gate-level trick cannot make a
controlled swap cheaper than one non-Clifford per exchanged qubit. The only way to
remove it is to stop exchanging — *pointer substitution*: keep the data where it
is and reassign which physical register plays the `A` role in each round. A
register name is a build-time object, so the exchange would cost **nothing**, and
— unlike every lever in the campaign — the saving would live in the *emitted*
circuit rather than in an executed count. That is the only kind of saving a build
without measurement opcodes can compile, which is why it is worth asking
(Experiment D, [`DQI_OFFICIAL_COMPATIBILITY.md`](DQI_OFFICIAL_COMPATIBILITY.md)).

It works if and only if the schedule is a function of the round index. The
exchange is controlled on `c3 and flag`, both of which are *data*, so the
question is a measurement: what does the set of exchanging rounds look like
across shots?

## 2. The prediction, declared before the arms existed

| # | claim | declared before |
|---|---|---|
| **C1** | reassigning the roles at build time removes the region, so the saving lands exactly on the control's exchange row — `64,224.0` executed, `198,024` static — and the arm is correct **if and only if** the exchanging-round set is shot-independent | any variant |
| **C2** | the exchanging-round set is a function of the shot's syndrome, so there is more than one pattern and C1 is unavailable; one pattern would make C1 exact | any variant |
| **C3** | the exchange's `flag` operand is redundant in a `c3` round (the campaign's S3b result said as much for the shift), so exchanging on `c3` alone saves `72 × 1 = 73.0` and stays correct | any variant |

C1's figure is not a projection: the arm deletes the region, so the region's whole
executed cost is what it would save. C3's 73 is the `init_and`/`del_and` pair the
incumbent wraps around its `cswap`, once per firing.

## 3. What was measured

All arms: `Q` = 1,913 unless noted, 222 EEA rounds, syndromes restored, and every
region row partitioning its own total.

| arm | executed T/shot | exchange row | correct | what it is |
|---|---|---|---|---|
| control (five gates) | **686,014.84** | 64,224.0 | yes | reproduces the incumbent to the Toffoli |
| `swap_cx` (harness check) | **686,014.84** | 64,224.0 | yes | the incumbent's exchange re-emitted from this script: ±0.0 |
| **`rebased` (C1)** | 505,092.77 | **0.0** | **no** | roles assigned from the round index; region deleted |
| `swap_on_c3` (C3) | 852,267.41 | 190,888.0 | **no** | exchange controlled on `c3` alone |
| `probe` (C2) | 686,014.84 | 64,224.0 | yes | `Q` = 2,135: `c3` read once per round into a per-round qubit |

Three things in that table are load-bearing:

1. **The rewrite is exact.** `swap_cx` assembles the incumbent's exchange from
   this script's own code and lands on 686,014.84 with no change, so the ±0.0 row
   licenses the comparison: what follows differs by the *change*, not by the
   rewrite.
2. **C1 removes the region, statically.** A future reader's first question is
   whether the rebased circuit is really cheaper. It is: the region's static
   count falls by its whole 198,024, which is the saving an
   official-compatible schedule would have delivered. Its *executed* total
   (505,092.77) is **not** a saving and is not quoted as one — the arm is wrong,
   a wrong state runs different control flow, and §6 rule 4 of the campaign
   forbids reading a saving off a wrong arm's total. That is precisely the trap
   the campaign's own falsifiers documented.
3. **It is wrong anyway.** The rebased arm restores the syndromes but leaves the
   rest of the register file dirty and the tracked phase unmatched. Build-time
   resolution is not available.

## 4. Why: the schedule's invariant is a count, not an order

The probe reads `c3` -- the exchange's own control -- once per round, into one
qubit per round, and the same 64 shots are read back:

| quantity | measured |
|---|---|
| rounds | 222 |
| shots | 64 |
| exchanging rounds per shot | **72, on every shot** |
| distinct patterns across the 64 shots | **10** |
| identical to the period-4 guess (`i % 4 == 3`) | **no** |

So the schedule is *cardinality-stable and order-unstable*: every shot exchanges
exactly 72 times, and no two shots agree on where. That is why the incumbent's
exchange costs exactly `64,224 = 72 × 892` on every shot while no build-time
period reproduces it, and why the naive period-4 rebasing is not merely
approximate — it is a different circuit from the first exchange.

The number 72 is also the answer to "how much is there to win": the exchange is
72 controlled 892-qubit permutations per shot, and a schedule cannot have fewer
of them without changing which rounds the EEA completes.

## 5. A correction to the campaign: two of its "0.0" arms are copies

The campaign reports two arms that measured exactly 0.0 — `composed` (the shift
fused with the exchange) and `drop_shift_c3` (the shift dropped in every `c3`
round) — and explains both as "the extra condition is unsatisfiable". This
experiment found the actual mechanism while reusing the campaign's own decoder,
and it is not that: the code implementing both lives in a method named
`_shift_a_and_b`, and nothing calls it. The reference and `MRGatedDecoder` call
`_shift_b_array_right`, so the campaign's `_shift_a_and_b` (and the `shift_under`
and `drop_shift_c3` branches inside it) never ran.

Measured, at a small instance (m=15, n=6, GF(2^4)), emitted instruction streams:

| arm | instructions | static Toffolis |
|---|---|---|
| `none` (control) | 149,337 | 16,610 |
| `composed` | 149,337 | 16,610 |
| `drop_shift_c3` | 149,337 | 16,610 |

The two are the control circuit, instruction for instruction. Their 0.0 is
therefore **vacuous**, not evidence that the composed condition is unsatisfiable.

What survives: the *argument* in the campaign's §2 table that the exchange's
`c3 ∧ flag` and the shift's `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))` cannot overlap is
an argument about the predicates, and it is untouched — `ctrl` is one-hot and the
flag is one value, so a round cannot be both. What does not survive is the claim
that the two arms *measured* it. The campaign's §3 table should be read with
`composed` and `drop_shift_c3` struck from the evidence column, and this
experiment's audit row is the replacement for the measurement they were meant to
provide.

## 6. The shift half

The same argument closes the shift's remaining 96,584.6 without a new arm:
its movement is `cswap(do_shift, reg_b[i-1], reg_b[i])` under a mask that is the
XOR of three threshold intervals over quantum registers (`n_B`, `n_b`, `n_A`), so
the mask's firing set is a function of the shot by construction — the campaign's
S2 arm already measured it as half-dense (65,895 saved, 65,325 kept). A schedule
that knew the window at build time would have to know the shot's syndrome.

So both halves are closed with one measurement rather than two, and the closing
statement is stronger than the campaign's: the decoder's data movement is a
permutation *parameterised by the data*, and no build-time assignment of register
names can express it.

## 7. What this does not settle

**C3 falsified, and the reason is not established.** The arm that exchanges on
`c3` alone is incorrect — that is measured, and it is the useful outcome, because
it says the exchange's control has an operand that matters. But the probe also
measures `c3` firing 72 times per shot while the incumbent's exchange costs
exactly 72 firings, which on its face is consistent with `flag` being set in every
`c3` round. The arm's own exchange count (190,888, i.e. 214 firings) is measured
on a trajectory that has already diverged, so it is not evidence either way
(campaign §6 rule 4). **The mechanism is an open question**, recorded as such:
either the `flag` operand is clear in some `c3` rounds, or the arm differs from
the incumbent in a way this experiment has not isolated. An exploratory probe
outside this record (m=15, n=6, GF(2^4), the same 8 sampled shots, logging `c3`
and the post-update `flag` at the exchange) points at the first: `c3 ∧ flag`
fires 1-3 times where `c3` fires 7-8 on every shot. That is a hint, not a
measurement this record pins, and nothing in the conclusion depends on which
explanation holds.

**No target arithmetic moves.** The arm stays at 686,014.8, 1.541× the 445,229
target. This experiment removes an option rather than a Toffoli.

**It does not produce a receipt either.** Deleting the exchange would have been
an official-compatible saving; it is not available, and the lever-based savings
that *are* measured still need `mr` (Experiment D).

## 8. Pins

`tests/test_dqi_movement_schedule.py` pins: the control's exact 686,014.84 at
`Q` = 1,913 and its 64,224 exchange row, the pre-declared C1 saving, the rebased
arm's zeroed region and its 198,024 static fall together with its incorrectness,
the probe's 10 distinct patterns at an invariant 72 firings, the falsified C3
prediction, and the audit showing the campaign's two zero-saving arms are copies
of its control.

A future edit that quotes the rebased arm's total as a saving, promotes a
build-time schedule on fewer than 64 distinct patterns, or restores the two
vacuous arms as evidence fails there.
