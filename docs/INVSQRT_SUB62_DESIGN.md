# Benchmark I: designing a circuit below 62 qubits

**Status:** design record with measured inputs · **Target:** below **62 qubits**, which this
repository has already reached: the 62-Q build staged in `submission_24_59.zip`
(`sha256 af469847…`) is `PASS` over the full domain at **97,921.58** (200,540,928 shots,
`benchmarks/quantum_compilation/invsqrt_62q_full_domain_receipt.json`) · **Evidence:**
`scripts/diagnose_invsqrt_qubit_peak.py`,
`docs/invsqrt_frontier_and_portfolio_synthesis.md` §5, and the adder probe below.

The score is `Q × toffoli_equivalent`, so a design is only better if `Q × te` is lower. At
Q = 62 the bar is `te < 1,579.38`; the allowed `te` at other widths is `97,921.58 / Q`:

| Q | 62 | 56 | 48 | 40 |
|---|---|---|---|---|
| allowed `te` | 1,579.38 | 1,748.6 | 2,040.0 | 2,448.0 |

Our submitted entry is 89 Q at 1,770.8 `te` (157,601.08), which is 191.4 `te` over the 62-qubit
bar; the staged 62-Q build sits on the bar at 1,579.38. Everything below is about converting
width into slack **below** 62 — the 18 carries and the 49 registers are what stands between
this arithmetic and Q = 60.

## 1. Where the 89 qubits are

The builder instrumented by qubit id, with `folded_gadd` wrapped to record the live count on
entry and with its carry chain allocated:

| region | live entering | live with carries | calls |
|---|---|---|---|
| 21-bit accumulate add (`outer`) | **71** | **89** | **28** |
| 12-bit add, `inner` first pass | 53 | 62 | 6 |
| 12-bit add, `inner` second pass | 81 | 89 | 6 |
| 10/11-bit adds | 81 | 87, 88 | 2 |

The 71 live at a 21-bit add decompose as `22` x + `5` flags + `6` c2 (curvature) +
`12` residual + `6` coarse + `21` low (the phase accumulator) = 72, of which one lane is
not yet allocated. `folded_gadd` then allocates `n − 2 − borrowed = 18` carries, which is
what takes the circuit to 89.

So the width has two independent parts: **71 qubits of named registers** and **18 qubits of
carry chain**, and the public circuit's `62 = 22 x + 40` has to absorb both in 40.

## 2. Lever A — segment the carry chain (priced, not built)

`folded_gadd` allocates one carry per bit above the first. Splitting a 21-bit add into
`m` chunks of equal width makes the carry chain `ceil(21/m) − 1` instead of 18, at the cost
of one carry-propagate step per chunk boundary:

| chunks | carries live | Q | extra Toffolis per call | Δ per shot (28 calls) |
|---|---|---|---|---|
| 1 (shipped) | 18 | 89 | 0 | 0 |
| 2 | 9 | **80** | ~1 | ~28 |
| 3 | 5 | **76** | ~2 | ~56 |
| 7 | 1 | **72** | ~5 | ~140 |

At 3 chunks: `Q = 76`, `te ≈ 1,770.8 + 56 = 1,826.8`, score **138,837** — **1.14× better than
our shipped entry**, still **1.38× behind** the public circuit. The carry chain cannot be
taken below 1 without abandoning `folded_gadd` altogether, so this lever bottoms out at
Q ≈ 72.

### 2.1 Built — and it does not compile

That table is a price, not a configuration. `scripts/build_invsqrt_chunked.py` builds the
variant through the champion's own path (same tables, shared PDDD plan and rotation
calibration) with `chunked_gadd` in place of `folded_gadd`:

| build | raw Q | compiled Q | static T | verifier |
|---|---|---|---|---|
| shipped control (`--chunk 0`) | 89 | 89 | 1,466 | `PASS`, 157,605.52, sha `ce04c9d5…` |
| segmented (`--chunk 7`) | 76 | **aborts** | — | `AssertionError: Scratch function is not zero.` (`portable_exact_domain.py:285`) |

The control reproduces the pinned circuit byte for byte, so the failure is the segmentation
and not the harness. The cause is structural: each segment's carry-out is handed to the next
segment as its carry-in, and `folded_gadd` *consumes but preserves* a carry-in, so the
hand-off qubit still holds the carry when the last segment finishes. Nothing in this build
can clear it — `del_and` clears an AND, and a carry-out is a majority of three qubits, not an
AND. The only clear that would work is a Z-basis measurement, which is the same surface the
DQI arm is blocked on (`docs/BENCHMARK2_STATUS_DECISION.md` D2: no `mr`), and `hmr` is not
phase-safe.

So the width lever is **blocked, not expensive**: the 18 carries are what it costs to keep
every carry reversible with the primitives that exist. Unless an upstream Z-basis read
appears, the champion's width cannot be segmented at all.

## 3. Lever B — conditional per-shot skipping (measured, already taken)

The verifier counts *executed* Toffolis, and a gate under `push_condition` runs only on the
shots where its measurement bit is 1. The champion already uses this in
`unnormalize` (`hmr` → `push_condition` → the normalization phase fixup):

| | static | executed mean | skipped | classical bits |
|---|---|---|---|---|
| ours | 1,466 | 1,455.5 | 10.5 (0.7%) | 60 |
| public 62-Q | 1,495 | 1,325.5 | **169.5 (11.3%)** | **30,001** |

The staged 62-Q build (`af469847…`) skips **255.3 of its 1,537 static Toffolis (16.6%)** — more
than the public circuit's 169.5 — which is why its executed stream (1,281.66) sits below the
89-Q champion's 1,455.50 despite the larger static count.

Ten and a half Toffolis is the whole of what the 89-Q circuit exposes: the only fair measurement in
it is the normalization flag, and the adders have none — every control in `folded_gadd` is a
data qubit that must be restored, so gating a carry or an addend on a measured value would
destroy the value the verifier requires back. Their 169.5 therefore comes from a branch this
arithmetic does not have, and it cannot be copied by wrapping gates in conditions.

## 4. Verdict for this architecture

Lever A is measured to be unavailable (§2.1), so the only lever left is Lever B, which is
already taken to the extent this arithmetic exposes it at 89 qubits. The 89 / 1,770.8 /
157,601 entry is therefore the endpoint of *that schedule*, not a waypoint — but it is not the
floor: the staged 62-Q build reaches 62 Q at 1,579.38 `te`, so the width has been taken by
something outside this note's instrumentation (its builder is not in this checkout, so the
result is recorded as measured, not derived). What this note measures is why **sub-62** is
still blocked.

The remaining factor is structural: their 40 working qubits against our 71 − 22 = 49
registers plus the carries. Closing it means not holding a 21-bit accumulator, a 12-bit
residual and a 6-bit curvature word at the same time — a different arithmetic, not a
schedule. Record-restoring those instead of holding them is priced in §4a and does not fit
the budget either.

The digit-serial schedule was the candidate for that and is measured out:
`scripts/prototype_bitserial_sqrt.py` at 22 input bits costs **34,997 qubits / 101,914
Toffolis**, because each digit's function is built as a full-prefix table whose transients
grow with the table.

## 4a. Record-restoring the registers instead (measured cost)

If a register is measured away and rebuilt when needed, its width leaves the peak and its
restore enters the Toffoli stream. Both halves are measurable without building the circuit —
the restore is a shared PDDD over the same 8-bit address:

| register | width | restore cost |
|---|---|---|
| `residual` (12 bits from the address) | 12 | **328 nodes = 328 Toffolis** |
| `c2` curvature (already restored this way) | 3 | **39 nodes = 39 Toffolis** |
| `coarse` | 7 | same order as the 28-node curve plan |

Deferring all three would take the working set to `71 − 12 − 6 − 7 − 5 = 41`, i.e. Q ≈ 60
with a carry, but at `+328 + 39 + ~30 ≈ +400` Toffolis per shot: `te ≈ 2,171`, score
**130,260** at Q = 60. That beats the submitted 157,601 by 1.21× and still loses to the
62-Q 97,921.58 by 1.33×.

## 4b. Digit-by-digit with a shared PDDD, measured at every width

`boolean_plan` is already a node-shared decision diagram, not a lookup table, so the open
question is not "PDDD instead of a table" but **one plan per digit against one multi-output
plan shared across all digits**. `scripts/prototype_digit_shared_pddd.py` builds both from
the same truth data:

| k | w bits | per-digit Q | per-digit T | shared Q | shared T | shared nodes |
|---|---|---|---|---|---|---|
| 3 | 6 | **15** | 22 | 21 | 22 | 11 |
| 4 | 8 | **29** | 60 | 41 | 56 | 28 |
| 5 | 10 | **58** | 152 | 86 | 140 | 70 |
| 6 | 12 | **134** | 392 | 205 | 372 | 186 |
| 7 | 14 | **364** | 1,088 | 548 | 1,052 | 526 |
| 8 | 16 | **1,050** | 3,152 | 1,568 | 3,086 | 1,543 |
| 9 | 18 | **3,431** | 9,974 | 4,963 | 9,870 | 4,935 |

Sharing is **1.4–1.5× wider at every width** measured, because the multi-output plan must
carry the whole `2k`-bit address for every digit while the per-digit plans only carry the
prefix each digit actually depends on. It buys 1% of the Toffolis and costs a third of the
width, which is the wrong trade when a qubit is worth ~26 te. So the per-digit shape is
already the better one, and at full width it measures **34,997 Q / 101,914 T**.

The earlier description of that prototype as building each digit from a "full-prefix table"
was wrong: there is no table in it, and swapping the plan for a shared one makes the width
worse, not better. What the champion has that the prototype does not is not sharing — it is
an **address of 8 bits plus a polynomial correction over 14**, which is why its functions
cost tens of Toffolis instead of thousands.

## 4c. Can one of the 8 correction rows go? Measured, no

`scripts/analyse_correction_rows.py` evaluates the emitted integer model for every supported
`x` against the verifier's own `target` and `tol`, with the full row set and with each single
row removed. The control is 0.9628 of tolerance at the shipped coefficients (the built
circuit measures 0.99962; the model under-reports by 3.7% relative, which matters below):

| dropped row | shift | worst error/tol | x over tolerance | effect on `R` |
|---|---|---|---|---|
| 13 | −2 | **48.08** | 2,996,903 | none (Rmax 260,996) |
| 12 | −1 | 24.40 | 2,863,145 | none |
| 11 | 0 | 12.55 | 2,597,858 | none |
| 10 | 1 | 6.72 | 2,070,978 | none |
| 9 | 2 | 3.71 | 1,096,820 | none |
| 8 | 3 | 2.12 | 219,783 | none |
| 7 | 4 | 1.47 | 7,731 | none |
| 6 | 5 | **1.14** | 110 | none |

**No row can be dropped at the same 24-bit precision with the shipped coefficients**, and
dropping one buys no width either: `Rmax` moves by less than 0.01%, so the residual stays 18
bits. Re-calibrating the (K2, K1, K0) tables against a 7-row model does rescue the least
valuable row — **row 6 lands at 0.9805** of tolerance — but that is inside the model's own
3.7% error bar against the built circuit's 0.99962, so it is a build question, not a pass.

Even granting it, the prize is small. The attribution of the 1,476 static Toffolis
(`scripts/attribute_invsqrt_toffolis.py`, which reconciles exactly) is:

| region | Toffolis | share |
|---|---|---|
| 21-bit adders (the 14 outer rows) | 560 | 37.9% |
| table load (`init_anf`) | 247 | 16.7% |
| unattributed (shift/phasing helpers) | 228 | 15.4% |
| 12-bit adders (the 8 correction rows) | 132 | 8.9% |
| normalize + unnormalize | 198 | 13.4% |
| PDDD restores + record phasing | 61 | 4.1% |
| other adders (11/10/7-bit) | 50 | 3.4% |

So the 8 correction rows are 132 Toffolis in total, i.e. **16.5 each** — dropping row 6 is
worth ~1.1% of the score (156,132 instead of 157,601), and the largest single block to
attack is the 21-bit outer adder at 560, not the correction.

## 5. What a winning design has to look like

To beat 97,921.58 the entry needs one of:

- **Q = 62 with `te ≤ 1,579.38`** — where the staged build already sits, so a Q = 62 entry only
  wins by shedding further `te`; or
- **Q ≤ 56 at `te ≤ 1,748.6`**, or **Q ≤ 48 with up to +460.7 `te` of restore cost**
  (`2,040.0 − 1,579.38`).

Both require the working set to be ~40 qubits. Neither measured route reaches it: segmenting
the carry chain is blocked by the missing Z-basis clear (§2.1), and record-restoring the
registers costs 400 Toffolis for 24 qubits (§4a), which is the wrong exchange rate at any
width above ~52. What is left is a decoder-class rewrite of the arithmetic — an accumulator
that is phased digit by digit rather than held, with each digit's function implemented as a
shared PDDD rather than the full-prefix table that makes the digit-serial prototype cost
34,997 qubits.

## 6. Reproducing the decomposition

```bash
.venv-kickmix/bin/python scripts/diagnose_invsqrt_qubit_peak.py     # allocation ladder, peak 90
# the adder probe is the wrapper shown in §1: instrument alloc/free by id, wrap
# scheduled_arithmetic90.folded_gadd, print (len(target), live on entry, live with carries)
```

The public circuit's numbers come from the same verifier at the same 25,600-shot basis:
`100,569.53`, `PASS`, `Q = 62`, 1,325.5 executed T, 41 rotations. The staged 62-Q build, same
verifier: **97,918.09** at that basis and **97,921.58** over the full 200,540,928-shot domain.
