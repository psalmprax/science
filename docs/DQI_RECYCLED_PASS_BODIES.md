# DQI recycled pass bodies: what a 1,885-qubit build needs

**Status:** design sketch, nothing built · **Why:**
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §7.7 — the 23-qubit
window saving cannot be re-measured at 1,853 until a recycled build exists · **Basis:**
`RECYCLING_MEASUREMENT`, `RECYCLING_APPLIED_IN_CIRCUIT = False`

## The invariant, as measured

Only the a-pair is recyclable: `n_A + n_a = n + 1` is constant (the documentation's `n` was wrong),
while `n_B + n_b` varies over 3, 4, 5 and cannot be derived from `n_B`. So a build removes exactly
one register — `n_a`, 8 qubits at `n = 70` — and nothing else: **Q = 1,885**, not 1,789.

## The read sites, measured

An earlier version of this page listed five bodies. Read off the vendored signatures, that is
wrong on two of them: `_shift_b_array_right(..., n_B, n_b, flag, ctrl, ...)` and
`_update_metadata(..., n_B, n_b, n_q, flag, ctrl)` take no `n_a` argument at all — there is
nothing in either to substitute. The register is touched in **three** bodies plus the driver:

| site | access |
|---|---|
| `_update_flag` | reads `n_a == n_b` twice (sub-step 2 and sub-step 3) |
| `_multiply_and_add_or_sub` | a unary cursor over `n_a`, plus `n_q += n_a` and `n_q -= n_a` |
| `_swap_a_and_b` | cswaps `n_a` with `n_b` — reads *and writes* it |
| `build_syndrome_decoding_mbuc` | allocates `n_a`, sets its leading bit, threads it through both EEA passes, frees it |

The reads are comparisons and cursor addresses rather than data dependencies, which is what makes
substitution conceivable: it changes *when* the value is produced, not what the pass computes. The
writer is the harder site, because a swap is where a live `n_a` can be *put back* non-derivably.

## The substitution, as built

The leading-zero count sketched earlier was not needed and is not what shipped. Two mechanisms do
the work instead:

1. **Mirrored cursor.** `_multiply_and_add_or_sub` addresses `n_a` through `unary_iteration(n_a)`;
   that becomes `unary_iteration(n_A)` behind a mirror (`move_to(v)` → `move_to(n + 1 - v)`), so the
   body keeps its own thresholds and equality tests while never naming the register.
2. **Arithmetic at the read.** `n_q += n_a` becomes `n_q += n + 1; n_q -= n_A` (and the `isub` form
   dually), which is the invariant applied as two constant/register adds.

`_swap_a_and_b` writes instead of reading: the reference cswaps `n_a` with `n_b`, and with no
register the new `n_b` must be `old n_a = n + 1 - new n_B`. The build clears `n_b` under the swap
control and reloads it with `iadd(n + 1)` then `isub(n_B)`.

The driver edit is what actually removes qubits: `n_a` is no longer allocated, initialised or freed,
and `n_A` is passed in its slot as a sentinel the substituted bodies assert on.

## Order of work

1. Implement behind `RECYCLING_APPLIED_IN_CIRCUIT`, default `False`.
2. Build: `num_qubits` must read **1,885**. Anything else means the register survived.
3. `verify()` at 64 shots against the reference, shot for shot — value and phase.
4. Re-measure the workspace window on *this* build, same protocol as §1.1's 42 → 19 cut.
5. Only then replace §7.5's 55 with the contemporaneous number.

## Falsifiers

| # | check | fails if |
|---|---|---|
| 1 | peak after build | ≠ 1,885 — register not removed, or scratch leaked |
| 2 | 64-shot equivalence | any value or phase mismatch |
| 3 | Toffoli delta | more than the counter-decode's measured cost — the substitution is not free and the trade must be re-priced |
| 4 | the b-pair | `n_b` had to be derived — the measurement says it varies, so this build cannot do it |

## What this unlocks, and what it does not

It makes the window figure contemporaneous, so §7.5's "55 < 64" becomes a same-build sum. It does
not reach 1,789: this lever is worth 8 qubits here, and 32 at its absolute limit.

## Status of the substitution: peak reached, equivalence failed

All three readers are substituted (`RecycledDegreeDecoder`) and the driver no longer allocates
`n_a`, so this is the build the accounting was waiting for. Measured at the scored 222-iteration
window (`dqi_recycled_degrees_window.json`):

| build | peak | Toffolis | Δ vs reference |
|---|---|---|---|
| reference | 1,893 | 5,725,461 | — |
| `_update_flag` only | 1,893 | 6,359,493 | +634,032 |
| all three sites + driver | **1,885** | 6,374,245 | +648,784 |

Falsifier #1 passes. The register is gone and the build peaks at 1,885: the accounting's
1,893 → 1,885 is reproduced by a circuit, for **+648,784 Toffolis** — 11.3% of the reference
total. At the short 4-iteration window the same build reads 1,880 against the reference's 1,888,
which is why a window-4 number is not the number the score sees, and why the earlier record
(`dqi_recycled_flag_body_build.json`, +11,424) understated the cost by a factor of 55.

**Falsifier #2 fails.** `RecycledUpdateFlagDecoder._update_flag` reads no `n_a`, and its predicate
identity holds over all 5,184 `(deg(A), n_b)` pairs (`derived_flag_predicate()`) — which is exactly
what makes the next table informative rather than ambiguous. `scripts/verify_recycled_equivalence.py`
drives the bundle's own `verify()` over the scored instance at 64 shots, every arm on the *same*
shots:

| arm | value errors | phase errors |
|---|---|---|
| reference | 0/64 | 0/64 |
| `_update_flag` only | **52/64** | 0/64 |
| all three sites + driver | **64/64** | 0/64 |

The first body substituted is already wrong, so the fault is in the derived predicate as built and
not only in the swap's write. An arithmetic identity that holds does not make the emitted cascade
compute it. So **1,885 is a peak, not a decoder**: `RECYCLING_APPLIED_IN_CIRCUIT` stays `False`,
the 1,885-scored projection stays analytical, and `dqi_recycled_equivalence.json` is pinned by a
test so the failure cannot be quietly dropped.

Contract `derive_n_a(deg(A)) = n + 1 - deg(A)` was verified exhaustively for `n = 4, 12, 70` in
both directions; the accounting still reports one 8-qubit register and the b-pair untouched.

## Root cause: the invariant dies at the first swap

The substituted comparison is not the fault. `scripts/diagnose_derived_flag_predicate.py` builds
the predicate alone — `A`, `B`, a register loaded with `n + 1 - A`, the reference comparison and
the derived loop — and simulates **every** `(A, B)` basis pair: all 5,184 agree, under both
`control=True` and a qubit control. So the cascade computes what it is asked to compute.

What it is asked to compute is the problem. Sweeping the EEA window
(`dqi_recycled_equivalence_bisect.json`, same 64 shots in every arm and window):

| `n_iters` | reference | `_update_flag` only | all sites + driver |
|---|---|---|---|
| 2 | 0v / 30p | 0v / 30p | 0v / 30p |
| 8 | 0v / 28p | 0v / 28p | **64v** / 0p |
| 32 | 0v / 27p | 0v / 27p | **64v** / 0p |
| 128 | 0v / 34p | 0v / 34p | **64v** / 0p |
| 222 | 0v / 0p | **52v** / 0p | **64v** / 0p |

The one-body substitution matches the reference at every window short of the full run and fails
only at 222; the all-sites build fails from the first window that fires a swap. That is the
signature of a premise that holds early and stops holding, not of a mis-emitted comparison.

**The invariant holds only while the a-pair has never been swapped.** `n_A + n_a = n + 1` is
established at initialisation, and the only thing that writes `n_a` afterwards is
`_swap_a_and_b`, which exchanges the a-pair with the b-pair. That pair's sum varies — measured at
`n_B + n_b ∈ {3, 4, 5}` on the `n = 4` trace — so after a swap `n_a = old n_b`, which is not
`n + 1 - new n_A`. Every derived read after the first swap that lands a non-complementary value is
then wrong. The one-body build still carries a real `n_a`, which is why it survives 128 iterations
and fails in the last phase where those swaps have accumulated; the all-sites build has no `n_a`
at all, so its swap write assumes the same false invariant and fails immediately.

This is falsifier #4 applied to the other register: `n_a` is derivable only pre-swap, and tracking
the pair sum to recover it needs a register — the one being removed. So **1,885 is not reachable by
this substitution**, and the 11.3% Toffoli cost below buys a peak that does not belong to a working
decoder.

## What that cost does to the projected score

The trade is not close. On the built circuits the recycled decoder costs **1.1133×** the
reference's Toffolis (6,374,245 vs 5,725,461, or 2,922 extra per EEA iteration against 25,790),
while the 8 qubits buy **0.9916×** in `Q²` (1,885² vs 1,893²). Applied to the projection's
T_mean target of 445,229 — the lever that supplies the rest of the gain — the recycling moves

| quantity | before | after recycling |
|---|---|---|
| `T_mean` | 445,229 | 495,680 |
| projected gain | 12.968× | **11.648×** |
| score factor vs the 12.968× projection | 1.000 | **0.898** |

So the register removal as built is **score-negative by about 10%**: the qubit saving is a 0.8%
improvement and the gate cost is a 10.2% regression. The ratio is first-order rather than exact,
because T_mean is a per-shot mean with early halting while the delta is a static count — but both
scale with iterations executed, so the ratio survives that correction. This is the second reason
the 1,885 build is not the campaign's answer even if the equivalence bug is found.
