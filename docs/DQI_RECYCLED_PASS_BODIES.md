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
