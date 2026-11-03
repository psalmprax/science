# DQI recycled pass bodies: what a 1,885-qubit build needs

**Status:** design sketch, nothing built · **Why:**
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §7.7 — the 23-qubit
window saving cannot be re-measured at 1,853 until a recycled build exists · **Basis:**
`RECYCLING_MEASUREMENT`, `RECYCLING_APPLIED_IN_CIRCUIT = False`

## The invariant, as measured

Only the a-pair is recyclable: `n_A + n_a = n + 1` is constant (the documentation's `n` was wrong),
while `n_B + n_b` varies over 3, 4, 5 and cannot be derived from `n_B`. So a build removes exactly
one register — `n_a`, 8 qubits at `n = 70` — and nothing else: **Q = 1,885**, not 1,789.

## The five read sites

The degree registers are read as unary loop controls in five bodies: `_update_flag`,
`_multiply_and_add_or_sub`, `_shift_b_array_right`, `_update_metadata` and `_swap_a_and_b`. Each
read is a comparison against a counter rather than a data dependency, which is what makes the
substitution admissible at all: it changes *when* the value is produced, not what the pass computes.

## The substitution

At each site, `n_a` is replaced by `n + 1 - n_A`, computed as a leading-zero count on `A`. The pass
already compares `A` against the counter to decide termination, so the count is the same comparator
tree it already runs, kept in the existing counter decode rather than in a new register. No
persistent register is added; the scratch the count needs is the pooling the V6 scaffolding already
performs.

Three edits per site:

1. delete the `n_a` register allocation and its uncompute;
2. at each use, emit the leading-zero count on `A` into the freed counter wire;
3. uncompute that count with the site's own adjoint, so the pass is exactly restored.

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

## Status of the substitution

The first body is implemented. `RecycledUpdateFlagDecoder._update_flag` reads no `n_a`: both of its
`flip_if_equal(n_a, n_b, …)` comparisons are replaced by `n_A == n + 1 - n_b`, swept through the
`n_b` cursor the pass already runs, so no persistent register is added. The identity is checked
exhaustively over every `(deg(A), n_b)` pair the decoder can hold
(`derived_flag_predicate()`, 5,184 pairs at `n = 70`).

**It does not move the peak, and that is the finding** (the accounting is still
1,893 → 1,885; the build is not). The other four bodies still read `n_a`, so
the register is still allocated: `build_candidate_with_recycled_flag_body()` builds the full
pipeline and reports the same peak as the reference build. **1,885 remains an accounting figure**
until all five bodies and the driver change; a one-body substitution cannot produce a 1,885-qubit
circuit, and claiming otherwise is exactly what the accounting is there to prevent.

Contract `derive_n_a(deg(A)) = n + 1 - deg(A)` was already verified exhaustively for
`n = 4, 12, 70` in both directions; `model_recycled_n_a_substitution()` still reports one 8-qubit
register and the b-pair untouched.
