# Integer relation search: branch-and-bound vs LLL/PSLQ

Two functions in the native kernel answer the same question — *is there a non-zero
integer vector `m` with `m·x ≈ 0` for these real constants?* — at different cost and
with different honesty about what they return.

| | `find_integer_relation` | `find_integer_relation_lll` |
|---|---|---|
| module | `rust/apedp-kernel/src/rime/number_theory.rs` | `rust/apedp-kernel/src/rime/pslq.rs` |
| method (`INTEGER_RELATION_METHOD*`) | `brute_force_branch_and_bound` | `lll_integer_relation_pslq` |
| search space | coefficients in `[-max_coeff, max_coeff]^n` | the lattice `[ I \| round(x · 10^digits) ]` |
| cost | `O((2·max_coeff+1)^n)` | polynomial in `n` for a fixed precision |
| input cap | refuses `n > 8` | any `n`, coefficients bounded only by reach |
| on success | minimal-l1 vector inside the box, residual attached | reduced vector, verified by exact substitution |
| certificate | none | exact rational residual, size margin, working precision |
| on failure | `None` — nothing in the box | `None` — nothing found at any precision on the ladder |

`None` never means "no relation exists". It means the search, at the precision it
reached and inside the space it was given, produced nothing it could verify.

## Acceptance rules in the reduction

A reduced row is not evidence by itself. Three filters sit between the lattice and a
returned relation, and each exists because of a specific way the search can lie:

1. **Exact substitution.** `m·x` is evaluated in exact rational arithmetic on the
   exact values of the input `f64`s. A candidate that only holds as a float dot
   product fails here.
2. **Size margin.** The embedding always contains some vector with a zero constant
   coordinate whose coefficients are about `10^(digits/n)` — any `n` integers of
   `digits` digits have a small relation. A candidate must be 10× shorter than that
   bound or it is arithmetic, not structure. (`SPURIOUS_SIZE_GUARD`.)
3. **Resolution margin.** Rounding the constant column creates residuals of order
   `0.5·sum|m_i|·10^-digits`. A residual at or below that floor was manufactured by
   the embedding. The residual must be 4× above it — unless it is exactly zero, which
   no rounding produced. (`RESOLUTION_GUARD`.)

Precision escalates in ten-digit steps from the input's own precision (17 digits for
`f64`). Escalation buys coefficient range, not information: an `f64` carries about
seventeen digits, and no ladder reaches past that.

## How the reduction is driven

The lattice is reduced by a two-stage loop rather than one exact pass per step:

1. **`f64` LLL pass** (`float_sweep`). The Gram-Schmidt decomposition is built once,
   then updated in place by the standard size-reduction identities
   (`mu[k][l] -= q·mu[j][l]`, `mu[k][j] -= q`; `norm[k]` is untouched because `b*_k`
   does not move) and rebuilt only after a swap. The vector updates stay in `BigInt`,
   so the reduction itself is still exact — only the *steering* numbers are floats.
2. **Exact certification** (`exact_violation`, `exact_one_step`). When the pass ends,
   `BigRational` Gram-Schmidt asks whether the resulting basis actually satisfies the
   LLL conditions. If it does, the basis is used. If not, one exact step repairs it
   and the pass repeats, up to a bounded number of repairs; a basis that is never
   certified is reported as `completed = false` and the caller falls back to the
   exact-only `lll_reduce`.

The worst failure mode of the float steering is therefore a missed relation, never a
wrong one: every returned vector is still checked by exact substitution.

### Measured cost of the two drivers

`cargo test --release -p apedp-kernel --lib rime::pslq -- --nocapture`, on one hard
basis — eight rows, twenty digits, no short relation, so the reduction cannot stop
early:

| driver | time | iterations | certificate |
|---|---|---|---|
| exact Gram-Schmidt rebuilt per step (`lll_reduce`) | 454.5 ms | 270 | exact LLL conditions |
| `f64` pass + one exact certification (`lll_reduce_fast`) | 265.5 ms | 489 | exact LLL conditions |

**1.7×**, with both resulting bases asserted to satisfy the same exact conditions.
The count of iterations is higher and the time is lower, which is the whole point:
iterations are cheap when the steering decomposition is floats.

A first version of this pass rebuilt the `f64` decomposition from `BigInt` **twice per
iteration** and restarted the sweep after every repair. On the same kind of basis
(twelve rows, twenty-seven digits) it measured **0.5×** — 79.9 s against the exact
path's 42.1 s — because the per-step `BigInt` size reduction dominates the loop and
the route through `f64` never got to save anything. That negative number is kept here
because "use floats for the Gram-Schmidt" is not the speedup; keeping the
decomposition in place is. It also explains why the timing test prints instead of
asserting a wall-clock threshold: what it asserts is that both drivers return a
certified basis.

## Measured comparison

`scripts/compare_integer_relation_methods.py` → `benchmarks/integer_relation_methods.json`.
Eight hand-built cases, each with a declared criterion and a declared scope; the
record states that this is a designed set, not a distribution over problem space.
Times are one process on one machine against an optimised build.

| case | n | branch-and-bound | LLL/PSLQ | ms (bnb / lll, one run; the record is authoritative) |
|---|---|---|---|---|
| stoichiometry `2 H₂ + O₂ − 2 H₂O` | 3 | `[2, 1, -2]` correct | `[2, 1, -2]` correct | 0.065 / 0.249 |
| harmonic ratio `3ω₁ − 2ω₂` | 2 | `[3, -2]` correct | `[3, -2]` correct | 0.008 / 0.045 |
| planted `√2 + √3` | 3 | `[1, 1, -1]` correct | `[1, 1, -1]` correct | 0.051 / 0.275 |
| coefficients outside the box (`137 : −89 : −1`, `max_coeff = 20`) | 3 | refused (out of scope) | `[137, -89, -1]` correct | 1.423 / 0.252 |
| ten exact inputs, first nine primes | 10 | refused (out of scope, `n > 8`) | exact relation, correct | 0.003 / 0.875 |
| twelve inputs, weighted exact sum | 12 | refused (out of scope, `n > 8`) | exact relation, correct | 0.003 / 2.241 |
| independent irrationals, `eps = 1e-300`, ladder stopped at 17 digits | 3 | refused, correct | refused, correct | 31.414 / 0.426 |
| same three values, ladder allowed to 40 digits | 3 | refused (out of scope) | `[82119140, 22504678, -69368753]`, exact, correct | 28.525 / 0.901 |

Summary as recorded: branch-and-bound **4 correct, 4 refused out of its declared
scope, 0 wrong**; LLL/PSLQ **8 correct, 0 wrong**; they agree on **4 of 8**. The
record's own `median_elapsed_ms` and `slowest_elapsed_ms` carry the timings, and
single-machine numbers move between runs: the median sits near 0.07 ms against
0.43 ms and the slowest near 31 ms against 2.2 ms. The box search is a few
microseconds to a few milliseconds of work wherever it can be asked and refuses half
these cases; the reduction pays a fixed few-hundred-microsecond cost per question.

These are the timings *after* the `f64`-driven pass described above. With the exact
pass per step in the same record's previous run the reduction took 7.4 / 0.31 / 6.8 /
2.7 / 69 / 207 / 18 / 51 ms on the same eight cases — 10–90× more, which is the
speedup this pass bought on the real comparison rather than on a synthetic basis.

### Randomized families

Eight designed cases say the methods work on cases someone chose. The record also
carries three seeded families, 40 cases each, `random.Random(20261102)`:

| family | construction | branch-and-bound | LLL/PSLQ |
|---|---|---|---|
| `random_planted_in_box` | dyadic relation, `|m_i| ≤ 4`, `max_coeff = 6` | 40/40 correct, pass 1.0 | 40/40 correct, pass 1.0 |
| `random_planted_out_of_box` | same construction, `|m_i| ∈ [20, 400]`, `max_coeff = 12` | 0/40 correct, 40 refused out of scope, pass 0.0 | 40/40 correct, pass 1.0 |
| `random_independent` | continuous draws, `eps = 1e-300`, ladder stopped at 17 digits | 40/40 refused, pass 1.0 | 40/40 refused, pass 1.0 |

Totals over all 128 cases: branch-and-bound **84 correct, 44 refused out of its
declared scope, 0 wrong**; LLL/PSLQ **128 correct, 0 wrong**. The `0.0` pass rate for
the box search on the out-of-box family is scope, not error: those coefficients are
20–400 when the box is 12, and the record counts them as `refused_out_of_scope`, which
is the outcome its documentation promises.

The planted relations are recovered **up to sign**: of 80 planted cases, 42 come back
as exactly the planted vector and the other 38 as its negation, because
canonicalisation makes the first non-zero coefficient positive. All 38 were checked to
be exact negations, so the searchers recover the planted vector in 80/80 cases and the
record's `planted_vector_recovered_*` counters (42) count only identical vectors — the
narrower, verifiable claim.

These rates are a distribution over *these three families only*. Both planted families
always carry a relation and the independent family never does, so the rates say the
contracts are honoured on this generator; they do not estimate how often relations
exist in real discovery output.

## What the numbers say

- **Where the box fits, it wins on time and loses nothing.** On the three
  relations inside `n ≤ 8` and a small coefficient box, both methods return the same
  vector, and the box search is 4–6× faster at this size: a fraction of a millisecond
  against a fraction of a millisecond. It is exact inside the box and needs no
  precision discipline.
- **The reduction is the only one that answers above the box.** `137 : −89 : −1`
  cannot be expressed with `max_coeff = 20`, and `n = 10` and `n = 12` are refused
  outright by the box search. Those are four of the eight cases, and they are the
  entire reason the second method exists. It is now also *cheaper* than the box on the
  two refusal cases (31.4 / 28.5 ms against 0.43 / 0.90 ms): exhausting a box to say
  nothing is more work than reducing a lattice to say nothing.
- **The reduction pays for reach in milliseconds, not seconds.** It is still the
  slower method on the three trivial cases, where the box answers in microseconds and
  the reduction still owes a precision ladder and one exact certification. What the
  `f64`-driven pass changed is the size of that penalty: median 0.43 ms against
  0.07 ms, and the two refusal cases now cost the reduction *less* than the box.
- **"No relation" is not a property of floating-point numbers.** The last two rows
  are the same three constants. Stopped at the input's own precision, the reduction
  correctly returns nothing. Allowed to escalate, it returns an exact relation with
  coefficients around `10^8` and a residual of exactly zero — a genuine integer
  relation among three 52-bit rationals, and not a relation among √2, √3, √5. Both
  facts are true; the record keeps them apart instead of presenting only the second.
  The same limit decides the coefficients-outside-the-box case: a three-term relation
  with coefficients around `10^2` that held only to `f64` rounding would need about
  `3·log10(137) + 16 ≈ 21` working digits and could not be certified at all, so that
  case uses exactly representable values and says why.
- **`max_coeff` is not passed to the reduction.** A positive value there filters
  candidates beyond it, which recreates the box the reduction exists to escape. The
  comparison calls it with `0` (unconstrained) and records that choice per row.

## Harvesting relations from recorded discovery runs

`scripts/harvest_integer_relations_from_discoveries.py` turns committed benchmark
artifacts into 235 questions — one per emitted formula, per fitted coefficient vector,
per set of replicates — and asks both searchers each of them. `--profile` reports where
the minutes go instead of only the total.

One profiled run, 429.5 s wall, gathered while an unrelated test suite was also using
the machine, so absolute times are inflated and the shares are the part to read:

| inputs `n` | groups | total ms | share of the run |
|---|---|---|---|
| 2 | 134 | 38.7 | <0.1% |
| 3 | 18 | 16.1 | <0.1% |
| 4 | 16 | 266.3 | 0.1% |
| 5 | 11 | 4,394.7 | 1.0% |
| 6 | 25 | 254,146.7 | 59.2% |
| 10 | 16 | 20,953.6 | 4.9% |
| 14 | 1 | 4,700.6 | 1.1% |
| 15 | 14 | 145,021.0 | 33.8% |

By source: `benchmarks/feynman_noise_rediscovery_state.json` 425.4 s (99.0%),
`benchmarks/lorenz_public_benchmark_results.json` 4.1 s. The 134 two-input groups cost
0.04 s together, so counting questions is a poor guide to cost — 93% of the run is
`n ≥ 6`.

The six slowest groups are 15-input ladders for `engine='rime_ad'` (worlds
`quadratic_constant`, `sinusoid`, `sqrt_law`, `saturated_growth`, `hill2_cooperative`,
`rational_transition`) at 12–22 s each, all of it in the reduction: their box search
returns in 0.01 ms because it refuses `n > 8`. The only branch-and-bound-dominated
questions in the whole run are two 6-input groups, `trials[66]` and `trials[67]`, at
11.2 s each against 0.09 s for the reduction — a 25^6 box is roughly 244 million
nodes, and that is the exponential the box search documents.

What that says for a future change: the harvest's cost is set by the reduction on
`n ≥ 10` groups, not by its 235 questions. The box search is a rounding error here
except where its box gets large, and there its cost is the one it warns about.

## Reproducing

```bash
make build-native                     # or maturin develop --release
python scripts/compare_integer_relation_methods.py
python scripts/harvest_integer_relations_from_discoveries.py --profile
python -m pytest tests/test_dyn_integer_relation_methods.py -q
cargo test -p apedp-kernel --lib rime::pslq
cargo test --release -p apedp-kernel --lib rime::pslq -- --nocapture   # the timings above
```

The record is pinned by `tests/test_dyn_integer_relation_methods.py`: the tallies must
be recomputable from the rows, no method may be recorded `wrong`, and a `specific`
case answered must carry the relation its construction fixed.

Status: the branch-and-bound is `production`; the LLL/PSLQ path is `experimental` —
it is exact about what it returns, but its comparison set is eight designed cases,
not a survey of which relations appear in real discovery work.
