# DQI single-array decoder: the correctness obligation, before any builder work

**Status:** design contract; the packing route is **measured and closed** (see §Measured closure) · **Basis:** `model_single_polynomial_array()` and
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) §7.6 ·
**Artifact:** `benchmarks/quantum_compilation/dqi_design_projection.json` (`single_array_trade`)

## What is claimed, and what it rests on

Holding one polynomial array instead of the live pair would put the peak at **1,001** qubits (852
saved from the 1,853 register floor) and cost one extra division per EEA round, priced at **378** Toffolis per call, counted off the emitted gate stream (one
`init_gf2_div_with_scaffold` at GF(2^12) is 378 CCX, matching `divide_scaffold_pair`). Over
222 rounds that is **1.015×** the reference Toffolis, and `R · Q² · T` lands at **0.284×**
the reference score — favourable. The 46,656 figure is the decoder's whole division bill per
shot (123.4 calls), not a per-call price, and the earlier 2.81× / 0.786× figures are
withdrawn as a per-shot quantity multiplied by the round count.

All of that is arithmetic under one sentence, and the sentence is the entire risk:

> the dropped remainder can be re-derived, at each point the EEA reads it, as a function of the
> syndrome and the step counter alone — so the re-derivation can be inserted without changing the
> state the rest of the pass sees.

## The obligation

The decoder must compute the same map as the reference Proos-Zalka pass set — syndrome to error
vector, tracked phase unchanged — for every error weight the instance admits (`l = 35`), and fail
through the same path above it. Concretely, before a gate is built:

1. **Derivation lemma.** For every step `t`, the dropped remainder `B_t` is a closed function of
   `(syndrome, t)`: a specific reversible circuit mapping `(S, t)` to `B_t` and back, with no side
   effects on the other registers.
2. **Induction.** Given the lemma, `A_t` and `a_t` computed with `B` re-derived each round equal the
   reference's for all 222 rounds — same qubit values, not merely the same final error vector.
3. **Phase invariance.** The predicates read during re-derivation are basis-state reads (`cx` into
   the readout ancilla, then `mr`), so the tracked phase is unchanged. This is the argument
   `EEA_SCHEDULE_CAMPAIGN.md` §1 makes for the schedule gates, and it has to be made again because
   the reads now sit inside the EEA body rather than around it.
4. **Failure equivalence.** Weight above `l` must still fail the same way: a wrong or uncomputed `B`
   that happens to decode is a correctness failure, not a pass.

## Falsifiers

Each of these kills the claim; none is optional.

| # | check | the claim dies if |
|---|---|---|
| 1 | reference control passes `verify()` at 64 shots | it does not — fix the harness before measuring anything |
| 2 | single-array build vs reference, shot for shot | any value or phase divergence, at any weight up to `l` |
| 3 | executed Toffolis per round | they rise by **less** than 378 — a gate went missing rather than being paid for |
| 4 | peak qubits | not 1,001 — the array was not actually dropped (an alias survived) |
| 5 | score | it beats 0.284× — suspect uncharged work; recompute from the executed stream |
| 6 | weight-`l+1` inputs | they decode instead of failing — the re-derivation erased the failure boundary |

## Measured closure of the packing route (2026-11-03)

The 1,001 peak assumes one 73-slot array can hold what `reg_a` and `reg_b` hold between them. That
is a claim about the BM update's write pattern, and it is decidable classically before a gate is
built:

```
C' = C + (d/b) x^m B        reads C_old and B_old, writes C's region
B' = C_old                  reads C_old,             writes B's region
```

Whichever region is written first destroys data the other still has to read. Both orders were run
over weights 0–35 (6 trials each, `benchmarks/quantum_compilation/dqi_single_array_packing.json`):

| arm | decoded | failed | first failure |
|---|---|---|---|
| three regions (`C`, `B`, scratch — the reference's live set) | **216/216** | 0 | — |
| two regions, `C'` in place then `C_old` copied | 6/216 | 210 | weight 1 (planted 1, decoded 2) |
| two regions, roles exchanged each step | 6/216 | 210 | weight 1 (planted 1, decoded 1) |

The 6 successes in each two-region arm are the weight-0 trials: **neither packing decodes even one
error**. The weight-`l+1` boundary still holds for the three-region core (20/20 fail to decode), so
the harness can see a false pass. `PACKING_ROUTE_CLOSED`: the 1,001 peak is not reachable by
repacking the reference update, and the contract's surviving route is re-deriving `B_t` by replay —
a nested EEA, not one division per round. That route has not been priced or built, so the ledger
figures (1.015× Toffolis, 0.284×) remain arithmetic on an assumption, and no circuit exists.

## What is not claimed

No construction exists. `RECYCLING_APPLIED_IN_CIRCUIT` stays `False` and the compiled circuit stays
reference-equivalent until obligations 1–4 are discharged and falsifiers 2, 3, 4 and 6 pass. The
1,001 figure is arithmetic on an assumption, not a measured peak.
