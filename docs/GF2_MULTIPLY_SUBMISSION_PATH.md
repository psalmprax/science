# How a sub-51 GF(2^12) multiplier would actually be submitted and verified

Status: audit-only. Written after the stub behind `field_tensor_search.search_sat_rank` was
replaced with a real, exact search (`apedp_dyn/quantum/gf2_rank_search.py`).

## 1. Why an algebraic certificate is stronger than a verifier receipt

A bilinear decomposition of the multiplication tensor is a **polynomial identity**. If

    T[i, j, k] == sum_l U[l, i] * V[l, j] * W[l, k]   (mod 2)

holds for all `12 * 12 * 12 = 1,728` coefficients, then the multiplier is correct on all
`2**24` input pairs at once. No shots, no confidence interval, no salt. The official verifiers
sample because a *circuit* is a physical object with a compiled gate stream; a *decomposition*
is not — it is an equation.

That asymmetry decides the whole submission question: the hard part is not proving the
arithmetic, it is turning the arithmetic into something a verifier is allowed to score.

## 2. The four gates, and which ones exist

| gate | what it decides | state in this checkout |
|---|---|---|
| 1. algebraic | the 1,728 coefficients reconstruct; `W` re-solved from `U`, `V` alone | **PASS** — `verify_tensor_decomposition`, and `gf2_rank_search.solve_weights` for the independent `W` |
| 2. bit-order cross-check | the decomposition rebuilds `gf_mul` for sampled pairs, catching convention errors the tensor identity would still pass | **PASS (sampled)** — `scripts/verify_gf2_multiplier_candidate.py` |
| 3. reversible netlist | an actual gate list that can be simulated end to end | **MISSING** — `compile_reversible_multiplier` returns counts (Toffolis, CNOTs, ancillas, depth) and no gate stream, so nothing can be simulated as a circuit |
| 4. embedded benchmark receipt | an official `PASS` on a scored circuit containing the multiplier | **MISSING** — a component rank is not a submission |

Gate 3 is the real blocker, and it is not a formality: the Toffoli figure `rank + 12` for a
controlled build is a *pricing model*, not a compilation. Until a netlist emitter exists, any
"sub-51 circuit" claim is a claim about a formula.

## 3. The rank search, measured

`scripts/search_gf2_mul_rank.py` runs exact deflation and `k -> k-1` rewrites, where the
residual space against the remaining atoms has at most `k` dimensions and every one of its
`2**k - 1` nonzero elements must be rank-one — a complete test over *all* rank-one matrices,
not a candidate pool.

| seed | start rank | after search | deflations | rewrites | moves tried |
|---|---|---|---|---|---|
| Schoolbook-144 | 144 | 144 | 0 | 0 | budget-bound |
| Karatsuba-63 | 63 | **54** | 9 | 0 | 26,235 |
| Karatsuba-54-BaseComposed | 54 | 54 | 0 | 0 | 26,235 (complete) |

Two things are worth reading off that table:

* the search **rediscovers the base-composed 54 from Karatsuba's atoms alone**, by spans, with
  no knowledge of how the shipped 54 was constructed — evidence the machinery is real rather
  than a restatement of the answer;
* for the 54, all `C(54, 2) = 1,431` two-atom and `C(54, 3) = 24,804` three-atom rewrites were
  tried and none exists. That is a *negative result over a declared family*, not a minimality
  proof, and it is recorded as such.

The target is still open: nothing that reconstructs reaches rank 50, and the published
51-incumbent ships all-zero factors (363 of 1,728 coefficients disagree), so it cannot even be
deflated — it has no atoms to remove.

## 4. What sub-51 would be worth, arithmetically

The DQI projection spends 386,061.7 Toffolis on its GF(2^12) product calls, at a measured
63 Toffolis per call (`c_gf_iadd`). That is about `386,061.7 / 63 ≈ 6,128` calls per shot, so
each AND removed is worth `6,128` Toffolis:

* rank 50 instead of 63 removes `13 * 6,128 ≈ 79,700` Toffolis, about **17.9%** of the
  445,229 denominator;
* the shipped rank 54 removes 9, i.e. `≈55,150`, about 12.4%.

Both figures are arithmetic on the numbers in the DQI projection, not measurements of a built
circuit: they assume the product call stays the 63-AND form, ignore the CNOT and ancilla
budget, and say nothing about liveness.

## 5. The path to a real submission

1. **Emit a netlist.** A `reversible_netlist()` that lowers `U`, `V`, `W` to `cx`/`ccx` with
   explicit ancilla allocation is the single missing piece; without it gate 3 cannot be closed.
2. **Simulate it.** Value-level simulation on all `2**24` pairs is affordable once a netlist
   exists, and it is stronger than any shot-based receipt.
3. **Embed it.** Replace the product call inside a scored benchmark circuit (Benchmark II's
   decoder is the natural host) and re-run the official verifier; only that receipt moves a
   score.
4. **Never promote on rank.** Rank, Toffoli count and score are three different quantities;
   this repository's convention is that a screen is not a score, and a component rank is not
   even a screen.
