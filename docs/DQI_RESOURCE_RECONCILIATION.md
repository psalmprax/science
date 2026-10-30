# Reconciling the DQI reference rows, and why the arm has no receipt

**Status:** reconciliation note · **Scope:** the three reference figures in circulation for
Benchmark II (DQI) at m=4095, n=70, GF(2^12) · **Evidence:**
`benchmarks/quantum_compilation/dqi_resource_reconciliation.json`, produced by
`scripts/reconcile_dqi_resource_baseline.py`; rules in
[`CIRCUIT_OPTIMISATION_RULES.md`](CIRCUIT_OPTIMISATION_RULES.md); target arithmetic in
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md).

The reviewer's objection is correct: three numbers were being quoted side by side without
their provenance attached. They are reconciled here, and the answer is reassuring for the
gate counts and blunt about the receipt.

## 1. The three rows, with their provenance

| Row | `Q` | `T_mean` | Circuit | Extension | Shot sample | Comparable to the official row? |
|---|---|---|---|---|---|---|
| **Locked official reference** | 1,893 | 5,725,108.382109375 | Proos–Zalka reference decoder | pinned `kickmix.so` (competition build) | the verifier's default **102,400** shots | this *is* the official row |
| **This repository's reference** | 1,893 | 5,725,071.609375 | the same decoder, gate for gate | pinned binary **and** local `MR` rebuild — identical per shot | this project's **64** shots, seed 20261028 | yes: same metric, 0.78 standard errors away |
| **The arm** | 1,913 | 1,221,466.8 | six-lever gated decoder, 54-AND product, 19-qubit workspace | local `MR` rebuild only | the same 64 shots | **no**: needs an opcode the competition build does not have |

The official row is not a different architecture, and the difference was never a counting
difference. Two checks settle it.

### 1.1 The official figure is a 102,400-shot mean (proved from the number itself)

A mean over `k` counts always has a fractional part that is a multiple of `1/k`, so the
figure itself says how many shots it averaged:

| hypothesis | product | integral? |
|---|---|---|
| 102,400 shots (the verifier's `DEFAULT_NUM_SHOTS`) | 586,251,098,328.0 | **yes** |
| 64 shots (this project's scripts) | 366,406,936.45 | no |

So 5,725,108.382109375 is the verifier's default-shot mean, and 5,725,071.609375 — whose
fractional part is 39/64 — is a 64-shot mean. Same decoder, same metric
(`num_toffoli_like`, which the verifier itself sums and divides by the shot count), same
instance, different sample. The per-shot spread is 375.8 Toffolis, so this 64-shot mean has
a standard error of 47.0 and the observed −36.77 is **0.78 standard errors**: well inside
sampling noise, not a discrepancy to resolve.

### 1.2 The local rebuild is count-neutral (measured, per shot)

The reference decoder was built and simulated on the **pinned competition binary** and on
the **local `MR` rebuild**, over the same 64 sampled syndromes (identical sample hash
`d3282209…`). The two per-shot Toffoli arrays are **identical element by element**, and
both means are 5,725,071.609375 at `Q` = 1,893. The `MR` patch adds opcodes and
statistics plumbing; for a circuit that emits no `MR`, it changes no count. The local
reference row is therefore the official metric, on another sample — not a different metric.

## 2. The arm is not a compilable official candidate

The arm is a complete circuit, and its numbers are produced by the same counting function
the verifier uses, but it **cannot be compiled for the competition binary**. Recorded
verbatim by the reconciliation run:

```
arm on the pinned binary: built = False, AttributeError: 'Builder' object has no attribute 'mr'
```

The arm's ledger clears (`del_and` targets and the 19-qubit product window) are
measurement-and-reset operations. In the competition build there is no `MR` opcode, no
`mr` method on the builder, no `MR` in the opcode table, and no parser entry — so the
circuit cannot even be constructed, let alone written to a `.kmb` and scored.
`apedp_dyn/quantum_compilation/contracts.py` makes that a rule rather than a limitation:
it hashes the verifier source tree and the `kickmix` extension, and only that pinned
extension may produce a receipt.

There is also no threshold to clear: `verify_submission_dqi.py` returns `status: PASS` on
correctness (`eps_plus < 1/16` over the shot sample) and reports `score = R(eps_plus) · Q² ·
T_mean`. It does not compare `T_mean` with 445,229 or any other number. So "official
verification" of the arm is not a button that was skipped — it does not exist for a circuit
the pinned binary cannot represent.

What *would* make a receipt possible: an upstream release whose instruction set includes
measurement-based reset (the proposal in `docs/MR_PATCH_REQUEST.md`), or a decoder variant
that is exact without measurement-based clears. Until then the arm's 1,221,466.8 is a
correct, measured, **local** figure, and quoting it as a submission result would be a
category error.

## 3. What the numbers may and may not be used for

| Claim | Status |
|---|---|
| "The reference decoder executes 5,725,071.6 ± ~50 Toffolis per shot on a 64-shot sample at the pinned instance." | measured, both binaries, pinned by script |
| "The official reference row is 5,725,108.38 over 102,400 shots." | recorded constant; shot count proved from integrality |
| "The official and local figures agree within one standard error of the sample." | measured |
| "The 54-AND product policy lowers the reference decoder 9.15% at unchanged correctness." | measured on the local rebuild |
| "The six-lever arm with that policy executes 1,221,466.8 per shot at `Q` = 1,913." | measured on the local rebuild |
| "The arm is an official Benchmark II result / has a receipt." | **not claimed, and not obtainable** |
| "445,229 is an acceptance threshold in the verifier." | **false** — see below |

## 4. What 445,229 actually is

It is a **target**, and a derived one. `apedp_dyn/quantum_compilation/dqi_gluzdov_v6.py`
records the target row as `Q = 1,789`, `T_mean = 445,229`, with

```
score_target = R(eps_plus) · 1,789² · 445,229 ≈ 1.6975e12
gain         = score_reference / score_target = 2.4439e13 / 1.6975e12 ≈ 14.397×
```

So the T figure exists to make the product with a *qubit* target reach the 14.396× portfolio
aim. It is not a measurement of any candidate, not a competing candidate's mean, and not an
acceptance threshold: the verifier has none. Every comparison against it in
[`BENCHMARK2_TARGET_REACHABILITY.md`](BENCHMARK2_TARGET_REACHABILITY.md) is therefore a
comparison against that derived aim at the same instance and the same metric, and is
labelled as such.

## 5. Pins

`tests/test_dyn_mr_gated_arms_kickmix.py` pins the reconciliation record: the official row's
102,400-shot integrality (and its non-integrality at 64), the identical per-shot arrays
across the two binaries, the identical means, and the pinned build's failure to construct
the arm with the exact `AttributeError`. A future edit that quietly re-samples the reference,
changes the counting, or claims a receipt for an `MR` circuit fails there.
