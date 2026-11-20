# Result: PREREG-FT-PREP-2026-11

**Run:** once, 2026-11-20 00:21–00:25 UTC, locally (2 cores, Python 3.13, Stim 1.16.0), at commit
`ddf34ef`, which froze the preregistration, the script and the module (digests repeated in
`results.json`). 185 s. No interruption.

Output, unedited: [`RESULT-FT-PREP-2026-11/results.json`](RESULT-FT-PREP-2026-11/results.json).

## Outcome

### Part A: any pair of qubits may interact

| | |
| :--- | ---: |
| Shortest encoders, up to symmetry and reordering | 22 976 (8 CNOTs each) |
| Fault-tolerant without a check | 0 |
| Made fault-tolerant by one weight-three check | 22 976 |
| Certified fault-tolerant circuits (11 CNOTs, one ancilla) | 275 712 |
| `c2` among them: lowest / median / highest | 19.12 / 21.40 / 27.88 |
| Circuits at the lowest `c2` | 2 880 |

Every certified circuit has `c1 = 0` exactly. The logical error rate of an accepted preparation is
`c2 * p**2`, so the best gate-optimal circuit has 11 % fewer logical errors than the median one and
31 % fewer than the worst, at the same gate count.

One circuit at the lowest `c2` (qubits 0–6 data, 7 the ancilla; qubits 0, 1, 3 start in `|+>`):
CNOTs `0→2, 0→4, 1→2, 0→5, 1→6, 3→4, 4→6, 6→5`, then the check `1→7, 3→7, 5→7`.

### Part B: a 2-by-4 grid, nearest-neighbour CNOTs, ancilla used for the check only

| | |
| :--- | ---: |
| Ways of laying the code on the grid | 30 |
| Shortest encoder: 10 CNOTs / 11 / 12 | 14 / 14 / 2 layings |
| Layings with a certified fault-tolerant circuit of this form | 5 |
| Best circuit | 14 CNOTs (11 + 3), one ancilla, `c2` = 25.49 |

None of the 14 layings whose encoder needs only 10 CNOTs admits a single-ancilla check on the
ancilla's three neighbours; the five that do need an 11- or 12-CNOT encoder. No laying reached the
enumeration limit (38 652 encoders listed in all).

### Part C: Stim, p = 0.001

| Circuit | Predicted (`c2 * p**2`, or `c1 * p`) | Sampled | Standard error | Sampled / predicted | Accepted runs |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All-to-all, lowest `c2` (19.12) | 1.912e-5 | 1.954e-5 | 0.031e-5 | 1.022 | 198 056 383 |
| All-to-all, median `c2` (21.40) | 2.140e-5 | 2.146e-5 | 0.033e-5 | 1.003 | 198 345 864 |
| All-to-all, highest `c2` (27.88) | 2.788e-5 | 2.773e-5 | 0.037e-5 | 0.995 | 198 044 941 |
| Grid, best (25.49) | 2.549e-5 | 2.601e-5 | 0.036e-5 | 1.021 | 197 834 912 |
| Bare 8-CNOT encoder (`c1` = 2.33) | 2.333e-3 | 2.337e-3 | 0.011e-3 | 1.002 | 20 000 000 |

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | 11 CNOTs and one ancilla, certified; no encoder fault-tolerant alone | 275 712 such circuits; 0 of 22 976 encoders | **held** |
| 2 | Stim within 10 % of `c2 * p**2` for the three all-to-all circuits | +2.2 %, +0.3 %, −0.5 % | **held** |
| 3 | Highest-`c2` circuit worse in Stim, ratio within 10 % of the `c2` ratio | sampled 1.420 ± 0.030, predicted 1.458 | **held** |
| 4 | Bare encoder within 10 % of `c1 * p`, and at least 20 times the checked circuit | +0.2 %; 120 times | **held** |
| 5 | A certified circuit on the grid, Stim within 10 % | 5 layings; best +2.1 % | **held** |

## What this does and does not show

- The search finds the known low-overhead construction for this code: an eight-CNOT encoder and a
  check with one ancilla (Goto, Sci. Rep. 2016; the encoder is Paetznick and Reichardt's). It does
  not find a circuit with fewer gates, and was not expected to.
- The certificate is right where it was checked: five circuits, one noise level, agreement within
  about two standard errors of 2·10^8 shots. The first-order coefficient of a certified circuit is
  zero by enumeration, not by sampling.
- Gate-optimal is not noise-optimal. Among 275 712 circuits of identical size the second-order
  coefficient spans a factor 1.46, and Stim reproduces the difference (1.42 ± 0.03). Choosing by
  `c2` is the part of this that the gate-count searches do not do.
- On the grid, the cheapest encoder is not the one to use: the layings with a 10-CNOT encoder
  cannot be checked by an ancilla that only verifies.

Not shown: anything above distance three; codes other than the Steane code; circuits with flag
qubits or an ancilla used as a bus (the published grid circuits use one, arXiv:2504.01083, and this
family cannot express them, so the 14-CNOT grid circuit is the best of a restricted family and not
a claim about the grid); idle noise; hardware.

## After the run (not preregistered)

- The fault-tolerant circuits number exactly twelve per encoder on average (275 712 / 22 976),
  which suggests two usable checks with six CNOT orders each for every encoder; this was not
  verified encoder by encoder.
- The lowest `c2` found in development (19.52, on the first 1 500 encoders) was not the lowest of
  the full list (19.12).
