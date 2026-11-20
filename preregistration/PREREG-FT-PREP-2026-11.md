# PREREG-FT-PREP-2026-11: fault-tolerant preparation of the Steane code's logical zero, searched and certified

**Status:** frozen before the run by `preregistration/DIGEST-FT-PREP.json`. Runs once, locally
(`python scripts/run_prereg_ft_prep.py`), writing `preregistration/RESULT-FT-PREP-2026-11/`.

## Why

The circuit-discovery engines in `apedp_dyn/vqc/` minimise operation count. Nothing in the
repository searched for circuits that keep working when a gate fails, or could certify that one
does. `apedp_dyn/vqc/fault_tolerant_prep.py` adds both: a referee that decides fault tolerance by
enumerating every single fault and counts exactly the fault pairs that cause a logical error (the
coefficient `c2` of the logical error rate `c2 * p**2`), and a search ranked by CNOT count, then
ancillas, then `c2`.

The task is a known one. For the Steane code an eight-CNOT encoder followed by a check with one
ancilla is the published low-overhead preparation (Goto, Sci. Rep. 2016, using Paetznick and
Reichardt's encoder), and circuits for a 2D grid have been found by reinforcement learning with the
ancilla also carrying interactions (arXiv:2504.01083). This run does not claim a better circuit
count. It tests whether the search finds the known optimum, whether the exact certificate agrees
with an outside simulator, and whether ranking gate-optimal circuits by `c2` predicts a real
difference.

## What is run

- **Part A, any pair of qubits may interact.** All 22 976 shortest encoders (eight CNOTs), listed
  once per class under the code's 168 symmetries and under reordering of gates on disjoint qubits;
  for each, every single-ancilla check of minimal weight that catches its dangerous errors and
  every order of the check's CNOTs. Every candidate is certified.
- **Part B, a 2-by-4 grid, CNOTs between nearest neighbours only.** Seven data qubits and one
  ancilla on a site with three neighbours; the ancilla is used for the check only. All 30 distinct
  ways of laying the code on the grid; up to 20 000 raw shortest gate sequences per laying.
- **Part C, Stim.** Four circuits are sampled under the same noise model at p = 0.001 with
  2·10^8 shots each: from Part A the circuits with the lowest, the median and the highest `c2`,
  and the best circuit of Part B if there is one. The bare encoder of the lowest-`c2` circuit is
  sampled with 2·10^7 shots.

Noise model: after each CNOT one of the 15 two-qubit Paulis with total probability p; each
preparation and each measurement fails with probability p; idle qubits are noiseless.

## Expectations, fixed now

1. **The search finds the known optimum, certified.** Part A yields circuits with 11 CNOTs and one
   ancilla that the referee certifies as fault-tolerant, and no eight-CNOT encoder is
   fault-tolerant without a check.
2. **The certificate agrees with an outside simulator.** For each of the Part A circuits sampled,
   Stim's logical error rate among accepted runs is within 10 % of `c2 * p**2`, allowing three
   standard errors of the sample.
3. **The ranking is real.** Stim's rate for the highest-`c2` circuit is above its rate for the
   lowest-`c2` circuit, and the ratio of the two sampled rates is within 10 % of the ratio of their
   `c2` (three standard errors allowed).
4. **The check buys an order in p.** The bare encoder's sampled rate is within 10 % of `c1 * p` and
   at least 20 times the rate of the checked circuit built on it.
5. **The grid.** At least one laying admits a certified fault-tolerant circuit of this form, and
   Stim's rate for the best one is within 10 % of its `c2 * p**2` (three standard errors allowed).
   Its CNOT count is reported, not predicted.
6. Reported without a criterion: how many fault-tolerant circuits there are, the distribution of
   `c2` among the gate-optimal ones, rejection rates, the grid's layings one by one, and run time.

A miss is reported as a miss.

## Disclosed before freezing

- Development ran the referee and the search on the first 3 000 encoders of the raw enumeration
  and on cut-down runs of this script (1 500 encoders; 300 gate sequences per grid laying). From
  those: every encoder seen needs a check and one weight-three check suffices, so expectation 1
  restates for the full list what was seen on a part of it; among 18 000 fault-tolerant circuits
  `c2` ran from 19.52 to about 27.9; five of the 30 grid layings had a fault-tolerant circuit, so
  the first half of expectation 5 was also seen in part.
- Stim was run in development at p = 0.01 and 0.02 only, on five circuits, with 10^6 to 4·10^6
  shots; the sampled rates were within 5 % of the prediction. No circuit has been sampled at
  p = 0.001, and none of the circuits Part C will select from the full list is known.
- The full list of 22 976 encoders has been enumerated (to count it and time it) and not
  certified.
- The symmetry reduction relies on a claim, tested on one pair of circuits in
  `tests/test_vqc_fault_tolerant_prep.py`: relabelling the qubits by a symmetry of the code leaves
  every number the referee produces unchanged. Counts of circuits are therefore counts of classes.
- Limits of the circuit family: no flag qubits, no X-basis checks, no ancilla used as a bus, no
  mid-circuit measurement, shortest encoders only. A layout that needs any of these gets "none
  found". The grid circuits in the literature use the ancilla as a bus; this family cannot.
- The criterion certified is the distance-three one (an accepted single fault leaves an X part and
  a Z part of reduced weight at most one). `c2` counts X-type logical failures after ideal
  minimum-weight decoding; for the logical zero a Z error cannot cause one.

## Deviation rule

The run is not repeated. If it is interrupted by the machine, it is started again from the
beginning with the output directory removed, and the interruption is recorded with its time; the
search is deterministic and the Stim seeds are fixed, so a restart reproduces the same numbers.
