# Result: PREREG-ODE-MAIN-SEARCH-2026-11

**Run:** once, 2026-11-19 21:40–23:32 UTC, GitHub Actions run `44621196833` on the branch
`odebench-run-1` at commit `0a5c2af`, which froze the preregistration and the harness (digests
repeated in `results.json`). The smoke job, 32 shards and the collect job all succeeded on the first
attempt; no shard was re-run.

Outputs, unedited, in [`RESULT-ODE-MAIN-SEARCH-2026-11/`](RESULT-ODE-MAIN-SEARCH-2026-11/):
`results.json`, `runs.jsonl` (468 runs) and the 32 shard files, rebuilt from the collect job's log;
all 34 match the SHA-256 the job computed.

## Outcome

ODEBench's 63 systems, 117 components, noise 0 and 1 %, two engines on the same rows. No system was
excluded, no run ended in an error or a timeout.

| | Noise | Components recovered (of 117) | Accurate | Law reported | Reported and wrong | Systems with every component recovered (of 63) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Main search path** | 0 | **80** | 89 | 90 | 1 | 35 |
| **Main search path** | 1 % | **69** | 84 | 94 | 8 | 27 |
| Polynomial sparse regression | 0 | 78 | 91 | 117 | 26 | 34 |
| Polynomial sparse regression | 1 % | 53 | 70 | 117 | 42 | 17 |

"Wrong" is neither recovered nor accurate (R² below 0.999 against the true vector field).

By kind of law (recovered):

| | Noise | Polynomial, degree ≤ 3 (79) | Everything else (38) |
| :--- | ---: | ---: | ---: |
| Main search path | 0 | 67 | 13 |
| Main search path | 1 % | 61 | 8 |
| Polynomial sparse regression | 0 | 78 | 0 |
| Polynomial sparse regression | 1 % | 53 | 0 |

## Expectations

| | Expectation | Result | |
| :--- | :--- | :--- | :--- |
| 1 | At most 5 % of the laws the path reports are wrong (both noise levels) | 9 of 184, 4.9 % | **held**, narrowly |
| 2 | The path recovers at least 65 of 117 at noise 0 and at least 45 at 1 % | 80 and 69 | **held** |
| 3 | Of the 38 non-polynomial components, at least 12 recovered at noise 0 | 13 | **held**, narrowly |
| 4 | At least 20 % of what the sparse regression reports is wrong | 68 of 234, 29.1 % | **held** |

On expectation 1 the 95 % interval is 2.3 to 9.1 %: the count is inside the criterion and the data
do not exclude a true rate above it. The split by noise is uneven, 1 of 90 without noise and 8 of 94
with it.

## Reported without a criterion

- **Side by side.** Without noise the two engines are level in count (80 and 78) and differ in
  kind: 67 components are recovered by both, 13 only by the path (all 13 outside the polynomial
  family), 11 only by the sparse regression (polynomials the path refused or approximated). At 1 %
  noise the path is ahead, 69 to 53: both recover 47, only the path 22, only the regression 6.
- **What each says when it has not found the law.** The regression always answers; the path
  refused 50 of 234 runs (27 without noise, 23 with), 22 of them after the adequacy check turned
  down a law that had survived confirmation. In 4 runs at each noise level the path refused a
  component the regression recovered.
- **Actions.** Of 184 reports, 177 are equivalence classes and 7 are confirmed against all rivals;
  none is unchallenged, because the search stage ran on every component. 156 came from PySR and 28
  from the first pass.
- **Reported laws by kind:** 147 recovered and accurate, 2 recovered and below R² 0.999, 26
  accurate without being the law's form (9 without noise, 17 with), 9 neither.
- **Time.** Median 259 s per component (PySR's budget is 240 s), maximum 426 s; the regression
  takes 0.06 s.
- **The five systems with imprecise clean rates** (11, 41, 55, 59, 61), noise 0, 12 components: the
  path recovered 6, reported an accurate approximation in 3, refused 3 and reported the one wrong
  law of the noise-free runs (below). The regression recovered all 12.

## After the run (not preregistered)

Everything in this section was looked at after the results were known.

**The one wrong report without noise is a correct law with a term that should not be there.**
System 11, `dx/dt = -x**3`. The path reported
`-0.9999999*x0**3 + 1.7e-8*x0 - 4.4e-9*exp(exp(x0))`, confirmed as an exact fit (relative error
5e-8 on the pool). On the development and pool states the last term is nothing; on the scoring
states, which reach larger `x0`, a double exponential is not nothing, and R² is -935 011. The
symbolic check names it: "superfluous term". So an exact-fit confirmation can carry a term whose
coefficient is at the level of the numerical error and whose growth outside the confirmed range is
unbounded. Recorded as audit M-15. Nothing in the frozen path removes such terms.

**The eight wrong reports at 1 % noise are near misses, not gross errors.** Their R² against the
true field runs from 0.990 to 0.9986; six are sine or cosine forms standing in for a rational or
polynomial law over the range visited. They passed contrast confirmation as members of an
equivalence class and passed the adequacy check: at 1 % measurement noise the rates carry errors of
a few percent, and a formula within that of the truth is not distinguishable on 200 rows. This is
the limit of the data, and it is also where the 5 % criterion was nearly missed.

**What this says about audit M-14** (the adequacy check on trajectory rows). The failure recorded
there was an unchallenged first-pass law. Here no law was unchallenged and no report was grossly
wrong at either noise level apart from the superfluous-term case. That is consistent with the
suspicion that the weak point is the unchallenged verdict, and it does not test it.

**Where the path loses to the plain regression.** Eleven polynomial components at noise 0. The
regression's library contains every one of them by construction; the path has to find them in
240 s among all expressions, and it refuses when candidates leave structure in residuals that are
exact to ten digits. A polynomial library as one more proposer in the path would cost nothing and
is the obvious repair.

## What changes

- ODE discovery now has a benchmark result (audit M-07): 80 and 69 of 117 components, 4.9 % of
  reports wrong, against 78 and 53 and 29.1 % for polynomial sparse regression on the same rows.
- Two repairs are indicated and neither is made in the frozen path: drop terms that contribute
  nothing on the confirmed rows before a law is reported (M-15), and add a polynomial library as a
  proposer.
- Nothing here integrates a system forward; the result is about the rate of change on the states
  visited.
