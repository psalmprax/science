# APEDP-QED — quantum-assisted equation discovery (experimental)

`Status: experimental`. Code in [`apedp_dyn/qed/`](../apedp_dyn/qed/), record in
[`benchmarks/equation_discovery/qed_001_recovery.json`](../benchmarks/equation_discovery/qed_001_recovery.json),
produced by [`scripts/measure_qed_recovery.py`](../scripts/measure_qed_recovery.py).

## 1. The question this answers, and the one it does not

APEDP-VQC asks whether a circuit can be *constructed and verified*. QED asks a
different question: can a quantum-assisted search act as a **discovery
instrument** — a proposer that suggests a governing equation, which APEDP-X then
adjudicates without knowing or caring that a circuit was involved?

The falsifiable version, and the only one this document claims to answer:

> Given noisy observations of a damped harmonic oscillator and a fixed ten-term
> library, does a QAOA-mediated search recover the generating equation as well as
> four classical proposers and a RIME variant, when every arm is held to the same
> number of objective fits?

**Measured answer: no.** At 256 fits the quantum arm recovered the generating
support in 1 of 3 runs, while hill climbing, simulated annealing, the genetic
algorithm and boolean GP each recovered it in 3 of 3. At 64 fits it recovered it
in none of 3 runs while boolean GP recovered it in 3 of 3. Three measured reasons
are given in §5, and one of them is a defect in this experiment's own first
version (§6).

What this is *not*: it is not a test of quantum hardware, not a claim about
quantum advantage in either direction, and not a claim that "quantum cannot do
physics". It is one task, one library, one noise level, one simulated circuit.

## 2. What was built

| Module | Responsibility |
|---|---|
| `library.py` | The term library: named scalar functions of the observed state. The whole hypothesis space, fixed before any seed is drawn |
| `objective.py` | `J(z) = mean residual² + λ|z|` with the coefficients re-fitted per selection, and a hard budget on distinct fits |
| `tasks.py` | A task: development rows, holdout trajectory, library, and the answer. Also the RK4 rollout used by the referee |
| `arms.py` | Six proposers over the selection vector, all charged in the same currency |
| `qaoa.py` | QUBO construction from exact singleton/pair fits, diagonal phase separator, mixer, angle search, measurement |
| `referee.py` | The five checks and the four statuses |
| `records.py` | The record: conditions, exhaustive fairness sweep, circuit probes, an advantage block computed from the rows |
| `proposer.py` | The `CandidateProposal` adapter, so QED competes inside APEDP-X and never adjudicates itself |

The six arms, all searching the same ten binary variables:

| Arm | Mechanism |
|---|---|
| `random_restart` | First-improvement hill climbing from random starts |
| `simulated_annealing` | Metropolis on single-bit flips, geometric cooling |
| `genetic_algorithm` | Bitstrings, tournament selection, uniform crossover, one-flip diversification |
| `boolean_gp` | Genetic programming over a **set-building language** (`and`/`or` over term leaves): variable-size trees, phenotype = support |
| `rime_relaxed` | RIME soft/hard operators on a relaxed vector in `[0,1]`, hardened by sampling `z ~ Bernoulli(r)` |
| `qaoa_surrogate` | QAOA over a quadratic surrogate of `J`, paid from the same budget, answer = best **scored candidate** |

Two deliberate decisions, both stated in the code:

* The GP arm evolves set-valued expressions, not arithmetic expression trees.
  Products and compositions are a different hypothesis space, so scoring them
  against a subset objective would compare quantities that mean different things.
  Arithmetic-tree GP is named as untested rather than smuggled in.
* The RIME arm is the RIME operator set on a *different* search space than
  `apedp_dyn/proposers/rime_proposer.py`, which evolves financial mechanisms
  through the native Rust kernel. No native RIME code is used here.

## 3. The task, and why it is fair

QED-001 is a damped harmonic oscillator, `x'' = −1.44 x − 0.15 v`, integrated
with RK4 at `dt = 0.02` from two different initial conditions: development from
`(2.5, 0.0)`, holdout from `(2.5, 1.2)`, so the holdout is an extrapolation in
state space rather than the saturated tail of one trajectory. Gaussian noise with
`σ = 1%` of the development velocity scale is added to both coordinates
independently, and the target is the centred finite difference of the observed
velocity.

The library is ten terms: `1, x, v, x², v², xv, x³, v³, sin(x), sin(v)`.

Only the acceleration equation is discovered. The kinematic identity `dx/dt = v`
is **supplied**, which is why the referee reports the companion coordinate and
does not count it: recovering `x` from `x' = v` is not evidence about the
discovered law.

Before any arm runs, the harness sweeps all `2¹⁰ = 1024` selections — outside
every search budget — to answer the question that decides whether the experiment
means anything:

| | |
|---|---|
| penalised optimum | `{x, v}`, `J = 0.347214` |
| runner-up | `{x, v, x³}`, `J = 0.348833` |
| margin | **0.464 %** |
| the optimum is the generating support | yes |

The margin is small because the parsimony penalty, not the residual, is what
ranks them: adding `x³` buys a residual improvement just smaller than
`λ = 0.002` costs. Without this number a missed optimum would be
indistinguishable from a badly posed problem; with it, a miss is a search
failure.

## 4. The referee

Five questions, each reported separately, and the record keeps them apart:

1. **Structural recovery** — is the support the generating support?
2. **Forward prediction** — integrate from the holdout initial condition; is
   `R² ≥ 0.98` on the discovered coordinate?
3. **Parameters** — does every true coefficient lie inside a 95 % residual
   bootstrap interval at the candidate's own support?
4. **Parameter stability** — do those coefficients survive 16 subsample refits at
   20 % relative tolerance?
5. **Ablation separability** — remove one term, add one term, and score each on
   the holdout rollout. A term that can be deleted at no cost makes the candidate
   inseparable from its own ablation.

The status is deliberately two statements: `status ∈ {VALID, ALTERNATIVE,
INVALID, ABSTAIN}` says whether the candidate predicts unseen data, and
`recovered` says whether the structure matched. A different equation that
predicts the holdout as well as the truth is `ALTERNATIVE` — a
non-identifiability result, not a win and not a failure. `ABSTAIN` is an arm that
could not run, with its reason.

`HOLDOUT_R2_FLOOR = 0.98`, `STABILITY_TOLERANCE = 0.2` and
`SEPARATION_TOLERANCE = 1e-4` are **experiment-local**. They are not in
`specifications/canonical_preregistered_spec_v1.json` and must not be cited as
preregistered values.

## 5. Results

### 5.1 Budget 256 fits (25 % of the space)

| Arm | Fits spent | Recovered | Statuses | Best `J` |
|---|---|---|---|---|
| `random_restart` | 200 / 156 / 192 | **3 / 3** | VALID ×3 | 0.347214 |
| `simulated_annealing` | 142 / 164 / 165 | **3 / 3** | VALID ×3 | 0.347214 |
| `genetic_algorithm` | 256 / 238 / 256 | **3 / 3** | VALID ×3 | 0.347214 |
| `boolean_gp` | 122 / 78 / 50 | **3 / 3** | VALID ×3 | 0.347214 |
| `rime_relaxed` | 102 / 104 / 48 | 0 / 3 | ALTERNATIVE ×3 | 0.348898 |
| `qaoa_surrogate` | 256 / 256 / 256 | **1 / 3** | VALID ×1, ALTERNATIVE ×2 | 0.347214 |

### 5.2 Budget 64 fits (6.25 % of the space)

| Arm | Fits spent | Recovered | Statuses | Best `J` |
|---|---|---|---|---|
| `random_restart` | 64 / 64 / 64 | 2 / 3 | VALID ×2, ALTERNATIVE ×1 | 0.347214 |
| `simulated_annealing` | 53 / 58 / 58 | 2 / 3 | VALID ×2, ALTERNATIVE ×1 | 0.347214 |
| `genetic_algorithm` | 64 / 64 / 64 | 1 / 3 | VALID ×1, ALTERNATIVE ×2 | 0.347214 |
| `boolean_gp` | 64 / 64 / 50 | **3 / 3** | VALID ×3 | 0.347214 |
| `rime_relaxed` | 64 / 64 / 48 | 0 / 3 | ALTERNATIVE ×3 | 0.348898 |
| `qaoa_surrogate` | 64 / 64 / 64 | 0 / 3 | ALTERNATIVE ×3 | 0.350877 |

`J` is the best of three seeds; the record keeps every per-seed row, including
the selections the arms returned. Note what the two tables say together: at 256
fits the classical arms are **not** separable — four of them tie at the optimum —
so that budget ranks nothing. The 64-fit condition is where the arms separate: the
quantum arm returned the worst best-objective of all six and recovered nothing,
the same recovery count as the RIME variant and worse than every bitstring or
tree search.

### 5.3 Why the quantum arm did not win — three measurements

**Its surrogate's minimiser is not the answer.** The quadratic pseudo-Boolean
approximation is built from exact fits of every singleton and every pair
(`1 + 10 + 45 = 56` fits). Its own minimum, on all three seeds, is an eight-term
selection — `{1, x, v, x², v², x³, v³, sin(v)}` — because a quadratic
reconstruction under-estimates the higher-order interactions and is not bounded
below at the dense corner. The thing QAOA optimised is therefore not the thing
the experiment wanted, and the record says so in its own block rather than
folding the two together.

**The circuit put almost no probability on the answer.** A harness-side probe
re-runs the arm's circuit from its own seed (same surrogate, same generator, no
search budget) and asks what probability it assigned to the generating support.
The arm cannot compute this — it does not know the answer. Measured:

| seed | `p(truth)` | uniform `2⁻¹⁰` | rank of the answer | effective selections `1/Σp²` | peak `p` |
|---|---|---|---|---|---|
| 11 | 0.001215 | 0.000977 | 447 / 1024 | 748.6 | 0.00195 |
| 12 | 0.001253 | 0.000977 | 412 / 1024 | 749.2 | 0.00192 |
| 13 | 0.001281 | 0.000977 | 347 / 1024 | 741.5 | 0.00198 |

The distribution is nearly flat over the space — it behaves as if choosing among
~749 of 1024 selections — and its mass sits on dense selections, which is where
the surrogate's minimum is. Whatever the arm found, it did not find it because
the circuit pointed at it.

**It spent most of a lean budget on the model.** At 64 fits, 56 went to the
surrogate and 8 candidates were scored. At 256, 200 were scored. That is a
legitimate strategy and it is reported as one — but it is a *classical*
model-then-sample strategy wrapped around a simulated circuit, and on this task
it was worse than spending the same 64 fits on hill climbing.

## 6. A false positive this experiment caught in itself

The first version of the arm reported the best selection **the whole run
evaluated**, and it recovered the generating support in 3 of 3 runs at both
budgets. That looked like parity at 256 and a win at 64.

The circuit probe is what exposed it: if the circuit's probability on the answer
is 0.0012 at rank 447, an armed arm cannot be finding it *through the circuit*.
It was not. The surrogate expansion evaluates every singleton and every pair, so
`{x, v}` — a two-term support — had already been scored as one of the 56
surrogate fits, and the arm was crediting its own **exhaustive pair enumeration**
to the circuit.

The fix is in `arms.py`: an arm may declare its answer, and the QAOA arm declares
only the best *scored candidate* from the sampled set. After the fix the same
record reads 1 / 3 and 0 / 3. Both versions are deterministic from the recorded
seeds, so the difference is a design decision, not noise.

This is recorded because it is the failure mode this platform exists to prevent:
a number that is correct, reproducible, and means something other than what it
looks like.

## 7. The non-identifiability result

`rime_relaxed` never recovered the generating support. It returned
`{x, v, sin(x)}` — coefficients `(−1.484, −0.152, 0.071)` — in every run at both
budgets, and the referee graded it **ALTERNATIVE**:

| Check | Result |
|---|---|
| structural recovery | false |
| forward prediction | `R² = 0.9993` (three seeds: 0.99927, 0.99970, 0.99970) |
| parameters inside the interval | true — the true coefficients are inside their intervals at this support |
| parameter stability | **0.0** — no subsample refit stays within 20 % of the full-data coefficients |
| ablation separability | **false** — the `sin(x)` term can be removed at no cost |

That is the honest meaning of an alternative: on this data the extra term is
free, the coefficients are unstable, and the forward prediction is
indistinguishable from the truth. Calling it wrong would be wrong; calling it a
discovery would be worse.

For contrast, the generating support *is* ablation-separable:

| Ablation of `{x, v}` | holdout `R²` | cost of removing |
|---|---|---|
| remove `x` | −1.072 | 2.071 |
| remove `v` | 0.846 | 0.154 |
| add the best term (`1` or `v²`) | 0.999704 | gain `1.5 × 10⁻⁶` |

so `SEPARATION_TOLERANCE = 1e-4` is what separates "an extra term is free" from
"the extra term buys 1.5 parts per million of explained variance".

## 8. What this does not show

* One synthetic system, one ten-term library, one noise level, one `dt`. Nothing
  here calibrates difficulty or transfers to another library.
* The derivative is a centred finite difference of a noisy signal, so its
  pointwise error is dominated by gradient noise (~2.1 against a target scale of
  2.0). A smoothing or total-variation estimator would change every number and
  was not measured.
* Only the acceleration equation is discovered; `dx/dt = v` is supplied.
* The task is fair only in the benchmark sense — an exhaustive sweep confirms the
  optimum is the truth. A real discovery has no oracle.
* The circuit is simulated densely with no shot noise in the angles, no device
  calibration and no error model. The rotation count (10 `RZ` + 45 `RZZ` + 10 `RX`
  per layer, 130 rotations at depth 2) is the honest circuit cost of the
  separator and nothing else.
* **No advantage is claimed in either direction.** Ten selection bits and a
  classical simulator cannot support a speedup claim; what is measured is search
  quality at a fixed budget.
* Approach A (differentiable quantum circuits, the published QMoD route) and
  Approach C (a quality-diversity sampler over circuits) were **not** tested.
  Only the combinatorial-search idea was.
* Three seeds per arm per condition. Enough to see a broken arm, not enough to
  estimate a variance.
* Two budgets, one space. The record says how the arms rank at those budgets and
  nothing about the budget where they rank stably.

## 9. Prior work and what would make this stronger

The idea is not new and the record does not present it as new: *Quantum
Model-Discovery* (QMoD, 2021) demonstrated differentiable quantum circuits plus
sparse optimisation for discovering differential equations from observational
data. QED-001 does not reproduce QMoD and should not be compared to it: QMoD
learns a *function and its derivatives* with a differentiable circuit, while this
experiment searches a **binary selection** over a classical library and uses the
circuit as a sampler for that selection.

The routes that could actually move the needle, in the order the evidence
suggests:

1. **Estimate derivatives properly.** The target's noise dominates the residual
   at 1 % observation noise, and every arm is fighting it. A weak-form or
   total-variation derivative estimator would raise the oracle margin, which is
   the single number that decides whether search differences are measurable at
   all.
2. **Widen the library until the surrogate is not obviously wrong.** Ten terms
   make the quadratic reconstruction lossy at the dense corner while the answer
   is at size two. A library of 20-30 terms is where a QUBO formulation has a
   chance of being a faithful model.
3. **Test Approach A.** A differentiable-circuit proposer is a different
   hypothesis generator, not a better sampler, and this experiment cannot say
   anything about it.
4. **Only then consider hardware.** Running a near-uniform 10-qubit distribution
   on a device would add noise to a search that is already not working; the
   circuit has to be worth executing first.

## 10. Reproducing

```bash
python scripts/measure_qed_recovery.py
```

Deterministic: every arm is seeded, every fit is `numpy.linalg.lstsq`, and the
record pins the SHA-256 of the script and all nine modules. The record's checks
assert that the task is fair, that the budget was respected, that the space is
larger than the budget, that no advantage is claimed, and that no proposer could
reach the holdout — and that last one is structural, not asserted: an arm is
handed a `SupportObjective` built from development rows only, and
`tests/test_qed_recovery.py` pins that the object carries no holdout attribute.
