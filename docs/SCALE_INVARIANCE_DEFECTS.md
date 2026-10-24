# Scale-invariance defects in model selection

`tests/test_dimensional_invariance.py` asserts a property that most of the
engine already holds: under a positive rescaling of `x`, every verdict is
unchanged. That is not a stylistic preference. It is the claim that the engine
measures the shape of a relationship rather than the units the caller happened
to choose, and commit `0153761` bought it by moving the regime detector's noise
floor into `y` units after a slope-based floor made 11 of 60 EXT-BENCH problems
change verdict under `x*10`.

The detector kept the property. Model selection did not. Three of the eight
problems in the sampled family check still fail it, and they fail it for two
different reasons.

## The measurement

Each problem synthesised at `x`, `x*0.1`, `x*10`. Same rows, same `y`, same
synthesiser, same seeds. Only the multiplier differs.

| problem | truth (from the corpus name) | `x` | `x*0.1` | `x*10` |
| --- | --- | --- | --- | --- |
| O29 `damped-2.8` | damped oscillation | damped_oscillation, MSE 3.30e-3 | **missing_structural_mechanism**, MSE 8.06e-1 | damped_oscillation, MSE 3.30e-3 |
| O30 `damped-2.0` | damped oscillation | damped_oscillation, MSE 3.37e-15 | **missing_structural_mechanism**, MSE 2.69e-2 | damped_oscillation, MSE 2.36e-15 |
| P07 `power-0.5` | power law | **rational_transition**, MSE 2.88e-4 | power_law, MSE 3.74e-4 | power_law, MSE 3.74e-4 |
| E16 `exp-0.2` | exponential decay | **polynomial_baseline**, MSE 2.28e-4 | **polynomial_baseline**, MSE 1.97e-4 | exponential_decay, MSE 3.35e-4 |

O29 is the one worth reading twice. At `x` the engine names the mechanism and
quotes a fit accurate to 0.3%. At `x*0.1` — the same 30 rows, the same values
of `y`, a multiplier applied to one column — it reports
`missing_structural_mechanism`: the mechanism is *absent*. It was not absent at
`x`. Nothing about the data was lost by multiplying a coordinate by 0.1.

## O29 and O30: a frequency grid in the caller's units. Fixed

`generate_candidate_structures` proposed oscillations on a literal list:

```python
omegas = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0]
```

O29's truth is 2.8 rad per unit `x`, and the list contains it. Rescale by 0.1
and the truth becomes 28 rad per unit `x`. The list's best candidate is 4.0 —
a seventh of the period — so no template in the pool can represent the
oscillation, the pool's best fit is a smooth envelope, the residual is
structured, and the abstention gate correctly reports that *some* mechanism is
missing. Every step of that chain is locally reasonable. The fault is at the
front: the grid is a prior on the units, not on the data, and it is a fair
prior only for a caller whose `x` is O(1).

The scale-free quantity is the phase swept across the observed window,
`w * (x_max - x_min)`, i.e. periods across the data. Gridded in periods, a
candidate transforms exactly the way the truth transforms under a rescaling of
`x`, so the family is reachable at every scale.

Two changes, both additive:

* **The coarse pool** gains candidates at
  `w = 2*pi*p/span` for twelve values of `p` in `_SPAN_RELATIVE_OMEGA_PERIODS`,
  keeping only those the absolute grid does not already reach. The absolute
  grid is not removed: it is a fine-grained prior in the units this corpus
  uses, and the union can only add reachable optima.
* **Stage D's frequency box** was `(0.2, 10.0)` in units of `1/x` — a second
  absolute prior, and a worse one, because a coarse candidate outside it is
  handed to L-BFGS-B as a *seed outside its own search region*. It is now the
  union of the legacy range and a periods-based range, for two reasons taken
  from the code rather than from taste: the pool still proposes the absolute
  grid, so a replacement box would invalidate seeds the pool itself produced;
  and a replacement would *narrow* the box at large spans, converting
  refinements that work today into invalid starts. A union cannot break a start
  that worked.

Result, same three-problem probe:

| problem | `x` | `x*0.1` | `x*10` |
| --- | --- | --- | --- |
| O29 | damped_oscillation, 3.303e-3 | damped_oscillation, 3.304e-3 | damped_oscillation, 3.304e-3 |
| O30 | damped_oscillation, 4.40e-15 | damped_oscillation, 3.43e-14 | damped_oscillation, 2.56e-15 |
| O34 | damped_oscillation, 2.701e-5 | damped_oscillation, 2.699e-5 | damped_oscillation, 2.700e-5 |

Invariant to four significant figures, and the base-scale answer is unchanged
(O29 moves in the fifth digit, from 3.304e-3 to 3.303e-3). Pinned by
`TestOscillationGridIsScaleFree` — cheap, in the default run, over three
problems and three scales — and by the existing `slow` end-to-end
`TestSelectedFamilyIsScaleFree`, which now passes O29.

## P07 and E16: a more complex family winning on a narrow window. Open

Different mechanism, and **not fixed**. Both are cases where the winner at
base scale is the wrong family and a simpler one wins after rescaling.

P07 is a power law, `A*x**0.5` (log-log slope fitted from the data: 0.49991,
intercept 1.0994). At base scale the engine returns
`220.7*x**1.0/(72.9*x**0.5 + 1)` with a *better* validation MSE than the power
law it rejected (2.88e-4 against 3.74e-4). Nothing is being fitted that the
data forbids: over `x` in `[0.5, 4]` a four-parameter rational form can
approximate a power law closely, and it does.

The selection rule that lets it through is not scale-dependent — the parsimony
tiebreak margin is `max(0.15 * best.val_mse, noise_floor)`, which is a
relative quantity. What is scale-dependent is *which families the optimiser can
actually reach*: the rational denominator coefficient `B` in `B*x**q` is
bounded by `(1e-5, a_max)` where `a_max = _amplitude_cap(y_max, family)` is a
**`y`-unit** bound, while `B`'s natural scale is `y * x**-q`. Rescaling `x`
moves the useful `B` out of a `y`-unit box, and the flexible family drops out
of the pool at some scales and not others.

The machinery that would fix this already exists in the file and is already
described as correct: `x_scale_all`, with the comment "Global input
normalization: optimize on `x_tilde = x / x_scale` so amplitudes AND rates
live where the y-scale bounds are meaningful ... `x_scale == 1.0` is an exact
no-op". It is gated to `max|x| < 0.1`, where amplitude bounds "demonstrably
fail". P07 at `x*0.1` has `max|x| = 0.4`, so the gate does not fire and the
normalization that would have made the three scales identical never runs.

Making that normalization unconditional for the families already allowed to
normalize (`_NORMALIZE_FAMILIES`) is the fix. It is not applied because the
same comment records the cost, and the cost was measured by whoever wrote it:
"Stage D landscapes are multimodal, and reparameterization alone can flip
basins (observed: Planck R^2 0.95 -> 0.94 with identical math)". That is a
trade between scale-invariance and byte-identity with a frozen benchmark
record, and it is a design decision rather than a bug fix.

E16 is the same shape with the families swapped: a quartic
`polynomial_baseline` beating `exponential_decay` over `x` in `[0.31, 6]`,
and losing to it once the window is rescaled.

**This is why the sampled test still fails on P07 and E16, and the failure is
the point of the test.** It is not a flaky assertion waiting to be retried; it
is the standing record of a defect that is diagnosed, bounded, and unfixed.

## What is deliberately not claimed

* The fix was measured on three problems from one block. It is additive, and
  the O block's recorded answers are unchanged, but "additive" is an argument,
  not a proof, and the O block was re-run before this was written.
* Nothing here says the twelve period values are optimal. They are a
  defensible range at a stated resolution; a finer grid is a different choice
  with a different cost, and the base-scale answers should be re-measured if it
  is made.
* The rate bounds in Stage D (`k` in `(0, 5.0 * x_scale_all)`) are absolute in
  the same way the frequency box was, and were left alone. They are absolute in
  the *caller's* units but reachable at every scale, so rescaling does not move
  them — which makes them a different (and weaker) version of the same problem,
  not the same defect. They would bind on a caller whose data has a genuinely
  fast decay rate.

## Reproducing

```bash
python3 /tmp/scaleprobe.py O29 O30 O34     # three scales, families + MSE
python3 -m pytest -p no:randomly -q tests/test_dimensional_invariance.py \
    -k OscillationGrid                     # cheap pin, default run
python3 -m pytest -p no:randomly -q tests/test_dimensional_invariance.py \
    -k "SelectedFamily" -m slow           # the deep check; P07/E16 fail here
```

The scaleprobe script is session scratch and is not committed; the tests are.
