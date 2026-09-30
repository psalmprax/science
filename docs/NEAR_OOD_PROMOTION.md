# The near-OOD guard: what it actually does, and what it should do

Commit `689740e` made the near-OOD signal live by passing `near_ood` from the
EXT-BENCH adapter into `synthesize()`. Three consumers had been running on a
permanently-absent input, and `ood_tested` was `False` for all 60 problems.
Wiring it up was correct and the wiring stays.

But the wiring changed 10 of 60 recorded families, and on the problems where
the corpus name determines ground truth, family accuracy went **down**, 24/27
to 22/27. The two regressions were P05 and P08, both `power_law` (the truth)
becoming `rational_transition`. This document is the investigation of why, and
it is worth the space because the answer was not the one the code's own comment
claimed.

## The claim under test

`benchmarks/external_60_adapters.py` states the contract for the near-OOD
signal in the adapter:

> it only ever breaks ties between candidates already inside the noise floor,
> and it can only disqualify a *more* complex model (parsimony tiebreak,
> model_synthesis.py:1660-1670). It never promotes anything.

Read as a specification, that is testable, and it is wrong. The selection loop
contains an `ood_superior` branch that lets a challenger more than 2x better on
near-OOD displace the incumbent **even when the challenger is more complex**.
That is a promotion, and it is unconditional on the tie margin.

## What the data says, in the order it arrived

**First measurement (wrong conclusion, kept because the correction matters).**
Running the P block with and without near-OOD handed to `synthesize`, holding
everything else identical:

| arm | P-block family correct | P05 | P08 |
| --- | --- | --- | --- |
| near-OOD passed | 7/10 | `rational_transition` | `rational_transition` |
| near-OOD withheld | 9/10 | `power_law` | `power_law` |

The natural reading is that the `ood_superior` promotion is what demoted P05
and P08. That reading is wrong, and the referee run is what caught it.

**The referee.** All 60 problems, two arms, identical code, differing only in
`_NEAR_OOD_MAY_PROMOTE`. The judge is **far-OOD MSE**, which the selector never
sees — read by the harness only after the winner was chosen. near-OOD is a
split the selector may look at; far-OOD is one it may not, so far-OOD is the
only honest test of whether looking at near-OOD bought anything.

| | promote | no-promote |
| --- | --- | --- |
| families differing between arms | 5 | — |
| family correct (27 named) | 23/27 | 23/27 |
| median far-OOD MSE (56 finite) | 2.189e-6 | 2.232e-6 |
| far-OOD better / worse / tied | — | 7 / 1 / 48 |
| non-finite far-OOD | 4 | 4 |

Two things to read here. First, **the promotion branch is not what demoted P05
and P08** — those flip in the no-promote arm too. Second, where the promotion
does fire it is mildly *helpful* on far-OOD (7 better, 1 worse, median 2.19e-6
against 2.23e-6), and the five families it moves are all on problems whose
truth the corpus name does not determine. It is not the defect.

## The actual mechanism: a bare ratio with no floor

`model_synthesis.py`'s parsimony tiebreak makes two adjacent decisions in one
loop:

```python
noise_floor = best.train_mse / max(len(y_val), 1)
tie_margin = max(0.15 * best.val_mse, noise_floor)      # floored
...
if not ood_superior and m_comp >= best_comp:
    continue
if m.near_ood_mse > best.near_ood_mse * 1.5:            # NOT floored
    continue
```

The margin is calibrated: a `val_mse` difference below what the train noise can
resolve is not evidence, so the floor is added. The veto beside it is a bare
ratio, and a ratio has no absolute scale — when the incumbent's near-OOD MSE is
small, so is the threshold.

P05 is the measured case. The two candidates the veto compares:

| candidate | val_mse | near-OOD MSE | params |
| --- | --- | --- | --- |
| `rational_transition` (incumbent) | 0.083419 | 0.0039136 | 4 |
| `power_law` (**the truth**, `power-1.5`) | 0.085677 | 0.0058819 | 1 |

The validation gap is 0.00226, comfortably inside `tie_margin` of 0.01251, so
the parsimony rule is entitled to prefer the one-parameter model. It does not,
because the veto threshold is `0.0039136 * 1.5 = 0.0058704` and the power law's
near-OOD MSE is `0.0058819`.

**It misses by 0.20%.** And both MSEs sit roughly 25x *below* the data's own
train noise floor (P05's robust noise estimate is 0.312, so the floor is
~0.097, against near-OOD MSEs of ~0.004). The veto is deciding family identity
on a difference the data cannot resolve.

## The fix, and its measured cost

Floor the veto with the same quantity the parsimony margin uses:

```
vetoed iff  m.near_ood_mse > 1.5 * best.near_ood_mse + floor
```

with `floor = best.train_mse / n_val` (`_NEAR_OOD_VETO_FLOOR = "auto"`, the
shipped default; `0.0` restores the bare ratio exactly).

All 60 problems, both arms in separate processes:

| | bare | floored |
| --- | --- | --- |
| families changed by the floor | — | **1 of 60** |
| family correct (27 named) | 23/27 | **24/27** |
| median far-OOD MSE | 2.2323e-6 | 2.2323e-6 |
| far-OOD better / worse / tied | — | 0 / 2 / 54 |
| non-finite far-OOD | 4 | 4 |

The single change is P05, which returns to `power_law` — the truth. The two
far-OOD movements are P05 (0.0167 to 0.0279, both poor, the truth being a
`0.5783`-MSE power law that this block does not resolve) and C44
(2.6e-14 to 3.2e-10, same family, both negligible). One problem's family label
becomes correct; no abstention changes; nothing else moves.

Reproducibility, checked rather than assumed: the bare arm was re-run in a
separate process and is bit-identical to the earlier no-promote run on all
60 x {family, val_mse, near_ood_mse, far_ood_mse}.

## The promotion branch: kept, documented, honestly

`_NEAR_OOD_MAY_PROMOTE = False`. The measurement does not support enabling it,
and the adapter's stated contract does not describe it. But the reason is not
the one first assumed:

* It is **not** the cause of the P05/P08 regressions. Those are the veto.
* Where it does fire, it is mildly positive on the withheld metric (7 better,
  1 worse across the 5 problems it moves).

So it is disabled because the documented contract forbids it and no evidence
demands it, not because it was measured harmful. The constant stays in the
source, with that reasoning, so the decision is reversible by measurement
rather than by re-deriving this argument.

The adapter's comment is **not** corrected to match the code, because the code
is now the thing that matches the comment. It was wrong when written; the
`ood_superior` branch was the discrepancy, and it is now off.

## What this does not settle

* The floor is calibrated to `train_mse / n_val`, inherited from the parsimony
  margin. That is one defensible noise estimate among several. A different
  estimate would move the threshold; the *shape* of the defect — a ratio with no
  absolute scale — is independent of the choice.
* 27 of 60 problems have a corpus name that determines ground truth. The other
  33 are graded on far-OOD MSE only, and "family correctness" is not defined
  for them. Every accuracy figure above is on the 27.
* P05 and P08 remain wrong at `x` and correct at `x*0.1`. That is a separate
  defect, documented in `SCALE_INVARIANCE_DEFECTS.md`, and the veto floor does
  not touch it. P08 in particular is unchanged by everything measured here.

## Reproducing

```bash
python3 /tmp/veto_ood.py floored    # _NEAR_OOD_VETO_FLOOR = "auto"
python3 /tmp/veto_ood.py bare       # _NEAR_OOD_VETO_FLOOR = 0.0
```

Both write `/tmp/veto_<arm>.json` with per-problem family, val/near/far MSE,
train MSE and n_val, plus the arm's floor recorded per problem so a run that
failed to bind its constant is visible in the output. The scripts are session
scratch; the constants they exercise are committed.
