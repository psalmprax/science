# Term-builder structure audit

A term is emitted as `(coefficient * <built expression>)`. Drop the inner
parentheses and the emission is correct for exactly as long as every builder
returns a single monomial — and silently wrong the first time one does not.

That happened. `dxr` is `XR - X`, so the two-DOF materialiser emitted
`(q2*-x1 + x2)`: the remote position entered with coefficient 1, gate 1 saw a
length added to an acceleration, and **every** coupling candidate died for a
formatting reason. The uncoupled rival that the coupled grammar exists to defeat
was the only thing left standing. The same line bug affected `u_relax`.

All four materialisation sites were fixed to parenthesise the built expression.
This audit asks the same question of everything else, numerically:

> does the emitted coefficient multiply the whole built expression?

- **Script:** `scripts/audit_term_builder_structure.py`
- **Record:** `benchmarks/term_builder_structure_audit.json`
- **Pinned by:** `tests/test_term_builder_structure_audit.py`

## Method

For a single-term equation, the emitted right-hand side must satisfy
`rhs(p=2) - 2*rhs(p=1) == 0` at arbitrary state values, where `p` is that term's
own parameter. A mis-bound coefficient cannot pass by coincidence: the sampled
values are deliberately distinct and non-unit, and the residual is a number
rather than a `simplify`-and-hope.

Single-term probes are load-bearing. With two terms on one right-hand side,
doubling one coefficient leaves the other's contribution behind, so
`rhs(2) - 2*rhs(1)` equals minus that other term and every multi-term equation
looks broken. The first version of the script reported the coupled site as
failing for exactly this reason; its residual was `-q0*u1`.

Each builder is also classified structurally and run through materialisation and
gate 1, because "not a single product" has a second face.

## What is not a single product, and which of those is a defect

| class | terms | verdict |
|---|---|---|
| sum, coherent units | `dxr` (`XR - X`), `u_relax` (`U - V`) | **correct after the parentheses fix**; `u_relax` is checked on a velocity-driven probe plant, because on a force-driven plant the sum is incoherent and the unit check refuses it — for a units reason, not a formatting one |
| quotient of sums, scaled | `langmuir_bind`, `hill2_bind`, `hill3_bind` | correct: the scale is a fitted parameter in the state's units |
| quotient of sums, literal `1` | `sat_x`, `sat_v`, `sat_u`, `inv_x`, `inv_x2`, `log_abs_x`, `sqrt_abs_x`, `ret_x` | refused by the unit check on **every** dimensional problem (`dimension mismatch in Add`) — documented dimensionless-only vocabulary, not a silent wrong answer |
| function of a dimensional state | `sin_x`, `cos_x`, `tanh_x` | refused (`dimensional argument to sin`) — a different refusal, same gate |
| unavailable builders | `c1`, `c2`, `grad_x`, `laplacian_x`, `zscore_x`, `ma_x`, `mom_x`, `rsi_like`, `bollinger`, `macd_like` | raise `UnavailableGrammarTerm` with the reason, from the earlier placeholder work |

The literal-`1` family is a real limit worth stating plainly: eight of the
vocabulary's terms cannot be built on any problem with units. They are correct
for a dimensionless state, and the record says so rather than pretending the
gate is a bug. `candidate_ir.TERM_FAMILIES` already labels them
"dimensionless-only vocabulary" and `test_dyn_typed_saturation_grammar.py` pins
it.

## Two emission sites that were still assembling by string

Both are outside the grammar materialisers, and both are the same defect class:
a coefficient formatted onto an operand without parentheses.

1. **`advanced_search.HierarchicalSearch._expansions`** — the mutation family
   applied to a surviving seed. Seeds are frequently sums (the coarse templates
   emit `v1/(1+v2)`), so `f"{expr}*2"` on the seed `x0 + 1` emitted `x0 + 1*2`,
   `f"{expr}**2"` emitted `x0 + 1**2`, and `f"{expr}*{var}"` emitted
   `x0 + 1*x1`. In each case the mutation reached the constant and the search
   silently explored a different family than the one it named. **Live defect**,
   fixed by parenthesising the seed once and formatting that.
2. **`proposers.sindy_proposer.SINDyProposer.fit_and_propose`** — emits
   `f"{c:.6g}*{t}"` for a library column named `t`. Every column is a product
   today (`x0*x1`, `sin(x0)`), so this is **latent**, but a column named for a
   sum would emit `2*x0 + x1` and claim the second variable's coefficient is 1.
   Fixed the same way.

`apedp/discovery/_core.py` builds its rebuilt expression with
`sp.Mul(coeff, part)` — sympy, so structurally immune — and needs no change.

## A third defect the audit found: the fast/slow labels never reached the emission

The multiscale probe emits its fast equation as `(q/tau_fast)*x` and its slow
equation as `(q/tau_slow)*x`. The first run of this audit reported
`q0*x1/tau_slow` for the **fast** equation, and `tau_fast` was `null` because
that symbol appeared nowhere in the emitted expression.

The cause: `enumerate_coupled` records `fast_variables`/`slow_variables` on each
*equation's* metadata, while `_materialize_multiscale` read them from the
**system's** metadata — which only ever holds `family`, `dae_mode`,
`timescales` and `constraint_terms`. Every term therefore fell through to
"unlabeled: default to slow". `tau_fast` was declared as a fitted parameter,
never appeared in any equation, and the two-timescale mode emitted one
timescale while reporting two.

That is a semantic defect of the same shape as the original: an emission site
assuming something about its inputs that the code producing those inputs does
not guarantee. The materialiser now unions the per-equation labels with the
system-level keys, and the audit pins `tau_fast` in the fast equation.

The union alone left the duplication that hid the defect. The function held
three near-identical branches — fast, slow, and unlabelled-falls-back-to-slow
— differing only in which symbol `tau` held, so the fast branch could be wrong
while the slow branch, exercising the same code shape, stayed right. Those
three are now one emission that picks the constant by label, and a term
labelled both fast and slow is refused rather than resolved silently.

The declaration follows the emission. `tau_fast` and `tau_slow` were appended
unconditionally, which is why the previous pin — "is `tau_fast` declared?" —
passed while the symbol appeared in no equation: a declared parameter is not a
used one. Each constant is now declared only if an emitted term used it, and
`tests/test_dyn_d3.py` asserts declared equals used over the emitted
right-hand sides rather than a fixed roster of two.

## The repo-wide sweep: every coefficient-prefix site

The three defects above were found by asking the question of the grammar. The
same question was then asked of the whole library, because the shape — a field,
a `*`, another field — is the defect rather than the file it lives in.

- **Sweep:** `scan_coefficient_prefix_sites()`, over `apedp/`, `apedp_dyn/`,
  `scripts/` and `benchmarks/`
- **Adjudication:** `STRING_ASSEMBLY_SITES`, one entry per site, in the record
  under `string_assembly`
- **Failure mode:** a site that is neither adjudicated nor converted fails
  `tests/test_term_builder_structure_audit.py`, so a new template has to be
  adjudicated before it is reviewed rather than after it mis-binds

47 sites were found, each with a registry entry, and one was a live defect of
the original kind. Twenty-one operands were already parenthesised; the rest are
numbered across the verdicts below.

### The live one

`apedp_dyn/ui/server.py` emitted the promoted θ̈ equation as
`f"ddot(theta) = {coef:.4g} * {expr}"`, where `expr` is the canonical expression
the L2 engine promoted — routinely a sum (`x + x**3`, `-x1 + x2`). The
coefficient applied to the first addend and the portrait of the promoted law was
wrong in the same way the materialisers were. The neighbouring θ and energy
strings already parenthesised, which is exactly how this kind of defect survives
review: the file gets it right twice and wrong once. Fixed in place.

### Two latent families, now closed

- **Caller-named operands.** `TermLibrary.render` (QED), `SINDyProposer`,
  `sparse_microstructure`, `regime_switching`, `invariant_transfer`,
  `distributed_lag_interaction`, `rime_ad` and `stem_proposers` all format a
  coefficient onto a name the caller supplies — a library term, a DataFrame
  column, a feature label. Every name in use today is atomic, so the defect is
  latent; a term called `a + b` would have split its coefficient.
  `apedp_dyn/grammar/term_format.py` now provides `scaled()` and `atom()`, which
  return an atomic operand untouched (so recorded equations keep their exact
  text, and formula-hash candidate ids do not move) and parenthesise the rest.
  A power on a sum-named operand goes through `atom()` too, so `2.0*(a + b)^2`
  never becomes `2.0*(a + b^2)`.
- **The exponential labels.** `discovery/_core.py` and `family_detection.py`
  emitted `exp({c:.4f}*{name})`. For an atomic name that is correct; for a
  sum-named column it shifts the exponent by the second addend — a different
  function, not a formatting nicety. Both now emit `exp({c:.4f}*({name}))`.

### What is recorded as safe, and why

The remaining sites each carry a verdict rather than a silent pass:

| verdict | what it asserts |
|---|---|
| `parenthesised` | the operand reaches the template already wrapped (the four materialisers, `multiterm_engine`, `coordinate_transmuter`, `rime_repair_engine`, `rime_proposer`, the two already-correct `ui/server.py` strings) |
| `atomic_operand` | the operand is a name built in the same module from a fixed table (PDE term names `u`, `u_sq`, `u*u_x`, `(u_x)^2`; `model_synthesis` onsets and extinctions; QASM rotation-parameter labels) or the field calls `atom()` |
| `numeric_product_label` | the string names a product whose values are multiplied numerically — design-matrix columns (`feats[f"{a}*{b}"]`), the reduced-library columns in `de_arm`, and `rime_repair_engine`'s interaction columns. No coefficient is formatted onto these |
| `diagnostic_text` | prose in an `OverflowError`, a budget message, an assertion message |
| `guarantee_itself` | `term_format.scaled`, whose field calls `atom()` |
| `converted` | the literal shape was removed and the module now calls the formatter; the record also checks the marker is still in the file |

`model_synthesis` is the instructive `atomic_operand`: its onsets and extinctions
are atomic fragments (`x**1.5`, `exp(-C * x)`), while the sum-valued saturation
`1.0 + B * x` only ever appears as `({s})`. The template table is therefore safe
as written, and the entry says which fragment is which so the next editor does
not have to re-derive it.

### What the sweep does not look at

`sandbox/` holds a vendored upstream tree and `scratch/` holds one-off analysis
scripts; neither emits an expression the repo records or adjudicates. The audit
script skips itself, because it carries the pattern as data in `STRING_ASSEMBLY_SITES`
and would otherwise report its own registry. The sweep is a source-shape scan:
it finds the formatting sites, and the registry — not the scanner — decides
whether each one is safe, which is why every entry has to carry a reason.

## The registry is a proof about the formatter, not about the proposers

An adjudicated site says that the string at that line calls a formatter that
brackets its operand. It does not say that the proposer feeding that formatter
ever hands it a name containing an operator — every feature column, library
term and design-matrix label in use today is atomic, which is precisely why the
defect was latent at all eight sites.

`tests/test_sum_named_column_binding.py` closes that gap behaviourally: it
builds each proposer's real input with a column literally named `a + b` and
asserts on the formula the emitting entry point returns. Four sites are driven
end to end — regime switching, distributed lag, sparse microstructure and the
RIME-AD multivariate emitter — and every probe fails if it produces no
proposal, so an empty result cannot pass for a clean one. The shared helper is
tested against the unparenthesised form, and the sparse probe re-evaluates the
printed formula against the scorer, so a formula that merely *looks* bound is
not enough.

## The sibling shapes: a power, a quotient, a call and a left factor

The coefficient sweep looked for `}*{`. Three further families fail the same way:

> is a field raised to a power, divided by, used as a function argument, or
> multiplied by a literal, without being bound as one syntactic unit?

`2*(a + b)` written `2*a + b` gives the second addend a coefficient of 1, and
each of these is that mistake one operator further out: `(a + b)**2` written
`a + b**2` raises only `b`; `1/(a + b)` written `1/a + b` puts `b` outside the
reciprocal; `x*(a + b)` written `x*a + b` splits the product; and a negated
argument, `exp(-(a + b))` written `exp(-a + b)`, flips only the first addend.

Six live sites, all latent only because every operand name in use is atomic:

| site | was | now |
|---|---|---|
| `rime_ad_proposer` multi-rational basis | `c1*(1/{feat_name})`, `c2*(1/{feat_name}^2)` | `1/{atom(feat_name)}`; the scorer divides the whole column, so the formula now says so |
| `rime_ad_proposer` Hill form | `{feat_name}**2` in numerator and denominator | `{atom(feat_name)}**2` |
| `epistemic.coordinate_transmuter` | `exp(-c / {feature_name})`, `({feature_name} ** k)` | both through `atom()`; the outer parentheses did not help, since `(x ** 2)` with `x` a sum still raises only the last addend |
| `_core.recombine_power_law` | `f"{name}**{p}"`, and `terms.append(name)` at `p == 1` | `powered()` and `atom()`; the exponent-1 branch mattered because the terms are joined with `*` |
| `feature_selection` atom set (reciprocal, square) | `f"1/{variable}"`, `f"{variable}**2"`, `f"{variable}**3"` | `over()` and `powered()` |
| `feature_selection` atom set (negated exponential) | `f"exp(-{variable})"` | `f"exp(-{atom(variable)})"` — a leading sign is not a delimiter |

`term_format` grew `powered(operand, exponent)` and `over(numerator,
denominator)` for two of the families, on the same rule `scaled` already
follows: return the operand untouched when it is already atomic, so no emitted
string churns and no candidate id moves. The other two need no new helper —
`atom()` is enough — and a call whose field is the entire argument needs
nothing at all: the call's own parentheses already delimit it, which is why most
call sites are recorded as safe rather than changed.

### What the sweep excluded, and why

330 sites carry one of the four shapes — 59 powers, 99
quotients, 43 function arguments and 129 left factors. 185
are not arithmetic at all — progress counters (`{done}/{total}`), filesystem
paths, markdown bold (`**{name}**`), derivative notation (`d{state}/dt`) and
shape tuples — and 94 render a formatted number, which cannot contain a
top-level operator. Those are reported with their reason in the record rather
than being required to carry a registry entry, and the scope rule is part of the
record so a reader can see what it skipped:

- **in scope** — the operand field is named like an operand (`name`, `feature`,
  `variable`, `term`, `expr`, `operand`, `seed`, `tokens[…]`, `reciprocal`, or a
  formatter call) and carries no numeric format spec;
- **numeric operand** — the field has a numeric format spec, so it renders a
  number;
- **display only** — the field is not named like an operand: a counter, a path,
  a markdown label or a shape tuple.

The naming rule is a heuristic, and it is stated as one: a field that holds an
expression but is named, say, `z`, would be reported as display-only and escape
the registry. What it buys is a decidable census — 51 sites adjudicated by
hand with the other 279 accounted for — rather than a 330-entry list
nobody reads.

## Reproducing

```bash
python3 scripts/audit_term_builder_structure.py
pytest tests/test_term_builder_structure_audit.py -v
pytest tests/test_sum_named_column_binding.py -v
```

## What this does not cover

One problem per materialiser: the Duffing plant for single-DOF and the coupled
two-DOF plant for the system paths. A builder can only be refused by the unit
check on a problem whose units expose it, so "dimensionless-only" is a statement
about dimensional problems. The SINDy and expansion sites are audited with a
synthetic sum-valued operand, which makes their defect latent rather than
absent. The constrained-DAE path is not audited here because it refuses to
enumerate at all (`dae_mode='constrained'` raises `NotImplementedError`).
