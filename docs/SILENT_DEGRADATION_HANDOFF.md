# Silent degradation handlers: classification and handoff

`except Exception: logger.debug(...)` is not one problem. It is two, with
opposite correct treatments, and lumping them together is why the pattern
looks unfixable.

**Class A — a crash produces a wrong or misleading result.** The guarded
operation computes a *verdict* (is this dimensionally valid, does this
candidate have a pole in the data, are these two candidates
indistinguishable). Logging and carrying on leaves the verdict at its initial
value, so a check that never ran is reported identically to a check that
passed. These are bugs. The fix is to mark the check failed *and* record why.

**Class B — a crash produces a still-correct result.** The guarded operation
computes something (evaluate this expression), the handler falls back to a
second implementation that also works, and the returned answer is right. The
only loss is observability: the run cannot prove which path executed. The fix
is a counter the caller can read, not a louder log line.

Every site in the repository is listed below with its class. Line numbers are
as of 2026-09-29 and will drift.

## Already fixed (Class A)

### `apedp/physics_constraints.py` — 5 sites

`PhysicsConstraintEngine.evaluate` ends with `is_sound = len(violations) == 0`
and `score = passed_checks / 7`. Handlers that only logged left the flag
`True` and appended nothing, so a check that raised was worth a full point on
a compliance report. `sp.limit` raising on an expression with no origin limit
is routine, not exotic, so a formula that diverges at the origin could be
certified physically sound on a check that never ran.

Each handler now clears its flag and appends `_unverified(check, exc)`, which
names both the check and the exception. Pinned by
`tests/test_failed_checks_are_not_passes.py::TestFailedPhysicsCheckIsNotAPass`,
including a source-level assertion that no `logger.debug` handler returns.

### `apedp/decision/stage_c.py` — 3 sites

1. **Pole extraction** and 2. **singular-root search**: the function had one
   exit for two findings. A candidate was refused when a pole was found and
   simply absent when extraction raised, and the caller could only test for
   presence — so "no pole" and "no answer" reached Stage A looking identical
   and the second one endorsed a model nobody had checked. It now returns
   `(refused, inconclusive)`; an unchecked candidate is withheld, like a
   refused one, and the reason is stated in the Stage A rationale.

   This surfaced a second bug underneath. A fractional-power singularity
   arrives as `Abs(x - c)**0.5`, whose component base is a `Pow`, not an
   `Abs`, so the existing unwrap loop left it intact and `sp.Poly` raised
   `PolynomialError` — the very exception being swallowed. The pole check had
   never examined those components. `_strip_abs_to_linear_form` now strips
   through a positive power.

3. **Paired bootstrap SE probe**: leaving `is_noise_equiv = False` did not
   merely withhold a trade-off note, it kept the candidate out of
   `competitive`, which is what promotes a run to `UNIQUE_SUPPORT`. A crashed
   probe was silently resolving the adjudication toward "one winner". The
   failure is now recorded and disclosed in the rationale, and the
   `UNIQUE_SUPPORT` state carries the missing evidence.

   The trade-off sentence in the same block was also asserting a check it had
   not performed: it hardcoded `|Δho_mse = … <= 1.5e-3 or paired SE p > 0.05`
   regardless of which branch fired, so a run decided by the bootstrap printed
   an inequality that was false next to a number that violated it. It now
   quotes the branch that actually established the equivalence.

## Class A, not fixed — files owned by another session

These are untracked (`??`) in git and being actively authored. Editing them
risks a write collision, and committing a fix would mean committing someone
else's work-in-progress wholesale. Handing them over instead.

**`apedp/singularity_probing.py` — 5 sites, Class A.** Root-finding failures
at `extract_singular_components` internals, the analytic solve, and the Brent
refinement. All currently `logger.debug` and continue. The consequence is the
sharpest version of this bug: a failed root search returns *fewer* poles, so
the probe reports "no singularity detected" — and downstream, a candidate with
a real pole in the observed data passes the Stage A pole refusal. Same
treatment as `stage_c.py`: record it, do not let it read as absence.
(Approximate line numbers: the handlers following lines 77, 120, 122, 148,
150.)

**`apedp/mathematical/mathematical_layer.py` — 2 sites, Class A.** The
critical-strip probe and the singularity probe both fail to `logger.debug` and
fall through with their verdict flags at their initial values, so
`MathematicalDomainAudit` reports an expression admissible on checks that
never ran. Same fix as `physics_constraints.py`. (Handlers following lines
118 and 134.)

**`apedp/special_functions.py:244` — Class A.** On a vectorized-evaluation
failure the handler falls back to `n_pts = len(...) if variable_data else 1`,
so a domain probe over 50 points silently becomes a probe over 1. That is a
shape change, not a fallback to an equivalent answer: the reduced grid can
miss a singularity the full grid would have found. Line 215 (`return expr` on
a simplification failure) is Class B and is fine.

## Class B — correct as written, left alone

These fall back to something that is still right. The only gap is that a run
cannot prove which path executed, and the fix is a counter, not a log line.

**`apedp/rust_bridge.py` — 8 sites.** Every one is a Rust-to-Python fallback
on kernel unavailability, with the same answer either way. This is the case
where a counter matters most: the whole point of the bridge is measurement,
and at present a run in which the Rust path never executed is
indistinguishable from one where it did. A single `fallback_count` on
`RustKernelBridge`, incremented at each of the 8 sites and readable by the
caller, would close it. `logger.debug` is arguably the correct level for the
message; it is the *absence of a summary* that is the gap.

**`apedp/symbolic_equivalence.py` — 3 sites.** Simplification returns the
input unchanged, skeleton recombination falls back to the abstracted form,
and polynomial classification falls through to a string probe. All three are
conservative: a comparison that cannot be simplified reports "not equivalent",
which understates a match rather than inventing one. No action needed beyond
the counter, and even that is optional — nothing downstream reads these as
evidence of equivalence.

**`apedp/composer.py:113`** — the correspondence probe returns `False` on
failure. Conservative, and correct: it declines to assert a correspondence it
could not verify. No action.

## Why the split matters for a lint rule

A rule that fires on every `except Exception` whose body is a log call would
flag 20 sites, of which 15 are correct and 3 of the remaining 5 are unowned.
That is a rule everyone learns to ignore. The distinction that is worth
enforcing mechanically is narrower and is the one this note is built around:

> If the guarded operation sets a verdict that a later boolean or score is
> derived from, a raised exception must leave that verdict *distinguishable
> from a pass*.

`apedp/physics_constraints.py` and `apedp/decision/stage_c.py` are the two
places that derive a boolean or a score from a check. A test asserting that
those two files contain no log-only handler is more likely to survive than a
lint rule over the whole package, and it fails on the case that actually
matters.
