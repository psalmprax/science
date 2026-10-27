## Stability Selection

**Module:** `apedp.stability_selection`  
**Status:** `experimental`  
**Basis:** Meinshausen & Bühlmann-style selection-frequency filter over repeated fits.

**What it exposes:**
- `stability_select(fit_fn, ..., k=20, pi=0.9, seed_fn=None, **kwargs)`

  Runs `fit_fn` `k` times with optional per-call seeding, records how often each
  term/feature survives the fitter's own selection/thresholding, keeps those
  with selection frequency >= `pi`, and returns the average coefficient over
  the kept draws.

- Returns a `dict[str, float]`; empty dict if every call fails.

**Intended use case:**
- Sparse regression / symbolic fitters that are *stochastic* (random subsampling,
  random test-function draws, random initialization). For deterministic fitters
  the utility does nothing useful; that is by design, not a bug.

**Known limitations (disclosed, not hidden):**
- It reduces but does not remove spurious terms on hard cases. In the blind
  PDE benchmark weak-form fitter (workspace `/tmp/buffy/blind/`) it improved
  benchmark recovery, but heat under higher noise still failed.
- Coefficients are simple averages over kept draws; no refit after selection.
- It is an *unverified helper* in the core repo right now: there is no
  production caller pinned to it yet. The blind-benchmark fitter path that
  uses it lives in `/tmp/buffy/blind/`, not in the repo core.

**Integrity note:**
- The blind-benchmark usage (`stability_fit.py`, choose_pi=0.9) was validated on
  VALID first and frozen before BENCH. That workflow lives outside this repo.
- The module itself has no repo integration test at this time; adding one is the
  next honest step if this is to graduate from experimental.
