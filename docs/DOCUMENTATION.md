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

## 89Q invsqrt Champion Reproduction

| Script | Purpose | Status |
|---|---|---|
| [`scripts/test_shared_pddd_circuit.py`](../scripts/test_shared_pddd_circuit.py) | Rebuilds the 89-qubit shared-PDDD invsqrt circuit and runs the official verifier on it | `audit-only` |
| [`scripts/search_claims_pddd.c`](../scripts/search_claims_pddd.c) | Exhaustive search for the shared multi-output PDDD parameters that script consumes | `audit-only` |
| [`scripts/search_mixed_polarity_pddd.py`](../scripts/search_mixed_polarity_pddd.py) | Mixed-polarity search for the four-output restoration: 26 nodes vs 28 | `audit-only` |
| [`scripts/build_mixed_polarity_candidate.py`](../scripts/build_mixed_polarity_candidate.py) | Builds and verifies that plan as a circuit; finds no static gate saving | `audit-only` |
| [`tests/test_dyn_invsqrt_champion_rebuild.py`](../tests/test_dyn_invsqrt_champion_rebuild.py) | Pins the champion rebuild circuit hash and stage timings | `audit-only` |

**Mixed polarity:** `scripts/search_mixed_polarity_pddd.py` finds a 26-node curve
restoration plan (28 in the incumbent) by letting each AND node choose its control
literal; `scripts/build_mixed_polarity_candidate.py` shows the saving does not reach
the static gate count (1466 either way) and that the residual score difference
shrinks with the shot budget. See §5.1 of the boundary doc.

**Module:** `scripts/test_shared_pddd_circuit`
**Status:** `audit-only`
**Basis:** The champion row in
[`docs/invsqrt_frontier_and_portfolio_synthesis.md`](invsqrt_frontier_and_portfolio_synthesis.md)
is a compiled artifact plus a verifier receipt. This script is the only code path
in the repository that builds that circuit, so it is what makes the claim
reproducible rather than merely recorded.

**Procedure:**

```bash
python3 scripts/test_shared_pddd_circuit.py
```

It reads three tracked inputs — `src/boolean_choices90.json` (coarse6 order and
polarity), `src/shared_davio_results.json` (curve4..6 order and polarity, 28
nodes), `src/invsqrt_calibration.json` (the 44 rotation phases) — plus the
`kickmix` extension and the official `invsqrt` verifier under
`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/`. It writes
`/tmp/invsqrt_89q_shared_pddd.kmb` and prints the verifier receipt. It does not
overwrite anything in `artifacts/`.

**What must match on a successful rebuild** (the first three are exact):

| Field | Expected |
|---|---|
| `num_qubits` | 89 |
| `max_toffoli_count` | 1466 |
| `worst_error_over_tolerance` | 0.9996227434818049 |
| `circuit_sha256` | `ce04c9d54dc42e42455444022b6652a6efb0be2630aca7b485d3efd4e58fa9de` |
| `score` | 157,605.5 at the script's 25,600 shots; the pinned 157,601.0808 in `artifacts/` was produced with 200,540,928 verified shots |

**Known limitations (disclosed, not hidden):**
- `artifacts/` is gitignored, so a clone has neither the committed `.kmb` nor the
  receipt JSON. The circuit hash above is the reproducibility check that survives
  that: a rebuild either produces it or does not.
- `toffoli_count` is a mean over shots and depends on mid-circuit measurement
  outcomes, so it moves in the fourth significant figure with the shot budget
  (1455.499 at full budget, 1455.549 at 25,600 shots). Only `max_toffoli_count`
  (1466) and the circuit hash are budget-independent.
- The script needs a Python environment with the `kickmix` C++ extension built
  and the official verifier importable; it is not importable under a bare
  `python3 -m pytest` run and is not collected by the test suite.

**Module:** `scripts/search_claims_pddd`
**Status:** `audit-only`
**Basis:** Proos–Zalka / Davio shared-node counting for the four restoration
outputs. The search enumerates all 8! variable orders and 256 polarities and
reports the order with the fewest shared AND nodes.

```bash
gcc -O2 -fopenmp -o /tmp/search_claims_pddd scripts/search_claims_pddd.c
/tmp/search_claims_pddd
```

**Measured output:** Test 1 (`{curve3..curve6}`) 46 shared nodes at order
`[3,4,2,1,0,7,6,5]`, polarity 96; Test 2 (`{coarse6, curve4..6}`) 36 shared nodes
at order `[7,4,3,2,1,0,6,5]`, polarity 32. Test 2's parameters are what
`scripts/test_viable_optimization_path.py` measures and what
`build_shared_plan` in the Python builder reproduces as 36 nodes.

**Known limitations (disclosed, not hidden):**
- The five truth tables in `main()` are hardcoded literals; the file does not
  derive them from the curve tables, so a changed residual split would silently
  leave this search answering the old question.
- The node set is capped at 128 entries with a silent drop past that; harmless at
  the 36/46 counts reported here, wrong for larger output groups.
- The reported optimum is never re-validated by building the resulting DAG; the
  cross-check is the Python builder's independent count, which is what the
  36-node agreement above relies on.

**Related:** [`docs/PHOTONIC_ACCELERATOR_BOUNDARY.md`](PHOTONIC_ACCELERATOR_BOUNDARY.md)
states what an external photonic accelerator may propose versus what only the
exact verifier may accept, with the measured pipeline cost (4.44 s end to end),
the precision wall (3.97e-12 turns of unused margin against ~1e-5 analog
conversion), and the measured ceiling on Boolean node sharing (36 nodes minimal
over all 10,320,960 order/polarity pairs; 1 qubit costs 9.95 nodes).
