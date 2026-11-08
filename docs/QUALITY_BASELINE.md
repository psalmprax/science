# Quality baseline (ruff + mypy)

**Measured:** 2026-11-07, on the working tree that produced this file.
**Gate:** `python scripts/check_quality_baseline.py` (run by `make lint` and
`make typecheck`); pinned in `quality_baseline.json`.

## Why a baseline instead of "all green"

The previous quality gate did not measure what the documentation claimed:

- `make lint` ran `ruff check apedp/ apedp_dyn/ tests/ --select E9,F`. That
  rule set (syntax errors and undefined names only) is narrow enough that the
  repository's own configured ruleset — `E,F,W,I,UP,B,C4,RUF,BLE,S` — could
  accumulate findings with the gate still green.
- `make typecheck` ran bare `mypy`, which `pyproject.toml` confines to
  `apedp_dyn/conf/paths.py` and `apedp_dyn/conf/canonical_spec.py`. A green
  type gate therefore said nothing about the rest of the tree.

Measured on this tree with the full configured ruleset and a full package type
pass:

| Path | ruff findings | mypy errors | Status |
|---|---:|---:|---|
| `apedp/` | **0** | 938 | ruff clean; types pinned |
| `tests/` | **0** | — | ruff clean |
| `apedp_dyn/` | **0** | 29 | ruff clean; types burned down 2026-11-08 |
| `scripts/` | 604 | — | pinned |
| `benchmarks/` | 217 | — | pinned |

`apedp/` and `tests/` are held at **zero**. The other three trees are pinned at
their measured counts because clearing the remaining findings is a long, separate
refactor; the gate's job today is to stop the debt growing while that work happens.

## Provenance exclusions (2026-11-08)

`apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src`
is a **vendored upstream Qualtran checkout** shipped alongside the prebuilt
`kickmix.so`, not repository code. It carried 12,186 of the package's 14,486 mypy
errors and thousands of `S101`-class lint findings. Excluding it by provenance
(`[tool.ruff] extend-exclude`, `[tool.mypy] exclude`) is a scope correction, not a
suppression of repository findings: editing that tree would also invalidate the
pinned extension build.

## Risk-first burn-down (2026-11-08)

Rule families were cleared in risk order rather than by count:

| Family | What it catches | Findings cleared |
|---|---|---:|
| `F` | undefined names, unused imports, missing f-string placeholders | 22 |
| `B` | loop-variable closures (`B023`), `zip` without `strict`, unused loop vars, mutable defaults | 35 |
| `S`/`BLE` | blind excepts, scheme-unchecked URL opens, temp-file and password heuristics | 14 |
| `UP`/`RUF`/`C4`/`I`/`E7`/`E4` | dead code, stale `noqa` directives, modern syntax, import order | 81 |

The last two families, `E702` (112) and `E701` (87), were cleared with the
repository's own formatter (`ruff format`, the same command `make format` runs) on
`invsqrt_signed_digit.py` (187 findings) and `run_official_receipts.py`. Formatting
is what exposed three real typing gaps that are now fixed at the source: an untyped
`f` helper, an untyped `C` closure, and the `Bits` truth-table harness. One
semicolon was split by hand in `sweep_dqi_eea_bound.py`, because
`docs/pddd_receipt_recording.txt` records that file as restored to patch-provided
content and a formatter pass would have rewritten it wholesale.

On the type side the high-risk codes were the 1,010-file `attr-defined` pile;
12,186 of those errors were the vendored tree above. The remaining 2,250 are
dominated by annotation debt (`no-untyped-def` 571, `no-untyped-call` 501,
`no-any-return` 392), plus real-type codes now counted honestly rather than
hidden by the vendored noise: `index` 28, `operator` 22, `union-attr` 65,
`assignment` 44. Two `valid-type` (`callable` used as a type) and one `no-redef`
were fixed in this pass.

The `no-untyped-def` programme has started: every `apedp_dyn/gates` entry point
(`run_all_gates`, the five gate functions, the two `_dao_*` helpers and
`ScalingInvarianceGate.add_regime`) is annotated with the real `PhysicalModel` /
`DynProblem` / `Callable[[float], np.ndarray]` types, along with the `Bits`
truth-table harness in `invsqrt_signed_digit.py`, two dataclass dunders, and
`two_region_probe`'s `Callable` annotations. It is 571 sites and 257 of them are in
`apedp_dyn/sandbox`; it needs annotating function by function from each body, not
a bulk edit, so it stays pinned here between passes.

## Rule

**Counts may only move down.** The check fails when any path reports more
findings than its pinned limit. Removing findings is always allowed, and
`--update` re-pins lower counts.

Raising a limit is a deliberate act, not a fix: edit `quality_baseline.json`
*after* recording here what grew, why it is acceptable, and what would remove
it. A baseline raised without that note is the same failure mode this file
exists to prevent — a documented gate that does not gate.

## Commands

```bash
make lint        # ruff check apedp/ tests/ (zero) + baseline check
make typecheck   # strict mypy on conf/ + baseline check
python scripts/check_quality_baseline.py --ruff-only
python scripts/check_quality_baseline.py --mypy-only
python scripts/check_quality_baseline.py --update   # re-pin downward counts
```

## What this baseline does not claim

- It does not claim the pinned findings are acceptable engineering — they are
  recorded debt, not a quality standard.
- It does not type-check `apedp_dyn/` strictly: the 2,302 errors are counted
  non-strictly (`--ignore-missing-imports`), and `scripts/` remains unmeasured
  by any type gate.
- It does not lint the vendored `_qualtran_src` tree at all (see Provenance
  exclusions); its quality is upstream's to certify.
- A count is not a severity ranking: the `apedp_dyn/` total includes rule
  families (blind-except, security heuristics on sandbox harnesses) whose
  triage matters more than their number.

## History

| Date | Change |
|---|---|
| 2026-11-07 | Baseline introduced: `apedp/` and `tests/` brought to 0 ruff findings (175 cleared); other trees pinned at measured counts; `make lint`/`make typecheck` re-pointed at the real ruleset and the baseline check. |
| 2026-11-07 | Re-pinned downward after the same pass: `scripts/` 610→604, `apedp/` mypy 942→938. No limit was raised. |
| 2026-11-08 | Vendored Qualtran tree excluded by provenance (`_qualtran_src`). Risk-first burn-down: `apedp_dyn/` ruff 351→199, mypy 14,486→2,302. Remaining findings are `E701`/`E702` only. No limit was raised. |
| 2026-11-08 | `apedp_dyn/` ruff 199→**0** via `ruff format` (plus one hand split in the patch-provided sweep script). mypy 2,302→2,250 after fixing the typing gaps the formatter exposed and annotating the `gates/` entry points. A transient `--update` that raised the mypy pin to 2,311 was reverted by fixing the cause. |
