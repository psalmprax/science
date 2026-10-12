# apedp-kernel

The native numerical and dimensional-analysis kernel for APEDP. It is the
module `apedp/rust_bridge.py` imports as `apedp_kernel`; the bridge uses it
for all eight kernel entry points when it is importable, and falls back to the
pure-Python reference engine when it is not.

The root `pyproject.toml` is plain setuptools and does **not** build this
crate, so the kernel is an explicit, separate install:

```bash
cd rust/apedp-kernel
maturin develop          # into the active virtualenv
```

CI does this in `.github/workflows/test.yml`, and then asserts the module
actually imports. That assertion is the point: without it, a missing kernel
turns every differential test into a silent skip, and a skip is
indistinguishable from a pass.

## What is implemented

| File | Contents |
| --- | --- |
| `src/parser.rs` | Tokenizer, precedence-climbing parser, and evaluator for the expression language |
| `src/numeric.rs` | Metrics, matrix conditioning, least squares, precision boundary |
| `src/units.rs` | 7-D unit algebra and compound-unit parsing |
| `src/genetic.rs` | Native multi-island genetic search over equation trees |
| `src/ode.rs` | RK45 ODE integration for compiled systems |
| `src/rime/` | RIME Stage-1 mechanism-graph engine (spec: `docs/RIME_V01_MASTER_SPEC.md`); see below |

### RIME entry points

| Module | Contents |
| --- | --- |
| `rime/graph.rs` | Typed mechanism DAG; series-at-a-time topological evaluation; recursive `State`; structural validation; canonical SHA-256 ids |
| `rime/rewrite.rs` | The five frozen structural rewrites |
| `rime/eval.rs` | Ridge-fitted readout scoring, charged degrees of freedom, parallel batches over Rayon |
| `rime/record.rs` | Immutable content-addressed candidate records |
| `rime/seeds.rs` | Deterministic 60-form seed catalogue (Stage-2 scaffolding) |
| `rime/search.rs` | Deterministic budgeted beam search (Stage-2 scaffolding) |

Python surface: `rime_search_json`, `rime_evaluate_json`, `rime_rewrite_json`,
`rime_evaluate_series_json`. The engine is a **proposer**: it never scores itself
against a holdout slice or decides its own promotion.

## Two things to know before changing any of it

**The parser must stay independent.** It is a from-scratch recursive-descent
parser, not a sympy binding. A sympy binding would make
`tests/test_cross_implementation_conformance.py` compare Python against Python
and pass while proving nothing. Its `Ident` nodes are resolved at `compile`
time, which is also where the Rust side and the Python side first have to agree
about which symbols exist.

**Branch structure and constants mirror the Python fallback exactly.** The
tolerances in the differential suite are sized for implementation differences,
not for rounding differences. If you "improve" a Rust branch so it no longer
takes the same path as `apedp/rust_bridge.py`'s `_fallback_*` functions, the
suite will start failing on rounding noise it was never meant to see — and the
failure will look like a numerical bug in the Rust.

`BASE_DIMENSIONS` in `units.rs` is ordered `(m, kg, s, A, K, mol, cd)` —
metre first, which is *not* SI base-unit order. The Rust code and the Python
test expectations only agree under that ordering, so it is a cross-file
contract, not a cosmetic choice.

## Tests

```bash
cd rust/apedp-kernel && cargo test     # 27 unit tests, no Python needed
```

The Python-side differential and dispatch coverage lives in
`tests/test_cross_implementation_conformance.py`,
`tests/test_rust_bridge_dispatch.py`, and
`tests/test_rust_bridge_fallback_safety.py`. Note that those need the kernel
installed; under an interpreter that cannot import `apedp_kernel` they will
skip rather than fail.