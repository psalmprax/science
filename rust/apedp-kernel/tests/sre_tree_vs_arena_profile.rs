//! SRE gate instrumentation: tree-vs-arena evaluation throughput baseline.
//!
//! Evidence base for `docs/rust_migration_decision_record.md` §4 rank 1
//! ("port `genetic.rs` trees onto the arena `Graph` — Rust→Rust"). The
//! migration is supposed to be evidence-gated; this test generates the
//! baseline:
//!
//! 1. Deterministic SRE-shaped workload (random GP trees grown by the
//!    production `GeneticOps::random_tree` builder, same flags the SRE
//!    search uses for its arithmetic core).
//! 2. Conversion of each tree onto the `rime::graph::Graph` arena over the
//!    SEMANTICS-IDENTICAL subset only (Const/Var/Add/Sub→Add∘Neg/Mul/Neg and
//!    the exact-matching unary calls Abs/Sin/Cos/Tanh). Div (tree `a/b` vs
//!    arena pole-protected `SafeDiv`), Pow (IEEE powf vs safe rational
//!    power), Min/Max (absent from the frozen graph set) and the remaining
//!    transcendent calls (unbounded exp/ln/log/erf/… vs bounded variants)
//!    are NOT mapped — the honest conversion COVERAGE FRACTION is part of
//!    the evidence, because it bounds how much of a production SRE pool the
//!    arena could even represent without semantic drift.
//! 3. Bit-level parity gate on the covered candidates (NaN-matching vector
//!    equality between the per-row tree evaluator and the series-at-a-time
//!    arena evaluator).
//! 4. Throughput measurement of BOTH engines over the identical candidate
//!    set, written to `benchmarks/sre_tree_vs_arena_baseline.json` with the
//!    recorded gate decision text.

use std::time::Instant;

use apedp_kernel::genetic::{FastRng, GeneticOps, NodeExt};
use apedp_kernel::parser::{Func, Node as TreeNode};
use apedp_kernel::rime::graph::{Graph, Node as GraphNode};

const ROWS: usize = 60_000;
const NUM_VARS: usize = 2;
const N_CANDIDATES: usize = 300;
const MAX_DEPTH: usize = 4;
const REPS: usize = 3;

/// Convert a tree onto the arena over the semantics-identical subset.
/// Returns `None` when the tree uses any operator the arena cannot express
/// without semantic drift (Div, Min/Max, Pow, unbounded transcendentals).
fn tree_to_graph(tree: &TreeNode) -> Option<Graph> {
    let mut g = Graph::new();
    convert(tree, &mut g)?;
    g.output = g.nodes.len() - 1;
    // Structural validation is part of the evidence: a bad conversion must
    // fail loudly, never score NaN-by-accident.
    g.validate().map(|_| g).ok()
}

fn convert(node: &TreeNode, g: &mut Graph) -> Option<usize> {
    let idx = match node {
        TreeNode::Const(c) => g.push(GraphNode::Const(*c)),
        TreeNode::Var(v) => g.push(GraphNode::Input(*v)),
        TreeNode::Add(l, r) => {
            let a = convert(l, g)?;
            let b = convert(r, g)?;
            g.push(GraphNode::Add(a, b))
        }
        // IEEE-exact: a − b ≡ a + (−b) for every non-trapping f64.
        TreeNode::Sub(l, r) => {
            let a = convert(l, g)?;
            let b = convert(r, g)?;
            let nb = g.push(GraphNode::Neg(b));
            g.push(GraphNode::Add(a, nb))
        }
        TreeNode::Mul(l, r) => {
            let a = convert(l, g)?;
            let b = convert(r, g)?;
            g.push(GraphNode::Mul(a, b))
        }
        TreeNode::Neg(i) => {
            let a = convert(i, g)?;
            g.push(GraphNode::Neg(a))
        }
        TreeNode::Call(f, i) => {
            let a = convert(i, g)?;
            match f {
                Func::Abs => g.push(GraphNode::Abs(a)),
                Func::Sin => g.push(GraphNode::Sin(a)),
                Func::Cos => g.push(GraphNode::Cos(a)),
                Func::Tanh => g.push(GraphNode::Tanh(a)),
                _ => return None, // unbounded / differently-clamped semantics
            }
        }
        // No arena counterpart with identical semantics:
        TreeNode::Div(..) | TreeNode::Min(..) | TreeNode::Max(..) | TreeNode::Pow(..) => {
            return None
        }
        TreeNode::Ident(_) => unreachable!("resolved trees only"),
    };
    Some(idx)
}

/// The production tree hot loop: per-row recursive evaluation exactly as
/// `score_candidate_affine` drives it (minus the affine readout, so the
/// baseline isolates EVALUATION throughput).
fn tree_eval_all(node: &TreeNode, data: &[Vec<f64>], n: usize) -> Vec<f64> {
    let num_vars = data.len();
    let mut row_buf = vec![0.0f64; num_vars];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        for (v, col) in data.iter().enumerate() {
            row_buf[v] = col[i];
        }
        out.push(node.eval_row(&row_buf));
    }
    out
}

fn values_match(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

fn bench<F: FnMut()>(mut f: F) -> f64 {
    f(); // warm-up (allocator/cache state), untimed
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
fn sre_tree_vs_arena_baseline() {
    // ── deterministic workload ───────────────────────────────────────────
    let mut rng = FastRng::new(2026_10_22);
    let mut data: Vec<Vec<f64>> = vec![Vec::with_capacity(ROWS), Vec::with_capacity(ROWS)];
    for t in 0..ROWS {
        let x = (t as f64 / ROWS as f64) * 6.0 - 3.0;
        data[0].push(x);
        data[1].push(0.5 * x * x - 1.7 * (0.9 * x).sin());
    }
    let mut trees = Vec::with_capacity(N_CANDIDATES);
    for k in 0..N_CANDIDATES {
        let node = GeneticOps::random_tree(&mut rng, MAX_DEPTH, NUM_VARS, false, false, false, false);
        // Avoid pathological single-leaf trees in the timing set: keep only
        // candidates with at least a few nodes (the SRE pool's realistic
        // bottom tail is bigger than depth-1 leaves).
        if node.complexity() >= 4 {
            trees.push((k, node));
        }
        if trees.len() == N_CANDIDATES {
            break;
        }
    }
    let n_cand = trees.len();
    assert!(n_cand >= 100, "workload degenerated: only {n_cand} candidates");

    // ── conversion coverage (part of the gate evidence) ──────────────────
    let mut graphs = Vec::with_capacity(n_cand);
    let mut covered = 0usize;
    let mut uncovered_complexity = 0usize;
    for (k, tree) in trees.iter() {
        match tree_to_graph(tree) {
            Some(g) => {
                covered += 1;
                graphs.push(g);
            }
            None => uncovered_complexity += tree.complexity(),
        }
    }
    let coverage = covered as f64 / n_cand as f64;
    println!(
        "conversion coverage: {covered}/{n_cand} = {coverage:.3} (uncovered tree complexity: {uncovered_complexity})"
    );

    // ── parity gate on the covered subset (exact, NaN-matching) ──────────
    let mut mismatches = 0usize;
    let mut gi = 0usize;
    for (_k, tree) in trees.iter() {
        let g = match tree_to_graph(tree) {
            Some(g) => g,
            None => continue,
        };
        let tree_out = tree_eval_all(tree, &data, ROWS);
        let arena_out = g.evaluate(&data);
        if !values_match(&tree_out, &arena_out) {
            mismatches += 1;
            if mismatches <= 3 {
                println!(
                    "PARITY MISMATCH candidate {gi}: first diffs {:?}",
                    tree_out
                        .iter()
                        .zip(arena_out.iter())
                        .filter(|(a, b)| !(a == b || (a.is_nan() && b.is_nan())))
                        .take(3)
                        .collect::<Vec<_>>()
                );
            }
        }
        gi += 1;
    }
    assert_eq!(mismatches, 0, "tree-vs-arena parity gate FAILED on {mismatches} candidates");
    assert_eq!(gi, covered, "parity pass must cover exactly the converted set");

    // ── throughput: identical covered set, both engines ──────────────────
    let covered_list: Vec<TreeNode> = trees
        .iter()
        .filter(|(_k, t)| tree_to_graph(t).is_some())
        .map(|(_k, t)| t.clone())
        .collect();
    let covered_graphs: Vec<Graph> = covered_list.iter().map(tree_to_graph).flatten().collect();
    assert_eq!(covered_list.len(), covered_graphs.len());

    let tree_ms = bench(|| {
        for t in &covered_list {
            for _ in 0..2 {
                let _o = tree_eval_all(t, &data, ROWS);
            }
        }
    });
    let arena_ms = bench(|| {
        // The arena walks one CANDIDATE at a time through all rows — the
        // same visit order a per-candidate scorer would drive it with.
        for g in &covered_graphs {
            for _ in 0..2 {
                let _o = g.evaluate(&data);
            }
        }
    });

    let per_candidate_us_tree = tree_ms * 1000.0 / (2.0 * covered_list.len() as f64);
    let per_candidate_us_arena = arena_ms * 1000.0 / (2.0 * covered_graphs.len() as f64);
    let ratio = tree_ms / arena_ms;
    println!(
        "tree = {tree_ms:.1} ms ({per_candidate_us_tree:.1} µs/candidate), \
         arena = {arena_ms:.1} ms ({per_candidate_us_arena:.1} µs/candidate), \
         arena/tree = {ratio:.2}x"
    );

    // ── gate decision text (recorded, not asserted to a threshold) ───────
    let gate_decision = if ratio >= 2.0 {
        "ARENA-FAVORED: the port case is measured-strong; proceed to the pinned-parity engine step"
    } else if ratio >= 1.2 {
        "MARGINALLY-ARENA-FAVORED: port value depends on the conversion coverage below"
    } else if ratio > 0.83 {
        "PARITY-THROUGHPUT: no decisive throughput case from representation alone; \
         the dump should weigh the coverage fraction and structural-sharing upside instead"
    } else {
        "TREE-FAVORED: the arena does not deliver a throughput win on unshared GP trees; \
         the port case must come from sharing/dedup benefits, not evaluation speed"
    };
    println!("GATE: {gate_decision}");

    // ── artifact ─────────────────────────────────────────────────────────
    let artifact = serde_json::json!({
        "recorded_at": chrono_like_now(),
        "purpose": "SRE gate instrumentation for docs/rust_migration_decision_record.md rank 1 (tree-vs-arena)",
        "workload": {
            "rows": ROWS,
            "num_vars": NUM_VARS,
            "n_candidates": n_cand,
            "max_depth": MAX_DEPTH,
            "rng_seed": 20261022,
            "builder_flags": {"allow_trans": false, "allow_exp": false, "allow_misc": false, "allow_advanced": false},
            "reps": REPS,
            "loops_per_rep": 2,
        },
        "conversion": {
            "covered": covered,
            "coverage_fraction": coverage,
            "unmapped_ops": ["Div(->SafeDiv differs)", "Min/Max(absent)", "Pow(safe-rational differs)", "unbounded transcendentals(exp/ln/log/erf/tan/... ...)"],
            "paranote": "Sub is mapped as Add∘Neg (IEEE-exact equality incl. NaN); parity gate is exact on the covered set",
        },
        "throughput": {
            "tree_ms": tree_ms,
            "arena_ms": arena_ms,
            "arena_over_tree_ratio": ratio,
            "tree_us_per_candidate": per_candidate_us_tree,
            "arena_us_per_candidate": per_candidate_us_arena,
        },
        "gates": {"parity_mismatches": 0, "gate_decision": gate_decision},
    });
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/sre_tree_vs_arena_baseline.json");
    std::fs::write(&path, serde_json::to_string_pretty(&artifact).unwrap())
        .expect("write sre baseline artifact");
    println!("artifact written: {}", path.display());

    // Guard against silent degenerate workloads only — NOT a throughput or
    // coverage threshold: the measured truth (2026-10-22) is coverage 10.1%
    // (Div-heavy random trees) and arena/tree 0.63x, and the gate must keep
    // recording exactly that rather than being force-passed.
    assert!(coverage > 0.02, "semantics-identical conversion coverage collapsed: {coverage}");
    assert!(tree_ms < 60_000.0, "tree timing implausible: {tree_ms}");
    assert!(arena_ms < 60_000.0, "arena timing implausible: {arena_ms}");
}
fn chrono_like_now() -> String {
    // Deterministic-enough stamp without adding a dependency: seconds since
    // epoch, to be read next to the artifact's seed.
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
