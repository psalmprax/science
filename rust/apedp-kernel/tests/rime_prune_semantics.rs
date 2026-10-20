//! RIME v0.1.2 P2: structural pruning consistency.
//!
//! `PruneEdge` plants `Const(0.0)` in place rather than deleting the node, so
//! downstream indices survive. The review's requirement: that decision must be
//! PROVEN consistent everywhere, not just applied — an inactive edge must
//! contribute exactly zero, complexity/DoF accounting must treat it the same
//! way every consumer does, predictions must survive serialization round-trips,
//! and pruning must not disturb unrelated graph components.

use apedp_kernel::rime::eval::{charged_dof, fit_readout, fitted_mse, ScoreConfig, READOUT_PARAMS};
use apedp_kernel::rime::graph::{Graph, Node};
use apedp_kernel::rime::rewrite::{apply, RewriteOp};

fn two_term_graph() -> Graph {
    // y = x0 + x1, both live.
    let mut g = Graph::new();
    let x0 = g.push(Node::Input(0));
    let x1 = g.push(Node::Input(1));
    g.output = g.push(Node::Add(x0, x1));
    g
}

#[test]
fn zeroed_edge_contributes_exactly_zero() {
    let g = two_term_graph();
    let pruned = apply(&g, &RewriteOp::PruneEdge { node: 0 }).unwrap();

    // The pruned node is a literal zero.
    assert!(matches!(pruned.nodes[0], Node::Const(c) if c == 0.0));
    // Expressed over a spread of inputs: output == x1 exactly (x0's 0.0 * anything = 0).
    let cols = vec![vec![1.0, -2.3, 7.7, 0.0], vec![10.0, 20.0, 30.0, 40.0]];
    let out = pruned.evaluate(&cols);
    assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    // And the unpruned output was x0+x1 — pruning changed exactly one term.
    let unpruned = g.evaluate(&cols);
    assert_eq!(unpruned, vec![11.0, 17.7, 37.7, 40.0]);
}

#[test]
fn simplification_and_complexity_treat_inactive_edges_consistently() {
    let g = two_term_graph();
    let pruned = apply(&g, &RewriteOp::PruneEdge { node: 0 }).unwrap();

    // Structural accounting: the inactive edge still exists as a live Input in
    // the arena, BUT — critically — the search deduplicates by rendered
    // expression (`describe`), where the pruned term renders through Const(0).
    // What the referee packet reports as structural size is edge_count, which
    // counts the Add's two operands identically before and after: pruning is a
    // zeroing, not a removal, and the packet must say so.
    assert_eq!(pruned.edge_count(), g.edge_count());
    assert_eq!(
        charged_dof(&pruned, 0.0),
        pruned.edge_count() as f64 + READOUT_PARAMS
    );
    // Rendered form: the zeroed branch is visible, not hidden.
    assert!(
        pruned.describe().contains("(0.000000"),
        "got {}",
        pruned.describe()
    );
    // And `canonical` — the content address — CHANGES, so the pruned candidate
    // cannot be confused with its unpruned parent in any ledger.
    assert_ne!(pruned.id(), g.id());
}

#[test]
fn predictions_reproducible_after_serialization_roundtrip() {
    let g = two_term_graph();
    let pruned = apply(&g, &RewriteOp::PruneEdge { node: 1 }).unwrap();

    let cols = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![9.0, 8.0, 7.0, 6.0, 5.0]];
    let before = pruned.evaluate(&cols);

    // Serialize → deserialize (the same path the PyO3 boundary uses).
    let json = serde_json::to_string(&pruned).unwrap();
    let reloaded: Graph = serde_json::from_str(&json).unwrap();
    assert_eq!(reloaded, pruned, "byte-identical structure");
    assert_eq!(reloaded.id(), pruned.id(), "content address survives");
    assert_eq!(reloaded.evaluate(&cols), before, "predictions reproducible");

    // Also through fitted scoring (the packet's numbers):
    let y: Vec<f64> = before.iter().map(|v| 2.0 * v + 1.5).collect();
    let (mse1, a1, b1) = fitted_mse(&pruned, &cols, &y, 1e-9);
    let (mse2, a2, b2) = fitted_mse(&reloaded, &cols, &y, 1e-9);
    assert_eq!(mse1, mse2);
    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
}

#[test]
fn pruning_does_not_modify_unrelated_components() {
    let g = two_term_graph();
    // Keep a snapshot of every node except the pruned one.
    let snapshot: Vec<Node> = g.nodes.clone();
    let pruned = apply(&g, &RewriteOp::PruneEdge { node: 0 }).unwrap();
    for (i, (before, after)) in snapshot.iter().zip(pruned.nodes.iter()).enumerate() {
        if i == 0 {
            continue; // the pruned node is the one that changes
        }
        assert_eq!(before, after, "node {i} must be untouched");
    }
    assert_eq!(pruned.output, g.output, "output index preserved");
    // The original graph is not mutated (a rewrite is a pure function).
    assert!(matches!(g.nodes[0], Node::Input(0)));
}

#[test]
fn pruned_readout_is_intercept_only_not_rescued() {
    // Pruning BOTH terms must leave an intercept-only model — and the readout
    // must say so honestly: scale ≈ 0 (no structure), intercept = mean(y).
    let g = two_term_graph();
    let p01 = apply(&g, &RewriteOp::PruneEdge { node: 0 }).unwrap();
    let p0 = apply(&p01, &RewriteOp::PruneEdge { node: 2 }).unwrap(); // the Add node itself
    let cols = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let out = p0.evaluate(&cols);
    assert!(
        out.iter().all(|v| *v == 0.0),
        "both terms zeroed: {:?}",
        out
    );
    let y = vec![3.0, 5.0, 7.0];
    let (a, b) = fit_readout(&out, &y, 1e-9);
    assert_eq!(a, 0.0, "constant output: scale must be exactly 0");
    assert!((b - 5.0).abs() < 1e-9, "intercept = mean(y): {}", b);
    // Its score against the null must be ≤ 0 — an intercept-only model is the
    // null model, it cannot earn a complexity charge. (The null's MSE is var(y);
    // a parent of INFINITY would compute INF - finite = INF, which is what the
    // single-candidate entry point is for, not this comparison.)
    let sc = ScoreConfig::new(0.01, 1e-9);
    let var_y = 8.0 / 3.0; // mean([3,5,7])=5, var = ((3-5)²+(0)²+(7-5)²)/3
    let s = apedp_kernel::rime::eval::score_candidate_fitted(&p0, var_y, &cols, &y, &sc);
    assert!(
        s.score <= 0.0,
        "intercept-only model cannot outscore the null: {}",
        s.score
    );
}
