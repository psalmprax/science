//! RIME structural rewrites: the frozen five (spec §5).
//!
//! Each op is a pure function `&Graph -> Graph`. No other rewrites exist in
//! v0.1; adding one requires a new RIME version.

use serde::{Deserialize, Serialize};
use super::graph::{Graph, Node, LAG_SET};

/// The five frozen rewrite operations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewriteOp {
    AddLagEdge { src: usize, k: usize },
    AddInteraction { a: usize, b: usize },
    AddState { decay_num: usize, decay_den: usize },
    WrapNonlinearity { src: usize },
    PruneEdge { node: usize },
}

impl RewriteOp {
    pub fn name(&self) -> &'static str {
        match self {
            RewriteOp::AddLagEdge { .. } => "AddLagEdge",
            RewriteOp::AddInteraction { .. } => "AddInteraction",
            RewriteOp::AddState { .. } => "AddState",
            RewriteOp::WrapNonlinearity { .. } => "WrapNonlinearity",
            RewriteOp::PruneEdge { .. } => "PruneEdge",
        }
    }
}

/// Apply a rewrite; returns `None` when indices are out of range or the lag
/// is outside the frozen set. Ops that push a new expression node
/// (AddLagEdge, AddInteraction, WrapNonlinearity) make it the new output —
/// the candidate under test IS the new expression. AddState likewise.
/// PruneEdge edits in place and preserves the output index.
pub fn apply(g: &Graph, op: &RewriteOp) -> Option<Graph> {
    let n = g.nodes.len();
    let valid = |i: &usize| *i < n;
    let mut h = g.clone();
    match op {
        RewriteOp::AddLagEdge { src, k } => {
            if !valid(src) || !LAG_SET.contains(k) {
                return None;
            }
            h.output = h.push(Node::Lag(*src, *k));
        }
        RewriteOp::AddInteraction { a, b } => {
            if !valid(a) || !valid(b) {
                return None;
            }
            h.output = h.push(Node::Mul(*a, *b));
        }
        RewriteOp::AddState { decay_num, decay_den } => {
            if *decay_den == 0 || *decay_num > *decay_den {
                return None;
            }
            // Persistent memory cell: state = d*state_prev + input.
            // Represented structurally as decay constant × output feedback
            // plus fresh input edge; evaluator treats Const + Mul + Add.
            let d = (*decay_num as f64) / (*decay_den as f64);
            let dc = h.push(Node::Const(d));
            let fb = h.push(Node::Lag(h.output, 1));
            let mem = h.push(Node::Mul(dc, fb));
            let inp = h.push(Node::Neg(h.output));
            let cell = h.push(Node::Add(mem, inp));
            h.output = cell;
        }
        RewriteOp::WrapNonlinearity { src } => {
            if !valid(src) {
                return None;
            }
            h.output = h.push(Node::Tanh(*src));
        }
        RewriteOp::PruneEdge { node } => {
            if !valid(node) {
                return None;
            }
            // Pruning replaces the node with Const(0.0) in place: downstream
            // indices stay valid, so the output index is preserved.
            h.nodes[*node] = Node::Const(0.0);
        }
    }
    Some(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Graph {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let y = g.push(Node::Input(1));
        g.output = g.push(Node::Add(x, y));
        g
    }

    #[test]
    fn all_five_apply() {
        let g = base();
        assert!(apply(&g, &RewriteOp::AddLagEdge { src: 0, k: 5 }).is_some());
        assert!(apply(&g, &RewriteOp::AddInteraction { a: 0, b: 1 }).is_some());
        assert!(apply(&g, &RewriteOp::AddState { decay_num: 9, decay_den: 10 }).is_some());
        assert!(apply(&g, &RewriteOp::WrapNonlinearity { src: 2 }).is_some());
        assert!(apply(&g, &RewriteOp::PruneEdge { node: 0 }).is_some());
    }

    #[test]
    fn invalid_inputs_rejected() {
        let g = base();
        assert!(apply(&g, &RewriteOp::AddLagEdge { src: 99, k: 5 }).is_none());
        assert!(apply(&g, &RewriteOp::AddLagEdge { src: 0, k: 7 }).is_none()); // k not frozen
        assert!(apply(&g, &RewriteOp::AddInteraction { a: 0, b: 50 }).is_none());
        assert!(apply(&g, &RewriteOp::AddState { decay_num: 3, decay_den: 0 }).is_none());
        assert!(apply(&g, &RewriteOp::PruneEdge { node: 77 }).is_none());
    }

    #[test]
    fn output_retargeted_to_new_expression() {
        let g = base();
        let h = apply(&g, &RewriteOp::AddInteraction { a: 0, b: 1 }).unwrap();
        assert_ne!(h.output, g.output);
        let l = apply(&g, &RewriteOp::AddLagEdge { src: 0, k: 5 }).unwrap();
        assert_ne!(l.output, g.output);
        let w = apply(&g, &RewriteOp::WrapNonlinearity { src: 2 }).unwrap();
        assert_ne!(w.output, g.output);
        let p = apply(&g, &RewriteOp::PruneEdge { node: 0 }).unwrap();
        assert_eq!(p.output, g.output);
    }

    #[test]
    fn prune_kills_contribution() {
        let g = base();
        let p = apply(&g, &RewriteOp::PruneEdge { node: 0 }).unwrap();
        let out = p.evaluate(&[vec![5.0], vec![7.0]]);
        assert_eq!(out, vec![7.0]); // x pruned to 0, y passes through
    }

    #[test]
    fn op_names_cover_frozen_set() {
        let ops = [
            RewriteOp::AddLagEdge { src: 0, k: 1 },
            RewriteOp::AddInteraction { a: 0, b: 0 },
            RewriteOp::AddState { decay_num: 1, decay_den: 2 },
            RewriteOp::WrapNonlinearity { src: 0 },
            RewriteOp::PruneEdge { node: 0 },
        ];
        let names: Vec<_> = ops.iter().map(|o| o.name()).collect();
        assert_eq!(names, vec!["AddLagEdge", "AddInteraction", "AddState", "WrapNonlinearity", "PruneEdge"]);
    }
}
