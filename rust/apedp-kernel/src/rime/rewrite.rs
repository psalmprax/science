//! RIME structural rewrites: the frozen five (spec §5).
//!
//! Each op is a pure function `&Graph -> Graph`. No other rewrites exist in
//! v0.1; adding one requires a new RIME version.

use serde::{Deserialize, Serialize};
use super::graph::{Graph, Node, LAG_SET};

/// The seven frozen rewrite operations (v0.1 core + two controlled expansions).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewriteOp {
    AddLagEdge { src: usize, k: usize },
    AddInteraction { a: usize, b: usize },
    AddState { decay_num: usize, decay_den: usize },
    WrapNonlinearity { src: usize },
    PruneEdge { node: usize },
    AbsNode { src: usize },
    SignNode { src: usize },
    AddSafeDiv { a: usize, b: usize },
    SinNode { src: usize },
    CosNode { src: usize },
    SafeExpNode { src: usize },
    SafeLogNode { src: usize },
    DiffNode { src: usize },
    AddPower { src: usize, p_num: i32, p_den: u32 },
}

/// Permitted rational power parameters: (numerator, denominator).
pub const PERMITTED_POWERS: [(i32, u32); 9] = [
    (1, 2),   // x^(1/2) = sqrt(|x|)
    (1, 3),   // x^(1/3) = cbrt(x)
    (2, 3),   // x^(2/3) = cbrt(x)^2
    (-1, 2),  // x^(-1/2) = 1 / sqrt(|x|)
    (-1, 3),  // x^(-1/3) = 1 / cbrt(|x|)
    (2, 1),   // x^2
    (3, 1),   // x^3
    (-1, 1),  // x^(-1) = 1 / x
    (-2, 1),  // x^(-2) = 1 / x^2
];

impl RewriteOp {
    pub fn name(&self) -> &'static str {
        match self {
            RewriteOp::AddLagEdge { .. } => "AddLagEdge",
            RewriteOp::AddInteraction { .. } => "AddInteraction",
            RewriteOp::AddState { .. } => "AddState",
            RewriteOp::WrapNonlinearity { .. } => "WrapNonlinearity",
            RewriteOp::PruneEdge { .. } => "PruneEdge",
            RewriteOp::AbsNode { .. } => "AbsNode",
            RewriteOp::SignNode { .. } => "SignNode",
            RewriteOp::AddSafeDiv { .. } => "AddSafeDiv",
            RewriteOp::SinNode { .. } => "SinNode",
            RewriteOp::CosNode { .. } => "CosNode",
            RewriteOp::SafeExpNode { .. } => "SafeExpNode",
            RewriteOp::SafeLogNode { .. } => "SafeLogNode",
            RewriteOp::DiffNode { .. } => "DiffNode",
            RewriteOp::AddPower { .. } => "AddPower",
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
            // Persistent memory cell (spec §4/§5): one State node whose
            // recurrence is h[t] = d*h[t-1] + u[t], with the current output as
            // its input `u` and h[-1] = init = 0 (a prior state is a parameter
            // the caller may set through a fit, not a structural choice).
            //
            // v0.1.1: this used to expand to `d*output[t-1] - output[t]`,
            // which is not a recursion (the input term has the wrong sign and
            // the feedback is read from the parent series, not from the state).
            let d = (*decay_num as f64) / (*decay_den as f64);
            h.output = h.push(Node::State { src: g.output, decay: d, init: 0.0 });
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
        RewriteOp::AbsNode { src } => {
            if *src >= n {
                return None;
            }
            h.push(Node::Abs(*src));
        }
        RewriteOp::SignNode { src } => {
            if *src >= n {
                return None;
            }
            h.push(Node::Sign(*src));
        }
        RewriteOp::AddSafeDiv { a, b } => {
            if !valid(a) || !valid(b) {
                return None;
            }
            h.output = h.push(Node::SafeDiv(*a, *b));
        }
        RewriteOp::SinNode { src } => {
            if !valid(src) {
                return None;
            }
            h.output = h.push(Node::Sin(*src));
        }
        RewriteOp::CosNode { src } => {
            if !valid(src) {
                return None;
            }
            h.output = h.push(Node::Cos(*src));
        }
        RewriteOp::SafeExpNode { src } => {
            if !valid(src) {
                return None;
            }
            h.output = h.push(Node::SafeExp(*src));
        }
        RewriteOp::SafeLogNode { src } => {
            if !valid(src) {
                return None;
            }
            h.output = h.push(Node::SafeLog(*src));
        }
        RewriteOp::DiffNode { src } => {
            if !valid(src) {
                return None;
            }
            h.output = h.push(Node::Diff(*src));
        }
        RewriteOp::AddPower { src, p_num, p_den } => {
            if !valid(src) || !PERMITTED_POWERS.contains(&(*p_num, *p_den)) {
                return None;
            }
            h.output = h.push(Node::Power { src: *src, p_num: *p_num, p_den: *p_den });
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
    fn add_state_is_a_recursion_not_a_sign_flip() {
        // Regression: AddState on an all-ones input with d = 0.5 previously
        // evaluated to [NaN, -0.5, -0.5, -0.5]. A true decaying memory gives
        // the geometric partial sums.
        let mut g = Graph::new();
        g.output = g.push(Node::Input(0));
        let s = apply(&g, &RewriteOp::AddState { decay_num: 1, decay_den: 2 }).unwrap();
        assert_eq!(s.evaluate(&[vec![1.0, 1.0, 1.0, 1.0]]), vec![1.0, 1.5, 1.75, 1.875]);
    }

    #[test]
    fn add_state_state_node_keeps_parent_as_input() {
        let g = base();
        let s = apply(&g, &RewriteOp::AddState { decay_num: 9, decay_den: 10 }).unwrap();
        match s.nodes[s.output] {
            Node::State { src, decay, init } => {
                assert_eq!(src, g.output);
                assert!((decay - 0.9).abs() < 1e-12);
                assert_eq!(init, 0.0);
            }
            ref other => panic!("AddState must push a State node, got {:?}", other),
        }
    }

    #[test]
    fn prune_kills_contribution() {
        let g = base();
        let p = apply(&g, &RewriteOp::PruneEdge { node: 0 }).unwrap();
        let out = p.evaluate(&[vec![5.0], vec![7.0]]);
        assert_eq!(out, vec![7.0]); // x pruned to 0, y passes through
    }

    #[test]
    fn safe_div_applies() {
        let g = base();
        let h = apply(&g, &RewriteOp::AddSafeDiv { a: 0, b: 1 }).unwrap();
        assert_ne!(h.output, g.output);
        let out = h.evaluate(&[vec![6.0], vec![2.0]]);
        // safediv(6.0, 2.0) = 6.0 / (1.0 + 2.0) = 2.0
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn add_power_applies() {
        let g = base();
        let h = apply(&g, &RewriteOp::AddPower { src: 0, p_num: 2, p_den: 3 }).unwrap();
        assert_ne!(h.output, g.output);
        let out = h.evaluate(&[vec![8.0], vec![1.0]]);
        assert!((out[0] - 4.0).abs() < 1e-10);

        // Invalid power rejected
        assert!(apply(&g, &RewriteOp::AddPower { src: 0, p_num: 5, p_den: 7 }).is_none());
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
