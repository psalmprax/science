//! RIME mechanism graph: typed DAG with executable discrete-time semantics.
//!
//! Spec: docs/RIME_V01_MASTER_SPEC.md §4. Nodes form an arena; evaluation is
//! synchronous over input series with NaN-poisoning. Canonical serialization
//! feeds SHA-256 graph ids (deterministic from structure).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A node in the mechanism arena. Operands are arena indices; `Lag` carries delay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Node {
    Input(usize),
    Const(f64),
    Add(usize, usize),
    Mul(usize, usize),
    Neg(usize),
    Tanh(usize),
    Sign(usize),
    Abs(usize),
    Lag(usize, usize),
}

/// Frozen lag set (spec §4).
pub const LAG_SET: [usize; 5] = [1, 2, 5, 10, 20];

/// A mechanism graph: arena + designated output node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub output: usize,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new(), output: 0 }
    }

    pub fn push(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    /// Canonical string form (structure only — deterministic).
    pub fn canonical(&self) -> String {
        let mut s = String::new();
        for n in &self.nodes {
            let frag = match n {
                Node::Input(i) => format!("I{}", i),
                Node::Const(c) => format!("K{:.17e}", c),
                Node::Add(a, b) => format!("A{}:{}", a, b),
                Node::Mul(a, b) => format!("M{}:{}", a, b),
                Node::Neg(a) => format!("N{}", a),
                Node::Tanh(a) => format!("T{}", a),
                Node::Sign(a) => format!("S{}", a),
                Node::Abs(a) => format!("B{}", a),
                Node::Lag(a, k) => format!("L{}:{}", a, k),
            };
            s.push_str(&frag);
            s.push(';');
        }
        s.push_str(&format!("O{}", self.output));
        s
    }

    /// SHA-256 hex id of the canonical form.
    pub fn id(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.canonical().as_bytes());
        h.finalize().iter().map(|b| format!("{:02x}", b)).collect::<String>()
    }

    fn eval_node(&self, idx: usize, t: usize, cols: &[Vec<f64>], memo: &mut Vec<Option<f64>>) -> f64 {
        if let Some(v) = memo[idx] {
            return v;
        }
        let v = match &self.nodes[idx] {
            Node::Input(i) => *cols.get(*i).and_then(|c| c.get(t)).unwrap_or(&f64::NAN),
            Node::Const(c) => *c,
            Node::Add(a, b) => self.eval_node(*a, t, cols, memo) + self.eval_node(*b, t, cols, memo),
            Node::Mul(a, b) => self.eval_node(*a, t, cols, memo) * self.eval_node(*b, t, cols, memo),
            Node::Neg(a) => -self.eval_node(*a, t, cols, memo),
            Node::Tanh(a) => self.eval_node(*a, t, cols, memo).tanh(),
            Node::Sign(a) => {
                let x = self.eval_node(*a, t, cols, memo);
                if x.is_nan() { f64::NAN } else if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
            }
            Node::Abs(a) => self.eval_node(*a, t, cols, memo).abs(),
            Node::Lag(a, k) => {
                if t >= *k {
                    self.eval_node(*a, t - k, cols, memo)
                } else {
                    f64::NAN
                }
            }
        };
        memo[idx] = Some(v);
        v
    }

    /// Evaluate the output over all timesteps. NaN-poisoning is structural:
    /// any NaN input propagates (IEEE arithmetic), warmup rows stay NaN.
    pub fn evaluate(&self, cols: &[Vec<f64>]) -> Vec<f64> {
        let n = cols.first().map(|c| c.len()).unwrap_or(0);
        (0..n)
            .map(|t| {
                let mut memo = vec![None; self.nodes.len()];
                self.eval_node(self.output, t, cols, &mut memo)
            })
            .collect()
    }

    pub fn edge_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| match n {
                Node::Input(_) | Node::Const(_) => 0,
                Node::Neg(_) | Node::Tanh(_) | Node::Sign(_) | Node::Abs(_) | Node::Lag(_, _) => 1,
                Node::Add(_, _) | Node::Mul(_, _) => 2,
            })
            .sum()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_inputs() -> (Graph, Vec<Vec<f64>>) {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let y = g.push(Node::Input(1));
        let s = g.push(Node::Add(x, y));
        g.output = s;
        (g, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
    }

    #[test]
    fn eval_add_purity() {
        let (g, cols) = two_inputs();
        assert_eq!(g.evaluate(&cols), vec![5.0, 7.0, 9.0]);
        assert_eq!(g.evaluate(&cols), vec![5.0, 7.0, 9.0]); // deterministic
    }

    #[test]
    fn lag_warmup_nan() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let l = g.push(Node::Lag(x, 2));
        g.output = l;
        let out = g.evaluate(&[vec![1.0, 2.0, 3.0, 4.0]]);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(&out[2..], &[1.0, 2.0]);
    }

    #[test]
    fn nan_poisoning() {
        let (mut g, mut cols) = two_inputs();
        cols[0][1] = f64::NAN;
        let t = g.push(Node::Tanh(0));
        g.output = t;
        let out = g.evaluate(&cols);
        assert!(out[1].is_nan());
        assert!(out[0].is_finite() && out[2].is_finite());
    }

    #[test]
    fn canonical_id_stable() {
        let (g, _) = two_inputs();
        let (h, _) = two_inputs();
        assert_eq!(g.canonical(), h.canonical());
        assert_eq!(g.id(), h.id());
        assert_eq!(g.id().len(), 64);
    }

    #[test]
    fn tanh_sign_abs_shapes() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = g.push(Node::Mul(x, x)); // x^2 >= 0
        let out = g.evaluate(&[vec![-2.0, 3.0]]);
        assert_eq!(out, vec![4.0, 9.0]);
    }
}
