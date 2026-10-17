//! RIME mechanism graph: typed DAG with executable discrete-time semantics.
//!
//! Spec: docs/RIME_V01_MASTER_SPEC.md §4. Nodes form an arena; evaluation is
//! synchronous over input series with NaN-poisoning. Canonical serialization
//! feeds SHA-256 graph ids (deterministic from structure).
//!
//! ## v0.1.1 correctness fixes (Stage-1 audit)
//!
//! * Evaluation is **series-at-a-time**, not row-at-a-time. The previous
//!   implementation memoised one value per *node*, so a node referenced both
//!   directly and through `Lag` shared a single cache slot and the lagged
//!   read returned the current row. Any difference/momentum form
//!   (`x[t] - x[t-1]`) silently evaluated to zero. Values are now arrays over
//!   the whole series, computed in arena (topological) order.
//! * `Node::State` is the persistent memory cell required by spec §4/§5, with
//!   the documented recurrence `h[t] = decay * h[t-1] + u[t]`, `h[-1] = init`.
//!   `AddState` used to be encoded as `d*output[t-1] - output[t]`, which is
//!   not a recursion and carries the wrong sign for the input term.

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
    /// Pole-protected rational division: `safediv(a, b) = a / (1.0 + |b|)`.
    /// Denominator is strictly >= 1.0, eliminating poles and division-by-zero.
    SafeDiv(usize, usize),
    /// STEM primitives (Option 1):
    /// Bounded sine: `sin(a)`
    Sin(usize),
    /// Bounded cosine: `cos(a)`
    Cos(usize),
    /// Clamped exponential: `exp(clamp(a, -60.0, 60.0))` to prevent float overflow
    SafeExp(usize),
    /// Non-pole natural logarithm: `ln(1.0 + |a|)`
    SafeLog(usize),
    /// Discrete first difference: `diff(a)[t] = a[t] - a[t-1]`, with NaN at warmup t=0
    Diff(usize),
    /// Persistent state cell (spec §4 `State`).
    ///
    /// `h[t] = decay * h[t-1] + u[t]` where `u` is the series of `src` and
    /// `h[-1] = init` (so `h[0] = decay*init + u[0]`, which is `u[0]` for the
    /// default `init = 0`). NaN in `u` poisons the state forward, exactly like
    /// every other node: warmup rows of a lagged input are NaN and are skipped
    /// (counted) by the scorer, never imputed.
    State { src: usize, decay: f64, init: f64 },
    /// Rational power operator: `x^(p_num / p_den)`.
    /// Supports fractional powers (e.g. 1/3, 2/3, 1/2) and integer powers (2, 3, -1).
    /// Evaluated safely: sign-preserving for odd denominators, pole-safe for negative powers.
    Power { src: usize, p_num: i32, p_den: u32 },
}

/// Frozen lag set (spec §4).
pub const LAG_SET: [usize; 5] = [1, 2, 5, 10, 20];

/// Why a graph cannot be evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// `output` does not name a node.
    OutputOutOfRange { output: usize, nodes: usize },
    /// A node references itself or a later node: the arena must be topological.
    NotTopological { node: usize, operand: usize },
    /// `Lag(k)` with `k` outside the frozen set.
    LagNotFrozen { node: usize, k: usize },
    /// A non-finite numeric literal.
    NonFiniteConstant { node: usize },
    /// Empty arena.
    Empty,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Empty => write!(f, "graph has no nodes"),
            GraphError::OutputOutOfRange { output, nodes } => {
                write!(f, "output index {} out of range ({} nodes)", output, nodes)
            }
            GraphError::NotTopological { node, operand } => write!(
                f,
                "node {} references operand {}: arena order must be topological",
                node, operand
            ),
            GraphError::LagNotFrozen { node, k } => {
                write!(f, "node {} uses lag {} outside the frozen set {:?}", node, k, LAG_SET)
            }
            GraphError::NonFiniteConstant { node } => {
                write!(f, "node {} carries a non-finite constant", node)
            }
        }
    }
}

impl std::error::Error for GraphError {}

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
                Node::SafeDiv(a, b) => format!("D{}:{}", a, b),
                Node::Sin(a) => format!("SI{}", a),
                Node::Cos(a) => format!("CO{}", a),
                Node::SafeExp(a) => format!("EX{}", a),
                Node::SafeLog(a) => format!("LG{}", a),
                Node::Diff(a) => format!("DF{}", a),
                Node::Power { src, p_num, p_den } => {
                    format!("P{}:{}:{}", src, p_num, p_den)
                }
                Node::State { src, decay, init } => {
                    format!("X{}:{:.17e}:{:.17e}", src, decay, init)
                }
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

    /// Human-readable closed form of the output expression (for referee
    /// packets). Node sharing is expanded, so the string can be exponential in
    /// the number of nodes; RIME graphs are small by construction.
    pub fn describe(&self) -> String {
        fn rec(g: &Graph, i: usize, memo: &mut Vec<Option<String>>) -> String {
            if let Some(s) = memo.get(i).and_then(|m| m.clone()) {
                return s;
            }
            let s = match g.nodes.get(i) {
                None => "<?>".to_string(),
                Some(Node::Input(v)) => format!("v{}", v),
                Some(Node::Const(c)) => format!("{:.6}", c),
                Some(Node::Add(a, b)) => format!("({} + {})", rec(g, *a, memo), rec(g, *b, memo)),
                Some(Node::Mul(a, b)) => format!("({} * {})", rec(g, *a, memo), rec(g, *b, memo)),
                Some(Node::SafeDiv(a, b)) => {
                    format!("safediv({}, {})", rec(g, *a, memo), rec(g, *b, memo))
                }
                Some(Node::Power { src, p_num, p_den }) => {
                    if *p_den == 1 {
                        format!("({}^{})", rec(g, *src, memo), p_num)
                    } else {
                        format!("({}^({}/{}))", rec(g, *src, memo), p_num, p_den)
                    }
                }
                Some(Node::Sin(a)) => format!("sin({})", rec(g, *a, memo)),
                Some(Node::Cos(a)) => format!("cos({})", rec(g, *a, memo)),
                Some(Node::SafeExp(a)) => format!("exp({})", rec(g, *a, memo)),
                Some(Node::SafeLog(a)) => format!("safelog({})", rec(g, *a, memo)),
                Some(Node::Diff(a)) => format!("diff({})", rec(g, *a, memo)),
                Some(Node::Neg(a)) => format!("(-{})", rec(g, *a, memo)),
                Some(Node::Tanh(a)) => format!("tanh({})", rec(g, *a, memo)),
                Some(Node::Sign(a)) => format!("sign({})", rec(g, *a, memo)),
                Some(Node::Abs(a)) => format!("abs({})", rec(g, *a, memo)),
                Some(Node::Lag(a, k)) => format!("{}[t-{}]", rec(g, *a, memo), k),
                Some(Node::State { src, decay, init }) => {
                    format!("state_{:.4}({}; init={:.4})", decay, rec(g, *src, memo), init)
                }
            };
            if i < memo.len() {
                memo[i] = Some(s.clone());
            }
            s
        }
        let mut memo = vec![None; self.nodes.len()];
        rec(self, self.output, &mut memo)
    }

    /// True when any node reads time: a `Lag` delay or a `State` memory cell.
    ///
    /// Static search mode (config `"static": true`) uses this to exclude every
    /// temporal form from the seed catalogue, so that on order-structured data
    /// (e.g. a log-spaced static law) no candidate can win by fitting grid
    /// order — a lag would be *unrepresentable*, not merely gated.
    pub fn is_temporal(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n, Node::Lag(_, _) | Node::State { .. } | Node::Diff(_)))
    }

    /// Structural validation: output in range, arena order topological, lags
    /// frozen, literals finite. Evaluation refuses non-topological graphs
    /// instead of silently reading a half-built value.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::Empty);
        }
        if self.output >= self.nodes.len() {
            return Err(GraphError::OutputOutOfRange { output: self.output, nodes: self.nodes.len() });
        }
        for (i, n) in self.nodes.iter().enumerate() {
            let check = |operand: usize| -> Result<(), GraphError> {
                if operand >= i {
                    Err(GraphError::NotTopological { node: i, operand })
                } else {
                    Ok(())
                }
            };
            match n {
                Node::Input(_) => {}
                Node::Const(c) => {
                    if !c.is_finite() {
                        return Err(GraphError::NonFiniteConstant { node: i });
                    }
                }
                Node::Add(a, b) | Node::Mul(a, b) | Node::SafeDiv(a, b) => {
                    check(*a)?;
                    check(*b)?;
                }
                Node::Neg(a)
                | Node::Tanh(a)
                | Node::Sign(a)
                | Node::Abs(a)
                | Node::Sin(a)
                | Node::Cos(a)
                | Node::SafeExp(a)
                | Node::SafeLog(a)
                | Node::Diff(a) => check(*a)?,
                Node::Power { src, p_den, .. } => {
                    check(*src)?;
                    if *p_den == 0 || *p_den > 10 {
                        return Err(GraphError::NonFiniteConstant { node: i });
                    }
                }
                Node::Lag(a, k) => {
                    check(*a)?;
                    if !LAG_SET.contains(k) {
                        return Err(GraphError::LagNotFrozen { node: i, k: *k });
                    }
                }
                Node::State { src, decay, init } => {
                    check(*src)?;
                    if !decay.is_finite() || !init.is_finite() {
                        return Err(GraphError::NonFiniteConstant { node: i });
                    }
                }
            }
        }
        Ok(())
    }

    /// Operand indices of a node.
    fn operands(n: &Node) -> Vec<usize> {
        match n {
            Node::Input(_) | Node::Const(_) => Vec::new(),
            Node::Add(a, b) | Node::Mul(a, b) | Node::SafeDiv(a, b) => vec![*a, *b],
            Node::Neg(a)
            | Node::Tanh(a)
            | Node::Sign(a)
            | Node::Abs(a)
            | Node::Lag(a, _)
            | Node::Sin(a)
            | Node::Cos(a)
            | Node::SafeExp(a)
            | Node::SafeLog(a)
            | Node::Diff(a)
            | Node::Power { src: a, .. } => vec![*a],
            Node::State { src, .. } => vec![*src],
        }
    }

    /// Input column indices read by the output expression, sorted and
    /// deduplicated (arena walk from the output, so unreachable nodes never
    /// contribute). Instrumentation only: reachability, not importance —
    /// a reachable-yet-canceling node still counts.
    pub fn inputs_used(&self) -> Vec<usize> {
        let mut seen_node = vec![false; self.nodes.len()];
        let mut stack = vec![self.output];
        let mut inputs = std::collections::BTreeSet::new();
        while let Some(i) = stack.pop() {
            if i >= self.nodes.len() || seen_node[i] {
                continue;
            }
            seen_node[i] = true;
            match &self.nodes[i] {
                Node::Input(k) => {
                    inputs.insert(*k);
                }
                n => {
                    stack.extend(Self::operands(n));
                }
            }
        }
        inputs.into_iter().collect()
    }

    /// Evaluate every node over the whole series, in arena (topological) order.
    ///
    /// One array per *live* node: a node's series is released as soon as the
    /// last consumer has read it, and nodes unreachable from the output are
    /// never computed. Memory is therefore O(depth × n), not O(nodes × n), and
    /// time is O(edges × n) instead of the old O(nodes × n) recursive walk that
    /// allocated a fresh memo per row.
    pub fn evaluate_checked<V: crate::data::DataCol>(&self, cols: &[V]) -> Result<Vec<f64>, GraphError> {
        self.validate()?;
        let n = cols.iter().map(|c| c.row_len()).max().unwrap_or(0);

        // Remaining consumer count per node (the output counts as a consumer).
        let mut uses = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            for op in Self::operands(node) {
                uses[op] += 1;
            }
        }
        // Pin the output node: its series must survive to the end even if a
        // later node also consumes it.
        uses[self.output] += 1;
        let mut vals: Vec<Option<Vec<f64>>> = (0..self.nodes.len()).map(|_| None).collect();

        for i in 0..self.nodes.len() {
            if uses[i] == 0 && i != self.output {
                continue; // unreachable from the output: never computed
            }
            let v: Vec<f64> = match &self.nodes[i] {
                Node::Input(k) => (0..n)
                    .map(|t| cols.get(*k).and_then(|c| c.row_opt(t)).unwrap_or(f64::NAN))
                    .collect(),
                Node::Const(c) => vec![*c; n],
                Node::Add(a, b) => {
                    let (x, y) = (vals[*a].as_ref().unwrap(), vals[*b].as_ref().unwrap());
                    (0..n).map(|t| x[t] + y[t]).collect()
                }
                Node::Mul(a, b) => {
                    let (x, y) = (vals[*a].as_ref().unwrap(), vals[*b].as_ref().unwrap());
                    (0..n).map(|t| x[t] * y[t]).collect()
                }
                Node::SafeDiv(a, b) => {
                    let (x, y) = (vals[*a].as_ref().unwrap(), vals[*b].as_ref().unwrap());
                    (0..n).map(|t| x[t] / (1.0 + y[t].abs())).collect()
                }
                Node::Sin(a) => vals[*a].as_ref().unwrap().iter().map(|v| v.sin()).collect(),
                Node::Cos(a) => vals[*a].as_ref().unwrap().iter().map(|v| v.cos()).collect(),
                Node::SafeExp(a) => vals[*a]
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|v| v.clamp(-60.0, 60.0).exp())
                    .collect(),
                Node::SafeLog(a) => vals[*a]
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|v| (1.0 + v.abs()).ln())
                    .collect(),
                Node::Diff(a) => {
                    let x = vals[*a].as_ref().unwrap();
                    (0..n)
                        .map(|t| if t >= 1 { x[t] - x[t - 1] } else { f64::NAN })
                        .collect()
                }
                Node::Power { src, p_num, p_den } => {
                    let x = vals[*src].as_ref().unwrap();
                    let exp = (*p_num as f64) / (*p_den as f64);
                    (0..n)
                        .map(|t| {
                            let v = x[t];
                            if v.is_nan() {
                                f64::NAN
                            } else if *p_num > 0 {
                                if *p_den % 2 != 0 && *p_num % 2 != 0 {
                                    v.abs().powf(exp) * v.signum()
                                } else {
                                    v.abs().powf(exp)
                                }
                            } else {
                                let denom = v.abs();
                                if denom < 1e-12 {
                                    0.0
                                } else if *p_den % 2 != 0 && *p_num % 2 != 0 {
                                    v.signum() / denom.powf(-exp)
                                } else {
                                    1.0 / denom.powf(-exp)
                                }
                            }
                        })
                        .collect()
                }
                Node::Neg(a) => vals[*a].as_ref().unwrap().iter().map(|v| -v).collect(),
                Node::Tanh(a) => vals[*a].as_ref().unwrap().iter().map(|v| v.tanh()).collect(),
                Node::Sign(a) => vals[*a]
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        if v.is_nan() {
                            f64::NAN
                        } else if *v > 0.0 {
                            1.0
                        } else if *v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                Node::Abs(a) => vals[*a].as_ref().unwrap().iter().map(|v| v.abs()).collect(),
                Node::Lag(a, k) => {
                    let x = vals[*a].as_ref().unwrap();
                    (0..n).map(|t| if t >= *k { x[t - k] } else { f64::NAN }).collect()
                }
                Node::State { src, decay, init } => {
                    let u = vals[*src].as_ref().unwrap();
                    let mut h = vec![f64::NAN; n];
                    let mut prev = *init;
                    for t in 0..n {
                        prev = decay * prev + u[t];
                        h[t] = prev;
                    }
                    h
                }
            };
            // Release consumed operands.
            for op in Self::operands(&self.nodes[i]) {
                uses[op] -= 1;
                if uses[op] == 0 {
                    vals[op] = None;
                }
            }
            vals[i] = Some(v);
        }
        Ok(vals[self.output].take().expect("output series computed"))
    }

    /// Evaluate the output over all timesteps. NaN-poisoning is structural:
    /// any NaN input propagates (IEEE arithmetic), warmup rows stay NaN.
    ///
    /// A structurally invalid graph yields an all-NaN series (so it can never
    /// score) — use [`Graph::evaluate_checked`] to see the reason.
    pub fn evaluate<V: crate::data::DataCol>(&self, cols: &[V]) -> Vec<f64> {
        let n = cols.iter().map(|c| c.row_len()).max().unwrap_or(0);
        self.evaluate_checked(cols).unwrap_or_else(|_| vec![f64::NAN; n])
    }

    pub fn edge_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| match n {
                Node::Input(_) | Node::Const(_) => 0,
                Node::Neg(_)
                | Node::Tanh(_)
                | Node::Sign(_)
                | Node::Abs(_)
                | Node::Lag(_, _)
                | Node::Sin(_)
                | Node::Cos(_)
                | Node::SafeExp(_)
                | Node::SafeLog(_)
                | Node::Diff(_)
                | Node::Power { .. }
                | Node::State { .. } => 1,
                Node::Add(_, _) | Node::Mul(_, _) | Node::SafeDiv(_, _) => 2,
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

    #[test]
    fn repeated_operand_is_not_memoised_across_time() {
        // Defect 1 regression: x is read at t (directly) and at t-1 (via Lag).
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let xl = g.push(Node::Lag(x, 1));
        g.output = g.push(Node::Mul(x, xl)); // x[t] * x[t-1]

        let out = g.evaluate(&[vec![2.0, 3.0, 4.0, 5.0]]);
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &[6.0, 12.0, 20.0]);
    }

    #[test]
    fn state_recurrence_and_init() {
        let mut g = Graph::new();
        let u = g.push(Node::Input(0));
        g.output = g.push(Node::State { src: u, decay: 0.5, init: 0.0 });
        let out = g.evaluate(&[vec![1.0, 1.0, 1.0, 1.0]]);
        assert_eq!(out, vec![1.0, 1.5, 1.75, 1.875]);

        // A non-zero init is a decaying imprint of pre-sample history: the
        // prior state is itself carried through one decay step to t = 0.
        let mut h = Graph::new();
        let u = h.push(Node::Input(0));
        h.output = h.push(Node::State { src: u, decay: 0.5, init: 1.0 });
        assert_eq!(h.evaluate(&[vec![0.0, 0.0]]), vec![0.5, 0.25]);
    }

    #[test]
    fn unreachable_nodes_do_not_affect_output() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let junk = g.push(Node::Input(7)); // unreachable, undefined column
        g.output = x;
        assert_eq!(g.evaluate(&[vec![1.0, 2.0]]), vec![1.0, 2.0]);
        assert!(!g.nodes.is_empty() && junk == 1);
    }

    #[test]
    fn validate_rejects_forward_reference_and_bad_lag() {
        let g = Graph { nodes: vec![Node::Const(1.0), Node::Add(0, 1)], output: 1 };
        assert_eq!(g.validate(), Err(GraphError::NotTopological { node: 1, operand: 1 }));
        // Violating graph must not silently produce numbers.
        assert!(g.evaluate(&[vec![1.0]]).iter().all(|v| v.is_nan()));

        let mut h = Graph::new();
        let x = h.push(Node::Input(0));
        h.output = h.push(Node::Lag(x, 3));
        assert_eq!(h.validate(), Err(GraphError::LagNotFrozen { node: 1, k: 3 }));

        let k = Graph { nodes: vec![], output: 0 };
        assert_eq!(k.validate(), Err(GraphError::Empty));
    }

    #[test]
    fn describe_is_readable() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let xl = g.push(Node::Lag(x, 2));
        let nxl = g.push(Node::Neg(xl));
        g.output = g.push(Node::Add(x, nxl));
        let s = g.describe();
        assert!(s.contains("v0"), "{}", s);
        assert!(s.contains("[t-2]"), "{}", s);
    }

    #[test]
    fn state_edge_is_charged() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = g.push(Node::State { src: x, decay: 0.5, init: 0.0 });
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn rational_power_evaluation() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        // Test x^(2/3)
        g.output = g.push(Node::Power { src: x, p_num: 2, p_den: 3 });
        let out = g.evaluate(&[vec![1.0, 8.0, 27.0]]);
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - 4.0).abs() < 1e-10);
        assert!((out[2] - 9.0).abs() < 1e-10);

        // Test odd-root sign preservation: (-8)^(1/3) = -2
        let mut h = Graph::new();
        let y = h.push(Node::Input(0));
        h.output = h.push(Node::Power { src: y, p_num: 1, p_den: 3 });
        let out_h = h.evaluate(&[vec![-8.0, 8.0]]);
        assert!((out_h[0] - (-2.0)).abs() < 1e-10);
        assert!((out_h[1] - 2.0).abs() < 1e-10);

        // Test negative power: 8^(-1/3) = 0.5, 0^(-1/3) = 0.0 (safe pole)
        let mut k = Graph::new();
        let z = k.push(Node::Input(0));
        k.output = k.push(Node::Power { src: z, p_num: -1, p_den: 3 });
        let out_k = k.evaluate(&[vec![8.0, 0.0]]);
        assert!((out_k[0] - 0.5).abs() < 1e-10);
        assert_eq!(out_k[1], 0.0);
    }
}
