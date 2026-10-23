//! Systematic Small-Expression Enumeration & Shared-Computation DAG (RIME-AD).
//!
//! Implements:
//!   1. Complexity-ordered enumeration of small canonical expression DAGs.
//!   2. Memoized, hash-consed subexpression evaluation (PSE-style shared computation).
//!   3. RED-style residual-guided subexpression targeting: isolates r = y - f(x).
//!
//! `include_exp` admits `exp()` nodes. The evaluator clamps the argument to
//! [-50, 50], so overflow degrades to saturation rather than non-finite
//! blowup; callers keep it off unless the problem family warrants it.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Op {
    Var(usize),
    Const(i32),
    Add,
    Sub,
    Mul,
    Div,
    Square,
    Inv,
    Cos,
    Sin,
    Exp,
    Floor,
    Neg,
    Log,
    Sqrt,
    Abs,
    Pi,
    Sign,
    Min,
    Max,
    Cbrt,
    Erf,
    BesselJ0,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Asin,
    Acos,
    Atan,
    Pow,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DagNode {
    pub op: Op,
    pub children: Vec<usize>, // indices into the shared DAG node pool
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedDag {
    pub nodes: Vec<DagNode>,
    #[serde(skip)]
    node_to_idx: HashMap<DagNode, usize>,
}

impl SharedDag {
    pub fn new() -> SharedDag {
        SharedDag {
            nodes: Vec::new(),
            node_to_idx: HashMap::new(),
        }
    }

    /// Add a node with hash-consing (returns existing index if already present)
    pub fn intern(&mut self, node: DagNode) -> usize {
        if let Some(&idx) = self.node_to_idx.get(&node) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(node.clone());
        self.node_to_idx.insert(node, idx);
        idx
    }

    /// Evaluate all nodes in the DAG up to node `root` on input columns `vars`.
    /// Reuses already evaluated nodes in `memo`.
    pub fn eval_node(
        &self,
        root: usize,
        vars: &[Vec<f64>],
        memo: &mut HashMap<usize, Vec<f64>>,
    ) -> Vec<f64> {
        if let Some(cached) = memo.get(&root) {
            return cached.clone();
        }

        let n = vars[0].len();
        let node = &self.nodes[root];

        let res = match node.op {
            Op::Var(v_idx) => vars[v_idx].clone(),
            Op::Const(c) => {
                vec![c as f64; n]
            }
            Op::Add => {
                let a = self.eval_node(node.children[0], vars, memo);
                let b = self.eval_node(node.children[1], vars, memo);
                a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
            }
            Op::Sub => {
                let a = self.eval_node(node.children[0], vars, memo);
                let b = self.eval_node(node.children[1], vars, memo);
                a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
            }
            Op::Mul => {
                let a = self.eval_node(node.children[0], vars, memo);
                let b = self.eval_node(node.children[1], vars, memo);
                a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
            }
            Op::Div => {
                let a = self.eval_node(node.children[0], vars, memo);
                let b = self.eval_node(node.children[1], vars, memo);
                a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| if y.abs() > 1e-12 { x / y } else { x / 1e-12 })
                    .collect()
            }
            Op::Square => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x * x).collect()
            }
            Op::Inv => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter()
                    .map(|&x| {
                        if x.abs() > 1e-12 {
                            1.0 / x
                        } else {
                            1.0 / 1e-12
                        }
                    })
                    .collect()
            }
            Op::Cos => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.cos()).collect()
            }
            Op::Sin => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.sin()).collect()
            }
            Op::Exp => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.clamp(-50.0, 50.0).exp()).collect()
            }
            Op::Floor => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.floor()).collect()
            }
            // Domain-guarded misc ops: out-of-domain inputs yield NaN, and
            // callers skip non-finite bases, so no spurious fits can arise.
            Op::Neg => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| -x).collect()
            }
            Op::Log => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter()
                    .map(|&x| if x > 0.0 { x.ln() } else { f64::NAN })
                    .collect()
            }
            Op::Sqrt => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter()
                    .map(|&x| if x >= 0.0 { x.sqrt() } else { f64::NAN })
                    .collect()
            }
            Op::Abs => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.abs()).collect()
            }
            Op::Pi => {
                vec![std::f64::consts::PI; vars[0].len()]
            }
            // Sign mirrors parser Func::Sign exactly (keeps -0.0 and NaN).
            Op::Sign => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter()
                    .map(|&x| {
                        if x > 0.0 {
                            1.0
                        } else if x < 0.0 {
                            -1.0
                        } else {
                            x
                        }
                    })
                    .collect()
            }
            Op::Min => {
                let a = self.eval_node(node.children[0], vars, memo);
                let b = self.eval_node(node.children[1], vars, memo);
                a.iter().zip(b.iter()).map(|(&x, &y)| x.min(y)).collect()
            }
            Op::Max => {
                let a = self.eval_node(node.children[0], vars, memo);
                let b = self.eval_node(node.children[1], vars, memo);
                a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect()
            }
            Op::Cbrt => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.cbrt()).collect()
            }
            Op::Erf => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| crate::numeric::erf_approx(x)).collect()
            }
            Op::BesselJ0 => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| crate::numeric::bessel_j0(x)).collect()
            }
            Op::Tan => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.tan()).collect()
            }
            Op::Sinh => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.sinh()).collect()
            }
            Op::Cosh => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.cosh()).collect()
            }
            Op::Tanh => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.tanh()).collect()
            }
            Op::Asin => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter()
                    .map(|&x| if x.abs() <= 1.0 { x.asin() } else { f64::NAN })
                    .collect()
            }
            Op::Acos => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter()
                    .map(|&x| if x.abs() <= 1.0 { x.acos() } else { f64::NAN })
                    .collect()
            }
            Op::Atan => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x.atan()).collect()
            }
            Op::Pow => {
                let a = self.eval_node(node.children[0], vars, memo);
                let b = self.eval_node(node.children[1], vars, memo);
                a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| {
                        if x < 0.0 && y.fract() != 0.0 {
                            f64::NAN
                        } else {
                            x.powf(y)
                        }
                    })
                    .collect()
            }
        };

        memo.insert(root, res.clone());
        res
    }

    /// Convert node to readable formula string
    pub fn describe_node(&self, idx: usize, var_names: &[String]) -> String {
        let node = &self.nodes[idx];
        match node.op {
            Op::Var(v) => var_names.get(v).cloned().unwrap_or_else(|| format!("v{v}")),
            Op::Const(c) => format!("{c}"),
            Op::Add => format!(
                "({} + {})",
                self.describe_node(node.children[0], var_names),
                self.describe_node(node.children[1], var_names)
            ),
            Op::Sub => format!(
                "({} - {})",
                self.describe_node(node.children[0], var_names),
                self.describe_node(node.children[1], var_names)
            ),
            Op::Mul => format!(
                "({} * {})",
                self.describe_node(node.children[0], var_names),
                self.describe_node(node.children[1], var_names)
            ),
            Op::Div => format!(
                "({} / {})",
                self.describe_node(node.children[0], var_names),
                self.describe_node(node.children[1], var_names)
            ),
            Op::Square => format!("({}^2)", self.describe_node(node.children[0], var_names)),
            Op::Inv => format!("(1 / {})", self.describe_node(node.children[0], var_names)),
            Op::Cos => format!("cos({})", self.describe_node(node.children[0], var_names)),
            Op::Sin => format!("sin({})", self.describe_node(node.children[0], var_names)),
            Op::Exp => format!("exp({})", self.describe_node(node.children[0], var_names)),
            Op::Floor => format!("floor({})", self.describe_node(node.children[0], var_names)),
            Op::Neg => format!("(-{})", self.describe_node(node.children[0], var_names)),
            Op::Log => format!("log({})", self.describe_node(node.children[0], var_names)),
            Op::Sqrt => format!("sqrt({})", self.describe_node(node.children[0], var_names)),
            Op::Abs => format!("abs({})", self.describe_node(node.children[0], var_names)),
            Op::Pi => "pi".to_string(),
            Op::Sign => format!("sign({})", self.describe_node(node.children[0], var_names)),
            Op::Min => format!(
                "min({}, {})",
                self.describe_node(node.children[0], var_names),
                self.describe_node(node.children[1], var_names)
            ),
            Op::Max => format!(
                "max({}, {})",
                self.describe_node(node.children[0], var_names),
                self.describe_node(node.children[1], var_names)
            ),
            Op::Cbrt => format!("cbrt({})", self.describe_node(node.children[0], var_names)),
            Op::Erf => format!("erf({})", self.describe_node(node.children[0], var_names)),
            Op::BesselJ0 => format!(
                "besselj({})",
                self.describe_node(node.children[0], var_names)
            ),
            Op::Tan => format!("tan({})", self.describe_node(node.children[0], var_names)),
            Op::Sinh => format!("sinh({})", self.describe_node(node.children[0], var_names)),
            Op::Cosh => format!("cosh({})", self.describe_node(node.children[0], var_names)),
            Op::Tanh => format!("tanh({})", self.describe_node(node.children[0], var_names)),
            Op::Asin => format!("asin({})", self.describe_node(node.children[0], var_names)),
            Op::Acos => format!("acos({})", self.describe_node(node.children[0], var_names)),
            Op::Atan => format!("atan({})", self.describe_node(node.children[0], var_names)),
            Op::Pow => format!(
                "({} ** {})",
                self.describe_node(node.children[0], var_names),
                self.describe_node(node.children[1], var_names)
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumerationCandidate {
    pub node_idx: usize,
    pub formula: String,
    pub slope: f64,
    pub intercept: f64,
    pub r2: f64,
    pub mse: f64,
}

/// Systematically enumerate small canonical expressions up to given depth.
///
/// Returns the shared DAG plus one candidate pool per depth level, so later
/// stages (e.g. rational pairing) can bound combinations by depth.
pub fn enumerate_canonical_dag(
    n_vars: usize,
    include_trig: bool,
    include_floor: bool,
    include_exp: bool,
    include_extended: bool,
    include_advanced: bool,
    max_depth: usize,
) -> (SharedDag, Vec<Vec<usize>>) {
    let mut dag = SharedDag::new();
    let mut depth_pools: Vec<Vec<usize>> = Vec::new();

    // Depth 0: variables and small constants (pi lives here too so it can
    // appear inside nonlinear positions like sin(pi*x)).
    let mut d0 = Vec::new();
    for v in 0..n_vars {
        d0.push(dag.intern(DagNode {
            op: Op::Var(v),
            children: vec![],
        }));
    }
    d0.push(dag.intern(DagNode {
        op: Op::Const(1),
        children: vec![],
    }));
    d0.push(dag.intern(DagNode {
        op: Op::Const(2),
        children: vec![],
    }));
    if include_extended {
        d0.push(dag.intern(DagNode {
            op: Op::Pi,
            children: vec![],
        }));
    }

    depth_pools.push(d0);

    // Iteratively expand depth up to max_depth
    for d in 1..=max_depth {
        let mut curr_pool = Vec::new();
        let prev_all: Vec<usize> = depth_pools.iter().flatten().cloned().collect();

        // 1. Unary ops on previous pool
        for &node_idx in &depth_pools[d - 1] {
            curr_pool.push(dag.intern(DagNode {
                op: Op::Square,
                children: vec![node_idx],
            }));
            curr_pool.push(dag.intern(DagNode {
                op: Op::Inv,
                children: vec![node_idx],
            }));

            if include_trig {
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Cos,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Sin,
                    children: vec![node_idx],
                }));
            }
            if include_floor {
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Floor,
                    children: vec![node_idx],
                }));
            }
            if include_exp {
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Exp,
                    children: vec![node_idx],
                }));
            }
            if include_extended {
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Neg,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Log,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Sqrt,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Abs,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Sign,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Cbrt,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Erf,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::BesselJ0,
                    children: vec![node_idx],
                }));
            }
            if include_advanced {
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Tan,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Sinh,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Cosh,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Tanh,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Asin,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Acos,
                    children: vec![node_idx],
                }));
                curr_pool.push(dag.intern(DagNode {
                    op: Op::Atan,
                    children: vec![node_idx],
                }));
            }
        }

        // 2. Binary ops between (d-1) and all previous
        if d <= 2 {
            for &idx1 in &depth_pools[d - 1] {
                for &idx2 in &prev_all {
                    if idx1 <= idx2 {
                        curr_pool.push(dag.intern(DagNode {
                            op: Op::Add,
                            children: vec![idx1, idx2],
                        }));
                        curr_pool.push(dag.intern(DagNode {
                            op: Op::Mul,
                            children: vec![idx1, idx2],
                        }));
                        // Min/Max are commutative like Add/Mul; keep them
                        // behind the extended flag with the same dedup rule.
                        if include_extended {
                            curr_pool.push(dag.intern(DagNode {
                                op: Op::Min,
                                children: vec![idx1, idx2],
                            }));
                            curr_pool.push(dag.intern(DagNode {
                                op: Op::Max,
                                children: vec![idx1, idx2],
                            }));
                        }
                        // General Pow is non-commutative; no dedup shortcut.
                        if include_advanced {
                            curr_pool.push(dag.intern(DagNode {
                                op: Op::Pow,
                                children: vec![idx1, idx2],
                            }));
                            curr_pool.push(dag.intern(DagNode {
                                op: Op::Pow,
                                children: vec![idx2, idx1],
                            }));
                        }
                    }
                    curr_pool.push(dag.intern(DagNode {
                        op: Op::Sub,
                        children: vec![idx1, idx2],
                    }));
                    curr_pool.push(dag.intern(DagNode {
                        op: Op::Div,
                        children: vec![idx1, idx2],
                    }));
                }
            }
        }

        depth_pools.push(curr_pool);
    }

    (dag, depth_pools)
}

/// Does the subtree rooted at `idx` contain operator `op`?
fn subtree_contains(dag: &SharedDag, idx: usize, op: Op, seen: &mut HashMap<usize, bool>) -> bool {
    if let Some(&v) = seen.get(&idx) {
        return v;
    }
    let node = &dag.nodes[idx];
    let mut v = node.op == op;
    if !v {
        for &c in &node.children {
            if subtree_contains(dag, c, op, seen) {
                v = true;
                break;
            }
        }
    }
    seen.insert(idx, v);
    v
}

/// Rational-pairing stage (Padé-style): form Div(num, den) with numerators
/// from the depth-bounded basis pool (excluding Div nodes, which would build
/// redundant rationals-of-rationals) and denominators from the
/// transcendental-denominator shortlist (basis nodes whose subtree contains
/// `exp`). Each pair is OLS-readout scored with the DAG-wide shared memo, so
/// every unique node is still evaluated once; only the top `top_k` pairs
/// are materialised into formula strings. The cap is an explicit caller
/// budget, not a validity verdict.
///
/// A second family covers saturating rational laws, `y ≈ slope*x/(c+x) +
/// intercept` with a **fitted** half-saturation `c > 0` (Langmuir
/// `q_max*K*p/(1+K*p)`, Michaelis–Menten). That interior constant is not a
/// DAG leaf (`Op::Const` is integer-only), so the existing exp-denominator
/// shortlist cannot express it; here `c` is searched on a log grid and
/// refined by golden-section. Both families compete for the same `top_k`.
pub fn search_rational_pairs(
    vars: &[Vec<f64>],
    target: &[f64],
    var_names: &[String],
    include_trig: bool,
    include_floor: bool,
    _include_exp: bool,
    _include_advanced: bool,
    max_depth: usize,
    top_k: usize,
) -> Vec<EnumerationCandidate> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone, Copy)]
    struct Scored {
        r2: f64,
        mse: f64,
        slope: f64,
        intercept: f64,
        root: usize,
    }
    impl PartialEq for Scored {
        fn eq(&self, other: &Self) -> bool {
            self.r2 == other.r2
        }
    }
    impl Eq for Scored {}
    impl PartialOrd for Scored {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.r2.partial_cmp(&other.r2)
        }
    }
    impl Ord for Scored {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    let n = target.len();
    let mean_y = target.iter().sum::<f64>() / (n as f64);
    let var_y = target
        .iter()
        .map(|v| (v - mean_y) * (v - mean_y))
        .sum::<f64>()
        / (n as f64);

    let (mut dag, pools) = enumerate_canonical_dag(
        vars.len(),
        include_trig,
        include_floor,
        true,  // rational pairing only runs with exp enabled
        false, // extended misc ops stay out of the denominator shortlist
        false, // advanced ops stay out of the denominator shortlist
        max_depth,
    );
    let mut seen = HashMap::new();
    let mut numerators = Vec::new();
    let mut denoms = Vec::new();
    for idx in pools.iter().flatten().cloned() {
        if dag.nodes[idx].op == Op::Div {
            continue;
        }
        numerators.push(idx);
        if subtree_contains(&dag, idx, Op::Exp, &mut seen) {
            denoms.push(idx);
        }
    }

    let mut memo = HashMap::new();
    let mut heap: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();
    let keep = top_k.max(1);
    for &num in &numerators {
        for &den in &denoms {
            if num == den {
                continue;
            }
            let root = dag.intern(DagNode {
                op: Op::Div,
                children: vec![num, den],
            });
            let out = dag.eval_node(root, vars, &mut memo);
            if out.iter().any(|v| !v.is_finite()) {
                continue;
            }
            // Spike guard: a denominator brushing zero produces huge ratios
            // that OLS can spuriously fit. Rational candidates must stay
            // bounded on the data.
            if out.iter().any(|v| v.abs() > 1e6) {
                continue;
            }
            let mean_b = out.iter().sum::<f64>() / (n as f64);
            let mut cov = 0.0f64;
            let mut vb = 0.0f64;
            for i in 0..n {
                let db = out[i] - mean_b;
                cov += db * (target[i] - mean_y);
                vb += db * db;
            }
            if vb <= 1e-12 {
                continue;
            }
            let slope = cov / vb;
            let intercept = mean_y - slope * mean_b;
            let mut sse = 0.0f64;
            for i in 0..n {
                let e = target[i] - (slope * out[i] + intercept);
                sse += e * e;
            }
            let mse = sse / (n as f64);
            let r2 = if var_y > 1e-15 {
                1.0 - (mse / var_y)
            } else {
                1.0
            };
            if heap.len() < keep || r2 > heap.peek().map(|s| s.0.r2).unwrap_or(f64::NEG_INFINITY) {
                heap.push(std::cmp::Reverse(Scored {
                    r2,
                    mse,
                    slope,
                    intercept,
                    root,
                }));
                if heap.len() > keep {
                    heap.pop();
                }
            }
        }
    }

    // ── Saturating rational family (Langmuir / Michaelis-Menten) ──────────
    // For a fixed c the readout in (slope, intercept) is linear, so each trial
    // costs one O(n) pass; the search is 1-D and stays inside the same heap.
    let mut formula_overrides: HashMap<usize, String> = HashMap::new();

    fn saturating_fit(
        x: &[f64],
        y: &[f64],
        c: f64,
        mean_y: f64,
    ) -> Option<(f64, f64, f64)> {
        if !(c.is_finite() && c > 0.0) {
            return None;
        }
        let n = x.len();
        let mut mean_b = 0.0f64;
        for i in 0..n {
            let den = c + x[i];
            if !x[i].is_finite() || !y[i].is_finite() || !den.is_finite() || den.abs() <= 1e-12
            {
                return None;
            }
            let b = x[i] / den;
            if !b.is_finite() || b.abs() > 1e6 {
                return None;
            }
            mean_b += b;
        }
        if n < 6 {
            return None;
        }
        mean_b /= n as f64;
        let mut cov = 0.0f64;
        let mut vb = 0.0f64;
        for i in 0..n {
            let db = x[i] / (c + x[i]) - mean_b;
            cov += db * (y[i] - mean_y);
            vb += db * db;
        }
        if vb <= 1e-15 {
            return None;
        }
        let slope = cov / vb;
        let intercept = mean_y - slope * mean_b;
        let mut sse = 0.0f64;
        for i in 0..n {
            let e = y[i] - (slope * (x[i] / (c + x[i])) + intercept);
            sse += e * e;
        }
        if sse.is_finite() {
            Some((slope, intercept, sse))
        } else {
            None
        }
    }

    for (vi, x) in vars.iter().enumerate() {
        if x.len() != n {
            continue;
        }
        let mut abs: Vec<f64> = x.iter().filter(|v| v.is_finite()).map(|v| v.abs()).collect();
        if abs.len() < 6 {
            continue;
        }
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let scale = abs[abs.len() / 2];
        if !(scale.is_finite() && scale > 0.0) {
            continue;
        }
        let lo = (scale * 1e-3).ln();
        let hi = (scale * 1e3).ln();
        let mut best: Option<(f64, f64)> = None; // (sse, c)
        for k in 0..48 {
            let c = (lo + (hi - lo) * (k as f64) / 47.0).exp();
            if let Some((_, _, sse)) = saturating_fit(x, target, c, mean_y) {
                if best.map(|(s, _)| sse < s).unwrap_or(true) {
                    best = Some((sse, c));
                }
            }
        }
        let Some((_, c0)) = best else { continue };

        // Golden-section on log(c): refine inside one grid cell's neighbourhood.
        let phi = 0.618_033_988_749_894_9_f64;
        let sse_at = |c: f64| {
            saturating_fit(x, target, c, mean_y)
                .map(|t| t.2)
                .unwrap_or(f64::INFINITY)
        };
        let mut a = (c0 / 3.0).ln();
        let mut b = (c0 * 3.0).ln();
        let mut x1 = b - phi * (b - a);
        let mut x2 = a + phi * (b - a);
        let mut f1 = sse_at(x1.exp());
        let mut f2 = sse_at(x2.exp());
        for _ in 0..32 {
            if f1 < f2 {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - phi * (b - a);
                f1 = sse_at(x1.exp());
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + phi * (b - a);
                f2 = sse_at(x2.exp());
            }
        }
        let c_best = (0.5 * (a + b)).exp();
        let Some((slope, intercept, sse)) = saturating_fit(x, target, c_best, mean_y) else {
            continue;
        };
        let mse = sse / (n as f64);
        let r2 = if var_y > 1e-15 {
            1.0 - (mse / var_y)
        } else {
            1.0
        };
        if !r2.is_finite() {
            continue;
        }
        // The fitted `c` has no DAG leaf (`Op::Const` is integer-only), so the
        // handle points at the variable and the formula carries the fitted
        // half-saturation constant.
        let var_node = dag.intern(DagNode {
            op: Op::Var(vi),
            children: vec![],
        });
        let name = var_names.get(vi).cloned().unwrap_or_else(|| format!("v{vi}"));
        formula_overrides.insert(
            var_node,
            format!("{slope:.6} * ({name} / ({c_best:.6} + {name})) + {intercept:.6}"),
        );
        if heap.len() < keep || r2 > heap.peek().map(|s| s.0.r2).unwrap_or(f64::NEG_INFINITY) {
            heap.push(std::cmp::Reverse(Scored {
                r2,
                mse,
                slope,
                intercept,
                root: var_node,
            }));
            if heap.len() > keep {
                heap.pop();
            }
        }
    }

    let mut top: Vec<Scored> = heap.into_iter().map(|s| s.0).collect();
    top.sort_by(|a, b| b.r2.partial_cmp(&a.r2).unwrap_or(Ordering::Equal));
    top.into_iter()
        .map(|s| EnumerationCandidate {
            node_idx: s.root,
            formula: formula_overrides.get(&s.root).cloned().unwrap_or_else(|| {
                format!(
                    "{:.4} * ({}) + {:.4}",
                    s.slope,
                    dag.describe_node(s.root, var_names),
                    s.intercept
                )
            }),
            slope: s.slope,
            intercept: s.intercept,
            r2: s.r2,
            mse: s.mse,
        })
        .collect()
}

/// Enumerate, batch-evaluate with shared memoization, and fit linear readout (slope * basis + intercept)
pub fn search_enumerated_shared(
    vars: &[Vec<f64>],
    target: &[f64],
    var_names: &[String],
    include_trig: bool,
    include_floor: bool,
    include_exp: bool,
    include_extended: bool,
    include_advanced: bool,
    max_depth: usize,
) -> Vec<EnumerationCandidate> {
    let n = target.len();
    let mean_y = target.iter().sum::<f64>() / (n as f64);
    let var_y = target
        .iter()
        .map(|v| (v - mean_y) * (v - mean_y))
        .sum::<f64>()
        / (n as f64);

    let (dag, pools) = enumerate_canonical_dag(
        vars.len(),
        include_trig,
        include_floor,
        include_exp,
        include_extended,
        include_advanced,
        max_depth,
    );
    let candidates: Vec<usize> = pools.iter().flatten().cloned().collect();
    let mut memo = HashMap::new();
    let mut results = Vec::new();

    for &node_idx in &candidates {
        let basis = dag.eval_node(node_idx, vars, &mut memo);

        // Fit a * basis + b = target via OLS
        let mean_b = basis.iter().sum::<f64>() / (n as f64);
        let mut cov_by = 0.0f64;
        let mut var_b = 0.0f64;

        for i in 0..n {
            let db = basis[i] - mean_b;
            cov_by += db * (target[i] - mean_y);
            var_b += db * db;
        }

        if var_b > 1e-12 {
            let slope = cov_by / var_b;
            let intercept = mean_y - slope * mean_b;

            let mut sum_sq_err = 0.0f64;
            for i in 0..n {
                let pred = slope * basis[i] + intercept;
                let err = target[i] - pred;
                sum_sq_err += err * err;
            }
            let mse = sum_sq_err / (n as f64);
            let r2 = if var_y > 1e-15 {
                1.0 - (mse / var_y)
            } else {
                1.0
            };

            let formula = format!(
                "{:.4} * {} + {:.4}",
                slope,
                dag.describe_node(node_idx, var_names),
                intercept
            );
            results.push(EnumerationCandidate {
                node_idx,
                formula,
                slope,
                intercept,
                r2,
                mse,
            });
        }
    }

    results.sort_by(|a, b| b.r2.partial_cmp(&a.r2).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_dag_hash_consing() {
        let mut dag = SharedDag::new();
        let v0 = dag.intern(DagNode {
            op: Op::Var(0),
            children: vec![],
        });
        let v0_dup = dag.intern(DagNode {
            op: Op::Var(0),
            children: vec![],
        });
        assert_eq!(v0, v0_dup);
        assert_eq!(dag.nodes.len(), 1);
    }

    #[test]
    fn test_enumerate_finds_harmonic_oscillation() {
        // target: y = 0.5 - 0.5 * cos(2*t) = (1 - cos(2t))/2
        let n = 50;
        let t: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64)).collect();
        let y: Vec<f64> = t.iter().map(|&tv| 0.5 - 0.5 * (2.0 * tv).cos()).collect();

        let res = search_enumerated_shared(
            &[t],
            &y,
            &["t".to_string()],
            true, // include_trig
            false,
            false, // include_exp
            false, // include_extended
            false, // include_advanced
            2,
        );

        assert!(!res.is_empty());
        let best = &res[0];
        assert!(
            best.r2 > 0.999,
            "Best R2: {}, formula: {}",
            best.r2,
            best.formula
        );
    }

    #[test]
    fn test_enumerate_with_exp_finds_exponential_growth() {
        // y = exp(v): Exp(v) is a depth-1 node once include_exp is set.
        let n = 40;
        let v: Vec<f64> = (0..n)
            .map(|i| 0.1 + 2.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let y: Vec<f64> = v.iter().map(|&vv| vv.exp()).collect();

        let res = search_enumerated_shared(
            &[v],
            &y,
            &["v".to_string()],
            false,
            false,
            true,  // include_exp
            false, // include_extended
            false, // include_advanced
            2,
        );

        assert!(!res.is_empty());
        let best = &res[0];
        assert!(
            best.r2 > 0.999,
            "Best R2: {}, formula: {}",
            best.r2,
            best.formula
        );
        assert!(best.formula.contains("exp"), "formula: {}", best.formula);
    }

    #[test]
    fn test_enumerate_extended_ops_recover_log_sqrt_abs_neg() {
        // Each extended op is a depth-1 node once include_extended is set.
        let n = 40;
        let v: Vec<f64> = (0..n)
            .map(|i| 0.5 + 3.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (v.iter().map(|&x| x.ln()).collect(), "log"),
            (v.iter().map(|&x| x.sqrt()).collect(), "sqrt"),
            (v.iter().map(|&x| (x - 2.0).abs()).collect(), "abs"),
            (v.iter().map(|&x| -x).collect(), "(-"),
        ];
        for (y, token) in cases {
            let arg: Vec<f64> = if token == "abs" {
                v.iter().map(|&x| x - 2.0).collect()
            } else {
                v.clone()
            };
            let res = search_enumerated_shared(
                &[arg],
                &y,
                &["v".to_string()],
                false,
                false,
                false,
                true,  // include_extended
                false, // include_advanced
                2,
            );
            assert!(!res.is_empty());
            let hit = res
                .iter()
                .any(|c| c.r2 > 0.999 && c.formula.contains(token));
            assert!(hit, "token {token} not recovered; best: {}", res[0].formula);
        }
    }

    #[test]
    fn test_rational_search_reaches_langmuir_saturating_law() {
        // Langmuir q = q_max*K*p/(1+K*p) with q_max=40, K=1.6 (c = 1/K = 0.625).
        // The exp-denominator shortlist cannot express a fitted c; the
        // saturating family must reach it inside the same top-k budget.
        let n = 200usize;
        let x: Vec<f64> = (0..n)
            .map(|i| 0.1 + 2.4 * (i as f64) / ((n - 1) as f64))
            .collect();
        let y: Vec<f64> = x.iter().map(|&p| 40.0 * 1.6 * p / (1.0 + 1.6 * p)).collect();
        let res = search_rational_pairs(
            &[x],
            &y,
            &["p".to_string()],
            false,
            false,
            false,
            false,
            2,
            8,
        );
        let hit = res
            .iter()
            .find(|c| c.formula.contains(" / (") && !c.formula.contains("exp(") && c.r2 > 0.9999)
            .unwrap_or_else(|| {
                panic!(
                    "no saturating candidate: {:?}",
                    res.iter().map(|c| (&c.formula, c.r2)).collect::<Vec<_>>()
                )
            });
        assert!(
            (hit.slope - 40.0).abs() / 40.0 < 0.01,
            "recovered q_max={} (formula {})",
            hit.slope,
            hit.formula
        );
        assert!(
            hit.intercept.abs() < 0.05,
            "recovered intercept={} (formula {})",
            hit.intercept,
            hit.formula
        );
    }

    #[test]
    fn test_extended_ops_are_gated_off_by_default() {
        // Without include_extended, log/sqrt/abs/neg nodes must not appear.
        let n = 20;
        let v: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64)).collect();
        let y: Vec<f64> = v.iter().map(|&x| x.ln()).collect();
        let res = search_enumerated_shared(
            &[v],
            &y,
            &["v".to_string()],
            false,
            false,
            false,
            false, // include_extended off
            false, // include_advanced off
            1,
        );
        for c in &res {
            assert!(!c.formula.contains("log("), "leaked: {}", c.formula);
            assert!(!c.formula.contains("sqrt("), "leaked: {}", c.formula);
            assert!(!c.formula.contains("abs("), "leaked: {}", c.formula);
        }
    }
}
