//! Systematic Small-Expression Enumeration & Shared-Computation DAG (RIME-AD).
//!
//! Implements:
//!   1. Complexity-ordered enumeration of small canonical expression DAGs.
//!   2. Memoized, hash-consed subexpression evaluation (PSE-style shared computation).
//!   3. RED-style residual-guided subexpression targeting: isolates r = y - f(x).

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

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
            Op::Var(v_idx) => {
                vars[v_idx].clone()
            }
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
                a.iter().zip(b.iter()).map(|(&x, &y)| {
                    if y.abs() > 1e-12 { x / y } else { x / 1e-12 }
                }).collect()
            }
            Op::Square => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| x * x).collect()
            }
            Op::Inv => {
                let a = self.eval_node(node.children[0], vars, memo);
                a.iter().map(|&x| if x.abs() > 1e-12 { 1.0 / x } else { 1.0 / 1e-12 }).collect()
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
            Op::Add => format!("({} + {})", self.describe_node(node.children[0], var_names), self.describe_node(node.children[1], var_names)),
            Op::Sub => format!("({} - {})", self.describe_node(node.children[0], var_names), self.describe_node(node.children[1], var_names)),
            Op::Mul => format!("({} * {})", self.describe_node(node.children[0], var_names), self.describe_node(node.children[1], var_names)),
            Op::Div => format!("({} / {})", self.describe_node(node.children[0], var_names), self.describe_node(node.children[1], var_names)),
            Op::Square => format!("({}^2)", self.describe_node(node.children[0], var_names)),
            Op::Inv => format!("(1 / {})", self.describe_node(node.children[0], var_names)),
            Op::Cos => format!("cos({})", self.describe_node(node.children[0], var_names)),
            Op::Sin => format!("sin({})", self.describe_node(node.children[0], var_names)),
            Op::Exp => format!("exp({})", self.describe_node(node.children[0], var_names)),
            Op::Floor => format!("floor({})", self.describe_node(node.children[0], var_names)),
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
pub fn enumerate_canonical_dag(
    n_vars: usize,
    include_trig: bool,
    include_floor: bool,
    max_depth: usize,
) -> (SharedDag, Vec<usize>) {
    let mut dag = SharedDag::new();
    let mut depth_pools: Vec<Vec<usize>> = Vec::new();

    // Depth 0: variables and small constants
    let mut d0 = Vec::new();
    for v in 0..n_vars {
        d0.push(dag.intern(DagNode { op: Op::Var(v), children: vec![] }));
    }
    d0.push(dag.intern(DagNode { op: Op::Const(1), children: vec![] }));
    d0.push(dag.intern(DagNode { op: Op::Const(2), children: vec![] }));

    depth_pools.push(d0);

    // Iteratively expand depth up to max_depth
    for d in 1..=max_depth {
        let mut curr_pool = Vec::new();
        let prev_all: Vec<usize> = depth_pools.iter().flatten().cloned().collect();

        // 1. Unary ops on previous pool
        for &node_idx in &depth_pools[d - 1] {
            curr_pool.push(dag.intern(DagNode { op: Op::Square, children: vec![node_idx] }));
            curr_pool.push(dag.intern(DagNode { op: Op::Inv, children: vec![node_idx] }));

            if include_trig {
                curr_pool.push(dag.intern(DagNode { op: Op::Cos, children: vec![node_idx] }));
                curr_pool.push(dag.intern(DagNode { op: Op::Sin, children: vec![node_idx] }));
            }
            if include_floor {
                curr_pool.push(dag.intern(DagNode { op: Op::Floor, children: vec![node_idx] }));
            }
        }

        // 2. Binary ops between (d-1) and all previous
        if d <= 2 {
            for &idx1 in &depth_pools[d - 1] {
                for &idx2 in &prev_all {
                    if idx1 <= idx2 {
                        curr_pool.push(dag.intern(DagNode { op: Op::Add, children: vec![idx1, idx2] }));
                        curr_pool.push(dag.intern(DagNode { op: Op::Mul, children: vec![idx1, idx2] }));
                    }
                    curr_pool.push(dag.intern(DagNode { op: Op::Sub, children: vec![idx1, idx2] }));
                    curr_pool.push(dag.intern(DagNode { op: Op::Div, children: vec![idx1, idx2] }));
                }
            }
        }

        depth_pools.push(curr_pool);
    }

    let all_candidates: Vec<usize> = depth_pools.into_iter().flatten().collect();
    (dag, all_candidates)
}

/// Enumerate, batch-evaluate with shared memoization, and fit linear readout (slope * basis + intercept)
pub fn search_enumerated_shared(
    vars: &[Vec<f64>],
    target: &[f64],
    var_names: &[String],
    include_trig: bool,
    include_floor: bool,
    max_depth: usize,
) -> Vec<EnumerationCandidate> {
    let n = target.len();
    let mean_y = target.iter().sum::<f64>() / (n as f64);
    let var_y = target.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>() / (n as f64);

    let (dag, candidates) = enumerate_canonical_dag(vars.len(), include_trig, include_floor, max_depth);
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
            let r2 = if var_y > 1e-15 { 1.0 - (mse / var_y) } else { 1.0 };

            let formula = format!("{:.4} * {} + {:.4}", slope, dag.describe_node(node_idx, var_names), intercept);
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
        let v0 = dag.intern(DagNode { op: Op::Var(0), children: vec![] });
        let v0_dup = dag.intern(DagNode { op: Op::Var(0), children: vec![] });
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
            true,  // include_trig
            false,
            2,
        );

        assert!(!res.is_empty());
        let best = &res[0];
        assert!(best.r2 > 0.999, "Best R2: {}, formula: {}", best.r2, best.formula);
    }
}
