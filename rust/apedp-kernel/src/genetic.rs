//! High-performance native Genetic Algorithm engine for multi-million candidate symbolic discovery.
//!
//! Evaluates candidate equation trees in native Rust using Rayon multi-core parallelism,
//! exact affine variable projection, subtree crossover, mutation, and Pareto archiving.

use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use crate::parser::{Func, Node};

/// Fast, reproducible, seedable 64-bit XorShift PRNG (zero-allocation).
#[derive(Clone, Debug)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    #[inline(always)]
    pub fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        min + (self.next_u64() as usize % (max - min))
    }

    #[inline(always)]
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

/// Evaluation and tree manipulation extensions on `Node`.
pub trait NodeExt {
    fn eval_row(&self, vars: &[f64]) -> f64;
    fn depth(&self) -> usize;
    fn complexity(&self) -> usize;
    fn to_equation_string(&self, var_names: &[String]) -> String;
    fn collect_paths(&self, current: &[usize], acc: &mut Vec<Vec<usize>>);
    fn get_subtree(&self, path: &[usize]) -> Option<Node>;
    fn set_subtree(&self, path: &[usize], replacement: &Node) -> Node;
}

impl NodeExt for Node {
    #[inline]
    fn eval_row(&self, vars: &[f64]) -> f64 {
        match self {
            Node::Const(c) => *c,
            Node::Var(idx) => vars.get(*idx).copied().unwrap_or(f64::NAN),
            Node::Ident(_) => f64::NAN,
            Node::Add(l, r) => l.eval_row(vars) + r.eval_row(vars),
            Node::Sub(l, r) => l.eval_row(vars) - r.eval_row(vars),
            Node::Mul(l, r) => l.eval_row(vars) * r.eval_row(vars),
            Node::Min(l, r) => l.eval_row(vars).min(r.eval_row(vars)),
            Node::Max(l, r) => l.eval_row(vars).max(r.eval_row(vars)),
            Node::Div(l, r) => {
                let den = r.eval_row(vars);
                if den.abs() < 1e-15 {
                    f64::NAN
                } else {
                    l.eval_row(vars) / den
                }
            }
            Node::Pow(l, r) => {
                let b = l.eval_row(vars);
                let p = r.eval_row(vars);
                let res = b.powf(p);
                if res.is_finite() { res } else { f64::NAN }
            }
            Node::Neg(inner) => -inner.eval_row(vars),
            Node::Call(func, inner) => func.apply(inner.eval_row(vars)),
        }
    }

    fn depth(&self) -> usize {
        match self {
            Node::Const(_) | Node::Var(_) | Node::Ident(_) => 0,
            Node::Neg(inner) | Node::Call(_, inner) => 1 + inner.depth(),
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) | Node::Min(l, r) | Node::Max(l, r) => {
                1 + l.depth().max(r.depth())
            }
        }
    }

    fn complexity(&self) -> usize {
        match self {
            Node::Const(_) | Node::Var(_) | Node::Ident(_) => 1,
            Node::Neg(inner) => 1 + inner.complexity(),
            Node::Call(_, inner) => 2 + inner.complexity(),
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) | Node::Min(l, r) | Node::Max(l, r) => {
                1 + l.complexity() + r.complexity()
            }
        }
    }

    fn to_equation_string(&self, var_names: &[String]) -> String {
        match self {
            Node::Const(c) => {
                if (c - std::f64::consts::PI).abs() < 1e-12 {
                    "pi".to_string()
                } else if (c - std::f64::consts::E).abs() < 1e-12 {
                    "e".to_string()
                } else if (c.round() - c).abs() < 1e-6 {
                    format!("{:.1}", c)
                } else {
                    format!("{:.4}", c)
                }
            }
            Node::Ident(s) => s.clone(),
            Node::Var(idx) => var_names.get(*idx).cloned().unwrap_or_else(|| format!("x{}", idx)),
            Node::Add(l, r) => format!("({} + {})", l.to_equation_string(var_names), r.to_equation_string(var_names)),
            Node::Sub(l, r) => format!("({} - {})", l.to_equation_string(var_names), r.to_equation_string(var_names)),
            Node::Mul(l, r) => format!("({} * {})", l.to_equation_string(var_names), r.to_equation_string(var_names)),
            Node::Div(l, r) => format!("({} / {})", l.to_equation_string(var_names), r.to_equation_string(var_names)),
            Node::Pow(l, r) => format!("({}**{})", l.to_equation_string(var_names), r.to_equation_string(var_names)),
            Node::Min(l, r) => format!("min({}, {})", l.to_equation_string(var_names), r.to_equation_string(var_names)),
            Node::Max(l, r) => format!("max({}, {})", l.to_equation_string(var_names), r.to_equation_string(var_names)),
            Node::Neg(inner) => format!("(-{})", inner.to_equation_string(var_names)),
            Node::Call(func, inner) => {
                let fname = match func {
                    Func::Sin => "sin",
                    Func::Cos => "cos",
                    Func::Tan => "tan",
                    Func::Exp => "exp",
                    Func::Ln => "log",
                    Func::Sqrt => "sqrt",
                    Func::Abs => "abs",
                    Func::Sign => "sign",
                    Func::Sinh => "sinh",
                    Func::Cosh => "cosh",
                    Func::Tanh => "tanh",
                    Func::Asin => "asin",
                    Func::Acos => "acos",
                    Func::Atan => "atan",
                    Func::Erf => "erf",
                    Func::BesselJ0 => "besselj",
                    Func::Cbrt => "cbrt",
                    _ => "f",
                };
                format!("{}({})", fname, inner.to_equation_string(var_names))
            }
        }
    }

    fn collect_paths(&self, current: &[usize], acc: &mut Vec<Vec<usize>>) {
        acc.push(current.to_vec());
        match self {
            Node::Const(_) | Node::Var(_) | Node::Ident(_) => {}
            Node::Neg(inner) | Node::Call(_, inner) => {
                let mut next = current.to_vec();
                next.push(0);
                inner.collect_paths(&next, acc);
            }
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) | Node::Min(l, r) | Node::Max(l, r) => {
                let mut next_l = current.to_vec();
                next_l.push(0);
                l.collect_paths(&next_l, acc);

                let mut next_r = current.to_vec();
                next_r.push(1);
                r.collect_paths(&next_r, acc);
            }
        }
    }

    fn get_subtree(&self, path: &[usize]) -> Option<Node> {
        if path.is_empty() {
            return Some(self.clone());
        }
        let head = path[0];
        let tail = &path[1..];
        match self {
            Node::Neg(inner) | Node::Call(_, inner) => {
                if head == 0 { inner.get_subtree(tail) } else { None }
            }
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) | Node::Min(l, r) | Node::Max(l, r) => {
                if head == 0 {
                    l.get_subtree(tail)
                } else if head == 1 {
                    r.get_subtree(tail)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn set_subtree(&self, path: &[usize], replacement: &Node) -> Node {
        if path.is_empty() {
            return replacement.clone();
        }
        let head = path[0];
        let tail = &path[1..];
        match self {
            Node::Neg(inner) => {
                if head == 0 {
                    Node::Neg(Box::new(inner.set_subtree(tail, replacement)))
                } else {
                    self.clone()
                }
            }
            Node::Call(func, inner) => {
                if head == 0 {
                    Node::Call(*func, Box::new(inner.set_subtree(tail, replacement)))
                } else {
                    self.clone()
                }
            }
            Node::Add(l, r) => {
                if head == 0 {
                    Node::Add(Box::new(l.set_subtree(tail, replacement)), r.clone())
                } else {
                    Node::Add(l.clone(), Box::new(r.set_subtree(tail, replacement)))
                }
            }
            Node::Sub(l, r) => {
                if head == 0 {
                    Node::Sub(Box::new(l.set_subtree(tail, replacement)), r.clone())
                } else {
                    Node::Sub(l.clone(), Box::new(r.set_subtree(tail, replacement)))
                }
            }
            Node::Mul(l, r) => {
                if head == 0 {
                    Node::Mul(Box::new(l.set_subtree(tail, replacement)), r.clone())
                } else {
                    Node::Mul(l.clone(), Box::new(r.set_subtree(tail, replacement)))
                }
            }
            Node::Div(l, r) => {
                if head == 0 {
                    Node::Div(Box::new(l.set_subtree(tail, replacement)), r.clone())
                } else {
                    Node::Div(l.clone(), Box::new(r.set_subtree(tail, replacement)))
                }
            }
            Node::Pow(l, r) => {
                if head == 0 {
                    Node::Pow(Box::new(l.set_subtree(tail, replacement)), r.clone())
                } else {
                    Node::Pow(l.clone(), Box::new(r.set_subtree(tail, replacement)))
                }
            }
            Node::Min(l, r) => {
                if head == 0 {
                    Node::Min(Box::new(l.set_subtree(tail, replacement)), r.clone())
                } else {
                    Node::Min(l.clone(), Box::new(r.set_subtree(tail, replacement)))
                }
            }
            Node::Max(l, r) => {
                if head == 0 {
                    Node::Max(Box::new(l.set_subtree(tail, replacement)), r.clone())
                } else {
                    Node::Max(l.clone(), Box::new(r.set_subtree(tail, replacement)))
                }
            }
            _ => self.clone(),
        }
    }
}

/// Random tree builder and genetic operators.
pub struct GeneticOps;

impl GeneticOps {
    /// `allow_trans` admits sin/cos/sqrt; `allow_exp` additionally admits
    /// exp(); `allow_misc` admits neg/log/abs/sign/erf/besselj/cbrt/min/max
    /// plus exact pi/e leaf constants; `allow_advanced` admits
    /// tan/sinh/cosh/tanh/asin/acos/atan and general Pow(x, y) with a
    /// random exponent. Overflowed or out-of-domain evaluations score as
    /// non-finite and are discarded by selection, so no clamp is applied.
    pub fn random_tree(rng: &mut FastRng, max_depth: usize, num_vars: usize, allow_trans: bool, allow_exp: bool, allow_misc: bool, allow_advanced: bool) -> Node {
        Self::random_tree_inner(rng, max_depth, num_vars, allow_trans, allow_exp, allow_misc, allow_advanced, true)
    }

    /// Trig frequency/phase template: `sin|cos(f*x + p)` with `f` and `p`
    /// drawn so selection can latch on and the polish stage can refine both
    /// constants. Physics laws with oscillatory structure are combinatorially
    /// unreachable from a uniform random tree at depth 4 (the frozen N04
    /// forensics: coarse frequency x no phase DOF -> R2 = 0.04); seeding the
    /// template directly is the recorded next-round fix.
    fn trig_template(rng: &mut FastRng, num_vars: usize) -> Node {
        let func = if rng.next_f64() < 0.5 { Func::Sin } else { Func::Cos };
        let v = rng.gen_range(0, num_vars);
        let f = ((rng.uniform(0.3, 2.8) * 10.0).round()) / 10.0;
        let arg = if rng.next_f64() < 0.35 {
            Node::Mul(Box::new(Node::Const(f)), Box::new(Node::Var(v)))
        } else {
            let p = ((rng.uniform(-3.14, 3.14) * 10.0).round()) / 10.0;
            Node::Add(
                Box::new(Node::Mul(Box::new(Node::Const(f)), Box::new(Node::Var(v)))),
                Box::new(Node::Const(p)),
            )
        };
        Node::Call(func, Box::new(arg))
    }

    fn random_tree_inner(rng: &mut FastRng, max_depth: usize, num_vars: usize, allow_trans: bool, allow_exp: bool, allow_misc: bool, allow_advanced: bool, top: bool) -> Node {
        if top && allow_trans && num_vars > 0 && max_depth >= 3 && rng.next_f64() < 0.18 {
            return Self::trig_template(rng, num_vars);
        }
        if max_depth <= 1 || rng.next_f64() < 0.35 {
            // Leaf: variable, exact pi/e constant (rare, documented), or
            // rounded uniform constant.
            if num_vars > 0 && rng.next_f64() < 0.75 {
                Node::Var(rng.gen_range(0, num_vars))
            } else if allow_misc && rng.next_f64() < 0.08 {
                Node::Const(if rng.next_f64() < 0.5 {
                    std::f64::consts::PI
                } else {
                    std::f64::consts::E
                })
            } else {
                let val = ((rng.uniform(-3.0, 3.0) * 10.0).round()) / 10.0;
                Node::Const(if val.abs() < 1e-4 { 1.0 } else { val })
            }
        } else {
            // Operator branch. Choices are packed contiguously per enabled
            // group (base, trans, exp, misc, advanced), so every reachable
            // index maps to an enabled operator under any flag combination.
            let n_trans = if allow_trans { 3 } else { 0 };
            let n_exp = if allow_exp { 1 } else { 0 };
            // Misc group: 7 unary (neg, log, abs, sign, erf, besselj,
            // cbrt) then 2 binary (min, max).
            let n_misc = if allow_misc { 9 } else { 0 };
            // Advanced group: 7 unary (tan, sinh, cosh, tanh, asin, acos,
            // atan). General Pow is handled inside the base Pow choice.
            let n_advanced = if allow_advanced { 7 } else { 0 };
            let op_choice = rng.gen_range(0, 5 + n_trans + n_exp + n_misc + n_advanced);
            let mk_sub = |rng: &mut FastRng| {
                Self::random_tree_inner(rng, max_depth - 1, num_vars, allow_trans, allow_exp, allow_misc, allow_advanced, false)
            };
            if op_choice < 5 {
                let (l, r) = (mk_sub(rng), mk_sub(rng));
                match op_choice {
                    0 => Node::Add(Box::new(l), Box::new(r)),
                    1 => Node::Sub(Box::new(l), Box::new(r)),
                    2 => Node::Mul(Box::new(l), Box::new(r)),
                    3 => Node::Div(Box::new(l), Box::new(r)),
                    _ => {
                        if allow_advanced && rng.next_f64() < 0.3 {
                            // General Pow(x, y) with a random exponent subtree.
                            Node::Pow(Box::new(l), Box::new(mk_sub(rng)))
                        } else {
                            let p = match rng.gen_range(0, 4) {
                                0 => 2.0,
                                1 => 3.0,
                                2 => -1.0,
                                _ => 0.5,
                            };
                            Node::Pow(Box::new(l), Box::new(Node::Const(p)))
                        }
                    }
                }
            } else if op_choice < 5 + n_trans {
                let f = match op_choice - 5 {
                    0 => Func::Sin,
                    1 => Func::Cos,
                    _ => Func::Sqrt,
                };
                Node::Call(f, Box::new(mk_sub(rng)))
            } else if op_choice < 5 + n_trans + n_exp {
                Node::Call(Func::Exp, Box::new(mk_sub(rng)))
            } else {
                let m = op_choice - 5 - n_trans - n_exp;
                match m {
                    0 => Node::Neg(Box::new(mk_sub(rng))),
                    1 => Node::Call(Func::Ln, Box::new(mk_sub(rng))),
                    2 => Node::Call(Func::Abs, Box::new(mk_sub(rng))),
                    3 => Node::Call(Func::Sign, Box::new(mk_sub(rng))),
                    4 => Node::Call(Func::Erf, Box::new(mk_sub(rng))),
                    5 => Node::Call(Func::BesselJ0, Box::new(mk_sub(rng))),
                    6 => Node::Call(Func::Cbrt, Box::new(mk_sub(rng))),
                    7 => {
                        let (l, r) = (mk_sub(rng), mk_sub(rng));
                        Node::Min(Box::new(l), Box::new(r))
                    }
                    8 => {
                        let (l, r) = (mk_sub(rng), mk_sub(rng));
                        Node::Max(Box::new(l), Box::new(r))
                    }
                    9 => Node::Call(Func::Tan, Box::new(mk_sub(rng))),
                    10 => Node::Call(Func::Sinh, Box::new(mk_sub(rng))),
                    11 => Node::Call(Func::Cosh, Box::new(mk_sub(rng))),
                    12 => Node::Call(Func::Tanh, Box::new(mk_sub(rng))),
                    13 => Node::Call(Func::Asin, Box::new(mk_sub(rng))),
                    14 => Node::Call(Func::Acos, Box::new(mk_sub(rng))),
                    _ => Node::Call(Func::Atan, Box::new(mk_sub(rng))),
                }
            }
        }
    }

    pub fn crossover(parent1: &Node, parent2: &Node, rng: &mut FastRng) -> Node {
        let mut paths1 = Vec::new();
        parent1.collect_paths(&[], &mut paths1);
        let mut paths2 = Vec::new();
        parent2.collect_paths(&[], &mut paths2);

        if paths1.is_empty() || paths2.is_empty() {
            return parent1.clone();
        }

        let p1 = &paths1[rng.gen_range(0, paths1.len())];
        let p2 = &paths2[rng.gen_range(0, paths2.len())];

        if let Some(sub) = parent2.get_subtree(p2) {
            let child = parent1.set_subtree(p1, &sub);
            if child.depth() <= 8 {
                return child;
            }
        }
        parent1.clone()
    }

    pub fn mutate(node: &Node, max_depth: usize, num_vars: usize, allow_trans: bool, allow_exp: bool, allow_misc: bool, allow_advanced: bool, rng: &mut FastRng) -> Node {
        let roll = rng.next_f64();
        if roll < 0.4 {
            // Subtree replacement
            let mut paths = Vec::new();
            node.collect_paths(&[], &mut paths);
            if !paths.is_empty() {
                let p = &paths[rng.gen_range(0, paths.len())];
                let sub = Self::random_tree(rng, 2, num_vars, allow_trans, allow_exp, allow_misc, allow_advanced);
                return node.set_subtree(p, &sub);
            }
        } else if roll < 0.7 {
            // Constant perturbation
            let mut paths = Vec::new();
            node.collect_paths(&[], &mut paths);
            for p in &paths {
                if let Some(Node::Const(c)) = node.get_subtree(p) {
                    let perturbed = Node::Const(c * rng.uniform(0.7, 1.3));
                    return node.set_subtree(p, &perturbed);
                }
            }
        }
        // Grow new
        Self::random_tree(rng, max_depth, num_vars, allow_trans, allow_exp, allow_misc, allow_advanced)
    }
}

/// Evaluates a candidate tree against a dataset using closed-form affine projection:
/// min_{a, b} || a * f(X) + b - y ||^2
#[inline]
pub fn score_candidate_affine<V: crate::data::DataCol>(
    node: &Node,
    data: &[V],
    target: &[f64],
    mean_target: f64,
    var_target: f64,
) -> (f64, f64, f64) {
    let n = target.len();
    let num_vars = data.len();

    let mut sum_f = 0.0f64;
    let mut sum_f2 = 0.0f64;
    let mut sum_fy = 0.0f64;

    let mut f_vals = Vec::with_capacity(n);
    let mut row_buf = vec![0.0f64; num_vars];

    for i in 0..n {
        for v in 0..num_vars {
            row_buf[v] = data[v].row(i);
        }
        let val = node.eval_row(&row_buf);
        if !val.is_finite() {
            return (f64::INFINITY, 0.0, 0.0);
        }
        sum_f += val;
        sum_f2 += val * val;
        sum_fy += val * target[i];
        f_vals.push(val);
    }

    let nf = n as f64;
    let mean_f = sum_f / nf;
    let var_f = (sum_f2 / nf) - (mean_f * mean_f);

    if var_f <= 1e-12 {
        // Constant predictor
        return (var_target, 0.0, mean_target);
    }

    let cov_fy = (sum_fy / nf) - (mean_f * mean_target);
    let a = cov_fy / var_f;
    let b = mean_target - a * mean_f;

    let mut sum_sq_err = 0.0f64;
    for i in 0..n {
        let pred = a * f_vals[i] + b;
        let err = pred - target[i];
        sum_sq_err += err * err;
    }

    let mse = sum_sq_err / nf;
    (mse, a, b)
}

/// Golden-section minimization of a unimodal closure on [lo, hi].
fn golden_section_min<F: FnMut(f64) -> f64>(
    mut lo: f64,
    mut hi: f64,
    iters: usize,
    mut f: F,
) -> (f64, f64) {
    const PHI: f64 = 0.6180339887498949;
    let mut x1 = hi - PHI * (hi - lo);
    let mut x2 = lo + PHI * (hi - lo);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..iters {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - PHI * (hi - lo);
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + PHI * (hi - lo);
            f2 = f(x2);
        }
    }
    if f1 < f2 { (x1, f1) } else { (x2, f2) }
}

/// Coordinate-descent polish of a candidate's inner constants via
/// golden-section line search (the LM-style local stage of the pipeline).
///
/// Each `Const` leaf is optimized in turn over `[v - span, v + span]` with
/// `span = max(|v|, 0.5)` (covers halving/doubling and fine moves), sweeping
/// the tree up to `sweeps` times. The affine readout is refit at every probe,
/// and a constant is only replaced when the full-pipeline MSE strictly
/// improves -- so objective-flat constants (e.g. a lone global multiplier
/// absorbed by the readout) leave the tree untouched.
pub fn polish_candidate_constants<V: crate::data::DataCol>(
    node: &Node,
    data: &[V],
    target: &[f64],
    mean_target: f64,
    var_target: f64,
    sweeps: usize,
    evals: &mut usize,
) -> (Node, f64, f64, f64) {
    let mut best = node.clone();
    let (mut best_mse, mut best_a, mut best_b) =
        score_candidate_affine(&best, data, target, mean_target, var_target);
    *evals += 1;

    let mut all_paths = Vec::new();
    best.collect_paths(&[], &mut all_paths);
    let const_paths: Vec<Vec<usize>> = all_paths
        .into_iter()
        .filter(|p| matches!(best.get_subtree(p), Some(Node::Const(_))))
        .collect();
    if const_paths.is_empty() {
        return (best, best_mse, best_a, best_b);
    }

    let values_of = |node: &Node| -> Vec<f64> {
        const_paths
            .iter()
            .map(|p| match node.get_subtree(p) {
                Some(Node::Const(v)) => v,
                _ => 0.0,
            })
            .collect()
    };

    for sweep in 0..sweeps {
        let mut improved_any = false;
        // Later sweeps narrow the search window: alternating 1D line searches
        // zigzag in coupled valleys (e.g. sin(f*x + p) frequency/phase), and a
        // fixed wide window keeps overshooting along each axis.
        let decay = 1.0 + 0.4 * sweep as f64;
        let before_vals = values_of(&best);
        for path in &const_paths {
            let Some(Node::Const(v)) = best.get_subtree(path) else {
                continue;
            };
            let span = v.abs().max(0.5) / decay;
            let (lo, hi) = (v - span, v + span);
            let mut probe = |c: f64| -> f64 {
                let cand = best.set_subtree(path, &Node::Const(c));
                let (mse, _, _) =
                    score_candidate_affine(&cand, data, target, mean_target, var_target);
                *evals += 1;
                mse
            };
            let (c_best, _) = golden_section_min(lo, hi, 28, &mut probe);
            let cand = best.set_subtree(path, &Node::Const(c_best));
            let (mse, a, b) =
                score_candidate_affine(&cand, data, target, mean_target, var_target);
            *evals += 1;
            if mse < best_mse {
                best = cand;
                best_mse = mse;
                best_a = a;
                best_b = b;
                improved_any = true;
            }
        }
        // Powell-style conjugate direction: line-search along this sweep's
        // net displacement vector to escape valleys where every single-axis
        // move is locally non-improving.
        let after_vals = values_of(&best);
        let dir: Vec<f64> = after_vals
            .iter()
            .zip(before_vals.iter())
            .map(|(a, b)| a - b)
            .collect();
        let norm: f64 = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
        if norm > 1e-12 && const_paths.len() <= 16 {
            let mut probe_dir = |t: f64| -> f64 {
                let mut cand = best.clone();
                for (path, (a, d)) in const_paths.iter().zip(after_vals.iter().zip(dir.iter())) {
                    cand = cand.set_subtree(path, &Node::Const(a + t * d));
                }
                let (mse, _, _) =
                    score_candidate_affine(&cand, data, target, mean_target, var_target);
                *evals += 1;
                mse
            };
            let (t_best, _) = golden_section_min(-1.5, 1.5, 28, &mut probe_dir);
            let mut cand = best.clone();
            for (path, (a, d)) in const_paths.iter().zip(after_vals.iter().zip(dir.iter())) {
                cand = cand.set_subtree(path, &Node::Const(a + t_best * d));
            }
            let (mse, a, b) =
                score_candidate_affine(&cand, data, target, mean_target, var_target);
            *evals += 1;
            if mse < best_mse {
                best = cand;
                best_mse = mse;
                best_a = a;
                best_b = b;
                improved_any = true;
            }
        }
        if !improved_any {
            break;
        }
    }
    (best, best_mse, best_a, best_b)
}

/// A candidate expression on the Pareto frontier.
#[derive(Clone, Debug)]
pub struct ParetoCandidate {
    pub formula: String,
    pub mse: f64,
    pub complexity: usize,
    pub scale_a: f64,
    pub bias_b: f64,
}

/// Configuration for native multi-threaded search.
#[derive(Clone, Debug)]
pub struct NativeSearchConfig {
    pub population_size: usize,
    pub generations: usize,
    pub islands: usize,
    pub migration_interval: usize,
    pub max_depth: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub allow_trans: bool,
    pub allow_exp: bool,
    pub allow_misc: bool,
    pub allow_advanced: bool,
    /// Coordinate-descent golden-section polish of inner constants applied to
    /// the top `polish_top_k` candidates of every island after evolution.
    pub polish_constants: bool,
    pub polish_top_k: usize,
    pub seed: u64,
}

impl Default for NativeSearchConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 25,
            islands: 4,
            migration_interval: 5,
            max_depth: 4,
            tournament_size: 4,
            mutation_rate: 0.35,
            crossover_rate: 0.70,
            allow_trans: true,
            allow_exp: true,
            allow_misc: true,
            allow_advanced: true,
            polish_constants: true,
            polish_top_k: 3,
            seed: 42,
        }
    }
}

#[inline(always)]
fn cmp_f64(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// Runs a high-throughput multi-island evolutionary search in pure Rust.
pub fn run_native_evolutionary_search<V: crate::data::DataCol>(
    data: &[V],
    target: &[f64],
    var_names: &[String],
    config: &NativeSearchConfig,
) -> (Vec<ParetoCandidate>, usize) {
    let n = target.len();
    let num_vars = data.len();
    let nf = n as f64;
    let mean_target = target.iter().sum::<f64>() / nf;
    let var_target = target.iter().map(|v| (v - mean_target) * (v - mean_target)).sum::<f64>() / nf;

    let total_evaluations = AtomicUsize::new(0);

    // Run islands in parallel using Rayon
    let island_results: Vec<Vec<(Node, f64, f64, f64)>> = (0..config.islands)
        .into_par_iter()
        .map(|island_id| {
            let mut rng = FastRng::new(config.seed.wrapping_add((island_id as u64) * 10007 + 1));
            let mut population: Vec<(Node, f64, f64, f64)> = Vec::with_capacity(config.population_size);

            // Initial population
            for _ in 0..config.population_size {
                let node = GeneticOps::random_tree(&mut rng, config.max_depth, num_vars, config.allow_trans, config.allow_exp, config.allow_misc, config.allow_advanced);
                let (mse, a, b) = score_candidate_affine(&node, data, target, mean_target, var_target);
                population.push((node, mse, a, b));
            }
            total_evaluations.fetch_add(config.population_size, Ordering::Relaxed);

            // Generational loop
            for _gen in 0..config.generations {
                population.sort_by(|a, b| cmp_f64(a.1, b.1));
                let elite_count = (config.population_size / 5).max(2);
                let mut next_gen: Vec<(Node, f64, f64, f64)> = population[..elite_count].to_vec();

                while next_gen.len() < config.population_size {
                    // Tournament selection
                    let mut pick = || {
                        let mut best_idx = rng.gen_range(0, population.len());
                        for _ in 1..config.tournament_size {
                            let cand = rng.gen_range(0, population.len());
                            if population[cand].1 < population[best_idx].1 {
                                best_idx = cand;
                            }
                        }
                        &population[best_idx].0
                    };

                    let p1 = pick();
                    let p2 = pick();

                    let child = if rng.next_f64() < config.crossover_rate {
                        GeneticOps::crossover(p1, p2, &mut rng)
                    } else {
                        p1.clone()
                    };

                    let mutated = if rng.next_f64() < config.mutation_rate {
                        GeneticOps::mutate(&child, config.max_depth, num_vars, config.allow_trans, config.allow_exp, config.allow_misc, config.allow_advanced, &mut rng)
                    } else {
                        child
                    };

                    let (mse, a, b) = score_candidate_affine(&mutated, data, target, mean_target, var_target);
                    next_gen.push((mutated, mse, a, b));
                }
                total_evaluations.fetch_add(config.population_size - elite_count, Ordering::Relaxed);
                population = next_gen;
            }

            // Local polish stage: refine inner constants of the best candidates
            // with a golden-section coordinate descent. Same-seed determinism is
            // preserved (islands still run in fixed order) and the pooled best
            // can only improve -- replacements require a strict MSE gain.
            if config.polish_constants && config.polish_top_k > 0 {
                population.sort_by(|a, b| cmp_f64(a.1, b.1));
                let k = config.polish_top_k.min(population.len());
                let mut polish_evals = 0usize;
                for slot in 0..k {
                    let (polished, mse, a, b) = polish_candidate_constants(
                        &population[slot].0,
                        data,
                        target,
                        mean_target,
                        var_target,
                        8,
                        &mut polish_evals,
                    );
                    population[slot] = (polished, mse, a, b);
                }
                total_evaluations.fetch_add(polish_evals, Ordering::Relaxed);
            }

            population
        })
        .collect();

    // Pool all candidates across islands and build Pareto frontier
    let mut all_scored = Vec::new();
    for pop in island_results {
        for (node, mse, a, b) in pop {
            if mse.is_finite() {
                all_scored.push((node, mse, a, b));
            }
        }
    }

    // Build Pareto frontier on (complexity, MSE)
    all_scored.sort_by(|a, b| cmp_f64(a.1, b.1));
    let mut pareto_front: Vec<ParetoCandidate> = Vec::new();

    for (node, mse, a, b) in all_scored {
        let compl = node.complexity();
        let dominated = pareto_front.iter().any(|p| p.complexity <= compl && p.mse <= mse);
        if !dominated {
            let raw_str = node.to_equation_string(var_names);
            let final_formula = if a.abs() > 1e-4 && (a - 1.0).abs() > 1e-4 && b.abs() > 1e-4 {
                format!("{:.4} * {} + {:.4}", a, raw_str, b)
            } else if a.abs() > 1e-4 && (a - 1.0).abs() > 1e-4 {
                format!("{:.4} * {}", a, raw_str)
            } else if b.abs() > 1e-4 {
                format!("{} + {:.4}", raw_str, b)
            } else {
                raw_str
            };

            pareto_front.push(ParetoCandidate {
                formula: final_formula,
                mse,
                complexity: compl,
                scale_a: a,
                bias_b: b,
            });
        }
    }

    pareto_front.sort_by_key(|p| p.complexity);
    let total_evals = total_evaluations.load(Ordering::Relaxed);
    (pareto_front, total_evals)
}

#[cfg(test)]
mod polish_tests {
    use super::*;
    use crate::parser::Func;

    fn sin_data(freq: f64, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let xs: Vec<f64> = (0..n)
            .map(|i| 0.05 + 3.0 * i as f64 / (n - 1) as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|x| (freq * x).sin()).collect();
        (vec![xs], ys)
    }

    fn stats(target: &[f64]) -> (f64, f64) {
        let nf = target.len() as f64;
        let mean = target.iter().sum::<f64>() / nf;
        let var = target.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / nf;
        (mean, var)
    }

    fn sin_const_tree(c: f64) -> Node {
        Node::Call(
            Func::Sin,
            Box::new(Node::Mul(
                Box::new(Node::Const(c)),
                Box::new(Node::Var(0)),
            )),
        )
    }

    fn recover_sin_const(node: &Node) -> f64 {
        match node {
            Node::Call(Func::Sin, inner) => match inner.as_ref() {
                Node::Mul(l, _) => match l.as_ref() {
                    Node::Const(c) => *c,
                    _ => f64::NAN,
                },
                _ => f64::NAN,
            },
            _ => f64::NAN,
        }
    }

    #[test]
    fn test_polish_refines_frequency_constant() {
        let (data, target) = sin_data(2.0, 60);
        let (mean, var) = stats(&target);
        let tree = sin_const_tree(1.8);
        let (mse_before, _, _) = score_candidate_affine(&tree, &data, &target, mean, var);
        let mut evals = 0usize;
        let (polished, mse_after, _, _) =
            polish_candidate_constants(&tree, &data, &target, mean, var, 2, &mut evals);
        let recovered = recover_sin_const(&polished);
        assert!(
            mse_after < mse_before * 1e-2,
            "polish must collapse the residual: {} -> {}",
            mse_before,
            mse_after
        );
        assert!(
            (recovered - 2.0).abs() < 5e-3,
            "frequency not refined to 2.0: {}",
            recovered
        );
        assert!(evals > 30, "eval budget must be counted: {}", evals);
    }

    #[test]
    fn test_polish_refines_phase_and_frequency_jointly() {
        // sin(1.7*x + 0.4): two coupled constants; coordinate descent must
        // improve both across sweeps.
        let xs: Vec<f64> = (0..80).map(|i| 0.02 + 3.0 * i as f64 / 79.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (1.7 * x + 0.4).sin()).collect();
        let data = vec![xs];
        let (mean, var) = stats(&ys);
        let tree = Node::Call(
            Func::Sin,
            Box::new(Node::Add(
                Box::new(Node::Mul(
                    Box::new(Node::Const(1.5)),
                    Box::new(Node::Var(0)),
                )),
                Box::new(Node::Const(0.2)),
            )),
        );
        let (mse_before, _, _) = score_candidate_affine(&tree, &data, &ys, mean, var);
        let mut evals = 0usize;
        let (polished, mse_after, _, _) =
            polish_candidate_constants(&tree, &data, &ys, mean, var, 8, &mut evals);
        assert!(
            mse_after < mse_before * 1e-3,
            "joint polish must collapse the residual: {} -> {}",
            mse_before,
            mse_after
        );
    }

    #[test]
    fn test_polish_leaves_constant_free_tree_untouched() {
        let (data, target) = sin_data(2.0, 50);
        let (mean, var) = stats(&target);
        // atan(x) tree: y ~ 0.5*atan(x) via readout; no inner constants.
        let tree = Node::Call(Func::Atan, Box::new(Node::Var(0)));
        let mut evals = 0usize;
        let (polished, mse_after, _, _) =
            polish_candidate_constants(&tree, &data, &target, mean, var, 2, &mut evals);
        assert_eq!(polished, tree);
        let (mse_before, _, _) = score_candidate_affine(&tree, &data, &target, mean, var);
        assert!((mse_after - mse_before).abs() < 1e-15);
        assert_eq!(evals, 1, "constant-free tree costs exactly one scoring eval");
    }

    #[test]
    fn test_native_search_polish_never_worsens_best() {
        let (data, target) = sin_data(2.0, 60);
        let names = vec!["x".to_string()];
        let base = NativeSearchConfig {
            population_size: 60,
            generations: 12,
            islands: 2,
            seed: 42,
            ..Default::default()
        };
        let off = NativeSearchConfig {
            polish_constants: false,
            ..base.clone()
        };
        let (pareto_off, evals_off) =
            run_native_evolutionary_search(&data, &target, &names, &off);
        let (pareto_on, evals_on) =
            run_native_evolutionary_search(&data, &target, &names, &base);
        let best_off = pareto_off.iter().map(|p| p.mse).fold(f64::INFINITY, f64::min);
        let best_on = pareto_on.iter().map(|p| p.mse).fold(f64::INFINITY, f64::min);
        assert!(
            best_on <= best_off + 1e-12,
            "polish worsened the best MSE: {} -> {}",
            best_off,
            best_on
        );
        assert!(evals_on > evals_off, "polish evals must be counted: {} vs {}", evals_on, evals_off);
    }

    #[test]
    fn test_golden_section_min_finds_quadratic_minimum() {
        let (x, f) = golden_section_min(-3.0, 5.0, 60, |t| (t - 1.25) * (t - 1.25));
        assert!((x - 1.25).abs() < 1e-6, "x={}", x);
        assert!(f < 1e-12);
    }
}

#[cfg(test)]
mod trig_seed_tests {
    use super::*;

    fn has_trig(node: &Node) -> bool {
        let mut paths = Vec::new();
        node.collect_paths(&[], &mut paths);
        paths.iter().any(|p| {
            matches!(
                node.get_subtree(p),
                Some(Node::Call(Func::Sin, _)) | Some(Node::Call(Func::Cos, _))
            )
        })
    }

    #[test]
    fn test_trig_templates_seeded_at_top_level() {
        let mut rng = FastRng::new(2026);
        let mut count = 0usize;
        for _ in 0..300 {
            let t = GeneticOps::random_tree(&mut rng, 4, 1, true, true, true, true);
            if has_trig(&t) {
                count += 1;
            }
        }
        assert!(
            (10..=110).contains(&count),
            "trig template seeding rate off: {}/300",
            count
        );
    }

    #[test]
    fn test_trig_templates_respect_flags() {
        let mut rng = FastRng::new(2027);
        for _ in 0..200 {
            let t = GeneticOps::random_tree(&mut rng, 4, 1, false, true, true, true);
            assert!(!has_trig(&t), "trig seeded while allow_trans=false");
        }
        // With num_vars = 0 the template is structurally impossible (it needs
        // a Var leaf) and is guarded before any rng draw.
        let mut rng = FastRng::new(2028);
        let t = GeneticOps::random_tree(&mut rng, 4, 0, true, true, true, true);
        let mut paths = Vec::new();
        t.collect_paths(&[], &mut paths);
        assert!(paths.iter().all(|p| t.get_subtree(p).is_some()));
    }
}
