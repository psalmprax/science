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
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) => {
                1 + l.depth().max(r.depth())
            }
        }
    }

    fn complexity(&self) -> usize {
        match self {
            Node::Const(_) | Node::Var(_) | Node::Ident(_) => 1,
            Node::Neg(inner) => 1 + inner.complexity(),
            Node::Call(_, inner) => 2 + inner.complexity(),
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) => {
                1 + l.complexity() + r.complexity()
            }
        }
    }

    fn to_equation_string(&self, var_names: &[String]) -> String {
        match self {
            Node::Const(c) => {
                if (c.round() - c).abs() < 1e-6 {
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
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) => {
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
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) | Node::Pow(l, r) => {
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
            _ => self.clone(),
        }
    }
}

/// Random tree builder and genetic operators.
pub struct GeneticOps;

impl GeneticOps {
    pub fn random_tree(rng: &mut FastRng, max_depth: usize, num_vars: usize, allow_trans: bool) -> Node {
        if max_depth <= 1 || rng.next_f64() < 0.35 {
            // Leaf: variable or constant
            if num_vars > 0 && rng.next_f64() < 0.75 {
                Node::Var(rng.gen_range(0, num_vars))
            } else {
                let val = ((rng.uniform(-3.0, 3.0) * 10.0).round()) / 10.0;
                Node::Const(if val.abs() < 1e-4 { 1.0 } else { val })
            }
        } else {
            // Operator branch
            let op_choice = rng.gen_range(0, if allow_trans { 8 } else { 5 });
            match op_choice {
                0 => Node::Add(
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                ),
                1 => Node::Sub(
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                ),
                2 => Node::Mul(
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                ),
                3 => Node::Div(
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                    Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                ),
                4 => {
                    let p = match rng.gen_range(0, 4) {
                        0 => 2.0,
                        1 => 3.0,
                        2 => -1.0,
                        _ => 0.5,
                    };
                    Node::Pow(
                        Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans)),
                        Box::new(Node::Const(p)),
                    )
                }
                5 => Node::Call(Func::Sin, Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans))),
                6 => Node::Call(Func::Cos, Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans))),
                _ => Node::Call(Func::Sqrt, Box::new(Self::random_tree(rng, max_depth - 1, num_vars, allow_trans))),
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

    pub fn mutate(node: &Node, max_depth: usize, num_vars: usize, allow_trans: bool, rng: &mut FastRng) -> Node {
        let roll = rng.next_f64();
        if roll < 0.4 {
            // Subtree replacement
            let mut paths = Vec::new();
            node.collect_paths(&[], &mut paths);
            if !paths.is_empty() {
                let p = &paths[rng.gen_range(0, paths.len())];
                let sub = Self::random_tree(rng, 2, num_vars, allow_trans);
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
        Self::random_tree(rng, max_depth, num_vars, allow_trans)
    }
}

/// Evaluates a candidate tree against a dataset using closed-form affine projection:
/// min_{a, b} || a * f(X) + b - y ||^2
#[inline]
pub fn score_candidate_affine(
    node: &Node,
    data: &[Vec<f64>],
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
            row_buf[v] = data[v][i];
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
pub fn run_native_evolutionary_search(
    data: &[Vec<f64>],
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
                let node = GeneticOps::random_tree(&mut rng, config.max_depth, num_vars, config.allow_trans);
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
                        GeneticOps::mutate(&child, config.max_depth, num_vars, config.allow_trans, &mut rng)
                    } else {
                        child
                    };

                    let (mse, a, b) = score_candidate_affine(&mutated, data, target, mean_target, var_target);
                    next_gen.push((mutated, mse, a, b));
                }
                total_evaluations.fetch_add(config.population_size - elite_count, Ordering::Relaxed);
                population = next_gen;
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
