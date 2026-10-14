//! RIME seed catalogue (spec §7: "grammar-exhaustive small forms").
//!
//! `AddLagEdge` can only attach a lag to a form that already exists, and no
//! frozen rewrite introduces an addition, a negation or a difference. Without a
//! seed catalogue the five rewrites therefore cannot reach the difference,
//! momentum and interaction forms that financial dynamics are made of, so the
//! catalogue is part of the mechanism, not a convenience.
//!
//! The catalogue is deterministic: fixed construction order, deduplicated by
//! graph id, truncated at [`MAX_SEED_FORMS`] (the spec's ≤60 budget). Identical
//! inputs give byte-identical forms on every platform. Families are emitted in
//! priority order (levels, delays, differences, cross-input structure, memory,
//! wrappers), so a truncation for many inputs drops multiplicity at the tail
//! rather than a whole family.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::graph::{Graph, Node, LAG_SET};

/// Spec §7 seed budget.
pub const MAX_SEED_FORMS: usize = 60;

/// Decay priors for state seeds and for the `AddState` rewrite: a **discrete**
/// grid, not a fitted parameter. Recovery of a memory law is exact when its
/// decay is on the grid and approximate otherwise; the grid is part of the
/// frozen search configuration, so it is declared here once and used by both
/// the catalogue and the rewrite enumeration.
pub const DECAY_PRIORS: [(usize, usize); 5] = [(1, 2), (7, 10), (4, 5), (9, 10), (1, 1)];

/// Lag multiplicity for the interaction families. Levels and differences use
/// the full frozen lag set; interaction forms are emitted for this reduced set
/// so that every family fits inside the 60-form budget for two inputs.
pub const INTERACTION_LAGS: [usize; 3] = [1, 2, 5];

fn add(out: &mut Vec<Graph>, seen: &mut HashSet<String>, g: Graph) {
    if out.len() >= MAX_SEED_FORMS {
        return;
    }
    if seen.insert(g.id()) {
        out.push(g);
    }
}

fn single_input(v: usize) -> Graph {
    let mut g = Graph::new();
    g.output = g.push(Node::Input(v));
    g
}

/// Wrap the current output in a new node.
fn wrap(g: &Graph, f: impl Fn(usize) -> Node) -> Graph {
    let mut h = g.clone();
    let o = h.output;
    h.output = h.push(f(o));
    h
}

fn lag(g: &Graph, k: usize) -> Graph {
    wrap(g, |i| Node::Lag(i, k))
}

pub(crate) fn shift(n: &Node, off: usize) -> Node {
    match n {
        Node::Input(i) => Node::Input(*i),
        Node::Const(c) => Node::Const(*c),
        Node::Add(a, b) => Node::Add(a + off, b + off),
        Node::Mul(a, b) => Node::Mul(a + off, b + off),
        Node::Neg(a) => Node::Neg(a + off),
        Node::Tanh(a) => Node::Tanh(a + off),
        Node::Sign(a) => Node::Sign(a + off),
        Node::Abs(a) => Node::Abs(a + off),
        Node::Lag(a, k) => Node::Lag(a + off, *k),
        Node::State { src, decay, init } => Node::State { src: src + off, decay: *decay, init: *init },
    }
}

/// Merge two graphs into one arena and combine their outputs. `f` receives the
/// merged arena so it can push auxiliary nodes (e.g. a negation for `a - b`).
fn combine(a: &Graph, b: &Graph, f: impl Fn(&mut Graph, usize, usize) -> Node) -> Graph {
    let mut h = a.clone();
    let off = h.nodes.len();
    let bo = b.output + off;
    for n in &b.nodes {
        let s = shift(n, off);
        h.push(s);
    }
    let ao = a.output;
    let out = f(&mut h, ao, bo);
    h.output = h.push(out);
    h
}

fn sub(a: &Graph, b: &Graph) -> Graph {
    combine(a, b, |h, x, y| {
        let ny = h.push(Node::Neg(y));
        Node::Add(x, ny)
    })
}

fn add_g(a: &Graph, b: &Graph) -> Graph {
    combine(a, b, |_, x, y| Node::Add(x, y))
}

fn mul_g(a: &Graph, b: &Graph) -> Graph {
    combine(a, b, |_, x, y| Node::Mul(x, y))
}

/// Seed-catalogue family-2 ordering (opt-in, `v0.1.6` diagnostics axis).
///
/// `Legacy` emits every input's five delayed levels before the next input's
/// (input-major). At more inputs than the 60-form budget can hold (≈7 at 5
/// lags each), truncation therefore removes whole inputs from the delayed
/// family — measured: positions 15/16/19 of the 22-signal frame lost their
/// lag-1 seed and could never be recovered, while a solo-frame control
/// recovered the identical dependence at R²>0.95.
///
/// `CoverageFirst` emits lag-major instead: one lag per input across ALL
/// inputs, then the next lag round. Same budget, same families, same forms
/// for small input counts (≤60/n... exactly: identical whenever the legacy
/// order fits inside the budget); different WHICH forms survive truncation
/// at high input counts. OFF by default so the frozen catalogue is unchanged
/// for every existing caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedOrder {
    Legacy,
    CoverageFirst,
}

impl SeedOrder {
    /// True for the frozen default (used by `skip_serializing_if`).
    pub fn is_legacy(&self) -> bool {
        matches!(self, SeedOrder::Legacy)
    }
}

impl Default for SeedOrder {
    fn default() -> Self {
        SeedOrder::Legacy
    }
}

/// Build the small-form catalogue over `n_inputs` variables (legacy family-2
/// order — input-major delayed levels).
pub fn small_forms(n_inputs: usize) -> Vec<Graph> {
    small_forms_with_order(n_inputs, SeedOrder::Legacy)
}

/// Build the small-form catalogue with an explicit family-2 ordering.
pub fn small_forms_with_order(n_inputs: usize, order: SeedOrder) -> Vec<Graph> {
    let n = n_inputs.max(1);
    let inputs: Vec<Graph> = (0..n).map(single_input).collect();
    let lagged: Vec<Vec<Graph>> = inputs.iter().map(|g| LAG_SET.iter().map(|k| lag(g, *k)).collect()).collect();
    let mut out: Vec<Graph> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let push = |out: &mut Vec<Graph>, seen: &mut HashSet<String>, g: Graph| add(out, seen, g);

    // 1. Levels.
    for g in &inputs {
        push(&mut out, &mut seen, g.clone());
    }
    // 2. Delayed levels. `CoverageFirst` walks lag rounds outer, inputs inner,
    // so every input's shortest lag survives truncation first.
    match order {
        SeedOrder::Legacy => {
            for ls in &lagged {
                for g in ls {
                    push(&mut out, &mut seen, g.clone());
                }
            }
        }
        SeedOrder::CoverageFirst => {
            for k in 0..LAG_SET.len() {
                for ls in &lagged {
                    push(&mut out, &mut seen, ls[k].clone());
                }
            }
        }
    }
    // 3. Differences: level - level[t-k] (momentum, reversal).
    for (i, g) in inputs.iter().enumerate() {
        for lk in &lagged[i] {
            push(&mut out, &mut seen, sub(g, lk));
        }
    }
    // 4. Momentum across horizons: level[t-1] - level[t-k].
    for (i, _) in inputs.iter().enumerate() {
        for k in [2usize, 5, 20] {
            let idx = LAG_SET.iter().position(|x| *x == k).expect("lag in set");
            push(&mut out, &mut seen, sub(&lagged[i][0], &lagged[i][idx]));
        }
    }
    // 5. Cross-input structure: level, difference and same-time product.
    for (i, ga) in inputs.iter().enumerate() {
        for gb in inputs.iter().skip(i + 1) {
            push(&mut out, &mut seen, add_g(ga, gb));
            push(&mut out, &mut seen, sub(ga, gb));
            push(&mut out, &mut seen, mul_g(ga, gb));
        }
    }
    // 6. Delayed interactions, both orientations: the lead-lag product family.
    for i in 0..inputs.len() {
        for j in 0..inputs.len() {
            if i == j {
                continue;
            }
            for k in INTERACTION_LAGS {
                let idx = LAG_SET.iter().position(|x| *x == k).expect("lag in set");
                push(&mut out, &mut seen, mul_g(&inputs[i], &lagged[j][idx]));
            }
        }
    }
    // 7. Self-interaction: level * level[t-k].
    for (i, g) in inputs.iter().enumerate() {
        for k in INTERACTION_LAGS {
            let idx = LAG_SET.iter().position(|x| *x == k).expect("lag in set");
            push(&mut out, &mut seen, mul_g(g, &lagged[i][idx]));
        }
    }
    // 8. Persistent memory cells (decay grid).
    for g in &inputs {
        for (num, den) in DECAY_PRIORS {
            let d = num as f64 / den as f64;
            push(&mut out, &mut seen, wrap(g, |i| Node::State { src: i, decay: d, init: 0.0 }));
        }
    }
    // 9. Bounded response of a difference (spec §6 heteroskedastic remedy).
    for (i, g) in inputs.iter().enumerate() {
        let diff = sub(g, &lagged[i][0]);
        push(&mut out, &mut seen, wrap(&diff, |i| Node::Tanh(i)));
    }
    // 10. Bounded response of a level.
    for g in &inputs {
        push(&mut out, &mut seen, wrap(g, |i| Node::Tanh(i)));
    }
    // 11. Shape wrappers (first input only: the remaining budget at n = 2).
    // `Neg` is deliberately absent: a negation is exactly what a fitted negative
    // readout scale already expresses.
    if let Some(g) = inputs.first() {
        push(&mut out, &mut seen, wrap(g, |i| Node::Abs(i)));
        push(&mut out, &mut seen, wrap(g, |i| Node::Sign(i)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_is_deterministic_and_deduplicated() {
        let a = small_forms(2);
        let b = small_forms(2);
        let ids_a: Vec<String> = a.iter().map(|g| g.id()).collect();
        let ids_b: Vec<String> = b.iter().map(|g| g.id()).collect();
        assert_eq!(ids_a, ids_b);
        let unique: HashSet<&String> = ids_a.iter().collect();
        assert_eq!(unique.len(), ids_a.len(), "duplicate seed forms");
        assert!(a.len() <= MAX_SEED_FORMS);
        assert!(a.len() >= 20, "catalogue too thin: {}", a.len());
    }

    #[test]
    fn coverage_first_is_identical_inside_budget() {
        // Small input counts: ordering cannot matter (nothing is truncated).
        for n in 1..=6usize {
            let legacy = small_forms(n);
            let cov = small_forms_with_order(n, SeedOrder::CoverageFirst);
            let li: HashSet<String> = legacy.iter().map(|g| g.id()).collect();
            let ci: HashSet<String> = cov.iter().map(|g| g.id()).collect();
            assert_eq!(li, ci, "ordering changed the set at n={}", n);
            assert_eq!(legacy.len(), cov.len(), "ordering changed the count at n={}", n);
        }
    }

    #[test]
    fn coverage_first_keeps_every_inputs_shortest_lag() {
        // 22 inputs: 22 levels + 110 delayed levels cannot fit 60. Legacy
        // truncation loses whole inputs; coverage-first keeps lag-1 for all.
        let cov = small_forms_with_order(22, SeedOrder::CoverageFirst);
        assert!(cov.len() <= MAX_SEED_FORMS);
        let described: Vec<String> = cov.iter().map(|g| g.describe()).collect();
        for k in 0..22 {
            assert!(described.contains(&format!("v{}[t-1]", k)), "input {} lost lag-1", k);
        }
        // Legacy, for contrast, loses at least the measured tail inputs.
        let legacy = small_forms(22);
        let ld: Vec<String> = legacy.iter().map(|g| g.describe()).collect();
        assert!(!ld.contains(&"v19[t-1]".to_string()), "legacy unexpectedly kept v19 lag-1");
    }

    #[test]
    fn orderings_share_budget_and_validity() {
        for &n in &[2usize, 8, 22] {
            for &o in &[SeedOrder::Legacy, SeedOrder::CoverageFirst] {
                let forms = small_forms_with_order(n, o);
                assert!(forms.len() <= MAX_SEED_FORMS, "n={} over budget", n);
                for g in &forms {
                    g.validate().unwrap_or_else(|e| panic!("invalid seed {}: {}", g.describe(), e));
                }
            }
        }
    }

    #[test]
    fn every_seed_is_valid_and_finite_on_a_series() {
        let cols = vec![vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6], vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5]];
        for g in small_forms(2) {
            g.validate().unwrap_or_else(|e| panic!("invalid seed {}: {}", g.describe(), e));
            let out = g.evaluate(&cols);
            assert_eq!(out.len(), 6);
        }
    }

    #[test]
    fn catalogue_contains_every_family_for_two_inputs() {
        let forms = small_forms(2);
        let described: Vec<String> = forms.iter().map(|g| g.describe()).collect();
        let has = |needle: &str| described.iter().any(|d| d.contains(needle));
        assert!(has("[t-2]"), "no lag-2 form");
        assert!(has("-"), "no difference form");
        assert!(has("state_"), "no memory form");
        assert!(has("tanh("), "no bounded-response form");
        assert!(has("abs(") && has("sign("), "no shape-wrapper form");
        // Lead-lag product in both orientations: v0[t] * v1[t-1] and v1[t] * v0[t-1].
        assert!(has("(v0 * v1[t-1])") || has("(v1[t-1] * v0)"), "missing v0 * v1[t-1]");
        assert!(has("(v1 * v0[t-1])") || has("(v0[t-1] * v1)"), "missing v1 * v0[t-1]");
    }

    #[test]
    fn one_input_catalogue_is_small_and_complete() {
        let forms = small_forms(1);
        assert!(forms.len() <= MAX_SEED_FORMS);
        let described: Vec<String> = forms.iter().map(|g| g.describe()).collect();
        for prior in DECAY_PRIORS {
            let d = format!("state_{:.4}", prior.0 as f64 / prior.1 as f64);
            assert!(described.iter().any(|s| s.contains(&d)), "missing decay prior {}", d);
        }
    }
}
