//! RIME additive refinement: multi-term laws with fitted weights.
//!
//! ## Why this exists (measured, not assumed)
//!
//! `search` returns a single mechanism read through ONE shared readout
//! `a·out + b`. The grammar has no addition rewrite, and the catalogue only
//! sums inputs at the same time step, so a law such as
//! `y = x[t-1] + 0.5·z[t-5]` is unreachable: the power-calibration harness
//! (`tests/rime_power_calibration.rs`) recovered `x[t-1] + x[t-2]` in 0 of 40
//! replicates at every signal strength and sample size (up to rho = 0.30,
//! n = 15000) while the engine happily "promoted" a half-explanatory single
//! lag 40 times out of 40. Sums of lagged terms with different weights are the
//! most common structure in financial dynamics, so this is a representation
//! defect, not a power limit.
//!
//! ## What it does
//!
//! Forward selection on top of the *unchanged* single-term search:
//!
//! 1. Run [`search`]; the best proposal is term 1 (it already cleared the
//!    frozen gate, so a null world never reaches this stage).
//! 2. Candidate pool = the seed catalogue ∪ every accepted proposal. Each
//!    round, every pool member `g` not already in the model is tried: the
//!    model `[1, o_1 … o_k, o_g]` is refitted jointly (least squares) and
//!    scored `ΔMSE − λ_c·(edges(g) + 1 + 2·ln M)`, `M` = pool size. That is the
//!    frozen §8 score with the incremental degrees of freedom: the new graph's
//!    edges, ONE fitted coefficient, and the selection charge for choosing the
//!    best of `M` candidates. A term is added only when the best such score is
//!    positive; the loop stops otherwise or at `max_terms`.
//! 3. The fitted model is emitted as an ordinary executable [`Graph`]
//!    (`Σ c_i · term_i` from `Const`, `Mul`, `Add`), so the referee, the
//!    holdout and the Lean bridge see the same object type as before.
//!
//! No node type and no frozen rewrite is added; the stage is opt-in
//! (`max_terms >= 2`), so every existing caller is byte-for-byte unchanged.
//!
//! ## Honest limits
//!
//! * Terms must lie in the catalogue ∪ accepted proposals; a term outside that
//!   set is not found.
//! * Fitted constants are real parameters. `Const` nodes cost 0 edges, so the
//!   proposal's `complexity` is reported explicitly (edges + coefficients +
//!   selection charges) instead of trusting `edge_count`.

use rayon::prelude::*;

use super::eval::{fitted_mse, score_batch_fitted};
use super::graph::{Graph, Node};
use super::search::{multiplicity_dof, search, Proposal, SearchConfig, SearchOutcome};
use super::seeds::shift;

/// Hard cap on additive terms (including the first). Frozen with the search
/// protocol: it bounds both the multiplicity paid and the model size.
pub const MAX_ADDITIVE_TERMS: usize = 4;

/// Minimum jointly-finite rows required to fit a multi-term model.
const MIN_JOINT_ROWS: usize = 30;

/// One selected term: its graph, evaluated series and fitted coefficient.
struct Term {
    graph: Graph,
    series: Vec<f64>,
}

/// Centered least squares of `y` on `xs` over `rows`, with a RELATIVE ridge
/// (`ridge · A_ii`), so the shrinkage is scale-free. Returns `(coeffs, mse)`,
/// or `None` when the system is numerically singular (collinear terms).
fn ols(xs: &[&[f64]], y: &[f64], rows: &[usize], ridge: f64) -> Option<(Vec<f64>, f64)> {
    let k = xs.len();
    let n = rows.len() as f64;
    if rows.len() < MIN_JOINT_ROWS || k == 0 {
        return None;
    }
    let mean = |v: &[f64]| rows.iter().map(|&r| v[r]).sum::<f64>() / n;
    let mx: Vec<f64> = xs.iter().map(|x| mean(x)).collect();
    let my = mean(y);
    let mut a = vec![0.0f64; k * k];
    let mut b = vec![0.0f64; k];
    let mut syy = 0.0f64;
    for &r in rows {
        let dy = y[r] - my;
        syy += dy * dy;
        for i in 0..k {
            let di = xs[i][r] - mx[i];
            b[i] += di * dy;
            for j in 0..k {
                a[i * k + j] += di * (xs[j][r] - mx[j]);
            }
        }
    }
    let a_raw = a.clone();
    for i in 0..k {
        if a[i * k + i] <= 0.0 {
            return None; // constant column: no information
        }
        a[i * k + i] += ridge.max(0.0) * a_raw[i * k + i];
    }
    // Gaussian elimination with partial pivoting on a copy.
    let mut m = a;
    let mut rhs = b.clone();
    for col in 0..k {
        let (mut piv, mut best) = (col, m[col * k + col].abs());
        for r in col + 1..k {
            if m[r * k + col].abs() > best {
                best = m[r * k + col].abs();
                piv = r;
            }
        }
        // Relative pivot test: collinear columns collapse the pivot.
        let scale = a_raw[col * k + col].abs().max(1e-300);
        if best <= 1e-10 * scale {
            return None;
        }
        if piv != col {
            for c in 0..k {
                m.swap(col * k + c, piv * k + c);
            }
            rhs.swap(col, piv);
        }
        for r in col + 1..k {
            let f = m[r * k + col] / m[col * k + col];
            for c in col..k {
                m[r * k + c] -= f * m[col * k + c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    let mut beta = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut s = rhs[i];
        for j in i + 1..k {
            s -= m[i * k + j] * beta[j];
        }
        beta[i] = s / m[i * k + i];
    }
    if beta.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // SSE = syy − 2βᵀb + βᵀ A_raw β  (un-ridged: the honest in-sample fit).
    let mut quad = 0.0;
    for i in 0..k {
        for j in 0..k {
            quad += beta[i] * a_raw[i * k + j] * beta[j];
        }
    }
    let lin: f64 = beta.iter().zip(b.iter()).map(|(c, v)| c * v).sum();
    let sse = (syy - 2.0 * lin + quad).max(0.0);
    Some((beta, sse / n))
}

/// Rows where the target and every series are finite.
fn joint_rows(target: &[f64], series: &[&[f64]]) -> Vec<usize> {
    (0..target.len())
        .filter(|&t| target[t].is_finite() && series.iter().all(|s| s.get(t).map_or(false, |v| v.is_finite())))
        .collect()
}

/// Merge `g` into `h`'s arena; return the index of `g`'s output inside `h`.
fn append(h: &mut Graph, g: &Graph) -> usize {
    let off = h.nodes.len();
    for n in &g.nodes {
        h.push(shift(n, off));
    }
    g.output + off
}

/// Executable graph of `Σ c_i · term_i`.
fn compose(terms: &[Term], coeffs: &[f64]) -> Graph {
    let mut h = Graph::new();
    let mut acc: Option<usize> = None;
    for (t, c) in terms.iter().zip(coeffs.iter()) {
        let out = append(&mut h, &t.graph);
        let k = h.push(Node::Const(*c));
        let scaled = h.push(Node::Mul(k, out));
        acc = Some(match acc {
            None => scaled,
            Some(prev) => h.push(Node::Add(prev, scaled)),
        });
    }
    h.output = acc.expect("compose needs at least one term");
    h
}

/// Frozen-search first, then forward-selected additive terms.
///
/// `max_terms <= 1` returns `search`'s outcome untouched.
pub fn search_additive<V: crate::data::DataCol>(
    seeds: &[Graph],
    cols: &[V],
    target: &[f64],
    cfg: &SearchConfig,
    max_terms: usize,
) -> SearchOutcome {
    let mut outcome = search(seeds, cols, target, cfg);
    let max_terms = max_terms.min(MAX_ADDITIVE_TERMS);
    if max_terms < 2 {
        return outcome;
    }
    let Some(first) = outcome.proposals.first().cloned() else {
        return outcome; // abstained: nothing to refine
    };

    // ---- candidate pool (deterministic order, expression-deduplicated) ----
    let mut pool: Vec<Graph> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let admissible = |g: &Graph| g.validate().is_ok() && !(cfg.static_ops && g.is_temporal());
    for g in seeds.iter().cloned().chain(outcome.proposals.iter().map(|p| p.graph.clone())) {
        if admissible(&g) && seen.insert(g.describe()) {
            pool.push(g);
        }
    }
    let series: Vec<Vec<f64>> = pool.par_iter().map(|g| g.evaluate(cols)).collect();
    let sel = multiplicity_dof(pool.len());

    let mut terms = vec![Term { graph: first.graph.clone(), series: first.graph.evaluate(cols) }];
    let mut used: Vec<String> = vec![first.graph.describe()];
    let mut added_dof = 0.0f64;
    let mut added_selection = 0.0f64;

    while terms.len() < max_terms {
        let cur: Vec<&[f64]> = terms.iter().map(|t| t.series.as_slice()).collect();
        let scored: Vec<Option<f64>> = pool
            .par_iter()
            .zip(series.par_iter())
            .map(|(g, s)| {
                if used.contains(&g.describe()) {
                    return None;
                }
                let mut with: Vec<&[f64]> = cur.clone();
                with.push(s.as_slice());
                let rows = joint_rows(target, &with);
                let (_, mse_new) = ols(&with, target, &rows, cfg.ridge)?;
                let (_, mse_cur) = ols(&cur, target, &rows, cfg.ridge)?;
                let charge = cfg.lambda_c * (g.edge_count() as f64 + 1.0 + sel);
                let score = (mse_cur - mse_new) - charge;
                score.is_finite().then_some(score)
            })
            .collect();
        // Best positive score; ties broken by expression text (total order).
        let mut best: Option<(usize, f64)> = None;
        for (i, s) in scored.iter().enumerate() {
            if let Some(v) = s {
                if *v > 0.0 {
                    let better = match best {
                        None => true,
                        Some((bi, bv)) => {
                            *v > bv || (*v == bv && pool[i].describe() < pool[bi].describe())
                        }
                    };
                    if better {
                        best = Some((i, *v));
                    }
                }
            }
        }
        let Some((bi, _)) = best else { break };
        added_dof += pool[bi].edge_count() as f64 + 1.0;
        added_selection += sel;
        used.push(pool[bi].describe());
        terms.push(Term { graph: pool[bi].clone(), series: series[bi].clone() });
    }

    if terms.len() < 2 {
        return outcome; // no additional term earned its charge
    }

    // ---- final joint fit on the rows every term can see ----
    let all: Vec<&[f64]> = terms.iter().map(|t| t.series.as_slice()).collect();
    let rows = joint_rows(target, &all);
    let Some((coeffs, _)) = ols(&all, target, &rows, cfg.ridge) else { return outcome };
    let graph = compose(&terms, &coeffs);
    if graph.validate().is_err() {
        return outcome;
    }

    // Score everything on the SAME rows so parent and child are comparable.
    let mut masked = vec![f64::NAN; target.len()];
    for &r in &rows {
        masked[r] = target[r];
    }
    let (parent_mse, _, _) = fitted_mse(&first.graph, cols, &masked, cfg.ridge);
    let fs = score_batch_fitted(&[(graph.clone(), parent_mse, masked.clone())], cols, &cfg.score_config())
        .into_iter()
        .next()
        .expect("one job in, one score out");
    if !fs.mse.is_finite() {
        return outcome;
    }
    // Honest accounting: first term's charge + each added term's edges, one
    // coefficient and selection charge (NOT edge_count of the composite, in
    // which fitted `Const`s are free and `Mul`/`Add` plumbing is over-counted).
    let complexity = first.complexity + added_dof + added_selection;
    let score = (parent_mse - fs.mse) - cfg.lambda_c * (added_dof + added_selection);
    let ref_score = (outcome.baseline_mse - fs.mse) - cfg.lambda_c * complexity;
    if !score.is_finite() || score <= 0.0 {
        return outcome; // joint refit on common rows did not pay: keep single term
    }

    let refined = Proposal {
        graph_id: graph.id(),
        parent_id: first.graph_id.clone(),
        rewrite_op: "AddTerm".to_string(),
        depth: first.depth + terms.len() - 1,
        equation: graph.describe(),
        canonical: graph.canonical(),
        score,
        ref_score,
        mse: fs.mse,
        r2: fs.r2,
        scale: fs.scale,
        intercept: fs.intercept,
        complexity,
        edges: graph.edge_count(),
        selection_dof: first.selection_dof + added_selection,
        n_eff: fs.n_eff,
        seed: cfg.seed,
        graph,
    };
    outcome.proposals.insert(0, refined);
    outcome.accepted = outcome.proposals.len();
    outcome
}

#[cfg(test)]
mod tests {
    use super::super::seeds::small_forms;
    use super::*;

    fn hash_noise(n: usize, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let v = ((t as f64 + phase) * 12.9898).sin() * 43758.5453;
                v - v.floor() - 0.5
            })
            .collect()
    }

    fn world(n: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = hash_noise(n, 0.0);
        let z = hash_noise(n, 7.0);
        let eps = hash_noise(n, 900.0);
        let y: Vec<f64> = (0..n)
            .map(|t| if t >= 5 { x[t - 1] + 0.5 * z[t - 5] + noise * eps[t] } else { f64::NAN })
            .collect();
        (vec![x, z], y)
    }

    fn cfg_for(y: &[f64]) -> SearchConfig {
        let n_dev = y.iter().filter(|v| v.is_finite()).count();
        SearchConfig::noise_scaled(y, n_dev, 1.5).with_max_depth(2)
    }

    #[test]
    fn single_term_search_cannot_express_a_two_term_law() {
        // Baseline for the claim in the module docs: without refinement the
        // best proposal explains only part of the variance.
        let (cols, y) = world(3000, 0.05);
        let out = search(&small_forms(2), &cols, &y, &cfg_for(&y));
        let best = out.best().expect("a partial term is still promoted");
        assert!(best.r2 < 0.9, "single term unexpectedly complete: r2 {}", best.r2);
    }

    #[test]
    fn two_term_law_is_recovered_with_fitted_weights() {
        let (cols, y) = world(3000, 0.05);
        let out = search_additive(&small_forms(2), &cols, &y, &cfg_for(&y), 3);
        let best = out.best().unwrap();
        assert_eq!(best.rewrite_op, "AddTerm", "{}", best.equation);
        assert!(best.r2 > 0.95, "r2 {}", best.r2);
        // Executable graph: evaluates to the planted law up to the readout.
        let pred = best.graph.evaluate(&cols);
        let truth: Vec<f64> = (0..y.len())
            .map(|t| if t >= 5 { cols[0][t - 1] + 0.5 * cols[1][t - 5] } else { f64::NAN })
            .collect();
        let (mut sab, mut saa, mut sbb, mut n) = (0.0, 0.0, 0.0, 0.0);
        let (mut ma, mut mb) = (0.0, 0.0);
        for (p, q) in pred.iter().zip(truth.iter()).filter(|(p, q)| p.is_finite() && q.is_finite()) {
            ma += p;
            mb += q;
            n += 1.0;
        }
        ma /= n;
        mb /= n;
        for (p, q) in pred.iter().zip(truth.iter()).filter(|(p, q)| p.is_finite() && q.is_finite()) {
            sab += (p - ma) * (q - mb);
            saa += (p - ma) * (p - ma);
            sbb += (q - mb) * (q - mb);
        }
        assert!(sab / (saa.sqrt() * sbb.sqrt()) > 0.99);
    }

    #[test]
    fn added_terms_pay_edges_coefficient_and_selection() {
        let (cols, y) = world(3000, 0.05);
        let cfg = cfg_for(&y);
        let single = search(&small_forms(2), &cols, &y, &cfg);
        let multi = search_additive(&small_forms(2), &cols, &y, &cfg, 2);
        let first = single.best().unwrap();
        let refined = multi.best().unwrap();
        assert_eq!(refined.rewrite_op, "AddTerm");
        // At least one coefficient + the selection charge beyond the first term.
        assert!(
            refined.complexity >= first.complexity + 1.0 + 1.0,
            "{} vs {}",
            refined.complexity,
            first.complexity
        );
        assert!(refined.selection_dof > first.selection_dof);
        assert!(refined.score > 0.0);
    }

    #[test]
    fn single_term_law_gets_no_spurious_extra_term() {
        let n = 3000;
        let x = hash_noise(n, 0.0);
        let z = hash_noise(n, 7.0);
        let eps = hash_noise(n, 900.0);
        let y: Vec<f64> = (0..n).map(|t| if t >= 2 { 3.0 * x[t - 2] + 0.5 * eps[t] } else { f64::NAN }).collect();
        let cols = vec![x, z];
        let out = search_additive(&small_forms(2), &cols, &y, &cfg_for(&y), 4);
        assert_ne!(out.best().unwrap().rewrite_op, "AddTerm", "{}", out.best().unwrap().equation);
    }

    #[test]
    fn null_world_still_abstains_and_off_switch_is_identity() {
        let n = 1500;
        let cols = vec![hash_noise(n, 0.0), hash_noise(n, 7.0)];
        let y = hash_noise(n, 4242.0);
        let cfg = cfg_for(&y);
        assert!(search_additive(&small_forms(2), &cols, &y, &cfg, 4).proposals.is_empty());
        // max_terms = 1 is exactly `search`.
        let (c2, y2) = world(2000, 0.05);
        let cfg2 = cfg_for(&y2);
        let a = search(&small_forms(2), &c2, &y2, &cfg2);
        let b = search_additive(&small_forms(2), &c2, &y2, &cfg2, 1);
        let ids = |o: &SearchOutcome| o.proposals.iter().map(|p| p.graph_id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
    }

    #[test]
    fn deterministic() {
        let (cols, y) = world(2000, 0.05);
        let cfg = cfg_for(&y);
        let a = search_additive(&small_forms(2), &cols, &y, &cfg, 3);
        let b = search_additive(&small_forms(2), &cols, &y, &cfg, 3);
        let ids = |o: &SearchOutcome| o.proposals.iter().map(|p| p.graph_id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
    }

    #[test]
    fn collinear_terms_are_refused_not_solved() {
        let x = hash_noise(200, 0.0);
        let y = hash_noise(200, 5.0);
        let dup = x.clone();
        let rows: Vec<usize> = (0..200).collect();
        assert!(ols(&[&x, &dup], &y, &rows, 0.0).is_none());
        assert!(ols(&[&x], &y, &rows, 1e-9).is_some());
    }
}
