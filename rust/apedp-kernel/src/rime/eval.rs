//! RIME batch evaluator: parallel candidate scoring (spec §8, §12).
//!
//! Score (frozen): S(G') = ΔMSE_dev − λc·(|E'| + readout parameters).
//!
//! ## v0.1.1 correctness fix (Stage-1 audit)
//!
//! Graphs used to be scored on their **raw** output: `x[t-2]` was compared
//! directly against `y = 3·x[t-2] + 100`, so every planted mechanism with a
//! fitted amplitude or offset scored worse than the null and could not be
//! discovered. Spec §7 already required a fit ("ridge fit on development data
//! ONLY"); it was never implemented. [`fit_readout`] now performs a
//! ridge-regularised least-squares fit of `a·out + b` before the MSE is taken,
//! and the two readout degrees of freedom are charged against the complexity
//! penalty so fitting alone cannot buy a promotion on a null world.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::graph::Graph;

/// Degrees of freedom spent by the fitted linear readout (scale + intercept).
/// Charged to every candidate so that "improvement" attributable purely to
/// in-sample fitting does not read as structure.
pub const READOUT_PARAMS: f64 = 2.0;

/// How a candidate is charged and fitted.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreConfig {
    /// Penalty per charged degree of freedom (spec §8 `lambda_c`).
    pub lambda_c: f64,
    /// Ridge coefficient for the readout fit.
    pub ridge: f64,
    /// Selection charge in degrees of freedom (EBIC-style `2·ln M` for a search
    /// over `M` candidates). Zero for a single-candidate evaluation, and set by
    /// the search so that the multiplicity the search itself created is paid
    /// for (spec §10, "Bonferroni over grammar size").
    pub selection_dof: f64,
}

impl ScoreConfig {
    /// Single-candidate evaluation: no selection charge.
    pub fn new(lambda_c: f64, ridge: f64) -> Self {
        ScoreConfig { lambda_c, ridge, selection_dof: 0.0 }
    }

    pub fn with_selection_dof(mut self, dof: f64) -> Self {
        self.selection_dof = dof.max(0.0);
        self
    }
}

/// A candidate's charged degrees of freedom: graph edges, the two readout
/// parameters, and the search's own selection charge.
pub fn charged_dof(graph: &Graph, selection_dof: f64) -> f64 {
    graph.edge_count() as f64 + READOUT_PARAMS + selection_dof.max(0.0)
}

/// Mean-squared error over finite entries only. NaN rows (lag warmup) are
/// skipped and counted, never imputed.
pub fn mse_finite(output: &[f64], target: &[f64]) -> (f64, usize) {
    let mut se = 0.0;
    let mut n = 0usize;
    let mut skipped = 0usize;
    for (o, t) in output.iter().zip(target.iter()) {
        if o.is_finite() && t.is_finite() {
            let d = o - t;
            se += d * d;
            n += 1;
        } else {
            skipped += 1;
        }
    }
    (if n > 0 { se / n as f64 } else { f64::INFINITY }, skipped)
}

/// Ridge-regularised least-squares readout.
///
/// Minimises `Σ (a·o_t + b − y_t)² + ridge·Σ o_t² · a²` over finite `(o, y)`
/// pairs, returning `(a, b)`. `ridge` shrinks the slope toward zero (the
/// intercept is free, since the target's mean is not evidence of mechanism).
/// Fewer than two finite pairs, or a constant output, yields `(0, ȳ)`.
///
/// v0.1.2 numerics fix: the ridge term is now RELATIVE to the regressor's raw
/// second moment (`Σ o²`), not an absolute literal added to `Σ(o−ō)²`. The old
/// form is scale-dependent — with the algebraic-transform columns (e.g. a
/// squared axis at 1e-8 magnitude) a fixed 1e-9 ridge exceeded the entire
/// signal energy by ~30× and silently zeroed the mechanism. For unit-scale
/// regressors (all pre-v0.1.2 planted tests) the two forms agree to well below
/// every test tolerance; for extreme scales this is the only correct form.
pub fn fit_readout(output: &[f64], target: &[f64], ridge: f64) -> (f64, f64) {
    let mut n = 0.0f64;
    let (mut so, mut sy, mut soo, mut soy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (o, t) in output.iter().zip(target.iter()) {
        if o.is_finite() && t.is_finite() {
            n += 1.0;
            so += *o;
            sy += *t;
            soo += o * o;
            soy += o * t;
        }
    }
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let mo = so / n;
    let my = sy / n;
    let sxx = (soo - n * mo * mo).max(0.0) + ridge.max(0.0) * soo.max(0.0);
    let sxy = soy - n * mo * my;
    let a = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (a, my - a * mo)
}

/// Residual statistics of the fitted readout: `(mse, tss, n_eff, skipped)`.
/// `tss` is the total sum of squares about the target mean over the same rows,
/// so `1 − mse/tss` is the readout's R².
///
/// `pub` for the audit-only phase-decomposition harness
/// (`tests/rime_profile.rs`), which needs fit+stats WITHOUT graph evaluation.
pub fn residual_stats(output: &[f64], target: &[f64], a: f64, b: f64) -> (f64, f64, usize, usize) {
    let mut se = 0.0;
    let mut sy = 0.0;
    let mut n = 0usize;
    let mut skipped = 0usize;
    for (o, t) in output.iter().zip(target.iter()) {
        if o.is_finite() && t.is_finite() {
            sy += *t;
            n += 1;
        }
    }
    if n == 0 {
        return (f64::INFINITY, 0.0, 0, skipped + output.len().min(target.len()));
    }
    let my = sy / n as f64;
    let mut tss = 0.0;
    for (o, t) in output.iter().zip(target.iter()) {
        if o.is_finite() && t.is_finite() {
            let d = a * o + b - t;
            se += d * d;
            tss += (t - my) * (t - my);
        } else {
            skipped += 1;
        }
    }
    (se / n as f64, tss, n, skipped)
}

/// A candidate scored on its fitted readout.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedScore {
    /// Spec §8 score: ΔMSE vs the parent minus the complexity charge.
    pub score: f64,
    /// Dev MSE of the fitted readout `a·out + b`.
    pub mse: f64,
    pub parent_mse: f64,
    /// Fitted readout coefficients, reported in the referee packet.
    pub scale: f64,
    pub intercept: f64,
    /// Readout R² on dev data (in-sample: prioritisation only, never evidence).
    pub r2: f64,
    /// Finite rows used for the fit.
    pub n_eff: usize,
    /// Rows skipped because the output or target was NaN (warmup, gaps).
    pub skipped: usize,
    /// Charged complexity: edges + [`READOUT_PARAMS`].
    pub complexity: f64,
}

/// Fit the readout of a graph and return `(mse, scale, intercept)`.
pub fn fitted_mse(graph: &Graph, cols: &[Vec<f64>], target: &[f64], ridge: f64) -> (f64, f64, f64) {
    let out = graph.evaluate(cols);
    let (a, b) = fit_readout(&out, target, ridge);
    let (mse, _, _, _) = residual_stats(&out, target, a, b);
    (mse, a, b)
}

/// Score one candidate against its parent's dev MSE, on the fitted readout.
pub fn score_candidate_fitted(
    graph: &Graph,
    parent_dev_mse: f64,
    cols: &[Vec<f64>],
    target: &[f64],
    cfg: &ScoreConfig,
) -> FittedScore {
    let out = graph.evaluate(cols);
    let (a, b) = fit_readout(&out, target, cfg.ridge);
    let (mse, tss, n_eff, skipped) = residual_stats(&out, target, a, b);
    let complexity = charged_dof(graph, cfg.selection_dof);
    let score = (parent_dev_mse - mse) - cfg.lambda_c * complexity;
    let r2 = if tss > 0.0 { 1.0 - mse * n_eff as f64 / tss } else { f64::NAN };
    FittedScore { score, mse, parent_mse: parent_dev_mse, scale: a, intercept: b, r2, n_eff, skipped, complexity }
}

/// Parallel batch scoring on fitted readouts. Returns scores in input order
/// (deterministic despite parallelism).
pub fn score_batch_fitted(
    jobs: &[(Graph, f64, Vec<f64>)],
    cols: &[Vec<f64>],
    cfg: &ScoreConfig,
) -> Vec<FittedScore> {
    jobs.par_iter()
        .map(|(g, parent_mse, target)| score_candidate_fitted(g, *parent_mse, cols, target, cfg))
        .collect()
}

/// Raw-output scoring (no readout fitted). Diagnostic only: it cannot see a
/// planted mechanism whose amplitude or offset differs from the target's, so it
/// must never be used to decide promotion. Kept so audits can reproduce the
/// pre-v0.1.1 numbers.
pub fn score_candidate_raw(
    graph: &Graph,
    parent_dev_mse: f64,
    cols: &[Vec<f64>],
    target: &[f64],
    lambda_c: f64,
) -> (f64, f64, usize) {
    let out = graph.evaluate(cols);
    let (mse, skipped) = mse_finite(&out, target);
    let score = (parent_dev_mse - mse) - lambda_c * graph.edge_count() as f64;
    (score, mse, skipped)
}

/// Parallel batch scoring on raw outputs (diagnostic; see [`score_candidate_raw`]).
pub fn score_batch_raw(
    jobs: &[(Graph, f64, Vec<f64>)],
    cols: &[Vec<f64>],
    lambda_c: f64,
) -> Vec<(f64, f64, usize)> {
    jobs.par_iter()
        .map(|(g, parent_mse, target)| score_candidate_raw(g, *parent_mse, cols, target, lambda_c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::graph::Node;
    use super::*;

    fn cols_target() -> (Vec<Vec<f64>>, Vec<f64>) {
        // target = x[t-1] + y[t]: needs a lag + add; tests ground-truth recovery path
        let cols = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![0.5, 1.0, 1.5, 2.0, 2.5]];
        let tgt = vec![f64::NAN, f64::NAN, 3.5, 5.0, 6.5];
        (cols, tgt)
    }

    #[test]
    fn perfect_graph_zero_mse() {
        let (cols, tgt) = cols_target();
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let y = g.push(Node::Input(1));
        let xl = g.push(Node::Lag(x, 1));
        g.output = g.push(Node::Add(xl, y));
        let (mse, skipped) = mse_finite(&g.evaluate(&cols), &tgt);
        assert!(mse < 1e-12);
        assert_eq!(skipped, 2); // warmup rows skipped, counted
    }

    #[test]
    fn fit_readout_recovers_scale_and_intercept() {
        // Mean-zero regressor, so the offset in the target is exactly the
        // fitted intercept.
        let out: Vec<f64> = (0..40).map(|t| t as f64 - 19.5).collect();
        let tgt: Vec<f64> = out.iter().map(|o| 2.5 * o + 7.0).collect();
        let (a, b) = fit_readout(&out, &tgt, 0.0);
        assert!((a - 2.5).abs() < 1e-9, "a = {}", a);
        assert!((b - 7.0).abs() < 1e-9, "b = {}", b);

        // NaN rows are excluded from the fit.
        let mut out_nan = out.clone();
        out_nan[0] = f64::NAN;
        let (a2, _) = fit_readout(&out_nan, &tgt, 0.0);
        assert!((a2 - 2.5).abs() < 1e-9, "a = {}", a2);
    }

    #[test]
    fn ridge_shrinks_the_slope_and_never_amplifies() {
        let out: Vec<f64> = (0..20).map(|t| t as f64).collect();
        let tgt: Vec<f64> = out.iter().map(|o| 3.0 * o).collect();
        let (a0, _) = fit_readout(&out, &tgt, 0.0);
        let (a1, _) = fit_readout(&out, &tgt, 1e6);
        assert!((a0 - 3.0).abs() < 1e-9);
        assert!(a1.abs() < a0.abs(), "ridge must shrink: {} vs {}", a1, a0);
    }

    #[test]
    fn constant_output_yields_intercept_only() {
        let out = vec![4.0; 10];
        let tgt = vec![0.0; 10];
        let (a, b) = fit_readout(&out, &tgt, 0.0);
        assert_eq!(a, 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn fitted_mse_beats_raw_mse_for_an_offset_mechanism() {
        let (cols, tgt) = cols_target();
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let xl = g.push(Node::Lag(x, 1));
        let y = g.push(Node::Input(1));
        g.output = g.push(Node::Add(xl, y));
        let shifted: Vec<f64> = tgt.iter().map(|t| 10.0 * t + 100.0).collect();
        let (raw, _) = mse_finite(&g.evaluate(&cols), &shifted);
        let (fitted, a, b) = fitted_mse(&g, &cols, &shifted, 1e-9);
        assert!(raw > 1e4, "raw MSE is dominated by the offset: {}", raw);
        assert!(fitted < 1e-12, "fitted MSE must vanish: {}", fitted);
        // The fit recovers the planted scale; the intercept absorbs 100 plus the
        // target mean (the graph output has mean 10).
        assert!((a - 10.0).abs() < 1e-6, "a={}", a);
        assert!(b.is_finite(), "b={}", b);
    }

    #[test]
    fn readout_parameters_are_charged() {
        let (cols, tgt) = cols_target();
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = x;
        let s0 = score_candidate_fitted(&g, 5.0, &cols, &tgt, &ScoreConfig::new(0.0, 1e-9));
        let s1 = score_candidate_fitted(&g, 5.0, &cols, &tgt, &ScoreConfig::new(1.0, 1e-9));
        let charged = g.edge_count() as f64 + READOUT_PARAMS;
        assert!((s0.score - s1.score - charged).abs() < 1e-9);
        assert_eq!(s0.complexity, charged);
    }

    #[test]
    fn selection_charge_is_paid_on_top_of_structure() {
        // A search over many candidates must pay for the multiplicity it created.
        let (cols, tgt) = cols_target();
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = x;
        let base = ScoreConfig::new(1.0, 1e-9);
        let searched = base.with_selection_dof(2.0 * 200f64.ln());
        let s_base = score_candidate_fitted(&g, 5.0, &cols, &tgt, &base);
        let s_search = score_candidate_fitted(&g, 5.0, &cols, &tgt, &searched);
        let gap = s_base.score - s_search.score;
        assert!((gap - 1.0 * searched.selection_dof).abs() < 1e-9, "gap {}", gap);
        assert_eq!(s_search.complexity, s_base.complexity + searched.selection_dof);
    }

    #[test]
    fn batch_order_deterministic() {
        let (cols, tgt) = cols_target();
        let mut g1 = Graph::new();
        let x = g1.push(Node::Input(0));
        g1.output = x;
        let mut g2 = Graph::new();
        let y = g2.push(Node::Input(1));
        g2.output = y;
        let jobs = vec![(g1, 10.0, tgt.clone()), (g2, 10.0, tgt.clone())];
        let cfg = ScoreConfig::new(0.01, 1e-9);
        let r1 = score_batch_fitted(&jobs, &cols, &cfg);
        let r2 = score_batch_fitted(&jobs, &cols, &cfg);
        assert_eq!(r1.len(), 2);
        assert!(r1[0].score.is_finite() && r1[1].score.is_finite());
        assert_eq!(r1, r2);
    }

    #[test]
    fn raw_scoring_is_still_available_for_audits() {
        let (cols, tgt) = cols_target();
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = x;
        let (score, mse, _) = score_candidate_raw(&g, 5.0, &cols, &tgt, 1.0);
        assert!((score - (5.0 - mse - g.edge_count() as f64)).abs() < 1e-12);
    }
}
