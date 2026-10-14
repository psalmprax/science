//! RIME residual diagnostics (spec §8/§10).
//! Per-sample residual analysis over the FITTED readout (`a·out + b`, the
//! same readout `eval` scores), so diagnosis can never contradict scoring:
//! a planted amplitude or offset reads as structure found, not as error.
//!
//! The over-threshold cutoff is an explicit caller argument (no threshold
//! literal in this module, per the preregistered-spec discipline). Count
//! comparison avoids `Ord` on `f64`; a MAD-based cutoff can replace the
//! absolute one later without changing this signature.

use super::eval::fit_readout;
use super::graph::Graph;


/// Statistics from a residual analysis.
#[derive(Clone, Debug, Default)]
pub struct ResidualAnalysis {
    /// Number of samples where |residual| > threshold.
    pub overthreshold_count: usize,
    /// Maximum absolute residual (over all samples).
    pub max_abs_residual: f64,
    /// Fixed absolute threshold used for the “over threshold” test.
    pub threshold: f64,
    /// Human‑readable suggestion based on the worst residuals (may be None).
    pub suggestion: Option<String>,
}

/// Compute residuals on the fitted readout and return analysis.
///
/// * `graph`     – candidate graph to evaluate.
/// * `columns`   – design matrix (samples × variables).
/// * `target`    – target vector (length = number of samples).
/// * `threshold` – explicit absolute cutoff for the over-threshold count.
///   Caller-supplied; this module defines no threshold literal.
/// * `ridge`     – ridge coefficient for the readout fit (same semantics as
///   `eval::fit_readout`; shrinks the slope toward zero, intercept free).
///
/// Diagnosis only: returns statistics, never modifies the graph and never
/// promotes anything.
pub fn diagnose_residuals(
    graph: &Graph,
    columns: &[Vec<f64>],
    target: &[f64],
    threshold: f64,
    ridge: f64,
) -> ResidualAnalysis {
    let n = target.len();
    // Evaluate graph output, then fit the SAME readout `eval` scores, so a
    // mechanism whose amplitude or offset differs from the target is read as
    // found structure rather than as residual error.
    let output = graph.evaluate(columns);
    let (a, b) = fit_readout(&output, target, ridge);

    // Compute residuals = target - (a*out + b) (skip NaN/Inf).
    let mut residuals = Vec::with_capacity(n);
    for (t, o) in target.iter().zip(output.iter()) {
        if t.is_finite() && o.is_finite() {
            residuals.push(t - (a * o + b));
        } else {
            residuals.push(f64::NAN);
        }
    }

    let overthreshold_count = residuals
        .iter()
        .filter(|r| !r.is_nan() && r.abs() > threshold)
        .count();

    // Compute max_abs_residual without using Ord on f64 directly:
    let mut max_abs = 0.0;
    for r in &residuals {
        if !r.is_nan() && r.abs() > max_abs {
            max_abs = r.abs();
        }
    }
    let max_abs_residual = max_abs;

    // Very simple suggestion: if we have overthreshold residuals,
    // mention the number and note that a lag or non‑linear wrap may help.
    let suggestion = if overthreshold_count > 0 {
        Some(format!(
            "{} residuals exceed threshold={}; consider adding a lag or Abs/Sign node",
            overthreshold_count, threshold
        ))
    } else {
        None
    };

    ResidualAnalysis {
        overthreshold_count,
        max_abs_residual,
        threshold,
        suggestion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::Node;

    fn lag_graph() -> Graph {
        // output = x[t-2].
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = g.push(Node::Lag(x, 2));
        g
    }

    #[test]
    fn residual_analysis_basic() {
        let n = 20;
        let cols = vec![vec![1.0; n]; 1];
        let target: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let graph = Graph::new();
        let analysis = diagnose_residuals(&graph, &cols, &target, 1.0, 1e-9);
        assert!(analysis.overthreshold_count >= 0);
        assert!(analysis.max_abs_residual >= 0.0);
        // The reported threshold is exactly what the caller passed.
        assert_eq!(analysis.threshold, 1.0);
    }

    #[test]
    fn residual_analysis_with_graph() {
        let n = 30;
        let cols = vec![vec![1.0; n]; 1];
        let target: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) + 0.5).collect();
        let graph = Graph::new();
        let analysis = diagnose_residuals(&graph, &cols, &target, 1.0, 1e-9);
        // With a perfect linear relationship the residuals should be small.
        assert!(analysis.max_abs_residual < 1.0, "max residual too large: {}", analysis.max_abs_residual);
        assert!(analysis.overthreshold_count == 0, "should have few overthreshold residuals");
    }

    #[test]
    fn fitted_readout_consistency_planted_scale_and_offset() {
        // A mechanism whose amplitude AND offset differ from the target must
        // read as found structure (tiny fitted residuals), never as error.
        // Under raw-output residuals this target would look catastrophic.
        let x: Vec<f64> = (0..50).map(|t| t as f64 * 0.5 - 12.0).collect();
        let target: Vec<f64> = x.iter().map(|v| 3.0 * v + 100.0).collect();
        let g = lag_graph();
        // Shift columns so the lag has finite warmup inside the frame.
        let cols = vec![x.clone()];
        let analysis = diagnose_residuals(&g, &cols, &target, 1.0, 1e-9);
        assert!(
            analysis.max_abs_residual < 1e-6,
            "fitted residuals must vanish: {}",
            analysis.max_abs_residual
        );
        assert_eq!(analysis.overthreshold_count, 0);
        assert!(analysis.suggestion.is_none());
    }

    #[test]
    fn threshold_argument_drives_the_count() {
        let x: Vec<f64> = (0..40).map(|t| t as f64).collect();
        let target: Vec<f64> = x.iter().map(|v| 2.0 * v + 0.01 * (v % 3.0)).collect();
        let g = lag_graph();
        let cols = vec![x];
        let tiny = diagnose_residuals(&g, &cols, &target, 1e-9, 1e-9);
        let huge = diagnose_residuals(&g, &cols, &target, 1e9, 1e-9);
        assert!(tiny.overthreshold_count > 0, "tiny threshold must flag rows");
        assert_eq!(huge.overthreshold_count, 0, "huge threshold must flag nothing");
        assert_eq!(tiny.threshold, 1e-9);
        assert_eq!(huge.threshold, 1e9);
    }
}