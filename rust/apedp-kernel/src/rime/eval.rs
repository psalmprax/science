//! RIME batch evaluator: parallel candidate scoring (spec §8, §12).
//!
//! Score (frozen): S(G') = mean dev-MSE reduction vs parent − λc·|E'|.
//! Ridge fitting of a final linear readout is done by the caller (kernel
//! numeric / Python side); this module evaluates raw graph outputs in
//! parallel and computes MSE against targets.

use rayon::prelude::*;

use super::graph::Graph;

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

/// Score one candidate against its parent's dev MSE.
pub fn score_candidate(
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

/// Parallel batch scoring: one (graph, target) pair per job, shared inputs.
/// Returns scores in input order (deterministic despite parallelism).
pub fn score_batch(
    jobs: &[(Graph, f64, Vec<f64>)],
    cols: &[Vec<f64>],
    lambda_c: f64,
) -> Vec<(f64, f64, usize)> {
    jobs.par_iter()
        .map(|(g, parent_mse, target)| score_candidate(g, *parent_mse, cols, target, lambda_c))
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
    fn batch_order_deterministic() {
        let (cols, tgt) = cols_target();
        let mut g1 = Graph::new();
        let x = g1.push(Node::Input(0));
        g1.output = x;
        let mut g2 = Graph::new();
        let y = g2.push(Node::Input(1));
        g2.output = y;
        let jobs = vec![(g1, 10.0, tgt.clone()), (g2, 10.0, tgt.clone())];
        let r1 = score_batch(&jobs, &cols, 0.01);
        let r2 = score_batch(&jobs, &cols, 0.01);
        assert_eq!(r1.len(), 2);
        assert!(r1[0].0.is_finite() && r1[1].0.is_finite());
        assert_eq!(format!("{:?}", r1), format!("{:?}", r2));
    }

    #[test]
    fn complexity_penalty_applies() {
        let (cols, tgt) = cols_target();
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = x;
        let (s0, _, _) = score_candidate(&g, 5.0, &cols, &tgt, 0.0);
        let (s1, _, _) = score_candidate(&g, 5.0, &cols, &tgt, 1.0);
        assert!((s0 - s1 - g.edge_count() as f64) < 1e-9);
    }
}
