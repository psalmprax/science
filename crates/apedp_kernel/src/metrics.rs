//! Vectorized numerical metrics, loss functions, and parallel scoring.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bytecode::CompiledExpression;
use crate::vm::ExpressionVM;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricResult {
    pub mse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub max_residual: f64,
    pub has_nan_or_inf: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoredCandidate {
    pub candidate_id: usize,
    pub metrics: MetricResult,
}

pub struct MetricsEngine;

impl MetricsEngine {
    pub fn compute_metrics(y_true: &[f64], y_pred: &[f64]) -> MetricResult {
        let n = y_true.len();
        if n == 0 || n != y_pred.len() {
            return MetricResult {
                mse: f64::NAN,
                mae: f64::NAN,
                r_squared: f64::NAN,
                max_residual: f64::NAN,
                has_nan_or_inf: true,
            };
        }

        let mut sum_sq_err = 0.0;
        let mut sum_abs_err = 0.0;
        let mut max_res = 0.0;
        let mut sum_y = 0.0;
        let mut has_nan_inf = false;

        for i in 0..n {
            let yt = y_true[i];
            let yp = y_pred[i];
            sum_y += yt;

            if !yp.is_finite() {
                has_nan_inf = true;
            } else {
                let err = (yt - yp).abs();
                sum_sq_err += err * err;
                sum_abs_err += err;
                if err > max_res {
                    max_res = err;
                }
            }
        }

        if has_nan_inf {
            return MetricResult {
                mse: f64::NAN,
                mae: f64::NAN,
                r_squared: f64::NAN,
                max_residual: f64::NAN,
                has_nan_or_inf: true,
            };
        }

        let mse = sum_sq_err / (n as f64);
        let mae = sum_abs_err / (n as f64);

        let mean_y = sum_y / (n as f64);
        let mut total_var = 0.0;
        for i in 0..n {
            let diff = y_true[i] - mean_y;
            total_var += diff * diff;
        }

        let r_squared = if total_var > 1e-15 {
            1.0 - (sum_sq_err / total_var)
        } else if mse < 1e-15 {
            1.0
        } else {
            0.0
        };

        MetricResult {
            mse,
            mae,
            r_squared,
            max_residual: max_res,
            has_nan_or_inf: false,
        }
    }

    /// Evaluates a pool of candidate expressions in parallel across Rayon threads.
    pub fn score_candidates_parallel(
        candidates: &[CompiledExpression],
        data: &[&[f64]],
        target: &[f64],
    ) -> Vec<ScoredCandidate> {
        let n = target.len();

        candidates
            .par_iter()
            .enumerate()
            .map(|(idx, expr)| {
                let mut vm = ExpressionVM::new();
                let mut pred_buf = vec![0.0; n];

                match vm.evaluate_batch(expr, data, &mut pred_buf) {
                    Ok(_) => {
                        let metrics = Self::compute_metrics(target, &pred_buf);
                        ScoredCandidate {
                            candidate_id: idx,
                            metrics,
                        }
                    }
                    Err(_) => ScoredCandidate {
                        candidate_id: idx,
                        metrics: MetricResult {
                            mse: f64::NAN,
                            mae: f64::NAN,
                            r_squared: f64::NAN,
                            max_residual: f64::NAN,
                            has_nan_or_inf: true,
                        },
                    },
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_metrics_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        let res = MetricsEngine::compute_metrics(&y_true, &y_pred);
        assert_eq!(res.mse, 0.0);
        assert_eq!(res.mae, 0.0);
        assert_eq!(res.r_squared, 1.0);
        assert!(!res.has_nan_or_inf);
    }

    #[test]
    fn test_compute_metrics_with_error() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.5, 2.0, 2.5, 4.0];
        let res = MetricsEngine::compute_metrics(&y_true, &y_pred);
        assert!((res.mse - 0.125).abs() < 1e-10);
        assert!((res.mae - 0.25).abs() < 1e-10);
        assert!(!res.has_nan_or_inf);
    }
}
