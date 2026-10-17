//! Active Cross-Regime Invariant Transfer & Extrapolation Guard Engine
//!
//! Provides:
//! 1. Invariant Risk Minimization (IRM) & Cross-Environment Retention scoring.
//! 2. Active Extrapolation Barrier Guard testing derivative bounds and singularity risks.
//! 3. Cross-Regime Generalization Index (CGI) ranking invariant physical laws over spurious correlations.

use serde::{Deserialize, Serialize};

/// An environment dataset containing features and targets under a specific regime/condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentData {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Detailed evaluation metrics for a hypothesis within a single environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentMetric {
    pub name: String,
    pub n_samples: usize,
    pub mse: f64,
    pub r2: f64,
    pub optimal_scale: f64,
}

/// Outcome of cross-regime invariant retention evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantTransferResult {
    pub candidate_name: String,
    pub formula: String,
    pub pooled_r2: f64,
    pub mean_env_r2: f64,
    pub worst_env_r2: f64,
    pub r2_dispersion_std: f64,
    pub irm_penalty: f64,
    pub cgi_score: f64,
    pub passed_retention_gate: bool,
    pub per_env_metrics: Vec<EnvironmentMetric>,
    pub extrapolation_safe: bool,
    pub extrapolation_verdict: String,
}

/// Evaluates a candidate model function `f(x)` across multiple environments.
pub fn evaluate_invariant_transfer<F>(
    candidate_name: &str,
    formula: &str,
    envs: &[EnvironmentData],
    f: F,
    min_worst_r2: f64,
    max_r2_dispersion: f64,
    extrap_margin_ratio: f64,
    max_gradient_ratio: f64,
) -> Option<InvariantTransferResult>
where
    F: Fn(f64) -> f64,
{
    if envs.len() < 2 {
        return None;
    }

    let mut per_env_metrics = Vec::with_capacity(envs.len());
    let mut env_r2s = Vec::with_capacity(envs.len());
    let mut env_scales = Vec::with_capacity(envs.len());
    let mut pooled_sq_err = 0.0;
    let mut pooled_var_y = 0.0;
    let mut _total_samples = 0;

    let mut global_min_x = f64::INFINITY;
    let mut global_max_x = f64::NEG_INFINITY;

    for env in envs {
        let n = env.x.len();
        if n < 4 || n != env.y.len() {
            return None;
        }
        _total_samples += n;

        let mean_y = env.y.iter().sum::<f64>() / n as f64;
        let var_y = env.y.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;
        let safe_var_y = if var_y < 1e-12 { 1e-12 } else { var_y };

        let mut sq_err = 0.0;
        let mut y_pred = Vec::with_capacity(n);
        let mut sum_pred_sq = 0.0;
        let mut sum_y_pred = 0.0;

        for i in 0..n {
            let xi = env.x[i];
            let yi = env.y[i];
            if xi < global_min_x { global_min_x = xi; }
            if xi > global_max_x { global_max_x = xi; }

            let pred = f(xi);
            if !pred.is_finite() {
                return None; // Singular/unstable candidate rejected immediately
            }
            y_pred.push(pred);
            sq_err += (yi - pred).powi(2);
            sum_pred_sq += pred * pred;
            sum_y_pred += yi * pred;
        }

        let mse = sq_err / n as f64;
        let r2 = 1.0 - (mse / safe_var_y);

        // Optimal environmental scaling factor alpha = (y^T y_hat) / (y_hat^T y_hat)
        let alpha = if sum_pred_sq > 1e-12 {
            sum_y_pred / sum_pred_sq
        } else {
            1.0
        };

        pooled_sq_err += sq_err;
        pooled_var_y += var_y * (n as f64);

        env_r2s.push(r2);
        env_scales.push(alpha);

        per_env_metrics.push(EnvironmentMetric {
            name: env.name.clone(),
            n_samples: n,
            mse,
            r2,
            optimal_scale: alpha,
        });
    }

    let pooled_r2 = 1.0 - (pooled_sq_err / (pooled_var_y.max(1e-12)));
    let num_envs = envs.len() as f64;
    let mean_env_r2 = env_r2s.iter().sum::<f64>() / num_envs;
    let worst_env_r2 = env_r2s.iter().copied().fold(f64::INFINITY, f64::min);

    let r2_var = env_r2s.iter().map(|&r| (r - mean_env_r2).powi(2)).sum::<f64>() / num_envs;
    let r2_std = r2_var.sqrt();

    // IRM penalty: variance of optimal environmental multipliers + variance of relative MSE
    let mean_scale = env_scales.iter().sum::<f64>() / num_envs;
    let scale_var = env_scales.iter().map(|&s| (s - mean_scale).powi(2)).sum::<f64>() / num_envs;
    let irm_penalty = scale_var + r2_var;

    // Cross-Regime Generalization Index (CGI)
    let cgi_score = if worst_env_r2 <= 0.0 {
        worst_env_r2
    } else {
        worst_env_r2 / (r2_std + 1e-4)
    };

    // Retention gate: worst-environment R2 must exceed threshold AND cross-env dispersion must be bounded
    let passed_retention_gate = worst_env_r2 >= min_worst_r2 && r2_std <= max_r2_dispersion;

    // Active Extrapolation Barrier Guard
    let domain_span = (global_max_x - global_min_x).max(1e-4);
    let extrap_margin = domain_span * extrap_margin_ratio;
    let (extrapolation_safe, extrapolation_verdict) = check_extrapolation_barrier(
        &f,
        global_min_x,
        global_max_x,
        extrap_margin,
        max_gradient_ratio,
    );

    Some(InvariantTransferResult {
        candidate_name: candidate_name.to_string(),
        formula: formula.to_string(),
        pooled_r2,
        mean_env_r2,
        worst_env_r2,
        r2_dispersion_std: r2_std,
        irm_penalty,
        cgi_score,
        passed_retention_gate,
        per_env_metrics,
        extrapolation_safe,
        extrapolation_verdict,
    })
}

/// Checks extrapolation stability outside [x_min, x_max].
pub fn check_extrapolation_barrier<F>(
    f: F,
    x_min: f64,
    x_max: f64,
    margin: f64,
    max_gradient_ratio: f64,
) -> (bool, String)
where
    F: Fn(f64) -> f64,
{
    let n_in = 50;
    let n_extrap = 25;
    let step_in = (x_max - x_min) / (n_in as f64);

    // Compute in-sample maximum gradient
    let mut max_in_grad: f64 = 0.0;
    for i in 0..n_in {
        let x1 = x_min + (i as f64) * step_in;
        let x2 = x1 + step_in * 0.1;
        let y1 = f(x1);
        let y2 = f(x2);
        if !y1.is_finite() || !y2.is_finite() {
            return (false, "In-sample non-finite value detected".to_string());
        }
        let grad = ((y2 - y1) / (step_in * 0.1)).abs();
        if grad > max_in_grad {
            max_in_grad = grad;
        }
    }
    let safe_in_grad = max_in_grad.max(1e-6);

    // Test left extrapolation: [x_min - margin, x_min]
    let step_extrap = margin / (n_extrap as f64);
    for i in 0..n_extrap {
        let x = (x_min - margin) + (i as f64) * step_extrap;
        let y = f(x);
        if !y.is_finite() {
            return (false, format!("Pole/Singularity in left extrapolation at x={:.4}", x));
        }
        let y_plus = f(x + step_extrap * 0.1);
        let grad = ((y_plus - y) / (step_extrap * 0.1)).abs();
        if grad / safe_in_grad > max_gradient_ratio {
            return (
                false,
                format!("Gradient explosion in left extrapolation (ratio {:.1} > {:.1})",
                        grad / safe_in_grad, max_gradient_ratio),
            );
        }
    }

    // Test right extrapolation: [x_max, x_max + margin]
    for i in 0..n_extrap {
        let x = x_max + (i as f64) * step_extrap;
        let y = f(x);
        if !y.is_finite() {
            return (false, format!("Pole/Singularity in right extrapolation at x={:.4}", x));
        }
        let y_plus = f(x + step_extrap * 0.1);
        let grad = ((y_plus - y) / (step_extrap * 0.1)).abs();
        if grad / safe_in_grad > max_gradient_ratio {
            return (
                false,
                format!("Gradient explosion in right extrapolation (ratio {:.1} > {:.1})",
                        grad / safe_in_grad, max_gradient_ratio),
            );
        }
    }

    (true, "Extrapolation stable and bounded".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariant_transfer_identifies_causal_law_and_rejects_spurious_correlations() {
        // Two environments under different distribution shifts:
        // True physical law: y = 2.0 * x
        // Spurious feature z:
        //   Env 1: z = 3.0 * x  => y = (2/3) * z (R2 = 1.0)
        //   Env 2: z = -1.5 * x => y = (-4/3) * z (sign flip!)

        let mut env1 = EnvironmentData {
            name: "Env1_Standard".to_string(),
            x: Vec::new(),
            y: Vec::new(),
        };
        let mut env2 = EnvironmentData {
            name: "Env2_Intervention".to_string(),
            x: Vec::new(),
            y: Vec::new(),
        };

        for i in 0..30 {
            let x = 1.0 + 0.2 * (i as f64);
            env1.x.push(x);
            env1.y.push(2.0 * x); // True target
        }
        for i in 0..30 {
            let x = 10.0 + 0.5 * (i as f64);
            env2.x.push(x);
            env2.y.push(2.0 * x); // True target invariant across domains
        }

        let envs = vec![env1, env2];

        // 1. True invariant candidate: f(x) = 2.0 * x
        let true_res = evaluate_invariant_transfer(
            "TrueInvariantLaw",
            "y = 2.0 * x",
            &envs,
            |x| 2.0 * x,
            0.80,
            0.10,
            0.50,
            10.0,
        ).expect("Evaluation should succeed");

        assert!(true_res.passed_retention_gate);
        assert!(true_res.worst_env_r2 > 0.999);
        assert!(true_res.r2_dispersion_std < 1e-4);
        assert!(true_res.irm_penalty < 1e-6);
        assert!(true_res.extrapolation_safe);

        // 2. Spurious candidate that overfits Env 1 but fails Env 2
        // Say f_spurious(x) = 0.5 * x + 1.5 (fits Env 1 around mean x=4, collapses on Env 2 x in [10, 25])
        let spurious_res = evaluate_invariant_transfer(
            "SpuriousFit",
            "y = 0.5 * x + 1.5",
            &envs,
            |x| 0.5 * x + 1.5,
            0.80,
            0.10,
            0.50,
            10.0,
        ).expect("Evaluation should succeed");

        assert!(!spurious_res.passed_retention_gate, "Spurious candidate must be rejected by retention gate");
        assert!(spurious_res.worst_env_r2 < 0.0);
    }

    #[test]
    fn test_extrapolation_barrier_detects_runge_explosion() {
        // High-degree polynomial: f(x) = x^8 - 100 x^6
        let (safe, reason) = check_extrapolation_barrier(
            |x| x.powi(8) - 100.0 * x.powi(6),
            -1.0,
            1.0,
            1.0,
            10.0,
        );
        assert!(!safe);
        assert!(reason.contains("Gradient explosion"));
    }

    #[test]
    fn test_extrapolation_barrier_detects_pole() {
        // Rational function with pole right outside domain at x = 2.2
        let (safe, reason) = check_extrapolation_barrier(
            |x| 1.0 / (2.2 - x),
            0.0,
            2.0,
            0.5,
            10.0,
        );
        assert!(!safe);
        assert!(reason.contains("Pole/Singularity") || reason.contains("Gradient explosion"));
    }
}
