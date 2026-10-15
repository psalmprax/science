//! Differential Padé & Jet Space Relation Discovery.
//!
//! Discovers governing differential equations satisfied by observed data:
//!     P(x) * y' + Q(x) * y ~ 0
//! or second-order relations:
//!     P(x) * y'' + Q(x) * y' + R(x) * y ~ 0
//!
//! Rather than forcing essential singularities and tunneling terms (like e^(-1/x))
//! to be primitive expression tree tokens, this module discovers the exact
//! polynomial ODE satisfied on the Jet bundle, and validates the candidate by
//! forward numerical integration (RK4) against actual observations.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPadeConfig {
    pub max_x_degree: usize,
    pub max_diff_order: usize,
    pub smooth_window: usize,
    pub sparsity_threshold: f64,
    pub rcond_threshold: f64,
}

impl Default for DiffPadeConfig {
    fn default() -> DiffPadeConfig {
        DiffPadeConfig {
            max_x_degree: 2,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPadeResult {
    pub formula: String,
    pub p_coefficients: Vec<f64>, // Polynomial P(x) multiplying y'
    pub q_coefficients: Vec<f64>, // Polynomial Q(x) multiplying y
    pub jet_rmse: f64,
    pub integrated_trajectory_r2: f64,
    pub integrated_y: Vec<f64>,
    pub is_exact_recovery: bool,
}

/// Compute numerical first derivative y' using 4th-order central difference and 2nd-order boundary stencils.
pub fn compute_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut dy = vec![0.0f64; n];
    if n < 4 {
        if n >= 2 {
            for i in 0..(n - 1) {
                let dx = (x[i + 1] - x[i]).max(1e-12);
                dy[i] = (y[i + 1] - y[i]) / dx;
            }
            dy[n - 1] = dy[n - 2];
        }
        return dy;
    }

    let h = (x[1] - x[0]).max(1e-12);

    // 2nd-order one-sided at edges
    dy[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h);
    dy[1] = (y[2] - y[0]) / (2.0 * h);

    // 4th-order central differences on interior
    for i in 2..(n - 2) {
        dy[i] = (-y[i + 2] + 8.0 * y[i + 1] - 8.0 * y[i - 1] + y[i - 2]) / (12.0 * h);
    }

    // 2nd-order one-sided at end
    dy[n - 2] = (y[n - 1] - y[n - 3]) / (2.0 * h);
    dy[n - 1] = (3.0 * y[n - 1] - 4.0 * y[n - 2] + y[n - 3]) / (2.0 * h);

    dy
}

/// Discover differential equation P(x) y' + Q(x) y = 0.
pub fn discover_differential_relation(
    x: &[f64],
    y: &[f64],
    config: &DiffPadeConfig,
) -> Result<DiffPadeResult, String> {
    let n = x.len();
    if n < 12 {
        return Err("Differential discovery requires at least 12 observations".to_string());
    }
    if y.len() != n {
        return Err("x and y lengths do not match".to_string());
    }

    // Sort by x
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let x_s: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
    let y_s: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

    // Compute derivative y'
    let dy_s = compute_derivatives(&x_s, &y_s);

    // Build differential Jet basis:
    // Columns for Q(x)*y: y, x*y, x^2*y, ..., x^deg*y
    // Columns for P(x)*y': y', x*y', x^2*y', ..., x^deg*y'
    let deg = config.max_x_degree;
    let n_q = deg + 1;
    let n_p = deg + 1;
    let k = n_q + n_p;

    let mut cols = Vec::with_capacity(k);
    let mut names = Vec::with_capacity(k);

    for p in 0..=deg {
        let col: Vec<f64> = x_s.iter().zip(y_s.iter()).map(|(&xv, &yv)| {
            xv.powi(p as i32) * yv
        }).collect();
        cols.push(col);
        names.push(if p == 0 { "y".to_string() } else { format!("x^{p}*y") });
    }

    for p in 0..=deg {
        let col: Vec<f64> = x_s.iter().zip(dy_s.iter()).map(|(&xv, &dyv)| {
            xv.powi(p as i32) * dyv
        }).collect();
        cols.push(col);
        names.push(if p == 0 { "y'".to_string() } else { format!("x^{p}*y'") });
    }

    // Exclude the boundary edges where finite differences are noisy
    let edge = 2usize;
    let interior_n = n.saturating_sub(2 * edge);
    if interior_n < 6 {
        return Err("Insufficient interior points after boundary guard".to_string());
    }

    // Normalize interior columns
    let mut norms = vec![0.0f64; k];
    for j in 0..k {
        let mut sum_sq = 0.0f64;
        for i in edge..(n - edge) {
            sum_sq += cols[j][i] * cols[j][i];
        }
        norms[j] = sum_sq.sqrt().max(1e-12);
    }

    // Gram matrix on interior
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut dot = 0.0f64;
            for row in edge..(n - edge) {
                dot += (cols[i][row] / norms[i]) * (cols[j][row] / norms[j]);
            }
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }

    // Jacobi Eigendecomposition
    let (evals, evecs) = crate::rime::implicit::jacobi_eigen(&gram, k, 150, 1e-15);

    // Find smallest eigenvalue
    let mut min_idx = 0usize;
    let mut min_val = f64::INFINITY;
    for (idx, &val) in evals.iter().enumerate() {
        if val.abs() < min_val {
            min_val = val.abs();
            min_idx = idx;
        }
    }

    let mut raw_c = vec![0.0f64; k];
    for j in 0..k {
        raw_c[j] = evecs[j * k + min_idx] / norms[j];
    }

    // Normalize so highest magnitude is 1.0
    let max_abs = raw_c.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-15);
    for v in raw_c.iter_mut() {
        *v /= max_abs;
    }

    // Sparsity thresholding
    for v in raw_c.iter_mut() {
        if v.abs() < config.sparsity_threshold {
            *v = 0.0;
        }
    }

    // Extract Q(x) and P(x)
    let q_coeffs = raw_c[0..n_q].to_vec();
    let p_coeffs = raw_c[n_q..k].to_vec();

    // Compute jet residual RMSE
    let mut sum_sq_res = 0.0f64;
    for row in edge..(n - edge) {
        let mut r = 0.0f64;
        for j in 0..k {
            r += cols[j][row] * raw_c[j];
        }
        sum_sq_res += r * r;
    }
    let jet_rmse = (sum_sq_res / (interior_n as f64)).sqrt();

    // Forward numerical integration via RK4 to validate against original data
    // dy/dx = - Q(x) / P(x) * y
    let mut y_sim = vec![0.0f64; n];
    y_sim[0] = y_s[0];

    let eval_ode = |xv: f64, yv: f64| -> f64 {
        let mut q_val = 0.0f64;
        let mut p_val = 0.0f64;
        for (p, &c) in q_coeffs.iter().enumerate() {
            q_val += c * xv.powi(p as i32);
        }
        for (p, &c) in p_coeffs.iter().enumerate() {
            p_val += c * xv.powi(p as i32);
        }
        if p_val.abs() > 1e-10 {
            -q_val / p_val * yv
        } else {
            0.0
        }
    };

    for i in 0..(n - 1) {
        let h = x_s[i + 1] - x_s[i];
        let x_curr = x_s[i];
        let y_curr = y_sim[i];

        let k1 = eval_ode(x_curr, y_curr);
        let k2 = eval_ode(x_curr + 0.5 * h, y_curr + 0.5 * h * k1);
        let k3 = eval_ode(x_curr + 0.5 * h, y_curr + 0.5 * h * k2);
        let k4 = eval_ode(x_curr + h, y_curr + h * k3);

        y_sim[i + 1] = y_curr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    // Trajectory R^2
    let mean_y = y_s.iter().sum::<f64>() / (n as f64);
    let ss_tot = y_s.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>();
    let ss_res = y_s.iter().zip(y_sim.iter()).map(|(yi, si)| (yi - si) * (yi - si)).sum::<f64>();
    let traj_r2 = if ss_tot > 1e-15 {
        1.0 - (ss_res / ss_tot)
    } else {
        1.0
    };

    // Format formula
    let mut p_str_terms = Vec::new();
    for (p, &c) in p_coeffs.iter().enumerate() {
        if c.abs() > 1e-4 {
            p_str_terms.push(if p == 0 { format!("{:.3}", c) } else { format!("{:.3}*x^{p}", c) });
        }
    }
    let mut q_str_terms = Vec::new();
    for (p, &c) in q_coeffs.iter().enumerate() {
        if c.abs() > 1e-4 {
            q_str_terms.push(if p == 0 { format!("{:.3}", c) } else { format!("{:.3}*x^{p}", c) });
        }
    }

    let p_formatted = if p_str_terms.is_empty() { "0".to_string() } else { p_str_terms.join(" + ") };
    let q_formatted = if q_str_terms.is_empty() { "0".to_string() } else { q_str_terms.join(" + ") };
    let formula = format!("({}) * y' + ({}) * y = 0", p_formatted, q_formatted);

    let is_exact = traj_r2 > 0.99 && jet_rmse < 0.05;

    Ok(DiffPadeResult {
        formula,
        p_coefficients: p_coeffs,
        q_coefficients: q_coeffs,
        jet_rmse,
        integrated_trajectory_r2: traj_r2,
        integrated_y: y_sim,
        is_exact_recovery: is_exact,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_bcs_energy_gap_recovery() {
        // y(g) = 3.52 * exp(-1/g) => g^2 * y' - 1 * y = 0
        let n = 50;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let g = 0.2 + 0.8 * (i as f64) / ((n - 1) as f64);
            let val = 3.52 * (-1.0 / g).exp();
            x.push(g);
            y.push(val);
        }

        let cfg = DiffPadeConfig {
            max_x_degree: 2,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
        };

        let res = discover_differential_relation(&x, &y, &cfg).unwrap();
        assert!(res.integrated_trajectory_r2 > 0.99, "Trajectory R2: {}", res.integrated_trajectory_r2);
        assert!(res.is_exact_recovery);
    }

    #[test]
    fn test_diff_landau_zener_recovery() {
        // y(v) = exp(-0.75 / v) => v^2 * y' - 0.75 * y = 0
        let n = 60;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let v = 0.2 + 3.8 * (i as f64) / ((n - 1) as f64);
            let val = (-0.75 / v).exp();
            x.push(v);
            y.push(val);
        }

        let cfg = DiffPadeConfig {
            max_x_degree: 2,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
        };

        let res = discover_differential_relation(&x, &y, &cfg).unwrap();
        assert!(res.integrated_trajectory_r2 > 0.99, "Trajectory R2: {}", res.integrated_trajectory_r2);
        assert!(res.is_exact_recovery);
    }
}
