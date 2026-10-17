//! Spatio-Temporal Partial Differential Equation (PDE) Discovery in 1D and 2D.
//!
//! Extends RIME from 1D ordinary differential relations to continuous
//! spatio-temporal field equations:
//!     u_t = F(u, u_x, u_xx, u_xxx, u*u_x, ...)
//! and 2D spatial fields:
//!     u_t = F(u, u_x, u_y, u_xx, u_yy, \nabla^2 u, u*u_x, u*u_y, ...)
//!
//! Uses high-order finite difference stencils (both uniform 4th-order and
//! Fornberg arbitrary non-uniform weights) and Sequentially Thresholded
//! Ridge Regression (STRidge) to discover exact governing PDEs.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdeConfig {
    pub max_spatial_order: usize,      // 1, 2, or 3 (u_x, u_xx, u_xxx)
    pub max_polynomial_degree: usize,  // 1, 2, or 3 (u, u^2, u^3)
    pub include_advection_terms: bool, // u*u_x, u^2*u_x, u*u_y
    pub sparsity_threshold: f64,
    pub ridge_alpha: f64,
}

impl Default for PdeConfig {
    fn default() -> PdeConfig {
        PdeConfig {
            max_spatial_order: 2,
            max_polynomial_degree: 2,
            include_advection_terms: true,
            sparsity_threshold: 0.01,
            ridge_alpha: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdeResult {
    pub formula: String,
    pub active_terms: Vec<String>,
    pub coefficients: Vec<f64>,
    pub residual_rmse: f64,
    pub r2: f64,
    pub is_exact_recovery: bool,
}

/// Gaussian elimination with partial pivoting for small K x K linear systems.
pub fn solve_linear_system(a: &[f64], b: &[f64], k: usize) -> Result<Vec<f64>, String> {
    let mut mat = vec![vec![0.0f64; k + 1]; k];
    for i in 0..k {
        for j in 0..k {
            mat[i][j] = a[i * k + j];
        }
        mat[i][k] = b[i];
    }

    for i in 0..k {
        let mut max_row = i;
        let mut max_val = mat[i][i].abs();
        for r in (i + 1)..k {
            if mat[r][i].abs() > max_val {
                max_val = mat[r][i].abs();
                max_row = r;
            }
        }
        if max_val < 1e-15 {
            return Err("Singular matrix in linear system solver".to_string());
        }
        mat.swap(i, max_row);

        let pivot = mat[i][i];
        for c in i..=k {
            mat[i][c] /= pivot;
        }

        for r in 0..k {
            if r != i {
                let factor = mat[r][i];
                for c in i..=k {
                    mat[r][c] -= factor * mat[i][c];
                }
            }
        }
    }

    let mut x = vec![0.0f64; k];
    for i in 0..k {
        x[i] = mat[i][k];
    }
    Ok(x)
}

/// Fornberg's algorithm (1988) for calculating exact finite difference weights
/// on arbitrarily spaced grids for derivatives up to order `m_max`.
pub fn fornberg_weights(x: &[f64], z: f64, m_max: usize) -> Vec<Vec<f64>> {
    let n = x.len() - 1;
    let mut c = vec![vec![vec![0.0f64; n + 1]; n + 1]; m_max + 1];
    c[0][0][0] = 1.0;
    let mut c1 = 1.0;

    for i in 1..=n {
        let mut c2 = 1.0;
        for j in 0..i {
            let c3 = x[i] - x[j];
            c2 *= c3;
            for m in 0..=m_max.min(i) {
                let term1 = (x[i] - z) * c[m][i - 1][j];
                let term2 = if m > 0 { (m as f64) * c[m - 1][i - 1][j] } else { 0.0 };
                c[m][i][j] = (term1 - term2) / c3;
            }
        }
        for m in 0..=m_max.min(i) {
            let term1 = (x[i - 1] - z) * c[m][i - 1][i - 1];
            let term2 = if m > 0 { (m as f64) * c[m - 1][i - 1][i - 1] } else { 0.0 };
            c[m][i][i] = (c1 / c2) * (term2 - term1);
        }
        c1 = c2;
    }

    let mut res = vec![vec![0.0f64; n + 1]; m_max + 1];
    for m in 0..=m_max {
        for j in 0..=n {
            res[m][j] = c[m][n][j];
        }
    }
    res
}

/// Generic Sequentially Thresholded Ridge Regression (STRidge) solver.
pub fn run_stridge(
    target: &[f64],
    columns: &[Vec<f64>],
    term_names: &[String],
    config: &PdeConfig,
    target_label: &str,
) -> Result<PdeResult, String> {
    let num_samples = target.len();
    let k = columns.len();
    if k == 0 || num_samples == 0 {
        return Err("Empty data for STRidge".to_string());
    }

    let mut active = vec![true; k];
    let mut coeffs = vec![0.0f64; k];
    let max_iter = 20;

    for _iter in 0..max_iter {
        let active_indices: Vec<usize> = (0..k).filter(|&i| active[i]).collect();
        let k_act = active_indices.len();
        if k_act == 0 {
            break;
        }

        let mut a_mat = vec![0.0f64; k_act * k_act];
        let mut b_vec = vec![0.0f64; k_act];

        for i in 0..k_act {
            let col_i = &columns[active_indices[i]];
            let mut dot_b = 0.0f64;
            for s in 0..num_samples {
                dot_b += col_i[s] * target[s];
            }
            b_vec[i] = dot_b;

            for j in i..k_act {
                let col_j = &columns[active_indices[j]];
                let mut dot_a = 0.0f64;
                for s in 0..num_samples {
                    dot_a += col_i[s] * col_j[s];
                }
                if i == j {
                    dot_a += config.ridge_alpha;
                }
                a_mat[i * k_act + j] = dot_a;
                a_mat[j * k_act + i] = dot_a;
            }
        }

        let sol = match solve_linear_system(&a_mat, &b_vec, k_act) {
            Ok(s) => s,
            Err(_) => break,
        };

        for (idx, &act_i) in active_indices.iter().enumerate() {
            coeffs[act_i] = sol[idx];
        }

        let mut changed = false;
        for &act_i in &active_indices {
            if coeffs[act_i].abs() < config.sparsity_threshold {
                active[act_i] = false;
                coeffs[act_i] = 0.0;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let mut ss_res = 0.0f64;
    let mean_target = target.iter().sum::<f64>() / (num_samples as f64);
    let mut ss_tot = 0.0f64;

    for s in 0..num_samples {
        let mut pred = 0.0f64;
        for j in 0..k {
            if coeffs[j] != 0.0 {
                pred += coeffs[j] * columns[j][s];
            }
        }
        let diff = target[s] - pred;
        ss_res += diff * diff;
        let d_mean = target[s] - mean_target;
        ss_tot += d_mean * d_mean;
    }

    let residual_rmse = (ss_res / (num_samples as f64)).sqrt();
    let r2 = if ss_tot > 1e-15 {
        1.0 - (ss_res / ss_tot)
    } else {
        1.0
    };

    let mut active_terms = Vec::new();
    let mut active_coeffs = Vec::new();
    let mut formula_terms = Vec::new();

    for j in 0..k {
        if coeffs[j].abs() >= config.sparsity_threshold {
            active_terms.push(term_names[j].clone());
            active_coeffs.push(coeffs[j]);
            formula_terms.push(format!("{:.4} * ({})", coeffs[j], term_names[j]));
        }
    }

    let rhs = if formula_terms.is_empty() {
        "0.0000".to_string()
    } else {
        formula_terms.join(" + ")
    };
    let formula = format!("{} = {}", target_label, rhs);
    let is_exact = r2 > 0.99 && residual_rmse < 0.05 && !active_terms.is_empty();

    Ok(PdeResult {
        formula,
        active_terms,
        coefficients: active_coeffs,
        residual_rmse,
        r2,
        is_exact_recovery: is_exact,
    })
}

/// Discover governing PDE from 1D spatio-temporal observation grid u(t, x).
pub fn discover_pde_1d(
    u_grid: &[Vec<f64>],
    dt: f64,
    dx: f64,
    config: &PdeConfig,
) -> Result<PdeResult, String> {
    let m = u_grid.len();
    if m < 5 {
        return Err("PDE discovery requires at least 5 time steps".to_string());
    }
    let n = u_grid[0].len();
    if n < 7 {
        return Err("PDE discovery requires at least 7 spatial points".to_string());
    }

    let dt = dt.max(1e-12);
    let dx = dx.max(1e-12);

    let t_interior = m - 2;
    let x_interior = n - 4;
    let num_samples = t_interior * x_interior;

    let mut ut = Vec::with_capacity(num_samples);
    let mut u_vals = Vec::with_capacity(num_samples);
    let mut ux = Vec::with_capacity(num_samples);
    let mut uxx = Vec::with_capacity(num_samples);
    let mut uxxx = Vec::with_capacity(num_samples);

    for i in 1..(m - 1) {
        for j in 2..(n - 2) {
            let d_t = (u_grid[i + 1][j] - u_grid[i - 1][j]) / (2.0 * dt);
            ut.push(d_t);
            u_vals.push(u_grid[i][j]);

            let d_x = (-u_grid[i][j + 2] + 8.0 * u_grid[i][j + 1] - 8.0 * u_grid[i][j - 1] + u_grid[i][j - 2])
                / (12.0 * dx);
            ux.push(d_x);

            let d_xx = (-u_grid[i][j + 2] + 16.0 * u_grid[i][j + 1] - 30.0 * u_grid[i][j]
                + 16.0 * u_grid[i][j - 1] - u_grid[i][j - 2])
                / (12.0 * dx * dx);
            uxx.push(d_xx);

            if config.max_spatial_order >= 3 {
                let d_xxx = (u_grid[i][j + 2] - 2.0 * u_grid[i][j + 1] + 2.0 * u_grid[i][j - 1] - u_grid[i][j - 2])
                    / (2.0 * dx.powi(3));
                uxxx.push(d_xxx);
            }
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }
    if config.max_polynomial_degree >= 3 {
        term_names.push("u^3".to_string());
        columns.push(u_vals.iter().map(|&v| v.powi(3)).collect());
    }

    term_names.push("u_x".to_string());
    columns.push(ux.clone());

    if config.include_advection_terms {
        term_names.push("u*u_x".to_string());
        columns.push(u_vals.iter().zip(ux.iter()).map(|(&u, &ux)| u * ux).collect());

        if config.max_polynomial_degree >= 2 {
            term_names.push("u^2*u_x".to_string());
            columns.push(u_vals.iter().zip(ux.iter()).map(|(&u, &ux)| u * u * ux).collect());
        }
    }

    if config.max_spatial_order >= 2 {
        term_names.push("u_xx".to_string());
        columns.push(uxx.clone());

        if config.include_advection_terms {
            term_names.push("u*u_xx".to_string());
            columns.push(u_vals.iter().zip(uxx.iter()).map(|(&u, &uxx)| u * uxx).collect());
        }
    }

    if config.max_spatial_order >= 3 {
        term_names.push("u_xxx".to_string());
        columns.push(uxxx.clone());
    }

    run_stridge(&ut, &columns, &term_names, config, "u_t")
}

/// Discover governing PDE from non-uniform spatial grid `x_coords` and temporal grid `t_coords`.
pub fn discover_pde_nonuniform(
    u_grid: &[Vec<f64>],
    t_coords: &[f64],
    x_coords: &[f64],
    config: &PdeConfig,
) -> Result<PdeResult, String> {
    let m = u_grid.len();
    let n = x_coords.len();
    if m < 5 || n < 7 {
        return Err("Non-uniform PDE requires at least 5 time steps and 7 spatial points".to_string());
    }

    let mut ut = Vec::new();
    let mut u_vals = Vec::new();
    let mut ux = Vec::new();
    let mut uxx = Vec::new();

    for i in 1..(m - 1) {
        // Temporal stencil (3-point Fornberg on [t_{i-1}, t_i, t_{i+1}])
        let t_sub = [t_coords[i - 1], t_coords[i], t_coords[i + 1]];
        let w_t = fornberg_weights(&t_sub, t_coords[i], 1);

        for j in 2..(n - 2) {
            let d_t = w_t[1][0] * u_grid[i - 1][j] + w_t[1][1] * u_grid[i][j] + w_t[1][2] * u_grid[i + 1][j];
            ut.push(d_t);
            u_vals.push(u_grid[i][j]);

            // Spatial 5-point Fornberg stencil around x_j
            let x_sub = [
                x_coords[j - 2],
                x_coords[j - 1],
                x_coords[j],
                x_coords[j + 1],
                x_coords[j + 2],
            ];
            let w_x = fornberg_weights(&x_sub, x_coords[j], 2);

            let mut d_x = 0.0;
            let mut d_xx = 0.0;
            for k in 0..5 {
                let u_val = u_grid[i][j - 2 + k];
                d_x += w_x[1][k] * u_val;
                d_xx += w_x[2][k] * u_val;
            }
            ux.push(d_x);
            uxx.push(d_xx);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push("u_x".to_string());
    columns.push(ux.clone());

    if config.include_advection_terms {
        term_names.push("u*u_x".to_string());
        columns.push(u_vals.iter().zip(ux.iter()).map(|(&u, &ux)| u * ux).collect());
    }

    term_names.push("u_xx".to_string());
    columns.push(uxx.clone());

    run_stridge(&ut, &columns, &term_names, config, "u_t")
}

/// Discover 2D spatial spatio-temporal PDE: u_t = F(u, u_x, u_y, u_xx, u_yy, \nabla^2 u, ...).
/// `u_grid` has dimensions: [N_t][N_x][N_y].
pub fn discover_pde_2d(
    u_grid: &[Vec<Vec<f64>>],
    dt: f64,
    dx: f64,
    dy: f64,
    config: &PdeConfig,
) -> Result<PdeResult, String> {
    let nt = u_grid.len();
    if nt < 5 {
        return Err("2D PDE requires at least 5 time steps".to_string());
    }
    let nx = u_grid[0].len();
    let ny = u_grid[0][0].len();
    if nx < 7 || ny < 7 {
        return Err("2D PDE requires at least 7x7 spatial grid".to_string());
    }

    let dt = dt.max(1e-12);
    let dx = dx.max(1e-12);
    let dy = dy.max(1e-12);

    let mut ut = Vec::new();
    let mut u_vals = Vec::new();
    let mut ux = Vec::new();
    let mut uy = Vec::new();
    let mut uxx = Vec::new();
    let mut uyy = Vec::new();
    let mut laplacian = Vec::new();

    for t in 1..(nt - 1) {
        for x in 2..(nx - 2) {
            for y in 2..(ny - 2) {
                let d_t = (u_grid[t + 1][x][y] - u_grid[t - 1][x][y]) / (2.0 * dt);
                ut.push(d_t);
                u_vals.push(u_grid[t][x][y]);

                let d_x = (-u_grid[t][x + 2][y] + 8.0 * u_grid[t][x + 1][y] - 8.0 * u_grid[t][x - 1][y] + u_grid[t][x - 2][y])
                    / (12.0 * dx);
                ux.push(d_x);

                let d_y = (-u_grid[t][x][y + 2] + 8.0 * u_grid[t][x][y + 1] - 8.0 * u_grid[t][x][y - 1] + u_grid[t][x][y - 2])
                    / (12.0 * dy);
                uy.push(d_y);

                let d_xx = (-u_grid[t][x + 2][y] + 16.0 * u_grid[t][x + 1][y] - 30.0 * u_grid[t][x][y]
                    + 16.0 * u_grid[t][x - 1][y] - u_grid[t][x - 2][y])
                    / (12.0 * dx * dx);
                uxx.push(d_xx);

                let d_yy = (-u_grid[t][x][y + 2] + 16.0 * u_grid[t][x][y + 1] - 30.0 * u_grid[t][x][y]
                    + 16.0 * u_grid[t][x][y - 1] - u_grid[t][x][y - 2])
                    / (12.0 * dy * dy);
                uyy.push(d_yy);

                laplacian.push(d_xx + d_yy);
            }
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push("u_x".to_string());
    columns.push(ux.clone());

    term_names.push("u_y".to_string());
    columns.push(uy.clone());

    // 2D Laplacian: \nabla^2 u = u_xx + u_yy
    term_names.push("laplacian(u)".to_string());
    columns.push(laplacian.clone());

    term_names.push("u_xx".to_string());
    columns.push(uxx.clone());

    term_names.push("u_yy".to_string());
    columns.push(uyy.clone());

    if config.include_advection_terms {
        term_names.push("u*u_x".to_string());
        columns.push(u_vals.iter().zip(ux.iter()).map(|(&u, &ux)| u * ux).collect());

        term_names.push("u*u_y".to_string());
        columns.push(u_vals.iter().zip(uy.iter()).map(|(&u, &uy)| u * uy).collect());
    }

    run_stridge(&ut, &columns, &term_names, config, "u_t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_equation_discovery() {
        let nx = 50;
        let nt = 40;
        let dx = 0.1;
        let dt = 0.005;
        let mut u = vec![vec![0.0f64; nx]; nt];

        for j in 0..nx {
            let x = (j as f64) * dx;
            u[0][j] = x.sin() + 0.5 * (2.0 * x).sin();
        }
        let d = 0.5f64;
        for i in 0..(nt - 1) {
            for j in 1..(nx - 1) {
                u[i + 1][j] = u[i][j] + dt * d * (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / (dx * dx);
            }
            u[i + 1][0] = u[i + 1][1];
            u[i + 1][nx - 1] = u[i + 1][nx - 2];
        }

        let cfg = PdeConfig::default();
        let res = discover_pde_1d(&u, dt, dx, &cfg).unwrap();
        assert!(res.r2 > 0.99, "Heat eq R2 = {}", res.r2);
        assert!(res.active_terms.contains(&"u_xx".to_string()));
        assert!(!res.active_terms.contains(&"u*u_x".to_string()));
        let uxx_idx = res.active_terms.iter().position(|t| t == "u_xx").unwrap();
        assert!((res.coefficients[uxx_idx] - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_viscous_burgers_equation_discovery() {
        let nx = 60;
        let nt = 50;
        let dx = 0.1;
        let dt = 0.002;
        let mut u = vec![vec![0.0f64; nx]; nt];

        for j in 0..nx {
            let x = (j as f64) * dx;
            u[0][j] = x.sin();
        }
        let nu = 0.1f64;
        for i in 0..(nt - 1) {
            for j in 1..(nx - 1) {
                let adv = -u[i][j] * (u[i][j + 1] - u[i][j - 1]) / (2.0 * dx);
                let diff = nu * (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / (dx * dx);
                u[i + 1][j] = u[i][j] + dt * (adv + diff);
            }
            u[i + 1][0] = u[i + 1][1];
            u[i + 1][nx - 1] = u[i + 1][nx - 2];
        }

        let cfg = PdeConfig::default();
        let res = discover_pde_1d(&u, dt, dx, &cfg).unwrap();
        assert!(res.r2 > 0.999, "Burgers R2 = {}", res.r2);
        assert!(res.active_terms.contains(&"u*u_x".to_string()));
        assert!(res.active_terms.contains(&"u_xx".to_string()));
    }

    #[test]
    fn test_fornberg_weights_uniform_match() {
        // Uniform 5-point stencil: [-2, -1, 0, 1, 2]
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let w = fornberg_weights(&x, 0.0, 2);
        assert!((w[1][0] - (1.0 / 12.0)).abs() < 1e-10);
        assert!((w[1][1] - (-8.0 / 12.0)).abs() < 1e-10);
        assert!((w[1][2] - 0.0).abs() < 1e-10);
        assert!((w[1][3] - (8.0 / 12.0)).abs() < 1e-10);
        assert!((w[1][4] - (-1.0 / 12.0)).abs() < 1e-10);

        // 2nd derivative central: 1/12 * [-1, 16, -30, 16, -1]
        assert!((w[2][0] - (-1.0 / 12.0)).abs() < 1e-10);
        assert!((w[2][1] - (16.0 / 12.0)).abs() < 1e-10);
        assert!((w[2][2] - (-30.0 / 12.0)).abs() < 1e-10);
        assert!((w[2][3] - (16.0 / 12.0)).abs() < 1e-10);
        assert!((w[2][4] - (-1.0 / 12.0)).abs() < 1e-10);
        assert!((w[2][1] - (16.0 / 12.0)).abs() < 1e-10);
        assert!((w[2][2] - (-30.0 / 12.0)).abs() < 1e-10);
    }

    #[test]
    fn test_2d_heat_equation_discovery() {
        // 2D Heat diffusion: u_t = 0.25 * (u_xx + u_yy) = 0.25 * laplacian(u)
        let nt = 30;
        let nx = 20;
        let ny = 20;
        let dt = 0.002;
        let dx = 0.1;
        let dy = 0.1;
        let diff = 0.25f64;

        let mut u = vec![vec![vec![0.0f64; ny]; nx]; nt];
        for t in 0..nt {
            let time = (t as f64) * dt;
            let decay = (-2.0 * diff * time).exp();
            for x in 0..nx {
                let px = (x as f64) * dx;
                for y in 0..ny {
                    let py = (y as f64) * dy;
                    u[t][x][y] = decay * px.sin() * py.sin();
                }
            }
        }

        let mut cfg = PdeConfig::default();
        cfg.sparsity_threshold = 0.02;
        let res = discover_pde_2d(&u, dt, dx, dy, &cfg).unwrap();
        assert!(res.r2 > 0.99, "2D Heat eq R2 = {}", res.r2);
        // Either laplacian(u) or u_xx + u_yy should be identified
        let has_laplacian = res.active_terms.contains(&"laplacian(u)".to_string());
        let has_uxx = res.active_terms.contains(&"u_xx".to_string());
        assert!(has_laplacian || has_uxx, "Must find diffusion operator: {:?}", res.active_terms);
    }
}
